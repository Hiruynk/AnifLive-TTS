from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .workstation_checkpoint_selection import (
    CHECKPOINT_CANDIDATES_SCHEMA,
    CheckpointSelectionError,
    build_selection_plan,
    consolidate_reference_evaluations,
    evaluate_checkpoint_evidence,
    materialize_deployment_checkpoints,
    rank_checkpoint_sweep,
    select_checkpoint_pair,
    validation_records,
)
from .workstation_evaluation import content_error_rate, spoken_content_error_rate
from .workstation_reference_selection import (
    load_preprocessed_speaker_embeddings,
    rank_reference_candidates,
    robust_speaker_centroid,
)
from .workstation_speaker import WorkstationSpeakerVerifier


class CheckpointSelectionWorkerError(RuntimeError):
    pass


def _emit_progress(progress: float, message: str) -> None:
    print(
        "ANIFLIVE_TTS_PROGRESS "
        + json.dumps(
            {"progress": max(0.0, min(1.0, progress)), "message": message},
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CheckpointSelectionWorkerError(f"{label} is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointSelectionWorkerError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise CheckpointSelectionWorkerError(f"{label} is malformed")
    return value


def _directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise CheckpointSelectionWorkerError(f"{label} directory is missing")
    return path.resolve(strict=True)


def _candidate_path(root: Path, record: Mapping[str, Any], label: str) -> Path:
    relative = Path(str(record.get("relative_path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise CheckpointSelectionWorkerError(f"{label} path is unsafe")
    path = root / relative
    if path.is_symlink() or not path.is_file() or _sha256_file(path) != record.get("sha256"):
        raise CheckpointSelectionWorkerError(f"{label} failed integrity validation")
    return path


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        raise CheckpointSelectionWorkerError("speaker cosine denominator is zero")
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    try:
        from scipy.io import wavfile
    except ImportError as error:
        raise CheckpointSelectionWorkerError(
            "checkpoint audio evidence requires scipy"
        ) from error
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise CheckpointSelectionWorkerError("generated audio is empty or invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, np.clip(values, -1.0, 1.0))


def _transcribe(model: Any, path: Path, language: str) -> str:
    routed = "zh" if language == "yue" else language
    segments, _ = model.transcribe(
        str(path),
        language=routed,
        beam_size=5,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    return "".join(segment.text for segment in segments).strip()


def _reference_records(
    records: Sequence[Mapping[str, Any]], embeddings: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    selection = rank_reference_candidates(records, embeddings, limit=3)
    by_item_id = {
        str(record.get("source_item_id")): record for record in records
    }
    candidates = selection["top_candidates"]
    resolved: list[Mapping[str, Any]] = []
    for candidate in candidates:
        item_id = str(candidate.get("item_id") or "")
        record = by_item_id.get(item_id)
        if record is None:
            raise CheckpointSelectionWorkerError(
                "provisional centroid reference did not resolve one train item"
            )
        resolved.append(record)
    if len(resolved) < 3:
        raise CheckpointSelectionWorkerError(
            "checkpoint selection requires three qualified train references"
        )
    automatic = candidates[0]
    return resolved, {
        "purpose": "checkpoint-selection-probe-only",
        "declared_deployment_reference": False,
        "policy": selection["policy"],
        "mode": "multi-reference-prototype-evaluation-v1",
        "item_id": automatic["item_id"],
        "path": automatic["path"],
        "speaker_centroid_cosine": automatic["speaker_centroid_cosine"],
        "selection_score": automatic["selection_score"],
        "prototype_size": len(resolved),
        "prototype": [
            {
                "item_id": candidate["item_id"],
                "path": candidate["path"],
                "speaker_centroid_cosine": candidate["speaker_centroid_cosine"],
                "selection_score": candidate["selection_score"],
            }
            for candidate in candidates
        ],
    }


class _PairEvaluator:
    def __init__(
        self,
        *,
        candidate_root: Path,
        training_bundle: Path,
        shared_dir: Path,
        asr_model: Path,
        speaker_component: Path,
        output: Path,
    ) -> None:
        candidates = _object(
            candidate_root / "checkpoint-candidates.json", "checkpoint candidates"
        )
        if candidates.get("schema") != CHECKPOINT_CANDIDATES_SCHEMA:
            raise CheckpointSelectionWorkerError("checkpoint candidates are unsupported")
        self.candidate_root = candidate_root
        self.training_bundle = training_bundle
        self.shared_dir = shared_dir
        self.output = output
        self.gpt = {int(row["epoch"]): row for row in candidates["gpt"]}
        self.sovits = {int(row["epoch"]): row for row in candidates["sovits"]}
        train_manifest = _object(training_bundle / "train" / "manifest.json", "train manifest")
        train_records = train_manifest.get("items")
        if train_manifest.get("split") != "train" or not isinstance(train_records, list):
            raise CheckpointSelectionWorkerError("train manifest is malformed")
        vectors = load_preprocessed_speaker_embeddings(
            train_records,
            candidate_root / "selection-assets" / "7-sv_cn",
        )
        self.centroid, _ = robust_speaker_centroid(list(vectors.values()))
        self.references, self.provisional_reference = _reference_records(
            train_records, vectors
        )
        self.reference_paths = [
            training_bundle / str(reference["path"])
            for reference in self.references
        ]
        self.speaker = WorkstationSpeakerVerifier(speaker_component)
        self._source_speaker_scores: dict[str, float] = {}
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise CheckpointSelectionWorkerError("offline ASR is missing") from error
        self.asr = WhisperModel(
            str(asr_model),
            device="cuda",
            device_index=0,
            compute_type="float16",
            local_files_only=True,
        )
        source_dir = Path(
            os.environ.get("ANIFLIVE_TTS_SOURCE_DIR", "/app/minimal_inference")
        ).resolve(strict=True)
        for value in (source_dir, source_dir / "GPT_SoVITS"):
            if str(value) not in sys.path:
                sys.path.insert(0, str(value))
        try:
            from run_inference import GPTSoVITSInference
        except ImportError as error:
            raise CheckpointSelectionWorkerError("PyTorch inference source is missing") from error
        self.inference_type = GPTSoVITSInference

    def _source_speaker_score(self, record: Mapping[str, Any]) -> float:
        try:
            import soundfile as sf
        except ImportError as error:
            raise CheckpointSelectionWorkerError(
                "source speaker calibration requires soundfile"
            ) from error
        item_id = str(record.get("source_item_id") or "")
        if not item_id:
            raise CheckpointSelectionWorkerError("validation item ID is missing")
        cached = self._source_speaker_scores.get(item_id)
        if cached is not None:
            return cached
        relative = Path(str(record.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise CheckpointSelectionWorkerError("validation audio path is unsafe")
        path = (self.training_bundle / relative).resolve(strict=True)
        try:
            path.relative_to(self.training_bundle)
        except ValueError as error:
            raise CheckpointSelectionWorkerError(
                "validation audio escaped its training bundle"
            ) from error
        if path.is_symlink() or _sha256_file(path) != record.get("sha256"):
            raise CheckpointSelectionWorkerError(
                "validation audio failed integrity validation"
            )
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim == 2:
            values = values.mean(axis=1)
        score = _cosine(self.speaker(values, int(sample_rate)), self.centroid)
        self._source_speaker_scores[item_id] = score
        return score

    def evaluate(
        self,
        *,
        gpt_epoch: int,
        sovits_epoch: int,
        records: Sequence[Mapping[str, Any]],
        seeds: Sequence[int],
        phase: str,
        reference_indices: Sequence[int] = (0,),
    ) -> dict[str, Any]:
        try:
            import torch
        except ImportError as error:
            raise CheckpointSelectionWorkerError("PyTorch is missing") from error
        engine = self.inference_type(
            str(_candidate_path(self.candidate_root, self.gpt[gpt_epoch], "GPT")),
            str(_candidate_path(self.candidate_root, self.sovits[sovits_epoch], "SoVITS")),
            str(self.shared_dir / "chinese-hubert-base"),
            str(self.shared_dir / "chinese-roberta-wwm-ext-large"),
            str(self.shared_dir / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"),
        )
        errors: list[float] = []
        orthographic_errors: list[float] = []
        speaker_scores: list[float] = []
        durations: dict[str, list[float]] = {}
        repetition = omission = invalid = nan = empty = clipped = duration_outliers = 0
        attempts = successes = 0
        audio_root = self.output / "audio" / phase / f"g{gpt_epoch}-s{sovits_epoch}"
        source_speaker_scores = [
            self._source_speaker_score(record) for record in records
        ]
        selected_indices = tuple(reference_indices)
        if (
            not selected_indices
            or len(set(selected_indices)) != len(selected_indices)
            or any(index < 0 or index >= len(self.references) for index in selected_indices)
        ):
            raise CheckpointSelectionWorkerError("reference probe indices are invalid")
        selected_references = list(
            (index, self.references[index], self.reference_paths[index])
            for index in selected_indices
        )
        for reference_index, reference, reference_path in selected_references:
            for record in records:
                text = str(record.get("transcript") or "").strip()
                language = str(record.get("language") or "").strip().casefold()
                item_id = str(record.get("source_item_id") or "item")
                if not text or language not in {"zh", "yue", "en", "ja", "ko"}:
                    raise CheckpointSelectionWorkerError(
                        "validation annotations are incomplete"
                    )
                for seed in seeds:
                    attempts += 1
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    try:
                        audio, sample_rate = engine.infer(
                            str(reference_path),
                            str(reference["transcript"]),
                            str(reference["language"]),
                            text,
                            language,
                            top_k=15,
                            top_p=1.0,
                            temperature=1.0,
                        )
                        values = np.asarray(audio, dtype=np.float32).reshape(-1)
                        if not np.isfinite(values).all():
                            nan += 1
                            continue
                        if values.size == 0:
                            empty += 1
                            continue
                        output_path = audio_root / (
                            f"r{reference_index + 1}-{item_id}-seed-{seed}.wav"
                        )
                        _write_audio(output_path, values, int(sample_rate))
                        hypothesis = _transcribe(self.asr, output_path, language)
                        errors.append(
                            spoken_content_error_rate(text, hypothesis, language)
                        )
                        orthographic_errors.append(
                            content_error_rate(text, hypothesis, language)
                        )
                        target_length = max(1, len(text.replace(" ", "")))
                        hypothesis_length = len(hypothesis.replace(" ", ""))
                        if hypothesis_length > target_length * 1.75:
                            repetition += 1
                        if hypothesis_length < target_length * 0.55:
                            omission += 1
                        duration = values.size / int(sample_rate)
                        durations.setdefault(
                            f"r{reference_index}:{item_id}", []
                        ).append(duration)
                        if duration < 0.25 or duration > max(
                            8.0, target_length * 0.9
                        ):
                            duration_outliers += 1
                        if float(np.mean(np.abs(values) >= 0.999)) > 0.001:
                            clipped += 1
                        speaker_scores.append(
                            _cosine(
                                self.speaker(values, int(sample_rate)),
                                self.centroid,
                            )
                        )
                        successes += 1
                    except Exception:
                        invalid += 1
        if not errors or not speaker_scores:
            errors = [1.0]
            orthographic_errors = [1.0]
            speaker_scores = [-1.0]
        variability = [float(np.std(values)) for values in durations.values() if len(values) > 1]
        stability = max(0.0, 1.0 - (float(np.mean(variability)) if variability else 0.0))
        return {
            "phase": phase,
            "gpt_epoch": gpt_epoch,
            "sovits_epoch": sovits_epoch,
            "seeds": list(seeds),
            "reference_probe": {
                "mode": "single-reference-pruning-probe"
                if phase != "joint-sweep"
                else "checkpoint-conditioned-reference-candidate-v1",
                "count": len(selected_references),
                "item_ids": [
                    str(reference.get("source_item_id"))
                    for _index, reference, _path in selected_references
                ],
            },
            "generation_success_rate": successes / attempts if attempts else 0.0,
            "content": {
                "primary_metric": "spoken-content-error",
                "median_error": float(np.median(errors)),
                "p95_error": float(np.percentile(errors, 95)),
                "orthographic_median_error": float(
                    np.median(orthographic_errors)
                ),
                "orthographic_p95_error": float(
                    np.percentile(orthographic_errors, 95)
                ),
                "repetition_failures": repetition,
                "omission_failures": omission,
            },
            "speaker": {
                "centroid_cosine_median": float(np.median(speaker_scores)),
                "centroid_cosine_p10": float(np.percentile(speaker_scores, 10)),
                "source_centroid_cosine_median": float(
                    np.median(source_speaker_scores)
                ),
                "source_centroid_cosine_p10": float(
                    np.percentile(source_speaker_scores, 10)
                ),
                "source_human_verified": all(
                    isinstance(record.get("verification_hashes"), Mapping)
                    and isinstance(record["verification_hashes"].get("speaker"), str)
                    and len(record["verification_hashes"]["speaker"]) == 64
                    for record in records
                ),
                "source_case_count": len(records),
            },
            "audio": {
                "invalid": invalid,
                "nan": nan,
                "empty": empty,
                "clipped": clipped,
                "duration_outliers": duration_outliers,
            },
            "stability": stability,
        }


def _write_diagnostics(
    output: Path,
    *,
    plan: Mapping[str, Any],
    passes: Mapping[str, Any],
    status: str,
    provisional_reference: Mapping[str, Any],
) -> Path:
    path = output / "checkpoint-selection-diagnostics.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema": "aniflive-tts-checkpoint-selection-diagnostics-v1",
                "status": status,
                "test_split_accessed": False,
                "provisional_reference": provisional_reference,
                "plan": plan,
                "passes": passes,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def run_checkpoint_selection(
    manifest: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[Path]]:
    if not sys.platform.startswith("linux"):
        raise CheckpointSelectionWorkerError("checkpoint selection runs only in Linux worker")
    inputs = manifest.get("container_input_paths")
    if not isinstance(inputs, Mapping):
        raise CheckpointSelectionWorkerError("worker manifest has no container inputs")
    required = ("checkpoint_candidates", "dataset", "shared_dir", "asr_model", "speaker_component")
    if any(not isinstance(inputs.get(key), str) for key in required):
        raise CheckpointSelectionWorkerError("checkpoint selection inputs are incomplete")
    candidate_root = _directory(Path(str(inputs["checkpoint_candidates"])), "candidates")
    training_bundle = _directory(Path(str(inputs["dataset"])), "training bundle")
    candidates = _object(candidate_root / "checkpoint-candidates.json", "checkpoint candidates")
    validation = validation_records(training_bundle)
    validation_sha = _sha256_file(training_bundle / "validation" / "manifest.json")
    plan = build_selection_plan(
        candidates, validation_item_ids=[str(row.get("source_item_id")) for row in validation]
    )
    output.mkdir(parents=True, exist_ok=True)
    evaluator = _PairEvaluator(
        candidate_root=candidate_root,
        training_bundle=training_bundle,
        shared_dir=_directory(Path(str(inputs["shared_dir"])), "shared models"),
        asr_model=_directory(Path(str(inputs["asr_model"])), "ASR model"),
        speaker_component=_directory(
            Path(str(inputs["speaker_component"])), "speaker component"
        ),
        output=output,
    )
    probe = validation[: min(5, len(validation))]
    pass_a_units = len(candidates["gpt"]) * len(probe)
    pass_b_units = len(candidates["sovits"]) * len(probe)
    joint_reference_count = min(3, len(evaluator.references))
    pass_c_units = (
        min(3, len(candidates["gpt"]))
        * min(3, len(candidates["sovits"]))
        * len(validation)
        * 3
        * joint_reference_count
    )
    total_units = max(1, pass_a_units + pass_b_units + pass_c_units)
    completed_units = 0

    def report_progress(message: str) -> None:
        _emit_progress(completed_units / total_units, message)

    report_progress("Checkpoint selection initialized")
    latest_sovits = int(plan["probe_dependencies"]["latest_sovits_epoch"])
    pass_a: list[dict[str, Any]] = []
    for index, row in enumerate(candidates["gpt"], start=1):
        pass_a.append(
            evaluator.evaluate(
                gpt_epoch=int(row["epoch"]),
                sovits_epoch=latest_sovits,
                records=probe,
                seeds=(1234,),
                phase="gpt-sweep",
            )
        )
        completed_units += len(probe)
        report_progress(f"GPT validation sweep {index}/{len(candidates['gpt'])}")
    evaluated_a = evaluate_checkpoint_evidence(
        pass_a,
        enforce_speaker_identity=False,
    )
    _write_diagnostics(
        output,
        plan=plan,
        passes={"gpt_sweep": evaluated_a},
        status="gpt-sweep-evaluated",
        provisional_reference=evaluator.provisional_reference,
    )
    top_gpt = rank_checkpoint_sweep(evaluated_a, "gpt")
    provisional_gpt = top_gpt[0]
    plan["passes"]["sovits_sweep"]["provisional_gpt_epoch"] = provisional_gpt
    pass_b: list[dict[str, Any]] = []
    for index, row in enumerate(candidates["sovits"], start=1):
        pass_b.append(
            evaluator.evaluate(
                gpt_epoch=provisional_gpt,
                sovits_epoch=int(row["epoch"]),
                records=probe,
                seeds=(1234,),
                phase="sovits-sweep",
            )
        )
        completed_units += len(probe)
        report_progress(f"SoVITS validation sweep {index}/{len(candidates['sovits'])}")
    # Pass B is a five-item pruning probe. Rank voice candidates by speaker
    # evidence here, then enforce the fixed speaker-identity gate on the full
    # validation set in Pass C.
    evaluated_b = evaluate_checkpoint_evidence(
        pass_b,
        enforce_speaker_identity=False,
    )
    _write_diagnostics(
        output,
        plan=plan,
        passes={"gpt_sweep": evaluated_a, "sovits_sweep": evaluated_b},
        status="sovits-sweep-evaluated",
        provisional_reference=evaluator.provisional_reference,
    )
    top_sovits = rank_checkpoint_sweep(evaluated_b, "sovits")
    joint_pairs = [
        (gpt_epoch, sovits_epoch)
        for gpt_epoch in top_gpt
        for sovits_epoch in top_sovits
    ]
    pass_c: list[dict[str, Any]] = []
    for index, (gpt_epoch, sovits_epoch) in enumerate(joint_pairs, start=1):
        reference_evidence = []
        for reference_index in range(joint_reference_count):
            reference_evidence.append(
                evaluator.evaluate(
                    gpt_epoch=gpt_epoch,
                    sovits_epoch=sovits_epoch,
                    records=validation,
                    seeds=(1234, 2026, 7),
                    phase="joint-sweep",
                    reference_indices=(reference_index,),
                )
            )
            completed_units += len(validation) * 3
        pass_c.append(consolidate_reference_evaluations(reference_evidence))
        report_progress(f"Joint checkpoint sweep {index}/{len(joint_pairs)}")
    evaluated_c = evaluate_checkpoint_evidence(pass_c)
    diagnostics_path = _write_diagnostics(
        output,
        plan=plan,
        passes={
            "gpt_sweep": evaluated_a,
            "sovits_sweep": evaluated_b,
            "joint_sweep": evaluated_c,
        },
        status="joint-sweep-evaluated",
        provisional_reference=evaluator.provisional_reference,
    )
    try:
        report = select_checkpoint_pair(
            evaluated_c, validation_manifest_sha256=validation_sha
        )
    except CheckpointSelectionError as error:
        diagnostics_path = _write_diagnostics(
            output,
            plan=plan,
            passes={
                "gpt_sweep": evaluated_a,
                "sovits_sweep": evaluated_b,
                "joint_sweep": evaluated_c,
            },
            status="failed",
            provisional_reference=evaluator.provisional_reference,
        )
        return {
            "schema": "aniflive-tts-checkpoint-selection-worker-v1",
            "status": "failed",
            "reason": str(error),
            "test_split_accessed": False,
        }, [diagnostics_path]
    report["provisional_reference"] = evaluator.provisional_reference
    report["plan"] = plan
    report["passes"] = {
        "gpt_sweep": evaluated_a,
        "sovits_sweep": evaluated_b,
        "joint_sweep": evaluated_c,
    }
    report_path = output / "checkpoint-selection-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected_root = output / "selected"
    deployment_path, deployment, selected = materialize_deployment_checkpoints(
        candidate_root=candidate_root,
        selection_report_path=report_path,
        output=selected_root,
    )
    payload = {
        "schema": "aniflive-tts-checkpoint-selection-worker-v1",
        "status": "passed",
        "winner": report["winner"],
        "runner_up": report["runner_up"],
        "deployment": deployment,
    }
    selection_assets_root = output / "selection-assets"
    shutil.copytree(candidate_root / "selection-assets", selection_assets_root)
    selection_assets = sorted(selection_assets_root.rglob("*"))
    selection_assets = [path for path in selection_assets if path.is_file()]
    audio = sorted((output / "audio").rglob("*.wav"))
    return payload, [
        report_path,
        diagnostics_path,
        deployment_path,
        *selected[:-1],
        *selection_assets,
        *audio,
    ]


__all__ = ["CheckpointSelectionWorkerError", "run_checkpoint_selection"]
