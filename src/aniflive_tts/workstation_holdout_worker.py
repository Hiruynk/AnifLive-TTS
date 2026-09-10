from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .workstation_evaluation import spoken_content_error_rate
from .workstation_production_gates import (
    holdout_evaluation_report,
    open_locked_test_holdout,
)
from .workstation_reference_selection import (
    load_preprocessed_speaker_embeddings,
    robust_speaker_centroid,
)
from .workstation_speaker import WorkstationSpeakerVerifier


class HoldoutEvaluationWorkerError(RuntimeError):
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
        raise HoldoutEvaluationWorkerError(f"{label} is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HoldoutEvaluationWorkerError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise HoldoutEvaluationWorkerError(f"{label} is malformed")
    return value


def _write_audio(path: Path, audio: Any, sample_rate: int) -> np.ndarray:
    try:
        from scipy.io import wavfile
    except ImportError as error:
        raise HoldoutEvaluationWorkerError(
            "holdout audio evidence requires scipy"
        ) from error
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise HoldoutEvaluationWorkerError("holdout generation returned invalid audio")
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, np.clip(values, -1.0, 1.0))
    return values


def _transcribe(model: Any, path: Path, language: str) -> str:
    segments, _ = model.transcribe(
        str(path),
        language="zh" if language == "yue" else language,
        beam_size=5,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    return "".join(segment.text for segment in segments).strip()


def _cosine(left: Any, right: Any) -> float:
    first = np.asarray(left, dtype=np.float64).reshape(-1)
    second = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        raise HoldoutEvaluationWorkerError("speaker cosine denominator is zero")
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def _speaker_verification_is_current(record: Mapping[str, Any]) -> bool:
    hashes = record.get("verification_hashes")
    if not isinstance(hashes, Mapping):
        return False
    value = hashes.get("speaker")
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def run_holdout_evaluation(
    manifest: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[Path]]:
    if not sys.platform.startswith("linux"):
        raise HoldoutEvaluationWorkerError("holdout evaluation runs only in Linux worker")
    inputs = manifest.get("container_input_paths")
    if not isinstance(inputs, Mapping):
        raise HoldoutEvaluationWorkerError("worker manifest has no container inputs")
    required = (
        "selected_checkpoints",
        "deployment_reference",
        "reference",
        "dataset",
        "shared_dir",
        "asr_model",
        "speaker_component",
    )
    if any(not isinstance(inputs.get(key), str) for key in required):
        raise HoldoutEvaluationWorkerError("holdout evaluation inputs are incomplete")
    selected_root = Path(str(inputs["selected_checkpoints"])).resolve(strict=True)
    training_bundle = Path(str(inputs["dataset"])).resolve(strict=True)
    deployment_path = selected_root / "selected" / "deployment-checkpoints.json"
    reference_manifest = Path(str(inputs["deployment_reference"])).resolve(strict=True)
    records, lock = open_locked_test_holdout(
        training_bundle=training_bundle,
        deployment_checkpoints=deployment_path,
        deployment_reference=reference_manifest,
    )
    deployment = _object(deployment_path, "deployment checkpoints")
    reference = _object(reference_manifest, "deployment reference")
    train_records = _object(
        training_bundle / "train" / "manifest.json", "train manifest"
    )["items"]
    embeddings = load_preprocessed_speaker_embeddings(
        train_records, selected_root / "selection-assets" / "7-sv_cn"
    )
    centroid, _ = robust_speaker_centroid(list(embeddings.values()))
    shared = Path(str(inputs["shared_dir"])).resolve(strict=True)
    source_dir = Path(
        os.environ.get("ANIFLIVE_TTS_SOURCE_DIR", "/app/minimal_inference")
    ).resolve(strict=True)
    for value in (source_dir, source_dir / "GPT_SoVITS"):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))
    try:
        import torch
        from faster_whisper import WhisperModel
        from run_inference import GPTSoVITSInference
    except ImportError as error:
        raise HoldoutEvaluationWorkerError("holdout neural dependencies are missing") from error
    engine = GPTSoVITSInference(
        str(selected_root / "selected" / deployment["gpt"]["relative_path"]),
        str(selected_root / "selected" / deployment["sovits"]["relative_path"]),
        str(shared / "chinese-hubert-base"),
        str(shared / "chinese-roberta-wwm-ext-large"),
        str(shared / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"),
    )
    asr = WhisperModel(
        str(Path(str(inputs["asr_model"])).resolve(strict=True)),
        device="cuda",
        device_index=0,
        compute_type="float16",
        local_files_only=True,
    )
    speaker = WorkstationSpeakerVerifier(
        Path(str(inputs["speaker_component"])).resolve(strict=True)
    )
    try:
        import soundfile as sf
    except ImportError as error:
        raise HoldoutEvaluationWorkerError(
            "source speaker calibration requires soundfile"
        ) from error
    source_speaker_scores: dict[str, float] = {}
    for index, record in enumerate(records, start=1):
        item_id = str(record.get("source_item_id") or "")
        relative = Path(str(record.get("path") or ""))
        if not item_id or relative.is_absolute() or ".." in relative.parts:
            raise HoldoutEvaluationWorkerError("test source audio path is unsafe")
        source_path = (training_bundle / relative).resolve(strict=True)
        try:
            source_path.relative_to(training_bundle)
        except ValueError as error:
            raise HoldoutEvaluationWorkerError(
                "test source audio escaped its training bundle"
            ) from error
        if source_path.is_symlink() or _sha256_file(source_path) != record.get("sha256"):
            raise HoldoutEvaluationWorkerError(
                "test source audio failed integrity validation"
            )
        source_audio, source_rate = sf.read(
            str(source_path), dtype="float32", always_2d=False
        )
        source_values = np.asarray(source_audio, dtype=np.float32)
        if source_values.ndim == 2:
            source_values = source_values.mean(axis=1)
        source_speaker_scores[item_id] = _cosine(
            speaker(source_values, int(source_rate)), centroid
        )
        _emit_progress(
            0.1 * index / len(records),
            f"Holdout source calibration {index}/{len(records)}",
        )
    output.mkdir(parents=True, exist_ok=True)
    errors: list[float] = []
    speaker_scores: list[float] = []
    repetition = omission = invalid = nan = empty = clipped = duration_outliers = 0
    successes = 0
    audio_paths: list[Path] = []
    case_evidence: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        text = str(record.get("transcript") or "").strip()
        language = str(record.get("language") or "").strip().casefold()
        case: dict[str, Any] = {
            "index": index,
            "source_item_id": record.get("source_item_id"),
            "language": language,
            "transcript": text,
        }
        try:
            np.random.seed(1234)
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            audio, sample_rate = engine.infer(
                str(Path(str(inputs["reference"])).resolve(strict=True)),
                str(reference["text"]),
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
            path = output / "audio" / f"holdout-{index:04d}.wav"
            values = _write_audio(path, values, int(sample_rate))
            audio_paths.append(path)
            hypothesis = _transcribe(asr, path, language)
            content_error = spoken_content_error_rate(text, hypothesis, language)
            errors.append(content_error)
            target_length = max(1, len(text.replace(" ", "")))
            hypothesis_length = len(hypothesis.replace(" ", ""))
            repetition += int(hypothesis_length > target_length * 1.75)
            omission += int(hypothesis_length < target_length * 0.55)
            duration = values.size / int(sample_rate)
            duration_outliers += int(duration < 0.25 or duration > max(8.0, target_length * 0.9))
            clipped += int(float(np.mean(np.abs(values) >= 0.999)) > 0.001)
            speaker_cosine = _cosine(speaker(values, int(sample_rate)), centroid)
            source_speaker_cosine = source_speaker_scores[
                str(record.get("source_item_id"))
            ]
            speaker_scores.append(speaker_cosine)
            successes += 1
            case.update(
                {
                    "status": "passed",
                    "hypothesis": hypothesis,
                    "content_error": content_error,
                    "speaker_centroid_cosine": speaker_cosine,
                    "source_speaker_centroid_cosine": source_speaker_cosine,
                    "duration_seconds": duration,
                    "output_audio": path.relative_to(output).as_posix(),
                }
            )
        except Exception as error:
            invalid += 1
            case.update(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                }
            )
        case_evidence.append(case)
        _emit_progress(
            0.1 + 0.85 * (index + 1) / len(records),
            f"Holdout evaluation {index + 1}/{len(records)}",
        )
    if not errors:
        errors = [1.0]
    if not speaker_scores:
        speaker_scores = [-1.0]
    metrics = {
        "generation_success_rate": successes / len(records),
        "content": {
            "median_error": float(np.median(errors)),
            "p95_error": float(np.percentile(errors, 95)),
            "repetition_failures": repetition,
            "omission_failures": omission,
        },
        "speaker": {
            "centroid_cosine_median": float(np.median(speaker_scores)),
            "centroid_cosine_p10": float(np.percentile(speaker_scores, 10)),
            "source_centroid_cosine_median": float(
                np.median(list(source_speaker_scores.values()))
            ),
            "source_centroid_cosine_p10": float(
                np.percentile(list(source_speaker_scores.values()), 10)
            ),
            "source_human_verified": all(
                _speaker_verification_is_current(record) for record in records
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
    }
    report = holdout_evaluation_report(metrics, lock=lock, cases=case_evidence)
    report_path = output / "holdout-evaluation.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report, [report_path, *audio_paths]


__all__ = ["HoldoutEvaluationWorkerError", "run_holdout_evaluation"]
