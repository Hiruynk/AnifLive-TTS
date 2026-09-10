from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .tse_pipeline import (
    SpeakerSegment,
    SpeakerVerificationConfig,
    VoiceActivityConfig,
    apply_review_decisions,
    detect_voice_activity,
    extract_target_audio,
    verify_target_speaker,
)


_MAX_MANUAL_SEPARATION_RANGES = 256


_RESULT_SCHEMA = "aniflive-tts-docker-worker-result-v1"
_MANIFEST_SCHEMA = "aniflive-tts-worker-preparation-v1"
_PLATFORM = "linux/amd64"
_SUPPORTED_COMMANDS = {
    "dataset": "dataset.process",
    "dataset-decode": "dataset.decode",
    "dataset-target-speaker": "dataset.target-speaker",
    "dataset-separate": "dataset.separate",
    "dataset-transcribe": "dataset.transcribe",
    "dataset-finalize": "dataset.finalize",
    "tse": "tse.prepare",
    "training": "training.prepare",
    "checkpoint-selection": "checkpoint.select",
    "reference-selection": "reference.select",
    "holdout-evaluation": "holdout.evaluate",
    "evaluation": "evaluation.prepare",
    "engine-build": "engine.prepare",
    "conversion-parity": "conversion.parity",
    "model-package": "model.package",
}


class ContainerWorkerError(RuntimeError):
    pass


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(values):
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ContainerWorkerError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ContainerWorkerError(f"non-finite JSON value: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContainerWorkerError(f"could not read worker manifest: {path}") from error
    if not isinstance(value, dict):
        raise ContainerWorkerError("worker manifest must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _regular_input(path: str, field: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.is_symlink():
        raise ContainerWorkerError(f"{field} must be an absolute regular file")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ContainerWorkerError(f"{field} must be a regular file")
    return resolved


def _directory_input(path: str, field: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.is_symlink():
        raise ContainerWorkerError(f"{field} must be an absolute directory")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir():
        raise ContainerWorkerError(f"{field} must be a directory")
    return resolved


def _settings(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = manifest.get("settings", {})
    if not isinstance(value, dict):
        raise ContainerWorkerError("worker settings must be a JSON object")
    return value


def _load_audio(path: Path, *, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    import soundfile as sf

    try:
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        with tempfile.TemporaryDirectory(prefix="aniflive-tse-") as temporary:
            decoded = Path(temporary) / "decoded.wav"
            completed = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(sample_rate or 16_000),
                    "-y",
                    str(decoded),
                ],
                check=False,
                capture_output=True,
            )
            if completed.returncode != 0:
                raise ContainerWorkerError(f"audio could not be decoded: {path.name}")
            audio, rate = sf.read(str(decoded), dtype="float32", always_2d=True)
    values = np.mean(np.asarray(audio, dtype=np.float32), axis=1).reshape(-1)
    if sample_rate is not None and int(rate) != int(sample_rate):
        from scipy.signal import resample_poly

        common = math.gcd(int(rate), int(sample_rate))
        values = resample_poly(
            values,
            int(sample_rate) // common,
            int(rate) // common,
        ).astype(np.float32, copy=False)
        rate = int(sample_rate)
    if values.size == 0 or not np.isfinite(values).all():
        raise ContainerWorkerError(f"audio is empty or malformed: {path.name}")
    return np.clip(values, -1.0, 1.0), int(rate)


def _active_engine_dir(package: Path) -> Path:
    manifest_path = package / "manifest.json"
    document = _strict_json(manifest_path)
    fingerprint = document.get("active_engine_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in fingerprint
    ):
        raise ContainerWorkerError("model package has an invalid engine fingerprint")
    engine_dir = (package / "engines" / fingerprint).resolve(strict=True)
    try:
        engine_dir.relative_to(package.resolve(strict=True))
    except ValueError as error:
        raise ContainerWorkerError("model package engine path escaped the package") from error
    engine = engine_dir / "sv_embedding.engine"
    if not engine.is_file():
        raise ContainerWorkerError("model package has no sv_embedding.engine")
    return engine_dir


class TensorRTSpeakerEmbedder:
    def __init__(self, model_package: Path) -> None:
        if sys.platform != "linux":
            raise ContainerWorkerError("TensorRT speaker verification only runs in Linux")
        import torch

        source_dir = Path(
            os.environ.get("ANIFLIVE_TTS_SOURCE_DIR", "/app/minimal_inference")
        ).resolve(strict=True)
        for import_root in (source_dir, source_dir / "GPT_SoVITS"):
            value = str(import_root)
            if value not in sys.path:
                sys.path.insert(0, value)
        from run_trt_inference import TRTModule

        if not torch.cuda.is_available():
            raise ContainerWorkerError("CUDA is unavailable inside the Linux worker")
        self._torch = torch
        self._device = torch.device("cuda")
        self._stream = torch.cuda.Stream(device=self._device)
        engine_dir = _active_engine_dir(model_package)
        self._module = TRTModule(
            str(engine_dir / "sv_embedding.engine"), self._device, self._stream
        )
        self._input_dtype = self._module.tensor_dtype["audio"]
        maximum = self._module.input_max_shapes.get("audio")
        self._maximum_samples = int(maximum[-1]) if maximum else 180_000

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != 16_000:
            from scipy.signal import resample_poly

            common = math.gcd(int(sample_rate), 16_000)
            audio = resample_poly(
                audio,
                16_000 // common,
                int(sample_rate) // common,
            ).astype(np.float32, copy=False)
        values = np.asarray(audio, dtype=np.float32).reshape(-1)
        minimum = 16_000
        if values.size < minimum:
            values = np.pad(values, (0, minimum - values.size))
        window = max(minimum, self._maximum_samples)
        step = max(minimum, window // 2)
        embeddings: list[np.ndarray] = []
        for start in range(0, values.size, step):
            part = values[start : start + window]
            if part.size < minimum and embeddings:
                break
            if part.size < minimum:
                part = np.pad(part, (0, minimum - part.size))
            tensor = self._torch.from_numpy(part.copy())[None, :].to(
                device=self._device, dtype=self._input_dtype
            )
            output = self._module({"audio": tensor})["sv_embedding"]
            embeddings.append(output.detach().float().cpu().numpy().reshape(-1))
            if start + window >= values.size:
                break
        if not embeddings:
            raise ContainerWorkerError("speaker embedding produced no windows")
        return np.mean(np.stack(embeddings), axis=0, dtype=np.float32)


def _possible_overlap_detector(
    embedder: TensorRTSpeakerEmbedder,
    reference_embedding: np.ndarray,
    *,
    threshold: float,
):
    reference = reference_embedding / max(float(np.linalg.norm(reference_embedding)), 1e-8)

    def detect(audio: np.ndarray, sample_rate: int) -> bool:
        minimum = int(sample_rate * 1.0)
        if audio.size < minimum * 2:
            return False
        scores: list[float] = []
        for start in range(0, audio.size - minimum + 1, minimum):
            value = embedder(audio[start : start + minimum], sample_rate)
            value /= max(float(np.linalg.norm(value)), 1e-8)
            scores.append(float(np.dot(reference, value)))
        if len(scores) < 2:
            return False
        # A sharp identity change within one VAD region is review evidence, not
        # an automatic rejection. This uses the same TensorRT speaker model and
        # never claims source separation has occurred.
        return max(scores) >= threshold and min(scores) < threshold - 0.16

    return detect


def _manual_separation_ranges(
    value: Any,
    *,
    sample_rate: int,
    sample_count: int,
) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ContainerWorkerError("separation_segments must be a JSON array")
    if len(value) > _MAX_MANUAL_SEPARATION_RANGES:
        raise ContainerWorkerError(
            f"separation_segments is limited to {_MAX_MANUAL_SEPARATION_RANGES} ranges"
        )
    ranges: list[tuple[int, int]] = []
    previous_end = 0
    previous_end_seconds = 0.0
    duration = sample_count / float(sample_rate)
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"start_seconds", "end_seconds"}:
            raise ContainerWorkerError(
                "separation_segments entries must contain only start_seconds and end_seconds"
            )
        start_value = item["start_seconds"]
        end_value = item["end_seconds"]
        if (
            isinstance(start_value, bool)
            or isinstance(end_value, bool)
            or not isinstance(start_value, (int, float))
            or not isinstance(end_value, (int, float))
        ):
            raise ContainerWorkerError("separation segment bounds must be JSON numbers")
        start_seconds = float(start_value)
        end_seconds = float(end_value)
        if (
            not math.isfinite(start_seconds)
            or not math.isfinite(end_seconds)
            or start_seconds < 0.0
            or end_seconds <= start_seconds
            or end_seconds > duration
        ):
            raise ContainerWorkerError(
                f"separation segment {index} is outside the source time bounds"
            )
        start = int(round(start_seconds * sample_rate))
        end = int(round(end_seconds * sample_rate))
        if not 0 <= start < end <= sample_count:
            raise ContainerWorkerError(
                f"separation segment {index} collapses outside the source sample bounds"
            )
        if ranges and (
            start_seconds < previous_end_seconds or start < previous_end
        ):
            raise ContainerWorkerError(
                "separation_segments must be sorted and must not overlap"
            )
        ranges.append((start, end))
        previous_end = end
        previous_end_seconds = end_seconds
    return tuple(ranges)


def _candidate_segments_with_manual_ranges(
    audio: np.ndarray,
    detected: Sequence[SpeakerSegment],
    manual_ranges: Sequence[tuple[int, int]],
) -> tuple[SpeakerSegment, ...]:
    bounds = [(segment.start_sample, segment.end_sample) for segment in detected]
    bounds.extend(manual_ranges)
    if not bounds:
        return ()
    merged: list[list[int]] = []
    for start, end in sorted(bounds):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    result: list[SpeakerSegment] = []
    for start, end in merged:
        region = audio[start:end]
        rms = float(np.sqrt(np.mean(np.square(region, dtype=np.float64)) + 1e-12))
        result.append(
            SpeakerSegment(
                index=len(result),
                start_sample=start,
                end_sample=end,
                rms_dbfs=20.0 * math.log10(max(rms, 1e-8)),
            )
        )
    return tuple(result)


def _range_intersects(
    start: int, end: int, ranges: Sequence[tuple[int, int]]
) -> bool:
    return any(start < range_end and range_start < end for range_start, range_end in ranges)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")


def _run_tse(manifest: Mapping[str, Any], output: Path) -> tuple[dict[str, Any], list[Path]]:
    inputs = manifest.get("container_input_paths")
    if not isinstance(inputs, dict):
        raise ContainerWorkerError("worker manifest has no container input paths")
    source = _regular_input(str(inputs.get("source", "")), "source")
    reference = _regular_input(str(inputs.get("reference", "")), "reference")
    package = _directory_input(str(inputs.get("model_package", "")), "model_package")
    separation_model = _directory_input(
        str(inputs.get("separation_model", "")), "separation_model"
    )
    settings = _settings(manifest)
    target_threshold = float(settings.get("target_threshold", 0.72))
    review_margin = float(settings.get("review_margin", 0.08))
    separation_ambiguity_margin = float(
        settings.get("separation_ambiguity_margin", 0.03)
    )
    if not math.isfinite(target_threshold) or not 0.0 <= target_threshold <= 1.0:
        raise ContainerWorkerError("target_threshold must be between 0 and 1")
    if not math.isfinite(review_margin) or not 0.0 <= review_margin <= 0.5:
        raise ContainerWorkerError("review_margin must be between 0 and 0.5")
    if (
        not math.isfinite(separation_ambiguity_margin)
        or not 0.0 <= separation_ambiguity_margin <= 0.5
    ):
        raise ContainerWorkerError(
            "separation_ambiguity_margin must be between 0 and 0.5"
        )

    audio, sample_rate = _load_audio(source, sample_rate=16_000)
    reference_audio, reference_rate = _load_audio(reference, sample_rate=16_000)
    manual_ranges = _manual_separation_ranges(
        settings.get("separation_segments", []),
        sample_rate=sample_rate,
        sample_count=audio.size,
    )
    embedder = TensorRTSpeakerEmbedder(package)
    reference_embedding = embedder(reference_audio, reference_rate)
    detected_segments = detect_voice_activity(
        audio,
        sample_rate,
        config=VoiceActivityConfig(
            frame_ms=float(settings.get("frame_ms", 20.0)),
            minimum_speech_ms=float(settings.get("minimum_speech_ms", 160.0)),
            maximum_gap_ms=float(settings.get("maximum_gap_ms", 120.0)),
            context_ms=float(settings.get("context_ms", 40.0)),
        ),
    )
    segments = _candidate_segments_with_manual_ranges(
        audio, detected_segments, manual_ranges
    )
    verified = verify_target_speaker(
        audio,
        sample_rate,
        segments,
        reference_audio=reference_audio,
        embedder=embedder,
        overlap_detector=_possible_overlap_detector(
            embedder, reference_embedding, threshold=target_threshold
        ),
        config=SpeakerVerificationConfig(
            target_threshold=target_threshold,
            review_margin=review_margin,
            extraction_gap_ms=float(settings.get("extraction_gap_ms", 120.0)),
        ),
    )
    separation_triggers: dict[int, tuple[str, ...]] = {}
    forced: list[Any] = []
    for segment in verified:
        triggers: list[str] = []
        if segment.overlap:
            triggers.append("detector")
        if _range_intersects(
            segment.start_sample, segment.end_sample, manual_ranges
        ):
            triggers.append("operator")
        separation_triggers[segment.index] = tuple(triggers)
        forced.append(
            replace(segment, overlap=True, decision="review") if triggers else segment
        )
    verified = tuple(forced)
    separated_audio: dict[int, np.ndarray] = {}
    separation_records: dict[int, dict[str, Any]] = {}
    overlap_segments = [segment for segment in verified if segment.overlap]
    if overlap_segments:
        from .tse_separation import (
            MOSSFORMER2_CHECKPOINT_SHA256,
            MOSSFORMER2_MODEL_REVISION,
            MOSSFORMER2_SOURCE_REVISION,
            MossFormer2TargetSeparator,
            SeparationBackendError,
        )

        try:
            separator = MossFormer2TargetSeparator(separation_model)
            recovered: list[Any] = []
            for segment in verified:
                if not segment.overlap:
                    recovered.append(segment)
                    continue
                region = audio[segment.start_sample : segment.end_sample]
                result = separator.separate_target(
                    region,
                    sample_rate,
                    reference_embedding=reference_embedding,
                    embedder=embedder,
                    target_threshold=target_threshold,
                    ambiguity_margin=separation_ambiguity_margin,
                )
                separated_audio[segment.index] = result.audio
                separation_records[segment.index] = {
                    "accepted": result.accepted,
                    "minimum_selected_similarity": result.minimum_selected_similarity,
                    "chunks": [asdict(chunk) for chunk in result.chunks],
                }
                recovered.append(
                    replace(segment, decision="target" if result.accepted else "review")
                )
            verified = tuple(recovered)
        except SeparationBackendError as error:
            raise ContainerWorkerError(str(error)) from error
        separation_backend = {
            "name": "MossFormer2_SS_16K",
            "execution_environment": "linux-docker-cuda",
            "source_revision": MOSSFORMER2_SOURCE_REVISION,
            "model_revision": MOSSFORMER2_MODEL_REVISION,
            "checkpoint_sha256": MOSSFORMER2_CHECKPOINT_SHA256,
            "target_selection": "TensorRT-11 sv_embedding.engine",
        }
    else:
        from .tse_separation import (
            MOSSFORMER2_CHECKPOINT_SHA256,
            MOSSFORMER2_MODEL_REVISION,
            MOSSFORMER2_SOURCE_REVISION,
        )

        separation_backend = {
            "name": "MossFormer2_SS_16K",
            "execution_environment": "linux-docker-cuda",
            "source_revision": MOSSFORMER2_SOURCE_REVISION,
            "model_revision": MOSSFORMER2_MODEL_REVISION,
            "checkpoint_sha256": MOSSFORMER2_CHECKPOINT_SHA256,
            "target_selection": "TensorRT-11 sv_embedding.engine",
        }
    decisions_value = settings.get("review_decisions", {})
    if not isinstance(decisions_value, dict):
        raise ContainerWorkerError("review_decisions must be a JSON object")
    decisions: dict[int, str] = {}
    for raw_index, raw_decision in decisions_value.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as error:
            raise ContainerWorkerError("review decision indices must be integers") from error
        if str(index) != str(raw_index) or index < 0:
            raise ContainerWorkerError("review decision indices must be canonical integers")
        if raw_decision not in {"target", "review", "rejected"}:
            raise ContainerWorkerError("review decisions must be target, review or rejected")
        decisions[index] = raw_decision
    if decisions and any(index >= len(verified) for index in decisions):
        raise ContainerWorkerError("review decision index is outside the detected segments")
    verified = apply_review_decisions(verified, decisions)
    extraction = extract_target_audio(
        audio,
        sample_rate,
        verified,
        replacement_audio=separated_audio,
    )
    audio_path = output / "target-speaker.wav"
    _write_wav(audio_path, extraction.audio, sample_rate)
    report_path = output / "tse-report.json"
    report = {
        "schema": "aniflive-tts-tse-report-v2",
        "source_sha256": _sha256_file(source),
        "reference_sha256": _sha256_file(reference),
        "sample_rate": sample_rate,
        "source_seconds": audio.size / sample_rate,
        "target_seconds": extraction.target_seconds,
        "review_seconds": extraction.review_seconds,
        "target_segments": sum(value.decision == "target" for value in verified),
        "review_segments": sum(value.decision == "review" for value in verified),
        "rejected_segments": sum(value.decision == "rejected" for value in verified),
        "segments": [
            {
                **asdict(value),
                "separation_triggers": list(separation_triggers[value.index]),
                "separation": separation_records.get(value.index),
            }
            for value in verified
        ],
        "speaker_backend": "TensorRT-11 sv_embedding.engine",
        "overlap_policy": "mossformer2-ss-16k-plus-trt-target-selection-v1",
        "separation_backend": separation_backend,
        "separation_attempted_segments": len(overlap_segments),
        "separation_accepted_segments": sum(
            bool(value.get("accepted")) for value in separation_records.values()
        ),
        "review_decisions_applied": len(decisions),
        "separation_requested_ranges": [
            {
                "start_sample": start,
                "end_sample": end,
                "start_seconds": start / sample_rate,
                "end_seconds": end / sample_rate,
            }
            for start, end in manual_ranges
        ],
        "separation_detector_segments": sum(
            "detector" in triggers for triggers in separation_triggers.values()
        ),
        "separation_operator_segments": sum(
            "operator" in triggers for triggers in separation_triggers.values()
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report, [audio_path, report_path]


def _dataset_asr_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    summary = {
        key: value[key]
        for key in (
            "schema",
            "backend",
            "execution_environment",
            "network_required",
            "model_qualification",
            "declared_language",
            "language",
            "review_required",
            "review_reasons",
        )
        if key in value
    }
    segments = value.get("segments")
    if isinstance(segments, list):
        summary["segment_count"] = len(segments)
    transcript = value.get("text")
    if isinstance(transcript, str):
        summary["transcript_character_count"] = len(transcript)
    model = value.get("model")
    if isinstance(model, Mapping) and isinstance(model.get("tree_sha256"), str):
        summary["model_tree_sha256"] = model["tree_sha256"]
    return summary


def _dataset_transcript_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    summary = {
        key: value[key]
        for key in (
            "language",
            "source",
            "asr_backend",
            "asr_model_tree_sha256",
            "review_required",
            "review_reasons",
        )
        if key in value
    }
    text = value.get("normalized_text")
    if not isinstance(text, str):
        text = value.get("canonical_input_text")
    if isinstance(text, str):
        summary["transcript_character_count"] = len(text)
    return summary


def _run_dataset(
    manifest: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[Path]]:
    from .dataset_factory import EnergyVadConfig
    from .dataset_pipeline import DatasetPipelineConfig, run_dataset_pipeline

    inputs = manifest.get("container_input_paths")
    if not isinstance(inputs, dict):
        raise ContainerWorkerError("worker manifest has no container input paths")
    source = _regular_input(str(inputs.get("source", "")), "source")
    asr_model_value = inputs.get("asr_model")
    if asr_model_value is not None and not isinstance(asr_model_value, str):
        raise ContainerWorkerError("asr_model must be an absolute directory")
    asr_model = (
        _directory_input(str(asr_model_value), "asr_model")
        if isinstance(asr_model_value, str)
        else None
    )
    vad_model_value = inputs.get("vad_model")
    if vad_model_value is not None and not isinstance(vad_model_value, str):
        raise ContainerWorkerError("vad_model must be an absolute directory")
    vad_model = (
        _directory_input(str(vad_model_value), "vad_model")
        if isinstance(vad_model_value, str)
        else None
    )
    settings = _settings(manifest)
    supported = {
        "acquisition_mode",
        "sources",
        "reference_audio",
        "speaker_threshold",
        "ambiguity_margin",
        "review_margin",
        "speaker_component",
        "vad_component",
        "diarization_component",
        "separation_component",
        "asr_component",
        "asr_backend",
        "transcript",
        "language",
        "declared_language",
        "require_expressions",
        "enable_afftdn",
        "afftdn_noise_reduction_db",
        "afftdn_noise_floor_db",
        "afftdn_gain_smooth",
        "dereverb_backend",
        "language_confidence_threshold",
        "max_segments",
        "parent_artifact_ids",
        "vad",
    }
    unknown = sorted(set(settings) - supported)
    if unknown:
        raise ContainerWorkerError(
            "dataset worker settings are unsupported: " + ", ".join(unknown)
        )
    transcript = settings.get("transcript")
    if transcript is not None and (
        not isinstance(transcript, str) or not transcript.strip() or len(transcript) > 16_000
    ):
        raise ContainerWorkerError(
            "dataset transcript must be non-empty text up to 16000 characters"
        )
    language = settings.get("language")
    declared_language = settings.get("declared_language")
    if language is not None and declared_language is not None:
        raise ContainerWorkerError("use either language or declared_language, not both")
    selected_language = declared_language if declared_language is not None else language
    if selected_language is not None and (
        not isinstance(selected_language, str) or len(selected_language) > 16
    ):
        raise ContainerWorkerError("dataset language must be a short language code")
    require_expressions = settings.get("require_expressions", True)
    if not isinstance(require_expressions, bool):
        raise ContainerWorkerError("require_expressions must be a boolean")
    enable_afftdn = settings.get("enable_afftdn", False)
    if not isinstance(enable_afftdn, bool):
        raise ContainerWorkerError("enable_afftdn must be a boolean")
    dereverb_backend = settings.get("dereverb_backend", "none")
    if not isinstance(dereverb_backend, str):
        raise ContainerWorkerError("dereverb_backend must be text")
    vad_value = settings.get("vad", {})
    if not isinstance(vad_value, dict):
        raise ContainerWorkerError("dataset vad settings must be a JSON object")
    try:
        max_segments = settings.get("max_segments", 1_000)
        if (
            not isinstance(max_segments, int)
            or isinstance(max_segments, bool)
            or not 1 <= max_segments <= 1_000
        ):
            raise ValueError("max_segments must be an integer between 1 and 1000")
        vad = EnergyVadConfig(**vad_value).validated()
        config = DatasetPipelineConfig(
            max_segments=max_segments,
            enable_afftdn=enable_afftdn,
            afftdn_noise_reduction_db=float(
                settings.get("afftdn_noise_reduction_db", 6.0)
            ),
            afftdn_noise_floor_db=float(
                settings.get("afftdn_noise_floor_db", -50.0)
            ),
            afftdn_gain_smooth=int(settings.get("afftdn_gain_smooth", 5)),
            dereverb_backend=dereverb_backend,
            language_confidence_threshold=float(
                settings.get("language_confidence_threshold", 0.80)
            ),
            vad=vad,
        ).validated()
    except (TypeError, ValueError) as error:
        raise ContainerWorkerError(f"dataset pipeline settings are invalid: {error}") from error

    destination = output / "dataset-pipeline"
    report = run_dataset_pipeline(
        source,
        destination,
        transcript=transcript,
        declared_language=selected_language,
        config=config,
        asr_model_path=asr_model,
        asr_backend=str(settings.get("asr_backend", "faster-whisper")),
        vad_model_path=vad_model,
    )
    canonical = report["output"]["canonical"]
    segment_records = report["output"]["segments"]
    artifacts = [
        destination / "canonical.wav",
        destination / "dataset-report.json",
        *(destination / record["path"] for record in segment_records),
    ]
    asr_artifact = report["output"].get("asr_transcripts")
    if isinstance(asr_artifact, Mapping) and isinstance(asr_artifact.get("path"), str):
        artifacts.append(destination / str(asr_artifact["path"]))
    payload = {
        "schema": report["schema"],
        "pipeline_identity_sha256": report["pipeline_identity_sha256"],
        "source_sha256": report["input"]["sha256"],
        "sample_rate": canonical["sample_rate"],
        "duration_seconds": round(
            float(canonical["frame_count"]) / float(canonical["sample_rate"]), 6
        ),
        "segment_count": len(segment_records),
        "denoise": canonical["denoise"],
        "dereverb": canonical["dereverb"],
        "transcript": _dataset_transcript_summary(report["transcript"]),
        "asr": _dataset_asr_summary(report.get("asr")),
        "execution_environment": report["runtime"]["execution_environment"],
        "neural_fallback": report["runtime"]["neural_fallback"],
    }
    return payload, artifacts


def _run_dataset_target_speaker(
    manifest: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[Path]]:
    from .dataset_acquisition_worker import run_target_speaker_routing

    inputs = manifest.get("container_input_paths")
    if not isinstance(inputs, dict):
        raise ContainerWorkerError("worker manifest has no container input paths")
    return run_target_speaker_routing(
        source=_regular_input(str(inputs.get("source", "")), "source"),
        reference=_regular_input(str(inputs.get("reference", "")), "reference"),
        speaker_component=_directory_input(
            str(inputs.get("speaker_engine", "")), "speaker_engine"
        ),
        vad_component=_directory_input(str(inputs.get("vad_model", "")), "vad_model"),
        diarization_component=_directory_input(
            str(inputs.get("diarization_model", "")), "diarization_model"
        ),
        output=output,
        settings=_settings(manifest),
    )


def _run_dataset_separation(
    manifest: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[Path]]:
    from .dataset_acquisition_worker import run_target_speaker_separation

    inputs = manifest.get("container_input_paths")
    if not isinstance(inputs, dict):
        raise ContainerWorkerError("worker manifest has no container input paths")
    return run_target_speaker_separation(
        dependency=_directory_input(str(inputs.get("dataset", "")), "dataset"),
        reference=_regular_input(str(inputs.get("reference", "")), "reference"),
        speaker_component=_directory_input(
            str(inputs.get("speaker_engine", "")), "speaker_engine"
        ),
        separation_model=_directory_input(
            str(inputs.get("separation_model", "")), "separation_model"
        ),
        output=output,
        settings=_settings(manifest),
    )


def _run_dataset_transcription(
    manifest: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[Path]]:
    from .dataset_acquisition_worker import run_target_speaker_transcription

    inputs = manifest.get("container_input_paths")
    if not isinstance(inputs, dict):
        raise ContainerWorkerError("worker manifest has no container input paths")
    return run_target_speaker_transcription(
        dependency=_directory_input(str(inputs.get("dataset", "")), "dataset"),
        asr_model=_directory_input(str(inputs.get("asr_model", "")), "asr_model"),
        output=output,
        settings=_settings(manifest),
    )


def _run_dataset_finalize(
    manifest: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[Path]]:
    from .dataset_acquisition_worker import run_target_speaker_finalize

    inputs = manifest.get("container_input_paths")
    if not isinstance(inputs, dict):
        raise ContainerWorkerError("worker manifest has no container input paths")
    return run_target_speaker_finalize(
        dependency=_directory_input(str(inputs.get("dataset", "")), "dataset"),
        output=output,
    )


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ContainerWorkerError("worker output destination already exists")
    shutil.copytree(source, destination, symlinks=False)


def _deployment_checkpoint_pair(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    from .model_package import resolve_contained_path

    root = _directory_input(str(root), "checkpoint")
    manifest_path = _regular_input(
        str(root / "deployment-checkpoints.json"), "deployment checkpoint manifest"
    )
    document = _strict_json(manifest_path)
    if (
        document.get("schema")
        != "aniflive-tts-v2proplus-deployment-checkpoints-v2"
        or document.get("model_family") != "gsv-v2proplus"
        or not isinstance(document.get("selection"), Mapping)
        or document["selection"].get("test_split_accessed") is not False
    ):
        raise ContainerWorkerError("deployment checkpoint manifest is unsupported")
    result: dict[str, Path] = {}
    for role, prefix, suffix in (
        ("gpt", "checkpoints/gpt/", ".ckpt"),
        ("sovits", "checkpoints/sovits/", ".pth"),
    ):
        record = document.get(role)
        if not isinstance(record, Mapping):
            raise ContainerWorkerError(f"deployment checkpoint manifest has no {role} record")
        relative = record.get("relative_path")
        if (
            not isinstance(relative, str)
            or not relative.startswith(prefix)
            or not relative.lower().endswith(suffix)
        ):
            raise ContainerWorkerError(f"deployment {role} checkpoint path is invalid")
        try:
            candidate = resolve_contained_path(root, relative, f"deployment {role} path")
        except Exception as error:
            raise ContainerWorkerError(f"deployment {role} checkpoint path is invalid") from error
        candidate = _regular_input(str(candidate), f"deployment {role} checkpoint")
        expected_sha = record.get("sha256")
        expected_size = record.get("size_bytes")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha)
            or not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size <= 0
        ):
            raise ContainerWorkerError(f"deployment {role} checkpoint metadata is invalid")
        if candidate.stat().st_size != expected_size or _sha256_file(candidate) != expected_sha:
            raise ContainerWorkerError(
                f"deployment {role} checkpoint does not match its immutable manifest"
            )
        result[role] = candidate
    return result["gpt"], result["sovits"], document


def _template_reference(
    package: Path,
) -> tuple[Path, str, str, str, str, dict[str, Any]]:
    from .model_backend import model_backend_for_manifest
    from .model_package import (
        resolve_contained_path,
        validate_checksums,
        validate_safe_identifier,
    )

    package = _directory_input(str(package), "model_package")
    validate_checksums(package)
    manifest = _strict_json(_regular_input(str(package / "manifest.json"), "package manifest"))
    if model_backend_for_manifest(manifest) is None:
        raise ContainerWorkerError("template package is not a supported V2ProPlus package")
    model_id = validate_safe_identifier(manifest.get("model_id"), "model_id")
    voice_profile = validate_safe_identifier(
        manifest.get("default_voice_profile"), "default_voice_profile"
    )
    profiles = manifest.get("voice_profiles")
    if not isinstance(profiles, list) or voice_profile not in profiles:
        raise ContainerWorkerError("template package default voice profile is invalid")
    profile_root = resolve_contained_path(
        package / "voices", voice_profile, "default voice profile"
    )
    profile = _strict_json(
        _regular_input(str(profile_root / "profile.json"), "voice profile manifest")
    )
    reference = resolve_contained_path(
        profile_root, profile.get("reference_audio"), "reference audio"
    )
    reference = _regular_input(str(reference), "reference audio")
    reference_text = profile.get("reference_text")
    reference_language = profile.get("reference_language")
    if not isinstance(reference_text, str) or not reference_text.strip():
        raise ContainerWorkerError("template voice profile reference text is empty")
    if reference_language not in {"zh", "yue", "en", "ja", "ko"}:
        raise ContainerWorkerError("template voice profile reference language is unsupported")
    return (
        reference,
        reference_text.strip(),
        str(reference_language),
        model_id,
        voice_profile,
        manifest,
    )


def _validate_package(package: Path) -> dict[str, Any]:
    from .validate import validate_model_package

    try:
        report = validate_model_package(package, enqueue=False)
    except Exception as error:
        raise ContainerWorkerError(f"model package validation failed: {error}") from error
    if report.get("status") != "passed" or report.get("engine_count") != 9:
        raise ContainerWorkerError("model package did not validate all nine TensorRT stages")
    return report


def _convert_checkpoint_model(**arguments: Any) -> Path:
    from .converter import convert_model

    return convert_model(**arguments)


def _run_engine_build(
    manifest: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[Path]]:
    inputs = manifest.get("container_input_paths")
    if not isinstance(inputs, dict):
        raise ContainerWorkerError("worker manifest has no container input paths")
    settings = _settings(manifest)
    workspace_mib = int(settings.get("workspace_mib", 4096))
    optimization_level = int(settings.get("optimization_level", 5))
    if workspace_mib < 256 or workspace_mib > 32768:
        raise ContainerWorkerError("workspace_mib must be between 256 and 32768")
    if optimization_level not in range(6):
        raise ContainerWorkerError("optimization_level must be between 0 and 5")
    destination = output / "model-package"
    checkpoint_value = inputs.get("checkpoint")
    package_value = inputs.get("model_package")
    mode: str
    source_sha256: str
    checkpoint_report: dict[str, Any] | None = None
    if checkpoint_value is not None:
        checkpoint = _directory_input(str(checkpoint_value), "checkpoint")
        gpt, sovits, checkpoint_manifest = _deployment_checkpoint_pair(checkpoint)
        shared_value = inputs.get("shared_dir")
        if shared_value is None:
            raise ContainerWorkerError("checkpoint conversion requires shared_dir")
        shared_dir = _directory_input(str(shared_value), "shared_dir")
        if package_value is not None:
            template = _directory_input(str(package_value), "model_package")
            (
                reference,
                reference_text,
                reference_language,
                model_id,
                voice_profile,
                template_manifest,
            ) = _template_reference(template)
            source_sha256 = _sha256_file(template / "manifest.json")
        else:
            reference_value = inputs.get("reference")
            if reference_value is None:
                raise ContainerWorkerError(
                    "checkpoint conversion requires model_package or reference"
                )
            reference = _regular_input(str(reference_value), "reference")
            reference_text = settings.get("reference_text")
            reference_language = settings.get("reference_language")
            model_id = settings.get("model_id")
            voice_profile = settings.get("voice_profile", "default")
            template_manifest = None
            if not isinstance(reference_text, str) or not reference_text.strip():
                raise ContainerWorkerError("reference_text is required for direct conversion")
            if reference_language not in {"zh", "yue", "en", "ja", "ko"}:
                raise ContainerWorkerError(
                    "reference_language must be zh, yue, en, ja or ko"
                )
            if not isinstance(model_id, str) or not isinstance(voice_profile, str):
                raise ContainerWorkerError("model_id and voice_profile must be text")
            source_sha256 = _sha256_file(reference)
        if isinstance(settings.get("model_id"), str):
            model_id = str(settings["model_id"])
        if isinstance(settings.get("voice_profile"), str):
            voice_profile = str(settings["voice_profile"])
        private = output / ".private-conversion"
        private.mkdir(parents=True)
        reference_text_file = private / "reference.txt"
        reference_text_file.write_text(reference_text.strip() + "\n", encoding="utf-8")
        allow_unsafe_pickle = settings.get("allow_unsafe_pickle", False)
        if not isinstance(allow_unsafe_pickle, bool):
            raise ContainerWorkerError("allow_unsafe_pickle must be boolean")
        try:
            _convert_checkpoint_model(
                gpt=gpt,
                sovits=sovits,
                reference_audio=reference,
                reference_text_file=reference_text_file,
                reference_language=str(reference_language),
                model_id=str(model_id),
                voice_profile=str(voice_profile),
                output=destination,
                shared_dir=shared_dir,
                source_dir=Path(
                    os.environ.get("ANIFLIVE_TTS_SOURCE_DIR", "/app/minimal_inference")
                ),
                allow_unsafe_pickle=allow_unsafe_pickle,
                max_len=int(settings.get("max_len", 1000)),
                stream_overlap_frames=int(settings.get("stream_overlap_frames", 12)),
                workspace_mib=workspace_mib,
                optimization_level=optimization_level,
            )
        except Exception as error:
            raise ContainerWorkerError(f"V2ProPlus checkpoint conversion failed: {error}") from error
        finally:
            shutil.rmtree(private, ignore_errors=True)
        mode = "checkpoint-conversion"
        checkpoint_report = {
            "schema": checkpoint_manifest["schema"],
            "gpt_sha256": _sha256_file(gpt),
            "sovits_sha256": _sha256_file(sovits),
            "template_model_id": (
                template_manifest.get("model_id")
                if isinstance(template_manifest, Mapping)
                else None
            ),
        }
    else:
        if package_value is None:
            raise ContainerWorkerError("engine build requires checkpoint or model_package")
        source = _directory_input(str(package_value), "model_package")
        _copy_tree(source, destination)
        from .converter import rebuild_engines

        try:
            rebuild_engines(
                model_package=destination,
                workspace_mib=workspace_mib,
                optimization_level=optimization_level,
                force=bool(settings.get("force", False)),
            )
        except Exception as error:
            raise ContainerWorkerError(f"TensorRT engine build failed: {error}") from error
        mode = "package-rebuild"
        source_sha256 = _sha256_file(source / "manifest.json")
    validation = _validate_package(destination)
    report_path = output / "engine-build.json"
    report = {
        "schema": "aniflive-tts-engine-build-report-v2",
        "status": "passed",
        "mode": mode,
        "source_sha256": source_sha256,
        "workspace_mib": workspace_mib,
        "optimization_level": optimization_level,
        "checkpoint": checkpoint_report,
        "validation": validation,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = [path for path in destination.rglob("*") if path.is_file()]
    return report, [report_path, *files]


def _run_model_package(
    manifest: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[Path]]:
    inputs = manifest.get("container_input_paths")
    if not isinstance(inputs, dict):
        raise ContainerWorkerError("worker manifest has no container input paths")
    package_value = inputs.get("model_package")
    if package_value is None:
        raise ContainerWorkerError("model packaging requires a model package directory")
    source = _directory_input(str(package_value), "model_package")
    destination = output / "model-package"
    _copy_tree(source, destination)
    report = _validate_package(destination)
    report_path = output / "package-validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = [path for path in destination.rglob("*") if path.is_file()]
    return {"validated": True, "report": report}, [report_path, *files]


def _artifact_kind(command: str, path: Path) -> str:
    del path
    return {
        "dataset": "dataset",
        "dataset-decode": "dataset",
        "dataset-target-speaker": "dataset",
        "dataset-separate": "dataset",
        "dataset-transcribe": "dataset",
        "dataset-finalize": "dataset",
        "tse": "dataset",
        "training": "checkpoint",
        "checkpoint-selection": "checkpoint",
        "reference-selection": "reference",
        "holdout-evaluation": "evaluation",
        "evaluation": "evaluation",
        "engine-build": "engine",
        "conversion-parity": "evaluation",
        "model-package": "package",
    }[command]


def _write_result(
    *,
    command: str,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    result_path: Path,
    payload: Mapping[str, Any],
    artifact_paths: Sequence[Path],
) -> None:
    output = result_path.parent.resolve(strict=True)
    artifacts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in artifact_paths:
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(output).as_posix()
        except ValueError as error:
            raise ContainerWorkerError("worker artifact escaped the output root") from error
        if relative == result_path.name or relative in seen:
            continue
        seen.add(relative)
        artifacts.append(
            {
                "kind": _artifact_kind(command, resolved),
                "relative_path": relative,
                "sha256": _sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    document = {
        "schema": os.environ.get("ANIFLIVE_TTS_WORKER_RESULT_SCHEMA", _RESULT_SCHEMA),
        "job_id": manifest["job_id"],
        "job_type": manifest["job_type"],
        "project_id": manifest["project_id"],
        "run_id": os.environ["ANIFLIVE_TTS_WORKER_RUN_ID"],
        "image_digest": os.environ["ANIFLIVE_TTS_WORKER_IMAGE_DIGEST"],
        "manifest_sha256": os.environ["ANIFLIVE_TTS_WORKER_MANIFEST_SHA256"],
        "platform": os.environ.get("ANIFLIVE_TTS_WORKER_PLATFORM", _PLATFORM),
        "outcome": "completed",
        "payload": dict(payload),
        "artifacts": artifacts,
    }
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, result_path)


def run(command: str, manifest_path: Path, result_path: Path) -> int:
    expected_job_type = _SUPPORTED_COMMANDS.get(command)
    if expected_job_type is None:
        raise ContainerWorkerError(f"unsupported worker command: {command}")
    manifest = _strict_json(manifest_path)
    if manifest.get("schema") != _MANIFEST_SCHEMA:
        raise ContainerWorkerError("worker manifest schema is unsupported")
    if manifest.get("job_type") != expected_job_type:
        raise ContainerWorkerError("worker command and manifest job type do not match")
    expected_digest = os.environ.get("ANIFLIVE_TTS_WORKER_MANIFEST_SHA256", "")
    if _sha256_file(manifest_path) != expected_digest:
        raise ContainerWorkerError("worker manifest checksum does not match the broker")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        raise ContainerWorkerError("worker result already exists")
    if command == "dataset":
        payload, artifacts = _run_dataset(manifest, result_path.parent)
    elif command == "dataset-decode":
        payload, artifacts = _run_dataset(manifest, result_path.parent)
    elif command == "dataset-target-speaker":
        payload, artifacts = _run_dataset_target_speaker(manifest, result_path.parent)
    elif command == "dataset-separate":
        payload, artifacts = _run_dataset_separation(manifest, result_path.parent)
    elif command == "dataset-transcribe":
        payload, artifacts = _run_dataset_transcription(manifest, result_path.parent)
    elif command == "dataset-finalize":
        payload, artifacts = _run_dataset_finalize(manifest, result_path.parent)
    elif command == "tse":
        payload, artifacts = _run_tse(manifest, result_path.parent)
    elif command == "training":
        from .workstation_training import run_training

        payload, artifacts = run_training(manifest, result_path.parent)
    elif command == "checkpoint-selection":
        from .workstation_checkpoint_worker import run_checkpoint_selection

        payload, artifacts = run_checkpoint_selection(manifest, result_path.parent)
    elif command == "reference-selection":
        from .workstation_reference_worker import run_reference_selection

        payload, artifacts = run_reference_selection(manifest, result_path.parent)
    elif command == "holdout-evaluation":
        from .workstation_holdout_worker import run_holdout_evaluation

        payload, artifacts = run_holdout_evaluation(manifest, result_path.parent)
    elif command == "evaluation":
        from .workstation_evaluation import run_evaluation

        payload, artifacts = run_evaluation(manifest, result_path.parent)
    elif command == "engine-build":
        payload, artifacts = _run_engine_build(manifest, result_path.parent)
    elif command == "conversion-parity":
        from .workstation_conversion_worker import run_conversion_parity

        payload, artifacts = run_conversion_parity(manifest, result_path.parent)
    elif command == "model-package":
        payload, artifacts = _run_model_package(manifest, result_path.parent)
    else:  # pragma: no cover - guarded by _SUPPORTED_COMMANDS above.
        raise ContainerWorkerError(f"unsupported worker command: {command}")
    _write_result(
        command=command,
        manifest=manifest,
        manifest_path=manifest_path,
        result_path=result_path,
        payload=payload,
        artifact_paths=artifacts,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="workstation-worker")
    parser.add_argument("command", choices=tuple(_SUPPORTED_COMMANDS))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args.command, args.manifest, args.result)
    except Exception as error:
        print(f"AnifLive-TTS workstation worker failed: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
