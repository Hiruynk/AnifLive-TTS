from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .dataset_quality import DatasetQualityError, canonical_language, normalize_transcript

DATASET_ASR_SCHEMA = "aniflive-dataset-asr-v1"
DATASET_ASR_BACKEND = "faster-whisper-ctranslate2-cuda-v1"
SENSEVOICE_ASR_BACKEND = "sensevoice-small-cuda-v1"
SUPPORTED_ASR_LANGUAGES = frozenset({"yue", "zh", "ja", "en", "ko"})
_WHISPER_LANGUAGE = {"yue": "zh", "zh": "zh", "ja": "ja", "en": "en", "ko": "ko"}
_MODEL_FILE_LIMIT = 4_096
_MODEL_BYTE_LIMIT = 20 * 1024**3
_TEXT_LIMIT = 16_000
_SENSEVOICE_TOKEN = re.compile(r"<\|([^|<>]{1,80})\|>")
_CJK_OR_JAPANESE_PUNCTUATION = (
    r"\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff"
    r"\u3000-\u303f\uff01-\uff60\uffe0-\uffee"
)
_CJK_ASR_SPACE = re.compile(
    rf"(?<=[{_CJK_OR_JAPANESE_PUNCTUATION}]) +"
    rf"(?=[{_CJK_OR_JAPANESE_PUNCTUATION}])"
)
_SENSEVOICE_EMOTIONS = {
    "HAPPY": "happy",
    "SAD": "sad",
    "ANGRY": "angry",
    "NEUTRAL": "neutral",
    "FEARFUL": "fearful",
    "DISGUSTED": "disgusted",
    "SURPRISED": "surprised",
}


class DatasetAsrError(ValueError):
    """Raised when offline Dataset Factory transcription cannot be proven safe."""


class WorkstationASRBackend(Protocol):
    backend_id: str

    def transcribe(
        self,
        segment_records: Sequence[Mapping[str, Any]],
        *,
        stage_root: Path,
        declared_language: str | None = None,
    ) -> dict[str, Any]: ...


def _normalize_sensevoice_spacing(text: str, language: str | None) -> tuple[str, bool]:
    """Remove decoder token gaps that are not orthographic spaces in CJK text."""

    normalized = normalize_transcript(text)
    if language not in {"ja", "zh", "yue"}:
        return normalized, False
    compact = _CJK_ASR_SPACE.sub("", normalized)
    return compact, compact != normalized


@dataclass(frozen=True)
class CTranslate2WhisperModel:
    root: Path
    tree_sha256: str
    file_count: int
    total_bytes: int
    critical_files: Mapping[str, Mapping[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": "ctranslate2-whisper",
            "tree_sha256": self.tree_sha256,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "critical_files": {
                key: dict(value) for key, value in sorted(self.critical_files.items())
            },
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_file(path: Path, *, maximum_bytes: int = 1024 * 1024) -> dict[str, Any]:
    if path.stat().st_size > maximum_bytes:
        raise DatasetAsrError(f"{path.name} exceeds the ASR metadata size limit")

    def pairs(values: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise DatasetAsrError(f"{path.name} contains duplicate JSON fields")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                DatasetAsrError(f"{path.name} contains a non-finite value: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetAsrError(f"{path.name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise DatasetAsrError(f"{path.name} must contain a JSON object")
    return value


def _model_inventory(root: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    total_bytes = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise DatasetAsrError("the ASR model directory could not be inspected") from error
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise DatasetAsrError("an ASR model entry could not be inspected") from error
            if entry.is_symlink():
                raise DatasetAsrError("the ASR model cannot contain symbolic links")
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise DatasetAsrError("the ASR model can contain only regular files")
            if len(records) >= _MODEL_FILE_LIMIT:
                raise DatasetAsrError("the ASR model exceeds the file-count limit")
            total_bytes += int(info.st_size)
            if total_bytes > _MODEL_BYTE_LIMIT:
                raise DatasetAsrError("the ASR model exceeds the byte limit")
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": int(info.st_size),
                    "sha256": _sha256_file(path),
                }
            )
    records.sort(key=lambda item: str(item["path"]))
    return records, total_bytes


def inspect_ct2_whisper_model(model_path: Path) -> CTranslate2WhisperModel:
    supplied = Path(model_path).expanduser()
    if supplied.is_symlink():
        raise DatasetAsrError("the ASR model root cannot be a symbolic link")
    try:
        root = supplied.resolve(strict=True)
    except OSError as error:
        raise DatasetAsrError("the local ASR model directory was not found") from error
    if not root.is_dir():
        raise DatasetAsrError("the local ASR model must be a directory")
    records, total_bytes = _model_inventory(root)
    by_path = {str(record["path"]): record for record in records}
    if "model.bin" not in by_path or "config.json" not in by_path:
        raise DatasetAsrError("asr_model is not a local CTranslate2 Whisper model")
    tokenizers = {"tokenizer.json", "vocabulary.json", "vocabulary.txt"}
    if not tokenizers.intersection(by_path):
        raise DatasetAsrError("the CTranslate2 Whisper tokenizer asset is missing")
    config = _strict_json_file(root / "config.json")
    language_ids = config.get("lang_ids")
    if not isinstance(language_ids, list) or len(language_ids) < len(SUPPORTED_ASR_LANGUAGES):
        raise DatasetAsrError("the ASR model is not a multilingual Whisper model")
    serialized = json.dumps(
        records,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    critical_names = {"model.bin", "config.json", *tokenizers}
    critical = {
        name: {
            "sha256": str(record["sha256"]),
            "size_bytes": int(record["size_bytes"]),
        }
        for name, record in by_path.items()
        if name in critical_names
    }
    return CTranslate2WhisperModel(
        root=root,
        tree_sha256=hashlib.sha256(serialized).hexdigest(),
        file_count=len(records),
        total_bytes=total_bytes,
        critical_files=critical,
    )


def _finite_probability(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or not 0 <= result <= 1:
        return None
    return result


def _finite_metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _has_repeated_text(value: str) -> bool:
    compact = "".join(character for character in value if not character.isspace())
    for width in range(1, min(8, len(compact) // 3) + 1):
        for start in range(len(compact) - width * 3 + 1):
            unit = compact[start : start + width]
            if any(character.isalnum() for character in unit) and (
                compact[start : start + width * 3] == unit * 3
            ):
                return True
    return False


def _canonical_detected_language(value: Any) -> str:
    try:
        result = canonical_language(str(value or ""))
    except DatasetQualityError as error:
        raise DatasetAsrError(
            "Whisper detected a language outside yue/zh/ja/en/ko; declare the dataset language"
        ) from error
    if result not in SUPPORTED_ASR_LANGUAGES:
        raise DatasetAsrError("Whisper returned an unsupported language")
    return result


def _load_model(path: Path) -> tuple[Any, dict[str, str]]:
    try:
        import ctranslate2
        import faster_whisper
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise DatasetAsrError(
            "offline faster-whisper/CTranslate2 dependencies are missing from the Linux worker"
        ) from error
    try:
        model = WhisperModel(
            str(path),
            device="cuda",
            device_index=0,
            compute_type="float16",
            cpu_threads=1,
            num_workers=1,
            local_files_only=True,
        )
    except Exception as error:
        raise DatasetAsrError(
            "the offline CTranslate2 Whisper model could not be loaded"
        ) from error
    return model, {
        "faster_whisper": str(getattr(faster_whisper, "__version__", "unknown")),
        "ctranslate2": str(getattr(ctranslate2, "__version__", "unknown")),
    }


def _verified_segment_path(
    stage_root: Path,
    record: Mapping[str, Any],
) -> tuple[Path, str, int]:
    relative = record.get("path")
    expected_hash = record.get("sha256")
    position = record.get("position")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise DatasetAsrError("ASR received a malformed segment path")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        raise DatasetAsrError("ASR received a malformed segment checksum")
    if not isinstance(position, int) or isinstance(position, bool) or position < 0:
        raise DatasetAsrError("ASR received a malformed segment position")
    path = (stage_root / relative).resolve(strict=True)
    if stage_root != path and stage_root not in path.parents:
        raise DatasetAsrError("ASR segment escaped the dataset staging directory")
    if path.is_symlink() or not path.is_file() or path.suffix.casefold() != ".wav":
        raise DatasetAsrError("ASR segments must be regular WAV files")
    if _sha256_file(path) != expected_hash:
        raise DatasetAsrError("ASR segment failed its pre-inference checksum")
    return path, expected_hash, position


def transcribe_dataset_segments(
    segment_records: Sequence[Mapping[str, Any]],
    *,
    stage_root: Path,
    model_path: Path,
    declared_language: str | None = None,
    model_factory: Callable[[Path], tuple[Any, Mapping[str, str]]] | None = None,
) -> dict[str, Any]:
    # Importing lazily keeps the lightweight host-side dataset modules free of
    # neural runtime dependencies while preserving a single Docker assertion.
    from .dataset_pipeline import assert_linux_docker_runtime

    assert_linux_docker_runtime()
    if not segment_records:
        raise DatasetAsrError("offline ASR requires at least one verified VAD segment")
    selected_language = None
    if declared_language is not None:
        try:
            selected_language = canonical_language(declared_language)
        except DatasetQualityError as error:
            raise DatasetAsrError(str(error)) from error
    model_before = inspect_ct2_whisper_model(model_path)
    factory = model_factory or _load_model
    model, versions_value = factory(model_before.root)
    versions = {str(key): str(value) for key, value in dict(versions_value).items()}
    results: list[dict[str, Any]] = []
    observed_languages: list[str] = []
    total_characters = 0
    for record in segment_records:
        segment_path, expected_hash, position = _verified_segment_path(
            Path(stage_root).resolve(strict=True), record
        )
        whisper_language = _WHISPER_LANGUAGE[selected_language] if selected_language else None
        try:
            generated, info = model.transcribe(
                str(segment_path),
                language=whisper_language,
                task="transcribe",
                beam_size=5,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                word_timestamps=False,
            )
            decoded_segments = list(generated)
            text = "".join(str(getattr(item, "text", "")) for item in decoded_segments).strip()
        except Exception as error:
            raise DatasetAsrError(f"offline ASR failed on segment {position}") from error
        detected_language = _canonical_detected_language(getattr(info, "language", None))
        probability = _finite_probability(getattr(info, "language_probability", None))
        language = selected_language or detected_language
        observed_languages.append(language)
        review_reasons = ["offline-asr-requires-human-review"]
        if selected_language is not None:
            expected_whisper = _WHISPER_LANGUAGE[selected_language]
            if detected_language != expected_whisper:
                review_reasons.append("declared-language-disagrees-with-whisper")
        if probability is None or probability < 0.80:
            review_reasons.append("whisper-language-confidence-below-threshold")
        if text:
            try:
                canonical_text = normalize_transcript(text)
            except DatasetQualityError as error:
                raise DatasetAsrError(str(error)) from error
        else:
            canonical_text = ""
            review_reasons.append("whisper-returned-empty-segment")
        compression_ratios = [
            value
            for item in decoded_segments
            if (value := _finite_metric(getattr(item, "compression_ratio", None))) is not None
        ]
        average_log_probabilities = [
            value
            for item in decoded_segments
            if (value := _finite_metric(getattr(item, "avg_logprob", None))) is not None
        ]
        no_speech_probabilities = [
            value
            for item in decoded_segments
            if (value := _finite_probability(getattr(item, "no_speech_prob", None))) is not None
        ]
        maximum_compression_ratio = max(compression_ratios, default=None)
        minimum_average_log_probability = min(average_log_probabilities, default=None)
        maximum_no_speech_probability = max(no_speech_probabilities, default=None)
        if canonical_text and _has_repeated_text(canonical_text):
            review_reasons.append("whisper-repetition-detected")
        if maximum_compression_ratio is not None and maximum_compression_ratio > 2.4:
            review_reasons.append("whisper-compression-ratio-above-threshold")
        if minimum_average_log_probability is not None and minimum_average_log_probability < -1.0:
            review_reasons.append("whisper-average-logprob-below-threshold")
        if maximum_no_speech_probability is not None and maximum_no_speech_probability > 0.6:
            review_reasons.append("whisper-no-speech-probability-above-threshold")
        total_characters += len(canonical_text)
        if total_characters > _TEXT_LIMIT:
            raise DatasetAsrError("offline ASR output exceeds the transcript size limit")
        if _sha256_file(segment_path) != expected_hash:
            raise DatasetAsrError("ASR segment changed during inference")
        results.append(
            {
                "position": position,
                "segment_sha256": expected_hash,
                "text": canonical_text,
                "language": language,
                "whisper_detected_language": detected_language,
                "whisper_language_probability": probability,
                "whisper_diagnostics": {
                    "maximum_compression_ratio": maximum_compression_ratio,
                    "minimum_average_log_probability": minimum_average_log_probability,
                    "maximum_no_speech_probability": maximum_no_speech_probability,
                },
                "review_required": bool(review_reasons),
                "review_reasons": review_reasons,
            }
        )
    if not any(record["text"] for record in results):
        raise DatasetAsrError("offline ASR produced no usable transcript")
    distinct_languages = set(observed_languages)
    if selected_language is None and len(distinct_languages) != 1:
        raise DatasetAsrError(
            "Whisper detected multiple languages; declare one yue/zh/ja/en/ko dataset language"
        )
    aggregate_language = selected_language or next(iter(distinct_languages))
    try:
        aggregate_text = normalize_transcript(
            " ".join(str(record["text"]) for record in results if record["text"])
        )
    except DatasetQualityError as error:
        raise DatasetAsrError(str(error)) from error
    model_after = inspect_ct2_whisper_model(model_before.root)
    if model_after.as_dict() != model_before.as_dict():
        raise DatasetAsrError("the ASR model changed during inference")
    return {
        "schema": DATASET_ASR_SCHEMA,
        "backend": DATASET_ASR_BACKEND,
        "execution_environment": "linux-docker-cuda-only",
        "network_required": False,
        "model": model_before.as_dict(),
        "model_qualification": "structurally-verified-content-review-required",
        "runtime_versions": versions,
        "decoding": {
            "device": "cuda:0",
            "compute_type": "float16",
            "beam_size": 5,
            "temperature": 0.0,
            "vad_filter": False,
            "condition_on_previous_text": False,
            "local_files_only": True,
        },
        "declared_language": selected_language,
        "language": aggregate_language,
        "text": aggregate_text,
        "segments": results,
        "review_required": any(bool(record["review_required"]) for record in results),
        "review_reasons": list(
            dict.fromkeys(reason for record in results for reason in record["review_reasons"])
        ),
    }


def _sensevoice_model_inventory(model_path: Path) -> dict[str, Any]:
    supplied = Path(model_path).expanduser()
    if supplied.is_symlink():
        raise DatasetAsrError("the SenseVoice model root cannot be a symbolic link")
    try:
        root = supplied.resolve(strict=True)
    except OSError as error:
        raise DatasetAsrError("the local SenseVoice model directory was not found") from error
    if not root.is_dir():
        raise DatasetAsrError("the local SenseVoice model must be a directory")
    records, total_bytes = _model_inventory(root)
    names = {str(record["path"]) for record in records}
    if not any(name.endswith(("model.pt", "model.bin", "model.safetensors")) for name in names):
        raise DatasetAsrError("the SenseVoice model has no local weight file")
    if not any(name.endswith(("config.json", "configuration.json")) for name in names):
        raise DatasetAsrError("the SenseVoice model has no local configuration")
    payload = json.dumps(
        records, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "format": "sensevoice-small",
        "root": root,
        "tree_sha256": hashlib.sha256(payload).hexdigest(),
        "file_count": len(records),
        "total_bytes": total_bytes,
    }


def _load_sensevoice(path: Path) -> tuple[Any, Mapping[str, str]]:
    try:
        import funasr
        from funasr import AutoModel
    except ImportError as error:
        raise DatasetAsrError(
            "offline FunASR/SenseVoice dependencies are missing from the Linux worker"
        ) from error
    try:
        model = AutoModel(
            model=str(path),
            trust_remote_code=False,
            disable_update=True,
            device="cuda:0",
        )
    except Exception as error:
        raise DatasetAsrError("the offline SenseVoice model could not be loaded") from error
    return model, {"funasr": str(getattr(funasr, "__version__", "unknown"))}


def _sensevoice_result(
    value: Any,
) -> tuple[str, str | None, str | None, list[str]]:
    if isinstance(value, list) and value:
        value = value[0]
    if not isinstance(value, Mapping):
        raise DatasetAsrError("SenseVoice returned a malformed result")
    raw = value.get("text")
    if not isinstance(raw, str):
        raise DatasetAsrError("SenseVoice returned no transcript")
    tokens = _SENSEVOICE_TOKEN.findall(raw)
    text = _SENSEVOICE_TOKEN.sub("", raw).strip()
    emotion = next((_SENSEVOICE_EMOTIONS[token.upper()] for token in tokens if token.upper() in _SENSEVOICE_EMOTIONS), None)
    detected_language = next(
        (
            token.casefold()
            for token in tokens
            if token.casefold() in {"zh", "yue", "ja", "en", "ko"}
        ),
        None,
    )
    events = [
        token.casefold()
        for token in tokens
        if token.upper() not in _SENSEVOICE_EMOTIONS
        and token.casefold() not in {"speech", "nospeech", "woitn", "withitn", "auto", "zh", "yue", "ja", "en", "ko"}
    ]
    return text, detected_language, emotion, list(dict.fromkeys(events))


def transcribe_sensevoice_segments(
    segment_records: Sequence[Mapping[str, Any]],
    *,
    stage_root: Path,
    model_path: Path,
    declared_language: str | None = None,
    model_factory: Callable[[Path], tuple[Any, Mapping[str, str]]] | None = None,
) -> dict[str, Any]:
    from .dataset_pipeline import assert_linux_docker_runtime

    assert_linux_docker_runtime()
    if not segment_records:
        raise DatasetAsrError("offline ASR requires at least one verified segment")
    language = None
    if declared_language is not None:
        try:
            language = canonical_language(declared_language)
        except DatasetQualityError as error:
            raise DatasetAsrError(str(error)) from error
    before = _sensevoice_model_inventory(model_path)
    root = before.pop("root")
    model, versions_value = (model_factory or _load_sensevoice)(root)
    versions = {str(key): str(value) for key, value in dict(versions_value).items()}
    results: list[dict[str, Any]] = []
    observed_languages: list[str] = []
    total_characters = 0
    stage = Path(stage_root).resolve(strict=True)
    for record in segment_records:
        segment_path, expected_hash, position = _verified_segment_path(stage, record)
        try:
            generated = model.generate(
                input=str(segment_path),
                cache={},
                language=language or "auto",
                use_itn=True,
                batch_size_s=0,
            )
        except Exception as error:
            raise DatasetAsrError(f"offline SenseVoice failed on segment {position}") from error
        raw_text, detected_language, emotion, events = _sensevoice_result(generated)
        segment_language = language or detected_language
        if segment_language is not None:
            observed_languages.append(segment_language)
        review_reasons = ["offline-asr-requires-human-review"]
        spacing_normalized = False
        if raw_text:
            try:
                text, spacing_normalized = _normalize_sensevoice_spacing(
                    raw_text, segment_language
                )
            except DatasetQualityError as error:
                raise DatasetAsrError(str(error)) from error
        else:
            text = ""
            review_reasons.append("sensevoice-returned-empty-segment")
        if spacing_normalized:
            review_reasons.append("sensevoice-cjk-token-spacing-normalized")
        if text and _has_repeated_text(text):
            review_reasons.append("sensevoice-repetition-detected")
        total_characters += len(text)
        if total_characters > _TEXT_LIMIT:
            raise DatasetAsrError("offline ASR output exceeds the transcript size limit")
        if _sha256_file(segment_path) != expected_hash:
            raise DatasetAsrError("ASR segment changed during inference")
        results.append(
            {
                "position": position,
                "segment_sha256": expected_hash,
                "text": text,
                "language": segment_language,
                "confidence": None,
                "emotion_suggestion": emotion,
                "audio_events": events,
                "review_required": True,
                "review_reasons": review_reasons,
            }
        )
    if not any(record["text"] for record in results):
        raise DatasetAsrError("offline ASR produced no usable transcript")
    after = _sensevoice_model_inventory(root)
    after.pop("root")
    if after != before:
        raise DatasetAsrError("the SenseVoice model changed during inference")
    aggregate = normalize_transcript(
        " ".join(str(record["text"]) for record in results if record["text"])
    )
    distinct_languages = set(observed_languages)
    aggregate_language = (
        language
        if language is not None
        else next(iter(distinct_languages))
        if len(distinct_languages) == 1
        else None
    )
    if language is None and len(distinct_languages) > 1:
        for record in results:
            reasons = record["review_reasons"]
            if "sensevoice-multiple-languages-detected" not in reasons:
                reasons.append("sensevoice-multiple-languages-detected")
    return {
        "schema": DATASET_ASR_SCHEMA,
        "backend": SENSEVOICE_ASR_BACKEND,
        "execution_environment": "linux-docker-cuda-only",
        "network_required": False,
        "model": before,
        "model_qualification": "pinned-assets-human-review-required",
        "runtime_versions": versions,
        "declared_language": language,
        "language": aggregate_language,
        "text": aggregate,
        "segments": results,
        "review_required": True,
        "review_reasons": list(
            dict.fromkeys(reason for record in results for reason in record["review_reasons"])
        ),
    }


@dataclass(frozen=True)
class FasterWhisperBackend:
    model_path: Path
    model_factory: Callable[[Path], tuple[Any, Mapping[str, str]]] | None = None
    backend_id: str = DATASET_ASR_BACKEND

    def transcribe(
        self,
        segment_records: Sequence[Mapping[str, Any]],
        *,
        stage_root: Path,
        declared_language: str | None = None,
    ) -> dict[str, Any]:
        return transcribe_dataset_segments(
            segment_records,
            stage_root=stage_root,
            model_path=self.model_path,
            declared_language=declared_language,
            model_factory=self.model_factory,
        )


@dataclass(frozen=True)
class SenseVoiceBackend:
    model_path: Path
    model_factory: Callable[[Path], tuple[Any, Mapping[str, str]]] | None = None
    backend_id: str = SENSEVOICE_ASR_BACKEND

    def transcribe(
        self,
        segment_records: Sequence[Mapping[str, Any]],
        *,
        stage_root: Path,
        declared_language: str | None = None,
    ) -> dict[str, Any]:
        return transcribe_sensevoice_segments(
            segment_records,
            stage_root=stage_root,
            model_path=self.model_path,
            declared_language=declared_language,
            model_factory=self.model_factory,
        )


def create_workstation_asr_backend(
    backend: str,
    model_path: Path,
) -> WorkstationASRBackend:
    normalized = str(backend).strip().casefold()
    if normalized in {"sensevoice", "sensevoice-small", SENSEVOICE_ASR_BACKEND}:
        return SenseVoiceBackend(Path(model_path))
    if normalized in {"faster-whisper", "whisper", DATASET_ASR_BACKEND}:
        return FasterWhisperBackend(Path(model_path))
    raise DatasetAsrError("Unsupported workstation ASR backend")


__all__ = [
    "DATASET_ASR_BACKEND",
    "DATASET_ASR_SCHEMA",
    "SENSEVOICE_ASR_BACKEND",
    "CTranslate2WhisperModel",
    "DatasetAsrError",
    "FasterWhisperBackend",
    "SenseVoiceBackend",
    "WorkstationASRBackend",
    "create_workstation_asr_backend",
    "inspect_ct2_whisper_model",
    "transcribe_dataset_segments",
    "transcribe_sensevoice_segments",
]
