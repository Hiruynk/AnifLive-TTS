from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4
import wave

import numpy as np

from .dataset_factory import DatasetFactory, EnergyVadConfig, SpeechRegion
from .dataset_quality import (
    DatasetQualityError,
    analyze_pcm_quality,
    canonical_language,
    infer_text_language,
    normalize_transcript,
)


DATASET_PIPELINE_SCHEMA = "aniflive-dataset-pipeline-v1"
FASTTEXT_LID176_SHA256 = (
    "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"
)
SUPPORTED_LANGUAGES = frozenset({"yue", "zh", "ja", "en", "ko"})
MEDIA_SUFFIXES = frozenset(
    {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".mp4", ".mkv", ".mov"}
)

_CANTONESE_MARKERS = frozenset(
    {"係", "唔", "咗", "喺", "嘅", "冇", "啲", "佢", "哋", "咩", "噉", "嗰", "呢", "啦"}
)
_DEFAULT_CONTAINER_MARKERS = (Path("/.dockerenv"), Path("/run/.containerenv"))
_DEFAULT_FASTTEXT_MODELS = (
    Path("/app/pretrained_models/fast_langdetect/lid.176.bin"),
    Path("/opt/aniflive-tts/pretrained_models/fast_langdetect/lid.176.bin"),
)


class DatasetPipelineError(ValueError):
    """Raised when deterministic dataset preparation cannot complete safely."""


@dataclass(frozen=True)
class DatasetPipelineConfig:
    sample_rate: int = 32_000
    max_input_bytes: int = 4 * 1024**3
    max_duration_seconds: float = 60 * 60
    max_samples: int = 32_000 * 60 * 60
    max_output_bytes: int = 256 * 1024**2
    max_artifact_bytes: int = 512 * 1024**2
    max_segments: int = 4_096
    ffprobe_timeout_seconds: float = 30.0
    ffmpeg_timeout_seconds: float = 900.0
    enable_afftdn: bool = False
    afftdn_noise_reduction_db: float = 6.0
    afftdn_noise_floor_db: float = -50.0
    afftdn_gain_smooth: int = 5
    dereverb_backend: str = "none"
    language_confidence_threshold: float = 0.80
    vad: EnergyVadConfig = field(default_factory=EnergyVadConfig)

    def validated(self) -> "DatasetPipelineConfig":
        if self.sample_rate != 32_000:
            raise DatasetPipelineError("canonical sample_rate must remain 32000 Hz")
        if not 1 <= self.max_input_bytes <= 16 * 1024**3:
            raise DatasetPipelineError("max_input_bytes is outside the supported range")
        if (
            not math.isfinite(self.max_duration_seconds)
            or not 0.1 <= self.max_duration_seconds <= 3_600
        ):
            raise DatasetPipelineError(
                "standard dataset max_duration_seconds must be between 0.1 and 3600"
            )
        if not 1 <= self.max_samples <= self.sample_rate * 3_600:
            raise DatasetPipelineError(
                "standard dataset max_samples is outside the supported range"
            )
        if not 45 <= self.max_output_bytes <= 512 * 1024**2:
            raise DatasetPipelineError("max_output_bytes is outside the supported range")
        if not self.max_output_bytes <= self.max_artifact_bytes <= 1024**3:
            raise DatasetPipelineError(
                "max_artifact_bytes must cover canonical output and stay below 1 GiB"
            )
        if not 1 <= self.max_segments <= 100_000:
            raise DatasetPipelineError("max_segments is outside the supported range")
        if (
            not math.isfinite(self.ffprobe_timeout_seconds)
            or not 1 <= self.ffprobe_timeout_seconds <= 600
        ):
            raise DatasetPipelineError(
                "ffprobe_timeout_seconds must be between 1 and 600"
            )
        if (
            not math.isfinite(self.ffmpeg_timeout_seconds)
            or not 1 <= self.ffmpeg_timeout_seconds <= 172_800
        ):
            raise DatasetPipelineError(
                "ffmpeg_timeout_seconds must be between 1 and 172800"
            )
        if (
            not math.isfinite(self.afftdn_noise_reduction_db)
            or not 0.01 <= self.afftdn_noise_reduction_db <= 12.0
        ):
            raise DatasetPipelineError(
                "conservative afftdn reduction must be between 0.01 and 12 dB"
            )
        if (
            not math.isfinite(self.afftdn_noise_floor_db)
            or not -80 <= self.afftdn_noise_floor_db <= -20
        ):
            raise DatasetPipelineError(
                "afftdn noise floor must be between -80 and -20 dBFS"
            )
        if not 0 <= self.afftdn_gain_smooth <= 50:
            raise DatasetPipelineError("afftdn_gain_smooth must be between 0 and 50")
        if self.dereverb_backend != "none":
            raise DatasetPipelineError(
                "dereverb_backend must be 'none'; v1.4 has no redistribution-safe "
                "qualified dereverb backend"
            )
        if (
            not math.isfinite(self.language_confidence_threshold)
            or not 0 < self.language_confidence_threshold <= 1
        ):
            raise DatasetPipelineError(
                "language_confidence_threshold must be in (0, 1]"
            )
        try:
            self.vad.validated()
        except ValueError as error:
            raise DatasetPipelineError(str(error)) from error
        return self


@dataclass(frozen=True)
class MediaProbe:
    audio_stream_index: int
    codec_name: str
    source_sample_rate: int
    source_channels: int
    duration_seconds: float
    format_name: str
    estimated_output_samples: int
    estimated_output_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_text(value: bytes, *, limit: int = 32_768) -> str:
    return value[:limit].decode("utf-8", errors="replace").strip()


def _round_seconds(value: float) -> float:
    return round(max(0.0, float(value)), 6)


def is_linux_docker_runtime(
    *,
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    marker_paths: Sequence[Path] | None = None,
    cgroup_text: str | None = None,
) -> bool:
    platform_value = platform_name if platform_name is not None else sys.platform
    if not platform_value.startswith("linux"):
        return False
    environment = os.environ if environ is None else environ
    explicit_worker = environment.get(
        "ANIFLIVE_TTS_WORKER_CONTAINER", ""
    ).strip()
    if explicit_worker not in {"", "1"}:
        return False
    markers = _DEFAULT_CONTAINER_MARKERS if marker_paths is None else marker_paths
    has_marker = any(Path(marker).is_file() for marker in markers)
    if cgroup_text is None:
        try:
            cgroup_text = Path("/proc/1/cgroup").read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            cgroup_text = ""
    lowered = cgroup_text.casefold()
    has_cgroup_evidence = any(
        marker in lowered for marker in ("docker", "containerd", "kubepods", "podman")
    )
    return has_marker or has_cgroup_evidence


def assert_linux_docker_runtime(**overrides: Any) -> None:
    if not is_linux_docker_runtime(**overrides):
        raise DatasetPipelineError(
            "dataset media execution is supported only inside the "
            "AnifLive-TTS Linux Docker worker"
        )


def _strict_input_file(path: Path, config: DatasetPipelineConfig) -> Path:
    supplied = Path(path).expanduser()
    if supplied.is_symlink():
        raise DatasetPipelineError("media input cannot be a symbolic link")
    try:
        source = supplied.resolve(strict=True)
    except OSError as error:
        raise DatasetPipelineError("media input does not exist") from error
    if not source.is_file():
        raise DatasetPipelineError("media input must be a regular file")
    if source.suffix.casefold() not in MEDIA_SUFFIXES:
        raise DatasetPipelineError(f"unsupported media suffix: {source.suffix}")
    byte_count = source.stat().st_size
    if byte_count < 1:
        raise DatasetPipelineError("media input cannot be empty")
    if byte_count > config.max_input_bytes:
        raise DatasetPipelineError("media input exceeds max_input_bytes")
    return source


def _copy_verified_input(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    copied_bytes = 0
    digest = hashlib.sha256()
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            while chunk := reader.read(1024 * 1024):
                copied_bytes += len(chunk)
                if copied_bytes > expected_bytes:
                    raise DatasetPipelineError(
                        "media input changed while it was being staged"
                    )
                digest.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
    except OSError as error:
        raise DatasetPipelineError("media input could not be staged safely") from error
    if copied_bytes != expected_bytes or digest.hexdigest() != expected_sha256:
        raise DatasetPipelineError("media input changed while it was being staged")


def _positive_int(value: Any, *, field_name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or (
        isinstance(value, float) and not value.is_integer()
    ):
        raise DatasetPipelineError(f"ffprobe returned invalid {field_name}")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise DatasetPipelineError(f"ffprobe returned invalid {field_name}") from error
    minimum = 0 if allow_zero else 1
    if result < minimum:
        raise DatasetPipelineError(f"ffprobe returned invalid {field_name}")
    return result


def _positive_float(value: Any, *, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise DatasetPipelineError(f"ffprobe returned invalid {field_name}") from error
    if not math.isfinite(result) or result <= 0:
        raise DatasetPipelineError(f"ffprobe returned invalid {field_name}")
    return result


def parse_ffprobe_document(
    document: Mapping[str, Any],
    config: DatasetPipelineConfig | None = None,
) -> MediaProbe:
    settings = (config or DatasetPipelineConfig()).validated()
    if not isinstance(document, Mapping):
        raise DatasetPipelineError("ffprobe output must be a JSON object")
    streams = document.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise DatasetPipelineError(
            "ffprobe must return exactly the selected first audio stream"
        )
    stream = streams[0]
    if not isinstance(stream, Mapping) or stream.get("codec_type") != "audio":
        raise DatasetPipelineError("input has no valid first audio stream")
    source_rate = _positive_int(stream.get("sample_rate"), field_name="sample_rate")
    channels = _positive_int(stream.get("channels"), field_name="channels")
    if channels > 64:
        raise DatasetPipelineError("audio channel count exceeds the safety limit")
    format_value = document.get("format")
    format_data = format_value if isinstance(format_value, Mapping) else {}
    duration_value = stream.get("duration")
    if duration_value is None or duration_value == "" or duration_value == "N/A":
        duration_value = format_data.get("duration")
    duration = _positive_float(duration_value, field_name="duration")
    if duration > settings.max_duration_seconds:
        raise DatasetPipelineError("media duration exceeds max_duration_seconds")
    estimated_samples = int(math.ceil(duration * settings.sample_rate))
    if estimated_samples > settings.max_samples:
        raise DatasetPipelineError("decoded audio would exceed max_samples")
    estimated_bytes = 44 + estimated_samples * 2
    if estimated_bytes > settings.max_output_bytes:
        raise DatasetPipelineError("decoded audio would exceed max_output_bytes")
    return MediaProbe(
        audio_stream_index=_positive_int(
            stream.get("index", 0), field_name="stream_index", allow_zero=True
        ),
        codec_name=str(stream.get("codec_name") or "unknown")[:80],
        source_sample_rate=source_rate,
        source_channels=channels,
        duration_seconds=duration,
        format_name=str(format_data.get("format_name") or "unknown")[:160],
        estimated_output_samples=estimated_samples,
        estimated_output_bytes=estimated_bytes,
    )


def probe_media(
    source_path: Path,
    *,
    config: DatasetPipelineConfig | None = None,
    ffprobe_binary: str = "ffprobe",
) -> MediaProbe:
    assert_linux_docker_runtime()
    settings = (config or DatasetPipelineConfig()).validated()
    source = _strict_input_file(source_path, settings)
    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-select_streams",
        "a:0",
        "-show_entries",
        (
            "stream=index,codec_name,codec_type,sample_rate,channels,duration:"
            "format=duration,format_name"
        ),
        "-of",
        "json",
        str(source),
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=settings.ffprobe_timeout_seconds,
        )
    except FileNotFoundError as error:
        raise DatasetPipelineError(
            "ffprobe is not installed in the Linux worker"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise DatasetPipelineError("ffprobe exceeded its hard timeout") from error
    if result.returncode != 0:
        detail = _bounded_text(result.stderr) or "unknown ffprobe error"
        raise DatasetPipelineError(f"ffprobe rejected media: {detail}")
    if len(result.stdout) > 256 * 1024:
        raise DatasetPipelineError("ffprobe output exceeded the safety limit")
    try:
        document = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DatasetPipelineError("ffprobe returned malformed JSON") from error
    return parse_ffprobe_document(document, settings)


def _read_canonical_wav(
    path: Path,
    config: DatasetPipelineConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            compression = audio.getcomptype()
            raw = audio.readframes(frame_count)
    except (OSError, EOFError, wave.Error) as error:
        raise DatasetPipelineError("ffmpeg did not produce a valid PCM WAV") from error
    if compression != "NONE" or channels != 1 or sample_width != 2:
        raise DatasetPipelineError("canonical WAV must be mono PCM16")
    if sample_rate != config.sample_rate:
        raise DatasetPipelineError("canonical WAV has the wrong sample rate")
    if frame_count < 1:
        raise DatasetPipelineError("canonical WAV is empty")
    if frame_count > config.max_samples:
        raise DatasetPipelineError("canonical WAV exceeds max_samples")
    if frame_count / sample_rate > config.max_duration_seconds + (1 / sample_rate):
        raise DatasetPipelineError("canonical WAV exceeds max_duration_seconds")
    if path.stat().st_size >= config.max_output_bytes:
        raise DatasetPipelineError("canonical WAV exceeds max_output_bytes")
    if len(raw) != frame_count * 2:
        raise DatasetPipelineError("canonical WAV payload is truncated")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if not np.isfinite(samples).all():
        raise DatasetPipelineError("canonical WAV contains invalid samples")
    return samples, {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
        "duration_seconds": frame_count / sample_rate,
        "byte_count": path.stat().st_size,
    }


def _write_pcm16_mono(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise DatasetPipelineError("segment samples must be finite mono audio")
    pcm = np.rint(np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with wave.open(str(temporary), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(sample_rate)
            audio.writeframes(pcm.tobytes(order="C"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _afftdn_filter(config: DatasetPipelineConfig) -> str:
    return (
        f"afftdn=nr={config.afftdn_noise_reduction_db:g}:"
        f"nf={config.afftdn_noise_floor_db:g}:tn=0:"
        f"gs={config.afftdn_gain_smooth}"
    )


def decode_media(
    source_path: Path,
    destination_path: Path,
    *,
    probe: MediaProbe | None = None,
    config: DatasetPipelineConfig | None = None,
    ffmpeg_binary: str = "ffmpeg",
) -> dict[str, Any]:
    assert_linux_docker_runtime()
    settings = (config or DatasetPipelineConfig()).validated()
    source = _strict_input_file(source_path, settings)
    media_probe = probe or probe_media(source, config=settings)
    if media_probe.duration_seconds > settings.max_duration_seconds:
        raise DatasetPipelineError("probe exceeds the configured duration cap")
    destination = Path(destination_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp.wav")
    filter_arguments: list[str] = []
    if settings.enable_afftdn:
        filter_arguments = ["-af", _afftdn_filter(settings)]
    command = [
        ffmpeg_binary,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-xerror",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-t",
        f"{settings.max_duration_seconds:.9f}",
        *filter_arguments,
        "-ac",
        "1",
        "-ar",
        str(settings.sample_rate),
        "-sample_fmt",
        "s16",
        "-c:a",
        "pcm_s16le",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:a",
        "+bitexact",
        "-threads",
        "1",
        "-fs",
        str(settings.max_output_bytes),
        "-f",
        "wav",
        "-y",
        str(temporary),
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=settings.ffmpeg_timeout_seconds,
        )
    except FileNotFoundError as error:
        raise DatasetPipelineError(
            "ffmpeg is not installed in the Linux worker"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise DatasetPipelineError("ffmpeg exceeded its hard timeout") from error
    try:
        if result.returncode != 0:
            detail = _bounded_text(result.stderr) or "unknown ffmpeg error"
            raise DatasetPipelineError(f"ffmpeg rejected media: {detail}")
        samples, metadata = _read_canonical_wav(temporary, settings)
        sample_tolerance = max(64, round(settings.sample_rate * 0.05))
        if abs(samples.size - media_probe.estimated_output_samples) > sample_tolerance:
            raise DatasetPipelineError(
                "decoded sample count materially disagrees with ffprobe duration"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        **metadata,
        "sha256": sha256_file(destination),
        "decoder": "ffmpeg-first-audio-pcm16-v1",
        "denoise": {
            "enabled": settings.enable_afftdn,
            "backend": "ffmpeg-afftdn" if settings.enable_afftdn else "none",
            "filter": _afftdn_filter(settings) if settings.enable_afftdn else None,
        },
        "dereverb": {
            "enabled": False,
            "backend": "none",
            "qualification": "no-qualified-redistribution-safe-backend",
        },
    }


def energy_speech_regions(
    samples: np.ndarray,
    sample_rate: int,
    config: EnergyVadConfig | None = None,
) -> list[SpeechRegion]:
    settings = (config or EnergyVadConfig()).validated()
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] < 1:
        raise DatasetPipelineError("VAD input must be mono or frame-major PCM")
    if values.shape[0] < 1 or not np.isfinite(values).all():
        raise DatasetPipelineError("VAD input must contain finite samples")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise DatasetPipelineError("sample_rate must be positive")
    return DatasetFactory._speech_regions(values, sample_rate, settings)


def resolve_fasttext_model(model_path: Path | None = None) -> Path:
    candidates = [Path(model_path)] if model_path is not None else list(_DEFAULT_FASTTEXT_MODELS)
    for candidate_value in candidates:
        candidate = candidate_value.expanduser()
        if candidate.is_symlink():
            raise DatasetPipelineError("fastText language model cannot be a symbolic link")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if not resolved.is_file():
            continue
        digest = sha256_file(resolved)
        if digest != FASTTEXT_LID176_SHA256:
            raise DatasetPipelineError(
                "fastText lid.176.bin failed its pinned SHA-256 integrity check"
            )
        return resolved
    raise DatasetPipelineError(
        "the pinned offline fastText lid.176.bin asset is unavailable"
    )


@lru_cache(maxsize=4)
def _cached_fasttext_detector(model_path: str, model_sha256: str) -> Any:
    if model_sha256 != FASTTEXT_LID176_SHA256:
        raise DatasetPipelineError("unexpected fastText language model digest")
    try:
        module = importlib.import_module("fast_langdetect")
        config_type = getattr(module, "LangDetectConfig")
        detector_type = getattr(module, "LangDetector")
    except (ImportError, AttributeError) as error:
        raise DatasetPipelineError(
            "fast-langdetect is unavailable in the Linux worker"
        ) from error
    config = config_type(
        custom_model_path=model_path,
        allow_fallback=False,
        disable_verify=True,
    )
    return detector_type(config)


def load_fasttext_detector(model_path: Path | None = None) -> Any:
    assert_linux_docker_runtime()
    resolved = resolve_fasttext_model(model_path)
    return _cached_fasttext_detector(str(resolved), sha256_file(resolved))


def _run_language_detector(detector: Any, text: str) -> Mapping[str, Any]:
    try:
        if hasattr(detector, "detect"):
            result = detector.detect(text, low_memory=True)
        elif callable(detector):
            result = detector(text)
        else:
            raise TypeError("detector is not callable")
    except Exception as error:
        raise DatasetPipelineError("offline fastText language detection failed") from error
    if not isinstance(result, Mapping):
        raise DatasetPipelineError("fastText language detector returned an invalid result")
    return result


def detect_dataset_language(
    text: str,
    *,
    detector: Any | None = None,
    model_path: Path | None = None,
    confidence_threshold: float = 0.80,
) -> dict[str, Any]:
    canonical_text = normalize_transcript(text)
    if not math.isfinite(confidence_threshold) or not 0 < confidence_threshold <= 1:
        raise DatasetPipelineError("confidence_threshold must be in (0, 1]")
    injected_detector = detector is not None
    active_detector = detector if injected_detector else load_fasttext_detector(model_path)
    result = _run_language_detector(active_detector, canonical_text)
    raw_label = str(result.get("lang") or "").strip().casefold()
    if raw_label.startswith("__label__"):
        raw_label = raw_label.removeprefix("__label__")
    try:
        score = float(result.get("score"))
    except (TypeError, ValueError) as error:
        raise DatasetPipelineError("fastText returned an invalid confidence score") from error
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise DatasetPipelineError("fastText returned an invalid confidence score")
    label_aliases = {
        "en": "en",
        "ja": "ja",
        "jp": "ja",
        "ko": "ko",
        "kr": "ko",
        "zh": "zh",
        "cmn": "zh",
        "yue": "yue",
    }
    model_language = label_aliases.get(raw_label, "und")
    marker_hits = sorted({character for character in canonical_text if character in _CANTONESE_MARKERS})
    suggested_language = (
        "yue"
        if model_language == "zh" and len(marker_hits) >= 2
        else model_language
    )
    review_reasons: list[str] = []
    if model_language not in SUPPORTED_LANGUAGES:
        review_reasons.append("unsupported-or-undetermined-fasttext-label")
    if score < confidence_threshold:
        review_reasons.append("fasttext-confidence-below-threshold")
    if model_language in {"zh", "yue"} or suggested_language == "yue":
        review_reasons.append("zh-yue-requires-human-review")
    if suggested_language != model_language:
        review_reasons.append("cantonese-lexeme-suggestion-is-diagnostic-only")
    return {
        "model": (
            "injected-language-detector"
            if injected_detector
            else "fastText-lid.176.bin"
        ),
        "model_sha256": None if injected_detector else FASTTEXT_LID176_SHA256,
        "provider_verified": not injected_detector,
        "raw_label": raw_label,
        "model_language": model_language,
        "model_score": round(score, 6),
        "suggested_language": suggested_language,
        "review_required": bool(review_reasons),
        "review_reasons": review_reasons,
        "cantonese_diagnostic": {
            "marker_hits": marker_hits,
            "suggested": suggested_language == "yue" and model_language != "yue",
            "classifier": False,
        },
        "script_diagnostic": infer_text_language(canonical_text),
    }


def _normalizer_roots() -> list[Path]:
    configured = os.environ.get("ANIFLIVE_TTS_GPT_SOVITS_ROOT", "").strip()
    roots = [
        Path(configured) if configured else None,
        Path("/app/minimal_inference/GPT_SoVITS"),
        Path(__file__).resolve().parents[2] / "minimal_inference" / "GPT_SoVITS",
        Path("/workspace/minimal_inference/GPT_SoVITS"),
        Path("/opt/aniflive-tts/gpt-sovits/GPT_SoVITS"),
    ]
    return [root for root in roots if root is not None]


@lru_cache(maxsize=1)
def load_gpt_sovits_normalizer() -> Callable[[str, str], str]:
    roots = [
        root.resolve()
        for root in _normalizer_roots()
        if (root / "text" / "cleaner.py").is_file()
    ]
    if not roots:
        raise DatasetPipelineError(
            "GPT-SoVITS five-language text normalizer is unavailable"
        )
    # Honor configured/bundled source priority even if a runner or an earlier
    # training import already added one of these roots to PYTHONPATH.
    search_paths = list(dict.fromkeys(str(path) for root in roots for path in (root, root.parent)))
    sys.path[:] = search_paths + [path for path in sys.path if path not in search_paths]
    # Training imports can pre-bind the generic `text` namespace to another root.
    for module_name in tuple(sys.modules):
        if module_name == "text" or module_name.startswith("text."):
            sys.modules.pop(module_name, None)
    try:
        cleaner_module = importlib.import_module("text.cleaner")
        clean_text = getattr(cleaner_module, "clean_text")
    except (ImportError, AttributeError) as error:
        raise DatasetPipelineError(
            "GPT-SoVITS five-language text normalizer is unavailable"
        ) from error
    module_file_value = getattr(cleaner_module, "__file__", None)
    if not module_file_value:
        raise DatasetPipelineError("GPT-SoVITS normalizer has no verifiable source file")
    module_file = Path(module_file_value).resolve()
    matching_root = next(
        (
            root
            for root in roots
            if module_file == root / "text" / "cleaner.py"
        ),
        None,
    )
    if matching_root is None:
        raise DatasetPipelineError(
            "imported text.cleaner is not from a trusted GPT-SoVITS source root"
        )
    provenance = {
        "name": "gpt-sovits-v2-five-language",
        "provider_verified": True,
        "source": "text/cleaner.py",
        "source_sha256": sha256_file(module_file),
    }

    def normalize(value: str, language: str) -> str:
        try:
            _phones, _word2ph, normalized = clean_text(value, language, "v2")
        except Exception as error:
            raise DatasetPipelineError(
                f"GPT-SoVITS text normalization failed for language {language}"
            ) from error
        if not isinstance(normalized, str):
            raise DatasetPipelineError("GPT-SoVITS normalizer returned invalid text")
        return normalize_transcript(normalized)

    setattr(normalize, "__aniflive_provenance__", provenance)
    return normalize


def normalize_dataset_transcript(
    transcript: str,
    language: str,
    *,
    normalizer: Callable[[str, str], str] | None = None,
) -> dict[str, Any]:
    if not isinstance(transcript, str):
        raise DatasetPipelineError("transcript must be text")
    original_text = transcript
    canonical_text = normalize_transcript(transcript)
    try:
        canonical_lang = canonical_language(language)
    except DatasetQualityError as error:
        raise DatasetPipelineError(str(error)) from error
    if canonical_lang not in SUPPORTED_LANGUAGES:
        raise DatasetPipelineError(f"unsupported dataset language: {canonical_lang}")
    injected_normalizer = normalizer is not None
    active_normalizer = normalizer or load_gpt_sovits_normalizer()
    try:
        normalized_text = active_normalizer(canonical_text, canonical_lang)
    except DatasetPipelineError:
        raise
    except Exception as error:
        raise DatasetPipelineError("text normalization failed") from error
    normalized_text = normalize_transcript(normalized_text)
    provenance = (
        {
            "name": "injected-normalizer",
            "provider_verified": False,
            "source": None,
            "source_sha256": None,
        }
        if injected_normalizer
        else getattr(active_normalizer, "__aniflive_provenance__")
    )
    return {
        "language": canonical_lang,
        "original_text": original_text,
        "canonical_input_text": canonical_text,
        "normalized_text": normalized_text,
        "normalizer": provenance,
    }


def build_transcript_record(
    transcript: str,
    *,
    declared_language: str | None = None,
    detector: Any | None = None,
    model_path: Path | None = None,
    normalizer: Callable[[str, str], str] | None = None,
    confidence_threshold: float = 0.80,
) -> dict[str, Any]:
    language_evidence = detect_dataset_language(
        transcript,
        detector=detector,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
    )
    review_reasons = list(language_evidence["review_reasons"])
    if declared_language is not None:
        try:
            selected_language = canonical_language(declared_language)
        except DatasetQualityError as error:
            raise DatasetPipelineError(str(error)) from error
        model_language = language_evidence["model_language"]
        if (
            model_language in SUPPORTED_LANGUAGES
            and {selected_language, model_language} != {"zh", "yue"}
            and selected_language != model_language
        ):
            review_reasons.append("declared-language-disagrees-with-fasttext")
    else:
        selected_language = str(language_evidence["suggested_language"])
        if selected_language not in SUPPORTED_LANGUAGES:
            raise DatasetPipelineError(
                "language could not be selected without a declared language"
            )
    normalized = normalize_dataset_transcript(
        transcript,
        selected_language,
        normalizer=normalizer,
    )
    return {
        **normalized,
        "language_evidence": language_evidence,
        "review_required": bool(review_reasons),
        "review_reasons": list(dict.fromkeys(review_reasons)),
    }


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DatasetPipelineError("dataset report is not canonical JSON") from error


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json_bytes(value)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stage(
    name: str,
    elapsed_seconds: float,
    *,
    status: str = "completed",
    **details: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "elapsed_seconds": _round_seconds(elapsed_seconds),
        **details,
    }


def _config_payload(config: DatasetPipelineConfig) -> dict[str, Any]:
    result = asdict(config)
    result["vad"] = asdict(config.vad)
    return result


def run_dataset_pipeline(
    source_path: Path,
    output_directory: Path,
    *,
    transcript: str | None = None,
    declared_language: str | None = None,
    config: DatasetPipelineConfig | None = None,
    detector: Any | None = None,
    fasttext_model_path: Path | None = None,
    asr_model_path: Path | None = None,
    asr_backend: str = "faster-whisper",
    vad_model_path: Path | None = None,
    normalizer: Callable[[str, str], str] | None = None,
    ffprobe_binary: str = "ffprobe",
    ffmpeg_binary: str = "ffmpeg",
) -> dict[str, Any]:
    assert_linux_docker_runtime()
    settings = (config or DatasetPipelineConfig()).validated()
    source = _strict_input_file(source_path, settings)
    destination = Path(output_directory).expanduser()
    if destination.exists() or destination.is_symlink():
        raise DatasetPipelineError("output_directory must not already exist")
    destination_parent = destination.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    destination_parent = destination_parent.resolve(strict=True)
    destination = destination_parent / destination.name
    stage_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".dataset-pipeline",
            dir=destination_parent,
        )
    )
    stages: list[dict[str, Any]] = []
    source_hash = sha256_file(source)
    source_size = source.stat().st_size
    try:
        started = time.perf_counter()
        staged_source = stage_root / f"source-input{source.suffix.casefold()}"
        _copy_verified_input(
            source,
            staged_source,
            expected_sha256=source_hash,
            expected_bytes=source_size,
        )
        stages.append(_stage("stage-input", time.perf_counter() - started))

        started = time.perf_counter()
        media_probe = probe_media(
            staged_source,
            config=settings,
            ffprobe_binary=ffprobe_binary,
        )
        stages.append(_stage("probe", time.perf_counter() - started))

        canonical_path = stage_root / "canonical.wav"
        started = time.perf_counter()
        decode_report = decode_media(
            staged_source,
            canonical_path,
            probe=media_probe,
            config=settings,
            ffmpeg_binary=ffmpeg_binary,
        )
        stages.append(
            _stage(
                "decode",
                time.perf_counter() - started,
                denoise_backend=decode_report["denoise"]["backend"],
            )
        )
        staged_source.unlink()

        started = time.perf_counter()
        samples, canonical_metadata = _read_canonical_wav(canonical_path, settings)
        artifact_bytes = canonical_path.stat().st_size
        try:
            quality = analyze_pcm_quality(samples, settings.sample_rate).as_dict()
        except DatasetQualityError as error:
            raise DatasetPipelineError(str(error)) from error
        stages.append(_stage("quality", time.perf_counter() - started))

        started = time.perf_counter()
        vad_report: dict[str, Any]
        if vad_model_path is None:
            regions = energy_speech_regions(samples, settings.sample_rate, settings.vad)
            vad_report = {
                "backend": "energy-vad-v1",
                "execution_environment": "linux-docker-cpu",
                "network_required": False,
                "segment_count": len(regions),
            }
        else:
            try:
                from .workstation_vad import WorkstationFsmnVad, WorkstationVadError

                mono = np.asarray(samples, dtype=np.float32).reshape(-1)
                if settings.sample_rate != 32_000:
                    raise DatasetPipelineError(
                        "managed FSMN-VAD requires the canonical 32000 Hz pipeline"
                    )
                vad_segments, vad_report = WorkstationFsmnVad(vad_model_path).detect(
                    mono[::2],
                    16_000,
                    minimum_speech_ms=float(settings.vad.min_speech_ms),
                    maximum_segment_ms=float(settings.vad.max_segment_ms),
                    context_ms=float(settings.vad.pad_ms),
                )
            except WorkstationVadError as error:
                raise DatasetPipelineError(str(error)) from error
            regions = []
            for segment in vad_segments:
                start = max(0, min(mono.size, int(segment.start_sample) * 2))
                end = max(start, min(mono.size, int(segment.end_sample) * 2))
                window = mono[start:end]
                peak = float(np.max(np.abs(window))) if window.size else 0.0
                rms = float(
                    np.sqrt(np.mean(np.square(window, dtype=np.float64)) + 1e-12)
                ) if window.size else 0.0
                regions.append(
                    SpeechRegion(
                        start_frame=start,
                        end_frame=end,
                        peak_dbfs=20.0 * math.log10(max(peak, 1e-8)),
                        mean_dbfs=20.0 * math.log10(max(rms, 1e-8)),
                    )
                )
            vad_report = {
                **vad_report,
                "canonical_source_sample_rate": settings.sample_rate,
                "segment_count": len(regions),
            }
        if len(regions) > settings.max_segments:
            raise DatasetPipelineError("VAD produced more than max_segments")
        segment_root = stage_root / "segments"
        segment_root.mkdir(parents=True, exist_ok=True)
        segment_reports: list[dict[str, Any]] = []
        for position, region in enumerate(regions):
            segment_path = segment_root / f"segment-{position:05d}.wav"
            segment_samples = samples[region.start_frame : region.end_frame]
            predicted_segment_bytes = 44 + int(segment_samples.size) * 2
            if artifact_bytes + predicted_segment_bytes > settings.max_artifact_bytes:
                raise DatasetPipelineError(
                    "canonical audio and segments exceed max_artifact_bytes"
                )
            _write_pcm16_mono(segment_path, segment_samples, settings.sample_rate)
            artifact_bytes += segment_path.stat().st_size
            if artifact_bytes > settings.max_artifact_bytes:
                raise DatasetPipelineError(
                    "canonical audio and segments exceed max_artifact_bytes"
                )
            try:
                segment_quality = analyze_pcm_quality(
                    segment_samples, settings.sample_rate
                ).as_dict()
            except DatasetQualityError as error:
                raise DatasetPipelineError(str(error)) from error
            segment_reports.append(
                {
                    "position": position,
                    "path": segment_path.relative_to(stage_root).as_posix(),
                    "sha256": sha256_file(segment_path),
                    "byte_count": segment_path.stat().st_size,
                    "region": region.as_dict(settings.sample_rate),
                    "quality": segment_quality,
                }
            )
        stages.append(
            _stage(
                "vad-and-segment",
                time.perf_counter() - started,
                backend=vad_report["backend"],
                segment_count=len(segment_reports),
            )
        )
        audio_artifact_bytes = artifact_bytes

        started = time.perf_counter()
        asr_report: dict[str, Any] | None = None
        asr_artifact: dict[str, Any] | None = None
        if asr_model_path is not None:
            if transcript is not None:
                raise DatasetPipelineError(
                    "use either a supplied transcript or offline ASR, not both"
                )
            try:
                from .dataset_asr import (
                    DatasetAsrError,
                    create_workstation_asr_backend,
                )

                workstation_asr = create_workstation_asr_backend(
                    asr_backend, asr_model_path
                )
                asr_report = workstation_asr.transcribe(
                    segment_reports,
                    stage_root=stage_root,
                    declared_language=declared_language,
                )
            except DatasetAsrError as error:
                raise DatasetPipelineError(str(error)) from error
            normalized_segments: list[dict[str, Any]] = []
            by_position = {int(item["position"]): item for item in segment_reports}
            for asr_segment in asr_report["segments"]:
                segment_position = int(asr_segment["position"])
                segment_text = str(asr_segment["text"])
                segment_language = asr_segment.get("language")
                normalized_record: dict[str, Any] | None = None
                if segment_text and isinstance(segment_language, str):
                    normalized_record = normalize_dataset_transcript(
                        segment_text,
                        segment_language,
                        normalizer=normalizer,
                    )
                transcript_payload = {
                    **dict(asr_segment),
                    "source": "offline-asr",
                    "normalized": normalized_record,
                }
                by_position[segment_position]["transcript"] = transcript_payload
                normalized_segments.append(transcript_payload)
            asr_report = {**asr_report, "segments": normalized_segments}
            aggregate_language = asr_report.get("language")
            if isinstance(aggregate_language, str):
                normalized = normalize_dataset_transcript(
                    str(asr_report["text"]),
                    aggregate_language,
                    normalizer=normalizer,
                )
                transcript_record = {
                    **normalized,
                    "source": "offline-asr",
                    "asr_backend": asr_report["backend"],
                    "asr_model_tree_sha256": asr_report["model"]["tree_sha256"],
                    "review_required": bool(asr_report["review_required"]),
                    "review_reasons": list(asr_report["review_reasons"]),
                }
            else:
                transcript_record = {
                    "original_text": str(asr_report["text"]),
                    "canonical_input_text": str(asr_report["text"]),
                    "normalized_text": str(asr_report["text"]),
                    "language": None,
                    "source": "offline-asr",
                    "asr_backend": asr_report["backend"],
                    "asr_model_tree_sha256": asr_report["model"]["tree_sha256"],
                    "review_required": True,
                    "review_reasons": list(
                        dict.fromkeys(
                            [
                                *asr_report["review_reasons"],
                                "asr-language-requires-human-confirmation",
                            ]
                        )
                    ),
                }
            asr_path = stage_root / "asr-transcripts.json"
            _write_canonical_json(asr_path, asr_report)
            artifact_bytes += asr_path.stat().st_size
            if artifact_bytes > settings.max_artifact_bytes:
                raise DatasetPipelineError(
                    "canonical audio, segments and ASR report exceed max_artifact_bytes"
                )
            asr_artifact = {
                "path": "asr-transcripts.json",
                "sha256": sha256_file(asr_path),
                "byte_count": asr_path.stat().st_size,
                "schema": asr_report["schema"],
            }
            stages.append(
                _stage(
                    "offline-asr",
                    time.perf_counter() - started,
                    backend=asr_report["backend"],
                    segment_count=len(normalized_segments),
                    review_required=transcript_record["review_required"],
                )
            )
        elif transcript is None:
            if declared_language is not None:
                raise DatasetPipelineError(
                    "declared_language requires a transcript or offline ASR model"
                )
            transcript_record = None
            stages.append(
                _stage(
                    "transcript",
                    time.perf_counter() - started,
                    status="skipped",
                    reason="no-transcript-supplied",
                )
            )
        else:
            transcript_record = build_transcript_record(
                transcript,
                declared_language=declared_language,
                detector=detector,
                model_path=fasttext_model_path,
                normalizer=normalizer,
                confidence_threshold=settings.language_confidence_threshold,
            )
            stages.append(
                _stage(
                    "transcript",
                    time.perf_counter() - started,
                    review_required=transcript_record["review_required"],
                )
            )

        config_payload = _config_payload(settings)
        identity_payload = {
            "schema": DATASET_PIPELINE_SCHEMA,
            "source_sha256": source_hash,
            "canonical_sha256": decode_report["sha256"],
            "config": config_payload,
            "transcript": transcript_record,
            "asr": asr_report,
            "vad": vad_report,
            "segments": [
                {
                    "position": item["position"],
                    "sha256": item["sha256"],
                    "region": item["region"],
                }
                for item in segment_reports
            ],
        }
        pipeline_identity = hashlib.sha256(
            _canonical_json_bytes(identity_payload)
        ).hexdigest()
        report: dict[str, Any] = {
            "schema": DATASET_PIPELINE_SCHEMA,
            "pipeline_identity_sha256": pipeline_identity,
            "determinism": {
                "canonical_audio": "ffmpeg-bitexact-pcm16-single-thread-v1",
                "report_serialization": "utf8-sorted-canonical-json-v1",
                "stage_timings_excluded_from_pipeline_identity": True,
            },
            "runtime": {
                "execution_environment": "linux-docker-only",
                "network_required": False,
                "neural_fallback": False,
                "neural_inference": [
                    *(
                        [str(vad_report["backend"])]
                        if vad_report["backend"] != "energy-vad-v1"
                        else []
                    ),
                    *([str(asr_report["backend"])] if asr_report is not None else []),
                ],
            },
            "config": config_payload,
            "input": {
                "name": source.name,
                "suffix": source.suffix.casefold(),
                "byte_count": source_size,
                "sha256": source_hash,
                "probe": media_probe.as_dict(),
            },
            "output": {
                "artifact_bytes": artifact_bytes,
                "audio_artifact_bytes": audio_artifact_bytes,
                "canonical": {
                    "path": "canonical.wav",
                    **canonical_metadata,
                    "sha256": decode_report["sha256"],
                    "decoder": decode_report["decoder"],
                    "denoise": decode_report["denoise"],
                    "dereverb": decode_report["dereverb"],
                    "quality": quality,
                },
                "segments": segment_reports,
                "asr_transcripts": asr_artifact,
            },
            "transcript": transcript_record,
            "asr": asr_report,
            "vad": vad_report,
            "stages": stages,
        }
        report_payload = _canonical_json_bytes(report)
        if artifact_bytes + len(report_payload) > settings.max_artifact_bytes:
            raise DatasetPipelineError(
                "dataset report would exceed max_artifact_bytes"
            )
        _write_canonical_json(stage_root / "dataset-report.json", report)
        os.replace(stage_root, destination)
        return report
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


__all__ = [
    "DATASET_PIPELINE_SCHEMA",
    "FASTTEXT_LID176_SHA256",
    "MEDIA_SUFFIXES",
    "SUPPORTED_LANGUAGES",
    "DatasetPipelineConfig",
    "DatasetPipelineError",
    "MediaProbe",
    "assert_linux_docker_runtime",
    "build_transcript_record",
    "decode_media",
    "detect_dataset_language",
    "energy_speech_regions",
    "is_linux_docker_runtime",
    "load_fasttext_detector",
    "load_gpt_sovits_normalizer",
    "normalize_dataset_transcript",
    "parse_ffprobe_document",
    "probe_media",
    "resolve_fasttext_model",
    "run_dataset_pipeline",
    "sha256_file",
]
