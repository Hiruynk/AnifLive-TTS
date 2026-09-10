from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence
import wave


REPORT_SCHEMA = "aniflive-tts-long-form-continuity-diagnostic-v1"
PCM16_FULL_SCALE = 32768.0
DEFAULT_MAX_FILE_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_SEGMENTS = 256
SUPPORTED_FORMATS = frozenset({"auto", "wav", "pcm16le"})


class ContinuityDiagnosticError(ValueError):
    pass


@dataclass(frozen=True)
class AudioSegment:
    path: Path
    input_format: str
    sample_rate: int
    samples: tuple[int, ...]

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate


def _validate_sample_rate(value: int | None, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContinuityDiagnosticError(f"{field} must be an integer sample rate")
    if value < 1 or value > 768_000:
        raise ContinuityDiagnosticError(f"{field} must be between 1 and 768000 Hz")
    return value


def _pcm16_samples(payload: bytes, *, source: Path) -> tuple[int, ...]:
    if not payload:
        raise ContinuityDiagnosticError(f"Audio segment is empty: {source}")
    if len(payload) % 2:
        raise ContinuityDiagnosticError(f"PCM16 payload has an incomplete sample: {source}")
    values = array("h")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return tuple(int(value) for value in values)


def _resolve_format(path: Path, input_format: str) -> str:
    if input_format not in SUPPORTED_FORMATS:
        raise ContinuityDiagnosticError(f"Unsupported input format: {input_format}")
    if input_format != "auto":
        return input_format
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "wav"
    if suffix in {".pcm", ".raw"}:
        return "pcm16le"
    raise ContinuityDiagnosticError(
        f"Cannot infer audio format for {path}; use .wav, .pcm, .raw, or an explicit format"
    )


def load_audio_segment(
    path: str | Path,
    *,
    input_format: str = "auto",
    raw_sample_rate: int | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> AudioSegment:
    if not isinstance(path, (str, os.PathLike)):
        raise ContinuityDiagnosticError("Audio segment path must be a string or filesystem path")
    source = Path(path).expanduser()
    if not source.is_file():
        raise ContinuityDiagnosticError(f"Audio segment does not exist or is not a file: {source}")
    if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int) or max_file_bytes < 1:
        raise ContinuityDiagnosticError("max_file_bytes must be a positive integer")
    try:
        size = source.stat().st_size
    except OSError as error:
        raise ContinuityDiagnosticError(f"Audio segment could not be inspected: {source}") from error
    if size > max_file_bytes:
        raise ContinuityDiagnosticError(
            f"Audio segment exceeds the {max_file_bytes}-byte diagnostic limit: {source}"
        )
    resolved_format = _resolve_format(source, input_format)
    if resolved_format == "pcm16le":
        sample_rate = _validate_sample_rate(raw_sample_rate, field="raw_sample_rate")
        try:
            payload = source.read_bytes()
        except OSError as error:
            raise ContinuityDiagnosticError(f"PCM segment could not be read: {source}") from error
        return AudioSegment(source.resolve(), resolved_format, sample_rate, _pcm16_samples(payload, source=source))

    try:
        with wave.open(str(source), "rb") as stream:
            if stream.getcomptype() != "NONE":
                raise ContinuityDiagnosticError(
                    f"WAV must contain uncompressed PCM audio: {source}"
                )
            if stream.getnchannels() != 1:
                raise ContinuityDiagnosticError(f"WAV must be mono: {source}")
            if stream.getsampwidth() != 2:
                raise ContinuityDiagnosticError(f"WAV must contain 16-bit PCM samples: {source}")
            sample_rate = _validate_sample_rate(stream.getframerate(), field="WAV sample rate")
            frame_count = stream.getnframes()
            if frame_count < 1:
                raise ContinuityDiagnosticError(f"Audio segment is empty: {source}")
            expected_bytes = frame_count * 2
            if expected_bytes > max_file_bytes:
                raise ContinuityDiagnosticError(
                    f"Decoded WAV exceeds the {max_file_bytes}-byte diagnostic limit: {source}"
                )
            payload = stream.readframes(frame_count)
            if len(payload) != expected_bytes:
                raise ContinuityDiagnosticError(f"WAV payload is truncated: {source}")
    except ContinuityDiagnosticError:
        raise
    except (EOFError, OSError, wave.Error) as error:
        raise ContinuityDiagnosticError(f"WAV is malformed or unreadable: {source}") from error
    return AudioSegment(source.resolve(), resolved_format, sample_rate, _pcm16_samples(payload, source=source))


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values)


def _energy(values: Sequence[int]) -> float:
    scale_squared = PCM16_FULL_SCALE * PCM16_FULL_SCALE
    return sum(float(value) * float(value) for value in values) / len(values) / scale_squared


def _rms(values: Sequence[int]) -> float:
    return math.sqrt(_energy(values))


def _dbfs(amplitude: float) -> float:
    return 20.0 * math.log10(max(amplitude, 1e-12))


def _silence_run(samples: Sequence[int], threshold: float, *, reverse: bool) -> int:
    values = reversed(samples) if reverse else iter(samples)
    count = 0
    for sample in values:
        if abs(sample) > threshold:
            break
        count += 1
    return count


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summary(values: Sequence[float], *, unit: str) -> dict[str, Any]:
    return {
        "unit": unit,
        "count": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def _boundary_metrics(
    left: AudioSegment,
    right: AudioSegment,
    *,
    index: int,
    window_samples: int,
    silence_threshold: float,
) -> dict[str, Any]:
    left_window = left.samples[-min(window_samples, len(left.samples)) :]
    right_window = right.samples[: min(window_samples, len(right.samples))]
    left_rms, right_rms = _rms(left_window), _rms(right_window)
    left_energy, right_energy = _energy(left_window), _energy(right_window)
    left_dc = _mean(left_window) / PCM16_FULL_SCALE
    right_dc = _mean(right_window) / PCM16_FULL_SCALE
    trailing = _silence_run(left.samples, silence_threshold, reverse=True)
    leading = _silence_run(right.samples, silence_threshold, reverse=False)
    discontinuity_raw = abs(right.samples[0] - left.samples[-1])
    return {
        "index": index,
        "left_segment_index": index,
        "right_segment_index": index + 1,
        "left_path": str(left.path),
        "right_path": str(right.path),
        "left_duration_seconds": left.duration_seconds,
        "right_duration_seconds": right.duration_seconds,
        "window_samples": {"left": len(left_window), "right": len(right_window)},
        "sample_discontinuity_raw": discontinuity_raw,
        "sample_discontinuity_normalized": discontinuity_raw / PCM16_FULL_SCALE,
        "left_rms": left_rms,
        "right_rms": right_rms,
        "rms_jump": abs(right_rms - left_rms),
        "rms_jump_db": abs(_dbfs(right_rms) - _dbfs(left_rms)),
        "left_energy": left_energy,
        "right_energy": right_energy,
        "energy_jump": abs(right_energy - left_energy),
        "left_dc": left_dc,
        "right_dc": right_dc,
        "dc_jump": abs(right_dc - left_dc),
        "trailing_pause_seconds": trailing / left.sample_rate,
        "leading_pause_seconds": leading / left.sample_rate,
        "pause_duration_seconds": (trailing + leading) / left.sample_rate,
    }


def evaluate_ordered_segments(
    paths: Sequence[str | Path],
    *,
    input_format: str = "auto",
    raw_sample_rate: int | None = None,
    expected_sample_rate: int | None = None,
    window_ms: float = 10.0,
    silence_threshold_dbfs: float = -50.0,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
) -> dict[str, Any]:
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise ContinuityDiagnosticError("paths must be an ordered sequence of audio segments")
    if isinstance(max_segments, bool) or not isinstance(max_segments, int) or max_segments < 2:
        raise ContinuityDiagnosticError("max_segments must be an integer of at least 2")
    if len(paths) < 2:
        raise ContinuityDiagnosticError("At least two ordered segments are required")
    if len(paths) > max_segments:
        raise ContinuityDiagnosticError(f"Segment count exceeds the configured limit of {max_segments}")
    if (
        isinstance(max_total_bytes, bool)
        or not isinstance(max_total_bytes, int)
        or max_total_bytes < 1
    ):
        raise ContinuityDiagnosticError("max_total_bytes must be a positive integer")
    if not isinstance(window_ms, (int, float)) or isinstance(window_ms, bool):
        raise ContinuityDiagnosticError("window_ms must be a number")
    if not math.isfinite(float(window_ms)) or not 0.1 <= float(window_ms) <= 1000.0:
        raise ContinuityDiagnosticError("window_ms must be between 0.1 and 1000")
    if not isinstance(silence_threshold_dbfs, (int, float)) or isinstance(
        silence_threshold_dbfs, bool
    ):
        raise ContinuityDiagnosticError("silence_threshold_dbfs must be a number")
    threshold_dbfs = float(silence_threshold_dbfs)
    if not math.isfinite(threshold_dbfs) or not -120.0 <= threshold_dbfs <= 0.0:
        raise ContinuityDiagnosticError("silence_threshold_dbfs must be between -120 and 0")
    required_sample_rate = (
        _validate_sample_rate(expected_sample_rate, field="expected_sample_rate")
        if expected_sample_rate is not None
        else None
    )
    segments: list[AudioSegment] = []
    decoded_bytes = 0
    for path in paths:
        segment = load_audio_segment(
            path,
            input_format=input_format,
            raw_sample_rate=raw_sample_rate,
            max_file_bytes=max_file_bytes,
        )
        decoded_bytes += len(segment.samples) * 2
        if decoded_bytes > max_total_bytes:
            raise ContinuityDiagnosticError(
                f"Decoded audio exceeds the configured total limit of {max_total_bytes} bytes"
            )
        segments.append(segment)
    sample_rate = segments[0].sample_rate
    if required_sample_rate is not None and sample_rate != required_sample_rate:
        raise ContinuityDiagnosticError(
            f"Sample rate {sample_rate} Hz does not match expected {required_sample_rate} Hz"
        )
    for segment in segments[1:]:
        if segment.sample_rate != sample_rate:
            raise ContinuityDiagnosticError(
                "All ordered segments must use the same sample rate "
                f"({sample_rate} Hz != {segment.sample_rate} Hz at {segment.path})"
            )
        if required_sample_rate is not None and segment.sample_rate != required_sample_rate:
            raise ContinuityDiagnosticError(
                f"Sample rate {segment.sample_rate} Hz does not match expected "
                f"{required_sample_rate} Hz at {segment.path}"
            )
    window_samples = max(1, round(sample_rate * float(window_ms) / 1000.0))
    silence_threshold = PCM16_FULL_SCALE * (10.0 ** (threshold_dbfs / 20.0))
    boundaries = [
        _boundary_metrics(
            segments[index],
            segments[index + 1],
            index=index,
            window_samples=window_samples,
            silence_threshold=silence_threshold,
        )
        for index in range(len(segments) - 1)
    ]
    durations = [segment.duration_seconds for segment in segments]
    metric_units = {
        "sample_discontinuity_normalized": "normalized_pcm16_amplitude",
        "rms_jump": "normalized_pcm16_rms",
        "rms_jump_db": "dB",
        "energy_jump": "normalized_pcm16_mean_square",
        "dc_jump": "normalized_pcm16_amplitude",
        "pause_duration_seconds": "seconds",
    }
    aggregate_boundaries = {
        metric: _summary([float(boundary[metric]) for boundary in boundaries], unit=unit)
        for metric, unit in metric_units.items()
    }
    return {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "qualification": False,
        "notice": (
            "Boundary waveform diagnostics do not establish speech quality, pronunciation, "
            "speaker identity, semantic correctness, or production qualification."
        ),
        "configuration": {
            "sample_rate_hz": sample_rate,
            "channels": 1,
            "pcm_format": "signed-pcm16-little-endian",
            "window_ms": float(window_ms),
            "window_samples": window_samples,
            "silence_threshold_dbfs": threshold_dbfs,
            "max_file_bytes": max_file_bytes,
            "max_total_bytes": max_total_bytes,
            "max_segments": max_segments,
        },
        "segments": [
            {
                "index": index,
                "path": str(segment.path),
                "input_format": segment.input_format,
                "sample_count": len(segment.samples),
                "duration_seconds": segment.duration_seconds,
            }
            for index, segment in enumerate(segments)
        ],
        "boundaries": boundaries,
        "aggregate": {
            "segment_count": len(segments),
            "boundary_count": len(boundaries),
            "total_duration_seconds": sum(durations),
            "segment_duration_seconds": _summary(durations, unit="seconds"),
            "boundary_metrics": aggregate_boundaries,
        },
    }


__all__ = [
    "AudioSegment",
    "ContinuityDiagnosticError",
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_TOTAL_BYTES",
    "DEFAULT_MAX_SEGMENTS",
    "PCM16_FULL_SCALE",
    "REPORT_SCHEMA",
    "evaluate_ordered_segments",
    "load_audio_segment",
]
