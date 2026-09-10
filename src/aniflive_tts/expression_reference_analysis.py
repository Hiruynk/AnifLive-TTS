from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import unicodedata
import wave
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

ANALYSIS_SCHEMA = "aniflive-expression-reference-analysis-v1"
ANALYSIS_SAMPLE_RATE = 16_000
MAX_ANALYSIS_SECONDS = 120.0
MAX_WAVE_SOURCE_BYTES = 64 * 1024 * 1024
WAVEFORM_BINS = 320


class ExpressionReferenceAnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class DecodedReference:
    samples: np.ndarray
    sample_rate: int
    decoder: str
    source_sample_rate: int | None
    source_channels: int | None
    truncated: bool


def _sha256_file(path: Path) -> tuple[str, os.stat_result]:
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as error:
        raise ExpressionReferenceAnalysisError(
            "Expression reference could not be read"
        ) from error
    identities = (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
    )
    if identities[0] != identities[1]:
        raise ExpressionReferenceAnalysisError(
            "Expression reference changed while it was being analyzed"
        )
    return digest.hexdigest(), after


def _pcm_bytes_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8)
        if octets.size % 3:
            raise ExpressionReferenceAnalysisError("PCM24 reference payload is malformed")
        triples = octets.reshape(-1, 3).astype(np.int32)
        values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8_388_608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2_147_483_648.0
    raise ExpressionReferenceAnalysisError("PCM reference sample width is unsupported")


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.ascontiguousarray(samples, dtype=np.float32)
    target_count = round(samples.size * target_rate / source_rate)
    if target_count < 1:
        raise ExpressionReferenceAnalysisError("Expression reference contains no audio samples")
    positions = np.arange(target_count, dtype=np.float64) * source_rate / target_rate
    source_positions = np.arange(samples.size, dtype=np.float64)
    return np.interp(positions, source_positions, samples).astype(np.float32)


def _decode_pcm_wave(path: Path) -> DecodedReference:
    try:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            sample_width = source.getsampwidth()
            frame_count = source.getnframes()
            compression = source.getcomptype()
            if compression != "NONE":
                raise ExpressionReferenceAnalysisError(
                    "Compressed WAVE references require the configured FFmpeg decoder"
                )
            if not 1 <= channels <= 32 or not 8_000 <= sample_rate <= 384_000:
                raise ExpressionReferenceAnalysisError("WAVE reference format is unsupported")
            analysis_frames = min(frame_count, int(MAX_ANALYSIS_SECONDS * sample_rate))
            decoded_bytes = analysis_frames * channels * sample_width
            if decoded_bytes > MAX_WAVE_SOURCE_BYTES:
                raise ExpressionReferenceAnalysisError(
                    "WAVE reference is too large for deterministic local analysis"
                )
            raw = source.readframes(analysis_frames)
    except ExpressionReferenceAnalysisError:
        raise
    except (EOFError, wave.Error, OSError) as error:
        raise ExpressionReferenceAnalysisError("Reference is not a supported PCM WAVE") from error
    values = _pcm_bytes_to_float(raw, sample_width)
    if values.size % channels:
        raise ExpressionReferenceAnalysisError("WAVE reference channel payload is malformed")
    mono = values.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return DecodedReference(
        samples=_resample_linear(mono, sample_rate, ANALYSIS_SAMPLE_RATE),
        sample_rate=ANALYSIS_SAMPLE_RATE,
        decoder="python-wave-pcm-v1",
        source_sample_rate=sample_rate,
        source_channels=channels,
        truncated=analysis_frames < frame_count,
    )


def _decode_ffmpeg(path: Path, *, ffmpeg_binary: str | None = None) -> DecodedReference:
    configured = ffmpeg_binary or os.environ.get("ANIFLIVE_TTS_FFMPEG_BINARY") or "ffmpeg"
    executable = shutil.which(configured)
    if executable is None:
        raise ExpressionReferenceAnalysisError(
            "This reference format requires the configured FFmpeg decoder"
        )
    argv = [
        executable,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-fflags",
        "+bitexact",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-t",
        f"{MAX_ANALYSIS_SECONDS:.0f}",
        "-ac",
        "1",
        "-ar",
        str(ANALYSIS_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=150,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as error:
        raise ExpressionReferenceAnalysisError(
            "FFmpeg reference analysis exceeded its hard timeout"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()[:300]
        raise ExpressionReferenceAnalysisError(
            f"FFmpeg could not decode the reference: {detail or 'unknown decoder error'}"
        )
    if len(completed.stdout) % 2:
        raise ExpressionReferenceAnalysisError("FFmpeg returned malformed PCM")
    samples = np.frombuffer(completed.stdout, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size < ANALYSIS_SAMPLE_RATE // 100:
        raise ExpressionReferenceAnalysisError("Expression reference contains no usable audio")
    cap = int(MAX_ANALYSIS_SECONDS * ANALYSIS_SAMPLE_RATE)
    return DecodedReference(
        samples=samples,
        sample_rate=ANALYSIS_SAMPLE_RATE,
        decoder="ffmpeg-bitexact-pcm16-mono-v1",
        source_sample_rate=None,
        source_channels=None,
        truncated=samples.size >= cap,
    )


def decode_reference(path: Path, *, ffmpeg_binary: str | None = None) -> DecodedReference:
    if path.suffix.lower() == ".wav":
        try:
            return _decode_pcm_wave(path)
        except ExpressionReferenceAnalysisError:
            if shutil.which(
                ffmpeg_binary
                or os.environ.get("ANIFLIVE_TTS_FFMPEG_BINARY")
                or "ffmpeg"
            ) is None:
                raise
    return _decode_ffmpeg(path, ffmpeg_binary=ffmpeg_binary)


def _dbfs(amplitude: float) -> float:
    return 20.0 * math.log10(max(float(amplitude), 1e-8))


def _frame_rms(samples: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if samples.size < frame_size:
        return np.asarray(
            [math.sqrt(float(np.mean(np.square(samples, dtype=np.float64))) + 1e-12)],
            dtype=np.float64,
        )
    starts = np.arange(0, samples.size - frame_size + 1, hop_size, dtype=np.int64)
    squared = np.square(samples, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    energy = cumulative[starts + frame_size] - cumulative[starts]
    return np.sqrt(np.maximum(energy / frame_size, 1e-12))


def _waveform_preview(samples: np.ndarray, bins: int = WAVEFORM_BINS) -> list[list[float]]:
    edges = np.linspace(0, samples.size, min(bins, samples.size) + 1, dtype=np.int64)
    preview: list[list[float]] = []
    for start, end in pairwise(edges):
        region = samples[start:max(start + 1, end)]
        preview.append([round(float(region.min()), 5), round(float(region.max()), 5)])
    return preview


def _pitch_summary(
    samples: np.ndarray,
    sample_rate: int,
    *,
    speech_threshold: float,
) -> dict[str, Any]:
    frame_size = round(0.04 * sample_rate)
    hop_size = round(0.02 * sample_rate)
    if samples.size < frame_size:
        return {
            "reliable": False,
            "reason": "reference-too-short",
            "method": "normalized-autocorrelation-v1",
        }
    starts = np.arange(0, samples.size - frame_size + 1, hop_size, dtype=np.int64)
    if starts.size > 800:
        starts = starts[np.linspace(0, starts.size - 1, 800, dtype=np.int64)]
    window = np.hanning(frame_size).astype(np.float64)
    minimum_lag = max(1, int(sample_rate / 500.0))
    maximum_lag = min(frame_size - 2, int(sample_rate / 60.0))
    estimates: list[float] = []
    confidences: list[float] = []
    eligible = 0
    fft_size = 1 << (2 * frame_size - 1).bit_length()
    for start in starts:
        frame = samples[start : start + frame_size].astype(np.float64)
        rms = math.sqrt(float(np.mean(np.square(frame))) + 1e-12)
        if rms < speech_threshold:
            continue
        eligible += 1
        frame = (frame - float(frame.mean())) * window
        spectrum = np.fft.rfft(frame, n=fft_size)
        autocorrelation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)
        zero = float(autocorrelation[0])
        if zero <= 1e-9:
            continue
        normalized = autocorrelation[minimum_lag : maximum_lag + 1] / zero
        relative = int(np.argmax(normalized))
        confidence = float(normalized[relative])
        if confidence < 0.58:
            continue
        lag = float(minimum_lag + relative)
        if 0 < relative < normalized.size - 1:
            left, center, right = (
                float(normalized[relative - 1]),
                confidence,
                float(normalized[relative + 1]),
            )
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-9:
                lag += 0.5 * (left - right) / denominator
        estimates.append(sample_rate / lag)
        confidences.append(confidence)
    voiced_ratio = len(estimates) / max(1, eligible)
    median_confidence = float(np.median(confidences)) if confidences else 0.0
    reliable = len(estimates) >= 12 and voiced_ratio >= 0.18 and median_confidence >= 0.62
    result: dict[str, Any] = {
        "reliable": reliable,
        "method": "normalized-autocorrelation-v1",
        "analyzed_frames": int(starts.size),
        "eligible_frames": eligible,
        "voiced_frames": len(estimates),
        "voiced_ratio": round(voiced_ratio, 4),
        "median_confidence": round(median_confidence, 4),
    }
    if not reliable:
        result["reason"] = "insufficient-periodic-evidence"
        return result
    values = np.asarray(estimates, dtype=np.float64)
    result.update(
        {
            "median_hz": round(float(np.median(values)), 2),
            "p10_hz": round(float(np.percentile(values, 10)), 2),
            "p90_hz": round(float(np.percentile(values, 90)), 2),
        }
    )
    return result


def _speaking_rate(
    transcript: str | None,
    language: str,
    speech_span_seconds: float | None,
) -> dict[str, Any] | None:
    if not isinstance(transcript, str) or not transcript.strip() or not speech_span_seconds:
        return None
    normalized = " ".join(transcript.strip().split())
    if not normalized:
        return None
    transcript_sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    root_language = language.lower().split("-", 1)[0]
    if root_language in {"zh", "yue", "ja", "ko"}:
        count = sum(
            1 for character in normalized if unicodedata.category(character)[0] in {"L", "N"}
        )
        if count < 1:
            return None
        return {
            "unit": "characters-per-second",
            "count": count,
            "value": round(count / speech_span_seconds, 3),
            "speech_span_seconds": round(speech_span_seconds, 4),
            "transcript_sha256": transcript_sha256,
        }
    words = re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", normalized, flags=re.UNICODE)
    if not words:
        return None
    return {
        "unit": "words-per-minute",
        "count": len(words),
        "value": round(len(words) * 60.0 / speech_span_seconds, 3),
        "speech_span_seconds": round(speech_span_seconds, 4),
        "transcript_sha256": transcript_sha256,
    }


def analyze_pcm(
    samples: np.ndarray,
    sample_rate: int,
    *,
    reference_sha256: str,
    decoder: str,
    language: str,
    transcript: str | None = None,
    source_sample_rate: int | None = None,
    source_channels: int | None = None,
    truncated: bool = False,
) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate <= 0 or values.size < max(1, sample_rate // 100):
        raise ExpressionReferenceAnalysisError("Expression reference contains no usable audio")
    if not np.all(np.isfinite(values)):
        raise ExpressionReferenceAnalysisError("Expression reference contains non-finite samples")
    values = np.clip(values, -1.0, 1.0)
    frame_size = max(1, round(0.01 * sample_rate))
    rms_frames = _frame_rms(values, frame_size, frame_size)
    rms_db_frames = 20.0 * np.log10(np.maximum(rms_frames, 1e-8))
    active = rms_db_frames >= -45.0
    onset_frame: int | None = None
    final_frame: int | None = None
    for index in range(max(0, active.size - 2)):
        if bool(np.all(active[index : index + 3])):
            onset_frame = index
            break
    if onset_frame is not None:
        active_indices = np.flatnonzero(active)
        final_frame = int(active_indices[-1]) if active_indices.size else None
    onset_seconds = onset_frame * 0.01 if onset_frame is not None else None
    speech_span_seconds = (
        (final_frame + 1 - onset_frame) * 0.01
        if onset_frame is not None and final_frame is not None
        else None
    )
    rms = math.sqrt(float(np.mean(np.square(values, dtype=np.float64))) + 1e-12)
    peak = float(np.max(np.abs(values)))
    speech_threshold = 10.0 ** (-45.0 / 20.0)
    result: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "reference_sha256": reference_sha256,
        "provenance": {
            "analyzer": "aniflive-nonneural-reference-signal-v1",
            "decoder": decoder,
            "neural_inference": False,
            "analysis_sample_rate_hz": sample_rate,
            "onset_policy": "three-consecutive-10ms-frames-at-minus-45-dbfs",
        },
        "source": {
            "source_sample_rate_hz": source_sample_rate,
            "source_channels": source_channels,
            "analyzed_samples": int(values.size),
            "analyzed_duration_seconds": round(values.size / sample_rate, 6),
            "truncated": bool(truncated),
        },
        "waveform": {
            "kind": "min-max",
            "bins": _waveform_preview(values),
        },
        "measurements": {
            "duration_seconds": round(values.size / sample_rate, 6),
            "onset_seconds": round(onset_seconds, 4) if onset_seconds is not None else None,
            "speech_span_seconds": (
                round(speech_span_seconds, 4) if speech_span_seconds is not None else None
            ),
            "rms_amplitude": round(rms, 7),
            "rms_dbfs": round(_dbfs(rms), 3),
            "peak_amplitude": round(peak, 7),
            "peak_dbfs": round(_dbfs(peak), 3),
            "pitch": _pitch_summary(
                values, sample_rate, speech_threshold=speech_threshold
            ),
        },
    }
    speaking_rate = _speaking_rate(transcript, language, speech_span_seconds)
    if speaking_rate is not None:
        result["measurements"]["speaking_rate"] = speaking_rate
    return result


def analyze_reference_file(
    path: Path,
    *,
    expected_sha256: str,
    language: str,
    transcript: str | None = None,
    ffmpeg_binary: str | None = None,
) -> dict[str, Any]:
    before_sha256, before = _sha256_file(path)
    if before_sha256 != expected_sha256:
        raise ExpressionReferenceAnalysisError(
            "Expression reference no longer matches its registered SHA256"
        )
    decoded = decode_reference(path, ffmpeg_binary=ffmpeg_binary)
    after_sha256, after = _sha256_file(path)
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if after_sha256 != before_sha256 or after_identity != before_identity:
        raise ExpressionReferenceAnalysisError(
            "Expression reference changed while it was being analyzed"
        )
    return analyze_pcm(
        decoded.samples,
        decoded.sample_rate,
        reference_sha256=expected_sha256,
        decoder=decoded.decoder,
        language=language,
        transcript=transcript,
        source_sample_rate=decoded.source_sample_rate,
        source_channels=decoded.source_channels,
        truncated=decoded.truncated,
    )


__all__ = [
    "ANALYSIS_SCHEMA",
    "ExpressionReferenceAnalysisError",
    "analyze_pcm",
    "analyze_reference_file",
    "decode_reference",
]
