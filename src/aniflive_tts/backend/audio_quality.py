from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import librosa
import matplotlib
import numpy as np
import soundfile as sf
from scipy import signal

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class AudioQualityConfig:
    analysis_sample_rate: int | None = None
    max_alignment_seconds: float = 1.0
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    clipping_threshold: float = 0.999
    si_sdr_ceiling_db: float = 120.0

    def validate(self) -> None:
        if self.analysis_sample_rate is not None and self.analysis_sample_rate <= 0:
            raise ValueError("analysis_sample_rate must be positive")
        if self.max_alignment_seconds < 0:
            raise ValueError("max_alignment_seconds must be non-negative")
        if self.n_fft < 16:
            raise ValueError("n_fft must be at least 16")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")
        if not 0 < self.clipping_threshold <= 1:
            raise ValueError("clipping_threshold must be in (0, 1]")
        if self.si_sdr_ceiling_db <= 0:
            raise ValueError("si_sdr_ceiling_db must be positive")


@dataclass(frozen=True)
class _LoadedAudio:
    path: Path
    samples: np.ndarray
    mono: np.ndarray
    sample_rate: int
    format: str
    subtype: str
    endian: str
    sha256: str
    size_bytes: int
    mtime_ns: int


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "sha256": _sha256_file(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _load_wav(path: Path | str) -> _LoadedAudio:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Audio input is not a regular file: {resolved}")

    fingerprint = _fingerprint(resolved)
    info = sf.info(resolved)
    if info.format not in {"WAV", "WAVEX"}:
        raise ValueError(f"Expected a WAV file, got {info.format}: {resolved}")

    samples, sample_rate = sf.read(resolved, dtype="float64", always_2d=True)
    if samples.shape[0] == 0:
        raise ValueError(f"Audio input is empty: {resolved}")
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate {sample_rate}: {resolved}")
    if not np.isfinite(samples).all():
        raise ValueError(f"Audio contains NaN or Inf values: {resolved}")

    mono = samples.mean(axis=1, dtype=np.float64)
    return _LoadedAudio(
        path=resolved,
        samples=samples,
        mono=mono,
        sample_rate=int(sample_rate),
        format=info.format,
        subtype=info.subtype,
        endian=info.endian,
        sha256=str(fingerprint["sha256"]),
        size_bytes=int(fingerprint["size_bytes"]),
        mtime_ns=int(fingerprint["mtime_ns"]),
    )


def _dbfs(value: float) -> float | None:
    if value <= 0:
        return None
    return float(20.0 * math.log10(value))


def _audio_statistics(audio: _LoadedAudio, clipping_threshold: float) -> dict[str, Any]:
    absolute = np.abs(audio.samples)
    peak = float(np.max(absolute))
    rms = float(np.sqrt(np.mean(np.square(audio.samples, dtype=np.float64))))
    clipped = int(np.count_nonzero(absolute >= clipping_threshold))
    sample_values = int(audio.samples.size)
    return {
        "path": str(audio.path),
        "sha256": audio.sha256,
        "size_bytes": audio.size_bytes,
        "format": audio.format,
        "subtype": audio.subtype,
        "endian": audio.endian,
        "sample_rate_hz": audio.sample_rate,
        "channels": int(audio.samples.shape[1]),
        "frames": int(audio.samples.shape[0]),
        "duration_seconds": float(audio.samples.shape[0] / audio.sample_rate),
        "peak_amplitude": peak,
        "peak_dbfs": _dbfs(peak),
        "rms_amplitude": rms,
        "rms_dbfs": _dbfs(rms),
        "dc_offset": float(np.mean(audio.samples, dtype=np.float64)),
        "clipping_threshold": clipping_threshold,
        "clipped_sample_values": clipped,
        "clipped_ratio": float(clipped / sample_values),
    }


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float64).copy()
    divisor = math.gcd(source_rate, target_rate)
    return signal.resample_poly(
        samples,
        target_rate // divisor,
        source_rate // divisor,
    ).astype(np.float64, copy=False)


def _correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_centered = reference - np.mean(reference)
    candidate_centered = candidate - np.mean(candidate)
    denominator = float(
        np.linalg.norm(reference_centered) * np.linalg.norm(candidate_centered)
    )
    if denominator <= np.finfo(np.float64).eps:
        return 1.0 if np.allclose(reference, candidate, rtol=0, atol=1e-12) else 0.0
    value = float(np.dot(reference_centered, candidate_centered) / denominator)
    return float(np.clip(value, -1.0, 1.0))


def _estimate_lag(
    reference: np.ndarray,
    candidate: np.ndarray,
    max_lag_samples: int,
) -> tuple[int, float]:
    if max_lag_samples <= 0:
        return 0, 0.0

    reference_centered = reference - np.mean(reference)
    candidate_centered = candidate - np.mean(candidate)
    if (
        np.linalg.norm(reference_centered) <= np.finfo(np.float64).eps
        or np.linalg.norm(candidate_centered) <= np.finfo(np.float64).eps
    ):
        return 0, 0.0

    cross_correlation = signal.correlate(
        candidate_centered,
        reference_centered,
        mode="full",
        method="fft",
    )
    lags = signal.correlation_lags(candidate.size, reference.size, mode="full")
    allowed = np.abs(lags) <= max_lag_samples
    allowed_lags = lags[allowed]
    scores = np.abs(cross_correlation[allowed])
    best_score = float(np.max(scores))
    tied = np.flatnonzero(np.isclose(scores, best_score, rtol=1e-12, atol=1e-15))
    best_index = min(tied, key=lambda index: (abs(int(allowed_lags[index])), index))
    integer_lag = int(allowed_lags[best_index])

    fractional_offset = 0.0
    if 0 < best_index < scores.size - 1:
        left, center, right = scores[best_index - 1 : best_index + 2]
        denominator = float(left - 2.0 * center + right)
        if abs(denominator) > np.finfo(np.float64).eps:
            fractional_offset = float(0.5 * (left - right) / denominator)
            fractional_offset = float(np.clip(fractional_offset, -0.5, 0.5))
    return integer_lag, integer_lag + fractional_offset


def _align(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    max_alignment_seconds: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    max_lag = min(
        int(round(max_alignment_seconds * sample_rate)),
        max(0, reference.size - 1),
        max(0, candidate.size - 1),
    )
    integer_lag, fractional_lag = _estimate_lag(reference, candidate, max_lag)
    if fractional_lag >= 0:
        reference_start = 0.0
        candidate_start = fractional_lag
    else:
        reference_start = -fractional_lag
        candidate_start = 0.0

    overlap = int(
        math.floor(
            min(
                reference.size - reference_start,
                candidate.size - candidate_start,
            )
        )
    )
    if overlap < 16:
        raise ValueError(
            "Aligned audio overlap is too short for quality analysis "
            f"({overlap} samples)"
        )

    sample_offsets = np.arange(overlap, dtype=np.float64)
    aligned_reference = np.interp(
        reference_start + sample_offsets,
        np.arange(reference.size, dtype=np.float64),
        reference,
    )
    aligned_candidate = np.interp(
        candidate_start + sample_offsets,
        np.arange(candidate.size, dtype=np.float64),
        candidate,
    )
    return aligned_reference, aligned_candidate, {
        "method": "bounded_fft_cross_correlation",
        "polarity_invariant_lag_search": True,
        "fractional_refinement": "three-point parabolic peak interpolation",
        "lag_definition": "positive means candidate is delayed relative to reference",
        "lag_samples": integer_lag,
        "fractional_lag_samples": fractional_lag,
        "delay_seconds": float(fractional_lag / sample_rate),
        "search_limit_samples": max_lag,
        "search_limit_seconds": float(max_lag / sample_rate),
        "reference_start_sample": float(reference_start),
        "candidate_start_sample": float(candidate_start),
        "overlap_samples": int(overlap),
        "overlap_seconds": float(overlap / sample_rate),
        "reference_coverage_ratio": float(overlap / reference.size),
        "candidate_coverage_ratio": float(overlap / candidate.size),
        "alignment_correlation": _correlation(aligned_reference, aligned_candidate),
    }


def _si_sdr(
    reference: np.ndarray,
    candidate: np.ndarray,
    ceiling_db: float,
) -> float:
    reference = reference - np.mean(reference)
    candidate = candidate - np.mean(candidate)
    epsilon = np.finfo(np.float64).eps
    reference_energy = float(np.dot(reference, reference))
    if reference_energy <= epsilon:
        return ceiling_db if np.allclose(reference, candidate, atol=1e-12) else -ceiling_db

    scale = float(np.dot(candidate, reference) / reference_energy)
    target = scale * reference
    residual = candidate - target
    target_energy = float(np.dot(target, target))
    residual_energy = float(np.dot(residual, residual))
    if target_energy <= epsilon:
        return -ceiling_db
    if residual_energy <= epsilon:
        return ceiling_db
    value = 10.0 * math.log10(target_energy / residual_energy)
    return float(np.clip(value, -ceiling_db, ceiling_db))


def _cosine_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(candidate))
    if denominator <= np.finfo(np.float64).eps:
        return 1.0 if np.allclose(reference, candidate, atol=1e-12) else 0.0
    value = float(np.dot(reference.ravel(), candidate.ravel()) / denominator)
    return float(np.clip(value, -1.0, 1.0))


def _speaker_embeddings(
    reference: _LoadedAudio,
    candidate: _LoadedAudio,
    model_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    model_path = model_path.expanduser().resolve(strict=True)
    module_dir = Path(__file__).resolve().parents[3] / "minimal_inference" / "GPT_SoVITS" / "eres2net"
    if not module_dir.is_dir():
        raise FileNotFoundError(f"Speaker verifier module directory is missing: {module_dir}")
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)
    eres2net = importlib.import_module("ERes2NetV2")
    kaldi = importlib.import_module("kaldi")
    model = eres2net.ERes2NetV2(baseWidth=24, scale=4, expansion=4)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    def embed(audio: _LoadedAudio) -> np.ndarray:
        waveform = _resample(audio.mono, audio.sample_rate, 16000)
        tensor = torch.from_numpy(waveform.astype(np.float32, copy=False)).to(device)
        with torch.inference_mode():
            features = kaldi.fbank(
                tensor.unsqueeze(0),
                num_mel_bins=80,
                sample_frequency=16000,
                dither=0,
            ).unsqueeze(0)
            embedding = model.forward3(features)
        result = embedding.detach().float().cpu().numpy().reshape(-1).astype(np.float64)
        if not np.isfinite(result).all() or not np.any(result):
            raise RuntimeError("Speaker verifier produced an invalid embedding")
        return result

    return embed(reference), embed(candidate), {
        "model_path": str(model_path),
        "model_sha256": _sha256_file(model_path),
        "architecture": "ERes2NetV2(baseWidth=24, scale=4, expansion=4)",
        "sample_rate_hz": 16000,
        "feature": "80-bin Kaldi fbank, dither=0",
        "device": str(device),
        "embedding_dimensions": 20480,
    }


def _spectral_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    config: AudioQualityConfig,
) -> tuple[dict[str, float], dict[str, Any], np.ndarray, np.ndarray]:
    largest_fft = 1 << int(math.floor(math.log2(reference.size)))
    n_fft = min(config.n_fft, largest_fft)
    n_fft = max(16, n_fft)
    hop_length = min(config.hop_length, max(1, n_fft // 4))
    n_mels = min(config.n_mels, n_fft // 2 + 1)

    reference_stft = np.abs(
        librosa.stft(
            reference,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window="hann",
            center=False,
        )
    )
    candidate_stft = np.abs(
        librosa.stft(
            candidate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window="hann",
            center=False,
        )
    )

    reference_mel = librosa.feature.melspectrogram(
        S=np.square(reference_stft),
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0,
        fmax=sample_rate / 2,
        power=2.0,
    )
    candidate_mel = librosa.feature.melspectrogram(
        S=np.square(candidate_stft),
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0,
        fmax=sample_rate / 2,
        power=2.0,
    )
    floor = 1e-10
    reference_log_mel = 10.0 * np.log10(np.maximum(reference_mel, floor))
    candidate_log_mel = 10.0 * np.log10(np.maximum(candidate_mel, floor))

    reference_norm = float(np.linalg.norm(reference_stft))
    difference_norm = float(np.linalg.norm(reference_stft - candidate_stft))
    if reference_norm <= np.finfo(np.float64).eps:
        spectral_convergence = 0.0 if difference_norm <= 1e-12 else config.si_sdr_ceiling_db
    else:
        spectral_convergence = difference_norm / reference_norm

    metrics = {
        "log_mel_mae_db": float(np.mean(np.abs(reference_log_mel - candidate_log_mel))),
        "log_mel_cosine_similarity": _cosine_similarity(
            reference_log_mel,
            candidate_log_mel,
        ),
        "spectral_convergence": float(spectral_convergence),
    }
    settings = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "window": "hann",
        "center": False,
        "mel_power": 2.0,
        "log_floor_power": floor,
    }
    return metrics, settings, reference_log_mel, candidate_log_mel


def _plot_waveforms(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    alignment: dict[str, Any],
    output_path: Path,
) -> None:
    max_points = 30_000
    step = max(1, int(math.ceil(reference.size / max_points)))
    indices = np.arange(0, reference.size, step)
    times = indices / sample_rate
    difference = candidate[indices] - reference[indices]

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(times, reference[indices], linewidth=0.6, label="Reference", alpha=0.85)
    axes[0].plot(times, candidate[indices], linewidth=0.6, label="Candidate", alpha=0.65)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(
        "Aligned waveform comparison "
        f"(candidate delay: {alignment['delay_seconds'] * 1000:.3f} ms)"
    )
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    axes[1].plot(times, difference, linewidth=0.6, color="#b42318")
    axes[1].axhline(0, linewidth=0.6, color="black", alpha=0.5)
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Candidate - reference")
    axes[1].set_title("Waveform residual")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_mels(
    reference_log_mel: np.ndarray,
    candidate_log_mel: np.ndarray,
    duration_seconds: float,
    output_path: Path,
) -> None:
    difference = np.abs(candidate_log_mel - reference_log_mel)
    difference_max = max(float(np.max(difference)), 1e-6)
    common_max = float(max(np.max(reference_log_mel), np.max(candidate_log_mel)))
    common_min = max(
        common_max - 80.0,
        float(min(np.min(reference_log_mel), np.min(candidate_log_mel))),
    )
    extent = [0, duration_seconds, 0, reference_log_mel.shape[0]]

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    reference_image = axes[0].imshow(
        reference_log_mel,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        vmin=common_min,
        vmax=common_max,
    )
    axes[0].set_title("Reference log-mel spectrogram")
    axes[0].set_ylabel("Mel bin")
    figure.colorbar(reference_image, ax=axes[0], label="dB")

    candidate_image = axes[1].imshow(
        candidate_log_mel,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        vmin=common_min,
        vmax=common_max,
    )
    axes[1].set_title("Candidate log-mel spectrogram")
    axes[1].set_ylabel("Mel bin")
    figure.colorbar(candidate_image, ax=axes[1], label="dB")

    difference_image = axes[2].imshow(
        difference,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        vmin=0,
        vmax=difference_max,
    )
    axes[2].set_title("Absolute log-mel difference")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Mel bin")
    figure.colorbar(difference_image, ax=axes[2], label="Absolute dB error")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def compare_audio(
    reference_path: Path | str,
    candidate_path: Path | str,
    output_dir: Path | str,
    *,
    config: AudioQualityConfig | None = None,
    report_path: Path | str | None = None,
    artifact_prefix: str = "",
    speaker_model_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compare two WAV files and write objective metrics, plots, and a JSON report."""
    config = config or AudioQualityConfig()
    config.validate()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if artifact_prefix and any(
        character not in "-_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        for character in artifact_prefix
    ):
        raise ValueError("artifact_prefix may contain only letters, digits, '-' and '_'")
    name_prefix = f"{artifact_prefix}_" if artifact_prefix else ""
    waveform_path = output_dir / f"{name_prefix}waveform_comparison.png"
    mel_path = output_dir / f"{name_prefix}mel_spectrogram_comparison.png"
    if report_path is None:
        resolved_report_path = output_dir / f"{name_prefix}audio_quality.json"
    else:
        resolved_report_path = Path(report_path).expanduser().resolve()
        resolved_report_path.parent.mkdir(parents=True, exist_ok=True)

    reference = _load_wav(reference_path)
    candidate = _load_wav(candidate_path)
    protected_inputs = {reference.path, candidate.path}
    output_paths = {waveform_path, mel_path, resolved_report_path}
    collisions = protected_inputs.intersection(output_paths)
    if collisions:
        raise ValueError(
            "Output path must not overwrite an input WAV: "
            + ", ".join(str(path) for path in sorted(collisions))
        )

    analysis_sample_rate = config.analysis_sample_rate or min(
        reference.sample_rate,
        candidate.sample_rate,
    )
    reference_resampled = _resample(
        reference.mono,
        reference.sample_rate,
        analysis_sample_rate,
    )
    candidate_resampled = _resample(
        candidate.mono,
        candidate.sample_rate,
        analysis_sample_rate,
    )
    aligned_reference, aligned_candidate, alignment = _align(
        reference_resampled,
        candidate_resampled,
        analysis_sample_rate,
        config.max_alignment_seconds,
    )

    spectral_metrics, spectral_settings, reference_mel, candidate_mel = (
        _spectral_metrics(
            aligned_reference,
            aligned_candidate,
            analysis_sample_rate,
            config,
        )
    )
    speaker_verification: dict[str, Any] | None = None
    if speaker_model_path is None:
        speaker_similarity: dict[str, Any] = {
            "status": "unavailable",
            "value": None,
            "reason": "No speaker-verification model was requested.",
        }
    else:
        reference_embedding, candidate_embedding, speaker_verification = _speaker_embeddings(
            reference,
            candidate,
            Path(speaker_model_path),
        )
        speaker_similarity = {
            "status": "measured",
            "value": _cosine_similarity(reference_embedding, candidate_embedding),
            "metric": "cosine_similarity",
            "higher_is_better": True,
        }

    metrics = {
        "waveform_correlation": _correlation(aligned_reference, aligned_candidate),
        "si_sdr_db": _si_sdr(
            aligned_reference,
            aligned_candidate,
            config.si_sdr_ceiling_db,
        ),
        **spectral_metrics,
        "speaker_similarity": speaker_similarity,
    }

    _plot_waveforms(
        aligned_reference,
        aligned_candidate,
        analysis_sample_rate,
        alignment,
        waveform_path,
    )
    _plot_mels(
        reference_mel,
        candidate_mel,
        alignment["overlap_seconds"],
        mel_path,
    )

    reference_after = _fingerprint(reference.path)
    candidate_after = _fingerprint(candidate.path)
    reference_unchanged = reference_after == {
        "sha256": reference.sha256,
        "size_bytes": reference.size_bytes,
        "mtime_ns": reference.mtime_ns,
    }
    candidate_unchanged = candidate_after == {
        "sha256": candidate.sha256,
        "size_bytes": candidate.size_bytes,
        "mtime_ns": candidate.mtime_ns,
    }
    if not reference_unchanged or not candidate_unchanged:
        raise RuntimeError("An input WAV changed while it was being analyzed")

    report: dict[str, Any] = {
        "schema_version": 1,
        "reference": _audio_statistics(reference, config.clipping_threshold),
        "candidate": _audio_statistics(candidate, config.clipping_threshold),
        "analysis": {
            "config": asdict(config),
            "sample_rate_hz": analysis_sample_rate,
            "channel_policy": "arithmetic mean downmix to mono",
            "resampling_method": "scipy.signal.resample_poly",
            "metric_scope": "delay-aligned overlapping samples only",
            "alignment": alignment,
            "spectral_settings": spectral_settings,
            "speaker_verification": speaker_verification,
        },
        "metrics": metrics,
        "metric_guidance": {
            "waveform_correlation": "higher is better; ideal 1",
            "si_sdr_db": (
                f"higher is better; capped to +/-{config.si_sdr_ceiling_db:g} dB"
            ),
            "log_mel_mae_db": "lower is better; ideal 0",
            "log_mel_cosine_similarity": "higher is better; ideal 1",
            "spectral_convergence": "lower is better; ideal 0",
            "speaker_similarity": "higher is better; ideal 1",
        },
        "input_integrity": {
            "inputs_opened_read_only": True,
            "reference_unchanged_during_analysis": reference_unchanged,
            "candidate_unchanged_during_analysis": candidate_unchanged,
        },
        "artifacts": {
            "waveform_png": str(waveform_path),
            "mel_spectrogram_png": str(mel_path),
            "json_report": str(resolved_report_path),
        },
    }
    resolved_report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return report
