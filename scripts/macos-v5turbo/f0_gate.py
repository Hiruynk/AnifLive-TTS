from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


F0_CONFIG = {
    "target_sample_rate_hz": 16000,
    "fmin_hz": 65.0,
    "fmax_hz": 1000.0,
    "frame_length": 1024,
    "hop_length": 160,
    "center": True,
}
THRESHOLDS = {
    "duration_relative_error_max": 0.03,
    "voiced_agreement_min": 0.99,
    "median_relative_error_max": 0.01,
    "p95_absolute_error_hz_max": 5.0,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_mono(path: Path, label: str) -> tuple[Path, np.ndarray, int, int]:
    resolved = path.expanduser().resolve(strict=True)
    samples, sample_rate = sf.read(str(resolved), dtype="float32", always_2d=False)
    values = np.asarray(samples, dtype=np.float32)
    if sample_rate <= 0:
        raise ValueError(f"{label} has an invalid sample rate")
    if values.ndim != 1:
        raise ValueError(f"{label} must be mono, got {values.ndim} dimensions")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{label} must be nonempty and finite")
    return resolved, values, int(sample_rate), int(values.size)


def _estimate_f0(samples: np.ndarray, sample_rate: int):
    resampled = librosa.resample(
        samples,
        orig_sr=sample_rate,
        target_sr=F0_CONFIG["target_sample_rate_hz"],
        res_type="soxr_hq",
    )
    f0, voiced, voiced_probability = librosa.pyin(
        resampled,
        sr=F0_CONFIG["target_sample_rate_hz"],
        fmin=F0_CONFIG["fmin_hz"],
        fmax=F0_CONFIG["fmax_hz"],
        frame_length=F0_CONFIG["frame_length"],
        hop_length=F0_CONFIG["hop_length"],
        center=F0_CONFIG["center"],
    )
    return resampled, f0, voiced, voiced_probability


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def compare(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference_path, reference_samples, reference_rate, reference_count = _read_mono(
        reference_path, "reference"
    )
    candidate_path, candidate_samples, candidate_rate, candidate_count = _read_mono(
        candidate_path, "candidate"
    )
    reference_duration = reference_count / reference_rate
    candidate_duration = candidate_count / candidate_rate
    duration_relative_error = abs(candidate_duration - reference_duration) / reference_duration

    ref_resampled, ref_f0, ref_voiced, ref_probability = _estimate_f0(
        reference_samples, reference_rate
    )
    cand_resampled, cand_f0, cand_voiced, cand_probability = _estimate_f0(
        candidate_samples, candidate_rate
    )
    sample_counts_match = ref_resampled.size == cand_resampled.size
    frame_counts_match = ref_f0.size == cand_f0.size
    jointly_voiced_count = 0
    voiced_agreement: float | None = None
    median_absolute_error_hz: float | None = None
    p95_absolute_error_hz: float | None = None
    median_relative_error: float | None = None
    p95_relative_error: float | None = None
    voiced_estimates_finite = False
    if frame_counts_match:
        ref_voiced = np.asarray(ref_voiced, dtype=bool)
        cand_voiced = np.asarray(cand_voiced, dtype=bool)
        voiced_agreement = float(np.mean(ref_voiced == cand_voiced))
        jointly_voiced = ref_voiced & cand_voiced
        jointly_voiced_count = int(np.count_nonzero(jointly_voiced))
        voiced_estimates_finite = bool(
            np.isfinite(ref_f0[ref_voiced]).all()
            and np.isfinite(cand_f0[cand_voiced]).all()
        )
        if jointly_voiced_count and voiced_estimates_finite:
            absolute_errors = np.abs(ref_f0[jointly_voiced] - cand_f0[jointly_voiced])
            relative_errors = absolute_errors / ref_f0[jointly_voiced]
            median_absolute_error_hz = float(np.median(absolute_errors))
            p95_absolute_error_hz = _percentile(absolute_errors, 95.0)
            median_relative_error = float(np.median(relative_errors))
            p95_relative_error = _percentile(relative_errors, 95.0)

    checks = {
        "sample_counts_match_after_resampling": sample_counts_match,
        "frame_counts_match": frame_counts_match,
        "voiced_estimates_finite": voiced_estimates_finite,
        "jointly_voiced_frames_present": jointly_voiced_count > 0,
        "duration_relative_error": {
            "observed": duration_relative_error,
            "threshold_max": THRESHOLDS["duration_relative_error_max"],
            "passed": duration_relative_error <= THRESHOLDS["duration_relative_error_max"],
        },
        "voiced_agreement": {
            "observed": voiced_agreement,
            "threshold_min": THRESHOLDS["voiced_agreement_min"],
            "passed": (
                voiced_agreement is not None
                and voiced_agreement >= THRESHOLDS["voiced_agreement_min"]
            ),
        },
        "median_relative_error": {
            "observed": median_relative_error,
            "threshold_max": THRESHOLDS["median_relative_error_max"],
            "passed": (
                median_relative_error is not None
                and median_relative_error <= THRESHOLDS["median_relative_error_max"]
            ),
        },
        "p95_absolute_error_hz": {
            "observed": p95_absolute_error_hz,
            "threshold_max": THRESHOLDS["p95_absolute_error_hz_max"],
            "passed": (
                p95_absolute_error_hz is not None
                and p95_absolute_error_hz <= THRESHOLDS["p95_absolute_error_hz_max"]
            ),
        },
    }
    valid_comparison = (
        sample_counts_match
        and frame_counts_match
        and voiced_estimates_finite
        and jointly_voiced_count > 0
    )
    passed = bool(valid_comparison and all(
        check["passed"]
        for key, check in checks.items()
        if isinstance(check, dict)
    ))
    if not valid_comparison:
        status = "unverified"
        if jointly_voiced_count == 0:
            status_reason = "no jointly voiced frames"
        elif not frame_counts_match:
            status_reason = "full-length pyin frame counts differ; no crop was applied"
        elif not sample_counts_match:
            status_reason = "full-length resampled sample counts differ; no crop was applied"
        else:
            status_reason = "non-finite F0 estimate in a voiced frame"
    else:
        status = "passed" if passed else "failed"
        status_reason = None

    return {
        "schema": "aniflive-tts-macos-v5turbo-f0-gate-v1",
        "kind": "single_utterance_f0_comparison",
        "qualification_scope": "one controlled utterance; not full multilingual qualification",
        "status": status,
        "status_reason": status_reason,
        "passed": passed,
        "reference": {
            "path": str(reference_path),
            "sha256": _sha256_file(reference_path),
            "sample_rate_hz": reference_rate,
            "sample_count": reference_count,
            "duration_seconds": reference_duration,
            "resampled_sample_count": int(ref_resampled.size),
            "pyin_frame_count": int(ref_f0.size),
            "voiced_frame_count": int(np.count_nonzero(ref_voiced)),
            "unvoiced_nan_f0_count": int(np.count_nonzero(~np.isfinite(ref_f0) & ~ref_voiced)),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": _sha256_file(candidate_path),
            "sample_rate_hz": candidate_rate,
            "sample_count": candidate_count,
            "duration_seconds": candidate_duration,
            "resampled_sample_count": int(cand_resampled.size),
            "pyin_frame_count": int(cand_f0.size),
            "voiced_frame_count": int(np.count_nonzero(cand_voiced)),
            "unvoiced_nan_f0_count": int(np.count_nonzero(~np.isfinite(cand_f0) & ~cand_voiced)),
        },
        "analysis": {
            **F0_CONFIG,
            "librosa_version": librosa.__version__,
            "resampling": "librosa.resample res_type=soxr_hq on each full waveform",
            "pyin_nan_policy": "NaNs are allowed in unvoiced frames; voiced F0 estimates must be finite",
            "crop_applied": False,
            "jointly_voiced_frame_count": jointly_voiced_count,
            "median_absolute_error_hz": median_absolute_error_hz,
            "p95_absolute_error_hz": p95_absolute_error_hz,
            "median_relative_error": median_relative_error,
            "p95_relative_error": p95_relative_error,
            "mean_reference_voiced_probability": float(np.mean(ref_probability)),
            "mean_candidate_voiced_probability": float(np.mean(cand_probability)),
        },
        "thresholds": THRESHOLDS,
        "checks": checks,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full-waveform F0 tracks using a bounded single-utterance gate."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = compare(args.reference, args.candidate)
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_name(report_path.name + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(report_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(json.dumps({"status": report["status"], "passed": report["passed"], "checks": report["checks"]}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
