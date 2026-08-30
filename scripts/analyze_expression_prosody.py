from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


FEATURE_NAMES = (
    "f0_median_hz",
    "f0_iqr_hz",
    "rms_mean",
    "rms_std",
    "spectral_centroid_hz",
)


def _load(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    trimmed, _ = librosa.effects.trim(mono, top_db=40)
    if trimmed.size < sample_rate // 4:
        trimmed = mono
    return trimmed, int(sample_rate)


def _features(path: Path) -> dict[str, float]:
    audio, sample_rate = _load(path)
    f0, voiced, _ = librosa.pyin(
        audio,
        fmin=65.0,
        fmax=800.0,
        sr=sample_rate,
        frame_length=1024,
        hop_length=256,
    )
    valid_f0 = f0[np.isfinite(f0)]
    if valid_f0.size < 3:
        raise RuntimeError(f"Insufficient voiced frames: {path}")
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sample_rate, n_fft=1024, hop_length=256
    )[0]
    return {
        "f0_median_hz": float(np.median(valid_f0)),
        "f0_iqr_hz": float(np.percentile(valid_f0, 75) - np.percentile(valid_f0, 25)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "spectral_centroid_hz": float(np.mean(centroid)),
        "voiced_fraction": float(np.mean(voiced)),
        "active_duration_seconds": float(audio.size / sample_rate),
    }


def _direction(value: dict[str, float], neutral: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            math.log(max(value[name], 1e-8) / max(neutral[name], 1e-8))
            for name in FEATURE_NAMES
        ],
        dtype=np.float64,
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-manifest", type=Path, required=True)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    profile_manifest = args.profile_manifest.resolve()
    profile_dir = profile_manifest.parent
    profile = json.loads(profile_manifest.read_text(encoding="utf-8"))
    expression = profile["expression"]
    descriptors = {
        item["emotion"]: item
        for item in expression["profiles"]
        if item["emotion"] != "neutral"
    }
    matrix_dir = args.matrix_dir.resolve()
    policy_report = json.loads((matrix_dir / "policy-ablation.json").read_text(encoding="utf-8"))
    neutral_generated = _features(matrix_dir / "neutral.wav")
    neutral_reference = _features(profile_dir / profile["reference_audio"])
    reference_features: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    for row in policy_report["rows"]:
        profile_id = row["profile"]
        descriptor = descriptors[profile_id]
        reference = reference_features.setdefault(
            profile_id,
            _features(profile_dir / descriptor["reference_audio"]),
        )
        candidate = _features(matrix_dir / f"{profile_id}-{row['policy']}.wav")
        target_direction = _direction(reference, neutral_reference)
        candidate_direction = _direction(candidate, neutral_generated)
        rows.append(
            {
                "profile": profile_id,
                "policy": row["policy"],
                "prosody_direction_cosine": _cosine(
                    candidate_direction, target_direction
                ),
                "candidate_direction_magnitude": float(np.linalg.norm(candidate_direction)),
                "target_direction_magnitude": float(np.linalg.norm(target_direction)),
                "candidate_features": candidate,
                "reference_features": reference,
            }
        )
    report = {
        "schema": 1,
        "method": "log-ratio direction over F0, RMS and spectral-centroid diagnostics",
        "quality_claim": False,
        "neutral_generated_features": neutral_generated,
        "neutral_reference_features": neutral_reference,
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
