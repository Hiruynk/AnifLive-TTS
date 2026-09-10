from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from aniflive_tts.workstation_production_gates import (
    SPEAKER_GENERATED_MEDIAN_FLOOR,
    SPEAKER_GENERATED_P10_FLOOR,
    SPEAKER_MEDIAN_RETENTION_LIMIT,
    SPEAKER_P10_RETENTION_LIMIT,
)
from aniflive_tts.workstation_reference_selection import robust_speaker_centroid
from aniflive_tts.workstation_speaker import WorkstationSpeakerVerifier


class CalibrationError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _audio_paths(root: Path, *, limit: int | None) -> list[Path]:
    root = root.resolve(strict=True)
    candidates = [
        path
        for path in root.rglob("*.wav")
        if path.is_file() and not path.is_symlink()
    ]
    candidates.sort(
        key=lambda path: hashlib.sha256(
            path.relative_to(root).as_posix().encode("utf-8")
        ).hexdigest()
    )
    if limit is not None:
        candidates = candidates[:limit]
    return candidates


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as error:
        raise CalibrationError("soundfile is required") from error
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1)
    values = values.reshape(-1)
    if values.size < int(sample_rate) or not np.isfinite(values).all():
        raise CalibrationError(f"audio is too short or malformed: {path}")
    return values, int(sample_rate)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise CalibrationError("speaker embedding has zero norm")
    return float(np.dot(left, right) / denominator)


def _distribution(scores: list[float]) -> dict[str, float | int]:
    if not scores:
        raise CalibrationError("speaker score distribution is empty")
    values = np.asarray(scores, dtype=np.float64)
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def calibrate(
    *,
    source_dir: Path,
    generated_dir: Path,
    speaker_component: Path,
    source_limit: int | None,
) -> dict[str, Any]:
    source_paths = _audio_paths(source_dir, limit=source_limit)
    generated_paths = _audio_paths(generated_dir, limit=None)
    if len(source_paths) < 20:
        raise CalibrationError("source calibration requires at least 20 WAV files")
    if len(generated_paths) < 10:
        raise CalibrationError("generated calibration requires at least 10 WAV files")

    verifier = WorkstationSpeakerVerifier(speaker_component)
    source_embeddings = []
    for path in source_paths:
        audio, sample_rate = _load_audio(path)
        source_embeddings.append(verifier(audio, sample_rate))
    generated_embeddings = []
    for path in generated_paths:
        audio, sample_rate = _load_audio(path)
        generated_embeddings.append(verifier(audio, sample_rate))

    control_count = max(5, len(source_embeddings) // 5)
    centroid_embeddings = source_embeddings[:-control_count]
    control_embeddings = source_embeddings[-control_count:]
    centroid, retained = robust_speaker_centroid(centroid_embeddings)
    control_scores = [_cosine(embedding, centroid) for embedding in control_embeddings]
    generated_scores = [
        _cosine(embedding, centroid) for embedding in generated_embeddings
    ]
    control = _distribution(control_scores)
    generated = _distribution(generated_scores)
    median_retention = float(generated["median"]) / float(control["median"])
    p10_retention = float(generated["p10"]) / float(control["p10"])
    passed = (
        float(generated["median"]) >= SPEAKER_GENERATED_MEDIAN_FLOOR
        and float(generated["p10"]) >= SPEAKER_GENERATED_P10_FLOOR
        and median_retention >= SPEAKER_MEDIAN_RETENTION_LIMIT
        and p10_retention >= SPEAKER_P10_RETENTION_LIMIT
    )
    return {
        "schema": "aniflive-tts-speaker-gate-calibration-v1",
        "status": "passed" if passed else "failed",
        "method": "qualified-model-source-control-retention-v1",
        "source": {
            "root": str(source_dir.resolve(strict=True)),
            "sample_count": len(source_paths),
            "centroid_count": len(centroid_embeddings),
            "robust_centroid_retained_count": len(retained),
            "control_count": len(control_embeddings),
            "inventory_sha256": hashlib.sha256(
                "\n".join(_sha256_file(path) for path in source_paths).encode("ascii")
            ).hexdigest(),
        },
        "generated": {
            "root": str(generated_dir.resolve(strict=True)),
            "sample_count": len(generated_paths),
            "inventory_sha256": hashlib.sha256(
                "\n".join(_sha256_file(path) for path in generated_paths).encode("ascii")
            ).hexdigest(),
        },
        "control_distribution": control,
        "generated_distribution": generated,
        "retention": {
            "median_ratio": median_retention,
            "p10_ratio": p10_retention,
        },
        "limits": {
            "generated_median_floor": SPEAKER_GENERATED_MEDIAN_FLOOR,
            "generated_p10_floor": SPEAKER_GENERATED_P10_FLOOR,
            "median_retention_ratio": SPEAKER_MEDIAN_RETENTION_LIMIT,
            "p10_retention_ratio": SPEAKER_P10_RETENTION_LIMIT,
        },
        "speaker_component": verifier.manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--speaker-component", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-limit", type=int, default=120)
    args = parser.parse_args()
    if args.source_limit < 20:
        parser.error("--source-limit must be at least 20")
    report = calibrate(
        source_dir=args.source_dir,
        generated_dir=args.generated_dir,
        speaker_component=args.speaker_component,
        source_limit=args.source_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
