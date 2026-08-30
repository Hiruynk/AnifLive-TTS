from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np


def _read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise RuntimeError(f"Expected mono PCM16 WAV: {path}")
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    return sample_rate, np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 1.0


def _compare(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    common = min(int(baseline.size), int(candidate.size))
    left = baseline[:common]
    right = candidate[:common]
    difference = right - left
    signal = float(np.mean(np.square(left)))
    noise = float(np.mean(np.square(difference)))
    changed = np.flatnonzero(np.abs(difference) > (1.0 / 32768.0))
    spans: list[dict[str, int]] = []
    if changed.size:
        start = int(changed[0])
        previous = start
        for value in changed[1:]:
            current = int(value)
            if current > previous + 1:
                spans.append({"start": start, "end": previous + 1})
                start = current
            previous = current
        spans.append({"start": start, "end": previous + 1})
    return {
        "baseline_samples": int(baseline.size),
        "candidate_samples": int(candidate.size),
        "duration_difference_samples": int(candidate.size - baseline.size),
        "waveform_cosine": _cosine(left, right),
        "snr_db_vs_hard_natural": (
            10.0 * math.log10(max(signal, 1e-30) / max(noise, 1e-30))
        ),
        "rms_difference": math.sqrt(noise),
        "max_absolute_difference": float(np.max(np.abs(difference))),
        "changed_samples": int(changed.size),
        "changed_sample_ratio": float(changed.size) / max(1, common),
        "changed_spans": spans,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_rate, baseline = _read_wav(args.baseline.resolve())
    rows = []
    for item in args.candidate:
        if "=" not in item:
            raise ValueError("--candidate must be NAME=PATH")
        name, raw_path = item.split("=", 1)
        candidate_rate, candidate = _read_wav(Path(raw_path).resolve())
        if candidate_rate != baseline_rate:
            raise RuntimeError(f"Sample-rate mismatch for {name}")
        rows.append({"name": name, **_compare(baseline, candidate)})
    report = {
        "schema": 1,
        "purpose": "Expression transition waveform delta against hard-natural",
        "quality_claim": False,
        "sample_rate": baseline_rate,
        "rows": rows,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
