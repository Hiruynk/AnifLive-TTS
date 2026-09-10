#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


RATE = 16_000
SUPPORTED = frozenset({".wav", ".flac"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic, private qualification recording from a target "
            "speaker corpus and an optional non-target interferer."
        )
    )
    parser.add_argument("--target-directory", type=Path, required=True)
    parser.add_argument("--interferer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-seconds", type=float, default=1_860.0)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> np.ndarray:
    from scipy.signal import resample_poly

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    values = np.mean(np.asarray(audio, dtype=np.float32), axis=1).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"audio is empty or malformed: {path.name}")
    if int(rate) != RATE:
        divisor = math.gcd(int(rate), RATE)
        values = resample_poly(values, RATE // divisor, int(rate) // divisor).astype(
            np.float32, copy=False
        )
    return np.clip(values, -1.0, 1.0)


def _cyclic(values: np.ndarray, length: int, offset: int) -> np.ndarray:
    if values.size == 0:
        raise ValueError("interferer is empty")
    indices = (np.arange(length, dtype=np.int64) + offset) % values.size
    return values[indices]


def main() -> int:
    arguments = _parser().parse_args()
    target_root = arguments.target_directory.resolve(strict=True)
    if not target_root.is_dir() or target_root.is_symlink():
        raise SystemExit("target-directory must be a regular local directory")
    if not math.isfinite(arguments.minimum_seconds) or arguments.minimum_seconds < 1_800:
        raise SystemExit("minimum-seconds must be at least 1800")
    target_paths = sorted(
        path
        for path in target_root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.suffix.casefold() in SUPPORTED
    )
    if not target_paths:
        raise SystemExit("target-directory contains no WAV or FLAC files")
    interferer = _read(arguments.interferer.resolve(strict=True)) if arguments.interferer else None
    silence = np.zeros(round(0.35 * RATE), dtype=np.float32)
    target_audio = [_read(path) for path in target_paths]
    target_hashes = [_sha256(path) for path in target_paths]
    interferer_hash = _sha256(arguments.interferer.resolve(strict=True)) if arguments.interferer else None
    pieces: list[np.ndarray] = []
    timeline: list[dict[str, int | str]] = []
    counts = {"target": 0, "overlap": 0, "non_target": 0}
    total_samples = 0
    position = 0
    interferer_offset = 0
    minimum_samples = math.ceil(arguments.minimum_seconds * RATE)
    while total_samples < minimum_samples:
        target = target_audio[position % len(target_audio)]
        kind = "target"
        payload = target
        if interferer is not None and position % 17 == 16:
            contaminant = _cyclic(interferer, target.size, interferer_offset)
            interferer_offset = (interferer_offset + target.size) % interferer.size
            payload = np.clip(target + 0.22 * contaminant, -1.0, 1.0)
            kind = "overlap"
        elif interferer is not None and position % 13 == 12:
            length = min(max(round(3.0 * RATE), target.size), round(12.0 * RATE))
            payload = 0.65 * _cyclic(interferer, length, interferer_offset)
            interferer_offset = (interferer_offset + length) % interferer.size
            kind = "non_target"
        start_sample = total_samples
        end_sample = start_sample + payload.size
        timeline.append(
            {
                "position": position,
                "kind": kind,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "source_sha256": (
                    interferer_hash
                    if kind == "non_target"
                    else target_hashes[position % len(target_hashes)]
                ),
            }
        )
        pieces.extend((payload.astype(np.float32, copy=False), silence))
        counts[kind] += 1
        total_samples += payload.size + silence.size
        position += 1
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), np.concatenate(pieces), RATE, subtype="PCM_16")
    duration = sf.info(str(output)).duration
    if duration < arguments.minimum_seconds:
        raise RuntimeError("qualification fixture did not reach the requested duration")
    report = {
        "schema": "aniflive-voice-acquisition-fixture-v1",
        "sample_rate": RATE,
        "duration_seconds": duration,
        "target_source_count": len(target_paths),
        "target_source_sha256": target_hashes,
        "interferer_sha256": interferer_hash,
        "composition": counts,
        "timeline": timeline,
        "output_sha256": _sha256(output),
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
