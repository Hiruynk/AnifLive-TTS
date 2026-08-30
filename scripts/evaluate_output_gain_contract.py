from __future__ import annotations

import argparse
import io
import json
import math
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np


def _read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as source:
        sample_rate = source.getframerate()
        pcm = source.readframes(source.getnframes())
    return sample_rate, np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def _wav_bytes(sample_rate: int, audio: np.ndarray) -> bytes:
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(pcm.tobytes())
    return output.getvalue()


def _complete_normalize(audio: np.ndarray) -> np.ndarray:
    result = np.asarray(audio, dtype=np.float32).copy()
    result -= float(np.mean(result, dtype=np.float64))
    peak = float(np.max(np.abs(result))) if result.size else 0.0
    if peak > 1e-5:
        result *= 0.9 / peak
    return result


def _causal_limiter(
    audio: np.ndarray,
    *,
    input_gain: float,
    ceiling: float,
    release_ms: float,
    sample_rate: int,
) -> np.ndarray:
    result = np.empty_like(audio, dtype=np.float32)
    release = math.exp(-1.0 / max(1.0, sample_rate * release_ms / 1000.0))
    envelope = 1.0
    for index, sample in enumerate(np.asarray(audio, dtype=np.float32)):
        driven = float(sample) * input_gain
        required = min(1.0, ceiling / max(abs(driven), 1e-12))
        if required < envelope:
            envelope = required
        else:
            envelope = 1.0 - (1.0 - envelope) * release
        result[index] = driven * envelope
    return result


def _amplitude(audio: np.ndarray) -> dict[str, float]:
    values = np.asarray(audio, dtype=np.float64)
    return {
        "peak": float(np.max(np.abs(values))) if values.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0,
        "dc_offset": float(np.mean(values)) if values.size else 0.0,
        "clipped_fraction": float(np.mean(np.abs(values) >= 0.999)) if values.size else 0.0,
    }


def _metrics(left: np.ndarray, right: np.ndarray, sample_rate: int) -> dict[str, Any]:
    from aniflive_tts.backend.audio_quality import (
        AudioQualityConfig,
        _align,
        _spectral_metrics,
    )

    aligned_left, aligned_right, alignment = _align(left, right, sample_rate, 1.0)
    spectral, _, _, _ = _spectral_metrics(
        aligned_left,
        aligned_right,
        sample_rate,
        AudioQualityConfig(),
    )
    return {"alignment": alignment, "spectral": spectral}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--complete-raw", type=Path, required=True)
    parser.add_argument("--stream-raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-gain", type=float, required=True)
    parser.add_argument("--limiter-gain", type=float, default=3.5)
    parser.add_argument("--limiter-ceiling", type=float, default=0.9)
    parser.add_argument("--limiter-release-ms", type=float, default=120.0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve() / "src"))
    if not 0.1 <= args.fixed_gain <= 8.0:
        raise ValueError("--fixed-gain must be from 0.1 to 8")
    if not 0.1 <= args.limiter_gain <= 8.0:
        raise ValueError("--limiter-gain must be from 0.1 to 8")
    if not 0.1 <= args.limiter_ceiling <= 0.99:
        raise ValueError("--limiter-ceiling must be from 0.1 to 0.99")
    complete_rate, complete_raw = _read_wav(args.complete_raw.resolve())
    stream_rate, stream_raw = _read_wav(args.stream_raw.resolve())
    if complete_rate != stream_rate:
        raise ValueError("Input sample rates differ")
    sample_rate = complete_rate
    variants = {
        "g0-current-mismatch": (
            _complete_normalize(complete_raw),
            stream_raw,
        ),
        "g1-shared-raw": (complete_raw, stream_raw),
        "g2-shared-fixed-gain": (
            complete_raw * args.fixed_gain,
            stream_raw * args.fixed_gain,
        ),
        "g3-shared-causal-limiter": (
            _causal_limiter(
                complete_raw,
                input_gain=args.limiter_gain,
                ceiling=args.limiter_ceiling,
                release_ms=args.limiter_release_ms,
                sample_rate=sample_rate,
            ),
            _causal_limiter(
                stream_raw,
                input_gain=args.limiter_gain,
                ceiling=args.limiter_ceiling,
                release_ms=args.limiter_release_ms,
                sample_rate=sample_rate,
            ),
        ),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": 1,
        "fixed_gain": args.fixed_gain,
        "limiter": {
            "input_gain": args.limiter_gain,
            "ceiling": args.limiter_ceiling,
            "release_ms": args.limiter_release_ms,
        },
        "variants": {},
    }
    for name, (complete, stream) in variants.items():
        complete_quantized = np.frombuffer(
            _wav_bytes(sample_rate, complete)[44:], dtype="<i2"
        ).astype(np.float32) / 32768.0
        stream_quantized = np.frombuffer(
            _wav_bytes(sample_rate, stream)[44:], dtype="<i2"
        ).astype(np.float32) / 32768.0
        metrics = _metrics(complete_quantized, stream_quantized, sample_rate)
        report["variants"][name] = {
            "stream_vs_complete": metrics,
            "complete_amplitude": _amplitude(complete_quantized),
            "stream_amplitude": _amplitude(stream_quantized),
            "passes_log_mel_0_99": (
                metrics["spectral"]["log_mel_cosine_similarity"] >= 0.99
            ),
        }
        (output_dir / f"{name}-complete.wav").write_bytes(
            _wav_bytes(sample_rate, complete)
        )
        (output_dir / f"{name}-stream.wav").write_bytes(
            _wav_bytes(sample_rate, stream)
        )
    report["decision"] = {
        "eligible": [
            name
            for name, value in report["variants"].items()
            if value["passes_log_mel_0_99"]
        ],
        "selection_rule": (
            "Prefer the least signal-altering shared causal contract that passes; "
            "subjective listening remains mandatory before promotion."
        ),
    }
    path = output_dir / "output-gain-contract.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["decision"], indent=2))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
