#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1, dtype=np.float32)
    if sample_rate != 16000:
        mono = soxr.resample(mono, sample_rate, 16000, quality="HQ")
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError(f"Invalid audio: {path}")
    # Complete WAV output is globally peak-normalized while true streaming
    # cannot know a future peak. Speaker identity evaluation must therefore
    # remove playback gain as a confounder before embedding both signals.
    mono = mono - float(np.mean(mono, dtype=np.float64))
    peak = float(np.max(np.abs(mono)))
    if peak > 1e-6:
        mono = mono / peak * 0.9
    return np.asarray(mono, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_dir.resolve()))
    from run_trt_inference import TRTModule

    module = TRTModule(str(args.engine.resolve()), device="cuda")

    def embed(path: Path) -> np.ndarray:
        waveform = torch.from_numpy(load_audio(path)).to("cuda", dtype=torch.float16)[None, :]
        embedding = module({"audio": waveform})["sv_embedding"]
        return embedding.detach().float().cpu().numpy().reshape(-1)

    reference = embed(args.reference)
    candidate = embed(args.candidate)
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(candidate))
    if denominator <= np.finfo(np.float32).eps:
        raise RuntimeError("TensorRT speaker engine produced a zero embedding")
    similarity = float(np.dot(reference, candidate) / denominator)
    report = {
        "schema": 1,
        "backend": "TensorRT-11 sv_embedding.engine",
        "pytorch_fallback": False,
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "embedding_dimensions": int(reference.size),
        "input_normalization": "mono, DC removal, peak=0.9",
        "speaker_cosine_similarity": similarity,
        "gate": {"minimum": 0.98, "passed": similarity >= 0.98},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["gate"]["passed"]:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
