#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_audio(path: Path, maximum_samples: int) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    values = np.mean(np.asarray(audio, dtype=np.float32), axis=1)
    if int(sample_rate) != 16_000:
        common = __import__("math").gcd(int(sample_rate), 16_000)
        values = resample_poly(values, 16_000 // common, int(sample_rate) // common)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < 16_000:
        values = np.pad(values, (0, 16_000 - values.size))
    values = values[:maximum_samples]
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"Audio is empty or malformed: {path.name}")
    return values[None, :]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(np.float32).eps:
        return 0.0
    return float(np.dot(left.reshape(-1), right.reshape(-1)) / denominator)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--audio", action="append", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=32)
    args = parser.parse_args()
    if len(args.audio) < 2:
        raise SystemExit("qualification requires at least two independent audio files")
    if not 8 <= args.iterations <= 1_000:
        raise SystemExit("iterations must be between 8 and 1000")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    source_dir = args.source_dir.resolve(strict=True)
    sys.path.insert(0, str(source_dir))
    from run_trt_inference import TRTModule

    engine = args.engine.resolve(strict=True)
    onnx = args.onnx.resolve(strict=True)
    session = ort.InferenceSession(
        str(onnx),
        providers=["CPUExecutionProvider"],
        sess_options=ort.SessionOptions(),
    )
    if [item.name for item in session.get_inputs()] != ["audio"]:
        raise SystemExit("ONNX speaker input contract is unsupported")
    if [item.name for item in session.get_outputs()] != ["sv_embedding"]:
        raise SystemExit("ONNX speaker output contract is unsupported")
    module = TRTModule(str(engine), device="cuda")
    maximum_shape = module.input_max_shapes.get("audio")
    maximum_samples = int(maximum_shape[-1]) if maximum_shape else 180_000
    records: list[dict[str, object]] = []
    tensors: list[torch.Tensor] = []
    for audio_path in args.audio:
        path = audio_path.resolve(strict=True)
        values = load_audio(path, maximum_samples)
        reference = session.run(["sv_embedding"], {"audio": values})[0].astype(
            np.float32, copy=False
        )
        tensor = torch.from_numpy(values.copy()).to("cuda", dtype=module.tensor_dtype["audio"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        candidate = module({"audio": tensor})["sv_embedding"]
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        candidate_numpy = candidate.detach().float().cpu().numpy()
        similarity = cosine(reference, candidate_numpy)
        maximum_absolute_error = float(np.max(np.abs(reference - candidate_numpy)))
        finite = bool(np.isfinite(candidate_numpy).all())
        records.append(
            {
                "audio_sha256": sha256_file(path),
                "samples": int(values.shape[1]),
                "embedding_dimensions": int(candidate_numpy.size),
                "cosine": similarity,
                "maximum_absolute_error": maximum_absolute_error,
                "finite": finite,
                "enqueue_ms": elapsed_ms,
            }
        )
        tensors.append(tensor)

    stability_reference = module({"audio": tensors[0]})["sv_embedding"].detach().clone()
    torch.cuda.synchronize()
    maximum_stability_error = 0.0
    elapsed: list[float] = []
    for index in range(args.iterations):
        tensor = tensors[index % len(tensors)]
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = module({"audio": tensor})["sv_embedding"]
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - started) * 1000.0)
        if not bool(torch.isfinite(output).all().item()):
            raise SystemExit("TensorRT speaker output became non-finite")
        if index % len(tensors) == 0:
            maximum_stability_error = max(
                maximum_stability_error,
                float(torch.max(torch.abs(output - stability_reference)).item()),
            )

    gates = {
        "minimum_onnx_trt_cosine": 0.999,
        "maximum_absolute_error": 0.05,
        "maximum_repeat_error": 1e-6,
    }
    passed = (
        all(
            bool(record["finite"])
            and float(record["cosine"]) >= gates["minimum_onnx_trt_cosine"]
            and float(record["maximum_absolute_error"]) <= gates["maximum_absolute_error"]
            for record in records
        )
        and maximum_stability_error <= gates["maximum_repeat_error"]
    )
    try:
        import tensorrt as trt
    except ImportError:
        trt_version = None
    else:
        trt_version = trt.__version__
    report = {
        "schema": "aniflive-workstation-speaker-qualification-v1",
        "passed": passed,
        "backend": "TensorRT-11",
        "neural_fallback": False,
        "engine_sha256": sha256_file(engine),
        "engine_bytes": engine.stat().st_size,
        "onnx_sha256": sha256_file(onnx),
        "onnx_bytes": onnx.stat().st_size,
        "gates": gates,
        "audio_cases": records,
        "stability": {
            "iterations": args.iterations,
            "maximum_repeat_error": maximum_stability_error,
            "enqueue_ms_p50": float(np.percentile(np.asarray(elapsed), 50)),
            "enqueue_ms_p95": float(np.percentile(np.asarray(elapsed), 95)),
        },
        "runtime": {
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tensorrt": trt_version,
            "onnxruntime": ort.__version__,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
