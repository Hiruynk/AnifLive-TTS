#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch

from aniflive_tts.backend.trt_builder import DetailedTensorRTLogger
from aniflive_tts.backend.trt_runtime import TensorRTRunner


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95 = ordered[max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))]
    return {
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "p95": p95,
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
    }


def _runner(path: Path) -> TensorRTRunner:
    return TensorRTRunner(path, logger=DetailedTensorRTLogger(echo=False))


def _outputs(runner: TensorRTRunner) -> dict[str, torch.Tensor]:
    dtype_map = {
        "DataType.HALF": torch.float16,
        "DataType.FLOAT": torch.float32,
        "DataType.INT64": torch.int64,
        "DataType.INT32": torch.int32,
    }
    return {
        name: torch.empty(
            tuple(int(value) for value in runner.engine.get_tensor_shape(name)),
            dtype=dtype_map[str(runner.engine.get_tensor_dtype(name))],
            device="cuda",
        )
        for name in runner.output_names
    }


def _inputs(
    samples: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    x_len: torch.Tensor,
    y_len: torch.Tensor,
    idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "samples": samples,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "x_len": x_len,
        "y_len": y_len,
        "idx": idx,
    }


def _timed(operation: Callable[[], None], stream: torch.cuda.Stream) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start.record(stream)
    operation()
    end.record(stream)
    stream.synchronize()
    return float(start.elapsed_time(end)), (time.perf_counter() - started) * 1000.0


def _measure(
    operations: dict[str, Callable[[], None]],
    stream: torch.cuda.Stream,
    *,
    warmup: int,
    runs: int,
) -> dict[str, dict[str, dict[str, float]]]:
    for _ in range(warmup):
        for operation in operations.values():
            operation()
    stream.synchronize()
    rows = {name: {"gpu": [], "wall": []} for name in operations}
    names = list(operations)
    for run in range(runs):
        order = names if run % 2 == 0 else list(reversed(names))
        for name in order:
            gpu_ms, wall_ms = _timed(operations[name], stream)
            rows[name]["gpu"].append(gpu_ms)
            rows[name]["wall"].append(wall_ms)
    return {
        name: {
            "gpu_ms": _summary(values["gpu"]),
            "wall_ms": _summary(values["wall"]),
        }
        for name, values in rows.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-step", type=Path, required=True)
    parser.add_argument("--baseline-block-h2", type=Path, required=True)
    parser.add_argument("--mtp-engine-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--acceptance-rate", type=float, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()
    if not 0.0 <= args.acceptance_rate <= 1.0:
        parser.error("--acceptance-rate must be between zero and one")

    baseline = _runner(args.baseline_step.resolve())
    block_h2 = _runner(args.baseline_block_h2.resolve())
    mtp_h1 = _runner(args.mtp_engine_dir.resolve() / "gpt_block_mtp_h1.engine")
    mtp_h2 = _runner(args.mtp_engine_dir.resolve() / "gpt_block_mtp_h2.engine")
    cache_shape = tuple(int(value) for value in baseline.engine.get_tensor_shape("k_cache"))
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    k_cache = torch.zeros(cache_shape, dtype=torch.float16, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    x_len = torch.tensor([50], dtype=torch.int64, device="cuda")
    y_len = torch.tensor([20], dtype=torch.int64, device="cuda")
    idx = torch.tensor([0], dtype=torch.int64, device="cuda")
    prefix = int(x_len.item() + y_len.item())
    k_cache[:, :, :prefix].normal_(mean=0.0, std=0.1)
    v_cache[:, :, :prefix].normal_(mean=0.0, std=0.1)
    sample_h1 = torch.randint(0, 1024, (1, 1), dtype=torch.int64, device="cuda")
    sample_h2 = torch.randint(0, 1024, (1, 2), dtype=torch.int64, device="cuda")
    stream = torch.cuda.Stream()
    runners = {
        "baseline_step": (baseline, sample_h1),
        "baseline_block_h2": (block_h2, sample_h2),
        "mtp_h1": (mtp_h1, sample_h1),
        "mtp_h2": (mtp_h2, sample_h2),
    }
    output_buffers = {name: _outputs(runner) for name, (runner, _) in runners.items()}

    def operation(name: str) -> Callable[[], None]:
        runner, samples = runners[name]

        def invoke() -> None:
            runner.infer(
                _inputs(samples, k_cache, v_cache, x_len, y_len, idx),
                outputs=output_buffers[name],
                stream=stream,
                synchronize=False,
                profile=False,
            )

        return invoke

    operations = {name: operation(name) for name in runners}
    performance = _measure(operations, stream, warmup=args.warmup, runs=args.runs)
    for name in operations:
        operations[name]()
    stream.synchronize()
    base_step = output_buffers["baseline_step"]
    base_block = output_buffers["baseline_block_h2"]
    mtp1 = output_buffers["mtp_h1"]
    mtp2 = output_buffers["mtp_h2"]
    numerical = {
        "h1_base_top1_exact": bool(
            torch.equal(
                base_step["topk_indices"][:, :1],
                mtp1["base_topk_indices"][:, 0, :1],
            )
        ),
        "h1_base_values_max_abs": float(
            (
                base_step["topk_values"].float()
                - mtp1["base_topk_values"][:, 0].float()
            )
            .abs()
            .max()
        ),
        "h2_base_top1_exact": bool(
            torch.equal(
                base_block["topk_indices"][:, :, :1],
                mtp2["base_topk_indices"][:, :, :1],
            )
        ),
        "h2_base_values_max_abs": float(
            (
                base_block["topk_values"].float()
                - mtp2["base_topk_values"].float()
            )
            .abs()
            .max()
        ),
    }
    step_ms = performance["baseline_step"]["gpu_ms"]["p50"]
    block_ms = performance["baseline_block_h2"]["gpu_ms"]["p50"]
    mtp_h1_ms = performance["mtp_h1"]["gpu_ms"]["p50"]
    mtp_h2_ms = performance["mtp_h2"]["gpu_ms"]["p50"]
    projection = {
        "accepted_tokens_per_h2_nfe": 1.0 + args.acceptance_rate,
        "steady_state_speedup_vs_step": (1.0 + args.acceptance_rate)
        * step_ms
        / mtp_h2_ms,
        "h1_overhead_ratio": mtp_h1_ms / step_ms,
        "h2_head_overhead_ratio": mtp_h2_ms / block_ms,
    }
    report: dict[str, Any] = {
        "schema": 1,
        "kind": "aniflive-tts-v1.2-mtp2-block-microbenchmark",
        "gpu": torch.cuda.get_device_name(0),
        "warmup": args.warmup,
        "runs": args.runs,
        "acceptance_rate": args.acceptance_rate,
        "performance": performance,
        "numerical": numerical,
        "projection": projection,
        "gate": {
            "base_numerical_parity": (
                numerical["h1_base_top1_exact"]
                and numerical["h2_base_top1_exact"]
                and numerical["h1_base_values_max_abs"] <= 0.03125
                and numerical["h2_base_values_max_abs"] <= 0.03125
            ),
            "projected_speedup_at_least_1_1x": (
                projection["steady_state_speedup_vs_step"] >= 1.1
            ),
        },
    }
    report["gate"]["passed"] = all(report["gate"].values())
    target = args.report.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": report["gate"], "projection": projection}, indent=2))
    print(target)
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
