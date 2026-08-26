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
    return TensorRTRunner(
        path,
        logger=DetailedTensorRTLogger(echo=False),
    )


def _output_buffers(runner: TensorRTRunner) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name in runner.output_names:
        shape = tuple(int(value) for value in runner.engine.get_tensor_shape(name))
        dtype = {
            "DataType.HALF": torch.float16,
            "DataType.FLOAT": torch.float32,
            "DataType.INT64": torch.int64,
            "DataType.INT32": torch.int32,
        }[str(runner.engine.get_tensor_dtype(name))]
        result[name] = torch.empty(shape, dtype=dtype, device="cuda")
    return result


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


def _numerical_check(
    baseline: TensorRTRunner,
    block: TensorRTRunner,
    samples: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    x_len: torch.Tensor,
    y_len: torch.Tensor,
) -> dict[str, Any]:
    sequential_values: list[torch.Tensor] = []
    sequential_indices: list[torch.Tensor] = []
    current_k = k_cache
    current_v = v_cache
    for offset in range(samples.shape[1]):
        result = baseline.infer(
            _inputs(
                samples[:, offset : offset + 1].contiguous(),
                current_k,
                current_v,
                x_len,
                y_len,
                torch.tensor([offset], dtype=torch.int64, device="cuda"),
            ),
            profile=False,
        ).outputs
        sequential_values.append(result["topk_values"][:, None, :].clone())
        sequential_indices.append(result["topk_indices"][:, None, :].clone())
        current_k = result["k_cache_new"]
        current_v = result["v_cache_new"]

    block_result = block.infer(
        _inputs(
            samples,
            k_cache,
            v_cache,
            x_len,
            y_len,
            torch.tensor([0], dtype=torch.int64, device="cuda"),
        ),
        profile=False,
    ).outputs
    expected_values = torch.cat(sequential_values, dim=1)
    expected_indices = torch.cat(sequential_indices, dim=1)
    value_error = float(
        (expected_values.float() - block_result["topk_values"].float()).abs().max()
    )
    k_error = float(
        (current_k.float() - block_result["k_cache_new"].float()).abs().max()
    )
    v_error = float(
        (current_v.float() - block_result["v_cache_new"].float()).abs().max()
    )
    position_matches = expected_indices == block_result["topk_indices"]
    return {
        "topk_indices_exact": bool(
            position_matches.all()
        ),
        "top1_indices_exact": bool(position_matches[:, :, :1].all()),
        "top15_indices_exact": bool(position_matches[:, :, :15].all()),
        "top15_sets_exact": bool(
            torch.equal(
                torch.sort(expected_indices[:, :, :15], dim=-1).values,
                torch.sort(block_result["topk_indices"][:, :, :15], dim=-1).values,
            )
        ),
        "top50_sets_exact": bool(
            torch.equal(
                torch.sort(expected_indices, dim=-1).values,
                torch.sort(block_result["topk_indices"], dim=-1).values,
            )
        ),
        "topk_position_match_rate": float(position_matches.float().mean()),
        "topk_values_max_abs": value_error,
        "k_cache_max_abs": k_error,
        "v_cache_max_abs": v_error,
    }


def _timed(
    operation: Callable[[], None], stream: torch.cuda.Stream
) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start.record(stream)
    operation()
    end.record(stream)
    stream.synchronize()
    return float(start.elapsed_time(end)), (time.perf_counter() - started) * 1000.0


def _benchmark_variant(
    baseline: TensorRTRunner,
    block: TensorRTRunner,
    samples: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    x_len: torch.Tensor,
    y_len: torch.Tensor,
    *,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    stream = torch.cuda.Stream()
    baseline_buffers = [_output_buffers(baseline), _output_buffers(baseline)]
    block_buffers = _output_buffers(block)
    idx_values = [
        torch.tensor([offset], dtype=torch.int64, device="cuda")
        for offset in range(samples.shape[1])
    ]

    def sequential() -> None:
        current_k = k_cache
        current_v = v_cache
        for offset in range(samples.shape[1]):
            outputs = baseline_buffers[offset % 2]
            baseline.infer(
                _inputs(
                    samples[:, offset : offset + 1].contiguous(),
                    current_k,
                    current_v,
                    x_len,
                    y_len,
                    idx_values[offset],
                ),
                outputs=outputs,
                stream=stream,
                synchronize=False,
                profile=False,
            )
            current_k = outputs["k_cache_new"]
            current_v = outputs["v_cache_new"]

    block_idx = torch.tensor([0], dtype=torch.int64, device="cuda")

    def one_block() -> None:
        block.infer(
            _inputs(samples, k_cache, v_cache, x_len, y_len, block_idx),
            outputs=block_buffers,
            stream=stream,
            synchronize=False,
            profile=False,
        )

    for _ in range(warmup):
        sequential()
        one_block()
    stream.synchronize()

    sequential_gpu: list[float] = []
    sequential_wall: list[float] = []
    block_gpu: list[float] = []
    block_wall: list[float] = []
    for index in range(runs):
        order = ((sequential, sequential_gpu, sequential_wall), (one_block, block_gpu, block_wall))
        if index % 2:
            order = tuple(reversed(order))
        for operation, gpu_rows, wall_rows in order:
            gpu_ms, wall_ms = _timed(operation, stream)
            gpu_rows.append(gpu_ms)
            wall_rows.append(wall_ms)

    result = {
        "sequential_gpu_ms": _summary(sequential_gpu),
        "sequential_wall_ms": _summary(sequential_wall),
        "block_gpu_ms": _summary(block_gpu),
        "block_wall_ms": _summary(block_wall),
    }
    result["gpu_speedup_p50"] = (
        result["sequential_gpu_ms"]["p50"] / result["block_gpu_ms"]["p50"]
    )
    result["wall_speedup_p50"] = (
        result["sequential_wall_ms"]["p50"] / result["block_wall_ms"]["p50"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-engine", type=Path, required=True)
    parser.add_argument("--block-engine-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        parser.error("warmup must be non-negative and runs must be positive")

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    baseline = _runner(args.baseline_engine.resolve())
    cache_shape = tuple(
        int(value) for value in baseline.engine.get_tensor_shape("k_cache")
    )
    k_cache = torch.zeros(cache_shape, dtype=torch.float16, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    x_len = torch.tensor([50], dtype=torch.int64, device="cuda")
    y_len = torch.tensor([20], dtype=torch.int64, device="cuda")
    prefix = int(x_len.item() + y_len.item())
    k_cache[:, :, :prefix].normal_(mean=0.0, std=0.1)
    v_cache[:, :, :prefix].normal_(mean=0.0, std=0.1)

    report: dict[str, Any] = {
        "schema": 1,
        "kind": "aniflive-tts-v1.2-gpt-block-microbenchmark",
        "gpu": torch.cuda.get_device_name(0),
        "warmup": args.warmup,
        "runs": args.runs,
        "variants": {},
    }
    for block_size in (2, 4):
        block = _runner(
            args.block_engine_dir.resolve() / f"gpt_block_h{block_size}.engine"
        )
        samples = torch.randint(
            0, 1024, (1, block_size), dtype=torch.int64, device="cuda"
        )
        numerical = _numerical_check(
            baseline, block, samples, k_cache, v_cache, x_len, y_len
        )
        performance = _benchmark_variant(
            baseline,
            block,
            samples,
            k_cache,
            v_cache,
            x_len,
            y_len,
            warmup=args.warmup,
            runs=args.runs,
        )
        report["variants"][str(block_size)] = {
            "numerical": numerical,
            "performance": performance,
            "gate": {
                "fp16_numerical_equivalence": (
                    numerical["top1_indices_exact"]
                    and numerical["top50_sets_exact"]
                    and numerical["topk_values_max_abs"] <= 0.03125
                    and numerical["k_cache_max_abs"] <= 0.03125
                    and numerical["v_cache_max_abs"] <= 0.03125
                ),
                "neural_speedup_at_least_1_5x": (
                    performance["gpu_speedup_p50"] >= 1.5
                ),
            },
        }
        print(json.dumps({"block_size": block_size, **report["variants"][str(block_size)]}))
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
