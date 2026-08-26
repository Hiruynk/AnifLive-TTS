#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inputs(
    samples: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    x_len: torch.Tensor,
    y_len: torch.Tensor,
    idx: int,
) -> dict[str, torch.Tensor]:
    return {
        "samples": samples,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "x_len": x_len,
        "y_len": y_len,
        "idx": torch.tensor([idx], dtype=torch.int64, device="cuda"),
    }


def _validate_variant(
    *,
    baseline_engine: Path,
    block_engine: Path,
    block_size: int,
    first_token: torch.Tensor,
    initial_k: torch.Tensor,
    initial_v: torch.Tensor,
    x_len: torch.Tensor,
    y_len: torch.Tensor,
    sample_topk: Any,
    groups: int,
    seed: int,
) -> dict[str, Any]:
    from aniflive_tts.backend.trt_builder import DetailedTensorRTLogger
    from aniflive_tts.backend.trt_runtime import TensorRTRunner

    logger = DetailedTensorRTLogger(echo=False)
    baseline = TensorRTRunner(baseline_engine, logger=logger)
    block = TensorRTRunner(block_engine, logger=logger)
    current = first_token.clone()
    current_k = initial_k.clone()
    current_v = initial_v.clone()
    rows: list[dict[str, Any]] = []
    torch.cuda.manual_seed_all(seed)
    for group_index in range(groups):
        start_step = group_index * block_size
        source_k = current_k
        source_v = current_v
        consumed: list[torch.Tensor] = []
        baseline_result: dict[str, torch.Tensor] | None = None
        baseline_next: torch.Tensor | None = None
        sampling_state: torch.Tensor | None = None
        sampling_state_after: torch.Tensor | None = None
        for offset in range(block_size):
            consumed.append(current)
            baseline_result = baseline.infer(
                _inputs(
                    current,
                    current_k,
                    current_v,
                    x_len,
                    y_len,
                    start_step + offset,
                ),
                profile=False,
            ).outputs
            current_k = baseline_result["k_cache_new"]
            current_v = baseline_result["v_cache_new"]
            if offset == block_size - 1:
                sampling_state = torch.cuda.get_rng_state()
            baseline_next = sample_topk(
                baseline_result["topk_values"],
                baseline_result["topk_indices"],
                temperature=1.0,
                top_k=15,
                top_p=1.0,
            )
            if offset == block_size - 1:
                sampling_state_after = torch.cuda.get_rng_state()
            current = baseline_next
        if (
            baseline_result is None
            or baseline_next is None
            or sampling_state is None
            or sampling_state_after is None
        ):
            raise RuntimeError("Sequential validation did not produce a complete block")
        block_result = block.infer(
            _inputs(
                torch.cat(consumed, dim=1),
                source_k,
                source_v,
                x_len,
                y_len,
                start_step,
            ),
            profile=False,
        ).outputs
        block_values = block_result["topk_values"][:, -1, :]
        block_indices = block_result["topk_indices"][:, -1, :]
        torch.cuda.set_rng_state(sampling_state)
        block_next = sample_topk(
            block_values,
            block_indices,
            temperature=1.0,
            top_k=15,
            top_p=1.0,
        )
        torch.cuda.set_rng_state(sampling_state_after)
        expected_indices = baseline_result["topk_indices"]
        expected_values = baseline_result["topk_values"]
        rows.append(
            {
                "group": group_index,
                "top1_exact": bool(
                    torch.equal(expected_indices[:, :1], block_indices[:, :1])
                ),
                "top15_order_exact": bool(
                    torch.equal(expected_indices[:, :15], block_indices[:, :15])
                ),
                "top15_set_exact": bool(
                    torch.equal(
                        torch.sort(expected_indices[:, :15]).values,
                        torch.sort(block_indices[:, :15]).values,
                    )
                ),
                "sample_exact": bool(torch.equal(baseline_next, block_next)),
                "topk_values_max_abs": float(
                    (expected_values.float() - block_values.float()).abs().max()
                ),
                "k_cache_max_abs": float(
                    (current_k.float() - block_result["k_cache_new"].float())
                    .abs()
                    .max()
                ),
                "v_cache_max_abs": float(
                    (current_v.float() - block_result["v_cache_new"].float())
                    .abs()
                    .max()
                ),
            }
        )
        if int(current.item()) == 1024:
            break
    if not rows:
        raise RuntimeError("Real-path validation produced no block rows")
    return {
        "seed": seed,
        "groups": len(rows),
        "top1_exact": all(row["top1_exact"] for row in rows),
        "top15_order_exact": all(row["top15_order_exact"] for row in rows),
        "top15_set_exact": all(row["top15_set_exact"] for row in rows),
        "sample_exact": all(row["sample_exact"] for row in rows),
        "sample_match_rate": sum(row["sample_exact"] for row in rows) / len(rows),
        "topk_values_max_abs": max(row["topk_values_max_abs"] for row in rows),
        "k_cache_max_abs": max(row["k_cache_max_abs"] for row in rows),
        "v_cache_max_abs": max(row["v_cache_max_abs"] for row in rows),
        "failed_groups": [row for row in rows if not row["sample_exact"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--baseline-engine", type=Path, required=True)
    parser.add_argument("--block-engine-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--text", default="今日はいい天気ですね。")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--groups", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 4567, 9876])
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[2, 4])
    args = parser.parse_args()
    if args.groups < 1:
        parser.error("--groups must be positive")
    if any(size not in (2, 4) for size in args.block_sizes):
        parser.error("--block-sizes currently accepts only 2 and 4")

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    os.environ.update(
        {
            "ANIFLIVE_TTS_MODEL_PACKAGE": str(args.model_package.resolve()),
            "ANIFLIVE_TTS_SHARED_DIR": str(args.shared_dir.resolve()),
            "ANIFLIVE_TTS_SOURCE_DIR": str((repo / "minimal_inference").resolve()),
            "ANIFLIVE_TTS_WARM_RETENTION_SECONDS": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    from aniflive_tts.api import configure_runtime

    configure_runtime()
    from aniflive_tts import service as service_module

    runtime = service_module.TensorRTService(service_module.RuntimeSettings.from_env())
    runtime.load()
    captured: dict[str, torch.Tensor] = {}
    semantic_runtime = runtime._streamer.semantic_runtime
    original_prepare = semantic_runtime.prepare

    def capture_prepare(inputs: Any, **options: Any) -> Any:
        state = original_prepare(inputs, **options)
        captured.update(
            {
                "first_token": state.first_token.clone(),
                "k_cache": state.cache_pair[0][0].clone(),
                "v_cache": state.cache_pair[0][1].clone(),
                "x_len": state.x_length.clone(),
                "y_len": state.y_length.clone(),
            }
        )
        return state

    semantic_runtime.prepare = capture_prepare
    try:
        runtime.synthesize(
            service_module.SynthesisOptions(
                text=args.text,
                text_language=args.language,
                top_k=15,
                top_p=1.0,
                temperature=1.0,
                speed=1.0,
                pause_length=0.0,
                noise_scale=0.5,
                cut_punc="",
                seed=1234,
            )
        )
        if not captured:
            raise RuntimeError("Failed to capture a real GPT encoder state")
        variants: dict[str, Any] = {}
        for size in sorted(set(args.block_sizes)):
            block_engine = (
                args.block_engine_dir.resolve() / f"gpt_block_h{size}.engine"
            )
            seed_reports = [
                _validate_variant(
                    baseline_engine=args.baseline_engine.resolve(),
                    block_engine=block_engine,
                    block_size=size,
                    first_token=captured["first_token"],
                    initial_k=captured["k_cache"],
                    initial_v=captured["v_cache"],
                    x_len=captured["x_len"],
                    y_len=captured["y_len"],
                    sample_topk=runtime._streamer.sample_topk,
                    groups=args.groups,
                    seed=seed,
                )
                for seed in args.seeds
            ]
            variants[str(size)] = {
                "block_engine_sha256": _sha256(block_engine),
                "seeds": seed_reports,
                "gate": {
                    "fixed_seed_sampling_exact": all(
                        report["sample_exact"] for report in seed_reports
                    ),
                    "top1_exact": all(report["top1_exact"] for report in seed_reports),
                    "fp16_error_within_0_03125": all(
                        report[name] <= 0.03125
                        for report in seed_reports
                        for name in (
                            "topk_values_max_abs",
                            "k_cache_max_abs",
                            "v_cache_max_abs",
                        )
                    ),
                },
            }
            variants[str(size)]["gate"]["passed"] = all(
                variants[str(size)]["gate"].values()
            )
        report = {
            "schema": 1,
            "kind": "aniflive-tts-v1.2-gpt-block-real-path-validation",
            "workload": {"text": args.text, "language": args.language},
            "baseline_engine_sha256": _sha256(args.baseline_engine.resolve()),
            "variants": variants,
            "gate": {"passed": all(item["gate"]["passed"] for item in variants.values())},
        }
        target = args.report.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report["gate"]))
        print(target)
        return 0 if report["gate"]["passed"] else 1
    finally:
        semantic_runtime.prepare = original_prepare
        runtime.unload()


if __name__ == "__main__":
    raise SystemExit(main())
