from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


def _runner_inputs(
    samples: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    x_len: torch.Tensor,
    y_len: torch.Tensor,
    index: int,
) -> dict[str, torch.Tensor]:
    return {
        "samples": samples.to(torch.int64),
        "k_cache": k_cache,
        "v_cache": v_cache,
        "x_len": x_len.to("cuda"),
        "y_len": y_len.to("cuda"),
        "idx": torch.tensor([index], dtype=torch.int64, device="cuda"),
    }


def _top1(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return outputs["base_topk_indices"][:, -1:, :1].reshape(1, 1).to(torch.int64)


def _base_row(outputs: dict[str, torch.Tensor], row: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        outputs["base_topk_values"][:, row, :],
        outputs["base_topk_indices"][:, row, :],
    )


def _compare(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    *,
    candidate_row: int,
) -> dict[str, Any]:
    ref_values, ref_indices = _base_row(reference, 0)
    candidate_values, candidate_indices = _base_row(candidate, candidate_row)
    return {
        "top1_exact": bool(
            torch.equal(ref_indices[:, :1], candidate_indices[:, :1])
        ),
        "top50_order_exact": bool(torch.equal(ref_indices, candidate_indices)),
        "topk_values_max_abs": float(
            (ref_values.float() - candidate_values.float()).abs().max()
        ),
        "k_cache_max_abs": float(
            (
                reference["k_cache_new"].float()
                - candidate["k_cache_new"].float()
            )
            .abs()
            .max()
        ),
        "v_cache_max_abs": float(
            (
                reference["v_cache_new"].float()
                - candidate["v_cache_new"].float()
            )
            .abs()
            .max()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--h1-engine", type=Path, required=True)
    parser.add_argument("--h2-engine", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--text", default="今日はいい天気ですね。")
    parser.add_argument("--language", default="ja")
    args = parser.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    os.environ.update(
        {
            "ANIFLIVE_TTS_MODEL_PACKAGE": str(args.model_package.resolve()),
            "ANIFLIVE_TTS_SHARED_DIR": str(args.shared_dir.resolve()),
            "ANIFLIVE_TTS_SOURCE_DIR": str((repo / "minimal_inference").resolve()),
            "ANIFLIVE_TTS_SEMANTIC_BACKEND": "transformer",
            "ANIFLIVE_TTS_WARM_RETENTION_SECONDS": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )

    from aniflive_tts.api import configure_runtime
    from aniflive_tts.backend.trt_builder import DetailedTensorRTLogger
    from aniflive_tts.backend.trt_runtime import TensorRTRunner

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

        logger = DetailedTensorRTLogger(echo=False)
        h1 = TensorRTRunner(args.h1_engine.resolve(), logger=logger)
        h2 = TensorRTRunner(args.h2_engine.resolve(), logger=logger)

        def run(
            runner: TensorRTRunner,
            samples: torch.Tensor,
            k_cache: torch.Tensor,
            v_cache: torch.Tensor,
            index: int,
        ) -> dict[str, torch.Tensor]:
            return runner.infer(
                _runner_inputs(
                    samples,
                    k_cache,
                    v_cache,
                    captured["x_len"],
                    captured["y_len"],
                    index,
                ),
                profile=False,
            ).outputs

        first = captured["first_token"]
        step0 = run(h1, first, captured["k_cache"], captured["v_cache"], 0)
        token1 = _top1(step0)
        step1 = run(h1, token1, step0["k_cache_new"], step0["v_cache_new"], 1)
        token2 = _top1(step1)
        step2 = run(h1, token2, step1["k_cache_new"], step1["v_cache_new"], 2)

        accepted = run(
            h2,
            torch.cat((token1, token2), dim=1),
            step0["k_cache_new"],
            step0["v_cache_new"],
            1,
        )

        wrong = (token2 + 1) % 1024
        rejected_block = run(
            h2,
            torch.cat((token1, wrong), dim=1),
            step0["k_cache_new"],
            step0["v_cache_new"],
            1,
        )
        corrected = run(
            h1,
            token2,
            rejected_block["k_cache_new"],
            rejected_block["v_cache_new"],
            2,
        )
        torch.cuda.synchronize()

        report = {
            "schema": 1,
            "kind": "aniflive-tts-v1.2-mtp-cache-repair-validation",
            "tokens": {
                "first": int(first.item()),
                "token1": int(token1.item()),
                "token2": int(token2.item()),
                "forced_wrong_draft": int(wrong.item()),
            },
            "h2_verification": {
                "accepted_row0_top1_matches_sequential_step1": bool(
                    torch.equal(
                        _base_row(step1, 0)[1][:, :1],
                        _base_row(accepted, 0)[1][:, :1],
                    )
                ),
                "rejected_row0_top1_matches_sequential_step1": bool(
                    torch.equal(
                        _base_row(step1, 0)[1][:, :1],
                        _base_row(rejected_block, 0)[1][:, :1],
                    )
                ),
            },
            "accepted_cache": _compare(step2, accepted, candidate_row=1),
            "rejected_then_corrected_cache": _compare(
                step2, corrected, candidate_row=0
            ),
        }
        tolerance = 0.03125
        for name in ("accepted_cache", "rejected_then_corrected_cache"):
            item = report[name]
            item["passed"] = bool(
                item["top1_exact"]
                and item["topk_values_max_abs"] <= tolerance
                and item["k_cache_max_abs"] <= tolerance
                and item["v_cache_max_abs"] <= tolerance
            )
        report["gate"] = {
            "passed": bool(
                all(report["h2_verification"].values())
                and report["accepted_cache"]["passed"]
                and report["rejected_then_corrected_cache"]["passed"]
            )
        }
        target = args.report.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["gate"]["passed"] else 1
    finally:
        semantic_runtime.prepare = original_prepare
        runtime.unload()


if __name__ == "__main__":
    raise SystemExit(main())
