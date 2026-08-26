#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import onnx
import torch
from torch import nn
from torch.nn import functional as F


class GPTBlockStep(nn.Module):
    def __init__(self, t2s_model: nn.Module) -> None:
        super().__init__()
        self.t2s_model = t2s_model

    def forward(
        self,
        samples: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        x_len: torch.Tensor,
        y_len: torch.Tensor,
        idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, k_new, v_new = self.t2s_model.model.infer_next_stage_block(
            samples,
            [value for value in k_cache],
            [value for value in v_cache],
            x_len.reshape(-1)[0],
            y_len.reshape(-1)[0],
            idx.reshape(-1)[0],
        )
        topk_values, topk_indices = torch.topk(logits, k=50, dim=-1)
        return (
            topk_values,
            topk_indices,
            torch.stack(k_new, dim=0),
            torch.stack(v_new, dim=0),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(checkpoint: Path, source_dir: Path, allow_unsafe: bool) -> nn.Module:
    sys.path.insert(0, str(source_dir.resolve()))
    from GPT_SoVITS.AR.models.t2s_lightning_module import (
        Text2SemanticLightningModule,
    )

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:
        if not allow_unsafe:
            raise RuntimeError(
                "Checkpoint requires unsafe pickle; use only a trusted local checkpoint "
                "with --allow-unsafe-pickle"
            ) from error
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Text2SemanticLightningModule(payload["config"], "output", is_train=False)
    model.load_state_dict(payload["weight"])
    model.eval()
    return model


def _initial_cache(
    model: nn.Module, max_len: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1234)
    phonemes = torch.randint(0, 512, (1, 50), dtype=torch.long)
    prompts = torch.randint(0, 1024, (1, 20), dtype=torch.long)
    bert = torch.randn(1, 1024, 50)
    with torch.no_grad():
        _, k_cache, v_cache, x_len, y_len = model.model.infer_first_stage(
            phonemes, prompts, bert
        )
    k_stacked = torch.stack(k_cache, dim=0)
    v_stacked = torch.stack(v_cache, dim=0)
    if k_stacked.shape[2] > max_len:
        raise ValueError("max_len is shorter than the prompt cache")
    k_padded = F.pad(k_stacked, (0, 0, 0, max_len - k_stacked.shape[2]))
    v_padded = F.pad(v_stacked, (0, 0, 0, max_len - v_stacked.shape[2]))
    return (
        k_padded,
        v_padded,
        torch.tensor([x_len], dtype=torch.long),
        torch.tensor([y_len], dtype=torch.long),
    )


def _verify_equivalence(
    model: nn.Module,
    samples: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    x_len: torch.Tensor,
    y_len: torch.Tensor,
) -> dict[str, float]:
    sequential_k = [value.clone() for value in k_cache]
    sequential_v = [value.clone() for value in v_cache]
    sequential_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for offset in range(samples.shape[1]):
            logits, sequential_k, sequential_v = model.model.infer_next_stage(
                samples[:, offset : offset + 1],
                sequential_k,
                sequential_v,
                x_len[0],
                y_len[0],
                torch.tensor(offset, dtype=torch.long),
            )
            sequential_logits.append(logits[:, None, :])
        block_logits, block_k, block_v = model.model.infer_next_stage_block(
            samples,
            [value.clone() for value in k_cache],
            [value.clone() for value in v_cache],
            x_len[0],
            y_len[0],
            torch.tensor(0, dtype=torch.long),
        )
    sequential = torch.cat(sequential_logits, dim=1)
    k_sequential = torch.stack(sequential_k, dim=0)
    v_sequential = torch.stack(sequential_v, dim=0)
    k_block = torch.stack(block_k, dim=0)
    v_block = torch.stack(block_v, dim=0)
    errors = {
        "logits_max_abs": float((sequential - block_logits).abs().max()),
        "k_cache_max_abs": float((k_sequential - k_block).abs().max()),
        "v_cache_max_abs": float((v_sequential - v_block).abs().max()),
    }
    torch.testing.assert_close(block_logits, sequential, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(k_block, k_sequential, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(v_block, v_sequential, rtol=1e-4, atol=1e-5)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--allow-unsafe-pickle", action="store_true")
    args = parser.parse_args()
    block_sizes = sorted(set(args.block_sizes))
    if any(size not in (2, 4) for size in block_sizes):
        parser.error("--block-sizes currently accepts only 2 and 4")

    checkpoint = args.gpt.resolve()
    model = _load_model(checkpoint, args.source_dir, args.allow_unsafe_pickle)
    k_cache, v_cache, x_len, y_len = _initial_cache(model, args.max_len)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "schema": 1,
        "kind": "aniflive-tts-v1.2-gpt-block-onnx",
        "base_gpt_sha256": _sha256(checkpoint),
        "max_len": args.max_len,
        "blocks": {},
    }
    for block_size in block_sizes:
        torch.manual_seed(1200 + block_size)
        samples = torch.randint(0, 1024, (1, block_size), dtype=torch.long)
        errors = _verify_equivalence(
            model, samples, k_cache, v_cache, x_len, y_len
        )
        wrapper = GPTBlockStep(model)
        target = output_dir / f"gpt_block_h{block_size}.onnx"
        torch.onnx.export(
            wrapper,
            (
                samples,
                k_cache,
                v_cache,
                x_len,
                y_len,
                torch.tensor([0], dtype=torch.long),
            ),
            str(target),
            input_names=["samples", "k_cache", "v_cache", "x_len", "y_len", "idx"],
            output_names=[
                "topk_values",
                "topk_indices",
                "k_cache_new",
                "v_cache_new",
            ],
            dynamic_axes={
                "k_cache": {1: "batch_size"},
                "v_cache": {1: "batch_size"},
                "x_len": {0: "one"},
                "y_len": {0: "one"},
                "idx": {0: "one"},
            },
            opset_version=20,
            dynamo=False,
        )
        onnx.checker.check_model(str(target), full_check=True)
        metadata["blocks"][str(block_size)] = {
            "onnx": target.name,
            "onnx_sha256": _sha256(target),
            "equivalence": errors,
        }
        print(json.dumps({"block_size": block_size, **errors}))
    (output_dir / "gpt_block_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
