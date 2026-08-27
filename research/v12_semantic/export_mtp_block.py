#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnx
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.v12_semantic.export_gpt_block import (  # noqa: E402
    _initial_cache,
    _load_model,
    _sha256,
)
from research.v12_semantic.mtp_adapter import load_adapter  # noqa: E402


class MTPBlockStep(nn.Module):
    def __init__(self, t2s_model: nn.Module, future_weight: torch.Tensor) -> None:
        super().__init__()
        self.t2s_model = t2s_model
        self.register_buffer("future_weight", future_weight.contiguous())
        self.future_head_count = int(future_weight.shape[0])
        self.vocab_size = int(future_weight.shape[1])

    def forward(
        self,
        samples: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        x_len: torch.Tensor,
        y_len: torch.Tensor,
        idx: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        core = self.t2s_model.model
        embedding = core.ar_audio_embedding(samples)
        block_size = samples.shape[1]
        offsets = torch.arange(block_size, dtype=idx.dtype, device=samples.device)
        positions = y_len.reshape(-1)[0] + idx.reshape(-1)[0] + offsets
        cache_index = x_len.reshape(-1)[0] + y_len.reshape(-1)[0] + idx.reshape(-1)[0]
        pe_slice = core.ar_audio_position.pe.index_select(1, positions.reshape(-1))
        positioned = (
            embedding * core.ar_audio_position.x_scale
            + core.ar_audio_position.alpha
            * pe_slice.to(dtype=embedding.dtype, device=embedding.device)
        )
        hidden, k_new, v_new = core.t2s_transformer.decode_token_block(
            positioned,
            [value for value in k_cache],
            [value for value in v_cache],
            idx=cache_index,
        )
        base_logits = core.ar_predict_layer(hidden)
        future_logits = torch.nn.functional.linear(
            hidden,
            self.future_weight.reshape(
                self.future_head_count * self.vocab_size,
                self.future_weight.shape[-1],
            ),
        ).reshape(
            hidden.shape[0],
            hidden.shape[1],
            self.future_head_count,
            self.vocab_size,
        )
        if self.future_head_count == 1:
            # Preserve the existing MTP-2 engine I/O contract.
            future_logits = future_logits[:, :, 0, :]
        base_values, base_indices = torch.topk(base_logits, k=50, dim=-1)
        future_values, future_indices = torch.topk(future_logits, k=50, dim=-1)
        return (
            base_values,
            base_indices,
            future_values,
            future_indices,
            torch.stack(k_new, dim=0),
            torch.stack(v_new, dim=0),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    block_sizes = sorted(set(args.block_sizes))
    if any(size not in (1, 2, 4) for size in block_sizes):
        parser.error("--block-sizes currently accepts only 1, 2 and 4")

    checkpoint = args.gpt.resolve()
    checkpoint_sha = _sha256(checkpoint)
    model = _load_model(checkpoint, args.source_dir, allow_unsafe=False)
    adapter = load_adapter(
        args.adapter.resolve(), expected_base_gpt_sha256=checkpoint_sha
    )
    if adapter.spec.heads not in (2, 4):
        raise ValueError("This exporter requires an MTP-2 or MTP-4 adapter")
    future_weight = adapter.weight.detach()
    k_cache, v_cache, x_len, y_len = _initial_cache(model, args.max_len)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema": 1,
        "kind": f"aniflive-tts-v1.2-mtp{adapter.spec.heads}-block-onnx",
        "heads": adapter.spec.heads,
        "future_heads": adapter.spec.future_head_count,
        "base_gpt_sha256": checkpoint_sha,
        "adapter_sha256": _sha256(args.adapter.resolve()),
        "max_len": args.max_len,
        "blocks": {},
    }
    for block_size in block_sizes:
        torch.manual_seed(2200 + block_size)
        samples = torch.randint(0, 1024, (1, block_size), dtype=torch.long)
        wrapper = MTPBlockStep(model, future_weight)
        target = output_dir / f"gpt_block_mtp_h{block_size}.onnx"
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
                "base_topk_values",
                "base_topk_indices",
                "mtp_topk_values",
                "mtp_topk_indices",
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
        manifest["blocks"][str(block_size)] = {
            "onnx": target.name,
            "onnx_sha256": _sha256(target),
        }
        print(json.dumps({"block_size": block_size, "onnx": str(target)}))
    (output_dir / "mtp-block-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
