#!/usr/bin/env python
"""Measure untrained Transformer-to-Mamba initialization on captured prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

from safetensors import safe_open
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.v12_semantic.hybrid_model import (  # noqa: E402
    HybridLayerState,
    HybridT2SBackbone,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint: Path, source_dir: Path) -> torch.nn.Module:
    sys.path.insert(0, str(source_dir.resolve()))
    from GPT_SoVITS.AR.models.t2s_lightning_module import (
        Text2SemanticLightningModule,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Text2SemanticLightningModule(payload["config"], "output", is_train=False)
    model.load_state_dict(payload["weight"])
    return model.eval()


def prompt_input(
    core: torch.nn.Module,
    phonemes: torch.Tensor,
    prompts: torch.Tensor,
    bert: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    text = core.ar_text_position(
        core.ar_text_embedding(phonemes) + core.bert_proj(bert.transpose(1, 2))
    )
    audio = core.ar_audio_position(core.ar_audio_embedding(prompts))
    x_len = int(text.shape[1])
    y_len = int(audio.shape[1])
    text_mask = torch.cat(
        (
            torch.zeros((x_len, x_len), dtype=torch.bool, device=text.device),
            torch.ones((x_len, y_len), dtype=torch.bool, device=text.device),
        ),
        dim=1,
    )
    audio_mask = torch.cat(
        (
            torch.zeros((y_len, x_len), dtype=torch.bool, device=text.device),
            torch.triu(
                torch.ones((y_len, y_len), dtype=torch.bool, device=text.device),
                diagonal=1,
            ),
        ),
        dim=1,
    )
    return torch.cat((text, audio), dim=1), torch.cat((text_mask, audio_mask), dim=0)


def tensor_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def transformer_state_bytes(k_cache: list[torch.Tensor], v_cache: list[torch.Tensor]) -> int:
    return sum(tensor_bytes(value) for value in (*k_cache, *v_cache))


def hybrid_state_bytes(states: list[HybridLayerState]) -> int:
    total = 0
    for state in states:
        if state.kind == "transformer":
            assert state.k_cache is not None and state.v_cache is not None
            total += tensor_bytes(state.k_cache) + tensor_bytes(state.v_cache)
        else:
            assert state.mamba is not None
            total += tensor_bytes(state.mamba.conv) + tensor_bytes(state.mamba.ssm)
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    device = torch.device(args.device)
    dtype = torch.float16 if args.precision == "float16" else torch.float32
    base = load_model(args.gpt.resolve(), args.source_dir.resolve())
    hybrid = HybridT2SBackbone(base.model).to(device=device, dtype=dtype).eval()
    core = hybrid.core
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    capture = args.capture.resolve()
    if manifest.get("capture_sha256") != sha256(capture):
        raise ValueError("Capture SHA256 does not match its manifest")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with safe_open(str(capture), framework="pt", device="cpu") as handle:
        with torch.inference_mode():
            for record in manifest["records"][: args.limit]:
                prefix = record["prefix"]
                phonemes = handle.get_tensor(f"{prefix}.phoneme_ids").long().to(device)
                prompts = handle.get_tensor(f"{prefix}.prompts").long().to(device)
                bert = handle.get_tensor(f"{prefix}.bert_feature").to(
                    device=device, dtype=dtype
                )
                inputs, mask = prompt_input(core, phonemes, prompts, bert)
                teacher, teacher_k, teacher_v = core.t2s_transformer.process_prompt(
                    inputs, mask, None, True
                )
                student, student_states = hybrid.process_prompt(inputs, mask)
                teacher_logits = core.ar_predict_layer(teacher[:, -1]).float()
                student_logits = core.ar_predict_layer(student[:, -1]).float()
                rows.append(
                    {
                        "item_id": record["item_id"],
                        "language": record["language"],
                        "prompt_length": int(inputs.shape[1]),
                        "hidden_cosine": float(
                            F.cosine_similarity(
                                teacher.float().reshape(-1, teacher.shape[-1]),
                                student.float().reshape(-1, student.shape[-1]),
                                dim=-1,
                            ).mean()
                        ),
                        "last_hidden_cosine": float(
                            F.cosine_similarity(
                                teacher[:, -1].float(), student[:, -1].float(), dim=-1
                            ).mean()
                        ),
                        "logit_kl": float(
                            F.kl_div(
                                F.log_softmax(student_logits, dim=-1),
                                F.softmax(teacher_logits, dim=-1),
                                reduction="batchmean",
                            )
                        ),
                        "top1_agreement": bool(
                            torch.equal(
                                teacher_logits.argmax(dim=-1),
                                student_logits.argmax(dim=-1),
                            )
                        ),
                        "teacher_state_bytes": transformer_state_bytes(
                            teacher_k, teacher_v
                        ),
                        "hybrid_state_bytes": hybrid_state_bytes(student_states),
                    }
                )
    elapsed = time.perf_counter() - started
    report = {
        "schema": 1,
        "kind": "aniflive-tts-v1.2-hybrid-initialization",
        "status": "measurement-only",
        "base_gpt_sha256": sha256(args.gpt.resolve()),
        "precision": args.precision,
        "device": str(device),
        "attention_layers": sorted(hybrid.attention_layers),
        "mamba_layers": sorted(int(index) for index in hybrid.mamba_layers),
        "trainable_parameters": sum(
            parameter.numel() for parameter in hybrid.trainable_parameters()
        ),
        "elapsed_seconds": elapsed,
        "records": rows,
        "summary": {
            "hidden_cosine_mean": sum(row["hidden_cosine"] for row in rows) / len(rows),
            "last_hidden_cosine_mean": sum(
                row["last_hidden_cosine"] for row in rows
            )
            / len(rows),
            "logit_kl_mean": sum(row["logit_kl"] for row in rows) / len(rows),
            "top1_agreement_rate": sum(row["top1_agreement"] for row in rows)
            / len(rows),
            "state_byte_ratio_mean": sum(
                row["hybrid_state_bytes"] / row["teacher_state_bytes"] for row in rows
            )
            / len(rows),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
