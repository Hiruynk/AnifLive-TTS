#!/usr/bin/env python
"""Distill a GPT-SoVITS Transformer into the v1.2 1:1 Hybrid student."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch import Tensor, nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.v12_semantic.hybrid_model import HybridT2SBackbone  # noqa: E402


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


def teacher_forcing_input(
    core: nn.Module,
    phonemes: Tensor,
    prompts: Tensor,
    bert: Tensor,
    targets: Tensor,
) -> tuple[Tensor, Tensor, slice]:
    if targets.ndim != 1 or targets.numel() == 0:
        raise ValueError("Semantic targets must be a non-empty vector")
    text = core.ar_text_position(
        core.ar_text_embedding(phonemes) + core.bert_proj(bert.transpose(1, 2))
    )
    audio_ids = torch.cat((prompts.reshape(-1), targets[:-1])).reshape(1, -1)
    audio = core.ar_audio_position(core.ar_audio_embedding(audio_ids))
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
    prediction_start = x_len + int(prompts.numel()) - 1
    prediction_stop = prediction_start + int(targets.numel())
    return (
        torch.cat((text, audio), dim=1),
        torch.cat((text_mask, audio_mask), dim=0),
        slice(prediction_start, prediction_stop),
    )


def teacher_forward(core: nn.Module, hidden: Tensor, mask: Tensor) -> Tensor:
    for block in core.t2s_transformer.blocks:
        hidden, _, _ = block.process_prompt(hidden, mask, None, True)
    return hidden


def student_forward(
    hybrid: HybridT2SBackbone, hidden: Tensor, mask: Tensor
) -> Tensor:
    for index in range(hybrid.num_layers):
        if index in hybrid.attention_layers:
            hidden, _, _ = hybrid.core.t2s_transformer.blocks[index].process_prompt(
                hidden, mask, None, True
            )
        else:
            hidden, _ = hybrid.mamba_layers[str(index)](hidden)
    return hidden


def skewed_reverse_kl(student_logits: Tensor, teacher_logits: Tensor, lam: float = 0.1) -> Tensor:
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(student_logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1.0 - lam) * teacher_probs + lam * student_probs
    student_log = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
    return (student_probs * (student_log - torch.log(mixed_probs))).sum(dim=-1).mean()


def split_records(
    records: list[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(records) < 2:
        raise ValueError("Hybrid distillation needs at least two records")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    count = max(1, min(len(records) - 1, round(len(records) * validation_fraction)))
    return shuffled[count:], shuffled[:count]


def read_record(
    handle: Any,
    record: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    prefix = record["prefix"]
    return (
        handle.get_tensor(f"{prefix}.phoneme_ids").long().to(device),
        handle.get_tensor(f"{prefix}.prompts").long().to(device),
        handle.get_tensor(f"{prefix}.bert_feature").to(device=device, dtype=dtype),
        handle.get_tensor(f"{prefix}.tokens").long().to(device),
    )


def evaluate(
    hybrid: HybridT2SBackbone,
    handle: Any,
    records: list[dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    hybrid.eval()
    totals = {
        "ce": 0.0,
        "skew_kl": 0.0,
        "hidden_mse": 0.0,
        "hidden_cosine": 0.0,
        "token_accuracy": 0.0,
    }
    token_count = 0
    with torch.inference_mode():
        for record in records:
            phonemes, prompts, bert, targets = read_record(
                handle, record, device=device, dtype=dtype
            )
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=device.type == "cuda" and dtype == torch.float16,
            ):
                inputs, mask, predicted = teacher_forcing_input(
                    hybrid.core, phonemes, prompts, bert, targets
                )
                teacher = teacher_forward(hybrid.core, inputs, mask)[:, predicted]
                student = student_forward(hybrid, inputs, mask)[:, predicted]
            teacher_logits = hybrid.core.ar_predict_layer(teacher).float()
            student_logits = hybrid.core.ar_predict_layer(student).float()
            length = int(targets.numel())
            totals["ce"] += float(
                F.cross_entropy(student_logits.reshape(-1, student_logits.shape[-1]), targets)
            ) * length
            totals["skew_kl"] += float(skewed_reverse_kl(student_logits, teacher_logits)) * length
            totals["hidden_mse"] += float(F.mse_loss(student.float(), teacher.float())) * length
            totals["hidden_cosine"] += float(
                F.cosine_similarity(student.float(), teacher.float(), dim=-1).mean()
            ) * length
            totals["token_accuracy"] += float(
                (student_logits.argmax(dim=-1).reshape(-1) == targets).float().mean()
            ) * length
            token_count += length
    return {name: value / token_count for name, value in totals.items()} | {
        "tokens": float(token_count)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-validation-records", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if args.epochs <= 0 or args.learning_rate <= 0:
        parser.error("epochs and learning rate must be positive")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("validation fraction must be in (0, 0.5)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16 if args.precision == "float16" else torch.float32
    checkpoint = args.gpt.resolve()
    capture = args.capture.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("capture_sha256") != sha256(capture):
        raise ValueError("Capture SHA256 does not match its manifest")
    train_records, validation_records = split_records(
        manifest["records"], args.validation_fraction, args.seed
    )
    if args.max_train_records:
        train_records = train_records[: args.max_train_records]
    if args.max_validation_records:
        validation_records = validation_records[: args.max_validation_records]

    base = load_model(checkpoint, args.source_dir.resolve())
    hybrid = HybridT2SBackbone(base.model).to(device=device).eval()
    optimizer = torch.optim.AdamW(
        hybrid.trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    best_loss = float("inf")
    best_state: dict[str, Tensor] = {}
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    with safe_open(str(capture), framework="pt", device="cpu") as handle:
        initial_validation = evaluate(
            hybrid,
            handle,
            validation_records,
            device=device,
            dtype=dtype,
        )
        for epoch in range(args.epochs):
            hybrid.train()
            hybrid.core.eval()
            random.Random(args.seed + epoch).shuffle(train_records)
            epoch_loss = 0.0
            epoch_tokens = 0
            for record in train_records:
                phonemes, prompts, bert, targets = read_record(
                    handle, record, device=device, dtype=dtype
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=device.type == "cuda" and dtype == torch.float16,
                ):
                    inputs, mask, predicted = teacher_forcing_input(
                        hybrid.core, phonemes, prompts, bert, targets
                    )
                    teacher = teacher_forward(hybrid.core, inputs, mask)[:, predicted]
                    teacher_logits = hybrid.core.ar_predict_layer(teacher).float()
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=device.type == "cuda" and dtype == torch.float16,
                ):
                    student = student_forward(hybrid, inputs, mask)[:, predicted]
                    student_logits = hybrid.core.ar_predict_layer(student).float()
                    ce = F.cross_entropy(
                        student_logits.reshape(-1, student_logits.shape[-1]), targets
                    )
                    logit_loss = skewed_reverse_kl(student_logits, teacher_logits)
                    hidden_loss = F.mse_loss(student.float(), teacher.float())
                    loss = ce + logit_loss + 0.25 * hidden_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(hybrid.trainable_parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                length = int(targets.numel())
                epoch_loss += float(loss.detach()) * length
                epoch_tokens += length
            validation = evaluate(
                hybrid,
                handle,
                validation_records,
                device=device,
                dtype=dtype,
            )
            row = {
                "epoch": epoch + 1,
                "training_loss": epoch_loss / epoch_tokens,
                "training_tokens": epoch_tokens,
                "validation": validation,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            score = validation["ce"] + validation["skew_kl"] + 0.25 * validation["hidden_mse"]
            if score < best_loss:
                best_loss = score
                best_state = {
                    name: value.detach().cpu().contiguous()
                    for name, value in hybrid.mamba_layers.state_dict().items()
                    if ".mamba." in f".{name}"
                }

    if not best_state:
        raise RuntimeError("Hybrid training did not produce an adapter state")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        best_state,
        str(output),
        metadata={
            "format": "aniflive-tts-v1.2-hybrid-adapter-v1",
            "base_gpt_sha256": sha256(checkpoint),
            "capture_sha256": sha256(capture),
            "layout": "blockbeg-1to1",
            "attention_layers": ",".join(map(str, sorted(hybrid.attention_layers))),
        },
    )
    report = {
        "schema": 1,
        "kind": "aniflive-tts-v1.2-hybrid-training",
        "status": "research",
        "base_gpt_sha256": sha256(checkpoint),
        "capture_sha256": sha256(capture),
        "adapter": str(output),
        "adapter_sha256": sha256(output),
        "precision": args.precision,
        "device": str(device),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "trainable_parameters": sum(
            parameter.numel() for parameter in hybrid.trainable_parameters()
        ),
        "initial_validation": initial_validation,
        "best_score": best_loss,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    target = args.report.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
