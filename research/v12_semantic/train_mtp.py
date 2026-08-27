#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.v12_semantic.mtp_adapter import (  # noqa: E402
    MTPAdapter,
    MTPAdapterSpec,
    save_adapter,
    sha256_file,
)


def _checkpoint_head(checkpoint: Path) -> tuple[torch.Tensor, dict[str, int]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = payload["config"]["model"]
    weight = payload["weight"]["model.ar_predict_layer.weight"].float()
    dimensions = {
        "hidden_dim": int(config["hidden_dim"]),
        "vocab_size": int(config["vocab_size"]),
    }
    expected = (dimensions["vocab_size"], dimensions["hidden_dim"])
    if tuple(weight.shape) != expected:
        raise ValueError(f"GPT prediction head has shape {tuple(weight.shape)}, expected {expected}")
    return weight, dimensions


def _dataset(
    path: Path, heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, str]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        hidden = handle.get_tensor("hidden").float()
        targets = handle.get_tensor("future_targets").long()
        record_ids = handle.get_tensor("record_ids").long()
    if hidden.ndim != 2:
        raise ValueError("Dataset hidden tensor must have shape [rows, hidden_dim]")
    if targets.ndim != 2 or targets.shape[0] != hidden.shape[0]:
        raise ValueError("Dataset future_targets must align with hidden rows")
    if record_ids.ndim != 1 or record_ids.shape[0] != hidden.shape[0]:
        raise ValueError("Dataset record_ids must align with hidden rows")
    required = heads - 1
    if targets.shape[1] < required:
        raise ValueError(
            f"Dataset contains {targets.shape[1]} future targets, {required} required"
        )
    return hidden, targets[:, :required].contiguous(), record_ids, metadata


def _split_indices(
    record_ids: torch.Tensor,
    validation_fraction: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    unique_records = torch.unique(record_ids, sorted=True)
    if unique_records.numel() < 2:
        raise ValueError("MTP dataset needs at least two records for grouped validation")
    validation_records = max(1, int(unique_records.numel() * validation_fraction))
    validation_records = min(validation_records, int(unique_records.numel()) - 1)
    permutation = torch.randperm(unique_records.numel(), generator=generator)
    validation_groups = unique_records[permutation[:validation_records]]
    validation_mask = torch.isin(record_ids, validation_groups)
    validation_index = torch.nonzero(validation_mask, as_tuple=False).reshape(-1)
    training_index = torch.nonzero(~validation_mask, as_tuple=False).reshape(-1)
    return training_index, validation_index


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> list[float]:
    predictions = logits.argmax(dim=-1)
    return [
        float((predictions[:, index] == targets[:, index]).float().mean())
        for index in range(targets.shape[1])
    ]


def _evaluate(
    adapter: MTPAdapter,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    adapter.eval()
    with torch.no_grad():
        logits = adapter(hidden.to(device))
        labels = targets.to(device)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
        accuracy = _accuracy(logits, labels)
    return {"loss": float(loss), "top1_accuracy_by_future_head": accuracy}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--heads", type=int, choices=(2, 4), default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("validation fraction must be in (0, 0.5)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    checkpoint = args.gpt.resolve()
    checkpoint_sha = sha256_file(checkpoint)
    base_weight, dimensions = _checkpoint_head(checkpoint)
    hidden, targets, record_ids, dataset_metadata = _dataset(
        args.dataset.resolve(), args.heads
    )
    dataset_sha = sha256_file(args.dataset.resolve())
    dataset_base_sha = dataset_metadata.get("base_gpt_sha256")
    if dataset_base_sha != checkpoint_sha:
        raise ValueError(
            "MTP dataset checkpoint mismatch: "
            f"dataset={dataset_base_sha}, checkpoint={checkpoint_sha}"
        )
    if hidden.shape[1] != dimensions["hidden_dim"]:
        raise ValueError("MTP dataset hidden dimension does not match the checkpoint")

    generator = torch.Generator().manual_seed(args.seed)
    training_index, validation_index = _split_indices(
        record_ids, args.validation_fraction, generator
    )
    training = TensorDataset(hidden[training_index], targets[training_index])
    loader = DataLoader(
        training,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = MTPAdapterSpec(
        checkpoint_sha,
        hidden_dim=dimensions["hidden_dim"],
        vocab_size=dimensions["vocab_size"],
        heads=args.heads,
    )
    adapter = MTPAdapter(spec).to(device)
    adapter.initialize_from_base_head(base_weight.to(device))
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_weight = adapter.weight.detach().cpu().clone()
    for epoch in range(args.epochs):
        adapter.train()
        losses: list[float] = []
        for batch_hidden, batch_targets in loader:
            batch_hidden = batch_hidden.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = adapter(batch_hidden)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), batch_targets.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = _evaluate(
            adapter,
            hidden[validation_index],
            targets[validation_index],
            device,
        )
        row = {
            "epoch": epoch + 1,
            "training_loss": sum(losses) / len(losses),
            "validation": validation,
        }
        history.append(row)
        print(json.dumps(row))
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_weight = adapter.weight.detach().cpu().clone()

    with torch.no_grad():
        adapter.weight.copy_(best_weight.to(device))
    final_validation = _evaluate(
        adapter,
        hidden[validation_index],
        targets[validation_index],
        device,
    )
    save_adapter(
        adapter,
        args.output.resolve(),
        metadata={
            "training_corpus_hash": dataset_metadata.get("corpus_sha256", "unknown"),
            "training_dataset_sha256": dataset_sha,
            "training_rows": str(training_index.numel()),
            "validation_rows": str(validation_index.numel()),
        },
    )
    report = {
        "schema": 1,
        "kind": "aniflive-tts-v1.2-mtp-training",
        "adapter": str(args.output.resolve()),
        "base_gpt_sha256": checkpoint_sha,
        "dataset_sha256": dataset_sha,
        "heads": args.heads,
        "device": str(device),
        "training_rows": int(training_index.numel()),
        "validation_rows": int(validation_index.numel()),
        "training_records": int(torch.unique(record_ids[training_index]).numel()),
        "validation_records": int(torch.unique(record_ids[validation_index]).numel()),
        "best_validation": final_validation,
        "history": history,
    }
    target = args.report.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
