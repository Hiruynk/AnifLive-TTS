#!/usr/bin/env python
"""Measure the offline upper bound of MTP-4 transition-aware decoding.

The transition matrix is built only from the training record split.  A first
half of held-out records selects top-k/transition weight; the remaining held-
out records report the selected configuration.  This tool does not alter the
production runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from safetensors import safe_open
import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.v12_semantic.mtp_adapter import (  # noqa: E402
    load_adapter,
    sha256_file,
)
from research.v12_semantic.train_mtp import _split_indices  # noqa: E402


def viterbi_decode(
    log_probs: Tensor,
    transition_log_probs: Tensor,
    *,
    top_k: int,
    transition_weight: float,
) -> Tensor:
    if log_probs.ndim != 3:
        raise ValueError("log_probs must have shape [batch, heads, vocab]")
    if transition_log_probs.ndim != 2:
        raise ValueError("transition matrix must have shape [vocab, vocab]")
    batch, heads, vocab = log_probs.shape
    if transition_log_probs.shape != (vocab, vocab):
        raise ValueError("transition matrix vocabulary does not match logits")
    if not 1 <= top_k <= vocab or transition_weight < 0:
        raise ValueError("invalid Viterbi configuration")
    values, ids = torch.topk(log_probs, top_k, dim=-1)
    score = values[:, 0]
    backpointers: list[Tensor] = []
    for head in range(1, heads):
        previous_ids = ids[:, head - 1]
        current_ids = ids[:, head]
        transitions = transition_log_probs[
            previous_ids.unsqueeze(-1), current_ids.unsqueeze(-2)
        ]
        candidates = (
            score.unsqueeze(-1)
            + transition_weight * transitions
            + values[:, head].unsqueeze(-2)
        )
        score, back = candidates.max(dim=1)
        backpointers.append(back)
    cursor = score.argmax(dim=-1)
    selected = [cursor]
    for back in reversed(backpointers):
        cursor = back.gather(1, cursor.unsqueeze(1)).squeeze(1)
        selected.append(cursor)
    selected.reverse()
    positions = torch.stack(selected, dim=1)
    return ids.gather(2, positions.unsqueeze(-1)).squeeze(-1)


def accepted_prefix_lengths(predictions: Tensor, targets: Tensor) -> Tensor:
    if predictions.shape != targets.shape or predictions.ndim != 2:
        raise ValueError("predictions and targets must share [batch, heads]")
    matches = predictions.eq(targets)
    return torch.cumprod(matches.to(torch.int64), dim=1).sum(dim=1)


def _metrics(predictions: Tensor, targets: Tensor, topk_recall: Tensor) -> dict[str, Any]:
    prefix = accepted_prefix_lengths(predictions, targets).float()
    return {
        "rows": int(targets.shape[0]),
        "head_accuracy": [
            float(predictions[:, index].eq(targets[:, index]).float().mean())
            for index in range(targets.shape[1])
        ],
        "full_path_exact": float(predictions.eq(targets).all(dim=1).float().mean()),
        "accepted_prefix_mean": float(prefix.mean()),
        "accepted_prefix_distribution": {
            str(length): float(prefix.eq(length).float().mean())
            for length in range(targets.shape[1] + 1)
        },
        "topk_recall_by_head": [
            float(topk_recall[:, index].float().mean())
            for index in range(targets.shape[1])
        ],
    }


def _evaluate(
    logits: Tensor,
    targets: Tensor,
    transition: Tensor,
    *,
    top_k: int,
    transition_weight: float,
) -> dict[str, Any]:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    predictions = viterbi_decode(
        log_probs,
        transition,
        top_k=top_k,
        transition_weight=transition_weight,
    )
    top_ids = torch.topk(log_probs, top_k, dim=-1).indices
    recall = top_ids.eq(targets.unsqueeze(-1)).any(dim=-1)
    return _metrics(predictions, targets, recall)


def _record_partition(
    record_ids: Tensor,
    validation_index: Tensor,
) -> tuple[Tensor, Tensor]:
    records = torch.unique(record_ids[validation_index], sorted=True)
    if records.numel() < 2:
        raise ValueError("Viterbi evaluation needs at least two validation records")
    selection_records = records[::2]
    holdout_records = records[1::2]
    selection = validation_index[torch.isin(record_ids[validation_index], selection_records)]
    holdout = validation_index[torch.isin(record_ids[validation_index], holdout_records)]
    return selection, holdout


def _transition_matrix(
    targets: Tensor,
    training_index: Tensor,
    *,
    vocab_size: int,
    smoothing: float,
) -> Tensor:
    counts = torch.full((vocab_size, vocab_size), smoothing, dtype=torch.float64)
    rows = targets[training_index]
    for head in range(rows.shape[1] - 1):
        flat = rows[:, head] * vocab_size + rows[:, head + 1]
        counts.view(-1).scatter_add_(
            0, flat, torch.ones_like(flat, dtype=counts.dtype)
        )
    return torch.log(counts / counts.sum(dim=1, keepdim=True)).float()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    if args.smoothing <= 0 or args.batch_size < 1:
        parser.error("smoothing and batch size must be positive")

    checkpoint_sha = sha256_file(args.gpt.resolve())
    adapter = load_adapter(
        args.adapter.resolve(),
        expected_base_gpt_sha256=checkpoint_sha,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ).eval()
    with safe_open(str(args.dataset.resolve()), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        hidden = handle.get_tensor("hidden").float()
        targets = handle.get_tensor("future_targets").long()
        record_ids = handle.get_tensor("record_ids").long()
    if metadata.get("base_gpt_sha256") != checkpoint_sha:
        raise ValueError("MTP dataset does not match the GPT checkpoint")
    if targets.shape[1] != adapter.spec.future_head_count:
        raise ValueError("MTP adapter head count does not match the dataset")

    generator = torch.Generator().manual_seed(args.seed)
    training_index, validation_index = _split_indices(
        record_ids, args.validation_fraction, generator
    )
    selection_index, holdout_index = _record_partition(record_ids, validation_index)
    transition = _transition_matrix(
        targets,
        training_index,
        vocab_size=adapter.spec.vocab_size,
        smoothing=args.smoothing,
    )
    device = next(adapter.parameters()).device
    logits_parts: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, hidden.shape[0], args.batch_size):
            logits_parts.append(
                adapter(hidden[start : start + args.batch_size].to(device)).cpu()
            )
    logits = torch.cat(logits_parts, dim=0)

    configurations: list[dict[str, Any]] = []
    for top_k in (5, 10, 15):
        for transition_weight in (0.0, 0.1, 0.25, 0.5, 1.0):
            selection = _evaluate(
                logits[selection_index],
                targets[selection_index],
                transition,
                top_k=top_k,
                transition_weight=transition_weight,
            )
            configurations.append(
                {
                    "top_k": top_k,
                    "transition_weight": transition_weight,
                    "selection": selection,
                }
            )
    selected = max(
        configurations,
        key=lambda row: (
            row["selection"]["accepted_prefix_mean"],
            row["selection"]["full_path_exact"],
        ),
    )
    holdout = _evaluate(
        logits[holdout_index],
        targets[holdout_index],
        transition,
        top_k=int(selected["top_k"]),
        transition_weight=float(selected["transition_weight"]),
    )
    independent = _evaluate(
        logits[holdout_index],
        targets[holdout_index],
        transition,
        top_k=1,
        transition_weight=0.0,
    )
    result = {
        "schema": 1,
        "kind": "aniflive-tts-v1.2-mtp4-viterbi-upper-bound",
        "scope": "offline-semantic-proxy",
        "base_gpt_sha256": checkpoint_sha,
        "dataset_sha256": sha256_file(args.dataset.resolve()),
        "adapter_sha256": sha256_file(args.adapter.resolve()),
        "training_rows": int(training_index.numel()),
        "selection_rows": int(selection_index.numel()),
        "holdout_rows": int(holdout_index.numel()),
        "smoothing": args.smoothing,
        "selected": {
            "top_k": selected["top_k"],
            "transition_weight": selected["transition_weight"],
            "selection": selected["selection"],
            "holdout": holdout,
        },
        "independent_argmax_holdout": independent,
        "configurations": configurations,
    }
    target = args.report.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["selected"], indent=2))
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
