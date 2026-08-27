from __future__ import annotations

from pathlib import Path

import pytest
import torch

from research.v12_semantic.mtp_adapter import (
    MTPAdapter,
    MTPAdapterSpec,
    future_targets,
    load_adapter,
    prompt_conditioned_targets,
    save_adapter,
)


BASE_SHA = "1" * 64


def test_adapter_shape_and_base_initialization() -> None:
    spec = MTPAdapterSpec(BASE_SHA, hidden_dim=4, vocab_size=7, heads=4)
    adapter = MTPAdapter(spec)
    base = torch.arange(28, dtype=torch.float32).reshape(7, 4)
    adapter.initialize_from_base_head(base)
    hidden = torch.ones(2, 4)
    logits = adapter(hidden)
    assert logits.shape == (2, 3, 7)
    assert torch.equal(adapter.weight[0], base)
    assert torch.equal(adapter.weight[1], base)
    assert torch.equal(adapter.weight[2], base)


def test_future_targets_align_t_plus_two_through_heads() -> None:
    rows, targets = future_targets(torch.tensor([10, 11, 12, 13, 14, 15]), heads=4)
    assert rows.tolist() == [0, 1]
    assert targets.tolist() == [[12, 13, 14], [13, 14, 15]]


def test_prompt_conditioned_targets_include_prompt_final_hidden() -> None:
    targets = prompt_conditioned_targets(
        torch.tensor([10, 11, 12, 13, 14]), heads=4
    )
    assert targets.tolist() == [[11, 12, 13], [12, 13, 14]]


def test_adapter_round_trip_and_checkpoint_gate(tmp_path: Path) -> None:
    spec = MTPAdapterSpec(BASE_SHA, hidden_dim=3, vocab_size=5, heads=2)
    adapter = MTPAdapter(spec)
    with torch.no_grad():
        adapter.weight.fill_(0.25)
    target = tmp_path / "adapter.safetensors"
    save_adapter(adapter, target, metadata={"training_corpus_hash": "abc"})
    loaded = load_adapter(target, expected_base_gpt_sha256=BASE_SHA)
    assert loaded.spec == spec
    assert torch.equal(loaded.weight, adapter.weight)
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        load_adapter(target, expected_base_gpt_sha256="2" * 64)
