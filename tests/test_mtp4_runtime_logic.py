from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from research.v12_semantic.mtp_semantic_runtime_experiment import (
    MTP4SpeculativeSemanticRuntime,
    _accepted_prefix_length,
    _filtered_topk_distribution,
    _mtp4_cycle_plan,
)


@pytest.mark.parametrize(
    ("verified", "drafts", "expected"),
    [
        ([1, 2, 3, 4], [1, 2, 3], 3),
        ([9, 2, 3, 4], [1, 2, 3], 0),
        ([1, 9, 3, 4], [1, 2, 3], 1),
        ([1, 2, 9, 4], [1, 2, 3], 2),
    ],
)
def test_accepted_prefix_length(
    verified: list[int], drafts: list[int], expected: int
) -> None:
    assert _accepted_prefix_length(verified, drafts, limit=3) == expected


@pytest.mark.parametrize(
    ("accepted_prefix", "expected"),
    [
        (0, (1, 18, 19)),
        (1, (2, 19, 20)),
        (2, (3, 20, 21)),
        (3, (3, None, 21)),
    ],
)
def test_mtp4_cycle_plan(
    accepted_prefix: int, expected: tuple[int, int | None, int]
) -> None:
    assert _mtp4_cycle_plan(index=17, accepted_prefix=accepted_prefix) == expected


def test_mtp4_cycle_plan_rejects_invalid_prefix() -> None:
    with pytest.raises(ValueError):
        _mtp4_cycle_plan(index=1, accepted_prefix=4)


def test_future_drafts_selects_top_one_per_head() -> None:
    indices = torch.arange(1 * 4 * 3 * 5, dtype=torch.int64).reshape(1, 4, 3, 5)
    drafts = MTP4SpeculativeSemanticRuntime._future_drafts(
        {"mtp_topk_indices": indices}, 2
    )
    assert drafts.shape == (1, 3)
    assert drafts.tolist() == [[30, 35, 40]]


def test_sample_rows_preserves_all_four_rows() -> None:
    runtime = MTP4SpeculativeSemanticRuntime.__new__(
        MTP4SpeculativeSemanticRuntime
    )
    runtime.engine = SimpleNamespace(device=torch.device("cpu"))
    runtime.sample_topk = lambda values, indices, **_: indices[..., :1]
    state = SimpleNamespace(temperature=1.0, top_k=15, top_p=1.0)
    indices = torch.tensor(
        [[[10, 11], [20, 21], [30, 31], [40, 41]]], dtype=torch.int64
    )
    outputs = {
        "base_topk_values": torch.ones((1, 4, 2), dtype=torch.float32),
        "base_topk_indices": indices,
    }
    sampled = runtime._sample_rows(state, outputs)
    assert sampled.shape == (1, 4)
    assert sampled.tolist() == [[10, 20, 30, 40]]


def test_filtered_topk_distribution_matches_requested_support() -> None:
    values = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])
    indices = torch.tensor([[[40, 30, 20, 10]]], dtype=torch.int64)
    probabilities, selected = _filtered_topk_distribution(
        values,
        indices,
        temperature=1.0,
        top_k=3,
        top_p=1.0,
    )
    assert selected.tolist() == [[[40, 30, 20]]]
    assert probabilities.shape == (1, 1, 3)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones((1, 1)))


def test_filtered_topk_distribution_keeps_at_least_one_nucleus_token() -> None:
    values = torch.tensor([[10.0, 0.0, -1.0]])
    indices = torch.tensor([[3, 2, 1]], dtype=torch.int64)
    probabilities, _ = _filtered_topk_distribution(
        values,
        indices,
        temperature=1.0,
        top_k=3,
        top_p=0.0,
    )
    assert probabilities[0, 0] == 1.0
    assert probabilities[0, 1:].count_nonzero() == 0
