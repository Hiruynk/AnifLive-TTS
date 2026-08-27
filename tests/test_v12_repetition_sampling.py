from __future__ import annotations

from types import SimpleNamespace

import torch

from aniflive_tts.backend.semantic_runtime import (
    _apply_repetition_penalty_to_topk,
    _resolve_minimum_semantic_tokens,
    _suppress_eos_in_topk,
    TransformerSemanticRuntime,
)


def test_repetition_penalty_is_noop_without_seen_tokens() -> None:
    values = torch.tensor([[4.0, 3.0, 2.0]], dtype=torch.float32)
    indices = torch.tensor([[7, 8, 9]], dtype=torch.int64)

    actual_values, actual_indices = _apply_repetition_penalty_to_topk(
        torch,
        values,
        indices,
        seen_token_mask=None,
        repetition_penalty=1.35,
    )

    assert torch.equal(actual_values, values)
    assert torch.equal(actual_indices, indices)


def test_repetition_penalty_penalizes_and_resorts_seen_candidates() -> None:
    values = torch.tensor([[4.0, 3.5, -1.0, -3.0]], dtype=torch.float32)
    indices = torch.tensor([[7, 8, 9, 10]], dtype=torch.int64)
    seen = torch.zeros(1025, dtype=torch.bool)
    seen[7] = True
    seen[9] = True

    actual_values, actual_indices = _apply_repetition_penalty_to_topk(
        torch,
        values,
        indices,
        seen_token_mask=seen,
        repetition_penalty=2.0,
    )

    assert torch.equal(actual_indices, torch.tensor([[8, 7, 9, 10]]))
    assert torch.equal(actual_values, torch.tensor([[3.5, 2.0, -2.0, -3.0]]))


def test_repetition_penalty_one_preserves_candidate_order() -> None:
    values = torch.tensor([[1.0, 3.0, 2.0]], dtype=torch.float32)
    indices = torch.tensor([[7, 8, 9]], dtype=torch.int64)
    seen = torch.ones(1025, dtype=torch.bool)

    actual_values, actual_indices = _apply_repetition_penalty_to_topk(
        torch,
        values,
        indices,
        seen_token_mask=seen,
        repetition_penalty=1.0,
    )

    assert torch.equal(actual_values, values)
    assert torch.equal(actual_indices, indices)


def test_seen_token_mask_starts_empty_instead_of_using_reference_prompt() -> None:
    runtime = TransformerSemanticRuntime(
        engine=SimpleNamespace(device=torch.device("cpu")),
        sample_topk=None,
        torch=torch,
        trt=None,
    )

    seen = runtime._prepare_seen_token_mask(1.35)

    assert seen is not None
    assert not bool(seen.any())


def test_eos_suppression_keeps_other_candidates_in_score_order() -> None:
    values = torch.tensor([[5.0, 4.0, 3.0]], dtype=torch.float32)
    indices = torch.tensor([[1024, 7, 8]], dtype=torch.int64)

    actual_values, actual_indices = _suppress_eos_in_topk(
        torch,
        values,
        indices,
    )

    assert torch.equal(actual_indices, torch.tensor([[7, 8, 1024]]))
    assert torch.equal(actual_values[:, :2], torch.tensor([[4.0, 3.0]]))
    assert torch.isneginf(actual_values[0, 2])


def test_minimum_semantic_tokens_follow_upstream_when_penalty_is_enabled(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ANIFLIVE_TTS_MIN_SEMANTIC_TOKENS", raising=False)

    assert _resolve_minimum_semantic_tokens(1.0) == 0
    assert _resolve_minimum_semantic_tokens(1.10) == 11
    assert _resolve_minimum_semantic_tokens(1.35) == 11


def test_minimum_semantic_tokens_can_be_explicitly_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ANIFLIVE_TTS_MIN_SEMANTIC_TOKENS", "17")

    assert _resolve_minimum_semantic_tokens(1.0) == 17
