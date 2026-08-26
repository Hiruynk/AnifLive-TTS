from __future__ import annotations

from dataclasses import fields

from aniflive_tts.backend.semantic_runtime import SemanticBatch, TransformerSemanticRuntime


def test_semantic_batch_contract_is_stable() -> None:
    assert [field.name for field in fields(SemanticBatch)] == [
        "tokens",
        "eos",
        "proposed_tokens",
        "accepted_tokens",
        "nfe",
    ]


def test_transformer_backend_identity_is_explicit() -> None:
    assert TransformerSemanticRuntime.backend_name == "transformer"
