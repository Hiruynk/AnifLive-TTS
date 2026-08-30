from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_validator():
    path = Path(__file__).parents[1] / "scripts" / "validate_expression_matrix.py"
    spec = importlib.util.spec_from_file_location("expression_matrix_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_neutral_uses_v12_regression_basis_without_hiding_speaker_diagnostic() -> None:
    gate = _load_validator()._build_quality_gate(
        controlled_expression=False,
        deterministic=True,
        semantic_tokens_exact=True,
        log_mel_cosine=0.9872,
        speaker_cosine=0.9787,
        duration_difference_ratio=0.0042,
        playback_continuity=True,
    )

    assert gate["passed"] is True
    assert gate["log_mel_diagnostic"] is False
    assert gate["speaker_diagnostic"] is False
    assert gate["speaker_required"] is False
    assert gate["quality_basis"] == "neutral-v1.2-regression-parity"


def test_controlled_expression_keeps_absolute_speaker_gate() -> None:
    gate = _load_validator()._build_quality_gate(
        controlled_expression=True,
        deterministic=True,
        semantic_tokens_exact=True,
        log_mel_cosine=0.9937,
        speaker_cosine=0.9787,
        duration_difference_ratio=0.0042,
        playback_continuity=True,
    )

    assert gate["passed"] is False
    assert gate["speaker"] is False
    assert gate["speaker_required"] is True
    assert gate["quality_basis"] == "controlled-expression-absolute-gates"
