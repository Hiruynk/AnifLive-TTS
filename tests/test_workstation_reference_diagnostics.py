from __future__ import annotations

import pytest

from aniflive_tts.workstation_reference_diagnostics import (
    ReferenceDiagnosticError,
    build_reference_rejection_diagnostic,
)


SHA = "a" * 64


def _case(label: str, item_id: str, speaker: float = 0.78) -> dict[str, object]:
    return {
        "label": label,
        "candidate_item_id": item_id,
        "content_error": 0.0,
        "speaker_centroid_cosine": speaker,
        "invalid": False,
        "nan": False,
        "empty": False,
        "clipped": False,
        "duration_outlier": False,
    }


def test_all_poor_is_authoritative_even_when_automatic_metrics_pass() -> None:
    cases = [_case(label, f"item-{label}") for label in "ABCDE" for _ in range(5)]
    report = build_reference_rejection_diagnostic(
        cases,
        source_speaker_scores={f"item-{label}": 0.81 for label in "ABCDE"},
        human_evidence={
            "status": "all-poor",
            "human_decision": "all-poor",
            "reference_selection_report_sha256": SHA,
        },
        reference_selection_report_sha256=SHA,
        blind_manifest_sha256="b" * 64,
    )

    assert report["status"] == "failed"
    assert report["human_gate"]["authoritative"] is True
    assert report["deployment_allowed"] is False
    assert report["test_split_accessed"] is False
    assert report["diagnosis"] == "objective-metric-human-perceptual-disagreement"


def test_reference_diagnostic_refuses_test_access() -> None:
    with pytest.raises(ReferenceDiagnosticError, match="must not access test data"):
        build_reference_rejection_diagnostic(
            [_case("A", "item-A")],
            source_speaker_scores={"item-A": 0.81},
            human_evidence={
                "status": "all-poor",
                "human_decision": "all-poor",
                "reference_selection_report_sha256": SHA,
            },
            reference_selection_report_sha256=SHA,
            blind_manifest_sha256="b" * 64,
            test_split_accessed=True,
        )


def test_reference_diagnostic_rejects_mismatched_human_evidence() -> None:
    with pytest.raises(ReferenceDiagnosticError, match="does not match"):
        build_reference_rejection_diagnostic(
            [_case("A", "item-A")],
            source_speaker_scores={"item-A": 0.81},
            human_evidence={
                "status": "all-poor",
                "human_decision": "all-poor",
                "reference_selection_report_sha256": "c" * 64,
            },
            reference_selection_report_sha256=SHA,
            blind_manifest_sha256="b" * 64,
        )
