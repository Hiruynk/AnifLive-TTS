from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .workstation_production_gates import speaker_identity_gate


REFERENCE_REJECTION_DIAGNOSTIC_SCHEMA = (
    "aniflive-tts-reference-rejection-diagnostic-v1"
)


class ReferenceDiagnosticError(RuntimeError):
    pass


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ReferenceDiagnosticError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReferenceDiagnosticError(f"{label} must be finite")
    return result


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ReferenceDiagnosticError("diagnostic metric collection is empty")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def build_reference_rejection_diagnostic(
    cases: Sequence[Mapping[str, Any]],
    *,
    source_speaker_scores: Mapping[str, float],
    human_evidence: Mapping[str, Any],
    reference_selection_report_sha256: str,
    blind_manifest_sha256: str,
    test_split_accessed: bool = False,
) -> dict[str, Any]:
    """Summarize an all-poor reference sweep without opening test data."""

    if human_evidence.get("status") != "all-poor" or human_evidence.get(
        "human_decision"
    ) != "all-poor":
        raise ReferenceDiagnosticError("diagnostic requires all-poor human evidence")
    evidence_report_sha = human_evidence.get("reference_selection_report_sha256")
    if evidence_report_sha != reference_selection_report_sha256:
        raise ReferenceDiagnosticError("human evidence does not match reference report")
    for value, label in (
        (reference_selection_report_sha256, "reference report SHA256"),
        (blind_manifest_sha256, "blind manifest SHA256"),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ReferenceDiagnosticError(f"{label} is invalid")
    if test_split_accessed:
        raise ReferenceDiagnosticError("reference diagnosis must not access test data")
    if not cases:
        raise ReferenceDiagnosticError("reference diagnosis requires generated cases")

    labels = sorted({str(case.get("label") or "") for case in cases})
    if any(not label for label in labels):
        raise ReferenceDiagnosticError("reference case label is missing")
    summaries: list[dict[str, Any]] = []
    for label in labels:
        rows = [case for case in cases if case.get("label") == label]
        if not rows:
            raise ReferenceDiagnosticError(f"reference label {label} has no cases")
        candidate_id = str(rows[0].get("candidate_item_id") or "")
        if not candidate_id or any(
            row.get("candidate_item_id") != candidate_id for row in rows
        ):
            raise ReferenceDiagnosticError(
                f"reference label {label} candidate lineage is inconsistent"
            )
        if candidate_id not in source_speaker_scores:
            raise ReferenceDiagnosticError(
                f"reference label {label} source speaker score is missing"
            )
        content_errors = [
            _finite(row.get("content_error"), f"{label} content error") for row in rows
        ]
        speaker_scores = [
            _finite(
                row.get("speaker_centroid_cosine"),
                f"{label} speaker centroid cosine",
            )
            for row in rows
        ]
        source_score = _finite(
            source_speaker_scores[candidate_id], f"{label} source speaker score"
        )
        audio_failures = {
            "invalid": sum(int(bool(row.get("invalid"))) for row in rows),
            "nan": sum(int(bool(row.get("nan"))) for row in rows),
            "empty": sum(int(bool(row.get("empty"))) for row in rows),
            "clipped": sum(int(bool(row.get("clipped"))) for row in rows),
            "duration_outliers": sum(
                int(bool(row.get("duration_outlier"))) for row in rows
            ),
        }
        speaker = {
            "centroid_cosine_median": _percentile(speaker_scores, 50),
            "centroid_cosine_p10": _percentile(speaker_scores, 10),
            "source_centroid_cosine_median": source_score,
            "source_centroid_cosine_p10": source_score,
            "source_human_verified": True,
            "source_case_count": 1,
        }
        gate = speaker_identity_gate(speaker)
        automatic_failures: list[str] = []
        if _percentile(content_errors, 50) > 0.08:
            automatic_failures.append("median-content-error")
        if _percentile(content_errors, 95) > 0.18:
            automatic_failures.append("p95-content-error")
        if not gate["passed"]:
            automatic_failures.append("speaker-identity")
        automatic_failures.extend(
            key.replace("_", "-")
            for key, count in audio_failures.items()
            if count
        )
        summaries.append(
            {
                "label": label,
                "candidate_item_id": candidate_id,
                "case_count": len(rows),
                "content": {
                    "median_error": _percentile(content_errors, 50),
                    "p95_error": _percentile(content_errors, 95),
                },
                "speaker": speaker,
                "speaker_identity_gate": gate,
                "audio": audio_failures,
                "automatic_status": (
                    "passed" if not automatic_failures else "failed"
                ),
                "automatic_failures": automatic_failures,
            }
        )

    automatic_pass_count = sum(
        summary["automatic_status"] == "passed" for summary in summaries
    )
    if automatic_pass_count == 0:
        diagnosis = "shared-objective-quality-failure"
    elif automatic_pass_count == len(summaries):
        diagnosis = "objective-metric-human-perceptual-disagreement"
    else:
        diagnosis = "mixed-reference-and-checkpoint-quality"

    return {
        "schema": REFERENCE_REJECTION_DIAGNOSTIC_SCHEMA,
        "status": "failed",
        "human_gate": {
            "decision": "all-poor",
            "authoritative": True,
        },
        "test_split_accessed": False,
        "evidence": {
            "reference_selection_report_sha256": (
                reference_selection_report_sha256
            ),
            "blind_manifest_sha256": blind_manifest_sha256,
        },
        "automatic_summary": {
            "labels_evaluated": len(summaries),
            "labels_passed": automatic_pass_count,
            "labels_failed": len(summaries) - automatic_pass_count,
        },
        "diagnosis": diagnosis,
        "next_action": "training-or-data-remediation",
        "deployment_allowed": False,
        "labels": summaries,
        "cases": [dict(case) for case in cases],
    }


__all__ = [
    "REFERENCE_REJECTION_DIAGNOSTIC_SCHEMA",
    "ReferenceDiagnosticError",
    "build_reference_rejection_diagnostic",
]
