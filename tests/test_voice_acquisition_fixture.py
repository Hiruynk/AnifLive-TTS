from __future__ import annotations

import copy

import pytest

from aniflive_tts.voice_acquisition_fixture import (
    VoiceAcquisitionFixtureError,
    evaluate_voice_acquisition_fixture,
)


SHA = "a" * 64


def _fixture() -> dict:
    return {
        "schema": "aniflive-voice-acquisition-fixture-v1",
        "sample_rate": 16_000,
        "output_sha256": SHA,
        "timeline": [
            {"kind": "target", "start_sample": 0, "end_sample": 16_000},
            {"kind": "overlap", "start_sample": 16_000, "end_sample": 32_000},
            {
                "kind": "non_target",
                "start_sample": 32_000,
                "end_sample": 48_000,
            },
        ],
    }


def _record(identifier: str, position: int, route: str, selected: str = "original") -> dict:
    start = position * 16_000
    return {
        "id": identifier,
        "route": route,
        "source_start_sample": start,
        "source_end_sample": start + 16_000,
        "audio": {"selected": selected},
    }


def test_fixture_evaluation_accepts_safe_review_and_salvage_routes() -> None:
    records = [
        _record("target", 0, "clean"),
        _record("overlap", 1, "salvage"),
        _record("other", 2, "reject"),
    ]
    routing = {"source_sha256": SHA, "records": records}
    finalized = {"records": copy.deepcopy(records)}

    report = evaluate_voice_acquisition_fixture(_fixture(), routing, finalized)

    assert report["passed"] is True
    assert all(report["gates"].values())
    assert report["routing_matrix"]["target"]["clean"] == 1
    assert report["routing_matrix"]["overlap"]["salvage"] == 1
    assert report["routing_matrix"]["non_target"]["reject"] == 1
    assert report["metrics"]["overlap_containment_recall"] == 1.0


def test_fixture_evaluation_exposes_unsafe_automatic_decisions() -> None:
    routing_records = [
        _record("target", 0, "reject"),
        _record("overlap", 1, "clean"),
        _record("other", 2, "clean"),
    ]
    final_records = copy.deepcopy(routing_records)
    final_records[0]["audio"]["selected"] = "processed"

    report = evaluate_voice_acquisition_fixture(
        _fixture(),
        {"source_sha256": SHA, "records": routing_records},
        {"records": final_records},
    )

    assert report["passed"] is False
    assert report["gates"]["overlap_containment"] is False
    assert report["gates"]["contamination_auto_clean"] is False
    assert report["gates"]["target_not_auto_rejected"] is False
    assert report["gates"]["non_target_not_auto_accepted"] is False
    assert report["gates"]["clean_target_not_replaced"] is False


def test_fixture_evaluation_rejects_unmatched_source() -> None:
    record = _record("target", 0, "clean")
    with pytest.raises(VoiceAcquisitionFixtureError, match="does not match"):
        evaluate_voice_acquisition_fixture(
            _fixture(),
            {"source_sha256": "b" * 64, "records": [record]},
            {"records": [record]},
        )


def test_fixture_evaluation_quarantines_mixed_truth_boundaries() -> None:
    fixture = _fixture()
    mixed = {
        "id": "mixed",
        "route": "review",
        "source_start_sample": 8_000,
        "source_end_sample": 24_000,
        "audio": {"selected": "original"},
    }

    report = evaluate_voice_acquisition_fixture(
        fixture,
        {"source_sha256": SHA, "records": [mixed]},
        {"records": [copy.deepcopy(mixed)]},
    )

    assert report["passed"] is True
    assert report["truth"]["mixed"]["kind"] == "mixed"
    assert report["routing_matrix"]["mixed"]["review"] == 1
    assert report["metrics"]["mixed_truth_count"] == 1
