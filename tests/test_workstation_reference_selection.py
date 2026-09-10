from __future__ import annotations

import numpy as np
import pytest

from aniflive_tts import workstation_reference_selection as reference


def _record(item_id: str, *, split: str = "train", route: str = "clean") -> dict:
    return {
        "source_item_id": item_id,
        "path": f"{split}/wav/{item_id}.wav",
        "sha256": item_id.ljust(64, "0")[:64],
        "split": split,
        "duration_seconds": 5.5,
        "transcript": "今日はいい天気ですね。",
        "language": "ja",
        "quality": {"quality_score": 95.0, "silence_ratio": 0.1, "clipped_ratio": 0.0},
        "verification_hashes": {"transcript": "a" * 64, "speaker": "b" * 64},
        "acquisition_route": route,
    }


def test_reference_candidates_are_train_only_and_transcript_verified() -> None:
    records = [_record("train-a"), _record("train-b"), _record("test", split="test")]
    unverified = _record("unverified")
    unverified["verification_hashes"].pop("transcript")
    records.append(unverified)
    report = reference.rank_reference_candidates(
        records,
        {
            "train-a": np.array([1.0, 0.0]),
            "train-b": np.array([0.98, 0.02]),
            "test": np.array([1.0, 0.0]),
            "unverified": np.array([1.0, 0.0]),
        },
    )
    assert {row["item_id"] for row in report["top_candidates"]} == {
        "train-a",
        "train-b",
    }
    reasons = {row["item_id"]: row["reason"] for row in report["rejected"]}
    assert "not-train" in reasons["test"]
    assert "transcript-unverified" in reasons["unverified"]


def test_reference_centroid_selection_is_deterministic() -> None:
    records = [_record("a"), _record("b"), _record("c")]
    embeddings = {
        "a": np.array([1.0, 0.0, 0.0]),
        "b": np.array([0.99, 0.01, 0.0]),
        "c": np.array([-1.0, 0.0, 0.0]),
    }
    first = reference.rank_reference_candidates(records, embeddings)
    second = reference.rank_reference_candidates(list(reversed(records)), embeddings)
    assert first["automatic_recommendation"] == second["automatic_recommendation"]
    assert first["top_candidates"] == second["top_candidates"]


def test_salvaged_audio_is_not_a_deployment_reference() -> None:
    with pytest.raises(reference.ReferenceSelectionError, match="fewer than two"):
        reference.rank_reference_candidates(
            [_record("clean"), _record("salvaged", route="salvaged")],
            {
                "clean": np.array([1.0, 0.0]),
                "salvaged": np.array([1.0, 0.0]),
            },
        )
