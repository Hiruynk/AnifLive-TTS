from __future__ import annotations

import numpy as np
import pytest

from aniflive_tts.dataset_acquisition import (
    DatasetAcquisitionConfig,
    DatasetAcquisitionError,
    SpeakerPurityEvidence,
    post_separation_quality_gate,
    route_speaker_purity,
    stationary_overlap_probe,
)


def test_dataset_language_is_an_optional_generic_acquisition_contract() -> None:
    automatic = DatasetAcquisitionConfig.from_mapping({"acquisition_mode": "standard"})
    japanese = DatasetAcquisitionConfig.from_mapping(
        {"acquisition_mode": "standard", "declared_language": "JA"}
    )

    assert automatic.declared_language is None
    assert japanese.declared_language == "ja"
    assert japanese.as_dict()["declared_language"] == "ja"


def test_dataset_language_rejects_unsupported_values() -> None:
    with pytest.raises(DatasetAcquisitionError, match="unsupported"):
        DatasetAcquisitionConfig.from_mapping(
            {"acquisition_mode": "standard", "declared_language": "fr"}
        )


def test_borderline_target_identity_is_reviewed_not_separated() -> None:
    route = route_speaker_purity(
        SpeakerPurityEvidence(
            target_similarity=0.69,
            winner_margin=1.0,
            embedding_variance=0.0,
        ),
        target_threshold=0.72,
        review_margin=0.08,
    )

    assert route.route == "review"
    assert "speaker-similarity-needs-review" in route.reasons


def test_phonetic_embedding_variance_alone_does_not_trigger_separation() -> None:
    route = route_speaker_purity(
        SpeakerPurityEvidence(
            target_similarity=0.91,
            winner_margin=1.0,
            embedding_variance=0.8,
        )
    )

    assert route.route == "clean"
    assert route.reasons == ()


def test_diarized_speaker_change_routes_target_clip_to_salvage() -> None:
    route = route_speaker_purity(
        SpeakerPurityEvidence(
            target_similarity=0.91,
            winner_margin=1.0,
            speaker_change_evidence=True,
        )
    )

    assert route.route == "salvage"
    assert route.reasons == ("multiple-speakers",)


def test_diarization_uncertainty_preserves_raw_clip_for_review() -> None:
    route = route_speaker_purity(
        SpeakerPurityEvidence(
            target_similarity=0.91,
            winner_margin=1.0,
            diarization_uncertain_evidence=True,
        )
    )

    assert route.route == "review"
    assert route.reasons == ("diarization-needs-review",)


def test_low_snr_target_clip_is_preserved_for_review() -> None:
    route = route_speaker_purity(
        SpeakerPurityEvidence(
            target_similarity=0.91,
            winner_margin=1.0,
            estimated_snr_db=19.9,
        )
    )

    assert route.route == "review"
    assert route.reasons == ("low-snr",)


def test_clean_snr_threshold_is_explicitly_configurable() -> None:
    route = route_speaker_purity(
        SpeakerPurityEvidence(
            target_similarity=0.91,
            winner_margin=1.0,
            estimated_snr_db=19.9,
        ),
        minimum_clean_snr_db=18.0,
    )

    assert route.route == "clean"


def test_diarized_speaker_fusion_distinguishes_contamination() -> None:
    from aniflive_tts.dataset_acquisition import fuse_diarized_speaker_scores

    contaminated = fuse_diarized_speaker_scores(
        {"speaker_1": 0.46, "speaker_2": 0.87}
    )
    split_target = fuse_diarized_speaker_scores(
        {"speaker_1": 0.84, "speaker_2": 0.88}
    )

    assert contaminated["distinct_speaker_evidence"] is True
    assert contaminated["decision"] == "distinct-speaker"
    assert split_target["distinct_speaker_evidence"] is False
    assert split_target["uncertain_multi_speaker"] is True
    assert split_target["decision"] == "review"


def test_failed_processed_audio_returns_raw_clip_to_review() -> None:
    route = post_separation_quality_gate(
        target_similarity=0.68,
        loser_similarity=0.66,
        similarity_before=0.76,
        asr_consistent=False,
        finite_audio=True,
        exact_sample_length=True,
        artifact_score=0.2,
        target_threshold=0.72,
        ambiguity_margin=0.03,
    )

    assert route.route == "review"
    assert "speaker-similarity-regression" in route.reasons
    assert "asr-content-regression" in route.reasons


def test_multi_window_stationary_probe_requires_consistent_evidence() -> None:
    sample_rate = 16_000
    mixed = np.concatenate(
        (
            np.full(sample_rate * 2, 0.25, dtype=np.float32),
            np.full(sample_rate * 2, -0.25, dtype=np.float32),
        )
    )

    def embedder(audio: np.ndarray, _sample_rate: int) -> np.ndarray:
        positive_ratio = float(np.mean(audio >= 0))
        return np.asarray([positive_ratio, 1.0 - positive_ratio], dtype=np.float32)

    result = stationary_overlap_probe(
        mixed,
        sample_rate,
        embedder=embedder,
        reference_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
    )

    assert result["window_seconds"] == [0.75, 1.0, 1.5]
    assert result["positive_votes"] >= result["required_votes"]
    assert result["overlap_evidence"] is True
    assert result["decision"] == "separator-probe"
