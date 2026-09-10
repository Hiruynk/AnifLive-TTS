from __future__ import annotations

import numpy as np
import pytest

from aniflive_tts.tse_pipeline import SpeakerSegment, extract_target_audio
from aniflive_tts.tse_separation import (
    SeparationBackendError,
    _normalise_model_input,
    select_target_candidate,
)


RATE = 16_000


def test_mossformer_input_normalization_preserves_restorable_gain() -> None:
    time = np.arange(RATE, dtype=np.float32) / RATE
    audio = 0.2 * np.sin(2 * np.pi * 220 * time)
    normalised, restore = _normalise_model_input(audio)
    assert np.sqrt(np.mean(np.square(normalised))) < 0.1
    assert np.allclose(normalised * restore, audio, rtol=1e-4, atol=1e-6)


def test_mossformer_input_normalization_rejects_silence() -> None:
    with pytest.raises(SeparationBackendError, match="no usable speech energy"):
        _normalise_model_input(np.zeros(RATE, dtype=np.float32))


def test_blind_sources_are_selected_by_the_injected_speaker_embedder() -> None:
    target = np.full(RATE, 0.2, dtype=np.float32)
    other = np.full(RATE, -0.2, dtype=np.float32)

    def embed(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        assert sample_rate == RATE
        return np.asarray([1.0, 0.0] if float(np.mean(audio)) > 0 else [0.0, 1.0])

    selected, similarities, accepted = select_target_candidate(
        (other, target),
        sample_rate=RATE,
        reference_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        embedder=embed,
        target_threshold=0.72,
        ambiguity_margin=0.03,
    )
    assert selected == 1
    assert similarities == pytest.approx((0.0, 1.0))
    assert accepted is True


def test_ambiguous_separated_sources_fail_closed() -> None:
    candidates = (np.ones(RATE, dtype=np.float32), np.ones(RATE, dtype=np.float32))
    selected, similarities, accepted = select_target_candidate(
        candidates,
        sample_rate=RATE,
        reference_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        embedder=lambda _audio, _rate: np.asarray([1.0, 0.0], dtype=np.float32),
        target_threshold=0.72,
        ambiguity_margin=0.03,
    )
    assert selected == 0
    assert similarities == pytest.approx((1.0, 1.0))
    assert accepted is False


def test_reference_fusion_scorer_controls_candidate_selection() -> None:
    candidates = (
        np.asarray([0.9, 0.1], dtype=np.float32),
        np.asarray([0.1, 0.9], dtype=np.float32),
    )
    selected, similarities, accepted = select_target_candidate(
        candidates,
        sample_rate=RATE,
        reference_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        embedder=lambda audio, _rate: audio,
        embedding_scorer=lambda embedding: float(embedding[1]),
        target_threshold=0.8,
        ambiguity_margin=0.03,
    )
    assert selected == 1
    assert similarities[1] > similarities[0]
    assert accepted is True


def test_malformed_separation_candidates_are_rejected() -> None:
    with pytest.raises(SeparationBackendError, match="exactly two"):
        select_target_candidate(
            (np.ones(RATE, dtype=np.float32),),
            sample_rate=RATE,
            reference_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            embedder=lambda _audio, _rate: np.asarray([1.0, 0.0], dtype=np.float32),
            target_threshold=0.72,
            ambiguity_margin=0.03,
        )


def test_extraction_uses_validated_separated_audio_for_target_segment() -> None:
    mixture = np.linspace(-0.8, 0.8, RATE, dtype=np.float32)
    separated = np.linspace(0.1, 0.2, RATE, dtype=np.float32)
    segment = SpeakerSegment(
        index=0,
        start_sample=0,
        end_sample=RATE,
        rms_dbfs=-12.0,
        similarity=0.9,
        overlap=True,
        decision="target",
    )
    result = extract_target_audio(
        mixture,
        RATE,
        (segment,),
        replacement_audio={0: separated},
    )
    assert np.array_equal(result.audio, separated)


def test_extraction_rejects_wrong_length_or_unknown_replacement() -> None:
    audio = np.ones(RATE, dtype=np.float32)
    segment = SpeakerSegment(0, 0, RATE, -12.0, decision="target")
    with pytest.raises(ValueError, match="length"):
        extract_target_audio(audio, RATE, (segment,), replacement_audio={0: audio[:-1]})
    with pytest.raises(ValueError, match="unknown"):
        extract_target_audio(audio, RATE, (segment,), replacement_audio={1: audio})
