from __future__ import annotations

import numpy as np
import pytest

from aniflive_tts.tse_pipeline import (
    SpeakerVerificationConfig,
    VoiceActivityConfig,
    apply_review_decisions,
    detect_voice_activity,
    extract_target_audio,
    verify_target_speaker,
)


RATE = 16_000


def _tone(seconds: float, frequency: float, amplitude: float = 0.25) -> np.ndarray:
    count = int(round(seconds * RATE))
    time = np.arange(count, dtype=np.float32) / RATE
    return amplitude * np.sin(2.0 * np.pi * frequency * time)


def _embedding(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    assert sample_rate == RATE
    spectrum = np.abs(np.fft.rfft(audio, n=4096))
    frequencies = np.fft.rfftfreq(4096, 1.0 / sample_rate)
    return np.array(
        [
            spectrum[(frequencies >= 180) & (frequencies < 300)].sum(),
            spectrum[(frequencies >= 300) & (frequencies < 520)].sum(),
            spectrum[(frequencies >= 520) & (frequencies < 900)].sum(),
        ],
        dtype=np.float32,
    )


def test_detect_verify_review_and_extract_target_regions() -> None:
    silence = np.zeros(int(0.24 * RATE), dtype=np.float32)
    source = np.concatenate(
        [silence, _tone(0.42, 220), silence, _tone(0.40, 680), silence]
    )
    segments = detect_voice_activity(
        source,
        RATE,
        config=VoiceActivityConfig(context_ms=0, maximum_gap_ms=60),
    )
    assert len(segments) == 2

    verified = verify_target_speaker(
        source,
        RATE,
        segments,
        reference_audio=_tone(0.5, 220),
        embedder=_embedding,
        config=SpeakerVerificationConfig(target_threshold=0.90, review_margin=0.08),
    )
    assert [item.decision for item in verified] == ["target", "rejected"]
    assert verified[0].similarity == pytest.approx(1.0, abs=0.01)

    result = extract_target_audio(source, RATE, verified)
    assert result.audio.size == verified[0].sample_count
    assert result.target_seconds == pytest.approx(verified[0].duration_seconds(RATE))
    assert result.review_seconds == 0.0


def test_overlap_is_never_automatically_accepted() -> None:
    source = np.concatenate([np.zeros(1600, dtype=np.float32), _tone(0.4, 220)])
    segments = detect_voice_activity(source, RATE, config=VoiceActivityConfig(context_ms=0))
    verified = verify_target_speaker(
        source,
        RATE,
        segments,
        reference_audio=_tone(0.4, 220),
        embedder=_embedding,
        overlap_detector=lambda _audio, _rate: True,
    )
    assert verified[0].decision == "review"
    assert verified[0].overlap is True
    assert extract_target_audio(source, RATE, verified).audio.size == 0
    assert extract_target_audio(source, RATE, verified, include_review=True).audio.size > 0


def test_human_review_override_is_explicit_and_validated() -> None:
    source = _tone(0.3, 220)
    segments = detect_voice_activity(source, RATE, config=VoiceActivityConfig(context_ms=0))
    verified = verify_target_speaker(
        source,
        RATE,
        segments,
        reference_audio=source,
        embedder=_embedding,
        overlap_detector=lambda _audio, _rate: True,
    )
    accepted = apply_review_decisions(verified, {0: "target"})
    assert accepted[0].decision == "target"
    with pytest.raises(ValueError, match="unsupported TSE decision"):
        apply_review_decisions(verified, {0: "accepted-without-review"})


def test_invalid_audio_and_embedding_fail_closed() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        detect_voice_activity(np.array([], dtype=np.float32), RATE)
    segment = detect_voice_activity(_tone(0.3, 220), RATE)[0]
    with pytest.raises(ValueError, match="zero magnitude"):
        verify_target_speaker(
            _tone(0.3, 220),
            RATE,
            (segment,),
            reference_audio=_tone(0.3, 220),
            embedder=lambda _audio, _rate: np.zeros(3, dtype=np.float32),
        )
