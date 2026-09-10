from __future__ import annotations

import numpy as np
import pytest

from aniflive_tts.workstation_speaker import (
    SpeakerReferencePrototype,
    SpeakerReferenceSegment,
    WorkstationSpeakerError,
    build_speaker_reference_prototype,
)


def test_reference_prototype_uses_top_two_matches() -> None:
    prototype = SpeakerReferencePrototype(
        (
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.8, 0.6], dtype=np.float32),
            np.asarray([0.0, 1.0], dtype=np.float32),
        ),
        (
            SpeakerReferenceSegment(0, 16_000),
            SpeakerReferenceSegment(16_000, 32_000),
            SpeakerReferenceSegment(32_000, 48_000),
        ),
    )

    score = prototype.score_embedding(np.asarray([1.0, 0.0], dtype=np.float32))

    assert score["aggregation"] == "top-k-mean"
    assert score["support_count"] == 2
    assert score["reference_count"] == 3
    assert score["similarity"] == pytest.approx(0.9)
    assert score["maximum_similarity"] == pytest.approx(1.0)


def test_reference_prototype_selects_longest_speech_regions_deterministically() -> None:
    audio = np.linspace(-0.5, 0.5, 80_000, dtype=np.float32)

    def embedder(values: np.ndarray, _rate: int) -> np.ndarray:
        return np.asarray([values.size, float(np.mean(values))], dtype=np.float32)

    prototype = build_speaker_reference_prototype(
        audio,
        16_000,
        embedder=embedder,
        segments=((0, 16_000), (16_000, 48_000), (48_000, 80_000)),
        maximum_references=2,
    )

    assert prototype.reference_count == 2
    assert [(item.start_sample, item.end_sample) for item in prototype.segments] == [
        (16_000, 48_000),
        (48_000, 80_000),
    ]
    assert prototype.as_dict(16_000)["support_count"] == 2


def test_reference_prototype_rejects_out_of_bounds_segments() -> None:
    with pytest.raises(WorkstationSpeakerError, match="escaped"):
        build_speaker_reference_prototype(
            np.ones(16_000, dtype=np.float32),
            16_000,
            embedder=lambda _audio, _rate: np.ones(2, dtype=np.float32),
            segments=((0, 16_001),),
        )
