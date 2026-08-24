from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_audio import analyze_pcm16_stream  # noqa: E402


def _pcm(samples: np.ndarray) -> bytes:
    return samples.astype("<i2", copy=False).tobytes()


def test_audible_ttfa_includes_leading_silence_after_first_packet() -> None:
    silence = np.zeros(4800, dtype=np.int16)
    speech = np.full(640, 12000, dtype=np.int16)
    payload = _pcm(np.concatenate((silence, speech)))
    chunks = [(0.080, payload[:8186]), (0.090, payload[8186:])]

    result = analyze_pcm16_stream(chunks, 32000)

    assert result.ttfp_seconds == pytest.approx(0.080)
    assert result.leading_silence_seconds == pytest.approx(0.150)
    assert result.audible_ttfa_seconds == pytest.approx(0.230)
    assert result.first_active_chunk_index == 1


def test_audible_ttfa_respects_late_active_chunk_arrival() -> None:
    silence = np.zeros(320, dtype=np.int16)
    speech = np.full(320, 12000, dtype=np.int16)
    chunks = [(0.050, _pcm(silence)), (0.200, _pcm(speech))]

    result = analyze_pcm16_stream(chunks, 32000)

    assert result.leading_silence_seconds == pytest.approx(0.010)
    assert result.audible_ttfa_seconds == pytest.approx(0.200)


def test_audible_ttfa_refines_activity_inside_first_frame() -> None:
    silence = np.zeros(224, dtype=np.int16)
    speech = np.full(320, 12000, dtype=np.int16)
    chunks = [(0.083, _pcm(np.concatenate((silence, speech))))]

    result = analyze_pcm16_stream(chunks, 32000)

    assert result.first_active_sample == 224
    assert result.leading_silence_seconds == pytest.approx(0.007)
    assert result.audible_ttfa_seconds == pytest.approx(0.090)


def test_audible_ttfa_rejects_all_silent_stream() -> None:
    with pytest.raises(RuntimeError, match="contains no audio"):
        analyze_pcm16_stream([(0.030, _pcm(np.zeros(640, dtype=np.int16)))], 32000)
