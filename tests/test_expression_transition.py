from __future__ import annotations

import numpy as np
import pytest

from aniflive_tts.expression_transition import (
    TransitionCurve,
    apply_edge_fade,
    transition_envelope,
)


@pytest.mark.parametrize(
    "curve",
    [TransitionCurve.HANN, TransitionCurve.EQUAL_POWER, TransitionCurve.SIGMOID],
)
def test_transition_envelopes_have_stable_endpoints(curve: TransitionCurve) -> None:
    fade_in = transition_envelope(65, curve=curve, fade_in=True)
    fade_out = transition_envelope(65, curve=curve, fade_in=False)
    assert fade_in[0] == pytest.approx(0.0, abs=1e-6)
    assert fade_in[-1] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(fade_in) >= 0.0)
    assert np.array_equal(fade_out, fade_in[::-1])


def test_hard_natural_is_bit_exact_noop() -> None:
    source = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    result = apply_edge_fade(
        source,
        samples=8,
        curve=TransitionCurve.HARD_NATURAL,
        fade_in=True,
    )
    assert result is source
    assert np.array_equal(result, source)


def test_edge_fade_does_not_change_length_or_input() -> None:
    source = np.ones(16, dtype=np.float32)
    result = apply_edge_fade(
        source,
        samples=4,
        curve=TransitionCurve.HANN,
        fade_in=False,
    )
    assert result.shape == source.shape
    assert np.array_equal(source, np.ones(16, dtype=np.float32))
    assert result[-1] == pytest.approx(0.0)
    assert result[-4] == pytest.approx(1.0)
