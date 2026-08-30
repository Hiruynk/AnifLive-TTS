from __future__ import annotations

from enum import Enum

import numpy as np


class TransitionCurve(str, Enum):
    HARD_NATURAL = "hard-natural"
    HANN = "hann"
    EQUAL_POWER = "equal-power"
    SIGMOID = "sigmoid"


def transition_envelope(
    samples: int,
    *,
    curve: TransitionCurve,
    fade_in: bool,
    sigmoid_k: float = 12.0,
) -> np.ndarray:
    if samples < 1:
        return np.empty(0, dtype=np.float32)
    if curve is TransitionCurve.HARD_NATURAL:
        return np.ones(samples, dtype=np.float32)
    position = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    if curve is TransitionCurve.HANN:
        envelope = 0.5 - 0.5 * np.cos(np.pi * position)
    elif curve is TransitionCurve.EQUAL_POWER:
        envelope = np.sin(0.5 * np.pi * position)
    elif curve is TransitionCurve.SIGMOID:
        steepness = max(1.0, float(sigmoid_k))
        raw = 1.0 / (1.0 + np.exp(-steepness * (position - 0.5)))
        envelope = (raw - raw[0]) / max(float(raw[-1] - raw[0]), 1e-12)
    else:  # pragma: no cover - exhaustive Enum protection
        raise ValueError(f"Unsupported transition curve: {curve}")
    if not fade_in:
        envelope = envelope[::-1]
    return envelope.astype(np.float32, copy=False)


def apply_edge_fade(
    audio: np.ndarray,
    *,
    samples: int,
    curve: TransitionCurve,
    fade_in: bool,
    sigmoid_k: float = 12.0,
) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float32)
    count = min(max(0, int(samples)), int(source.size))
    if count == 0 or curve is TransitionCurve.HARD_NATURAL:
        return source
    result = source.copy()
    envelope = transition_envelope(
        count,
        curve=curve,
        fade_in=fade_in,
        sigmoid_k=sigmoid_k,
    )
    if fade_in:
        result[:count] *= envelope
    else:
        result[-count:] *= envelope
    return result
