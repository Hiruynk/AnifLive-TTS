from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from quality_trt_speaker import trim_edge_silence  # noqa: E402


def test_trim_edge_silence_removes_transport_padding() -> None:
    speech = np.linspace(-0.5, 0.5, 3200, dtype=np.float32)
    padded = np.concatenate(
        (np.zeros(800, dtype=np.float32), speech, np.zeros(1200, dtype=np.float32))
    )

    trimmed = trim_edge_silence(padded)

    assert trimmed.size < padded.size
    assert trimmed.size >= 1600
    assert abs(float(trimmed[0])) >= 0.005
    assert abs(float(trimmed[-1])) >= 0.005


def test_trim_edge_silence_keeps_short_or_silent_inputs() -> None:
    silent = np.zeros(3200, dtype=np.float32)
    short = np.concatenate(
        (
            np.zeros(100, dtype=np.float32),
            np.ones(800, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
        )
    )

    assert trim_edge_silence(silent) is silent
    assert trim_edge_silence(short) is short
