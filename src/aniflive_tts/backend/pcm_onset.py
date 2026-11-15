"""Conservative, explicit initial PCM transport trimming; no neural operations."""
from __future__ import annotations

import math
import numpy as np


def trim_initial_pcm(pcm: bytes, original_total_samples: int, maximum_cut_ms: int = 60) -> tuple[bytes, dict]:
    if (not pcm or len(pcm) % 2 or isinstance(original_total_samples, bool)
            or not isinstance(original_total_samples, int) or original_total_samples < len(pcm) // 2):
        raise ValueError("Initial trim needs aligned PCM and the exact original utterance sample count")
    if type(maximum_cut_ms) is not int or not 0 <= maximum_cut_ms <= 60:
        raise ValueError("Initial PCM trim cap must be an integer in 0..60 ms")
    frame_samples = 480
    samples = np.frombuffer(pcm, dtype="<i2")
    frame_count = samples.size // frame_samples
    frames = samples[:frame_count * frame_samples].astype(np.float64).reshape(-1, frame_samples) / 32768.0
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    peak = np.max(np.abs(frames), axis=1)
    active = np.flatnonzero((rms > 10 ** (-65 / 20)) | (peak > 10 ** (-55 / 20)))
    if not active.size:
        raise ValueError("Canonical first chunk has no low-energy onset to bound trimming")
    onset_frame = int(active[0])
    cap = min(max(0, onset_frame - 4) * frame_samples, maximum_cut_ms * 48, original_total_samples // 40)
    cut = cap // 48 * 48
    # Never discard even a transient sample above the separately declared
    # quiet-prefix limit. Back off; do not fade or modify retained samples.
    above_quiet = np.flatnonzero(np.abs(samples[:cut].astype(np.int32)) > 32768 * 10 ** (-65 / 20))
    if above_quiet.size:
        cut = min(cut, int(above_quiet[0]) // 48 * 48)
    while cut and abs(int(samples[cut])) > 32:
        cut -= 48
    output = pcm[cut * 2:]
    prefix = samples[:cut].astype(np.float64) / 32768.0
    prefix_rms = float(np.sqrt(np.mean(prefix * prefix))) if cut else 0.0
    prefix_peak = float(np.max(np.abs(prefix))) if cut else 0.0
    return output, {
        "policy": "conservative-pcm-head-v3-energy", "activity_rms_dbfs": -65,
        "activity_peak_dbfs": -55, "guard_ms": 40, "maximum_cut_ms": maximum_cut_ms,
        "removed_prefix_peak_limit_dbfs": -65,
        "cut_resolution_ms": 1,
        "maximum_utterance_fraction": .025, "removed_samples": cut,
        "removed_ms": cut / 48, "original_total_samples": original_total_samples,
        "duration_bound_scope": "first native model utterance in this stream request",
        "onset_frame": onset_frame, "removed_prefix_rms_dbfs": None if prefix_rms == 0 else 20 * math.log10(prefix_rms),
        "removed_prefix_peak_dbfs": None if prefix_peak == 0 else 20 * math.log10(prefix_peak),
        "preserved_suffix_bit_exact": output == pcm[cut * 2:],
        "reference_or_neural_output_modified": False, "fade_or_gain_applied": False,
    }
