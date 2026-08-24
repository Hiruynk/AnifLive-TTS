from __future__ import annotations

from dataclasses import dataclass

import numpy as np


AUDIBLE_THRESHOLD_DBFS = -45.0
AUDIBLE_FRAME_MS = 10.0


@dataclass(frozen=True)
class StreamAudibility:
    ttfp_seconds: float
    audible_ttfa_seconds: float
    leading_silence_seconds: float
    first_active_sample: int
    first_active_chunk_index: int
    total_samples: int


def analyze_pcm16_stream(
    chunks: list[tuple[float, bytes]],
    sample_rate: int,
    *,
    threshold_dbfs: float = AUDIBLE_THRESHOLD_DBFS,
    frame_ms: float = AUDIBLE_FRAME_MS,
) -> StreamAudibility:
    """Measure first packet and device-independent first audible playback time."""

    if not chunks or not chunks[0][1]:
        raise RuntimeError("Streaming response did not contain a PCM payload")
    if sample_rate <= 0:
        raise RuntimeError("Streaming response declared an invalid sample rate")
    payload = b"".join(chunk for _, chunk in chunks)
    if len(payload) % 2:
        raise RuntimeError("PCM16 stream ended with an incomplete sample")

    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    frame_samples = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    first_active_sample: int | None = None
    for begin in range(0, int(samples.size), frame_samples):
        frame = samples[begin : begin + frame_samples].astype(np.float64, copy=False)
        rms = float(np.sqrt(np.mean(np.square(frame))))
        if rms > threshold:
            threshold_crossings = np.flatnonzero(np.abs(frame) > threshold)
            if threshold_crossings.size == 0:
                continue
            first_active_sample = begin + int(threshold_crossings[0])
            break
    if first_active_sample is None:
        raise RuntimeError(
            f"PCM stream contains no audio above {threshold_dbfs:g} dBFS"
        )

    first_active_byte_end = first_active_sample * 2 + 2
    cumulative_bytes = 0
    first_active_chunk_index = 0
    active_chunk_arrival = chunks[0][0]
    active_chunk_start_sample = 0
    for index, (arrival_seconds, chunk) in enumerate(chunks):
        if cumulative_bytes + len(chunk) >= first_active_byte_end:
            first_active_chunk_index = index
            active_chunk_arrival = arrival_seconds
            active_chunk_start_sample = cumulative_bytes // 2
            break
        cumulative_bytes += len(chunk)

    local_active_seconds = max(
        0.0,
        float(first_active_sample - active_chunk_start_sample) / float(sample_rate),
    )
    ttfp_seconds = float(chunks[0][0])
    ideal_playback_seconds = ttfp_seconds + float(first_active_sample) / float(
        sample_rate
    )
    arrival_limited_seconds = active_chunk_arrival + local_active_seconds
    return StreamAudibility(
        ttfp_seconds=ttfp_seconds,
        audible_ttfa_seconds=max(ideal_playback_seconds, arrival_limited_seconds),
        leading_silence_seconds=float(first_active_sample) / float(sample_rate),
        first_active_sample=first_active_sample,
        first_active_chunk_index=first_active_chunk_index,
        total_samples=int(samples.size),
    )
