from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_PREBUFFER_SWEEP_MS = (0, 16, 32, 48, 64, 80, 96, 112, 128, 160, 208, 256)


@dataclass(frozen=True)
class PlaybackChunk:
    arrival_seconds: float
    samples: int


@dataclass(frozen=True)
class PlaybackTrace:
    sample_rate: int
    recommended_prebuffer_ms: int
    chunks: tuple[PlaybackChunk, ...]

    @classmethod
    def from_pcm16_chunks(
        cls,
        *,
        sample_rate: int,
        recommended_prebuffer_ms: int,
        chunks: Iterable[tuple[float, bytes]],
    ) -> PlaybackTrace:
        records = tuple(
            PlaybackChunk(arrival_seconds=float(arrival), samples=len(payload) // 2)
            for arrival, payload in chunks
        )
        return cls(
            sample_rate=sample_rate,
            recommended_prebuffer_ms=recommended_prebuffer_ms,
            chunks=records,
        )


def _validate(trace: PlaybackTrace) -> None:
    if trace.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if trace.recommended_prebuffer_ms < 0:
        raise ValueError("recommended_prebuffer_ms must not be negative")
    if not trace.chunks:
        raise ValueError("playback trace must contain at least one chunk")
    previous = -1.0
    for chunk in trace.chunks:
        if chunk.arrival_seconds < 0.0:
            raise ValueError("chunk arrival time must not be negative")
        if chunk.arrival_seconds < previous:
            raise ValueError("chunk arrival times must be monotonic")
        if chunk.samples <= 0:
            raise ValueError("chunk sample count must be positive")
        previous = chunk.arrival_seconds


def simulate_playback(
    trace: PlaybackTrace,
    *,
    prebuffer_ms: int | None = None,
) -> dict[str, Any]:
    """Model buffered playback with integer sample accounting."""

    _validate(trace)
    selected_prebuffer_ms = (
        trace.recommended_prebuffer_ms if prebuffer_ms is None else int(prebuffer_ms)
    )
    if selected_prebuffer_ms < 0:
        raise ValueError("prebuffer_ms must not be negative")

    rate = trace.sample_rate
    prebuffer_samples = round(selected_prebuffer_ms * rate / 1000.0)
    arrivals = [round(chunk.arrival_seconds * rate) for chunk in trace.chunks]
    playback_end = arrivals[0] + prebuffer_samples
    events: list[dict[str, float | int]] = []
    total_gap_samples = 0

    for index, (arrival, chunk) in enumerate(zip(arrivals, trace.chunks, strict=True)):
        gap_samples = max(0, arrival - playback_end)
        if index > 0 and gap_samples:
            total_gap_samples += gap_samples
            events.append(
                {
                    "chunk_index": index,
                    "arrival_seconds": arrival / rate,
                    "buffer_depleted_seconds": playback_end / rate,
                    "gap_samples": gap_samples,
                    "gap_seconds": gap_samples / rate,
                }
            )
        playback_end = max(playback_end, arrival) + chunk.samples

    largest_gap_samples = max(
        (int(event["gap_samples"]) for event in events), default=0
    )
    return {
        "prebuffer_ms": selected_prebuffer_ms,
        "underrun_count": len(events),
        "underrun_samples": total_gap_samples,
        "underrun_seconds": total_gap_samples / rate,
        "largest_underrun_samples": largest_gap_samples,
        "largest_underrun_seconds": largest_gap_samples / rate,
        "gap_events": events,
    }


def analyze_playback_trace(
    trace: PlaybackTrace,
    *,
    prebuffer_sweep_ms: Iterable[int] = DEFAULT_PREBUFFER_SWEEP_MS,
) -> dict[str, Any]:
    sweep: dict[str, dict[str, Any]] = {}
    minimum_stable: int | None = None
    for value in prebuffer_sweep_ms:
        prebuffer = int(value)
        result = simulate_playback(trace, prebuffer_ms=prebuffer)
        sweep[str(prebuffer)] = result
        if minimum_stable is None and result["underrun_count"] == 0:
            minimum_stable = prebuffer
    return {
        "recommended_prebuffer_ms": trace.recommended_prebuffer_ms,
        "zero_prebuffer_gap_stress": simulate_playback(trace, prebuffer_ms=0),
        "contractual_playback": simulate_playback(trace),
        "prebuffer_sweep": sweep,
        "minimum_stable_prebuffer_ms": minimum_stable,
        "trace": [
            {
                "arrival_seconds": chunk.arrival_seconds,
                "samples": chunk.samples,
            }
            for chunk in trace.chunks
        ],
    }
