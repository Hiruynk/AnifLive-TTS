from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np


EmbeddingFunction = Callable[[np.ndarray, int], np.ndarray]
OverlapFunction = Callable[[np.ndarray, int], bool]


@dataclass(frozen=True)
class VoiceActivityConfig:
    frame_ms: float = 20.0
    minimum_speech_ms: float = 160.0
    maximum_gap_ms: float = 120.0
    context_ms: float = 40.0
    threshold_above_noise_db: float = 12.0
    minimum_threshold_dbfs: float = -48.0


@dataclass(frozen=True)
class SpeakerVerificationConfig:
    target_threshold: float = 0.72
    review_margin: float = 0.08
    extraction_gap_ms: float = 120.0


@dataclass(frozen=True)
class SpeakerSegment:
    index: int
    start_sample: int
    end_sample: int
    rms_dbfs: float
    similarity: float | None = None
    overlap: bool = False
    decision: str = "pending"
    silence_validated: bool = True
    forced_split: bool = False

    @property
    def sample_count(self) -> int:
        return self.end_sample - self.start_sample

    def duration_seconds(self, sample_rate: int) -> float:
        return self.sample_count / float(sample_rate)


@dataclass(frozen=True)
class ExtractionResult:
    segments: tuple[SpeakerSegment, ...]
    audio: np.ndarray
    sample_rate: int

    @property
    def target_seconds(self) -> float:
        return sum(
            segment.duration_seconds(self.sample_rate)
            for segment in self.segments
            if segment.decision == "target"
        )

    @property
    def review_seconds(self) -> float:
        return sum(
            segment.duration_seconds(self.sample_rate)
            for segment in self.segments
            if segment.decision == "review"
        )


def mono_float32(audio: np.ndarray | Sequence[float]) -> np.ndarray:
    values = np.asarray(audio)
    if values.ndim == 2:
        values = np.mean(values.astype(np.float32), axis=0)
    elif values.ndim != 1:
        raise ValueError("audio must be mono or channels-first stereo")
    if np.issubdtype(values.dtype, np.integer):
        scale = float(max(abs(np.iinfo(values.dtype).min), np.iinfo(values.dtype).max))
        values = values.astype(np.float32) / scale
    else:
        values = values.astype(np.float32, copy=False)
    if values.size == 0:
        raise ValueError("audio cannot be empty")
    if not np.isfinite(values).all():
        raise ValueError("audio contains non-finite samples")
    return np.clip(values, -1.0, 1.0)


def detect_voice_activity(
    audio: np.ndarray | Sequence[float],
    sample_rate: int,
    *,
    config: VoiceActivityConfig = VoiceActivityConfig(),
) -> tuple[SpeakerSegment, ...]:
    """Detect speech-sized active regions without a neural fallback.

    This detector is deliberately a deterministic front-end. Production TSE
    still verifies every region with the injected TensorRT speaker embedder.
    """

    samples = mono_float32(audio)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    frame_samples = _milliseconds_to_samples(config.frame_ms, sample_rate)
    minimum_frames = max(1, math.ceil(config.minimum_speech_ms / config.frame_ms))
    gap_frames = max(0, math.ceil(config.maximum_gap_ms / config.frame_ms))
    context_samples = _milliseconds_to_samples(config.context_ms, sample_rate)

    padded = np.pad(samples, (0, (-samples.size) % frame_samples))
    frames = padded.reshape(-1, frame_samples)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
    levels = 20.0 * np.log10(np.maximum(rms, 1e-8))
    noise_floor = float(np.percentile(levels, 20.0))
    threshold = min(
        max(
            float(config.minimum_threshold_dbfs),
            noise_floor + float(config.threshold_above_noise_db),
        ),
        float(np.max(levels)) - 3.0,
    )
    active = levels >= threshold

    runs: list[tuple[int, int]] = []
    start: int | None = None
    last_active: int | None = None
    for index, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = index
            last_active = index
            continue
        if start is not None and last_active is not None and index - last_active > gap_frames:
            runs.append((start, last_active + 1))
            start = None
            last_active = None
    if start is not None and last_active is not None:
        runs.append((start, last_active + 1))

    segments: list[SpeakerSegment] = []
    for frame_start, frame_end in runs:
        if frame_end - frame_start < minimum_frames:
            continue
        begin = max(0, frame_start * frame_samples - context_samples)
        end = min(samples.size, frame_end * frame_samples + context_samples)
        region = samples[begin:end]
        region_rms = float(np.sqrt(np.mean(np.square(region, dtype=np.float64)) + 1e-12))
        segments.append(
            SpeakerSegment(
                index=len(segments),
                start_sample=begin,
                end_sample=end,
                rms_dbfs=20.0 * math.log10(max(region_rms, 1e-8)),
            )
        )
    return tuple(_merge_overlapping_segments(segments, samples))


def verify_target_speaker(
    audio: np.ndarray | Sequence[float],
    sample_rate: int,
    segments: Iterable[SpeakerSegment],
    *,
    reference_audio: np.ndarray | Sequence[float],
    embedder: EmbeddingFunction,
    overlap_detector: OverlapFunction | None = None,
    config: SpeakerVerificationConfig = SpeakerVerificationConfig(),
) -> tuple[SpeakerSegment, ...]:
    samples = mono_float32(audio)
    reference = mono_float32(reference_audio)
    target = _normalised_embedding(embedder(reference, sample_rate))
    verified: list[SpeakerSegment] = []
    review_floor = config.target_threshold - config.review_margin

    for expected_index, segment in enumerate(segments):
        if segment.index != expected_index:
            raise ValueError("segments must use contiguous ordered indices")
        if not 0 <= segment.start_sample < segment.end_sample <= samples.size:
            raise ValueError("segment bounds are outside the source audio")
        region = samples[segment.start_sample : segment.end_sample]
        candidate = _normalised_embedding(embedder(region, sample_rate))
        similarity = float(np.dot(target, candidate))
        overlap = bool(overlap_detector(region, sample_rate)) if overlap_detector else False
        if overlap:
            decision = "review"
        elif similarity >= config.target_threshold:
            decision = "target"
        elif similarity >= review_floor:
            decision = "review"
        else:
            decision = "rejected"
        verified.append(
            replace(
                segment,
                similarity=max(-1.0, min(1.0, similarity)),
                overlap=overlap,
                decision=decision,
            )
        )
    return tuple(verified)


def apply_review_decisions(
    segments: Iterable[SpeakerSegment], decisions: dict[int, str]
) -> tuple[SpeakerSegment, ...]:
    allowed = {"target", "review", "rejected"}
    result: list[SpeakerSegment] = []
    for segment in segments:
        decision = decisions.get(segment.index, segment.decision)
        if decision not in allowed:
            raise ValueError(f"unsupported TSE decision: {decision}")
        result.append(replace(segment, decision=decision))
    return tuple(result)


def extract_target_audio(
    audio: np.ndarray | Sequence[float],
    sample_rate: int,
    segments: Iterable[SpeakerSegment],
    *,
    include_review: bool = False,
    replacement_audio: Mapping[int, np.ndarray | Sequence[float]] | None = None,
    config: SpeakerVerificationConfig = SpeakerVerificationConfig(),
) -> ExtractionResult:
    samples = mono_float32(audio)
    selected: list[np.ndarray] = []
    records = tuple(segments)
    gap = np.zeros(
        _milliseconds_to_samples(config.extraction_gap_ms, sample_rate),
        dtype=np.float32,
    )
    replacements = dict(replacement_audio or {})
    known_indices = {segment.index for segment in records}
    unknown = sorted(set(replacements) - known_indices)
    if unknown:
        raise ValueError("replacement audio references an unknown segment")
    for segment in records:
        if segment.decision != "target" and not (
            include_review and segment.decision == "review"
        ):
            continue
        if selected and gap.size:
            selected.append(gap)
        replacement = replacements.get(segment.index)
        if replacement is None:
            selected.append(samples[segment.start_sample : segment.end_sample])
            continue
        separated = mono_float32(replacement)
        if separated.size != segment.sample_count:
            raise ValueError("replacement audio length does not match its segment")
        selected.append(separated)
    output = np.concatenate(selected) if selected else np.empty(0, dtype=np.float32)
    return ExtractionResult(records, output.astype(np.float32, copy=False), sample_rate)


def _merge_overlapping_segments(
    segments: Sequence[SpeakerSegment], samples: np.ndarray
) -> tuple[SpeakerSegment, ...]:
    if not segments:
        return ()
    bounds: list[tuple[int, int]] = []
    for segment in segments:
        if bounds and segment.start_sample <= bounds[-1][1]:
            bounds[-1] = (bounds[-1][0], max(bounds[-1][1], segment.end_sample))
        else:
            bounds.append((segment.start_sample, segment.end_sample))
    merged: list[SpeakerSegment] = []
    for begin, end in bounds:
        region = samples[begin:end]
        rms = float(np.sqrt(np.mean(np.square(region, dtype=np.float64)) + 1e-12))
        merged.append(
            SpeakerSegment(
                index=len(merged),
                start_sample=begin,
                end_sample=end,
                rms_dbfs=20.0 * math.log10(max(rms, 1e-8)),
            )
        )
    return tuple(merged)


def _normalised_embedding(value: np.ndarray | Sequence[float]) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    if embedding.size == 0 or not np.isfinite(embedding).all():
        raise ValueError("speaker embedding is empty or non-finite")
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-8:
        raise ValueError("speaker embedding has zero magnitude")
    return embedding / norm


def _milliseconds_to_samples(milliseconds: float, sample_rate: int) -> int:
    if not math.isfinite(milliseconds) or milliseconds < 0:
        raise ValueError("millisecond values must be finite and non-negative")
    return max(1, int(round(milliseconds * sample_rate / 1000.0)))


__all__ = [
    "EmbeddingFunction",
    "ExtractionResult",
    "OverlapFunction",
    "SpeakerSegment",
    "SpeakerVerificationConfig",
    "VoiceActivityConfig",
    "apply_review_decisions",
    "detect_voice_activity",
    "extract_target_audio",
    "mono_float32",
    "verify_target_speaker",
]
