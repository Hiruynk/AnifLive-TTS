from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable, Sequence

import numpy as np

from .dataset_pipeline import assert_linux_docker_runtime


SORTFORMER_COMPONENT_ID = "sortformer-overlap-diarization"
SORTFORMER_MODEL_REVISION = "5240a64075176943f677d30fa2171c780229f341"
SORTFORMER_MODEL_FILENAME = "diar_streaming_sortformer_4spk-v2.q8_0.gguf"
SORTFORMER_MODEL_SHA256 = (
    "0679cfeb1ce356d0dea9470b31274f4bfc7eb927497d82005483770666da998a"
)
SORTFORMER_MODEL_BYTES = 147_075_776
SORTFORMER_FRAME_SECONDS = 0.08
NEMO_SPEECH_VERSION = "0.1.0"
SORTFORMER_POSTPROCESS_SCHEMA = "aniflive-sortformer-postprocess-v1"
_RTTM_SPEAKER = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_MAX_RTTM_BYTES = 32 * 1024 * 1024
_MAX_RTTM_SEGMENTS = 1_000_000


class WorkstationDiarizationError(RuntimeError):
    """Raised when the managed overlap-diarization stage is unusable."""


@dataclass(frozen=True)
class DiarizationPostprocess:
    """Pinned Sortformer hysteresis used by the acquisition worker.

    These values were selected on the labelled 31-minute voice-acquisition
    fixture. A lower onset recovered one additional overlap interval but
    produced substantially more false overlap evidence on clean target speech.
    """

    onset: float = 0.4
    offset: float = 0.7
    pad_onset_seconds: float = 0.05
    pad_offset_seconds: float = 0.0
    minimum_on_seconds: float = 0.2
    minimum_off_seconds: float = 0.2

    def __post_init__(self) -> None:
        values = (
            self.onset,
            self.offset,
            self.pad_onset_seconds,
            self.pad_offset_seconds,
            self.minimum_on_seconds,
            self.minimum_off_seconds,
        )
        if any(not math.isfinite(value) for value in values):
            raise WorkstationDiarizationError(
                "Sortformer post-processing values must be finite"
            )
        if not 0.0 <= self.onset <= 1.0 or not 0.0 <= self.offset <= 1.0:
            raise WorkstationDiarizationError(
                "Sortformer activation thresholds must be between zero and one"
            )
        if any(value < 0.0 or value > 10.0 for value in values[2:]):
            raise WorkstationDiarizationError(
                "Sortformer duration controls are outside the supported range"
            )

    def command_arguments(self) -> tuple[str, ...]:
        return (
            "--onset",
            str(self.onset),
            "--offset",
            str(self.offset),
            "--pad-onset",
            str(self.pad_onset_seconds),
            "--pad-offset",
            str(self.pad_offset_seconds),
            "--min-duration-on",
            str(self.minimum_on_seconds),
            "--min-duration-off",
            str(self.minimum_off_seconds),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SORTFORMER_POSTPROCESS_SCHEMA,
            "onset": self.onset,
            "offset": self.offset,
            "pad_onset_seconds": self.pad_onset_seconds,
            "pad_offset_seconds": self.pad_offset_seconds,
            "minimum_on_seconds": self.minimum_on_seconds,
            "minimum_off_seconds": self.minimum_off_seconds,
        }


@dataclass(frozen=True)
class DiarizationSegment:
    speaker: str
    start_sample: int
    end_sample: int

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample

    def as_dict(self, sample_rate: int) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "start_seconds": self.start_sample / sample_rate,
            "end_seconds": self.end_sample / sample_rate,
        }


@dataclass(frozen=True)
class DiarizationResult:
    segments: tuple[DiarizationSegment, ...]
    sample_rate: int
    backend: str
    model_sha256: str
    postprocess: DiarizationPostprocess | None = None

    def as_dict(self) -> dict[str, Any]:
        speakers = sorted({segment.speaker for segment in self.segments})
        return {
            "schema": "aniflive-sortformer-diarization-v1",
            "backend": self.backend,
            "runtime": f"NeMo-Speech.cpp-{NEMO_SPEECH_VERSION}",
            "model": "nvidia/diar_streaming_sortformer_4spk-v2",
            "model_revision": SORTFORMER_MODEL_REVISION,
            "model_sha256": self.model_sha256,
            "sample_rate": self.sample_rate,
            "frame_seconds": SORTFORMER_FRAME_SECONDS,
            "postprocess": (
                self.postprocess.as_dict() if self.postprocess is not None else None
            ),
            "speaker_count": len(speakers),
            "speakers": speakers,
            "segment_count": len(self.segments),
            "segments": [segment.as_dict(self.sample_rate) for segment in self.segments],
        }


@dataclass(frozen=True)
class DiarizationBatchResult:
    clips: tuple[DiarizationResult, ...]
    sample_rate: int
    backend: str
    model_sha256: str
    postprocess: DiarizationPostprocess | None = None

    def as_dict(self) -> dict[str, Any]:
        hypothesis_multi_speaker_clip_count = sum(
            len({segment.speaker for segment in result.segments}) >= 2
            for result in self.clips
        )
        return {
            "schema": "aniflive-sortformer-diarization-batch-v1",
            "backend": self.backend,
            "runtime": f"NeMo-Speech.cpp-{NEMO_SPEECH_VERSION}",
            "model": "nvidia/diar_streaming_sortformer_4spk-v2",
            "model_revision": SORTFORMER_MODEL_REVISION,
            "model_sha256": self.model_sha256,
            "sample_rate": self.sample_rate,
            "frame_seconds": SORTFORMER_FRAME_SECONDS,
            "mode": "vad-segment-offline-batch",
            "postprocess": (
                self.postprocess.as_dict() if self.postprocess is not None else None
            ),
            "clip_count": len(self.clips),
            "hypothesis_segment_count": sum(
                len(result.segments) for result in self.clips
            ),
            # This is raw model hypothesis telemetry. The acquisition router
            # only promotes it after duration filtering and speaker-score fusion.
            "hypothesis_multi_speaker_clip_count": (
                hypothesis_multi_speaker_clip_count
            ),
            "multi_speaker_clip_count": hypothesis_multi_speaker_clip_count,
        }


@dataclass(frozen=True)
class DiarizationCommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


DiarizationRunner = Callable[[Sequence[str], Path, float], DiarizationCommandResult]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _bounded_text(value: str, limit: int = 4_000) -> str:
    text = value.replace("\x00", "").strip()
    return text[:limit]


def _default_runner(
    argv: Sequence[str], working_directory: Path, timeout_seconds: float
) -> DiarizationCommandResult:
    environment = dict(os.environ)
    environment["HOME"] = str(working_directory)
    environment["XDG_CACHE_HOME"] = str(working_directory / "cache")
    try:
        completed = subprocess.run(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=working_directory,
            env=environment,
            check=False,
            shell=False,
            timeout=timeout_seconds,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as error:
        raise WorkstationDiarizationError(
            "the pinned NeMo-Speech.cpp runtime is missing from the Linux worker"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise WorkstationDiarizationError(
            "Sortformer diarization exceeded its hard timeout"
        ) from error
    return DiarizationCommandResult(
        int(completed.returncode),
        _bounded_text(completed.stdout),
        _bounded_text(completed.stderr),
    )


def _model_file(component_root: Path) -> Path:
    supplied = Path(component_root).expanduser()
    if supplied.is_symlink():
        raise WorkstationDiarizationError(
            "Sortformer component root cannot be a symbolic link"
        )
    try:
        root = supplied.resolve(strict=True)
    except OSError as error:
        raise WorkstationDiarizationError(
            "Sortformer component directory was not found"
        ) from error
    if not root.is_dir():
        raise WorkstationDiarizationError(
            "Sortformer component must be a directory"
        )
    files = sorted(path for path in root.rglob("*") if path.is_file())
    expected = root / SORTFORMER_MODEL_FILENAME
    if any(path.is_symlink() for path in root.rglob("*")) or files != [expected]:
        raise WorkstationDiarizationError(
            "Sortformer component inventory is incomplete or contains extra files"
        )
    if expected.stat().st_size != SORTFORMER_MODEL_BYTES:
        raise WorkstationDiarizationError("Sortformer model size does not match")
    if _sha256_file(expected) != SORTFORMER_MODEL_SHA256:
        raise WorkstationDiarizationError("Sortformer model checksum does not match")
    return expected


def parse_rttm(
    text: str,
    *,
    sample_rate: int,
    maximum_samples: int,
) -> tuple[DiarizationSegment, ...]:
    if not isinstance(text, str) or len(text.encode("utf-8")) > _MAX_RTTM_BYTES:
        raise WorkstationDiarizationError("Sortformer RTTM output is too large")
    if (
        not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or not 8_000 <= sample_rate <= 96_000
        or not isinstance(maximum_samples, int)
        or isinstance(maximum_samples, bool)
        or maximum_samples < 1
    ):
        raise WorkstationDiarizationError("RTTM audio bounds are malformed")
    records: list[DiarizationSegment] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if len(records) >= _MAX_RTTM_SEGMENTS:
            raise WorkstationDiarizationError("Sortformer RTTM has too many segments")
        fields = line.split()
        if len(fields) != 10 or fields[0] != "SPEAKER":
            raise WorkstationDiarizationError(
                f"Sortformer RTTM line {line_number} is malformed"
            )
        speaker = fields[7]
        if _RTTM_SPEAKER.fullmatch(speaker) is None:
            raise WorkstationDiarizationError(
                f"Sortformer RTTM line {line_number} has an invalid speaker label"
            )
        try:
            start_seconds = float(fields[3])
            duration_seconds = float(fields[4])
        except ValueError as error:
            raise WorkstationDiarizationError(
                f"Sortformer RTTM line {line_number} has invalid timing"
            ) from error
        if (
            not math.isfinite(start_seconds)
            or not math.isfinite(duration_seconds)
            or start_seconds < 0.0
            or duration_seconds <= 0.0
        ):
            raise WorkstationDiarizationError(
                f"Sortformer RTTM line {line_number} has invalid timing"
            )
        start = round(start_seconds * sample_rate)
        end = round((start_seconds + duration_seconds) * sample_rate)
        if start >= maximum_samples:
            continue
        start = max(0, start)
        end = min(maximum_samples, end)
        if end > start:
            records.append(DiarizationSegment(speaker, start, end))
    by_speaker: dict[str, list[DiarizationSegment]] = {}
    for record in records:
        by_speaker.setdefault(record.speaker, []).append(record)
    merged: list[DiarizationSegment] = []
    for speaker, speaker_records in by_speaker.items():
        speaker_records.sort(key=lambda item: (item.start_sample, item.end_sample))
        for record in speaker_records:
            if merged and merged[-1].speaker == speaker and (
                record.start_sample <= merged[-1].end_sample
            ):
                previous = merged[-1]
                merged[-1] = DiarizationSegment(
                    speaker,
                    previous.start_sample,
                    max(previous.end_sample, record.end_sample),
                )
            else:
                merged.append(record)
    merged.sort(key=lambda item: (item.start_sample, item.end_sample, item.speaker))
    return tuple(merged)


def _union_samples(intervals: Sequence[tuple[int, int]]) -> int:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    total = 0
    cursor_start = cursor_end = None
    for start, end in ordered:
        if cursor_start is None:
            cursor_start, cursor_end = start, end
        elif start <= cursor_end:
            cursor_end = max(cursor_end, end)
        else:
            total += cursor_end - cursor_start
            cursor_start, cursor_end = start, end
    if cursor_start is not None and cursor_end is not None:
        total += cursor_end - cursor_start
    return total


def interval_diarization_evidence(
    segments: Sequence[DiarizationSegment],
    *,
    start_sample: int,
    end_sample: int,
    sample_rate: int,
    minimum_speaker_seconds: float = 0.24,
    minimum_overlap_seconds: float = 0.16,
) -> dict[str, Any]:
    if (
        not isinstance(start_sample, int)
        or isinstance(start_sample, bool)
        or not isinstance(end_sample, int)
        or isinstance(end_sample, bool)
        or start_sample < 0
        or end_sample <= start_sample
        or not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or sample_rate < 1
    ):
        raise WorkstationDiarizationError("diarization interval is malformed")
    for name, value in (
        ("minimum_speaker_seconds", minimum_speaker_seconds),
        ("minimum_overlap_seconds", minimum_overlap_seconds),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 10.0:
            raise WorkstationDiarizationError(f"{name} is malformed")
    clipped: list[tuple[str, int, int]] = []
    coverage: dict[str, list[tuple[int, int]]] = {}
    for segment in segments:
        start = max(start_sample, segment.start_sample)
        end = min(end_sample, segment.end_sample)
        if end <= start:
            continue
        clipped.append((segment.speaker, start, end))
        coverage.setdefault(segment.speaker, []).append((start, end))
    minimum_speaker_samples = round(minimum_speaker_seconds * sample_rate)
    speaker_samples = {
        speaker: _union_samples(intervals)
        for speaker, intervals in coverage.items()
    }
    speakers = sorted(
        speaker
        for speaker, duration in speaker_samples.items()
        if duration >= minimum_speaker_samples
    )
    events: list[tuple[int, int, str]] = []
    for speaker, start, end in clipped:
        events.append((start, 1, speaker))
        events.append((end, -1, speaker))
    events.sort(key=lambda item: (item[0], item[1]))
    active_counts: dict[str, int] = {}
    overlap_samples = 0
    previous = start_sample
    index = 0
    while index < len(events):
        position = events[index][0]
        active_speakers = sum(count > 0 for count in active_counts.values())
        if active_speakers >= 2 and position > previous:
            overlap_samples += position - previous
        while index < len(events) and events[index][0] == position:
            _, delta, speaker = events[index]
            active_counts[speaker] = active_counts.get(speaker, 0) + delta
            index += 1
        previous = position
    minimum_overlap_samples = round(minimum_overlap_seconds * sample_rate)
    duration_samples = end_sample - start_sample
    overlap_evidence = overlap_samples >= minimum_overlap_samples
    speaker_change_evidence = len(speakers) >= 2
    return {
        "schema": "aniflive-diarization-interval-evidence-v1",
        "speakers": speakers,
        "speaker_count": len(speakers),
        "speaker_samples": speaker_samples,
        "overlap_samples": overlap_samples,
        "overlap_seconds": overlap_samples / sample_rate,
        "overlap_ratio": overlap_samples / duration_samples,
        "overlap_evidence": overlap_evidence,
        "speaker_change_evidence": speaker_change_evidence,
        "decision": (
            "overlap"
            if overlap_evidence
            else "multiple-speakers"
            if speaker_change_evidence
            else "single-speaker"
        ),
    }


class SortformerDiarizer:
    """Pinned native Sortformer v2 diarization for the isolated Linux worker."""

    def __init__(
        self,
        component_root: Path,
        *,
        executable: Path = Path("/opt/nemo-speech/bin/nemo-speech"),
        runner: DiarizationRunner | None = None,
        postprocess: DiarizationPostprocess | None = None,
    ) -> None:
        assert_linux_docker_runtime()
        self.model = _model_file(component_root)
        self.executable = Path(executable)
        if runner is None:
            try:
                resolved = self.executable.resolve(strict=True)
            except OSError as error:
                raise WorkstationDiarizationError(
                    "the pinned NeMo-Speech.cpp runtime is missing from the Linux worker"
                ) from error
            if not resolved.is_file() or resolved.is_symlink():
                raise WorkstationDiarizationError(
                    "the pinned NeMo-Speech.cpp executable is invalid"
                )
            self.executable = resolved
        self.runner = runner or _default_runner
        self.postprocess = postprocess or DiarizationPostprocess()

    def diarize_clips(
        self,
        audio_clips: Sequence[np.ndarray],
        sample_rate: int,
    ) -> DiarizationBatchResult:
        if (
            isinstance(audio_clips, (str, bytes))
            or not isinstance(audio_clips, Sequence)
            or len(audio_clips) > 100_000
            or sample_rate != 16_000
        ):
            raise WorkstationDiarizationError(
                "Sortformer clip batch requires finite 16000 Hz mono audio"
            )
        values: list[np.ndarray] = []
        for audio in audio_clips:
            clip = np.asarray(audio, dtype=np.float32).reshape(-1)
            if clip.size == 0 or not np.isfinite(clip).all():
                raise WorkstationDiarizationError(
                    "Sortformer clip batch contains empty or non-finite audio"
                )
            values.append(clip)
        backend = "sortformer-v2-nemo-speech-cpp-cuda-offline"
        if not values:
            return DiarizationBatchResult(
                (), sample_rate, backend, SORTFORMER_MODEL_SHA256, self.postprocess
            )
        duration_seconds = sum(clip.size for clip in values) / sample_rate
        timeout_seconds = max(
            120.0, min(6 * 60 * 60.0, duration_seconds * 2.0 + 60.0)
        )
        with tempfile.TemporaryDirectory(
            prefix="aniflive-sortformer-batch-"
        ) as temporary:
            working = Path(temporary)
            inputs = working / "input"
            outputs = working / "output"
            inputs.mkdir()
            outputs.mkdir()
            try:
                import soundfile as sf

                for position, clip in enumerate(values):
                    sf.write(
                        str(inputs / f"clip_{position:06d}.wav"),
                        clip,
                        sample_rate,
                        subtype="PCM_16",
                    )
            except Exception as error:
                raise WorkstationDiarizationError(
                    "canonical Sortformer clip batch could not be written"
                ) from error
            argv = (
                str(self.executable),
                "--quiet",
                "diarize",
                str(inputs),
                "--model",
                str(self.model),
                "--device",
                "cuda:0",
                "--offline",
                "--format",
                "rttm",
                "--output-dir",
                str(outputs),
                "--force",
                *self.postprocess.command_arguments(),
            )
            result = self.runner(argv, working, timeout_seconds)
            if result.returncode != 0:
                detail = _bounded_text(result.stderr) or "unknown native runtime error"
                raise WorkstationDiarizationError(
                    f"Sortformer clip diarization failed: {detail}"
                )
            expected = {
                outputs / f"clip_{position:06d}.rttm"
                for position in range(len(values))
            }
            actual = {path for path in outputs.rglob("*") if path.is_file()}
            if actual != expected or any(path.is_symlink() for path in outputs.rglob("*")):
                raise WorkstationDiarizationError(
                    "Sortformer clip batch produced an unexpected RTTM inventory"
                )
            clip_results: list[DiarizationResult] = []
            for position, clip in enumerate(values):
                output = outputs / f"clip_{position:06d}.rttm"
                try:
                    if output.stat().st_size > _MAX_RTTM_BYTES:
                        raise WorkstationDiarizationError(
                            "Sortformer RTTM output is too large"
                        )
                    rttm = output.read_text(encoding="utf-8")
                except UnicodeError as error:
                    raise WorkstationDiarizationError(
                        "Sortformer RTTM output is not UTF-8"
                    ) from error
                clip_results.append(
                    DiarizationResult(
                        parse_rttm(
                            rttm,
                            sample_rate=sample_rate,
                            maximum_samples=clip.size,
                        ),
                        sample_rate,
                        backend,
                        SORTFORMER_MODEL_SHA256,
                        self.postprocess,
                    )
                )
        return DiarizationBatchResult(
            tuple(clip_results),
            sample_rate,
            backend,
            SORTFORMER_MODEL_SHA256,
            self.postprocess,
        )

    def diarize(self, audio: np.ndarray, sample_rate: int) -> DiarizationResult:
        values = np.asarray(audio, dtype=np.float32).reshape(-1)
        if (
            values.size == 0
            or not np.isfinite(values).all()
            or sample_rate != 16_000
        ):
            raise WorkstationDiarizationError(
                "Sortformer requires finite 16000 Hz mono audio"
            )
        duration_seconds = values.size / sample_rate
        timeout_seconds = max(120.0, min(6 * 60 * 60.0, duration_seconds * 2.0 + 60.0))
        with tempfile.TemporaryDirectory(prefix="aniflive-sortformer-") as temporary:
            working = Path(temporary)
            source = working / "source.wav"
            output = working / "segments.rttm"
            try:
                import soundfile as sf

                sf.write(str(source), values, sample_rate, subtype="PCM_16")
            except Exception as error:
                raise WorkstationDiarizationError(
                    "canonical Sortformer input could not be written"
                ) from error
            argv = (
                str(self.executable),
                "--quiet",
                "diarize",
                str(source),
                "--model",
                str(self.model),
                "--device",
                "cuda:0",
                "--format",
                "rttm",
                "--output",
                str(output),
                *self.postprocess.command_arguments(),
            )
            result = self.runner(argv, working, timeout_seconds)
            if result.returncode != 0:
                detail = _bounded_text(result.stderr) or "unknown native runtime error"
                raise WorkstationDiarizationError(
                    f"Sortformer diarization failed: {detail}"
                )
            if not output.is_file() or output.is_symlink():
                raise WorkstationDiarizationError(
                    "Sortformer did not produce a regular RTTM output"
                )
            try:
                if output.stat().st_size > _MAX_RTTM_BYTES:
                    raise WorkstationDiarizationError(
                        "Sortformer RTTM output is too large"
                    )
                rttm = output.read_text(encoding="utf-8")
            except UnicodeError as error:
                raise WorkstationDiarizationError(
                    "Sortformer RTTM output is not UTF-8"
                ) from error
            segments = parse_rttm(
                rttm,
                sample_rate=sample_rate,
                maximum_samples=values.size,
            )
        return DiarizationResult(
            segments,
            sample_rate,
            "sortformer-v2-nemo-speech-cpp-cuda",
            SORTFORMER_MODEL_SHA256,
            self.postprocess,
        )


__all__ = [
    "DiarizationBatchResult",
    "DiarizationPostprocess",
    "DiarizationResult",
    "DiarizationSegment",
    "SORTFORMER_COMPONENT_ID",
    "SORTFORMER_FRAME_SECONDS",
    "SORTFORMER_MODEL_BYTES",
    "SORTFORMER_MODEL_FILENAME",
    "SORTFORMER_MODEL_REVISION",
    "SORTFORMER_MODEL_SHA256",
    "SORTFORMER_POSTPROCESS_SCHEMA",
    "SortformerDiarizer",
    "WorkstationDiarizationError",
    "interval_diarization_evidence",
    "parse_rttm",
]
