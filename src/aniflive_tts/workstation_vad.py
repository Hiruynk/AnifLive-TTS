from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .dataset_pipeline import assert_linux_docker_runtime
from .tse_pipeline import SpeakerSegment, mono_float32


FSMN_VAD_BACKEND = "fsmn-vad-funasr-1.4.11-cuda-v1"
_MODEL_FILE_LIMIT = 64
_MODEL_BYTE_LIMIT = 1024 * 1024 * 1024


class WorkstationVadError(RuntimeError):
    """Raised when managed workstation VAD cannot run deterministically."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _inspect_model(root: Path) -> dict[str, Any]:
    supplied = Path(root).expanduser()
    if supplied.is_symlink():
        raise WorkstationVadError("the FSMN-VAD model root cannot be a symbolic link")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as error:
        raise WorkstationVadError("the local FSMN-VAD model directory was not found") from error
    if not resolved.is_dir():
        raise WorkstationVadError("the FSMN-VAD model must be a directory")
    records: list[dict[str, Any]] = []
    total_bytes = 0
    pending = [resolved]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise WorkstationVadError("the FSMN-VAD model could not be inspected") from error
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise WorkstationVadError("an FSMN-VAD asset could not be inspected") from error
            if entry.is_symlink():
                raise WorkstationVadError("the FSMN-VAD model cannot contain symbolic links")
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise WorkstationVadError("the FSMN-VAD model can contain only regular files")
            if len(records) >= _MODEL_FILE_LIMIT:
                raise WorkstationVadError("the FSMN-VAD model exceeds the file-count limit")
            total_bytes += int(info.st_size)
            if total_bytes > _MODEL_BYTE_LIMIT:
                raise WorkstationVadError("the FSMN-VAD model exceeds the byte limit")
            records.append(
                {
                    "path": path.relative_to(resolved).as_posix(),
                    "bytes": int(info.st_size),
                    "sha256": _sha256_file(path),
                }
            )
    names = {str(record["path"]) for record in records}
    required = {"am.mvn", "config.yaml", "configuration.json", "model.pt"}
    if names != required:
        raise WorkstationVadError("the FSMN-VAD model inventory is incomplete or contains extra files")
    encoded = "\n".join(
        f"{record['path']}:{record['bytes']}:{record['sha256']}"
        for record in sorted(records, key=lambda item: str(item["path"]))
    ).encode("utf-8")
    return {
        "root": resolved,
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(records),
        "total_bytes": total_bytes,
    }


def _load_model(path: Path) -> tuple[Any, Mapping[str, str]]:
    try:
        import funasr
        from funasr import AutoModel
    except ImportError as error:
        raise WorkstationVadError(
            "offline FunASR/FSMN-VAD dependencies are missing from the Linux worker"
        ) from error
    try:
        model = AutoModel(
            model=str(path),
            trust_remote_code=False,
            disable_update=True,
            device="cuda:0",
        )
    except Exception as error:
        raise WorkstationVadError("the offline FSMN-VAD model could not be loaded") from error
    return model, {"funasr": str(getattr(funasr, "__version__", "unknown"))}


def _result_ranges(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, list) and value:
        value = value[0]
    if not isinstance(value, Mapping):
        raise WorkstationVadError("FSMN-VAD returned a malformed result")
    ranges = value.get("value")
    if not isinstance(ranges, list):
        raise WorkstationVadError("FSMN-VAD returned no speech ranges")
    result: list[tuple[float, float]] = []
    previous_end = -1.0
    for item in ranges:
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) != 2
        ):
            raise WorkstationVadError("FSMN-VAD returned a malformed speech range")
        try:
            start_ms = float(item[0])
            end_ms = float(item[1])
        except (TypeError, ValueError) as error:
            raise WorkstationVadError("FSMN-VAD returned a non-numeric speech range") from error
        if (
            not math.isfinite(start_ms)
            or not math.isfinite(end_ms)
            or start_ms < 0
            or end_ms <= start_ms
            or start_ms < previous_end
        ):
            raise WorkstationVadError("FSMN-VAD speech ranges are invalid or unordered")
        result.append((start_ms, end_ms))
        previous_end = end_ms
    return result


def _silence_validated_cut(
    samples: np.ndarray,
    *,
    start: int,
    desired_end: int,
    minimum_tail: int,
    sample_rate: int,
    search_ms: float = 1_500.0,
    silence_dbfs: float = -45.0,
    minimum_silence_ms: float = 50.0,
) -> int | None:
    """Choose a stable quiet boundary before a hard limit, or quarantine the split."""

    frame = max(1, round(sample_rate * 0.01))
    minimum_frames = max(1, math.ceil(minimum_silence_ms / 10.0))
    search_start = max(start + frame, desired_end - round(search_ms * sample_rate / 1000.0))
    search_end = min(desired_end, samples.size - minimum_tail)
    if search_end - search_start < frame * minimum_frames:
        return None
    quiet_runs: list[tuple[int, int, float]] = []
    run_start: int | None = None
    run_energy: list[float] = []
    cursor = search_start
    while cursor + frame <= search_end:
        window = samples[cursor : cursor + frame]
        rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64)) + 1e-12))
        dbfs = 20.0 * math.log10(max(rms, 1e-8))
        if dbfs <= silence_dbfs:
            if run_start is None:
                run_start = cursor
                run_energy = []
            run_energy.append(dbfs)
        else:
            if run_start is not None and len(run_energy) >= minimum_frames:
                quiet_runs.append((run_start, cursor, min(run_energy)))
            run_start = None
            run_energy = []
        cursor += frame
    if run_start is not None and len(run_energy) >= minimum_frames:
        quiet_runs.append((run_start, cursor, min(run_energy)))
    if not quiet_runs:
        return None
    run_start, run_end, _ = min(
        quiet_runs,
        key=lambda value: (abs(desired_end - ((value[0] + value[1]) // 2)), value[2]),
    )
    return max(start + frame, min(desired_end, (run_start + run_end) // 2))


@dataclass(frozen=True)
class WorkstationFsmnVad:
    model_path: Path
    model_factory: Callable[[Path], tuple[Any, Mapping[str, str]]] | None = None
    backend_id: str = FSMN_VAD_BACKEND

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        minimum_speech_ms: float = 160.0,
        maximum_segment_ms: float = 30_000.0,
        context_ms: float = 40.0,
    ) -> tuple[tuple[SpeakerSegment, ...], dict[str, Any]]:
        assert_linux_docker_runtime()
        samples = mono_float32(audio)
        if sample_rate != 16_000:
            raise WorkstationVadError("FSMN-VAD requires canonical 16000 Hz mono audio")
        for field, value, minimum, maximum in (
            ("minimum_speech_ms", minimum_speech_ms, 20.0, 10_000.0),
            ("maximum_segment_ms", maximum_segment_ms, 200.0, 120_000.0),
            ("context_ms", context_ms, 0.0, 2_000.0),
        ):
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise WorkstationVadError(f"{field} is outside the supported range")
        if maximum_segment_ms < minimum_speech_ms:
            raise WorkstationVadError("maximum_segment_ms must cover minimum_speech_ms")
        before = _inspect_model(self.model_path)
        root = before.pop("root")
        model, versions_value = (self.model_factory or _load_model)(root)
        try:
            generated = model.generate(input=samples, fs=sample_rate)
        except Exception as error:
            raise WorkstationVadError("offline FSMN-VAD inference failed") from error
        raw_ranges = _result_ranges(generated)
        context_samples = round(context_ms * sample_rate / 1000.0)
        minimum_samples = max(1, math.ceil(minimum_speech_ms * sample_rate / 1000.0))
        maximum_samples = max(minimum_samples, math.floor(maximum_segment_ms * sample_rate / 1000.0))
        bounds: list[tuple[int, int, bool, bool]] = []
        for start_ms, end_ms in raw_ranges:
            start = max(0, math.floor(start_ms * sample_rate / 1000.0) - context_samples)
            end = min(samples.size, math.ceil(end_ms * sample_rate / 1000.0) + context_samples)
            if end - start < minimum_samples:
                continue
            if bounds and start <= bounds[-1][1]:
                start = bounds[-1][0]
                bounds.pop()
            cursor = start
            while end - cursor > maximum_samples:
                desired_end = cursor + maximum_samples
                cut = _silence_validated_cut(
                    samples,
                    start=cursor,
                    desired_end=desired_end,
                    minimum_tail=minimum_samples,
                    sample_rate=sample_rate,
                )
                forced = cut is None
                selected_end = desired_end if cut is None else cut
                bounds.append((cursor, selected_end, not forced, forced))
                cursor = selected_end
            if end - cursor >= minimum_samples:
                bounds.append((cursor, end, True, False))
        segments: list[SpeakerSegment] = []
        for start, end, silence_validated, forced_split in bounds:
            region = samples[start:end]
            rms = float(np.sqrt(np.mean(np.square(region, dtype=np.float64)) + 1e-12))
            segments.append(
                SpeakerSegment(
                    index=len(segments),
                    start_sample=start,
                    end_sample=end,
                    rms_dbfs=20.0 * math.log10(max(rms, 1e-8)),
                    silence_validated=silence_validated,
                    forced_split=forced_split,
                )
            )
        after = _inspect_model(root)
        after.pop("root")
        if after != before:
            raise WorkstationVadError("the FSMN-VAD model changed during inference")
        return tuple(segments), {
            "backend": self.backend_id,
            "execution_environment": "linux-docker-cuda-only",
            "network_required": False,
            "model": before,
            "runtime_versions": {
                str(key): str(value) for key, value in dict(versions_value).items()
            },
            "raw_range_count": len(raw_ranges),
            "segment_count": len(segments),
            "forced_split_count": sum(segment.forced_split for segment in segments),
            "silence_validated_count": sum(
                segment.silence_validated for segment in segments
            ),
            "minimum_speech_ms": minimum_speech_ms,
            "maximum_segment_ms": maximum_segment_ms,
            "context_ms": context_ms,
        }


__all__ = [
    "FSMN_VAD_BACKEND",
    "WorkstationFsmnVad",
    "WorkstationVadError",
]
