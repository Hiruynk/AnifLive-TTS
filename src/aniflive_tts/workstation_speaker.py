from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


WORKSTATION_SPEAKER_COMPONENT_SCHEMA = "aniflive-workstation-speaker-verifier-v1"


class WorkstationSpeakerError(RuntimeError):
    """Raised when the shared speaker-verification component is unusable."""


EmbeddingFunction = Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class SpeakerReferenceSegment:
    start_sample: int
    end_sample: int

    @property
    def sample_count(self) -> int:
        return self.end_sample - self.start_sample


class SpeakerReferencePrototype:
    """A generic, multi-reference identity prototype for acquisition workers.

    Each reference region is embedded independently and L2-normalized. Candidate
    identity uses the mean of the two strongest reference matches (or the single
    available match), which is robust to expressive/phonetic variation without
    allowing one coincidental reference match to become authoritative.
    """

    def __init__(
        self,
        embeddings: Sequence[np.ndarray],
        segments: Sequence[SpeakerReferenceSegment],
    ) -> None:
        if not embeddings or len(embeddings) != len(segments) or len(embeddings) > 16:
            raise WorkstationSpeakerError("reference prototype inventory is malformed")
        normalized: list[np.ndarray] = []
        dimension: int | None = None
        for embedding in embeddings:
            value = np.asarray(embedding, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(value))
            if (
                value.size == 0
                or not np.isfinite(value).all()
                or norm <= 1e-8
                or (dimension is not None and value.size != dimension)
            ):
                raise WorkstationSpeakerError("reference prototype embedding is malformed")
            dimension = value.size
            normalized.append((value / norm).astype(np.float32, copy=False))
        checked_segments: list[SpeakerReferenceSegment] = []
        for segment in segments:
            if (
                not isinstance(segment, SpeakerReferenceSegment)
                or isinstance(segment.start_sample, bool)
                or isinstance(segment.end_sample, bool)
                or segment.start_sample < 0
                or segment.end_sample <= segment.start_sample
            ):
                raise WorkstationSpeakerError("reference prototype segment is malformed")
            checked_segments.append(segment)
        matrix = np.stack(normalized)
        centroid = np.mean(matrix, axis=0, dtype=np.float32)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm <= 1e-8:
            raise WorkstationSpeakerError("reference prototype centroid is malformed")
        self._embeddings = matrix
        self._centroid = (centroid / centroid_norm).astype(np.float32, copy=False)
        self.segments = tuple(checked_segments)

    @property
    def centroid_embedding(self) -> np.ndarray:
        return self._centroid.copy()

    @property
    def reference_count(self) -> int:
        return int(self._embeddings.shape[0])

    def score_embedding(self, embedding: np.ndarray) -> dict[str, Any]:
        value = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(value))
        if (
            value.size != self._embeddings.shape[1]
            or not np.isfinite(value).all()
            or norm <= 1e-8
        ):
            raise WorkstationSpeakerError("candidate speaker embedding is malformed")
        scores = np.clip(self._embeddings @ (value / norm), -1.0, 1.0)
        support = min(2, scores.size)
        strongest = np.sort(scores)[-support:]
        return {
            "schema": "aniflive-speaker-reference-score-v1",
            "similarity": float(np.mean(strongest, dtype=np.float64)),
            "aggregation": "top-k-mean",
            "support_count": int(support),
            "reference_count": int(scores.size),
            "maximum_similarity": float(np.max(scores)),
            "mean_similarity": float(np.mean(scores, dtype=np.float64)),
        }

    def score_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        embedder: EmbeddingFunction,
    ) -> dict[str, Any]:
        return self.score_embedding(embedder(audio, sample_rate))

    def as_dict(self, sample_rate: int) -> dict[str, Any]:
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate < 1:
            raise WorkstationSpeakerError("reference prototype sample rate is malformed")
        return {
            "schema": "aniflive-speaker-reference-prototype-v1",
            "reference_count": self.reference_count,
            "aggregation": "top-k-mean",
            "support_count": min(2, self.reference_count),
            "segments": [
                {
                    "start_sample": item.start_sample,
                    "end_sample": item.end_sample,
                    "duration_seconds": item.sample_count / sample_rate,
                }
                for item in self.segments
            ],
        }


def build_speaker_reference_prototype(
    audio: np.ndarray,
    sample_rate: int,
    *,
    embedder: EmbeddingFunction,
    segments: Sequence[tuple[int, int]] = (),
    maximum_references: int = 8,
) -> SpeakerReferencePrototype:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise WorkstationSpeakerError("reference audio is empty or malformed")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate < 1:
        raise WorkstationSpeakerError("reference sample rate is malformed")
    if (
        not isinstance(maximum_references, int)
        or isinstance(maximum_references, bool)
        or not 1 <= maximum_references <= 16
    ):
        raise WorkstationSpeakerError("maximum_references must be between 1 and 16")
    candidates: list[SpeakerReferenceSegment] = []
    for start, end in segments:
        item = SpeakerReferenceSegment(int(start), int(end))
        if (
            item.start_sample < 0
            or item.end_sample > values.size
            or item.end_sample <= item.start_sample
        ):
            raise WorkstationSpeakerError("reference segment escaped the audio bounds")
        if item.sample_count >= sample_rate:
            candidates.append(item)
    if not candidates:
        candidates = [SpeakerReferenceSegment(0, values.size)]
    selected = sorted(
        sorted(candidates, key=lambda item: (-item.sample_count, item.start_sample))[
            :maximum_references
        ],
        key=lambda item: item.start_sample,
    )
    embeddings = [
        embedder(values[item.start_sample : item.end_sample], sample_rate)
        for item in selected
    ]
    return SpeakerReferencePrototype(embeddings, selected)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _component_contract(root: Path) -> tuple[Path, dict[str, Any]]:
    root = Path(root).expanduser()
    if root.is_symlink():
        raise WorkstationSpeakerError("speaker component cannot be a symbolic link")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise WorkstationSpeakerError("speaker component was not found") from error
    if not root.is_dir():
        raise WorkstationSpeakerError("speaker component must be a directory")
    manifest_path = root / "component.json"
    engine = root / "workstation_sv_embedding.engine"
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not engine.is_file()
        or engine.is_symlink()
    ):
        raise WorkstationSpeakerError(
            "speaker component requires component.json and workstation_sv_embedding.engine"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WorkstationSpeakerError("speaker component manifest is unreadable") from error
    if not isinstance(manifest, Mapping):
        raise WorkstationSpeakerError("speaker component manifest must be an object")
    if (
        manifest.get("schema") != WORKSTATION_SPEAKER_COMPONENT_SCHEMA
        or manifest.get("component_id") != "eres2netv2-speaker-verifier"
        or manifest.get("architecture")
        != "ERes2NetV2(baseWidth=24,scale=4,expansion=4)"
        or manifest.get("backend") != "TensorRT-11"
        or manifest.get("input_sample_rate") != 16_000
    ):
        raise WorkstationSpeakerError("speaker component contract is unsupported")
    expected_sha = manifest.get("engine_sha256")
    expected_bytes = manifest.get("engine_bytes")
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
        or not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 1
    ):
        raise WorkstationSpeakerError("speaker component fingerprint is malformed")
    if engine.stat().st_size != expected_bytes or _sha256_file(engine) != expected_sha:
        raise WorkstationSpeakerError("speaker component engine failed integrity validation")
    return engine, dict(manifest)


class WorkstationSpeakerVerifier:
    """V2ProPlus-independent TensorRT 11 ERes2NetV2 verifier.

    The engine is workstation-owned and uses the same fixed architecture and
    weights for every target-speaker dataset. No trained voice package is needed.
    """

    def __init__(self, component_root: Path) -> None:
        if not sys.platform.startswith("linux"):
            raise WorkstationSpeakerError(
                "speaker verification runs only inside the Linux Docker worker"
            )
        engine, manifest = _component_contract(component_root)
        try:
            import torch
        except ImportError as error:
            raise WorkstationSpeakerError("PyTorch CUDA orchestration is unavailable") from error
        if not torch.cuda.is_available():
            raise WorkstationSpeakerError("CUDA is unavailable inside the Linux worker")
        source_dir = Path(
            os.environ.get("ANIFLIVE_TTS_SOURCE_DIR", "/app/minimal_inference")
        ).resolve(strict=True)
        for import_root in (source_dir, source_dir / "GPT_SoVITS"):
            value = str(import_root)
            if value not in sys.path:
                sys.path.insert(0, value)
        try:
            from run_trt_inference import TRTModule
        except ImportError as error:
            raise WorkstationSpeakerError("TensorRT runtime wrapper is unavailable") from error
        self._torch = torch
        self._device = torch.device("cuda")
        self._stream = torch.cuda.Stream(device=self._device)
        self._module = TRTModule(str(engine), self._device, self._stream)
        self._input_dtype = self._module.tensor_dtype["audio"]
        maximum = self._module.input_max_shapes.get("audio")
        self._maximum_samples = int(maximum[-1]) if maximum else 180_000
        self.manifest = manifest

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != 16_000:
            from scipy.signal import resample_poly

            common = math.gcd(int(sample_rate), 16_000)
            audio = resample_poly(
                audio, 16_000 // common, int(sample_rate) // common
            ).astype(np.float32, copy=False)
        values = np.asarray(audio, dtype=np.float32).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise WorkstationSpeakerError("speaker verifier input is empty or malformed")
        minimum = 16_000
        if values.size < minimum:
            values = np.pad(values, (0, minimum - values.size))
        window = max(minimum, self._maximum_samples)
        step = max(minimum, window // 2)
        embeddings: list[np.ndarray] = []
        for start in range(0, values.size, step):
            part = values[start : start + window]
            if part.size < minimum and embeddings:
                break
            if part.size < minimum:
                part = np.pad(part, (0, minimum - part.size))
            tensor = self._torch.from_numpy(part.copy())[None, :].to(
                device=self._device, dtype=self._input_dtype
            )
            output = self._module({"audio": tensor})["sv_embedding"]
            embedding = output.detach().float().cpu().numpy().reshape(-1)
            if embedding.size == 0 or not np.isfinite(embedding).all():
                raise WorkstationSpeakerError(
                    "speaker verifier produced an invalid embedding"
                )
            embeddings.append(embedding)
            if start + window >= values.size:
                break
        if not embeddings:
            raise WorkstationSpeakerError("speaker verifier produced no windows")
        return np.mean(np.stack(embeddings), axis=0, dtype=np.float32)


def speaker_component_manifest(engine: Path, *, build_fingerprint: str) -> dict[str, Any]:
    engine = Path(engine).expanduser().resolve(strict=True)
    if not engine.is_file() or engine.is_symlink():
        raise WorkstationSpeakerError("speaker engine must be a regular file")
    if (
        not isinstance(build_fingerprint, str)
        or not build_fingerprint
        or len(build_fingerprint) > 256
    ):
        raise WorkstationSpeakerError("build_fingerprint is malformed")
    return {
        "schema": WORKSTATION_SPEAKER_COMPONENT_SCHEMA,
        "component_id": "eres2netv2-speaker-verifier",
        "architecture": "ERes2NetV2(baseWidth=24,scale=4,expansion=4)",
        "backend": "TensorRT-11",
        "input_sample_rate": 16_000,
        "engine_sha256": _sha256_file(engine),
        "engine_bytes": engine.stat().st_size,
        "build_fingerprint": build_fingerprint,
    }


__all__ = [
    "WORKSTATION_SPEAKER_COMPONENT_SCHEMA",
    "SpeakerReferencePrototype",
    "SpeakerReferenceSegment",
    "WorkstationSpeakerError",
    "WorkstationSpeakerVerifier",
    "build_speaker_reference_prototype",
    "speaker_component_manifest",
]
