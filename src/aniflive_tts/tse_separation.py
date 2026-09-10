from __future__ import annotations

import hashlib
import math
import os
import sys
from argparse import Namespace
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MOSSFORMER2_SOURCE_REVISION = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"
MOSSFORMER2_SOURCE_ARCHIVE_SHA256 = (
    "f8f8d2f2190b9909b51e91ce886d1c7efedb7349b3ff6d3ab521451165fbd8da"
)
MOSSFORMER2_MODEL_REVISION = "407cb030cd66340918ebb6c8cc63b18f8592cdbe"
MOSSFORMER2_CHECKPOINT_SHA256 = (
    "00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a"
)
MOSSFORMER2_CHECKPOINT_BYTES = 670_353_271
MOSSFORMER2_SAMPLE_RATE = 16_000


EmbeddingFunction = Callable[[np.ndarray, int], np.ndarray]
EmbeddingScorer = Callable[[np.ndarray], float]


class SeparationBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeparationChunk:
    index: int
    start_sample: int
    end_sample: int
    candidate_similarities: tuple[float, float]
    selected_candidate: int
    accepted: bool
    reconstruction_rmse: float


@dataclass(frozen=True)
class TargetSeparationResult:
    audio: np.ndarray
    sample_rate: int
    chunks: tuple[SeparationChunk, ...]
    accepted: bool

    @property
    def minimum_selected_similarity(self) -> float:
        if not self.chunks:
            return -1.0
        return min(
            chunk.candidate_similarities[chunk.selected_candidate]
            for chunk in self.chunks
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _normalised_embedding(value: np.ndarray | Sequence[float]) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    if embedding.size == 0 or not np.isfinite(embedding).all():
        raise SeparationBackendError("speaker embedding is empty or non-finite")
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-8:
        raise SeparationBackendError("speaker embedding has zero magnitude")
    return embedding / norm


def _normalise_model_input(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Match ClearerVoice's two-stage -25 dB input normalization."""

    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    epsilon = 1e-6
    target = 10 ** (-25 / 20)
    rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
    if not math.isfinite(rms) or rms <= 1e-8:
        raise SeparationBackendError("separator window has no usable speech energy")
    first_scalar = target / (rms + epsilon)
    normalised = values * np.float32(first_scalar)
    power = np.square(normalised, dtype=np.float64)
    active_power = power[power > float(np.mean(power, dtype=np.float64))]
    if active_power.size == 0:
        raise SeparationBackendError("separator window normalization is undefined")
    active_rms = float(np.sqrt(np.mean(active_power, dtype=np.float64)))
    second_scalar = target / (active_rms + epsilon)
    normalised *= np.float32(second_scalar)
    restore_scalar = 1.0 / (first_scalar * second_scalar + epsilon)
    return normalised, restore_scalar


def select_target_candidate(
    candidates: Sequence[np.ndarray],
    *,
    sample_rate: int,
    reference_embedding: np.ndarray,
    embedder: EmbeddingFunction,
    embedding_scorer: EmbeddingScorer | None = None,
    target_threshold: float,
    ambiguity_margin: float,
) -> tuple[int, tuple[float, float], bool]:
    """Select one separated stream using the existing speaker verifier.

    MossFormer2 performs blind two-speaker separation. It is never trusted to
    identify the target: the workstation-owned TensorRT speaker verifier remains
    authoritative for that decision.
    """

    if len(candidates) != 2:
        raise SeparationBackendError("MossFormer2 must return exactly two candidates")
    if not math.isfinite(target_threshold) or not 0.0 <= target_threshold <= 1.0:
        raise SeparationBackendError("target_threshold must be between 0 and 1")
    if not math.isfinite(ambiguity_margin) or not 0.0 <= ambiguity_margin <= 0.5:
        raise SeparationBackendError("ambiguity_margin must be between 0 and 0.5")
    reference = _normalised_embedding(reference_embedding)
    similarities: list[float] = []
    for candidate in candidates:
        audio = np.asarray(candidate, dtype=np.float32).reshape(-1)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise SeparationBackendError("separator produced empty or non-finite audio")
        embedding = _normalised_embedding(embedder(audio, sample_rate))
        score = (
            float(embedding_scorer(embedding))
            if embedding_scorer is not None
            else float(np.dot(reference, embedding))
        )
        if not math.isfinite(score):
            raise SeparationBackendError("speaker scorer returned a non-finite value")
        similarities.append(float(np.clip(score, -1.0, 1.0)))
    selected = int(np.argmax(np.asarray(similarities)))
    rejected = 1 - selected
    accepted = (
        similarities[selected] >= target_threshold
        and similarities[selected] - similarities[rejected] >= ambiguity_margin
    )
    return selected, (similarities[0], similarities[1]), accepted


class MossFormer2TargetSeparator:
    """Pinned, offline MossFormer2 separator for Linux Docker TSE workers."""

    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        source_dir: Path | None = None,
        window_seconds: float = 2.0,
        stride_ratio: float = 0.75,
    ) -> None:
        if sys.platform != "linux":
            raise SeparationBackendError("MossFormer2 separation only runs in Linux Docker")
        if not math.isfinite(window_seconds) or not 1.0 <= window_seconds <= 8.0:
            raise SeparationBackendError("separator window_seconds must be between 1 and 8")
        if not math.isfinite(stride_ratio) or not 0.5 <= stride_ratio < 1.0:
            raise SeparationBackendError("separator stride_ratio must be between 0.5 and 1")

        self.checkpoint_dir = checkpoint_dir.resolve(strict=True)
        self.source_dir = (
            source_dir
            or Path(
                os.environ.get(
                    "ANIFLIVE_TTS_CLEARVOICE_SOURCE_DIR",
                    "/opt/aniflive-tts/clearvoice-source",
                )
            )
        ).resolve(strict=True)
        self._validate_source()
        checkpoint = self._validate_checkpoint()

        import torch

        if not torch.cuda.is_available():
            raise SeparationBackendError("CUDA is unavailable inside the Linux TSE worker")
        source_text = str(self.source_dir)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        try:
            from clearvoice.models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K
        except ImportError as error:
            raise SeparationBackendError(
                "pinned MossFormer2 source could not be imported"
            ) from error

        args = Namespace(
            encoder_embedding_dim=512,
            mossformer_sequence_dim=512,
            num_mossformer_layer=24,
            encoder_kernel_size=16,
            num_spks=2,
        )
        model = MossFormer2_SS_16K(args).model
        try:
            document = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except Exception as error:
            raise SeparationBackendError("MossFormer2 checkpoint could not be loaded") from error
        weights = document.get("model") if isinstance(document, dict) else None
        if not isinstance(weights, dict):
            raise SeparationBackendError("MossFormer2 checkpoint has no model state")
        state = model.state_dict()
        loaded: dict[str, object] = {}
        missing: list[str] = []
        for key, value in state.items():
            candidates = (key, key.removeprefix("module."), f"module.{key}")
            matched = next(
                (
                    weights[candidate]
                    for candidate in candidates
                    if candidate in weights
                    and getattr(weights[candidate], "shape", None) == value.shape
                ),
                None,
            )
            if matched is None:
                missing.append(key)
            else:
                loaded[key] = matched
        if missing:
            raise SeparationBackendError(
                "MossFormer2 checkpoint does not match the pinned architecture "
                f"({len(missing)} tensors)"
            )
        model.load_state_dict(loaded, strict=True)
        self._torch = torch
        self._device = torch.device("cuda")
        self._model = model.to(self._device).eval()
        self.window_samples = round(window_seconds * MOSSFORMER2_SAMPLE_RATE)
        self.stride_samples = round(self.window_samples * stride_ratio)

    def _validate_source(self) -> None:
        revision_path = self.source_dir / "ANIFLIVE_TTS_CLEARVOICE_REVISION"
        try:
            revision = revision_path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as error:
            raise SeparationBackendError("ClearVoice source provenance is missing") from error
        if revision != MOSSFORMER2_SOURCE_REVISION:
            raise SeparationBackendError(
                "ClearVoice source revision does not match the pinned build"
            )
        module = self.source_dir / "clearvoice" / "models" / "mossformer2_ss" / "mossformer2.py"
        if not module.is_file() or module.is_symlink():
            raise SeparationBackendError("pinned MossFormer2 source module is missing")

    def _validate_checkpoint(self) -> Path:
        checkpoint = self.checkpoint_dir / "last_best_checkpoint.pt"
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise SeparationBackendError("MossFormer2 checkpoint is missing")
        if checkpoint.stat().st_size != MOSSFORMER2_CHECKPOINT_BYTES:
            raise SeparationBackendError("MossFormer2 checkpoint size does not match")
        if _sha256_file(checkpoint) != MOSSFORMER2_CHECKPOINT_SHA256:
            raise SeparationBackendError("MossFormer2 checkpoint checksum does not match")
        return checkpoint

    def separate_target(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        reference_embedding: np.ndarray,
        embedder: EmbeddingFunction,
        embedding_scorer: EmbeddingScorer | None = None,
        target_threshold: float,
        ambiguity_margin: float = 0.03,
    ) -> TargetSeparationResult:
        if sample_rate != MOSSFORMER2_SAMPLE_RATE:
            raise SeparationBackendError("MossFormer2 separator requires 16 kHz input")
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0 or not np.isfinite(samples).all():
            raise SeparationBackendError("separator input is empty or non-finite")
        starts = self._window_starts(samples.size)
        accumulated = np.zeros(samples.size, dtype=np.float64)
        weights = np.zeros(samples.size, dtype=np.float64)
        records: list[SeparationChunk] = []
        with self._torch.inference_mode():
            for index, start in enumerate(starts):
                end = min(samples.size, start + self.window_samples)
                valid = end - start
                part = samples[start:end]
                if valid < self.window_samples:
                    part = np.pad(part, (0, self.window_samples - valid))
                model_input, restore_scalar = _normalise_model_input(part)
                tensor = self._torch.from_numpy(model_input.copy())[None, :].to(
                    device=self._device, dtype=self._torch.float32
                )
                outputs = self._model(tensor)
                if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
                    raise SeparationBackendError("MossFormer2 returned a malformed output")
                input_rms = float(
                    np.sqrt(np.mean(np.square(model_input), dtype=np.float64))
                )
                candidates_list: list[np.ndarray] = []
                for output in outputs:
                    candidate = output.detach().float().cpu().numpy().reshape(-1)
                    candidate_rms = float(
                        np.sqrt(np.mean(np.square(candidate), dtype=np.float64))
                    )
                    if not math.isfinite(candidate_rms) or candidate_rms <= 1e-8:
                        raise SeparationBackendError(
                            "MossFormer2 produced a zero-energy candidate"
                        )
                    candidate = (
                        candidate
                        * np.float32(input_rms / candidate_rms)
                        * np.float32(restore_scalar)
                    )
                    candidates_list.append(candidate[:valid])
                candidates = (candidates_list[0], candidates_list[1])
                selected, similarities, accepted = select_target_candidate(
                    candidates,
                    sample_rate=sample_rate,
                    reference_embedding=reference_embedding,
                    embedder=embedder,
                    embedding_scorer=embedding_scorer,
                    target_threshold=target_threshold,
                    ambiguity_margin=ambiguity_margin,
                )
                reconstructed = candidates[0] + candidates[1]
                reconstruction_rmse = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                reconstructed - samples[start:end], dtype=np.float64
                            )
                        )
                    )
                )
                window = self._crossfade_window(index, len(starts), valid)
                accumulated[start:end] += candidates[selected].astype(np.float64) * window
                weights[start:end] += window
                records.append(
                    SeparationChunk(
                        index=index,
                        start_sample=start,
                        end_sample=end,
                        candidate_similarities=similarities,
                        selected_candidate=selected,
                        accepted=accepted,
                        reconstruction_rmse=reconstruction_rmse,
                    )
                )
        if np.any(weights <= 0):
            raise SeparationBackendError("separator overlap-add left uncovered samples")
        result = (accumulated / weights).astype(np.float32)
        if not np.isfinite(result).all():
            raise SeparationBackendError("separator overlap-add produced non-finite audio")
        return TargetSeparationResult(
            np.clip(result, -1.0, 1.0),
            sample_rate,
            tuple(records),
            all(record.accepted for record in records),
        )

    def _window_starts(self, sample_count: int) -> tuple[int, ...]:
        if sample_count <= self.window_samples:
            return (0,)
        starts = list(range(0, sample_count - self.window_samples + 1, self.stride_samples))
        last = sample_count - self.window_samples
        if starts[-1] != last:
            starts.append(last)
        return tuple(starts)

    def _crossfade_window(self, index: int, count: int, valid: int) -> np.ndarray:
        window = np.ones(valid, dtype=np.float64)
        overlap = max(0, self.window_samples - self.stride_samples)
        fade = min(overlap, valid)
        if fade and index > 0:
            window[:fade] = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float64)
        if fade and index + 1 < count:
            window[-fade:] = np.minimum(
                window[-fade:],
                np.linspace(1.0, 0.0, fade, endpoint=False, dtype=np.float64),
            )
        return np.maximum(window, np.finfo(np.float64).eps)


__all__ = [
    "MOSSFORMER2_CHECKPOINT_BYTES",
    "MOSSFORMER2_CHECKPOINT_SHA256",
    "MOSSFORMER2_MODEL_REVISION",
    "MOSSFORMER2_SAMPLE_RATE",
    "MOSSFORMER2_SOURCE_ARCHIVE_SHA256",
    "MOSSFORMER2_SOURCE_REVISION",
    "MossFormer2TargetSeparator",
    "SeparationBackendError",
    "SeparationChunk",
    "TargetSeparationResult",
    "select_target_candidate",
]
