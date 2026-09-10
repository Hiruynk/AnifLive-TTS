from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from aniflive_tts.container_worker import TensorRTSpeakerEmbedder, _load_audio
from aniflive_tts.tse_separation import (
    MOSSFORMER2_CHECKPOINT_SHA256,
    MOSSFORMER2_MODEL_REVISION,
    MOSSFORMER2_SOURCE_REVISION,
    MossFormer2TargetSeparator,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _normalise_embedding(value: np.ndarray) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if embedding.size == 0 or not np.isfinite(embedding).all() or norm <= 1e-8:
        raise RuntimeError("speaker embedding is empty, non-finite, or zero")
    return embedding / norm


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(_normalise_embedding(left), _normalise_embedding(right)))


def _active_rms(audio: np.ndarray) -> float:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    active = values[np.abs(values) >= 10 ** (-45 / 20)]
    selected = active if active.size >= 160 else values
    return float(np.sqrt(np.mean(np.square(selected), dtype=np.float64)))


def _normalise_rms(audio: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    rms = _active_rms(audio)
    if not math.isfinite(rms) or rms <= 1e-8:
        raise RuntimeError("qualification input contains no usable speech energy")
    return np.asarray(audio, dtype=np.float32) * np.float32(target_rms / rms)


def _repeat_to_length(audio: np.ndarray, sample_count: int) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise RuntimeError("qualification input is empty")
    repeats = math.ceil(sample_count / values.size)
    return np.tile(values, repeats)[:sample_count]


def _si_sdr(estimate: np.ndarray, target: np.ndarray) -> float:
    estimate64 = np.asarray(estimate, dtype=np.float64).reshape(-1)
    target64 = np.asarray(target, dtype=np.float64).reshape(-1)
    if estimate64.shape != target64.shape:
        raise RuntimeError("SI-SDR inputs have different shapes")
    target_energy = float(np.dot(target64, target64))
    if target_energy <= 1e-12:
        raise RuntimeError("SI-SDR target has no energy")
    projection = target64 * (float(np.dot(estimate64, target64)) / target_energy)
    residual = estimate64 - projection
    ratio = float(np.dot(projection, projection)) / max(float(np.dot(residual, residual)), 1e-12)
    return 10.0 * math.log10(max(ratio, 1e-12))


def _build_mixture(
    target: np.ndarray,
    interferer: np.ndarray,
    *,
    sample_rate: int,
    duration_seconds: float,
    interferer_delay_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_count = round(duration_seconds * sample_rate)
    delay = round(interferer_delay_seconds * sample_rate)
    if delay < 0 or delay >= sample_count:
        raise RuntimeError("interferer delay must be inside the qualification duration")
    target_track = _normalise_rms(_repeat_to_length(target, sample_count))
    interferer_track = np.zeros(sample_count, dtype=np.float32)
    interferer_track[delay:] = _normalise_rms(_repeat_to_length(interferer, sample_count - delay))
    mixture = target_track + interferer_track
    peak = float(np.max(np.abs(mixture)))
    gain = min(1.0, 0.9 / max(peak, 1e-8))
    return (
        mixture * np.float32(gain),
        target_track * np.float32(gain),
        interferer_track * np.float32(gain),
    )


def qualify(args: argparse.Namespace) -> dict[str, object]:
    target, target_rate = _load_audio(args.target, sample_rate=16_000)
    interferer, interferer_rate = _load_audio(args.interferer, sample_rate=16_000)
    if target_rate != 16_000 or interferer_rate != 16_000:
        raise RuntimeError("qualification inputs were not decoded at 16 kHz")
    mixture, clean_target, clean_interferer = _build_mixture(
        target,
        interferer,
        sample_rate=16_000,
        duration_seconds=args.duration_seconds,
        interferer_delay_seconds=args.interferer_delay_seconds,
    )

    embedder = TensorRTSpeakerEmbedder(args.model_package)
    reference_embedding = embedder(target, 16_000)
    mixture_embedding = embedder(mixture, 16_000)
    interferer_embedding = embedder(clean_interferer, 16_000)
    separator = MossFormer2TargetSeparator(args.separation_model)

    import torch

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = separator.separate_target(
        mixture,
        16_000,
        reference_embedding=reference_embedding,
        embedder=embedder,
        target_threshold=args.target_threshold,
        ambiguity_margin=args.ambiguity_margin,
    )
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    separated_embedding = embedder(result.audio, 16_000)

    mixture_si_sdr = _si_sdr(mixture, clean_target)
    separated_si_sdr = _si_sdr(result.audio, clean_target)
    improvement = separated_si_sdr - mixture_si_sdr
    separated_similarity = _cosine(reference_embedding, separated_embedding)
    mixture_similarity = _cosine(reference_embedding, mixture_embedding)
    interferer_similarity = _cosine(reference_embedding, interferer_embedding)
    minimum_margin = min(
        chunk.candidate_similarities[chunk.selected_candidate]
        - chunk.candidate_similarities[1 - chunk.selected_candidate]
        for chunk in result.chunks
    )
    passed = bool(
        result.accepted
        and result.audio.shape == mixture.shape
        and np.isfinite(result.audio).all()
        and separated_similarity >= args.target_threshold
        and minimum_margin >= args.ambiguity_margin
        and improvement >= args.minimum_si_sdr_improvement_db
    )

    args.output.mkdir(parents=True, exist_ok=True)
    sf.write(args.output / "overlap-mixture.wav", mixture, 16_000, subtype="PCM_16")
    sf.write(args.output / "clean-target.wav", clean_target, 16_000, subtype="PCM_16")
    sf.write(args.output / "separated-target.wav", result.audio, 16_000, subtype="PCM_16")
    report: dict[str, object] = {
        "schema": "aniflive-tts-tse-overlap-qualification-v1",
        "passed": passed,
        "environment": {
            "platform": "linux-docker-cuda",
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "provenance": {
            "source_revision": MOSSFORMER2_SOURCE_REVISION,
            "model_revision": MOSSFORMER2_MODEL_REVISION,
            "checkpoint_sha256": MOSSFORMER2_CHECKPOINT_SHA256,
        },
        "inputs": {
            "target_sha256": _sha256_file(args.target),
            "interferer_sha256": _sha256_file(args.interferer),
            "duration_seconds": args.duration_seconds,
            "interferer_delay_seconds": args.interferer_delay_seconds,
        },
        "gates": {
            "target_threshold": args.target_threshold,
            "ambiguity_margin": args.ambiguity_margin,
            "minimum_si_sdr_improvement_db": args.minimum_si_sdr_improvement_db,
        },
        "metrics": {
            "mixture_si_sdr_db": mixture_si_sdr,
            "separated_si_sdr_db": separated_si_sdr,
            "si_sdr_improvement_db": improvement,
            "mixture_target_speaker_cosine": mixture_similarity,
            "separated_target_speaker_cosine": separated_similarity,
            "interferer_target_speaker_cosine": interferer_similarity,
            "minimum_selected_similarity": result.minimum_selected_similarity,
            "minimum_candidate_margin": minimum_margin,
            "wall_seconds": wall_seconds,
            "realtime_factor": wall_seconds / args.duration_seconds,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "chunks": [
            {
                "index": chunk.index,
                "start_sample": chunk.start_sample,
                "end_sample": chunk.end_sample,
                "candidate_similarities": list(chunk.candidate_similarities),
                "selected_candidate": chunk.selected_candidate,
                "accepted": chunk.accepted,
                "reconstruction_rmse": chunk.reconstruction_rmse,
            }
            for chunk in result.chunks
        ],
    }
    (args.output / "qualification-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qualify MossFormer2 overlap separation with TensorRT target selection."
    )
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--interferer", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--separation-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=4.0)
    parser.add_argument("--interferer-delay-seconds", type=float, default=1.0)
    parser.add_argument("--target-threshold", type=float, default=0.72)
    parser.add_argument("--ambiguity-margin", type=float, default=0.03)
    parser.add_argument("--minimum-si-sdr-improvement-db", type=float, default=3.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = qualify(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
