from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


REFERENCE_SELECTION_SCHEMA = "aniflive-tts-reference-selection-v2"
BLIND_REFERENCE_SCHEMA = "aniflive-tts-blind-reference-sweep-v1"


class ReferenceSelectionError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _normalize(vector: Any, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.size == 0 or not np.isfinite(value).all():
        raise ReferenceSelectionError(f"{label} speaker embedding is invalid")
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ReferenceSelectionError(f"{label} speaker embedding has zero norm")
    return value / norm


def robust_speaker_centroid(embeddings: Sequence[Any]) -> tuple[np.ndarray, list[int]]:
    if len(embeddings) < 2:
        raise ReferenceSelectionError("reference selection requires at least two embeddings")
    vectors = [_normalize(value, f"item {index}") for index, value in enumerate(embeddings)]
    width = vectors[0].size
    if any(vector.size != width for vector in vectors):
        raise ReferenceSelectionError("speaker embedding dimensions do not match")
    matrix = np.stack(vectors)
    initial = _normalize(matrix.mean(axis=0), "initial centroid")
    similarities = matrix @ initial
    median = float(np.median(similarities))
    mad = float(np.median(np.abs(similarities - median)))
    threshold = median - max(0.02, 3.0 * mad)
    retained = [index for index, value in enumerate(similarities) if value >= threshold]
    if len(retained) < 2:
        retained = list(range(len(vectors)))
    centroid = _normalize(matrix[retained].mean(axis=0), "robust centroid")
    return centroid, retained


def rank_reference_candidates(
    records: Sequence[Mapping[str, Any]],
    embeddings: Mapping[str, Any],
    *,
    limit: int = 5,
) -> dict[str, Any]:
    eligible: list[tuple[Mapping[str, Any], np.ndarray]] = []
    rejected: list[dict[str, str]] = []
    for record in records:
        item_id = str(record.get("source_item_id") or "")
        failures: list[str] = []
        verification = record.get("verification_hashes")
        quality = record.get("quality")
        duration = float(record.get("duration_seconds") or 0.0)
        if record.get("split") != "train":
            failures.append("not-train")
        if not isinstance(verification, Mapping) or not verification.get("transcript"):
            failures.append("transcript-unverified")
        if not isinstance(verification, Mapping) or not verification.get("speaker"):
            failures.append("speaker-unverified")
        if not 3.0 <= duration <= 10.0:
            failures.append("duration")
        if not isinstance(quality, Mapping) or float(quality.get("quality_score") or 0.0) < 70:
            failures.append("quality")
        if not isinstance(quality, Mapping) or not isinstance(
            quality.get("silence_ratio"), (int, float)
        ) or float(quality["silence_ratio"]) > 0.4:
            failures.append("silence")
        if not isinstance(quality, Mapping) or float(quality.get("clipped_ratio") or 0.0) > 0.001:
            failures.append("clipping")
        if record.get("acquisition_route", "clean") != "clean":
            failures.append("not-clean")
        if item_id not in embeddings:
            failures.append("speaker-vector-missing")
        if failures:
            rejected.append({"item_id": item_id, "reason": ",".join(failures)})
            continue
        eligible.append((record, _normalize(embeddings[item_id], item_id)))
    if len(eligible) < 2:
        raise ReferenceSelectionError("fewer than two clean train reference candidates passed")
    centroid, retained = robust_speaker_centroid([embedding for _, embedding in eligible])
    retained_ids = {str(eligible[index][0]["source_item_id"]) for index in retained}
    ranked: list[dict[str, Any]] = []
    for record, embedding in eligible:
        quality = record["quality"]
        duration = float(record["duration_seconds"])
        silence = float(quality["silence_ratio"])
        quality_score = float(quality["quality_score"]) / 100.0
        centroid_cosine = float(embedding @ centroid)
        duration_score = max(0.0, 1.0 - abs(duration - 5.5) / 4.5)
        score = (
            0.40 * quality_score
            + 0.35 * centroid_cosine
            + 0.15 * duration_score
            + 0.10 * (1.0 - silence)
        )
        ranked.append(
            {
                "item_id": str(record["source_item_id"]),
                "path": str(record["path"]),
                "sha256": str(record["sha256"]),
                "text": str(record["transcript"]),
                "language": str(record["language"]),
                "duration_seconds": duration,
                "quality_score": float(quality["quality_score"]),
                "silence_ratio": silence,
                "speaker_centroid_cosine": round(centroid_cosine, 9),
                "robust_centroid_member": str(record["source_item_id"]) in retained_ids,
                "selection_score": round(score, 9),
            }
        )
    ranked.sort(key=lambda row: (-row["selection_score"], row["item_id"]))
    return {
        "schema": REFERENCE_SELECTION_SCHEMA,
        "status": "pending-human-selection",
        "policy": "reviewed-quality-speaker-centroid-v2",
        "automatic_recommendation": ranked[0]["item_id"],
        "candidate_count": len(ranked),
        "top_candidates": ranked[:limit],
        "rejected": rejected,
        "human_decision": None,
    }


def load_preprocessed_speaker_embeddings(
    records: Sequence[Mapping[str, Any]], speaker_vector_dir: Path
) -> dict[str, np.ndarray]:
    if speaker_vector_dir.is_symlink() or not speaker_vector_dir.is_dir():
        raise ReferenceSelectionError("speaker vector directory is missing")
    try:
        import torch
    except ImportError as error:
        raise ReferenceSelectionError("PyTorch is required in the training worker") from error
    result: dict[str, np.ndarray] = {}
    for record in records:
        if record.get("split") != "train":
            continue
        item_id = str(record.get("source_item_id") or "")
        name = Path(str(record.get("path") or "")).name
        path = speaker_vector_dir / f"{name}.pt"
        if path.is_symlink() or not path.is_file():
            continue
        value = torch.load(path, map_location="cpu", weights_only=True)
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        result[item_id] = np.asarray(value)
    return result


def write_blind_reference_manifest(
    *,
    selection_report: Mapping[str, Any],
    generated_cases: Mapping[str, Sequence[Mapping[str, Any]]],
    output: Path,
) -> Path:
    if selection_report.get("schema") != REFERENCE_SELECTION_SCHEMA:
        raise ReferenceSelectionError("reference selection report is unsupported")
    candidates = selection_report.get("top_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ReferenceSelectionError("reference selection has no candidates")
    labels = tuple("ABCDE"[: len(candidates)])
    if set(generated_cases) != set(labels):
        raise ReferenceSelectionError("blind reference audio is incomplete")
    entries: list[dict[str, Any]] = []
    for label, candidate in zip(labels, candidates, strict=True):
        cases = generated_cases[label]
        if len(cases) != 5:
            raise ReferenceSelectionError("every reference candidate requires five cases")
        entries.append(
            {
                "label": label,
                "candidate_item_id": candidate["item_id"],
                "cases": list(cases),
            }
        )
    payload = {
        "schema": BLIND_REFERENCE_SCHEMA,
        "status": "blocked-pending-human-evidence",
        "labels": list(labels),
        "entries": entries,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "BLIND_REFERENCE_SCHEMA",
    "REFERENCE_SELECTION_SCHEMA",
    "ReferenceSelectionError",
    "load_preprocessed_speaker_embeddings",
    "rank_reference_candidates",
    "robust_speaker_centroid",
    "write_blind_reference_manifest",
]
