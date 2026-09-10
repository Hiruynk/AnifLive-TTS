from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from aniflive_tts.workstation_evaluation import spoken_content_error_rate
from aniflive_tts.workstation_reference_selection import robust_speaker_centroid
from aniflive_tts.workstation_speaker import WorkstationSpeakerVerifier


SCHEMA = "aniflive-tts-training-validation-audit-v1"


class ValidationAuditError(RuntimeError):
    pass


def _object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValidationAuditError(f"{label} is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationAuditError(f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _normalize(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size == 0 or not np.isfinite(vector).all() or norm <= 1e-12:
        raise ValidationAuditError("speaker embedding is invalid")
    return vector / norm


def _load_train_embeddings(
    train_records: list[dict[str, Any]], vector_root: Path
) -> tuple[dict[str, np.ndarray], np.ndarray, set[str]]:
    try:
        import torch
    except ImportError as error:
        raise ValidationAuditError("PyTorch is required for speaker-vector audit") from error
    values: dict[str, np.ndarray] = {}
    for record in train_records:
        item_id = str(record.get("source_item_id") or "")
        name = Path(str(record.get("path") or "")).name
        vector_path = vector_root / f"{name}.pt"
        if vector_path.is_symlink() or not vector_path.is_file():
            raise ValidationAuditError(f"speaker vector is missing for {item_id}")
        tensor = torch.load(vector_path, map_location="cpu", weights_only=True)
        values[item_id] = _normalize(tensor.detach().cpu().numpy())
    centroid, retained = robust_speaker_centroid(list(values.values()))
    keys = list(values)
    return values, centroid, {keys[index] for index in retained}


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _transcribe(model: Any, audio: Path) -> str:
    segments, _ = model.transcribe(
        str(audio),
        language="ja",
        beam_size=5,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    return "".join(segment.text for segment in segments).strip()


def audit(
    *,
    bundle: Path,
    vectors: Path,
    asr_model: Path,
    speaker_component: Path,
) -> dict[str, Any]:
    bundle = bundle.resolve(strict=True)
    vectors = vectors.resolve(strict=True)
    descriptor_path = bundle / "training-input.json"
    descriptor = _object(descriptor_path, "training descriptor")
    if descriptor.get("schema") != "aniflive-v2proplus-training-input-v2":
        raise ValidationAuditError("training descriptor schema is unsupported")
    train_path = bundle / "train" / "manifest.json"
    validation_path = bundle / "validation" / "manifest.json"
    train = _object(train_path, "train manifest")
    validation = _object(validation_path, "validation manifest")
    if train.get("split") != "train" or validation.get("split") != "validation":
        raise ValidationAuditError("split manifests do not match the requested audit")
    train_records = train.get("items")
    validation_records = validation.get("items")
    if (
        not isinstance(train_records, list)
        or not train_records
        or not all(isinstance(row, dict) for row in train_records)
        or not isinstance(validation_records, list)
        or not validation_records
        or not all(isinstance(row, dict) for row in validation_records)
    ):
        raise ValidationAuditError("split manifest inventory is malformed")

    train_embeddings, centroid, retained = _load_train_embeddings(train_records, vectors)
    train_rows: list[dict[str, Any]] = []
    for record in train_records:
        item_id = str(record["source_item_id"])
        quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
        train_rows.append(
            {
                "item_id": item_id,
                "path": str(record["path"]),
                "centroid_cosine": float(train_embeddings[item_id] @ centroid),
                "robust_centroid_member": item_id in retained,
                "duration_seconds": float(record.get("duration_seconds") or 0.0),
                "silence_ratio": float(quality.get("silence_ratio") or 0.0),
                "quality_score": float(quality.get("quality_score") or 0.0),
            }
        )
    train_scores = [row["centroid_cosine"] for row in train_rows]

    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise ValidationAuditError("Faster Whisper is required for validation audit") from error
    asr = WhisperModel(
        str(asr_model.resolve(strict=True)),
        device="cuda",
        device_index=0,
        compute_type="float16",
        local_files_only=True,
    )
    speaker = WorkstationSpeakerVerifier(speaker_component.resolve(strict=True))
    validation_rows: list[dict[str, Any]] = []
    for record in validation_records:
        relative = Path(str(record.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValidationAuditError("validation audio path is unsafe")
        audio = (bundle / relative).resolve(strict=True)
        try:
            audio.relative_to(bundle)
        except ValueError as error:
            raise ValidationAuditError("validation audio escaped its bundle") from error
        if audio.is_symlink() or _sha256(audio) != record.get("sha256"):
            raise ValidationAuditError("validation audio failed integrity validation")
        import soundfile as sf

        samples, sample_rate = sf.read(str(audio), dtype="float32", always_2d=False)
        if np.asarray(samples).ndim == 2:
            samples = np.asarray(samples).mean(axis=1)
        reference = str(record.get("transcript") or "").strip()
        hypothesis = _transcribe(asr, audio)
        content_error = spoken_content_error_rate(reference, hypothesis, "ja")
        validation_rows.append(
            {
                "item_id": str(record.get("source_item_id") or ""),
                "path": str(record["path"]),
                "duration_seconds": float(record.get("duration_seconds") or 0.0),
                "verified_transcript": reference,
                "independent_asr": hypothesis,
                "asr_disagreement": content_error,
                "source_speaker_centroid_cosine": float(
                    _normalize(speaker(np.asarray(samples), int(sample_rate))) @ centroid
                ),
            }
        )
    validation_asr = [row["asr_disagreement"] for row in validation_rows]
    validation_speaker = [row["source_speaker_centroid_cosine"] for row in validation_rows]
    suspicious = [
        row
        for row in validation_rows
        if row["asr_disagreement"] > 0.18
        or row["source_speaker_centroid_cosine"] < 0.72
    ]
    suspicious.sort(
        key=lambda row: (-row["asr_disagreement"], row["source_speaker_centroid_cosine"])
    )
    low_train = sorted(
        (row for row in train_rows if not row["robust_centroid_member"]),
        key=lambda row: row["centroid_cosine"],
    )
    return {
        "schema": SCHEMA,
        "status": "review-required" if suspicious else "passed",
        "scope": {
            "train_manifest_sha256": _sha256(train_path),
            "validation_manifest_sha256": _sha256(validation_path),
            "test_split_accessed": False,
        },
        "train_speaker_distribution": {
            "count": len(train_rows),
            "median": float(np.median(train_scores)),
            "p10": _percentile(train_scores, 10),
            "minimum": min(train_scores),
            "maximum": max(train_scores),
            "robust_centroid_members": len(retained),
            "outliers": low_train,
        },
        "validation": {
            "count": len(validation_rows),
            "asr_disagreement_median": float(np.median(validation_asr)),
            "asr_disagreement_p95": _percentile(validation_asr, 95),
            "source_speaker_cosine_median": float(np.median(validation_speaker)),
            "source_speaker_cosine_p10": _percentile(validation_speaker, 10),
            "suspicious_count": len(suspicious),
            "suspicious": suspicious,
            "items": validation_rows,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--speaker-component", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        bundle=args.bundle,
        vectors=args.vectors,
        asr_model=args.asr_model,
        speaker_component=args.speaker_component,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
