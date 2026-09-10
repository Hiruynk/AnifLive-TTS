from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4


TRAINING_INPUT_SCHEMA = "aniflive-v2proplus-training-input-v2"
VALIDATION_AUDIT_SCHEMA = "aniflive-tts-training-validation-audit-v1"
FILTER_SCHEMA = "aniflive-tts-training-quality-filter-v1"
SPLITS = ("train", "validation", "test")


class TrainingFilterError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TrainingFilterError(f"{label} is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingFilterError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise TrainingFilterError(f"{label} must be a JSON object")
    return value


def _records(bundle: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _object(bundle / split / "manifest.json", f"{split} manifest")
    values = manifest.get("items")
    if (
        manifest.get("split") != split
        or not isinstance(values, list)
        or any(not isinstance(value, Mapping) for value in values)
    ):
        raise TrainingFilterError(f"{split} manifest is malformed")
    return manifest, [dict(value) for value in values]


def derive_quality_filtered_training_bundle(
    *,
    parent_bundle: Path,
    audit_report: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Create an immutable derivative excluding audited train speaker outliers."""

    parent = parent_bundle.expanduser().resolve(strict=True)
    descriptor_path = parent / "training-input.json"
    descriptor = _object(descriptor_path, "parent training descriptor")
    if descriptor.get("schema") != TRAINING_INPUT_SCHEMA:
        raise TrainingFilterError("parent training descriptor is unsupported")
    audit_path = audit_report.expanduser().resolve(strict=True)
    audit = _object(audit_path, "training validation audit")
    scope = audit.get("scope")
    distribution = audit.get("train_speaker_distribution")
    if (
        audit.get("schema") != VALIDATION_AUDIT_SCHEMA
        or not isinstance(scope, Mapping)
        or scope.get("test_split_accessed") is not False
        or not isinstance(distribution, Mapping)
    ):
        raise TrainingFilterError("training validation audit is unsupported")

    manifests: dict[str, dict[str, Any]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    inventory_ids: set[str] = set()
    for split in SPLITS:
        manifests[split], records[split] = _records(parent, split)
        for record in records[split]:
            item_id = record.get("source_item_id")
            relative = Path(str(record.get("path") or ""))
            expected = record.get("sha256")
            if (
                not isinstance(item_id, str)
                or not item_id
                or item_id in inventory_ids
                or relative.is_absolute()
                or ".." in relative.parts
                or len(relative.parts) != 3
                or relative.parts[:2] != (split, "wav")
                or not isinstance(expected, str)
                or len(expected) != 64
            ):
                raise TrainingFilterError("parent training inventory is unsafe")
            audio = (parent / relative).resolve(strict=True)
            try:
                audio.relative_to(parent)
            except ValueError as error:
                raise TrainingFilterError("parent audio escaped its bundle") from error
            if audio.is_symlink() or not audio.is_file() or _sha256(audio) != expected:
                raise TrainingFilterError("parent audio failed integrity validation")
            inventory_ids.add(item_id)
    if scope.get("train_manifest_sha256") != _sha256(parent / "train" / "manifest.json"):
        raise TrainingFilterError("audit does not describe this train manifest")

    raw_outliers = distribution.get("outliers")
    if not isinstance(raw_outliers, list) or not raw_outliers:
        raise TrainingFilterError("audit contains no speaker outliers to filter")
    excluded = sorted(
        {
            str(row.get("item_id"))
            for row in raw_outliers
            if isinstance(row, Mapping)
            and row.get("robust_centroid_member") is False
            and isinstance(row.get("item_id"), str)
        }
    )
    train_ids = {str(record["source_item_id"]) for record in records["train"]}
    if not excluded or any(item_id not in train_ids for item_id in excluded):
        raise TrainingFilterError("audit outliers are not confined to the train split")
    retained = {
        "train": [
            record
            for record in records["train"]
            if record["source_item_id"] not in set(excluded)
        ],
        "validation": records["validation"],
        "test": records["test"],
    }
    if len(retained["train"]) < 2 or any(
        len(retained[split]) < 5 for split in ("validation", "test")
    ):
        raise TrainingFilterError("quality filter would violate production split minimums")

    provenance = {
        "schema": FILTER_SCHEMA,
        "policy": "robust-speaker-centroid-train-filter-v1",
        "parent_descriptor_sha256": _sha256(descriptor_path),
        "audit_report_sha256": _sha256(audit_path),
        "audit_test_split_accessed": False,
        "excluded_train_item_ids": excluded,
        "excluded_train_item_ids_sha256": hashlib.sha256(
            json.dumps(excluded, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "split_counts": {split: len(retained[split]) for split in SPLITS},
    }
    identity = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    destination = output_root.expanduser().resolve() / identity

    def verify_existing() -> dict[str, Any]:
        current = _object(destination / "training-input.json", "filtered descriptor")
        if current.get("training_quality_filter") != provenance:
            raise TrainingFilterError("quality-filtered bundle identity collided")
        for record in current.get("audio_files", []):
            if not isinstance(record, Mapping):
                raise TrainingFilterError("filtered inventory is malformed")
            audio = destination / str(record.get("path") or "")
            if audio.is_symlink() or not audio.is_file() or _sha256(audio) != record.get(
                "sha256"
            ):
                raise TrainingFilterError("filtered audio failed integrity validation")
        return {
            "schema": TRAINING_INPUT_SCHEMA,
            "path": str(destination),
            "descriptor_sha256": _sha256(destination / "training-input.json"),
            "split_counts": provenance["split_counts"],
            "training_quality_filter": provenance,
        }

    if destination.exists():
        return verify_existing()

    temporary = destination.with_name(f".{identity}.{uuid4().hex}.tmp")
    inventory: list[dict[str, Any]] = []
    training_rows: list[str] = []
    try:
        for split in SPLITS:
            target_root = temporary / split
            target_audio = target_root / "wav"
            target_audio.mkdir(parents=True)
            split_records: list[dict[str, Any]] = []
            for record in retained[split]:
                source = parent / str(record["path"])
                target = target_audio / source.name
                shutil.copy2(source, target)
                if _sha256(target) != record["sha256"]:
                    raise TrainingFilterError("filtered audio copy failed integrity validation")
                copied = {**record, "path": f"{split}/wav/{target.name}", "split": split}
                split_records.append(copied)
                inventory.append(copied)
                if split == "train":
                    training_rows.append(
                        "|".join(
                            (
                                target.name,
                                str(record["speaker"]).strip(),
                                str(record["language"]).strip().casefold(),
                                str(record["transcript"]).strip(),
                            )
                        )
                    )
            _write_json(
                target_root / "manifest.json",
                {
                    **manifests[split],
                    "count": len(split_records),
                    "items": split_records,
                    "training_quality_filter": provenance,
                },
            )
        training_list = temporary / "train" / "voice.list"
        training_list.write_text("\n".join(training_rows) + "\n", encoding="utf-8")
        frozen = parent / "frozen-manifest.json"
        if frozen.is_symlink() or not frozen.is_file():
            raise TrainingFilterError("parent frozen manifest is missing")
        shutil.copy2(frozen, temporary / "frozen-manifest.json")
        _write_json(
            temporary / "training-input.json",
            {
                **{
                    key: value
                    for key, value in descriptor.items()
                    if key not in {"audio_files", "derived_split", "training_quality_filter"}
                },
                "examples": len(retained["train"]),
                "split_counts": provenance["split_counts"],
                "training_list_sha256": _sha256(training_list),
                "audio_files": inventory,
                "training_quality_filter": provenance,
            },
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return verify_existing()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "FILTER_SCHEMA",
    "TrainingFilterError",
    "derive_quality_filtered_training_bundle",
]
