from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4


DERIVED_SPLIT_SCHEMA = "aniflive-tts-derived-training-split-v1"
TRAINING_INPUT_SCHEMA = "aniflive-v2proplus-training-input-v2"
SPLITS = ("train", "validation", "test")


class TrainingResplitError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TrainingResplitError(f"{label} is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingResplitError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise TrainingResplitError(f"{label} must be a JSON object")
    return value


def _split_counts(total: int) -> dict[str, int]:
    ratios = {"train": 0.85, "validation": 0.10, "test": 0.05}
    raw = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    priority = {"train": 0, "validation": 1, "test": 2}
    for name in sorted(
        ratios,
        key=lambda value: (-(raw[value] - counts[value]), priority[value]),
    )[:remaining]:
        counts[name] += 1
    if total >= 40:
        for name in ("validation", "test"):
            shortfall = max(0, 5 - counts[name])
            counts[name] += shortfall
            counts["train"] -= shortfall
        if counts["train"] < 2:
            raise TrainingResplitError("derived production split is too small")
    return counts


def _ordered(
    records: Sequence[Mapping[str, Any]], *, seed: str, purpose: str
) -> list[Mapping[str, Any]]:
    return sorted(
        records,
        key=lambda row: hashlib.sha256(
            f"{seed}:{purpose}:{row['source_item_id']}".encode("utf-8")
        ).digest(),
    )


def derive_training_bundle(
    *,
    parent_bundle: Path,
    output_root: Path,
    seed: str,
    excluded_test_item_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Create an immutable split-only derivative without mutating reviewed data."""

    parent = parent_bundle.expanduser().resolve(strict=True)
    if not isinstance(seed, str) or not seed.strip() or len(seed) > 200:
        raise TrainingResplitError("derived split seed must be 1-200 characters")
    seed = seed.strip()
    descriptor_path = parent / "training-input.json"
    descriptor = _read_object(descriptor_path, "parent training descriptor")
    if descriptor.get("schema") != TRAINING_INPUT_SCHEMA:
        raise TrainingResplitError("parent training descriptor is unsupported")
    raw_records = descriptor.get("audio_files")
    if not isinstance(raw_records, list) or len(raw_records) < 40:
        raise TrainingResplitError("derived production split requires at least 40 items")
    records: list[dict[str, Any]] = []
    item_ids: set[str] = set()
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise TrainingResplitError("parent audio inventory is malformed")
        record = dict(raw)
        item_id = record.get("source_item_id")
        relative = record.get("path")
        expected_sha256 = record.get("sha256")
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in item_ids
            or not isinstance(relative, str)
            or not any(relative.startswith(f"{name}/wav/") for name in SPLITS)
            or ".." in Path(relative).parts
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
        ):
            raise TrainingResplitError("parent audio inventory is unsafe")
        source = parent / relative
        if source.is_symlink() or not source.is_file() or _sha256_file(source) != expected_sha256:
            raise TrainingResplitError("parent audio inventory failed checksum validation")
        if any(
            not isinstance(record.get(field), str) or not str(record[field]).strip()
            for field in ("transcript", "language", "speaker")
        ):
            raise TrainingResplitError("parent annotations are incomplete")
        verifications = record.get("verification_hashes")
        if not isinstance(verifications, Mapping) or any(
            not isinstance(verifications.get(kind), str)
            or len(str(verifications[kind])) != 64
            for kind in ("transcript", "speaker")
        ):
            raise TrainingResplitError("parent verification evidence is incomplete")
        item_ids.add(item_id)
        records.append(record)

    excluded = tuple(sorted(set(excluded_test_item_ids)))
    if any(not isinstance(item_id, str) or item_id not in item_ids for item_id in excluded):
        raise TrainingResplitError("excluded test item is not in the parent bundle")
    counts = _split_counts(len(records))
    test_candidates = [record for record in records if record["source_item_id"] not in excluded]
    if len(test_candidates) < counts["test"]:
        raise TrainingResplitError("not enough unseen items remain for a derived test split")
    test_records = _ordered(test_candidates, seed=seed, purpose="test")[: counts["test"]]
    test_ids = {str(record["source_item_id"]) for record in test_records}
    remaining = [record for record in records if record["source_item_id"] not in test_ids]
    validation_records = _ordered(remaining, seed=seed, purpose="validation")[: counts["validation"]]
    validation_ids = {str(record["source_item_id"]) for record in validation_records}
    train_records = [
        record for record in remaining if record["source_item_id"] not in validation_ids
    ]
    assigned = {
        "train": train_records,
        "validation": validation_records,
        "test": test_records,
    }
    if any(len(assigned[name]) != counts[name] for name in SPLITS):
        raise TrainingResplitError("derived split assignment is inconsistent")

    provenance = {
        "schema": DERIVED_SPLIT_SCHEMA,
        "policy": "production-85-10-5-excluding-consumed-test-v1",
        "seed": seed,
        "parent_descriptor_sha256": _sha256_file(descriptor_path),
        "parent_frozen_manifest_sha256": descriptor.get("frozen_manifest_sha256"),
        "excluded_test_item_ids": list(excluded),
        "excluded_test_item_ids_sha256": hashlib.sha256(
            json.dumps(list(excluded), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "split_counts": counts,
    }
    identity = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    destination = output_root.expanduser().resolve() / identity
    descriptor_destination = destination / "training-input.json"

    def verify_existing() -> dict[str, Any]:
        current = _read_object(descriptor_destination, "derived training descriptor")
        if current.get("schema") != TRAINING_INPUT_SCHEMA or current.get(
            "derived_split"
        ) != provenance:
            raise TrainingResplitError("derived training bundle identity collided")
        for record in current.get("audio_files", []):
            path = destination / str(record.get("path", ""))
            if path.is_symlink() or not path.is_file() or _sha256_file(path) != record.get(
                "sha256"
            ):
                raise TrainingResplitError("derived audio failed integrity validation")
        return {
            "schema": TRAINING_INPUT_SCHEMA,
            "path": str(destination),
            "descriptor_sha256": _sha256_file(descriptor_destination),
            "split_counts": counts,
            "derived_split": provenance,
        }

    if destination.exists():
        return verify_existing()

    temporary = destination.with_name(f".{identity}.{uuid4().hex}.tmp")
    audio_inventory: list[dict[str, Any]] = []
    list_rows: list[str] = []
    try:
        for split_name in SPLITS:
            split_root = temporary / split_name
            audio_root = split_root / "wav"
            audio_root.mkdir(parents=True)
            split_inventory: list[dict[str, Any]] = []
            names: set[str] = set()
            for record in assigned[split_name]:
                source = parent / str(record["path"])
                name = Path(str(record["path"])).name
                if name in names:
                    raise TrainingResplitError("derived split contains duplicate basenames")
                names.add(name)
                target = audio_root / name
                shutil.copy2(source, target)
                if _sha256_file(target) != record["sha256"]:
                    raise TrainingResplitError("derived audio copy failed checksum validation")
                derived = {
                    **record,
                    "path": f"{split_name}/wav/{name}",
                    "split": split_name,
                }
                split_inventory.append(derived)
                audio_inventory.append(derived)
                if split_name == "train":
                    list_rows.append(
                        "|".join(
                            (
                                name,
                                str(record["speaker"]).strip(),
                                str(record["language"]).strip().casefold(),
                                str(record["transcript"]).strip(),
                            )
                        )
                    )
            (split_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "aniflive-training-split-manifest-v1",
                        "dataset_id": descriptor.get("dataset_id"),
                        "split": split_name,
                        "count": len(split_inventory),
                        "items": split_inventory,
                        "derived_split": provenance,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        training_list = temporary / "train" / "voice.list"
        training_list.write_text("\n".join(list_rows) + "\n", encoding="utf-8")
        source_frozen = parent / "frozen-manifest.json"
        if source_frozen.is_symlink() or not source_frozen.is_file():
            raise TrainingResplitError("parent frozen manifest is missing")
        shutil.copy2(source_frozen, temporary / "frozen-manifest.json")
        derived_descriptor = {
            "schema": TRAINING_INPUT_SCHEMA,
            "dataset_id": descriptor.get("dataset_id"),
            "frozen_manifest_sha256": descriptor.get("frozen_manifest_sha256"),
            "training_list_sha256": _sha256_file(training_list),
            "examples": counts["train"],
            "split_counts": counts,
            "training_entrypoint": "train/voice.list",
            "validation_entrypoint": "validation/manifest.json",
            "test_entrypoint": "test/manifest.json",
            "audio_files": audio_inventory,
            "derived_split": provenance,
        }
        (temporary / "training-input.json").write_text(
            json.dumps(derived_descriptor, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return verify_existing()


__all__ = [
    "DERIVED_SPLIT_SCHEMA",
    "TRAINING_INPUT_SCHEMA",
    "TrainingResplitError",
    "derive_training_bundle",
]
