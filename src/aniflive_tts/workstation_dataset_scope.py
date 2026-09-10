from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4


TRAINING_INPUT_SCHEMA = "aniflive-v2proplus-training-input-v2"
SPLITS = ("train", "validation", "test")


class DatasetScopeError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DatasetScopeError(f"{label} is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetScopeError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise DatasetScopeError(f"{label} must be a JSON object")
    return value


def materialize_worker_dataset_scope(
    *,
    source: Path,
    destination: Path,
    visible_splits: Sequence[str],
) -> dict[str, Any]:
    """Copy only the split files one worker phase is permitted to observe."""

    source = source.expanduser().resolve(strict=True)
    if source.is_symlink() or not source.is_dir():
        raise DatasetScopeError("training bundle is missing")
    selected = tuple(dict.fromkeys(visible_splits))
    if (
        not selected
        or any(split not in SPLITS for split in selected)
        or tuple(split for split in SPLITS if split in selected) != selected
    ):
        raise DatasetScopeError("worker dataset scope has invalid split order")
    descriptor_path = source / "training-input.json"
    descriptor = _object(descriptor_path, "training descriptor")
    if descriptor.get("schema") != TRAINING_INPUT_SCHEMA:
        raise DatasetScopeError("worker dataset scope requires training input v2")
    inventory = descriptor.get("audio_files")
    if not isinstance(inventory, list) or any(
        not isinstance(record, Mapping) for record in inventory
    ):
        raise DatasetScopeError("training descriptor inventory is malformed")

    destination = destination.expanduser().resolve()
    if destination.exists():
        raise DatasetScopeError("worker dataset scope already exists")
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    scoped_inventory: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    split_hashes: dict[str, str] = {}
    try:
        for split in selected:
            source_manifest = source / split / "manifest.json"
            manifest = _object(source_manifest, f"{split} manifest")
            records = manifest.get("items")
            if (
                manifest.get("split") != split
                or not isinstance(records, list)
                or any(not isinstance(record, Mapping) for record in records)
            ):
                raise DatasetScopeError(f"{split} manifest is malformed")
            target_root = temporary / split
            target_audio = target_root / "wav"
            target_audio.mkdir(parents=True)
            for record in records:
                relative = Path(str(record.get("path") or ""))
                expected_sha256 = record.get("sha256")
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or len(relative.parts) != 3
                    or relative.parts[:2] != (split, "wav")
                    or not isinstance(expected_sha256, str)
                    or len(expected_sha256) != 64
                ):
                    raise DatasetScopeError(f"{split} audio inventory is unsafe")
                source_audio = (source / relative).resolve(strict=True)
                try:
                    source_audio.relative_to(source)
                except ValueError as error:
                    raise DatasetScopeError(f"{split} audio escaped its bundle") from error
                if source_audio.is_symlink() or _sha256_file(source_audio) != expected_sha256:
                    raise DatasetScopeError(f"{split} audio failed integrity validation")
                target_audio_path = target_audio / relative.name
                shutil.copy2(source_audio, target_audio_path)
                if _sha256_file(target_audio_path) != expected_sha256:
                    raise DatasetScopeError(f"{split} audio copy failed integrity validation")
                scoped_inventory.append(dict(record))
            shutil.copy2(source_manifest, target_root / "manifest.json")
            split_counts[split] = len(records)
            split_hashes[split] = _sha256_file(source_manifest)
            if split == "train":
                source_list = source / "train" / "voice.list"
                expected_list_sha = descriptor.get("training_list_sha256")
                if (
                    source_list.is_symlink()
                    or not source_list.is_file()
                    or not isinstance(expected_list_sha, str)
                    or _sha256_file(source_list) != expected_list_sha
                ):
                    raise DatasetScopeError("training list failed integrity validation")
                shutil.copy2(source_list, target_root / "voice.list")

        scoped_descriptor = {
            "schema": TRAINING_INPUT_SCHEMA,
            "dataset_id": descriptor.get("dataset_id"),
            "frozen_manifest_sha256": descriptor.get("frozen_manifest_sha256"),
            "training_list_sha256": (
                descriptor.get("training_list_sha256") if "train" in selected else None
            ),
            "examples": split_counts.get("train", 0),
            "split_counts": split_counts,
            "audio_files": scoped_inventory,
            "worker_scope": {
                "source_descriptor_sha256": _sha256_file(descriptor_path),
                "visible_splits": list(selected),
                "sealed_splits": [split for split in SPLITS if split not in selected],
                "split_manifest_sha256": split_hashes,
            },
        }
        if "train" in selected:
            scoped_descriptor["training_entrypoint"] = "train/voice.list"
        if "validation" in selected:
            scoped_descriptor["validation_entrypoint"] = "validation/manifest.json"
        if "test" in selected:
            scoped_descriptor["test_entrypoint"] = "test/manifest.json"
        (temporary / "training-input.json").write_text(
            json.dumps(scoped_descriptor, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return {
        "path": str(destination),
        "visible_splits": list(selected),
        "sealed_splits": [split for split in SPLITS if split not in selected],
        "descriptor_sha256": _sha256_file(destination / "training-input.json"),
    }


__all__ = [
    "DatasetScopeError",
    "materialize_worker_dataset_scope",
]
