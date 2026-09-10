from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aniflive_tts.workstation_dataset_scope import (
    DatasetScopeError,
    materialize_worker_dataset_scope,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _bundle(root: Path) -> Path:
    inventory: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        audio_dir = root / split / "wav"
        audio_dir.mkdir(parents=True)
        audio = audio_dir / f"{split}.wav"
        audio.write_bytes(f"RIFF-{split}-private-value".encode())
        record = {
            "source_item_id": f"item-{split}",
            "path": f"{split}/wav/{audio.name}",
            "sha256": _sha256(audio),
            "split": split,
            "transcript": f"private-{split}-transcript",
            "language": "ja",
            "speaker": "voice",
        }
        inventory.append(record)
        _write_json(
            root / split / "manifest.json",
            {"split": split, "count": 1, "items": [record]},
        )
    training_list = root / "train" / "voice.list"
    training_list.write_text("train.wav|voice|ja|train\n", encoding="utf-8")
    _write_json(
        root / "training-input.json",
        {
            "schema": "aniflive-v2proplus-training-input-v2",
            "dataset_id": "dataset-test",
            "frozen_manifest_sha256": "a" * 64,
            "training_list_sha256": _sha256(training_list),
            "training_entrypoint": "train/voice.list",
            "validation_entrypoint": "validation/manifest.json",
            "test_entrypoint": "test/manifest.json",
            "audio_files": inventory,
        },
    )
    return root


@pytest.mark.parametrize(
    ("visible", "sealed"),
    [
        (("train",), ("validation", "test")),
        (("train", "validation"), ("test",)),
        (("train", "test"), ("validation",)),
    ],
)
def test_worker_dataset_scope_exposes_only_permitted_splits(
    tmp_path: Path,
    visible: tuple[str, ...],
    sealed: tuple[str, ...],
) -> None:
    source = _bundle(tmp_path / "source")
    destination = tmp_path / "scope"

    result = materialize_worker_dataset_scope(
        source=source,
        destination=destination,
        visible_splits=visible,
    )

    descriptor_text = (destination / "training-input.json").read_text(encoding="utf-8")
    descriptor = json.loads(descriptor_text)
    assert result["visible_splits"] == list(visible)
    assert descriptor["worker_scope"]["sealed_splits"] == list(sealed)
    assert {record["split"] for record in descriptor["audio_files"]} == set(visible)
    for split in visible:
        assert (destination / split / "manifest.json").is_file()
        assert f"private-{split}-transcript" in descriptor_text
    for split in sealed:
        assert not (destination / split).exists()
        assert f"private-{split}-transcript" not in descriptor_text
        assert f"{split}/manifest.json" not in descriptor_text
    assert (destination / "train" / "voice.list").is_file() == ("train" in visible)


def test_worker_dataset_scope_rejects_tampered_audio(tmp_path: Path) -> None:
    source = _bundle(tmp_path / "source")
    (source / "train" / "wav" / "train.wav").write_bytes(b"tampered")

    with pytest.raises(DatasetScopeError, match="integrity"):
        materialize_worker_dataset_scope(
            source=source,
            destination=tmp_path / "scope",
            visible_splits=("train",),
        )


def test_worker_dataset_scope_rejects_invalid_split_contract(tmp_path: Path) -> None:
    source = _bundle(tmp_path / "source")

    with pytest.raises(DatasetScopeError, match="split order"):
        materialize_worker_dataset_scope(
            source=source,
            destination=tmp_path / "scope",
            visible_splits=("validation", "train"),
        )
