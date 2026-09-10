from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aniflive_tts.workstation_training_resplit import derive_training_bundle


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parent_bundle(root: Path, count: int = 40) -> Path:
    root.mkdir(parents=True)
    inventory = []
    split_names = ("train", "validation", "test")
    for index in range(count):
        split = split_names[index % len(split_names)]
        path = root / split / "wav" / f"seg_{index:04d}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"audio-{index}".encode())
        inventory.append(
            {
                "path": f"{split}/wav/{path.name}",
                "sha256": _sha256(path),
                "source_item_id": f"item_{index:04d}",
                "source_parent_item_id": None,
                "split": split,
                "duration_seconds": 2.0,
                "transcript": f"sentence {index}",
                "language": "ja",
                "speaker": "voice",
                "quality": {"quality_score": 90.0},
                "acquisition_route": "clean",
                "verification_hashes": {
                    "transcript": f"{index:064x}",
                    "speaker": f"{index + 1:064x}",
                },
            }
        )
    (root / "frozen-manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "training-input.json").write_text(
        json.dumps(
            {
                "schema": "aniflive-v2proplus-training-input-v2",
                "dataset_id": "dataset_test",
                "frozen_manifest_sha256": "f" * 64,
                "audio_files": inventory,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_derived_split_is_immutable_and_excludes_consumed_test(tmp_path: Path) -> None:
    parent = _parent_bundle(tmp_path / "parent")
    consumed = [f"item_{index:04d}" for index in range(5)]

    result = derive_training_bundle(
        parent_bundle=parent,
        output_root=tmp_path / "derived",
        seed="run-2",
        excluded_test_item_ids=consumed,
    )
    repeated = derive_training_bundle(
        parent_bundle=parent,
        output_root=tmp_path / "derived",
        seed="run-2",
        excluded_test_item_ids=consumed,
    )

    assert result == repeated
    assert result["split_counts"] == {"train": 30, "validation": 5, "test": 5}
    test_manifest = json.loads(
        (Path(result["path"]) / "test" / "manifest.json").read_text(encoding="utf-8")
    )
    assert not {row["source_item_id"] for row in test_manifest["items"]}.intersection(
        consumed
    )
    assert json.loads(
        (parent / "training-input.json").read_text(encoding="utf-8")
    )["audio_files"][0]["split"] == "train"
