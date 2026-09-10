from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aniflive_tts.workstation_training_filter import (
    TrainingFilterError,
    derive_quality_filtered_training_bundle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "bundle"
    inventory = []
    for split, count in (("train", 7), ("validation", 5), ("test", 5)):
        audio_root = bundle / split / "wav"
        audio_root.mkdir(parents=True)
        rows = []
        for index in range(count):
            audio = audio_root / f"{split}-{index}.wav"
            audio.write_bytes(f"RIFF-{split}-{index}".encode())
            row = {
                "source_item_id": f"item-{split}-{index}",
                "path": f"{split}/wav/{audio.name}",
                "sha256": _sha256(audio),
                "split": split,
                "speaker": "voice",
                "language": "ja",
                "transcript": f"text {split} {index}",
            }
            rows.append(row)
            inventory.append(row)
        _json(
            bundle / split / "manifest.json",
            {"schema": "split-v1", "split": split, "count": count, "items": rows},
        )
    voice_list = bundle / "train" / "voice.list"
    voice_list.write_text("parent\n", encoding="utf-8")
    (bundle / "frozen-manifest.json").write_text("{}\n", encoding="utf-8")
    _json(
        bundle / "training-input.json",
        {
            "schema": "aniflive-v2proplus-training-input-v2",
            "dataset_id": "dataset-test",
            "frozen_manifest_sha256": "f" * 64,
            "training_list_sha256": _sha256(voice_list),
            "audio_files": inventory,
        },
    )
    audit = tmp_path / "audit.json"
    _json(
        audit,
        {
            "schema": "aniflive-tts-training-validation-audit-v1",
            "status": "passed",
            "scope": {
                "train_manifest_sha256": _sha256(bundle / "train" / "manifest.json"),
                "validation_manifest_sha256": _sha256(
                    bundle / "validation" / "manifest.json"
                ),
                "test_split_accessed": False,
            },
            "train_speaker_distribution": {
                "outliers": [
                    {
                        "item_id": "item-train-0",
                        "robust_centroid_member": False,
                        "centroid_cosine": 0.5,
                    }
                ]
            },
        },
    )
    return bundle, audit


def test_quality_filter_removes_only_audited_train_outliers(tmp_path: Path) -> None:
    bundle, audit = _fixture(tmp_path)

    result = derive_quality_filtered_training_bundle(
        parent_bundle=bundle,
        audit_report=audit,
        output_root=tmp_path / "derived",
    )

    output = Path(result["path"])
    descriptor = json.loads((output / "training-input.json").read_text(encoding="utf-8"))
    assert result["split_counts"] == {"train": 6, "validation": 5, "test": 5}
    assert not (output / "train" / "wav" / "train-0.wav").exists()
    assert (output / "validation" / "wav" / "validation-0.wav").is_file()
    assert (output / "test" / "wav" / "test-0.wav").is_file()
    assert descriptor["training_quality_filter"]["audit_test_split_accessed"] is False
    assert "item-train-0" not in (output / "train" / "voice.list").read_text(
        encoding="utf-8"
    )
    assert derive_quality_filtered_training_bundle(
        parent_bundle=bundle,
        audit_report=audit,
        output_root=tmp_path / "derived",
    ) == result


def test_quality_filter_rejects_an_audit_that_accessed_test(tmp_path: Path) -> None:
    bundle, audit = _fixture(tmp_path)
    value = json.loads(audit.read_text(encoding="utf-8"))
    value["scope"]["test_split_accessed"] = True
    _json(audit, value)

    with pytest.raises(TrainingFilterError, match="unsupported"):
        derive_quality_filtered_training_bundle(
            parent_bundle=bundle,
            audit_report=audit,
            output_root=tmp_path / "derived",
        )


def test_quality_filter_rejects_validation_exclusion(tmp_path: Path) -> None:
    bundle, audit = _fixture(tmp_path)
    value = json.loads(audit.read_text(encoding="utf-8"))
    value["train_speaker_distribution"]["outliers"][0]["item_id"] = (
        "item-validation-0"
    )
    _json(audit, value)

    with pytest.raises(TrainingFilterError, match="train split"):
        derive_quality_filtered_training_bundle(
            parent_bundle=bundle,
            audit_report=audit,
            output_root=tmp_path / "derived",
        )
