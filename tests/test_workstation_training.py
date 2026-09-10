from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
import yaml

from aniflive_tts import workstation_training as training


def _dataset(root: Path) -> Path:
    root.mkdir()
    (root / "2-name2text.txt").write_text(
        "a.wav\ta\t1\ta\nb.wav\tb\t1\tb\n", encoding="utf-8"
    )
    (root / "6-name2semantic.tsv").write_text(
        "item_name\tsemantic_audio\na.wav\t1 2\nb.wav\t2 3\n", encoding="utf-8"
    )
    for name in ("3-bert", "4-cnhubert", "5-wav32k", "7-sv_cn"):
        path = root / name
        path.mkdir()
        if name == "7-sv_cn":
            (path / "a.wav.pt").write_bytes(b"asset")
            (path / "b.wav.pt").write_bytes(b"asset")
        else:
            (path / "a.bin").write_bytes(b"asset")
    return root


def test_fixed_presets_and_advanced_bounds() -> None:
    quick = training.resolve_training_plan(
        {"preset": "Quick", "stage": "both"}, project_id="training_example"
    )
    assert quick.gpt_epochs == 4
    assert quick.sovits_epochs == 4
    assert quick.experiment_name == "training_example"
    with pytest.raises(training.TrainingWorkerError, match="custom fields require Advanced"):
        training.resolve_training_plan(
            {"preset": "balanced", "gpt_epochs": 3}, project_id="training_example"
        )
    advanced = training.resolve_training_plan(
        {
            "preset": "advanced",
            "gpt_epochs": 9,
            "sovits_epochs": 7,
            "gpt_batch_size": 3,
            "sovits_batch_size": 2,
            "gpt_learning_rate": 0.005,
            "sovits_learning_rate": 0.0002,
            "gradient_checkpointing": True,
        },
        project_id="training_example",
    )
    assert advanced.gradient_checkpointing is True
    assert advanced.gpt_epochs == 9
    with pytest.raises(training.TrainingWorkerError, match="between 1 and 100"):
        training.resolve_training_plan(
            {"preset": "advanced", "gpt_epochs": 0}, project_id="training_example"
        )


def test_preprocessed_v2proplus_dataset_contract(tmp_path: Path) -> None:
    report = training.validate_preprocessed_dataset(_dataset(tmp_path / "dataset"))
    assert report["text_rows"] == 2
    assert report["semantic_rows"] == 2
    assert report["speaker_embedding_rows"] == 2
    (tmp_path / "dataset" / "7-sv_cn").rename(tmp_path / "removed")
    with pytest.raises(training.TrainingWorkerError, match="7-sv_cn"):
        training.validate_preprocessed_dataset(tmp_path / "dataset")


def test_preprocessed_dataset_rejects_zero_or_missing_speaker_vectors(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset")
    (dataset / "7-sv_cn" / "b.wav.pt").unlink()
    with pytest.raises(training.TrainingWorkerError, match="speaker-vector preprocessing"):
        training.validate_preprocessed_dataset(dataset)


def test_training_input_bundle_verifies_list_and_every_audio_digest(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    wav = root / "wav"
    wav.mkdir(parents=True)
    records = []
    rows = []
    for index in range(2):
        path = wav / f"seg_{index}.wav"
        path.write_bytes(f"audio-{index}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"path": f"wav/{path.name}", "sha256": digest})
        rows.append(f"{path.name}|voice|ja|テスト{index}")
    training_list = root / "voice.list"
    training_list.write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / "training-input.json").write_text(
        json.dumps(
            {
                "schema": "aniflive-v2proplus-training-input-v1",
                "dataset_id": "dataset_voice",
                "training_list_sha256": hashlib.sha256(training_list.read_bytes()).hexdigest(),
                "audio_files": records,
            }
        ),
        encoding="utf-8",
    )

    verified_list, verified_wav, descriptor = training._validate_training_input_bundle(root)

    assert verified_list == training_list
    assert verified_wav == wav
    assert descriptor["dataset_id"] == "dataset_voice"
    (wav / "seg_1.wav").write_bytes(b"tampered")
    with pytest.raises(training.TrainingWorkerError, match="audio failed integrity"):
        training._validate_training_input_bundle(root)


def test_v2_training_bundle_requires_two_train_files_not_holdout_files(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    train_wav = root / "train" / "wav"
    validation_wav = root / "validation" / "wav"
    train_wav.mkdir(parents=True)
    validation_wav.mkdir(parents=True)
    train_audio = train_wav / "train.wav"
    validation_audio = validation_wav / "validation.wav"
    train_audio.write_bytes(b"train")
    validation_audio.write_bytes(b"validation")
    training_list = root / "train" / "voice.list"
    training_list.write_text("train.wav|voice|ja|訓練\n", encoding="utf-8")
    records = [
        {
            "path": "train/wav/train.wav",
            "sha256": hashlib.sha256(train_audio.read_bytes()).hexdigest(),
            "split": "train",
        },
        {
            "path": "validation/wav/validation.wav",
            "sha256": hashlib.sha256(validation_audio.read_bytes()).hexdigest(),
            "split": "validation",
        },
    ]
    (root / "training-input.json").write_text(
        json.dumps(
            {
                "schema": "aniflive-v2proplus-training-input-v2",
                "training_list_sha256": hashlib.sha256(
                    training_list.read_bytes()
                ).hexdigest(),
                "audio_files": records,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(training.TrainingWorkerError, match="two train audio files"):
        training._validate_training_input_bundle(root)


def test_generated_configs_are_fixed_to_v2proplus_and_resume_safe(tmp_path: Path) -> None:
    source = tmp_path / "source"
    config_root = source / "GPT_SoVITS" / "configs"
    config_root.mkdir(parents=True)
    (config_root / "s1longer-v2.yaml").write_text(
        yaml.safe_dump(
            {
                "train": {},
                "optimizer": {"lr": 0.01},
                "data": {},
                "model": {},
            }
        ),
        encoding="utf-8",
    )
    (config_root / "s2v2ProPlus.json").write_text(
        json.dumps({"train": {}, "data": {}, "model": {}}), encoding="utf-8"
    )
    output = tmp_path / "output"
    (output / "work" / "gpt" / "ckpt").mkdir(parents=True)
    dataset = _dataset(tmp_path / "dataset")
    work = output / "work" / "dataset"
    shutil.copytree(dataset, work)
    (work / "logs_s2_v2ProPlus").mkdir()
    inputs = {
        "pretrained_gpt": tmp_path / "gpt.ckpt",
        "pretrained_sovits_g": tmp_path / "G.pth",
        "pretrained_sovits_d": tmp_path / "D.pth",
    }
    plan = training.resolve_training_plan(
        {"preset": "balanced"}, project_id="training_example"
    )
    plan, budget = training.plan_training_budget(
        plan, duration_seconds=3600, clips=1000
    )
    gpt_path, sovits_path = training._write_configs(source, output, work, plan, inputs)
    gpt = yaml.safe_load(gpt_path.read_text(encoding="utf-8"))
    sovits = json.loads(sovits_path.read_text(encoding="utf-8"))
    assert gpt["output_dir"].endswith("/work/gpt") or gpt["output_dir"].endswith("\\work\\gpt")
    assert gpt["pretrained_s1"] == str(inputs["pretrained_gpt"])
    assert gpt["train"]["if_save_latest"] is True
    assert gpt["train"]["epochs"] == budget["stages"]["gpt"]["resolved_epochs"]
    assert sovits["train"]["epochs"] == budget["stages"]["sovits"]["resolved_epochs"]
    assert sovits["model"]["version"] == "v2ProPlus"
    assert sovits["train"]["if_save_latest"] is True
    assert sovits["train"]["pretrained_s2G"] == str(inputs["pretrained_sovits_g"])
    assert sovits["train"]["pretrained_s2D"] == str(inputs["pretrained_sovits_d"])


def test_training_pythonpath_always_includes_both_pinned_source_roots(tmp_path: Path) -> None:
    source = tmp_path / "gpt-sovits"
    inherited = os.pathsep.join((str(tmp_path / "site-a"), str(source)))
    values = training._training_pythonpath(source, inherited).split(os.pathsep)
    assert values == [
        str(source),
        str(source / "GPT_SoVITS"),
        str(tmp_path / "site-a"),
    ]


def test_checkpoint_manifest_retains_every_epoch_for_validation_selection(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    gpt_root = output / "checkpoints" / "gpt"
    sovits_root = output / "checkpoints" / "sovits"
    gpt_root.mkdir(parents=True)
    sovits_root.mkdir(parents=True)
    gpt_e9 = gpt_root / "voice-e9.ckpt"
    gpt_e10 = gpt_root / "voice-e10.ckpt"
    sovits_e9 = sovits_root / "voice_e9_s90.pth"
    sovits_e10 = sovits_root / "voice_e10_s100.pth"
    for path in (gpt_e9, gpt_e10, sovits_e9, sovits_e10):
        path.write_bytes(path.name.encode("ascii"))

    manifest_path, manifest = training._write_checkpoint_candidates_manifest(
        output,
        gpt_weights=[gpt_e10, gpt_e9],
        sovits_weights=[sovits_e10, sovits_e9],
    )

    assert manifest["schema"] == "aniflive-tts-v2proplus-checkpoint-candidates-v1"
    assert manifest["selection_required"] is True
    assert [row["epoch"] for row in manifest["gpt"]] == [9, 10]
    assert [row["epoch"] for row in manifest["sovits"]] == [9, 10]
    assert sorted(path.name for path in gpt_root.iterdir()) == [
        "voice-e10.ckpt",
        "voice-e9.ckpt",
    ]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_checkpoint_manifest_requires_a_complete_candidate_pair(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(training.TrainingWorkerError, match="SoVITS"):
        training._write_checkpoint_candidates_manifest(
            output,
            gpt_weights=[output / "voice-e1.ckpt"],
            sovits_weights=[],
        )


def test_candidate_checkpoints_remain_until_selection_and_resume_keeps_latest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    gpt_weights = output / "checkpoints" / "gpt"
    gpt_resume = output / "work" / "gpt" / "ckpt"
    sovits_weights = output / "checkpoints" / "sovits"
    sovits_resume = output / "work" / "dataset" / "logs_s2_v2ProPlus"
    for directory in (gpt_weights, gpt_resume, sovits_weights, sovits_resume):
        directory.mkdir(parents=True, exist_ok=True)
    for name in ("voice-e9.ckpt", "voice-e10.ckpt"):
        (gpt_weights / name).write_bytes(name.encode())
    for name in ("voice_e9_s90.pth", "voice_e10_s100.pth"):
        (sovits_weights / name).write_bytes(name.encode())
    for name in ("epoch=8-step=80.ckpt", "epoch=9-step=90.ckpt"):
        (gpt_resume / name).write_bytes(name.encode())
    for name in ("G_80.pth", "G_90.pth", "D_80.pth", "D_90.pth"):
        (sovits_resume / name).write_bytes(name.encode())

    training._write_checkpoint_candidates_manifest(
        output,
        gpt_weights=list(gpt_weights.glob("*.ckpt")),
        sovits_weights=list(sovits_weights.glob("*.pth")),
    )
    resume = training._copy_resume_outputs(output, output / "work" / "dataset")

    assert len(resume) == 3
    assert {path.name for path in resume} == {
        "epoch=9-step=90.ckpt",
        "G_90.pth",
        "D_90.pth",
    }
    assert sorted(path.name for path in gpt_weights.iterdir()) == [
        "voice-e10.ckpt",
        "voice-e9.ckpt",
    ]
    assert sorted(path.name for path in sovits_weights.iterdir()) == [
        "voice_e10_s100.pth",
        "voice_e9_s90.pth",
    ]

def test_native_gpt_epoch_progress_is_one_based(tmp_path, monkeypatch):
    import os
    import sys
    import aniflive_tts.workstation_training as training
    events=[]
    monkeypatch.setattr(training,"_emit_progress",lambda *event:events.append(event))
    training._run_stage(
        name="gpt",argv=(sys.executable,"-c","print('Epoch 0:');print('Epoch 1:');print('Epoch 2:')"),
        cwd=tmp_path,environment=dict(os.environ),log_path=tmp_path/"gpt.log",
        start_progress=0.1,end_progress=0.5,epochs=3,
    )
    assert [event[2] for event in events]==["gpt epoch 1/3","gpt epoch 2/3","gpt epoch 3/3"]
