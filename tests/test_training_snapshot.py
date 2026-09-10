import json
from pathlib import Path

import pytest

from aniflive_tts.training_snapshot import (
    TrainingSnapshotError,
    commit_training_snapshot,
    read_training_snapshot,
)


def write_gpt(root: Path, epoch=1, fingerprint="dataset-A"):
    with commit_training_snapshot(root, metadata={
        "stage": "gpt", "epoch": epoch, "global_step": epoch * 3,
        "job_id": "job-test", "source_fingerprint": fingerprint,
    }) as stage:
        (stage / "gpt").mkdir()
        (stage / "gpt/trainer.ckpt").write_bytes(b"framework-checkpoint")
        (stage / "gpt/random-state.pt").write_bytes(b"rng-state")
        (stage / "training-plan.json").write_text('{"epochs": 4}')
        (stage / "training-budget.json").write_text('{"resolved_epochs": 4}')


def test_complete_snapshot_preserves_budget_and_validates_source(tmp_path):
    write_gpt(tmp_path)
    root, manifest = read_training_snapshot(tmp_path, expected_source_fingerprint="dataset-A")
    assert manifest["metadata"]["epoch"] == 1
    assert json.loads((root / "training-budget.json").read_text())["resolved_epochs"] == 4
    with pytest.raises(TrainingSnapshotError, match="different training inputs"):
        read_training_snapshot(tmp_path, expected_source_fingerprint="dataset-B")


def test_interrupted_publication_keeps_previous_recovery_point(tmp_path):
    write_gpt(tmp_path)
    before = (tmp_path / "latest.json").read_bytes()
    with pytest.raises(RuntimeError, match="simulated crash"):
        with commit_training_snapshot(tmp_path, metadata={
            "stage": "gpt", "epoch": 2, "global_step": 6,
            "job_id": "job-test", "source_fingerprint": "dataset-A",
        }) as stage:
            (stage / "half-written.ckpt").write_bytes(b"partial")
            raise RuntimeError("simulated crash")
    assert (tmp_path / "latest.json").read_bytes() == before
    assert read_training_snapshot(tmp_path)[1]["metadata"]["epoch"] == 1


def test_partial_framework_state_is_never_committed(tmp_path):
    with pytest.raises(TrainingSnapshotError, match="incomplete"):
        with commit_training_snapshot(tmp_path, metadata={
            "stage": "sovits", "epoch": 1, "global_step": 3,
            "job_id": "job-test", "source_fingerprint": "dataset-A",
        }) as stage:
            (stage / "sovits").mkdir()
            (stage / "sovits/G.pth").write_bytes(b"only generator")
    assert not (tmp_path / "latest.json").exists()


def test_corrupt_checkpoint_rejected_before_framework_load(tmp_path):
    write_gpt(tmp_path)
    root, _ = read_training_snapshot(tmp_path)
    (root / "gpt/trainer.ckpt").write_bytes(b"corrupted")
    with pytest.raises(TrainingSnapshotError, match="checksum"):
        read_training_snapshot(tmp_path)


def test_symlink_checkpoint_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    with pytest.raises(TrainingSnapshotError, match="symbolic"):
        with commit_training_snapshot(tmp_path / "snapshots", metadata={
            "stage": "gpt", "epoch": 1, "global_step": 3,
            "job_id": "job-test", "source_fingerprint": "dataset-A",
        }) as stage:
            (stage / "link").symlink_to(outside)


def test_pointer_cannot_escape_snapshot_root(tmp_path):
    write_gpt(tmp_path)
    (tmp_path / "latest.json").write_text(json.dumps({
        "schema": "aniflive-training-snapshot-v1", "snapshot_id": "../outside",
        "manifest_sha256": "0" * 64,
    }))
    with pytest.raises(TrainingSnapshotError, match="pointer"):
        read_training_snapshot(tmp_path)


def test_unlisted_checkpoint_cannot_be_selected_by_a_glob(tmp_path):
    write_gpt(tmp_path)
    root, _ = read_training_snapshot(tmp_path)
    (root / "gpt/epoch=999-step=999.ckpt").write_bytes(b"unlisted checkpoint")
    with pytest.raises(TrainingSnapshotError, match="unlisted"):
        read_training_snapshot(tmp_path)


def test_resume_restores_resolved_budget_without_recomputing(tmp_path, monkeypatch):
    from dataclasses import asdict, replace
    from aniflive_tts import workstation_training as training

    requested = training.resolve_training_plan(
        {"preset": "balanced", "experiment_name": "recovery"},
        project_id="training-recovery",
    )
    resolved, budget = training.plan_training_budget(
        requested, duration_seconds=3600, clips=900,
    )
    assert resolved.gpt_epochs < requested.gpt_epochs
    with commit_training_snapshot(tmp_path, metadata={
        "stage": "gpt", "epoch": 1, "global_step": 3,
        "job_id": "job-test", "source_fingerprint": "same-training-inputs",
    }) as stage:
        (stage / "gpt").mkdir()
        (stage / "gpt/trainer.ckpt").write_bytes(b"framework state")
        (stage / "gpt/random-state.pt").write_bytes(b"random state")
        (stage / "training-plan.json").write_text(json.dumps({
            "schema": "aniflive-training-resume-plan-v1",
            "requested_plan": asdict(requested), "resolved_plan": asdict(resolved),
        }))
        (stage / "training-budget.json").write_text(json.dumps(budget))
    def forbidden(*args, **kwargs):
        raise AssertionError("Resume must not recompute an adaptive budget")
    monkeypatch.setattr(training, "plan_training_budget", forbidden)
    restored, resumed_budget, directory = training.restore_snapshot_budget(
        tmp_path, requested, source_fingerprint="same-training-inputs",
    )
    assert restored == resolved
    assert resumed_budget["stages"] == budget["stages"]
    assert resumed_budget["resume_source"]["budget_recomputed"] is False
    assert json.loads((directory / "training-budget.json").read_text())["resumed"] is False
    with pytest.raises(training.TrainingWorkerError, match="settings changed"):
        training.restore_snapshot_budget(
            tmp_path, replace(requested, seed=requested.seed + 1),
            source_fingerprint="same-training-inputs",
        )

def test_retains_two_published_snapshots_and_leaves_unknown_directories(tmp_path):
    write_gpt(tmp_path, epoch=1)
    first, _ = read_training_snapshot(tmp_path)
    unknown = tmp_path / ("checkpoint-" + "a" * 32)
    unknown.mkdir()
    (unknown / "evidence.txt").write_text("unpublished diagnostic")
    write_gpt(tmp_path, epoch=2)
    second, _ = read_training_snapshot(tmp_path)
    assert first.exists() and second.exists()
    write_gpt(tmp_path, epoch=3)
    third, _ = read_training_snapshot(tmp_path)
    assert not first.exists()
    assert second.exists() and third.exists()
    assert (unknown / "evidence.txt").read_text() == "unpublished diagnostic"
    pointer = json.loads((tmp_path / "latest.json").read_text())
    assert pointer["retained"][0]["snapshot_id"] == second.name


def test_bad_retention_history_cannot_delete_outside_root(tmp_path):
    write_gpt(tmp_path)
    pointer = json.loads((tmp_path / "latest.json").read_text())
    pointer["retained"] = [{"snapshot_id": "../outside", "manifest_sha256": "0" * 64}]
    (tmp_path / "latest.json").write_text(json.dumps(pointer))
    before = (tmp_path / "latest.json").read_bytes()
    with pytest.raises(TrainingSnapshotError, match="retained history"):
        write_gpt(tmp_path, epoch=2)
    assert (tmp_path / "latest.json").read_bytes() == before


def test_failed_new_snapshot_does_not_prune_last_two(tmp_path):
    write_gpt(tmp_path, epoch=1)
    first, _ = read_training_snapshot(tmp_path)
    write_gpt(tmp_path, epoch=2)
    second, _ = read_training_snapshot(tmp_path)
    with pytest.raises(RuntimeError, match="interrupted"):
        with commit_training_snapshot(tmp_path, metadata={
            "stage": "gpt", "epoch": 3, "global_step": 9,
            "job_id": "job-test", "source_fingerprint": "dataset-A",
        }) as pending:
            (pending / "partial").write_bytes(b"incomplete")
            raise RuntimeError("interrupted")
    assert first.exists() and second.exists()
    assert read_training_snapshot(tmp_path)[0] == second

def test_retention_preserves_modified_old_bundle_for_diagnosis(tmp_path):
    write_gpt(tmp_path, epoch=1)
    first, _ = read_training_snapshot(tmp_path)
    write_gpt(tmp_path, epoch=2)
    (first / "gpt/trainer.ckpt").write_bytes(b"unexpected modification")
    write_gpt(tmp_path, epoch=3)
    assert first.exists()
    assert read_training_snapshot(tmp_path)[1]["metadata"]["epoch"] == 3

def test_completed_gpt_stage_requires_its_candidates_in_sovits_snapshot(tmp_path):
    with pytest.raises(TrainingSnapshotError, match="missing its deployable"):
        with commit_training_snapshot(tmp_path, metadata={
            "stage": "sovits", "completed_stages": ["gpt"], "epoch": 1, "global_step": 1,
            "job_id": "job-test", "source_fingerprint": "dataset-A",
        }) as pending:
            (pending / "sovits").mkdir()
            for name in ("G.pth", "D.pth", "runtime-state.pt"):
                (pending / "sovits" / name).write_bytes(b"framework state")
            (pending / "training-plan.json").write_text("{}")
            (pending / "training-budget.json").write_text("{}")
    assert not (tmp_path / "latest.json").exists()
