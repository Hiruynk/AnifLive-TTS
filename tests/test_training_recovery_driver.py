import json
import tarfile
from pathlib import Path
from aniflive_tts.training_recovery_driver import prepare_training_recovery

def prepare(root, text="same"):
    root.mkdir()
    dataset=root/"dataset";dataset.mkdir()
    (dataset/"features").mkdir()
    (dataset/"features/a.pt").write_bytes(text.encode())
    (dataset/"labels.txt").write_text("sample")
    weight=root/"weight.pth";weight.write_bytes(b"weights")
    output=root/"output";output.mkdir()
    return prepare_training_recovery(
        manifest={"job_id":"job-test"},output=output,dataset=dataset,
        dataset_entries=("features","labels.txt"),
        requested_plan={"epochs":20},resolved_plan={"epochs":12},
        budget={"resolved_epochs":12},pretrained_paths={"gpt":weight},source_revision="pinned",
    )

def test_recovery_identity_is_portable_and_preserves_original_budget_and_data(tmp_path):
    first=prepare(tmp_path/"one")
    second=prepare(tmp_path/"two")
    assert first["source_fingerprint"]==second["source_fingerprint"]
    assert first["requested_plan"]["epochs"]==20
    assert first["resolved_plan"]["epochs"]==12
    with tarfile.open(first["carry_files"]["assets/dataset.tar"]) as archive:
        assert set(archive.getnames())=={"features","features/a.pt","labels.txt"}
        assert archive.extractfile("features/a.pt").read()==b"same"

def test_changed_training_features_change_recovery_identity(tmp_path):
    assert prepare(tmp_path/"one")["source_fingerprint"]!=prepare(tmp_path/"two",text="changed")["source_fingerprint"]

def test_restored_archive_has_identical_feature_bytes(tmp_path):
    import shutil
    from aniflive_tts.training_recovery_driver import restore_recovery_dataset
    context = prepare(tmp_path / "original")
    bundle = tmp_path / "bundle"
    (bundle / "assets").mkdir(parents=True)
    shutil.copy2(context["carry_files"]["assets/dataset.tar"], bundle / "assets/dataset.tar")
    restored = restore_recovery_dataset(bundle, tmp_path / "restored")
    assert (restored / "features/a.pt").read_bytes() == b"same"
    assert (restored / "labels.txt").read_text() == "sample"


def test_archive_restore_rejects_traversal_and_links(tmp_path):
    import io
    import pytest
    from aniflive_tts.training_recovery_driver import restore_recovery_dataset
    for index, kind in enumerate(("traversal", "symlink")):
        bundle = tmp_path / str(index)
        (bundle / "assets").mkdir(parents=True)
        with tarfile.open(bundle / "assets/dataset.tar", "w") as archive:
            item = tarfile.TarInfo("../outside" if kind == "traversal" else "link")
            if kind == "symlink":
                item.type = tarfile.SYMTYPE
                item.linkname = "/tmp/outside"
                archive.addfile(item)
            else:
                item.size = 1
                archive.addfile(item, io.BytesIO(b"x"))
        with pytest.raises(RuntimeError, match="unsafe"):
            restore_recovery_dataset(bundle, tmp_path / ("restore-" + kind))
    assert not (tmp_path / "outside").exists()


def test_materializes_native_checkpoint_and_prior_candidates(tmp_path):
    from aniflive_tts.training_recovery_driver import materialize_recovery_checkpoint
    bundle = tmp_path / "bundle"
    (bundle / "gpt").mkdir(parents=True)
    (bundle / "gpt/trainer.ckpt").write_bytes(b"optimizer and trainer")
    (bundle / "candidates/gpt").mkdir(parents=True)
    (bundle / "candidates/gpt/e2.ckpt").write_bytes(b"candidate")
    output = tmp_path / "output"
    materialize_recovery_checkpoint(bundle, {"stage":"gpt","epoch":2,"global_step":16},
                                    output=output,dataset_work=tmp_path / "dataset")
    assert (output / "work/gpt/ckpt/epoch=1-step=16.ckpt").read_bytes() == b"optimizer and trainer"
    assert (output / "checkpoints/gpt/e2.ckpt").read_bytes() == b"candidate"
