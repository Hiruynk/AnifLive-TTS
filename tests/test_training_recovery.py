import json
from dataclasses import asdict
from pathlib import Path

import pytest
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from aniflive_tts.training_recovery import gpt_recovery_callback, install_epoch_loader_policy
from aniflive_tts.training_snapshot import read_training_snapshot
from aniflive_tts.workstation_training import plan_training_budget, resolve_training_plan


class RandomDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return torch.tensor([float(index), float(np.random.uniform())]), torch.tensor([float(index) * 0.5])


class TinyTrainingModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(2, 1))

    def training_step(self, batch, batch_idx):
        features, targets = batch
        return torch.nn.functional.mse_loss(self.network(features), targets)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return {"optimizer": optimizer,
                "lr_scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)}


class ManualAccumulationModel(TinyTrainingModel):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        features, targets = batch
        loss = torch.nn.functional.mse_loss(self.network(features), targets)
        self.manual_backward(loss)
        if batch_idx > 0 and batch_idx % 4 == 0:
            optimizer = self.optimizers()
            optimizer.step()
            optimizer.zero_grad()
            self.lr_schedulers().step()
        return loss


class TailDataset(RandomDataset):
    def __len__(self):
        return 12
    def get_sample_length(self, index):
        return 1.0 + index * 0.01

    @staticmethod
    def collate(items):
        from torch.utils.data import default_collate
        return default_collate(items)


def context_file(root: Path, *, pause=False, resume=False, stage="gpt"):
    root.mkdir()
    plan = resolve_training_plan({
        "preset": "advanced", "stage": stage, "gpt_epochs": 3, "sovits_epochs": 3,
        "experiment_name": "recovery-test",
    }, project_id="training-test")
    resolved, budget = plan_training_budget(plan, duration_seconds=30, clips=8)
    context = {
        "schema": "aniflive-training-recovery-context-v1",
        "job_id": "test-job", "run_id": "test-run", "source_fingerprint": "toy-training",
        "snapshot_root": str(root / "snapshots"), "control_path": str(root / "control.json"),
        "pause_ack_path": str(root / "pause-ack.json"),
        "requested_plan": asdict(plan), "resolved_plan": asdict(resolved), "budget": budget,
        "resume_stage": stage if resume else None,
    }
    path = root / "context.json"
    path.write_text(json.dumps(context))
    if pause:
        Path(context["control_path"]).write_text(json.dumps({
            "schema": "aniflive-training-control-v1", "action": "pause",
            "job_id": "test-job", "run_id": "test-run", "request_id": "test-request",
        }))
    return path


def train(context, monkeypatch, checkpoint=None, managed=False, manual=False, native_sampler=False):
    monkeypatch.setenv("ANIFLIVE_TTS_TRAINING_RECOVERY_CONTEXT", str(context))
    pl.seed_everything(1234, workers=True)
    model = ManualAccumulationModel() if manual else TinyTrainingModel()
    class UpstreamBoundary(pl.callbacks.ModelCheckpoint):
        def on_train_epoch_end(self, trainer, pl_module):
            from aniflive_tts.training_recovery import finish_gpt_checkpoint_boundary
            finish_gpt_checkpoint_boundary(trainer, pl_module)

    callbacks = [gpt_recovery_callback(managed_boundary=managed)]
    if managed:
        callbacks.insert(0, UpstreamBoundary(dirpath=context.parent / "upstream"))
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=3, logger=False,
        enable_checkpointing=managed, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0,
        callbacks=callbacks, deterministic=True,
    )
    if native_sampler:
        from AR.data.data_module import Text2SemanticDataModule
        def setup(module, stage=None, **kwargs):
            module._train_dataset = TailDataset()
            module._dev_dataset = module._train_dataset
        monkeypatch.setattr(Text2SemanticDataModule, "setup", setup)
        data = Text2SemanticDataModule(
            {"data":{"num_workers":2},"train":{"batch_size":2}}, "semantic.tsv", "phoneme.txt",
        )
        trainer.fit(model, datamodule=data, ckpt_path=checkpoint)
    else:
        trainer.fit(model, DataLoader(TailDataset() if manual else RandomDataset(), batch_size=2, shuffle=True,
                                     num_workers=2, persistent_workers=True),
                    ckpt_path=checkpoint)
    return trainer, model


@pytest.fixture(autouse=True)
def restore_thread_count():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("managed", [False, True])
def test_cpu_training_pause_resume_preserves_optimizer_scheduler_and_randomness(tmp_path, monkeypatch, managed):
    from aniflive_tts import training_recovery
    monkeypatch.setattr(torch.utils.data.DataLoader, "__init__", torch.utils.data.DataLoader.__init__)
    monkeypatch.setattr(training_recovery, "_loader_policy_installed", False)
    install_epoch_loader_policy()
    baseline_context = context_file(tmp_path / "baseline")
    baseline_trainer, baseline_model = train(baseline_context, monkeypatch, managed=managed)
    paused_context = context_file(tmp_path / "paused", pause=True)
    paused_trainer, _ = train(paused_context, monkeypatch, managed=managed)
    assert paused_trainer.global_step == 4
    directory, metadata = read_training_snapshot(tmp_path / "paused/snapshots")
    assert metadata["metadata"]["epoch"] == 1
    assert (tmp_path / "paused/pause-ack.json").is_file()
    resumed_context = context_file(tmp_path / "resumed", resume=True)
    resumed_trainer, resumed_model = train(
        resumed_context, monkeypatch, checkpoint=directory / "gpt/trainer.ckpt", managed=managed,
    )
    assert resumed_trainer.global_step == baseline_trainer.global_step == 12
    for key, value in baseline_model.state_dict().items():
        torch.testing.assert_close(value, resumed_model.state_dict()[key], rtol=0, atol=0)
    assert resumed_trainer.optimizers[0].param_groups[0]["lr"] == baseline_trainer.optimizers[0].param_groups[0]["lr"]


def test_sovits_runtime_restores_lr_scheduler_amp_and_rng(tmp_path, monkeypatch):
    import random
    from aniflive_tts.training_recovery import save_sovits_boundary, restore_sovits_runtime

    from aniflive_tts import training_recovery
    monkeypatch.setattr(training_recovery, "_dataset_order_fingerprint", "f" * 64)
    path = context_file(tmp_path / "sovits", stage="sovits")
    context = json.loads(path.read_text())
    candidate = tmp_path / "completed-gpt.ckpt"
    candidate.write_bytes(b"completed GPT candidate")
    context.update(completed_stages=["gpt"], carry_files={"candidates/gpt/epoch-3.ckpt": str(candidate)})
    path.write_text(json.dumps(context))
    monkeypatch.setenv("ANIFLIVE_TTS_TRAINING_RECOVERY_CONTEXT", str(path))
    learning_rate = context["resolved_plan"]["sovits_learning_rate"]
    torch.manual_seed(42)
    models = [torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)]
    optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate) for model in models]
    schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
                  for optimizer in optimizers]
    scaler = torch.amp.GradScaler("cpu")
    for model, optimizer in zip(models, optimizers, strict=True):
        optimizer.zero_grad()
        loss = model(torch.ones(1, 2)).square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
    scaler.update()
    for scheduler in schedulers:
        scheduler.step()
    expected_lrs = [optimizer.param_groups[0]["lr"] for optimizer in optimizers]
    expected_schedulers = [scheduler.state_dict() for scheduler in schedulers]
    expected_scaler = scaler.state_dict()
    assert save_sovits_boundary(1, 1, models, optimizers, schedulers, scaler) is False
    expected_random = (torch.rand(4), np.random.rand(4), random.random())
    directory, _ = read_training_snapshot(Path(context["snapshot_root"]))
    candidate.write_bytes(b"changed after snapshot")
    assert (directory / "candidates/gpt/epoch-3.ckpt").read_bytes() == b"completed GPT candidate"

    restored_models = [torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)]
    restored_optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate)
                           for model in restored_models]
    for name, model, optimizer in zip(("G", "D"), restored_models, restored_optimizers, strict=True):
        data = torch.load(directory / "sovits" / (name + ".pth"), weights_only=False)
        model.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optimizer"])
    restored_schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
                           for optimizer in restored_optimizers]
    # Simulate the legacy bootstrap that advances an already-restored LR again.
    for scheduler in restored_schedulers:
        scheduler.step()
        scheduler.step()
    restored_scaler = torch.amp.GradScaler("cpu", init_scale=8)
    context.update(resume_stage="sovits", resume_directory=str(directory))
    path.write_text(json.dumps(context))
    result = restore_sovits_runtime(restored_optimizers, restored_schedulers, restored_scaler)
    assert result == {"epoch": 1, "global_step": 1}
    assert [optimizer.param_groups[0]["lr"] for optimizer in restored_optimizers] == expected_lrs
    assert [scheduler.state_dict() for scheduler in restored_schedulers] == expected_schedulers
    assert restored_scaler.state_dict() == expected_scaler
    torch.testing.assert_close(torch.rand(4), expected_random[0], rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.rand(4), expected_random[1])
    assert random.random() == expected_random[2]


def test_native_sovits_initializes_python_numpy_and_torch_rng():
    import random
    from aniflive_tts.training_recovery import capture_rng, restore_rng, seed_training_randomness
    previous = capture_rng()
    try:
        seed_training_randomness(1234)
        expected = (random.random(), np.random.rand(4), torch.rand(4))
        seed_training_randomness(1234)
        assert random.random() == expected[0]
        np.testing.assert_array_equal(np.random.rand(4), expected[1])
        torch.testing.assert_close(torch.rand(4), expected[2], rtol=0, atol=0)
    finally:
        restore_rng(previous)

def test_carry_files_cannot_replace_framework_or_escape_bundle(tmp_path):
    from aniflive_tts.training_recovery import _common_files, TrainingRecoveryError
    source = tmp_path / "source"
    source.write_bytes(b"candidate")
    for name in ("../outside", "gpt/trainer.ckpt", "/absolute", "assets"):
        with pytest.raises(TrainingRecoveryError, match="unsafe"):
            _common_files(tmp_path, {"carry_files": {name: str(source)}})


def test_manual_accumulation_tail_survives_epoch_boundary_resume(tmp_path, monkeypatch):
    from aniflive_tts import training_recovery
    monkeypatch.setattr(torch.utils.data.DataLoader, "__init__", torch.utils.data.DataLoader.__init__)
    monkeypatch.setattr(training_recovery, "_loader_policy_installed", False)
    install_epoch_loader_policy()
    baseline_trainer, baseline_model = train(context_file(tmp_path / "manual-baseline"), monkeypatch,
                                            managed=True, manual=True)
    paused_trainer, _ = train(context_file(tmp_path / "manual-paused", pause=True), monkeypatch,
                             managed=True, manual=True)
    directory, _ = read_training_snapshot(tmp_path / "manual-paused/snapshots")
    saved = torch.load(directory / "gpt/trainer.ckpt", weights_only=False)
    state = saved["callbacks"]["aniflive-training-recovery-v1"]
    assert state["schema"] == "aniflive-training-rng-v2"
    assert any(value is not None and torch.count_nonzero(value) for value in state["pending_gradients"].values())
    resumed_trainer, resumed_model = train(context_file(tmp_path / "manual-resumed", resume=True), monkeypatch,
                                          checkpoint=directory / "gpt/trainer.ckpt", managed=True, manual=True)
    assert paused_trainer.global_step == 1
    assert resumed_trainer.global_step == baseline_trainer.global_step == 3
    for key, value in baseline_model.state_dict().items():
        torch.testing.assert_close(value, resumed_model.state_dict()[key], rtol=0, atol=0)

def test_manual_resume_rejects_legacy_checkpoint_without_pending_gradients(tmp_path, monkeypatch):
    from aniflive_tts.training_recovery import capture_rng, TrainingRecoveryError
    context = context_file(tmp_path / "legacy-manual", resume=True)
    monkeypatch.setenv("ANIFLIVE_TTS_TRAINING_RECOVERY_CONTEXT", str(context))
    callback = gpt_recovery_callback()
    callback.load_state_dict({"schema":"aniflive-training-rng-v1","rng":capture_rng()})
    with pytest.raises(TrainingRecoveryError, match="accumulated gradient state"):
        callback.on_train_start(None, ManualAccumulationModel())

def test_sovits_canonical_order_is_independent_of_initial_set_iteration():
    from types import SimpleNamespace
    import random
    from aniflive_tts.training_recovery import _canonicalize_sovits_dataset
    first=SimpleNamespace(audiopaths_sid_text=[["b",[2]],["a",[1]],["b",[2]]],lengths=[20,10,20])
    second=SimpleNamespace(audiopaths_sid_text=[["b",[2]],["b",[2]],["a",[1]]],lengths=[20,20,10])
    before=random.getstate()
    assert _canonicalize_sovits_dataset(first,1234)==_canonicalize_sovits_dataset(second,1234)
    assert first.audiopaths_sid_text==second.audiopaths_sid_text
    assert first.lengths==second.lengths
    assert random.getstate()==before

def test_native_bucket_sampler_and_manual_gradient_resume_match(tmp_path, monkeypatch):
    from aniflive_tts import training_recovery
    source=Path("/opt/aniflive-tts/gpt-sovits")
    if not source.exists():
        pytest.skip("Requires native training source")
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.syspath_prepend(str(source/"GPT_SoVITS"))
    from AR.data.data_module import Text2SemanticDataModule
    monkeypatch.setattr(Text2SemanticDataModule,"train_dataloader",Text2SemanticDataModule.train_dataloader)
    monkeypatch.setattr(training_recovery,"_gpt_loader_epoch_installed",False)
    monkeypatch.setattr(torch.utils.data.DataLoader,"__init__",torch.utils.data.DataLoader.__init__)
    monkeypatch.setattr(training_recovery,"_loader_policy_installed",False)
    install_epoch_loader_policy()
    training_recovery.install_gpt_dataloader_epoch_policy()
    baseline,_model=train(context_file(tmp_path/"native-baseline"),monkeypatch,managed=True,manual=True,native_sampler=True)
    train(context_file(tmp_path/"native-paused",pause=True),monkeypatch,managed=True,manual=True,native_sampler=True)
    directory,_=read_training_snapshot(tmp_path/"native-paused/snapshots")
    resumed,resumed_model=train(context_file(tmp_path/"native-resumed",resume=True),monkeypatch,
                                checkpoint=directory/"gpt/trainer.ckpt",managed=True,manual=True,native_sampler=True)
    assert baseline.global_step==resumed.global_step==3
    for key,value in _model.state_dict().items():
        torch.testing.assert_close(value,resumed_model.state_dict()[key],rtol=0,atol=0)
