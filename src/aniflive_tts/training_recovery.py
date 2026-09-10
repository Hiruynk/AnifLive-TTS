"""Framework hooks for checkpoint-boundary training recovery in Docker workers."""
from __future__ import annotations

import json
import hashlib
import os
import random
import shutil
from pathlib import Path, PurePosixPath
from uuid import uuid4

import numpy as np
import torch

from .training_snapshot import commit_training_snapshot, read_training_snapshot

_CONTEXT_ENV = "ANIFLIVE_TTS_TRAINING_RECOVERY_CONTEXT"
_loader_policy_installed = False
_dataset_order_fingerprint = None
_sovits_order_installed = False
_gpt_loader_epoch_installed = False


class TrainingRecoveryError(RuntimeError):
    pass


def seed_training_randomness(seed):
    """Initialize all host RNGs used by native SoVITS before its first epoch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise TrainingRecoveryError("Saved CUDA RNG state requires CUDA")
        torch.cuda.set_rng_state_all(state["cuda"])


def install_epoch_loader_policy():
    """New workers per epoch make worker-local RNG reproducible at resume boundaries."""
    global _loader_policy_installed
    if _loader_policy_installed:
        return
    original = torch.utils.data.DataLoader.__init__
    def initialize(self, *args, **kwargs):
        kwargs["persistent_workers"] = False
        workers = kwargs.get("num_workers", args[5] if len(args) > 5 else 0)
        if workers:
            kwargs["multiprocessing_context"] = "spawn"
        return original(self, *args, **kwargs)
    torch.utils.data.DataLoader.__init__ = initialize
    _loader_policy_installed = True


def install_gpt_dataloader_epoch_policy():
    global _gpt_loader_epoch_installed
    if _gpt_loader_epoch_installed:
        return
    from AR.data.data_module import Text2SemanticDataModule
    original = Text2SemanticDataModule.train_dataloader
    def create(module, *args, **kwargs):
        loader = original(module, *args, **kwargs)
        # Lightning eagerly creates the first iterator before on_advance_start.
        # Its saved processed counter is the epoch that iterator must sample.
        epoch = int(module.trainer.fit_loop.epoch_progress.current.processed)
        loader.sampler.set_epoch(epoch)
        return loader
    Text2SemanticDataModule.train_dataloader = create
    _gpt_loader_epoch_installed = True


def _canonicalize_sovits_dataset(dataset, seed):
    rows = sorted(
        zip(dataset.audiopaths_sid_text, dataset.lengths, strict=True),
        key=lambda pair: (pair[0][0], tuple(pair[0][1]), int(pair[1])),
    )
    random.Random(seed).shuffle(rows)
    dataset.audiopaths_sid_text = [item for item, _ in rows]
    dataset.lengths = [length for _, length in rows]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def install_sovits_dataset_order_policy(seed):
    global _sovits_order_installed
    if _sovits_order_installed:
        return
    from module.data_utils import TextAudioSpeakerLoader
    original = TextAudioSpeakerLoader.__init__
    def initialize(dataset, *args, **kwargs):
        global _dataset_order_fingerprint
        original(dataset, *args, **kwargs)
        fingerprint = _canonicalize_sovits_dataset(dataset, seed)
        if not getattr(dataset, "val", False):
            _dataset_order_fingerprint = fingerprint
    TextAudioSpeakerLoader.__init__ = initialize
    _sovits_order_installed = True


def _context():
    path = Path(os.environ[_CONTEXT_ENV])
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise TrainingRecoveryError("Training recovery context is invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "aniflive-training-recovery-context-v1":
        raise TrainingRecoveryError("Training recovery context schema is invalid")
    return value


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "-" + uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _common_files(directory, context):
    carry = context.get("carry_files", {})
    if not isinstance(carry, dict):
        raise TrainingRecoveryError("Recovery carry files must be a mapping")
    carry = dict(carry)
    candidate_root = context.get("candidate_root")
    if candidate_root is not None:
        for stage, pattern in (("gpt", "*.ckpt"), ("sovits", "*.pth")):
            for candidate in sorted((Path(candidate_root) / stage).glob(pattern)):
                carry["candidates/" + stage + "/" + candidate.name] = str(candidate)
    if not isinstance(carry, dict):
        raise TrainingRecoveryError("Recovery carry files must be a mapping")
    for relative, source in carry.items():
        if not isinstance(relative, str) or not isinstance(source, str):
            raise TrainingRecoveryError("Recovery carry file is invalid")
        name = PurePosixPath(relative)
        if (name.is_absolute() or ".." in name.parts or "\\" in relative
                or len(name.parts) < 2 or name.parts[0] not in {"assets", "candidates"}):
            raise TrainingRecoveryError("Recovery carry file path is unsafe")
        path = Path(source)
        if path.is_symlink() or not path.is_file():
            raise TrainingRecoveryError("Recovery carry source must be a regular file")
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (directory / "training-plan.json").write_text(json.dumps({
        "schema": "aniflive-training-resume-plan-v1",
        "requested_plan": context["requested_plan"],
        "resolved_plan": context["resolved_plan"],
    }, sort_keys=True), encoding="utf-8")
    (directory / "training-budget.json").write_text(
        json.dumps(context["budget"], sort_keys=True), encoding="utf-8",
    )


def _metadata(context, stage, epoch, step):
    return {
        "job_id": context["job_id"], "source_fingerprint": context["source_fingerprint"],
        "stage": stage, "epoch": int(epoch), "global_step": int(step),
        "completed_stages": context.get("completed_stages", []),
        "worker_rng_policy": "epoch-workers-spawn-v1",
        **({"dataset_order_policy": "canonical-valid-items-shuffle-v1",
            "dataset_order_sha256": _dataset_order_fingerprint} if stage == "sovits" else {}),
    }


def _pause_if_requested(context):
    path = Path(context["control_path"])
    if not path.exists():
        return False
    if path.is_symlink() or path.stat().st_size > 4096:
        raise TrainingRecoveryError("Training pause request is invalid")
    request = json.loads(path.read_text(encoding="utf-8"))
    if (request.get("schema") != "aniflive-training-control-v1"
            or request.get("action") != "pause"
            or request.get("job_id") != context["job_id"]
            or request.get("run_id") != context["run_id"]):
        raise TrainingRecoveryError("Training pause request identity does not match")
    root = Path(context["snapshot_root"])
    directory, manifest = read_training_snapshot(
        root, expected_source_fingerprint=context["source_fingerprint"],
    )
    _atomic_json(context["pause_ack_path"], {
        "schema": "aniflive-training-pause-ack-v1",
        "job_id": context["job_id"], "run_id": context["run_id"],
        "request_id": request.get("request_id"), "snapshot_id": directory.name,
        "stage": manifest["metadata"]["stage"],
        "epoch": manifest["metadata"]["epoch"],
        "global_step": manifest["metadata"]["global_step"],
    })
    return True


def gpt_recovery_callback(*, managed_boundary=False):
    from pytorch_lightning.callbacks import Callback

    context = _context()
    class RecoveryCallback(Callback):
        @property
        def state_key(self):
            return "aniflive-training-recovery-v1"

        def __init__(self):
            self.restored = context.get("resume_stage") != "gpt"
            self.module = None
            self.pending_gradients = None
            self.saved_manual_optimization = None

        def state_dict(self):
            manual = self.module is not None and not self.module.automatic_optimization
            gradients = None
            if manual:
                gradients = {
                    name: parameter.grad.detach().cpu() if parameter.grad is not None else None
                    for name, parameter in self.module.named_parameters()
                }
            return {"schema": "aniflive-training-rng-v2", "rng": capture_rng(),
                    "manual_optimization": manual, "pending_gradients": gradients}

        def load_state_dict(self, state):
            if state.get("schema") not in {"aniflive-training-rng-v1", "aniflive-training-rng-v2"}:
                raise TrainingRecoveryError("GPT checkpoint RNG schema is invalid")
            restore_rng(state["rng"])
            self.pending_gradients = state.get("pending_gradients")
            self.saved_manual_optimization = state.get("manual_optimization")
            self.restored = True

        def on_train_start(self, trainer, pl_module):
            if not self.restored:
                raise TrainingRecoveryError("GPT checkpoint lacks verified RNG recovery state")
            self.module = pl_module
            if context.get("resume_stage") == "gpt" and not pl_module.automatic_optimization:
                gradients = self.pending_gradients
                parameters = dict(pl_module.named_parameters())
                if (self.saved_manual_optimization is not True
                        or not isinstance(gradients, dict) or gradients.keys() != parameters.keys()):
                    raise TrainingRecoveryError(
                        "GPT checkpoint lacks accumulated gradient state; refusing inexact resume"
                    )
                for name, parameter in parameters.items():
                    value = gradients[name]
                    if value is None:
                        parameter.grad = None
                    else:
                        if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
                            raise TrainingRecoveryError("GPT accumulated gradient shape/type mismatch")
                        parameter.grad = value.to(device=parameter.device)
                self.pending_gradients = None

        def on_train_epoch_end(self, trainer, pl_module):
            if not managed_boundary:
                self.checkpoint_boundary(trainer, pl_module)

        def checkpoint_boundary(self, trainer, pl_module):
            if not trainer.is_global_zero:
                return
            with commit_training_snapshot(
                Path(context["snapshot_root"]),
                metadata=_metadata(context, "gpt", trainer.current_epoch + 1, trainer.global_step),
            ) as directory:
                (directory / "gpt").mkdir()
                trainer.save_checkpoint(directory / "gpt/trainer.ckpt", weights_only=False)
                torch.save(capture_rng(), directory / "gpt/random-state.pt")
                _common_files(directory, context)
            if _pause_if_requested(context):
                trainer.should_stop = True
    return RecoveryCallback()


def save_sovits_boundary(epoch, step, models, optimizers, schedulers, scaler):
    context = _context()
    if not isinstance(_dataset_order_fingerprint, str):
        raise TrainingRecoveryError("SoVITS dataset order was not recorded")
    with commit_training_snapshot(
        Path(context["snapshot_root"]), metadata=_metadata(context, "sovits", epoch, step),
    ) as directory:
        (directory / "sovits").mkdir()
        for name, model, optimizer in zip(("G", "D"), models, optimizers, strict=True):
            module = model.module if hasattr(model, "module") else model
            torch.save({
                "model": module.state_dict(), "optimizer": optimizer.state_dict(),
                "learning_rate": context["resolved_plan"]["sovits_learning_rate"],
                "iteration": int(epoch),
            }, directory / "sovits" / (name + ".pth"))
        torch.save({
            "schema": "aniflive-sovits-runtime-state-v1",
            "dataset_order_policy": "canonical-valid-items-shuffle-v1",
            "dataset_order_sha256": _dataset_order_fingerprint,
            "epoch": int(epoch), "global_step": int(step),
            "schedulers": [scheduler.state_dict() for scheduler in schedulers],
            "optimizer_lrs": [[group["lr"] for group in optimizer.param_groups]
                              for optimizer in optimizers],
            "scaler": scaler.state_dict(), "rng": capture_rng(),
        }, directory / "sovits/runtime-state.pt")
        _common_files(directory, context)
    return _pause_if_requested(context)


def restore_sovits_runtime(optimizers, schedulers, scaler):
    context = _context()
    if context.get("resume_stage") != "sovits":
        return None
    directory = Path(context["resume_directory"])
    # The parent runner verifies the complete checksummed snapshot before launch.
    state = torch.load(directory / "sovits/runtime-state.pt", map_location="cpu", weights_only=False)
    if state.get("schema") != "aniflive-sovits-runtime-state-v1":
        raise TrainingRecoveryError("SoVITS checkpoint runtime schema is invalid")
    if (state.get("dataset_order_policy") != "canonical-valid-items-shuffle-v1"
            or _dataset_order_fingerprint is None
            or state.get("dataset_order_sha256") != _dataset_order_fingerprint):
        raise TrainingRecoveryError("SoVITS checkpoint dataset order is missing or does not match")
    for optimizer, scheduler, lrs, saved in zip(
        optimizers, schedulers, state["optimizer_lrs"], state["schedulers"], strict=True,
    ):
        scheduler.load_state_dict(saved)
        for group, value in zip(optimizer.param_groups, lrs, strict=True):
            group["lr"] = value
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    return {"epoch": state["epoch"], "global_step": state["global_step"]}


def require_resume_stage(stage):
    return _context().get("resume_stage") == stage


def finish_gpt_checkpoint_boundary(trainer, pl_module):
    """Snapshot after upstream has saved this epoch's deployable candidates."""
    callbacks = [callback for callback in trainer.callbacks
                 if getattr(callback, "state_key", None) == "aniflive-training-recovery-v1"]
    if len(callbacks) != 1:
        raise TrainingRecoveryError("GPT recovery callback is missing or duplicated")
    callbacks[0].checkpoint_boundary(trainer, pl_module)
