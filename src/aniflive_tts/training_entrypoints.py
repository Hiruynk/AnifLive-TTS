"""Stage recovery hooks against the pinned upstream training entrypoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class TrainingEntrypointError(RuntimeError):
    pass


def _replace_once(text: str, marker: str, replacement: str) -> str:
    if text.count(marker) != 1:
        raise TrainingEntrypointError("Pinned training entrypoint contract changed")
    return text.replace(marker, replacement, 1)


def stage_recovery_entrypoint(source: Path, output: Path, stage: str) -> Path:
    """Copy, instrument, and compile upstream code; never modify its source tree.

    The parent must supply a verified recovery context and resume checkpoint.
    These runners support the workstation's single visible GPU contract.
    """
    filename = {"gpt": "s1_train.py", "sovits": "s2_train.py"}.get(stage)
    if filename is None:
        raise TrainingEntrypointError("Unknown recovery stage")
    original = source / "GPT_SoVITS" / filename
    if original.is_symlink() or not original.is_file():
        raise TrainingEntrypointError("Training entrypoint must be a regular file")
    raw = original.read_bytes()
    text = raw.decode("utf-8")
    hook = (
        "from aniflive_tts.training_recovery import (\n"
        "    install_epoch_loader_policy, gpt_recovery_callback,\n"
        "    finish_gpt_checkpoint_boundary, restore_sovits_runtime,\n"
        "    save_sovits_boundary, require_resume_stage, seed_training_randomness,\n"
        "    install_sovits_dataset_order_policy, install_gpt_dataloader_epoch_policy,\n"
        ")\n"
        "install_epoch_loader_policy()\n"
    )
    if stage == "gpt":
        hook += "install_gpt_dataloader_epoch_policy()\n"
    text = _replace_once(text, "import torch\n", "import torch\n" + hook)
    if stage == "gpt":
        text = _replace_once(
            text, "        callbacks=[ckpt_callback],",
            "        callbacks=[ckpt_callback, gpt_recovery_callback(managed_boundary=True)],",
        )
        text = _replace_once(
            text, "            self._save_last_checkpoint(trainer, monitor_candidates)\n",
            "            self._save_last_checkpoint(trainer, monitor_candidates)\n"
            "        finish_gpt_checkpoint_boundary(trainer, pl_module)\n",
        )
        text = _replace_once(
            text, '    print("ckpt_path:", ckpt_path)\n',
            '    if require_resume_stage("gpt") and ckpt_path is None:\n'
            '        raise RuntimeError("Required GPT recovery checkpoint is missing")\n'
            '    print("ckpt_path:", ckpt_path)\n',
        )
    else:
        text = _replace_once(
            text, "    torch.manual_seed(hps.train.seed)\n",
            "    seed_training_randomness(hps.train.seed)\n"
            "    install_sovits_dataset_order_policy(hps.train.seed)\n",
        )
        text = _replace_once(
            text, "def run(rank, n_gpus, hps):\n    global global_step\n",
            "def run(rank, n_gpus, hps):\n    global global_step\n"
            "    if n_gpus != 1:\n"
            '        raise RuntimeError("Training recovery requires one visible GPU")\n',
        )
        text = _replace_once(
            text, "    except:  # 如果首次不能加载，加载pretrain\n",
            "    except Exception:  # Initial training may load pretrained weights.\n"
            '        if require_resume_stage("sovits"):\n'
            "            raise\n",
        )
        marker = "        _, _, _, epoch_str = utils.load_checkpoint("
        if text.count(marker) != 2:
            raise TrainingEntrypointError("Pinned SoVITS G/D load contract changed")
        text = text.replace(marker, "        _, _, _, discriminator_epoch = utils.load_checkpoint(", 1)
        text = _replace_once(
            text, "        epoch_str += 1\n",
            "        if discriminator_epoch != epoch_str:\n"
            '            raise RuntimeError("SoVITS G/D recovery epochs differ")\n'
            "        epoch_str += 1\n",
        )
        text = _replace_once(
            text, "    scaler = GradScaler(enabled=hps.train.fp16_run)\n",
            "    scaler = GradScaler(enabled=hps.train.fp16_run)\n"
            "    restored = restore_sovits_runtime([optim_g, optim_d], [scheduler_g, scheduler_d], scaler)\n"
            "    if restored is not None:\n"
            '        if epoch_str != restored["epoch"] + 1:\n'
            '            raise RuntimeError("SoVITS model and runtime recovery epochs differ")\n'
            '        global_step = restored["global_step"]\n',
        )
        text = _replace_once(
            text, '        scheduler_d.step()\n    print("training done")',
            "        scheduler_d.step()\n"
            "        if save_sovits_boundary(epoch, global_step, [net_g, net_d],\n"
            "                                [optim_g, optim_d], [scheduler_g, scheduler_d], scaler):\n"
            "            break\n"
            '    print("training done")',
        )
    compile(text, filename, "exec")
    target_root = output / "recovery-runners"
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / filename
    target.write_text(text, encoding="utf-8")
    target.with_suffix(".source.json").write_text(json.dumps({
        "schema": "aniflive-training-recovery-runner-v1",
        "stage": stage,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "runner_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "worker_rng_policy": "epoch-workers-spawn-v1",
    }, indent=2) + "\n", encoding="utf-8")
    return target
