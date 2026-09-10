"""Create complete recovery inputs for new native training runs."""
from __future__ import annotations

import hashlib
import json
import os
import tarfile
import shutil
from pathlib import Path, PurePosixPath


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identify_training_inputs(*, dataset, dataset_entries, requested_plan, pretrained_paths, source_revision):
    records = {}
    for name in dataset_entries:
        entry = dataset / name
        paths = [entry, *entry.rglob("*")] if entry.is_dir() else [entry]
        for path in paths:
            if path.is_symlink():
                raise RuntimeError("Recovery dataset must not contain symbolic links")
            if path.is_file():
                records[path.relative_to(dataset).as_posix()] = _sha(path)
    identity = {
        "source_revision": source_revision,
        "requested_plan": requested_plan,
        "dataset": records,
        "pretrained": {name: _sha(path) for name, path in pretrained_paths.items()},
        "worker_rng_policy": "epoch-workers-spawn-v1",
    }
    return identity


def identity_fingerprint(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def prepare_training_recovery(
    *, manifest, output, dataset, dataset_entries, requested_plan, resolved_plan,
    budget, pretrained_paths, source_revision, identity=None,
):
    if identity is None:
        identity = identify_training_inputs(
            dataset=dataset, dataset_entries=dataset_entries, requested_plan=requested_plan,
            pretrained_paths=pretrained_paths, source_revision=source_revision,
        )
    fingerprint = identity_fingerprint(identity)
    asset_root = output / "recovery-assets"
    asset_root.mkdir()
    archive = asset_root / "dataset.tar"
    with tarfile.open(archive, "w") as stream:
        for name in dataset_entries:
            stream.add(dataset / name, arcname=name, recursive=True)
    context = {
        "schema": "aniflive-training-recovery-context-v1",
        "job_id": manifest["job_id"],
        "run_id": os.environ.get("ANIFLIVE_TTS_WORKER_RUN_ID", "local-training"),
        "source_fingerprint": fingerprint,
        "snapshot_root": str(output / "recovery-snapshots"),
        "control_path": str(output / "training-control.json"),
        "pause_ack_path": str(output / "training-pause-ack.json"),
        "requested_plan": requested_plan,
        "resolved_plan": resolved_plan,
        "budget": budget,
        "candidate_root": str(output / "checkpoints"),
        "carry_files": {"assets/dataset.tar": str(archive)},
        "completed_stages": [],
    }
    (asset_root / "source-identity.json").write_text(
        json.dumps(identity, sort_keys=True, indent=2), encoding="utf-8",
    )
    return context


def restore_recovery_dataset(directory, destination):
    """Extract only regular dataset files from the already verified checkpoint."""
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(directory / "assets/dataset.tar", "r|") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or ".." in name.parts or "\\" in member.name
                    or not name.parts or not (member.isdir() or member.isfile())):
                raise RuntimeError("Recovery dataset archive contains an unsafe entry")
            target = destination / name
            if not target.resolve().is_relative_to(destination.resolve()):
                raise RuntimeError("Recovery dataset archive escapes its root")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=256 * 1024)
    return destination


def materialize_recovery_checkpoint(directory, metadata, *, output, dataset_work):
    stage = metadata["stage"]
    if stage == "gpt":
        target = output / "work/gpt/ckpt"
        target.mkdir(parents=True, exist_ok=True)
        name = f"epoch={metadata['epoch'] - 1}-step={metadata['global_step']}.ckpt"
        shutil.copy2(directory / "gpt/trainer.ckpt", target / name)
    else:
        target = dataset_work / "logs_s2_v2ProPlus"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("G", "D"):
            shutil.copy2(directory / "sovits" / (name + ".pth"), target / (name + "_0.pth"))
    candidates = directory / "candidates"
    if candidates.exists():
        for path in candidates.rglob("*"):
            if path.is_file():
                target = output / "checkpoints" / path.relative_to(candidates)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
