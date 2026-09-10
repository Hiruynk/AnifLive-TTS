"""Publish a qualified registered package into the local model registry."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .model_package import validate_checksums, validate_safe_identifier
from .sampling_policy import LEGACY, NATIVE
from .workstation import (
    WorkstationError, _checked_regular_file, validated_artifact_relative_path,
)


def _start_managed_runtime(store):
    if os.environ.get("ANIFLIVE_TTS_MANAGED_RUNTIME_START") != "1":
        return {"state": "external-management"}
    from .workstation_handoff import RuntimeHandoff
    coordinator = None
    try:
        coordinator = RuntimeHandoff(store, store.root / "runtime-handoff.json")
        coordinator._acquire()
        state = store.get_runtime_handoff()
        if state and state.get("phase") != "idle":
            return {"state": "deferred", "reason": "GPU work is active; rerun the launcher when it finishes"}
        runtime = coordinator._inspect()
        if runtime["State"]["Running"]:
            return {"state": "running"}
        coordinator._docker("start", runtime["Id"])
        return {"state": "starting"}
    except (OSError, WorkstationError) as error:
        return {"state": "not-started", "reason": str(error)}
    finally:
        if coordinator is not None:
            coordinator.close()


def _installed_result(store, result, model_id, target):
    return {**result, "deployment": {
        "status": "installed", "model_id": model_id, "path": str(target),
        "runtime_start": _start_managed_runtime(store),
    }}


def promote_and_publish(store, artifact_id: str, qualification_id: str):
    artifact = store.get_artifact(artifact_id)
    metadata = artifact.get("metadata", {})
    if (artifact.get("type") != "package"
            or metadata.get("worker_relative_path") != "model-package/manifest.json"):
        return store.promote_artifact(artifact_id, qualification_id=qualification_id)
    with store._connect() as connection:
        qualification = store._assert_qualification_passes(
            connection, qualification_id=qualification_id,
            subject_kind="artifact", subject_id=artifact_id,
        )
        evaluation_artifact_id = qualification["evaluation_artifact_id"]
        report_sha256 = qualification["report_sha256"]
    qualified = store._verified_evaluation_report(store.get_artifact(evaluation_artifact_id))
    auto_ref = qualified["run_metadata"]["composition"]["sources"]["automated"]
    automated, _ = store._verified_json_evaluation_artifact(store.get_artifact(auto_ref["artifact_id"]))
    policies = {
        row.get("sampling_policy", {}).get(
            "semantic_sampling", automated.get("plan", {}).get("semantic_sampling", LEGACY)
        )
        for row in automated["languages"].values()
    }
    if len(policies) != 1:
        raise WorkstationError("Qualified evaluation has inconsistent sampling contracts")
    policy = policies.pop()
    if policy not in {LEGACY, NATIVE}:
        raise WorkstationError("Qualified sampling contract is unsupported")
    job = store.get_job(metadata["job_id"])
    ids = job.get("result", {}).get("registered_artifact_ids", [])
    if job["type"] != "model.package" or job["status"] != "succeeded" or artifact_id not in ids:
        raise WorkstationError("Package must belong to a completed registered package job")
    source_root = store.artifact_root.resolve(strict=True)
    manifest_path = _checked_regular_file(
        source_root, source_root / artifact["local_path"], field="Package manifest")
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != artifact["sha256"]:
        raise WorkstationError("Package manifest changed after registration")
    manifest = json.loads(manifest_bytes)
    model_id = validate_safe_identifier(manifest.get("model_id"), "model_id")
    if manifest.get("semantic_sampling", policy) != policy:
        raise WorkstationError("Package sampling differs from its qualified evaluation")
    manifest["semantic_sampling"] = policy
    manifest["qualification"] = {
        "id": qualification_id, "report_sha256": report_sha256,
        "source_artifact_id": artifact_id,
    }
    if automated.get("baseline", {}).get("available") is False:
        manifest["qualification"]["performance_comparison"] = "unavailable-measured-only"
    registry = Path(os.environ.get("ANIFLIVE_TTS_MODEL_REGISTRY", str(store.root.parent / "models")))
    if not registry.is_absolute():
        raise WorkstationError("Model registry must use an absolute path")
    if registry.is_symlink():
        raise WorkstationError("Model registry cannot be a symbolic link")
    registry.mkdir(parents=True, exist_ok=True)
    target = registry.resolve() / model_id
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_dir():
            raise WorkstationError("Model registry name is already occupied")
        existing = json.loads((target / "manifest.json").read_text())
        if existing != manifest:
            raise WorkstationError("Model ID already exists with different content; use a new model ID")
        validate_checksums(target)
        result = store.promote_artifact(artifact_id, qualification_id=qualification_id)
        return _installed_result(store, result, model_id, target)
    # Keep the atomic rename within the destination mount. Registry discovery
    # reads only immediate children, so packages nested here are not visible.
    staging_root = registry.resolve() / ".aniflive-staging"
    if staging_root.is_symlink() or (staging_root / "manifest.json").exists():
        raise WorkstationError("Model staging location conflicts with existing content")
    staging_root.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="package-", dir=staging_root))
    hashes = {}
    try:
        for item_id in ids:
            item = store.get_artifact(item_id)
            meta = item.get("metadata", {})
            if item.get("status") != "ready" or meta.get("job_id") != job["id"]:
                raise WorkstationError("Package files do not belong to the completed job")
            relative = validated_artifact_relative_path(
                meta.get("worker_relative_path"), field="Package worker path")
            if len(relative.parts) < 2 or relative.parts[0] != "model-package":
                continue
            relative = Path(*relative.parts[1:])
            source_relative = validated_artifact_relative_path(item["local_path"], field="Package source")
            source = _checked_regular_file(
                source_root, source_root.joinpath(*source_relative.parts), field="Package source")
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with source.open("rb") as incoming, destination.open("xb") as outgoing:
                for block in iter(lambda: incoming.read(1024 * 1024), b""):
                    digest.update(block)
                    outgoing.write(block)
            if digest.hexdigest() != item["sha256"]:
                raise WorkstationError("Package file changed after registration")
            hashes[relative.as_posix()] = digest.hexdigest()
        checksums = json.loads((stage / "checksums.json").read_text())
        observed = {name: digest for name, digest in hashes.items() if name != "checksums.json"}
        if checksums != observed:
            raise WorkstationError("Registered package inventory does not match its checksum manifest")
        encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        (stage / "manifest.json").write_bytes(encoded)
        checksums["manifest.json"] = hashlib.sha256(encoded).hexdigest()
        (stage / "checksums.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
        def publish():
            if target.exists() or target.is_symlink():
                raise WorkstationError("Model registry name was occupied during publication")
            stage.rename(target)
        result = store.promote_artifact(
            artifact_id, qualification_id=qualification_id, _before_commit=publish)
        return _installed_result(store, result, model_id, target)
    except OSError as error:
        raise WorkstationError(
            "Model installation could not complete: " + (error.strerror or str(error))
        ) from error
    finally:
        if stage.exists():
            shutil.rmtree(stage)
