"""Atomic, checksummed checkpoint bundles for Docker training recovery.

The training runner supplies complete framework state. This module only commits
and verifies bundles; a directory or a partial file is never a recovery point.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping
from uuid import uuid4

SCHEMA = "aniflive-training-snapshot-v1"
_SNAPSHOT_ID = re.compile(r"checkpoint-[0-9a-f]{32}")


class TrainingSnapshotError(RuntimeError):
    pass


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _regular(path: Path, root: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise TrainingSnapshotError("Checkpoint entry must be a regular file")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or any(
        parent.is_symlink() for parent in path.parents if parent != root.parent
    ):
        raise TrainingSnapshotError("Checkpoint entry escapes its bundle")
    return path


def _metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(_json_bytes(dict(value)))
    if value.get("stage") not in {"gpt", "sovits"}:
        raise TrainingSnapshotError("Checkpoint stage is invalid")
    for key in ("epoch", "global_step"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise TrainingSnapshotError(f"Checkpoint {key} is invalid")
    for key in ("job_id", "source_fingerprint"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise TrainingSnapshotError(f"Checkpoint {key} is required")
    completed = value.get("completed_stages", [])
    if (not isinstance(completed, list)
            or any(not isinstance(stage, str) or stage not in {"gpt", "sovits"} for stage in completed)
            or len(completed) != len(set(completed))
            or value["stage"] in completed):
        raise TrainingSnapshotError("Completed training stages are invalid")
    return value


def _validate_completed_candidates(metadata, files):
    for stage in metadata.get("completed_stages", []):
        suffix = ".ckpt" if stage == "gpt" else ".pth"
        if not any(isinstance(name, str) and name.startswith("candidates/" + stage + "/") and name.endswith(suffix)
                   for name in files):
            raise TrainingSnapshotError("Completed stage is missing its deployable candidates")


def _write_durable(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _required_files(stage: str) -> set[str]:
    common = {"training-plan.json", "training-budget.json"}
    return common | (
        {"gpt/trainer.ckpt", "gpt/random-state.pt"} if stage == "gpt"
        else {"sovits/G.pth", "sovits/D.pth", "sovits/runtime-state.pt"}
    )


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def commit_training_snapshot(
    root: Path, *, metadata: Mapping[str, Any],
) -> Iterator[Path]:
    """Expose a private staging directory, then atomically publish its manifest."""
    if root.is_symlink():
        raise TrainingSnapshotError("Checkpoint root cannot be a symbolic link")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = _metadata(metadata)
    history = []
    if (root / "latest.json").exists():
        _, previous_manifest = read_training_snapshot(
            root, expected_source_fingerprint=identity["source_fingerprint"],
        )
        if previous_manifest["metadata"]["job_id"] != identity["job_id"]:
            raise TrainingSnapshotError("Checkpoint root belongs to a different job")
        previous = json.loads((root / "latest.json").read_text())
        retained = previous.get("retained", [])
        if not isinstance(retained, list) or len(retained) > 1:
            raise TrainingSnapshotError("Checkpoint retained history is invalid")
        history = [{key: previous[key] for key in ("snapshot_id", "manifest_sha256")}, *retained]
        for record in history:
            if (not isinstance(record, dict)
                    or not isinstance(record.get("snapshot_id"), str)
                    or not _SNAPSHOT_ID.fullmatch(record["snapshot_id"])
                    or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("manifest_sha256", "")))):
                raise TrainingSnapshotError("Checkpoint retained history is invalid")
        if len({record["snapshot_id"] for record in history}) != len(history):
            raise TrainingSnapshotError("Checkpoint retained history is duplicated")
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
    committed = False
    try:
        yield staging
        required = _required_files(identity["stage"])
        files = {}
        for path in staging.rglob("*"):
            if path.is_symlink():
                raise TrainingSnapshotError("Checkpoint bundle cannot contain symbolic links")
            if path.is_dir():
                continue
            _regular(path, staging)
            relative = path.relative_to(staging).as_posix()
            if relative == "manifest.json":
                raise TrainingSnapshotError("Checkpoint manifest is reserved")
            files[relative] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        if not required.issubset(files):
            raise TrainingSnapshotError("Checkpoint framework state or budget is incomplete")
        _validate_completed_candidates(identity, files)
        manifest = {"schema": SCHEMA, "metadata": identity, "files": files}
        _write_durable(staging / "manifest.json", _json_bytes(manifest))
        snapshot_id = "checkpoint-" + uuid4().hex
        destination = root / snapshot_id
        for directory in staging.rglob("*"):
            if directory.is_dir():
                _sync_directory(directory)
        _sync_directory(staging)
        os.replace(staging, destination)
        _sync_directory(root)
        committed = True
        pointer = {"schema": SCHEMA, "snapshot_id": snapshot_id,
                   "manifest_sha256": _sha256(destination / "manifest.json"),
                   "retained": history[:1]}
        temporary_pointer = root / (".latest-" + uuid4().hex)
        _write_durable(temporary_pointer, _json_bytes(pointer))
        os.replace(temporary_pointer, root / "latest.json")
        _sync_directory(root)
        # Only remove a previously published, unchanged bundle after its successor
        # and the last-known-good pointer are durable. Never enumerate unknown dirs.
        for record in history[1:]:
            obsolete = root / record["snapshot_id"]
            if not obsolete.exists():
                continue
            manifest_path = obsolete / "manifest.json"
            if (obsolete.is_symlink() or obsolete.resolve().parent != root
                    or manifest_path.is_symlink() or not manifest_path.is_file()):
                continue
            try:
                _, old_manifest = _read_snapshot_pointer(
                    root, {"schema": SCHEMA, **record},
                    expected_source_fingerprint=identity["source_fingerprint"],
                )
            except (TrainingSnapshotError, OSError, ValueError, KeyError, TypeError):
                # A modified/corrupt bundle is diagnostic evidence, not cleanup input.
                continue
            if (old_manifest.get("schema") != SCHEMA
                    or old_manifest.get("metadata", {}).get("job_id") != identity["job_id"]
                    or old_manifest.get("metadata", {}).get("source_fingerprint") != identity["source_fingerprint"]):
                continue
            shutil.rmtree(obsolete)
        _sync_directory(root)
    finally:
        if not committed and staging.is_dir() and not staging.is_symlink():
            if staging.resolve().parent != root:
                raise TrainingSnapshotError("Refusing to clean a checkpoint outside its root")
            shutil.rmtree(staging)


def read_training_snapshot(
    root: Path, *, expected_source_fingerprint: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Verify every byte before a caller may load checkpoint framework state."""
    if root.is_symlink():
        raise TrainingSnapshotError("Checkpoint root cannot be a symbolic link")
    root = root.resolve(strict=True)
    pointer_path = _regular(root / "latest.json", root)
    if pointer_path.stat().st_size > 4096:
        raise TrainingSnapshotError("Checkpoint pointer is oversized")
    pointer = json.loads(pointer_path.read_text())
    return _read_snapshot_pointer(
        root, pointer, expected_source_fingerprint=expected_source_fingerprint,
    )


def _read_snapshot_pointer(
    root: Path, pointer: Mapping[str, Any], *,
    expected_source_fingerprint: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    snapshot_id = pointer.get("snapshot_id")
    if pointer.get("schema") != SCHEMA or not isinstance(snapshot_id, str) or not _SNAPSHOT_ID.fullmatch(snapshot_id):
        raise TrainingSnapshotError("Checkpoint pointer is invalid")
    directory = root / snapshot_id
    manifest_path = _regular(directory / "manifest.json", root)
    if manifest_path.stat().st_size > 4 * 1024 * 1024:
        raise TrainingSnapshotError("Checkpoint manifest is oversized")
    if _sha256(manifest_path) != pointer.get("manifest_sha256"):
        raise TrainingSnapshotError("Checkpoint manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise TrainingSnapshotError("Checkpoint schema is invalid")
    metadata = _metadata(manifest["metadata"])
    if expected_source_fingerprint is not None and metadata["source_fingerprint"] != expected_source_fingerprint:
        raise TrainingSnapshotError("Checkpoint belongs to different training inputs")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or len(files) > 10000:
        raise TrainingSnapshotError("Checkpoint inventory is invalid")
    if not _required_files(metadata["stage"]).issubset(files):
        raise TrainingSnapshotError("Checkpoint framework state or budget is incomplete")
    _validate_completed_candidates(metadata, files)
    inventory = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise TrainingSnapshotError("Checkpoint bundle contains a symbolic link")
        if path.is_file():
            inventory.add(path.relative_to(directory).as_posix())
    if inventory != set(files) | {"manifest.json"}:
        raise TrainingSnapshotError("Checkpoint contains unlisted or missing files")
    for name, record in files.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            raise TrainingSnapshotError("Checkpoint file record is invalid")
        if type(record.get("bytes")) is not int or record["bytes"] < 0:
            raise TrainingSnapshotError("Checkpoint file size is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))):
            raise TrainingSnapshotError("Checkpoint file checksum is invalid")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise TrainingSnapshotError("Checkpoint inventory contains an unsafe path")
        path = _regular(directory / relative, directory.resolve())
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise TrainingSnapshotError("Checkpoint file checksum mismatch")
    return directory, manifest
