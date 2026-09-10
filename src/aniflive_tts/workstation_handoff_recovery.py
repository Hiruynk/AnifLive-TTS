"""Recover a verified checkpoint result after an artifact transport-limit failure.

The original job and every retained byte remain immutable. Recovery creates an
explicit retry, imports the original Docker evidence and uses normal downstream
gates. It never executes training, opens test data or changes quality reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .workstation import WorkstationError, WorkstationStore
from .workstation_adapters import AdapterResult
from .workstation_docker import parse_result_manifest
from .workstation_worker import WorkstationWorker, _LeaseMonitor

_LIMIT_FAILURE = "Docker worker artifacts must be a bounded JSON array"
_RECOVERY_METADATA_ERRORS = {"result is too large", "Recovery lost its job lease"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _contained_file(root: Path, path: Path) -> Path:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise WorkstationError("Retained evidence cannot contain symbolic links")
    path.resolve(strict=True).relative_to(root.resolve(strict=True))
    if not path.is_file():
        raise WorkstationError("Retained evidence must be a regular file")
    return path


def recover_checkpoint_handoff(
    store: WorkstationStore, source_job_id: str, *, apply: bool = False
) -> dict[str, Any]:
    retry_source = store.get_job(source_job_id)
    source = retry_source
    root = store.root
    for _ in range(8):
        if (
            source["type"] != "checkpoint.select"
            or source["status"] != "failed"
            or source.get("error") not in {_LIMIT_FAILURE, *_RECOVERY_METADATA_ERRORS}
            or source.get("result")
        ):
            raise WorkstationError("Recovery requires a checkpoint transport-limit failure")
        source_job_id = source["id"]
        candidates = list(
            (root / "docker-worker-output" / source_job_id).glob("*/result.json")
        )
        if len(candidates) == 1:
            break
        previous_id = source.get("retry_of")
        if candidates or source.get("error") not in _RECOVERY_METADATA_ERRORS or not previous_id:
            raise WorkstationError("Recovery requires exactly one retained Docker result")
        previous = store.get_job(previous_id)
        if any(
            previous[key] != source[key]
            for key in ("type", "project_id", "parameters", "depends_on")
        ):
            raise WorkstationError("Recovery ancestry changed the original work")
        source = previous
    else:
        raise WorkstationError("Recovery ancestry is too deep")
    result_path = _contained_file(root, candidates[0])
    manifest_path = _contained_file(
        root, root / "worker-manifests" / "checkpoint-select" / f"{source_job_id}.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        manifest.get("job_id") != source_job_id
        or manifest.get("job_type") != source["type"]
        or manifest.get("project_id") != source["project_id"]
        or manifest.get("readiness") != "ready"
    ):
        raise WorkstationError("Retained preparation manifest does not match its source job")
    image = manifest.get("backend", {}).get("image")
    image_digest = raw.get("image_digest")
    if not isinstance(image, str) or not isinstance(image_digest, str) or not image.endswith(
        "@" + image_digest
    ):
        raise WorkstationError("Retained worker image does not match preparation evidence")
    manifest_sha = _sha(manifest_path)
    result_sha = _sha(result_path)
    parsed = parse_result_manifest(
        result_path,
        output_root=result_path.parent,
        expected_job_id=source_job_id,
        expected_job_type=source["type"],
        expected_project_id=source["project_id"],
        expected_run_id=result_path.parent.name,
        expected_image_digest=image_digest,
        expected_manifest_sha256=manifest_sha,
    )
    if len(parsed["artifacts"]) <= 1024:
        raise WorkstationError("Retained inventory does not explain the original limit failure")
    report_entries = [
        row for row in parsed["artifacts"]
        if row["relative_path"] == "checkpoint-selection-report.json"
    ]
    if len(report_entries) != 1:
        raise WorkstationError("Retained checkpoint report is absent or duplicated")
    report = json.loads(
        (result_path.parent / "checkpoint-selection-report.json").read_text(encoding="utf-8")
    )
    if (
        report.get("status") != "passed"
        or report.get("test_split_accessed") is not False
        or parsed["payload"].get("status") != "passed"
    ):
        raise WorkstationError("Recovery cannot bypass checkpoint quality or sealed-test gates")
    audit = {
        "schema": "aniflive-tts-checkpoint-handoff-recovery-v1",
        "source_job_id": source_job_id,
        "retry_source_job_id": retry_source["id"],
        "source_run_id": result_path.parent.name,
        "source_result_sha256": result_sha,
        "source_manifest_sha256": manifest_sha,
        "source_image_digest": image_digest,
        "artifact_count": len(parsed["artifacts"]),
        "test_split_accessed": False,
        "neural_execution_reused": True,
        "applied": False,
    }
    if not apply:
        return audit

    retry = store.create_job(
        job_type=source["type"],
        project_id=source["project_id"],
        parameters=source["parameters"],
        depends_on=source["depends_on"],
        priority=source["priority"],
        retry_of=retry_source["id"],
        attempt=retry_source["attempt"] + 1,
        start_paused=True,
    )
    claim = store.claim_job(retry["id"], allow_paused=True)
    monitor = _LeaseMonitor(store, retry["id"], claim.token, 5.0)
    worker = WorkstationWorker(store)
    monitor.start()
    try:
        store.append_job_log(
            retry["id"], "info",
            f"Recovering verified Docker handoff from {source_job_id}; no neural rerun",
            claim_token=claim.token,
        )
        registered = store.register_worker_artifacts(
            job_id=retry["id"],
            claim_token=claim.token,
            job_type=source["type"],
            project_id=source["project_id"],
            source_root=result_path.parent,
            artifacts=parsed["artifacts"],
            image_digest=image_digest,
            parent_artifact_ids=worker._artifact_parent_ids(source, source["parameters"]),
        )
        if _sha(result_path) != result_sha or _sha(manifest_path) != manifest_sha:
            raise WorkstationError("Retained handoff evidence changed during recovery")
        if monitor.error is not None:
            raise WorkstationError(f"Recovery lost its job lease: {monitor.error}")
        if store.job_cancel_requested(retry["id"], claim.token):
            raise WorkstationError("Recovery was cancelled before completion")
    except Exception as error:
        monitor.stop()
        worker._finish_failed(retry["id"], source["type"], claim.token, error)
        raise
    monitor.stop()
    result = AdapterResult(
        source["type"], source["resource_class"], "completed",
        {
            "execution_performed": True,
            "execution_reused": True,
            "readiness": "ready",
            "missing_inputs": [],
            "backend_available": True,
            "backend": {
                "backend": "linux-docker",
                "image": image,
                "image_digest": image_digest,
                "run_id": result_path.parent.name,
                "payload": parsed["payload"],
                "recovery": audit,
            },
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "registered_artifact_ids": [row["id"] for row in registered],
        },
    )
    worker._finish_result(retry["id"], source["type"], claim.token, result)
    status = store.get_job(retry["id"])["status"]
    if status != "succeeded":
        raise WorkstationError("Recovered handoff did not pass normal completion gates")
    return {**audit, "applied": True, "retry_job_id": retry["id"], "status": status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workstation-dir", type=Path, required=True)
    parser.add_argument("--source-job-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not Path("/.dockerenv").is_file():
        parser.error("Handoff recovery must run inside Linux Docker")
    result = recover_checkpoint_handoff(
        WorkstationStore(args.workstation_dir), args.source_job_id, apply=args.apply
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
