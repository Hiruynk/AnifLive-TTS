
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aniflive_tts.workstation import WorkstationError, WorkstationStore
from aniflive_tts.workstation_docker import RESULT_MANIFEST_SCHEMA, parse_result_manifest
from aniflive_tts.workstation_handoff_recovery import recover_checkpoint_handoff

DIGEST = "sha256:" + "a" * 64


def _failed_handoff(tmp_path: Path):
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="training", name="Test training", config={"auto_build_production": True})
    training = store.create_job(job_type="training.prepare", project_id=project["id"])
    claim = store.claim_job(training["id"])
    store.update_job(training["id"], status="succeeded", progress=1.0, claim_token=claim.token)
    job = store.create_job(
        job_type="checkpoint.select", project_id=project["id"], depends_on=[training["id"]]
    )
    claim = store.claim_job(job["id"])
    store.update_job(
        job["id"], status="failed", claim_token=claim.token,
        error="Docker worker artifacts must be a bounded JSON array",
    )
    job = store.get_job(job["id"])
    manifest = {
        "job_id": job["id"], "job_type": job["type"], "project_id": project["id"],
        "readiness": "ready", "backend": {"image": "worker@" + DIGEST},
    }
    manifest_path = store.root / "worker-manifests" / "checkpoint-select" / (job["id"] + ".json")
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))
    output = store.root / "docker-worker-output" / job["id"] / ("b" * 32)
    output.mkdir(parents=True)
    paths = []
    for index in range(1995):
        path = output / f"evidence-{index}.bin"
        path.write_bytes(b"verified")
        paths.append(path)
    report = output / "checkpoint-selection-report.json"
    report.write_text(json.dumps({"status": "passed", "test_split_accessed": False}))
    paths.append(report)
    document = {
        "schema": RESULT_MANIFEST_SCHEMA, "job_id": job["id"], "job_type": job["type"],
        "project_id": project["id"], "run_id": output.name, "image_digest": DIGEST,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "platform": "linux/amd64", "outcome": "completed",
        "payload": {"schema": "aniflive-tts-checkpoint-selection-worker-v1", "status": "passed"},
        "artifacts": [
            {"kind": "checkpoint", "relative_path": path.name,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "size_bytes": path.stat().st_size}
            for path in paths
        ],
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(document))
    return store, job, result_path, document


def test_handoff_recovery_preserves_failed_job_and_original_evidence(tmp_path: Path) -> None:
    store, source, result_path, _ = _failed_handoff(tmp_path)
    original = result_path.read_bytes()
    preview = recover_checkpoint_handoff(store, source["id"])
    assert preview["artifact_count"] == 1996
    assert preview["applied"] is False
    assert len(store.list_jobs()) == 2
    applied = recover_checkpoint_handoff(store, source["id"], apply=True)
    assert applied["status"] == "succeeded"
    assert store.get_job(source["id"]) == source
    assert result_path.read_bytes() == original
    retry = store.get_job(applied["retry_job_id"])
    assert retry["retry_of"] == source["id"]
    assert retry["result"]["execution_reused"] is True
    assert len(retry["result"]["registered_artifact_ids"]) == 1996
    references = [job for job in store.list_jobs() if job["type"] == "reference.select"]
    assert len(references) == 1
    assert retry["id"] in references[0]["depends_on"]


def test_recovery_rejects_modified_artifact_before_creating_retry(tmp_path: Path) -> None:
    store, source, result_path, _ = _failed_handoff(tmp_path)
    (result_path.parent / "evidence-0.bin").write_bytes(b"tampered")
    with pytest.raises(WorkstationError, match="checksum"):
        recover_checkpoint_handoff(store, source["id"], apply=True)
    assert len(store.list_jobs()) == 2
    assert store.get_job(source["id"]) == source


@pytest.mark.parametrize("status,test_access", [("failed", False), ("passed", True)])
def test_recovery_never_bypasses_quality_or_test_gate(
    tmp_path: Path, status: str, test_access: bool
) -> None:
    store, source, result_path, document = _failed_handoff(tmp_path)
    report = result_path.parent / "checkpoint-selection-report.json"
    report.write_text(json.dumps({"status": status, "test_split_accessed": test_access}))
    entry = document["artifacts"][-1]
    entry["size_bytes"] = report.stat().st_size
    entry["sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
    result_path.write_text(json.dumps(document))
    with pytest.raises(WorkstationError, match="quality or sealed-test"):
        recover_checkpoint_handoff(store, source["id"], apply=True)
    assert len(store.list_jobs()) == 2


def test_checkpoint_inventory_remains_bounded(tmp_path: Path) -> None:
    _, _, result_path, document = _failed_handoff(tmp_path)
    document["artifacts"] = [document["artifacts"][0]] * 4097
    result_path.write_text(json.dumps(document))
    with pytest.raises(WorkstationError, match="bounded JSON array"):
        parse_result_manifest(
            result_path, output_root=result_path.parent,
            expected_job_id=document["job_id"], expected_job_type=document["job_type"],
            expected_project_id=document["project_id"], expected_run_id=document["run_id"],
            expected_image_digest=DIGEST, expected_manifest_sha256=document["manifest_sha256"],
        )


def test_non_checkpoint_inventory_retains_original_budget(tmp_path: Path) -> None:
    _, _, result_path, document = _failed_handoff(tmp_path)
    document["job_type"] = "training.prepare"
    result_path.write_text(json.dumps(document))
    with pytest.raises(WorkstationError, match="bounded JSON array"):
        parse_result_manifest(
            result_path, output_root=result_path.parent,
            expected_job_id=document["job_id"], expected_job_type="training.prepare",
            expected_project_id=document["project_id"], expected_run_id=document["run_id"],
            expected_image_digest=DIGEST, expected_manifest_sha256=document["manifest_sha256"],
        )


def test_paused_recovery_job_requires_explicit_atomic_claim(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="training", name="Test", config={})
    job = store.create_job(
        job_type="checkpoint.select", project_id=project["id"], start_paused=True
    )
    assert job["status"] == "paused"
    with pytest.raises(WorkstationError, match="queued"):
        store.claim_job(job["id"])
    claim = store.claim_job(job["id"], allow_paused=True)
    assert claim.job["status"] == "running"
    with pytest.raises(WorkstationError, match="queued"):
        store.claim_job(job["id"], allow_paused=True)


def test_result_limit_failure_recovers_through_unchanged_ancestry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aniflive_tts import workstation

    store, source, result_path, _ = _failed_handoff(tmp_path)
    original = result_path.read_bytes()
    with monkeypatch.context() as scoped:
        scoped.setattr(workstation, "MAX_JOB_RESULT_BYTES", 64 * 1024)
        with pytest.raises(WorkstationError, match="normal completion"):
            recover_checkpoint_handoff(store, source["id"], apply=True)
    failed = next(job for job in store.list_jobs() if job["retry_of"] == source["id"])
    assert failed["error"] == "result is too large"
    recovered = recover_checkpoint_handoff(store, failed["id"], apply=True)
    assert recovered["source_job_id"] == source["id"]
    assert recovered["retry_source_job_id"] == failed["id"]
    assert recovered["status"] == "succeeded"
    assert store.get_job(source["id"]) == source
    assert store.get_job(failed["id"]) == failed
    assert result_path.read_bytes() == original


def test_job_result_budget_is_bounded_without_expanding_project_config(tmp_path: Path) -> None:
    from aniflive_tts.workstation import MAX_JOB_RESULT_BYTES

    store = WorkstationStore(tmp_path / "workstation")
    with pytest.raises(WorkstationError, match="too large"):
        store.create_project(kind="training", name="Too large", config={"text": "x" * 70000})
    project = store.create_project(kind="training", name="Bounded", config={})
    job = store.create_job(job_type="checkpoint.select", project_id=project["id"])
    claim = store.claim_job(job["id"])
    with pytest.raises(WorkstationError, match="result is too large"):
        store.update_job(
            job["id"], result={"text": "x" * MAX_JOB_RESULT_BYTES}, claim_token=claim.token
        )


def test_failed_metadata_lease_reuses_only_matching_completed_ancestry(tmp_path: Path) -> None:
    store, source, result_path, _ = _failed_handoff(tmp_path)
    original = result_path.read_bytes()
    failed = store.create_job(
        job_type=source["type"], project_id=source["project_id"],
        parameters=source["parameters"], depends_on=source["depends_on"],
        retry_of=source["id"], attempt=2,
    )
    claim = store.claim_job(failed["id"])
    store.update_job(
        failed["id"], status="failed", error="Recovery lost its job lease",
        claim_token=claim.token,
    )
    failed = store.get_job(failed["id"])
    recovered = recover_checkpoint_handoff(store, failed["id"], apply=True)
    assert recovered["source_job_id"] == source["id"]
    assert recovered["retry_source_job_id"] == failed["id"]
    assert recovered["status"] == "succeeded"
    assert store.get_job(failed["id"]) == failed
    assert result_path.read_bytes() == original
