from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import subprocess
import time
from uuid import uuid4

import pytest

from aniflive_tts import workstation as workstation_module
from aniflive_tts.workstation import GPU_RESOURCE_KEY, WorkstationError, WorkstationStore


def _project_and_job(store: WorkstationStore, *, job_type: str = "dataset.inventory"):
    project = store.create_project(kind="dataset", name="Dataset")
    job = store.create_job(job_type=job_type, project_id=project["id"])
    return project, job


def test_error_message_preserves_context_and_final_cause() -> None:
    error = RuntimeError("CONTEXT:" + (" progress" * 300) + ":FINAL_CAUSE")

    message = workstation_module._error_message(error)

    assert len(message) == 1000
    assert message.startswith("CONTEXT:")
    assert "[middle omitted]" in message
    assert message.endswith(":FINAL_CAUSE")


def test_dataset_process_is_a_gpu_exclusive_dataset_job(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="dataset", name="Dataset")
    job = store.create_job(job_type="dataset.process", project_id=project["id"])
    assert job["resource_class"] == "gpu-exclusive"
    inference_token = store.acquire_resource_lease(
        GPU_RESOURCE_KEY, purpose="inference:test", owner_id="inference-server"
    )
    with pytest.raises(WorkstationError, match="GPU resource is busy"):
        store.claim_job(job["id"])
    store.release_resource_lease(inference_token)
    claim = store.claim_job(job["id"])
    store.update_job(job["id"], status="cancelled", claim_token=claim.token)
    training = store.create_project(kind="training", name="Training")
    with pytest.raises(WorkstationError, match="project of kind: dataset"):
        store.create_job(job_type="dataset.process", project_id=training["id"])


def test_model_package_validation_holds_the_gpu_lease(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="evaluation", name="Package validation")
    job = store.create_job(job_type="model.package", project_id=project["id"])
    assert job["resource_class"] == "gpu-exclusive"
    inference_token = store.acquire_resource_lease(
        GPU_RESOURCE_KEY, purpose="inference:test", owner_id="inference-server"
    )
    with pytest.raises(WorkstationError, match="GPU resource is busy"):
        store.claim_job(job["id"])
    store.release_resource_lease(inference_token)


def test_existing_dataset_process_job_is_migrated_to_gpu_exclusive(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workstation"
    store = WorkstationStore(root)
    project = store.create_project(kind="dataset", name="Dataset")
    job = store.create_job(job_type="dataset.process", project_id=project["id"])
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE jobs SET resource_class = 'io-heavy' WHERE id = ?", (job["id"],)
        )

    migrated = WorkstationStore(root).get_job(job["id"])
    assert migrated["resource_class"] == "gpu-exclusive"


def test_workstation_project_and_job_metadata_survive_restart(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(
        kind="training", name="Roxy V2ProPlus", config={"preset": "balanced"}
    )
    job = store.create_job(
        job_type="training.prepare", project_id=project["id"],
        parameters={"checkpoint": "epoch-8"},
    )
    reloaded = WorkstationStore(tmp_path / "workstation")
    assert reloaded.get_project(project["id"])["config"]["preset"] == "balanced"
    assert reloaded.list_jobs()[0]["id"] == job["id"]
    with sqlite3.connect(reloaded.database_path) as connection:
        schema = connection.execute("SELECT value FROM metadata WHERE key = 'schema'").fetchone()
        journal = connection.execute("PRAGMA journal_mode").fetchone()
    assert schema == ("9",)
    assert journal and journal[0].lower() == "wal"


def test_queued_gpu_wait_reason_is_stable_and_clears_on_claim(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="training", name="Training")
    job = store.create_job(job_type="training.prepare", project_id=project["id"])
    reason = "Waiting for gpu:0; TensorRT inference or another GPU job is resident"

    waiting = store.set_job_wait_reason(job["id"], reason)
    repeated = store.set_job_wait_reason(job["id"], reason)

    assert waiting["status"] == "queued"
    assert repeated["wait_reason"] == reason
    assert [event["message"] for event in store.list_job_logs(job["id"])].count(reason) == 1

    claim = store.claim_job(job["id"])
    assert claim.job["status"] == "running"
    assert claim.job["wait_reason"] is None
    store.update_job(job["id"], status="cancelled", claim_token=claim.token)


def test_project_summary_tracks_job_lifecycle_transactionally(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="training", name="Training")
    job = store.create_job(job_type="training.prepare", project_id=project["id"])

    queued = store.get_project(project["id"])
    assert queued["status"] == "queued"
    assert queued["progress"] == 0.0
    assert queued["metrics"]["jobs"] == {
        "total": 1,
        "cancelled": 0,
        "failed": 0,
        "paused": 0,
        "queued": 1,
        "running": 0,
        "succeeded": 0,
        "terminal": 0,
        "cancel_requested": 0,
        "pause_requested": 0,
        "latest": {
            "id": job["id"],
            "type": "training.prepare",
            "status": "queued",
        },
    }

    claim = store.claim_job(job["id"])
    running = store.get_project(project["id"])
    assert running["status"] == "running"
    assert running["progress"] == pytest.approx(0.01)
    assert running["metrics"]["jobs"]["running"] == 1

    store.update_job(job["id"], progress=0.4, claim_token=claim.token)
    progressed = store.get_project(project["id"])
    assert progressed["status"] == "running"
    assert progressed["progress"] == pytest.approx(0.4)

    store.update_job(
        job["id"], status="succeeded", result={"ok": True}, claim_token=claim.token
    )
    succeeded = store.get_project(project["id"])
    assert succeeded["status"] == "succeeded"
    assert succeeded["progress"] == 1.0
    assert succeeded["metrics"]["jobs"]["succeeded"] == 1
    assert succeeded["metrics"]["jobs"]["terminal"] == 1


def test_project_summary_keeps_completed_work_when_a_later_job_is_cancelled(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="training", name="Pipeline")
    first = store.create_job(job_type="training.prepare", project_id=project["id"])
    first_claim = store.claim_job(first["id"])
    store.update_job(first["id"], status="succeeded", claim_token=first_claim.token)
    second = store.create_job(job_type="model.package", project_id=project["id"])

    queued = store.get_project(project["id"])
    assert queued["status"] == "queued"
    assert queued["progress"] == pytest.approx(0.5)
    assert queued["metrics"]["jobs"]["total"] == 2
    assert queued["metrics"]["jobs"]["succeeded"] == 1
    assert queued["metrics"]["jobs"]["queued"] == 1

    cancelled_job = store.cancel_job(second["id"])
    cancelled = store.get_project(project["id"])
    assert cancelled_job["status"] == "cancelled"
    assert cancelled["status"] == "succeeded"
    assert cancelled["progress"] == 1.0
    assert cancelled["metrics"]["jobs"]["cancelled"] == 1
    assert cancelled["metrics"]["jobs"]["terminal"] == 2


def test_running_cancel_request_is_visible_in_project_metrics(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project, job = _project_and_job(store)
    claim = store.claim_job(job["id"])

    store.cancel_job(job["id"])
    cancelling = store.get_project(project["id"])

    assert cancelling["status"] == "running"
    assert cancelling["metrics"]["jobs"]["cancel_requested"] == 1
    store.update_job(job["id"], status="cancelled", claim_token=claim.token)
    assert store.get_project(project["id"])["status"] == "cancelled"


def test_concurrent_fresh_database_bootstrap_is_safe(tmp_path: Path) -> None:
    root = tmp_path / "workstation"
    with ThreadPoolExecutor(max_workers=8) as executor:
        stores = list(executor.map(lambda _: WorkstationStore(root), range(16)))
    assert len(stores) == 16
    assert WorkstationStore(root).snapshot().projects == ()


def test_schema_two_database_upgrades_before_lease_index_creation(tmp_path: Path) -> None:
    root = tmp_path / "workstation"
    root.mkdir()
    database = root / "workstation.sqlite3"
    project_id = f"training_{uuid4()}"
    job_id = f"job_{uuid4()}"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO metadata(key, value) VALUES('schema', '2');
            CREATE TABLE projects (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
                status TEXT NOT NULL, progress REAL NOT NULL, config_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY, type TEXT NOT NULL,
                project_id TEXT REFERENCES projects(id), status TEXT NOT NULL,
                progress REAL NOT NULL, parameters_json TEXT NOT NULL,
                result_json TEXT NOT NULL, error TEXT, resource_class TEXT NOT NULL,
                depends_on_json TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, started_at TEXT, finished_at TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO projects VALUES (?, 'training', 'Legacy', 'draft', 0, '{}', '{}', '2026-01-01Z', '2026-01-01Z')",
            (project_id,),
        )
        connection.execute(
            "INSERT INTO jobs VALUES (?, 'training.prepare', ?, 'queued', 0, '{}', '{}', NULL, 'cpu-shared', '[]', '2026-01-01Z', '2026-01-01Z', NULL, NULL)",
            (job_id, project_id),
        )
    store = WorkstationStore(root)
    upgraded = store.get_job(job_id)
    assert upgraded["resource_class"] == "gpu-exclusive"
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(jobs)")}
    assert {
        "worker_id", "claim_token", "lease_expires_at", "cancel_requested",
        "pause_requested", "priority", "attempt", "retry_of", "wait_reason",
    } <= columns
    assert "jobs_lease_expiry" in indexes
    assert "jobs_queue_priority" in indexes
    assert "jobs_single_retry_attempt" in indexes
    assert upgraded["priority"] == 0
    assert upgraded["attempt"] == 1
    assert upgraded["pause_requested"] is False


def test_schema_seven_database_adds_queue_wait_reason(tmp_path: Path) -> None:
    root = tmp_path / "workstation"
    original = WorkstationStore(root)
    with sqlite3.connect(original.database_path) as connection:
        connection.execute("ALTER TABLE jobs DROP COLUMN wait_reason")
        connection.execute("UPDATE metadata SET value = '7' WHERE key = 'schema'")

    migrated = WorkstationStore(root)
    with sqlite3.connect(migrated.database_path) as connection:
        schema = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema'"
        ).fetchone()
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}

    assert schema == ("9",)
    assert "wait_reason" in columns


def test_schema_eight_adds_reference_artifacts_without_losing_lineage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workstation"
    original = WorkstationStore(root)
    project = original.create_project(kind="training", name="Existing Training")
    parent = original.register_artifact(
        artifact_type="dataset", name="Frozen Dataset", project_id=None
    )
    child = original.register_artifact(
        artifact_type="checkpoint",
        name="Selected Checkpoint",
        project_id=project["id"],
        parent_artifact_ids=(parent["id"],),
    )
    with sqlite3.connect(original.database_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("PRAGMA legacy_alter_table = ON")
        connection.execute("ALTER TABLE artifacts RENAME TO artifacts_schema_v9")
        connection.execute(
            """CREATE TABLE artifacts (
                id TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT NOT NULL,
                project_id TEXT REFERENCES projects(id), status TEXT NOT NULL,
                local_path TEXT, sha256 TEXT, metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                CHECK(length(id) = 45 AND substr(id, 1, 9) = 'artifact_'),
                CHECK(type IN ('dataset', 'checkpoint', 'expression-bank',
                    'evaluation', 'engine', 'package')),
                CHECK(status IN ('planned', 'building', 'ready', 'failed',
                    'rejected', 'archived'))
            )"""
        )
        connection.execute(
            "INSERT INTO artifacts SELECT * FROM artifacts_schema_v9"
        )
        connection.execute("DROP TABLE artifacts_schema_v9")
        connection.execute("UPDATE metadata SET value = '8' WHERE key = 'schema'")

    migrated = WorkstationStore(root)
    assert migrated.get_artifact(child["id"])["parent_artifact_ids"] == [parent["id"]]
    with sqlite3.connect(migrated.database_path) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        schema = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema'"
        ).fetchone()
        sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'artifacts'"
        ).fetchone()[0]
    assert schema == ("9",)
    assert "'reference'" in sql


def test_schema_three_database_migrates_to_artifact_registry(tmp_path: Path) -> None:
    root = tmp_path / "workstation"
    original = WorkstationStore(root)
    project = original.create_project(kind="training", name="Existing v1.4 Project")
    with sqlite3.connect(original.database_path) as connection:
        connection.execute("DROP TABLE artifact_parents")
        connection.execute("DROP TABLE artifacts")
        connection.execute("UPDATE metadata SET value = '3' WHERE key = 'schema'")
    migrated = WorkstationStore(root)
    assert migrated.get_project(project["id"])["name"] == "Existing v1.4 Project"
    with sqlite3.connect(migrated.database_path) as connection:
        schema = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema'"
        ).fetchone()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert schema == ("9",)
    assert {"artifacts", "artifact_parents", "expression_drafts"} <= tables


def test_legacy_manifest_migration_is_concurrent_and_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "workstation"
    root.mkdir()
    (root / "workstation.json").write_text(
        '{"projects":[{"id":"dataset_deadbeef1234","kind":"dataset",'
        '"name":"Legacy","config":{},"metrics":{}}],"jobs":[]}',
        encoding="utf-8",
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        stores = list(executor.map(lambda _: WorkstationStore(root), range(2)))
    projects = stores[0].list_projects()
    assert len(projects) == 1
    assert projects[0]["id"].startswith("dataset_")
    assert len(projects[0]["id"].partition("_")[2]) == 36
    assert (root / "workstation.json.migrated").is_file()


def test_second_store_does_not_recover_a_live_job(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    _, job = _project_and_job(store)
    claim = store.claim_job(job["id"])
    observer = WorkstationStore(tmp_path / "workstation")
    assert observer.get_job(job["id"])["status"] == "running"
    assert observer.recover_expired_jobs() == 0
    assert store.update_job(
        job["id"], status="succeeded", result={"ok": True}, claim_token=claim.token
    )["status"] == "succeeded"


def test_claim_lease_clock_starts_after_waiting_for_writer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANIFLIVE_TTS_WORKSTATION_LEASE_SECONDS", "1")
    store = WorkstationStore(tmp_path / "workstation")
    _, job = _project_and_job(store)
    blocker = sqlite3.connect(store.database_path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(store.claim_job, job["id"])
        time.sleep(1.1)
        blocker.commit()
        claim = future.result(timeout=5)
    blocker.close()
    assert claim.job["status"] == "running"
    assert store.recover_expired_jobs() == 0
    store.update_job(job["id"], status="cancelled", claim_token=claim.token)


def test_only_expired_job_leases_are_recovered(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    _, job = _project_and_job(store)
    claim = store.claim_job(job["id"])
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at = '2000-01-01T00:00:00.000Z' WHERE id = ?",
            (job["id"],),
        )
    assert WorkstationStore(tmp_path / "workstation").get_job(job["id"])["status"] == "failed"
    with pytest.raises(WorkstationError, match="immutable"):
        store.update_job(job["id"], status="succeeded", claim_token=claim.token)


def test_expired_worker_with_pending_pause_recovers_as_paused(tmp_path: Path) -> None:
    root = tmp_path / "workstation"
    store = WorkstationStore(root)
    _, job = _project_and_job(store)
    store.claim_job(job["id"])
    store.pause_job(job["id"])
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at = '2000-01-01T00:00:00.000Z' "
            "WHERE id = ?",
            (job["id"],),
        )

    recovered = WorkstationStore(root).get_job(job["id"])
    assert recovered["status"] == "paused"
    assert recovered["pause_requested"] is False
    assert recovered["finished_at"] is None


def test_claim_token_is_required_and_terminal_jobs_are_immutable(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    _, job = _project_and_job(store)
    with pytest.raises(WorkstationError, match="Only claim_job"):
        store.update_job(job["id"], status="running")
    claim = store.claim_job(job["id"])
    with pytest.raises(WorkstationError, match="claim token"):
        store.update_job(job["id"], progress=0.5)
    completed = store.update_job(
        job["id"], status="succeeded", progress=0.9,
        result={"done": True}, claim_token=claim.token,
    )
    assert completed["progress"] == 1.0
    with pytest.raises(WorkstationError, match="immutable"):
        store.update_job(job["id"], status="failed", error="changed")


def test_job_resource_class_is_derived_and_inference_lease_blocks_gpu_job(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="training", name="Training")
    with pytest.raises(WorkstationError, match="requires resource class"):
        store.create_job(
            job_type="training.prepare", project_id=project["id"], resource_class="cpu-shared"
        )
    job = store.create_job(job_type="training.prepare", project_id=project["id"])
    assert job["resource_class"] == "gpu-exclusive"
    inference_token = store.acquire_resource_lease(
        GPU_RESOURCE_KEY, purpose="inference:test", owner_id="inference-server"
    )
    with pytest.raises(WorkstationError, match="GPU resource is busy"):
        store.claim_job(job["id"])
    store.release_resource_lease(inference_token)
    claim = store.claim_job(job["id"])
    store.update_job(job["id"], status="cancelled", claim_token=claim.token)


def test_create_job_and_initial_event_are_atomic(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="dataset", name="Dataset")
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_job_event BEFORE INSERT ON job_events "
            "BEGIN SELECT RAISE(ABORT, 'event rejected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="event rejected"):
        store.create_job(job_type="dataset.inventory", project_id=project["id"])
    assert store.list_jobs() == []


def test_job_type_rejects_incompatible_project_kind(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="dataset", name="Dataset")
    with pytest.raises(WorkstationError, match="project of kind"):
        store.create_job(job_type="training.prepare", project_id=project["id"])


def test_dependency_failure_is_propagated_without_stuck_queue(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="dataset", name="Dataset")
    parent = store.create_job(job_type="dataset.inventory", project_id=project["id"])
    child = store.create_job(
        job_type="dataset.inventory", project_id=project["id"], depends_on=[parent["id"]]
    )
    store.update_job(parent["id"], status="failed", error="input invalid")
    assert store.propagate_dependency_failures() == 1
    failed = store.get_job(child["id"])
    assert failed["status"] == "failed"
    assert parent["id"] in failed["error"]


def test_running_cancellation_is_cooperative_and_worker_observable(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    _, job = _project_and_job(store)
    claim = store.claim_job(job["id"])
    requested = store.cancel_job(job["id"])
    assert requested["status"] == "running"
    assert requested["cancel_requested"] is True
    assert store.job_cancel_requested(job["id"], claim.token) is True
    assert store.update_job(job["id"], status="cancelled", claim_token=claim.token)["status"] == "cancelled"


def test_job_priority_and_dependencies_persist_across_restart(tmp_path: Path) -> None:
    root = tmp_path / "workstation"
    store = WorkstationStore(root)
    project = store.create_project(kind="dataset", name="Dataset")
    parent = store.create_job(
        job_type="dataset.inventory", project_id=project["id"], priority=-5
    )
    child = store.create_job(
        job_type="dataset.inventory",
        project_id=project["id"],
        depends_on=[parent["id"]],
        priority=80,
    )

    reloaded = WorkstationStore(root).get_job(child["id"])
    assert reloaded["depends_on"] == [parent["id"]]
    assert reloaded["priority"] == 80
    assert reloaded["attempt"] == 1
    assert reloaded["retry_of"] is None
    with pytest.raises(WorkstationError, match="priority"):
        store.create_job(
            job_type="dataset.inventory", project_id=project["id"], priority=True
        )


def test_queued_job_can_pause_and_resume_without_a_worker_claim(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project, job = _project_and_job(store)

    paused = store.pause_job(job["id"])
    assert paused["status"] == "paused"
    assert paused["pause_requested"] is False
    assert store.get_project(project["id"])["status"] == "paused"
    assert store.get_project(project["id"])["metrics"]["jobs"]["paused"] == 1

    resumed = store.resume_job(job["id"])
    assert resumed["status"] == "queued"
    assert resumed["progress"] == 0.0
    assert resumed["result"] == {}
    assert resumed["error"] is None
    assert [event["message"] for event in store.list_job_logs(job["id"])][-2:] == [
        "Job paused",
        "Job resumed and queued",
    ]


def test_running_pause_is_cooperative_and_releases_gpu_claim(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="training", name="Training")
    job = store.create_job(job_type="training.prepare", project_id=project["id"])
    claim = store.claim_job(job["id"])

    requested = store.pause_job(job["id"])
    assert requested["status"] == "running"
    assert requested["pause_requested"] is True
    assert store.job_pause_requested(job["id"], claim.token) is True
    paused = store.update_job(job["id"], status="paused", claim_token=claim.token)
    assert paused["status"] == "paused"
    assert paused["pause_requested"] is False

    with sqlite3.connect(store.database_path) as connection:
        leases = connection.execute("SELECT COUNT(*) FROM resource_leases").fetchone()[0]
    assert leases == 0
    with pytest.raises(WorkstationError, match="running job"):
        store.heartbeat_job(job["id"], claim.token)


def test_retry_creates_a_traced_attempt_without_mutating_terminal_source(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project, job = _project_and_job(store)
    job = store.get_job(job["id"])
    failed = store.update_job(job["id"], status="failed", error="bad source")

    retried = store.retry_job(job["id"])
    assert retried["id"] != job["id"]
    assert retried["status"] == "queued"
    assert retried["attempt"] == 2
    assert retried["retry_of"] == job["id"]
    assert store.get_job(job["id"]) == failed
    assert "Retry queued as" in store.list_job_logs(job["id"])[-1]["message"]
    assert store.get_project(project["id"])["status"] == "queued"
    claim = store.claim_job(retried["id"])
    store.update_job(retried["id"], status="succeeded", claim_token=claim.token)
    assert store.get_project(project["id"])["status"] == "succeeded"
    with pytest.raises(WorkstationError, match="already retried"):
        store.retry_job(job["id"])
    with pytest.raises(WorkstationError, match="failed or cancelled"):
        store.retry_job(retried["id"])


def test_dataset_inventory_is_limited_to_configured_import_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "voice.wav").write_bytes(b"RIFF")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.wav").write_bytes(b"RIFF")
    monkeypatch.setenv("ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS", str(allowed))
    store = WorkstationStore(tmp_path / "workstation")
    accepted = store.create_project(kind="dataset", name="Allowed", config={"source": str(allowed)})
    accepted_job = store.create_job(job_type="dataset.inventory", project_id=accepted["id"])
    result = store.run_dataset_inventory(accepted_job["id"])
    assert result["status"] == "succeeded"
    assert result["result"]["media_files"] == 1
    rejected = store.create_project(kind="dataset", name="Rejected", config={"source": str(outside)})
    rejected_job = store.create_job(job_type="dataset.inventory", project_id=rejected["id"])
    with pytest.raises(WorkstationError, match="outside"):
        store.run_dataset_inventory(rejected_job["id"])
    assert store.get_job(rejected_job["id"])["status"] == "queued"


def test_target_dataset_media_expansion_accepts_files_and_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "allowed"
    nested = allowed / "recordings" / "session-a"
    nested.mkdir(parents=True)
    first = nested / "voice.wav"
    second = allowed / "interview.mp4"
    ignored = nested / "notes.txt"
    first.write_bytes(b"RIFF")
    second.write_bytes(b"video")
    ignored.write_text("not media", encoding="utf-8")
    monkeypatch.setenv("ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS", str(allowed))
    store = WorkstationStore(tmp_path / "workstation")

    paths = store.resolve_dataset_media_paths(
        [str(allowed / "recordings"), str(second), str(first)]
    )

    assert paths == tuple(sorted((first.resolve(), second.resolve()), key=lambda path: path.as_posix().casefold()))
    with pytest.raises(WorkstationError, match="1-file preparation limit"):
        store.resolve_dataset_media_paths([str(allowed)], maximum_files=1)


def test_inventory_environment_is_validated_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS", str(allowed))
    monkeypatch.setenv("ANIFLIVE_TTS_WORKSTATION_MAX_INVENTORY_FILES", "invalid")
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="dataset", name="Dataset", config={"source": str(allowed)})
    job = store.create_job(job_type="dataset.inventory", project_id=project["id"])
    with pytest.raises(WorkstationError, match="positive integer"):
        store.run_dataset_inventory(job["id"])
    assert store.get_job(job["id"])["status"] == "queued"
    assert store.get_job(job["id"])["started_at"] is None


def test_inventory_rejects_symlink_or_reparse_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed, outside = tmp_path / "allowed", tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (outside / "private.wav").write_bytes(b"RIFF")
    link = allowed / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows account cannot create test symbolic links")
    monkeypatch.setenv("ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS", str(allowed))
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="dataset", name="Dataset", config={"source": str(allowed)})
    job = store.create_job(job_type="dataset.inventory", project_id=project["id"])
    result = store.run_dataset_inventory(job["id"])
    assert result["status"] == "failed"
    assert "symbolic link or reparse point" in result["error"]


@pytest.mark.skipif(os.name != "nt", reason="Windows junction semantics")
def test_inventory_rejects_windows_junction_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed, outside = tmp_path / "allowed", tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (outside / "private.wav").write_bytes(b"RIFF")
    junction = allowed / "escape"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip("This Windows account cannot create a test junction")
    try:
        monkeypatch.setenv("ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS", str(allowed))
        store = WorkstationStore(tmp_path / "workstation")
        project = store.create_project(
            kind="dataset", name="Dataset", config={"source": str(allowed)}
        )
        job = store.create_job(job_type="dataset.inventory", project_id=project["id"])
        result = store.run_dataset_inventory(job["id"])
        assert result["status"] == "failed"
        assert "symbolic link or reparse point" in result["error"]
    finally:
        os.rmdir(junction)


def test_workstation_rejects_invalid_json_shapes_progress_and_ids(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    with pytest.raises(WorkstationError, match="JSON-compatible"):
        store.create_project(kind="dataset", name="Invalid", config={"path": object()})
    with pytest.raises(WorkstationError, match="JSON object"):
        store.create_project(kind="dataset", name="Invalid", config=[("x", 1)])  # type: ignore[arg-type]
    with pytest.raises(WorkstationError, match="too large"):
        store.create_project(kind="dataset", name="Large", config={"value": "x" * 70000})
    with pytest.raises(WorkstationError, match="malformed"):
        store.get_job("job_deadbeef")
    _, job = _project_and_job(store)
    claim = store.claim_job(job["id"])
    with pytest.raises(WorkstationError, match="finite"):
        store.update_job(job["id"], progress=float("nan"), claim_token=claim.token)
    with pytest.raises(WorkstationError, match="between 0 and 1"):
        store.update_job(job["id"], progress=1.5, claim_token=claim.token)


def test_snapshot_is_consistent_and_database_connections_are_released(tmp_path: Path) -> None:
    root = tmp_path / "workstation"
    store = WorkstationStore(root)
    _project_and_job(store)
    snapshot = store.snapshot()
    assert len(snapshot.projects) == 1
    assert len(snapshot.jobs) == 1
    store.database_path.rename(root / "renamed.sqlite3")
    assert (root / "renamed.sqlite3").is_file()


def test_artifact_registry_preserves_full_production_lineage(tmp_path: Path) -> None:
    root = tmp_path / "workstation"
    store = WorkstationStore(root)
    dataset_project = store.create_project(kind="dataset", name="Roxy Dataset")
    training_project = store.create_project(kind="training", name="Roxy V2ProPlus")
    evaluation_project = store.create_project(kind="evaluation", name="Qualification")

    def ready_file(relative: str, content: bytes) -> Path:
        path = store.artifact_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    dataset = store.register_artifact(
        artifact_type="dataset",
        name="roxy-clean-v8",
        status="ready",
        project_id=dataset_project["id"],
        local_path=ready_file("datasets/roxy-clean-v8.json", b"dataset"),
        metadata={"accepted_hours": 8.7, "split": {"train": 0.95, "validation": 0.05}},
    )
    checkpoint = store.register_artifact(
        artifact_type="checkpoint",
        name="v2proplus-epoch-15",
        status="ready",
        project_id=training_project["id"],
        local_path=ready_file("checkpoints/v2proplus-epoch-15.ckpt", b"checkpoint"),
        parent_artifact_ids=[dataset["id"]],
        metadata={"base": "GPT-SoVITS V2ProPlus", "epoch": 15},
    )
    expression_bank = store.register_artifact(
        artifact_type="expression-bank",
        name="expression-bank-v1",
        status="ready",
        project_id=training_project["id"],
        local_path=ready_file("expressions/expression-bank-v1.json", b"expressions"),
        parent_artifact_ids=[dataset["id"], checkpoint["id"]],
        metadata={"profiles": ["neutral", "happy", "relieved"]},
    )
    evaluation = store.register_artifact(
        artifact_type="evaluation",
        name="qualification-2026-08-30",
        status="ready",
        project_id=evaluation_project["id"],
        local_path=ready_file("evaluations/qualification.json", b"evaluation"),
        parent_artifact_ids=[checkpoint["id"], expression_bank["id"]],
        metadata={"five_languages": "pass", "speaker_cosine": 0.992},
    )
    engine_path = store.artifact_root / "engines" / "rtx-5070-ti.plan"
    engine_path.parent.mkdir(parents=True)
    engine_path.write_bytes(b"test-only-engine-placeholder")
    engine = store.register_artifact(
        artifact_type="engine",
        name="rtx-5070-ti-cuda-12.8",
        status="ready",
        project_id=training_project["id"],
        local_path=engine_path,
        sha256=hashlib.sha256(engine_path.read_bytes()).hexdigest(),
        parent_artifact_ids=[checkpoint["id"], evaluation["id"]],
        metadata={"runtime": "TensorRT 11"},
    )
    package = store.register_artifact(
        artifact_type="package",
        name="roxy-v2proplus",
        status="planned",
        project_id=training_project["id"],
        parent_artifact_ids=[engine["id"], expression_bank["id"], evaluation["id"]],
        metadata={"schema": 2},
    )

    assert engine["local_path"] == "engines/rtx-5070-ti.plan"
    assert engine["sha256"] == hashlib.sha256(engine_path.read_bytes()).hexdigest()
    assert package["parent_artifact_ids"] == [
        engine["id"], expression_bank["id"], evaluation["id"]
    ]
    assert store.list_artifacts(artifact_type="checkpoint") == [checkpoint]
    assert {item["id"] for item in store.list_artifacts(status="ready")} == {
        dataset["id"], checkpoint["id"], expression_bank["id"], evaluation["id"], engine["id"]
    }
    reloaded = WorkstationStore(root)
    assert reloaded.get_artifact(package["id"]) == package


def test_artifact_registry_rejects_invalid_contract_values(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="training", name="Training")
    with pytest.raises(WorkstationError, match="Unsupported artifact type"):
        store.register_artifact(artifact_type="weights", name="Invalid")
    with pytest.raises(WorkstationError, match="Unsupported artifact status"):
        store.register_artifact(artifact_type="checkpoint", name="Invalid", status="published")
    with pytest.raises(WorkstationError, match="64 hexadecimal"):
        store.register_artifact(artifact_type="checkpoint", name="Invalid", sha256="deadbeef")
    with pytest.raises(WorkstationError, match="JSON object"):
        store.register_artifact(
            artifact_type="checkpoint", name="Invalid", metadata=[("epoch", 1)]  # type: ignore[arg-type]
        )
    with pytest.raises(WorkstationError, match="Project was not found"):
        store.register_artifact(
            artifact_type="checkpoint", name="Invalid", project_id=f"training_{uuid4()}"
        )
    with pytest.raises(WorkstationError, match="Parent artifact was not found"):
        store.register_artifact(
            artifact_type="checkpoint", name="Invalid",
            parent_artifact_ids=[f"artifact_{uuid4()}"],
        )
    parent = store.register_artifact(artifact_type="dataset", name="Dataset")
    with pytest.raises(WorkstationError, match="duplicates"):
        store.register_artifact(
            artifact_type="checkpoint", name="Invalid",
            parent_artifact_ids=[parent["id"], parent["id"]],
        )
    self_id = f"artifact_{uuid4()}"
    with pytest.raises(WorkstationError, match="own parent"):
        store.register_artifact(
            artifact_id=self_id, artifact_type="checkpoint", name="Invalid",
            project_id=project["id"], parent_artifact_ids=[self_id],
        )


def test_artifact_local_path_is_relative_contained_and_reparse_safe(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    contained = store.artifact_root / "packages" / "voice.pkg"
    contained.parent.mkdir(parents=True)
    contained.write_bytes(b"package")
    artifact = store.register_artifact(
        artifact_type="package", name="Voice", local_path=contained
    )
    assert artifact["local_path"] == "packages/voice.pkg"
    assert artifact["sha256"] == hashlib.sha256(b"package").hexdigest()
    assert not Path(artifact["local_path"]).is_absolute()
    with pytest.raises(WorkstationError, match="contained"):
        store.register_artifact(
            artifact_type="package", name="Outside", local_path=tmp_path / "outside.pkg"
        )
    with pytest.raises(WorkstationError, match="traverse"):
        store.register_artifact(
            artifact_type="package", name="Traversal", local_path="nested/../../outside.pkg"
        )
    missing = store.artifact_root / "packages" / "missing.pkg"
    with pytest.raises(WorkstationError, match="does not exist"):
        store.register_artifact(
            artifact_type="package", name="Missing", local_path=missing
        )
    with pytest.raises(WorkstationError, match="regular file"):
        store.register_artifact(
            artifact_type="package", name="Directory", local_path=contained.parent
        )
    with pytest.raises(WorkstationError, match="does not match"):
        store.register_artifact(
            artifact_type="package", name="Wrong checksum", local_path=contained,
            sha256="0" * 64,
        )
    with pytest.raises(WorkstationError, match="requires an existing"):
        store.register_artifact(
            artifact_type="package", name="Checksum only", sha256="0" * 64
        )
    with pytest.raises(WorkstationError, match="Ready artifacts require"):
        store.register_artifact(
            artifact_type="package", name="Ready without file", status="ready"
        )


@pytest.mark.parametrize(
    "local_path",
    (
        "artifact.bin:stream",
        "CON",
        "prn.txt",
        "folder./artifact.bin",
        "folder /artifact.bin",
        "artifact.bin.",
        "artifact.bin ",
    ),
)
def test_artifact_store_rejects_windows_special_relative_paths(
    tmp_path: Path, local_path: str
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    with pytest.raises(WorkstationError, match="portable to Windows"):
        store.register_artifact(
            artifact_type="checkpoint",
            name="Invalid portable path",
            status="ready",
            local_path=local_path,
        )


def test_artifact_registration_rejects_symlinked_file_or_parent(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    outside = tmp_path / "outside.pkg"
    outside.write_bytes(b"outside")
    linked = store.artifact_root / "linked.pkg"
    linked_parent = store.artifact_root / "linked-parent"
    try:
        linked.symlink_to(outside)
        linked_parent.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this Windows host")
    with pytest.raises(WorkstationError, match="symbolic link|reparse"):
        store.register_artifact(
            artifact_type="package", name="Linked file", local_path=linked
        )
    with pytest.raises(WorkstationError, match="symbolic link|reparse"):
        store.register_artifact(
            artifact_type="package",
            name="Linked parent",
            local_path=linked_parent / outside.name,
        )


def test_worker_artifact_import_is_host_verified_idempotent_and_type_bounded(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="training", name="Training")
    parent = store.register_artifact(artifact_type="dataset", name="Dataset")
    source_root = tmp_path / "docker-output"
    source_file = source_root / "artifacts" / "checkpoint.bin"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"checkpoint")
    digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
    declared = [
        {
            "kind": "checkpoint",
            "relative_path": "artifacts/checkpoint.bin",
            "sha256": digest,
            "size_bytes": source_file.stat().st_size,
        }
    ]
    job = store.create_job(
        job_type="training.prepare", project_id=project["id"]
    )
    job_id = job["id"]
    claim = store.claim_job(job_id)
    arguments = {
        "job_id": job_id,
        "claim_token": claim.token,
        "job_type": "training.prepare",
        "project_id": project["id"],
        "source_root": source_root,
        "artifacts": declared,
        "image_digest": "sha256:" + "a" * 64,
        "parent_artifact_ids": [parent["id"]],
    }

    first = store.register_worker_artifacts(**arguments)
    registered = first[0]
    registered_path = store.artifact_root.joinpath(*Path(registered["local_path"]).parts)
    registered_path.unlink()
    second = store.register_worker_artifacts(**arguments)
    assert second == first
    assert len(store.list_artifacts(artifact_type="checkpoint")) == 1
    assert registered_path.is_file()
    assert registered_path.read_bytes() == b"checkpoint"
    assert registered["sha256"] == digest
    assert registered["status"] == "building"
    assert store.list_artifacts(status="ready") == []

    wrong_digest = [{**declared[0], "sha256": "0" * 64}]
    with pytest.raises(WorkstationError, match="wrong checksum"):
        store.register_worker_artifacts(**{**arguments, "artifacts": wrong_digest})
    wrong_kind = [{**declared[0], "kind": "engine"}]
    with pytest.raises(WorkstationError, match="may publish only"):
        store.register_worker_artifacts(**{**arguments, "artifacts": wrong_kind})
    with pytest.raises(WorkstationError, match="job does not match"):
        store.register_worker_artifacts(
            **{**arguments, "job_type": "engine.prepare"}
        )
    with pytest.raises(WorkstationError, match="lineage does not match"):
        store.register_worker_artifacts(
            **{**arguments, "image_digest": "sha256:" + "b" * 64}
        )


def test_reference_selection_publishes_only_reference_artifacts(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="training", name="Training")
    source_root = tmp_path / "docker-output"
    source_file = source_root / "blind-reference-manifest.json"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
    job = store.create_job(job_type="reference.select", project_id=project["id"])
    claim = store.claim_job(job["id"])
    arguments = {
        "job_id": job["id"],
        "claim_token": claim.token,
        "job_type": "reference.select",
        "project_id": project["id"],
        "source_root": source_root,
        "artifacts": [
            {
                "kind": "reference",
                "relative_path": source_file.name,
                "sha256": digest,
                "size_bytes": source_file.stat().st_size,
            }
        ],
        "image_digest": "sha256:" + "a" * 64,
    }

    registered = store.register_worker_artifacts(**arguments)
    assert len(registered) == 1
    assert registered[0]["type"] == "reference"

    with pytest.raises(WorkstationError, match="may publish only"):
        store.register_worker_artifacts(
            **{
                **arguments,
                "artifacts": [{**arguments["artifacts"][0], "kind": "checkpoint"}],
            }
        )


def test_worker_artifacts_promote_only_with_job_success_and_cancel_cleans_files(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="training", name="Training")
    source_root = tmp_path / "docker-output"
    source_root.mkdir()
    source_file = source_root / "checkpoint.bin"
    source_file.write_bytes(b"checkpoint")
    declared = [
        {
            "kind": "checkpoint",
            "relative_path": source_file.name,
            "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
            "size_bytes": source_file.stat().st_size,
        }
    ]

    success_job = store.create_job(
        job_type="training.prepare", project_id=project["id"]
    )
    success_claim = store.claim_job(success_job["id"])
    staged = store.register_worker_artifacts(
        job_id=success_job["id"],
        claim_token=success_claim.token,
        job_type="training.prepare",
        project_id=project["id"],
        source_root=source_root,
        artifacts=declared,
        image_digest="sha256:" + "a" * 64,
    )
    assert staged[0]["status"] == "building"
    assert store.list_artifacts(status="ready") == []
    completed = store.update_job(
        success_job["id"],
        status="succeeded",
        result={"registered_artifact_ids": [staged[0]["id"]]},
        claim_token=success_claim.token,
    )
    assert completed["status"] == "succeeded"
    assert store.get_artifact(staged[0]["id"])["status"] == "ready"

    cancel_job = store.create_job(
        job_type="training.prepare", project_id=project["id"]
    )
    cancel_claim = store.claim_job(cancel_job["id"])
    cancelled_staged = store.register_worker_artifacts(
        job_id=cancel_job["id"],
        claim_token=cancel_claim.token,
        job_type="training.prepare",
        project_id=project["id"],
        source_root=source_root,
        artifacts=declared,
        image_digest="sha256:" + "a" * 64,
    )
    cancelled_path = store.artifact_root.joinpath(
        *PurePosixPath(cancelled_staged[0]["local_path"]).parts
    )
    store.cancel_job(cancel_job["id"])
    cancelled = store.update_job(
        cancel_job["id"],
        status="succeeded",
        result={"registered_artifact_ids": [cancelled_staged[0]["id"]]},
        claim_token=cancel_claim.token,
    )
    assert cancelled["status"] == "cancelled"
    assert "registered_artifact_ids" not in cancelled["result"]
    assert all(
        artifact["id"] != cancelled_staged[0]["id"]
        for artifact in store.list_artifacts()
    )
    assert not cancelled_path.exists()


def test_failed_quality_gate_retains_verified_artifacts_as_rejected(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="training", name="Training")
    job = store.create_job(job_type="holdout.evaluate", project_id=project["id"])
    claim = store.claim_job(job["id"])
    source_root = tmp_path / "docker-output"
    source_root.mkdir()
    source_file = source_root / "holdout-evaluation.json"
    source_file.write_text('{"status":"failed"}\n', encoding="utf-8")
    staged = store.register_worker_artifacts(
        job_id=job["id"],
        claim_token=claim.token,
        job_type="holdout.evaluate",
        project_id=project["id"],
        source_root=source_root,
        artifacts=[
            {
                "kind": "evaluation",
                "relative_path": source_file.name,
                "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
                "size_bytes": source_file.stat().st_size,
            }
        ],
        image_digest="sha256:" + "a" * 64,
    )
    staged_path = store.artifact_root.joinpath(
        *PurePosixPath(staged[0]["local_path"]).parts
    )

    failed = store.update_job(
        job["id"],
        status="failed",
        result={"diagnostic_artifact_ids": [staged[0]["id"]]},
        error="quality gate failed",
        claim_token=claim.token,
    )

    assert failed["status"] == "failed"
    assert failed["result"]["diagnostic_artifact_ids"] == [staged[0]["id"]]
    assert store.get_artifact(staged[0]["id"])["status"] == "rejected"
    assert staged_path.read_text(encoding="utf-8") == '{"status":"failed"}\n'
    assert store.list_artifacts(status="ready") == []


def test_worker_artifact_promotion_failure_cannot_leave_ready_artifacts(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="training", name="Training")
    job = store.create_job(job_type="training.prepare", project_id=project["id"])
    claim = store.claim_job(job["id"])
    source_root = tmp_path / "docker-output"
    source_root.mkdir()
    source_file = source_root / "checkpoint.bin"
    source_file.write_bytes(b"checkpoint")
    staged = store.register_worker_artifacts(
        job_id=job["id"],
        claim_token=claim.token,
        job_type="training.prepare",
        project_id=project["id"],
        source_root=source_root,
        artifacts=[
            {
                "kind": "checkpoint",
                "relative_path": source_file.name,
                "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
                "size_bytes": source_file.stat().st_size,
            }
        ],
        image_digest="sha256:" + "a" * 64,
    )
    staged_path = store.artifact_root.joinpath(
        *PurePosixPath(staged[0]["local_path"]).parts
    )
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_worker_promotion BEFORE UPDATE OF status ON artifacts "
            "WHEN NEW.status = 'ready' BEGIN SELECT RAISE(ABORT, 'reject'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="reject"):
        store.update_job(
            job["id"],
            status="succeeded",
            result={"registered_artifact_ids": [staged[0]["id"]]},
            claim_token=claim.token,
        )
    assert store.get_job(job["id"])["status"] == "running"
    assert store.get_artifact(staged[0]["id"])["status"] == "building"
    store.update_job(
        job["id"], status="failed", error="promotion failed", claim_token=claim.token
    )
    assert store.list_artifacts(status="ready") == []
    assert not staged_path.exists()


def test_worker_artifact_identity_is_stable_when_manifest_order_changes(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="training", name="Training")
    job = store.create_job(
        job_type="training.prepare", project_id=project["id"]
    )
    claim = store.claim_job(job["id"])
    source_root = tmp_path / "docker-output"
    first_path = source_root / "artifacts" / "first.bin"
    second_path = source_root / "artifacts" / "second.bin"
    first_path.parent.mkdir(parents=True)
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")

    def declared(path: Path) -> dict[str, object]:
        return {
            "kind": "checkpoint",
            "relative_path": f"artifacts/{path.name}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }

    artifacts = [declared(first_path), declared(second_path)]
    arguments = {
        "job_id": job["id"],
        "claim_token": claim.token,
        "job_type": "training.prepare",
        "project_id": project["id"],
        "source_root": source_root,
        "image_digest": "sha256:" + "a" * 64,
    }
    first = store.register_worker_artifacts(**arguments, artifacts=artifacts)
    retried = store.register_worker_artifacts(
        **arguments, artifacts=list(reversed(artifacts))
    )

    first_by_name = {artifact["name"]: artifact for artifact in first}
    retry_by_name = {artifact["name"]: artifact for artifact in retried}
    assert retry_by_name == first_by_name
    assert len(store.list_artifacts(artifact_type="checkpoint")) == 2


def test_worker_artifact_destination_rejects_linked_parent_before_copy(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="training", name="Training")
    job = store.create_job(
        job_type="training.prepare", project_id=project["id"]
    )
    claim = store.claim_job(job["id"])
    source_root = tmp_path / "docker-output"
    source_file = source_root / "artifact.bin"
    source_root.mkdir()
    source_file.write_bytes(b"checkpoint")
    outside = tmp_path / "outside"
    outside.mkdir()
    worker_parent = store.artifact_root / "worker"
    try:
        worker_parent.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this Windows host")

    with pytest.raises(WorkstationError, match="symbolic link|reparse"):
        store.register_worker_artifacts(
            job_id=job["id"],
            claim_token=claim.token,
            job_type="training.prepare",
            project_id=project["id"],
            source_root=source_root,
            artifacts=[
                {
                    "kind": "checkpoint",
                    "relative_path": source_file.name,
                    "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
                    "size_bytes": source_file.stat().st_size,
                }
            ],
            image_digest="sha256:" + "a" * 64,
        )
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("race", ["cancel", "expire"])
def test_worker_artifact_import_rechecks_claim_after_copy_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, race: str
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="training", name="Training")
    job = store.create_job(
        job_type="training.prepare", project_id=project["id"]
    )
    claim = store.claim_job(job["id"])
    source_root = tmp_path / "docker-output"
    source_file = source_root / "checkpoint.bin"
    source_root.mkdir()
    source_file.write_bytes(b"checkpoint")
    original_copy = workstation_module._copy_verified_file

    def copy_then_lose_claim(*args, **kwargs):
        created = original_copy(*args, **kwargs)
        if race == "cancel":
            store.cancel_job(job["id"])
        else:
            with sqlite3.connect(store.database_path) as connection:
                connection.execute(
                    "UPDATE jobs SET lease_expires_at = ? WHERE id = ?",
                    ("1970-01-01T00:00:00.000000Z", job["id"]),
                )
        return created

    monkeypatch.setattr(workstation_module, "_copy_verified_file", copy_then_lose_claim)
    expected = "cancellation was requested" if race == "cancel" else "lease has expired"
    with pytest.raises(WorkstationError, match=expected):
        store.register_worker_artifacts(
            job_id=job["id"],
            claim_token=claim.token,
            job_type="training.prepare",
            project_id=project["id"],
            source_root=source_root,
            artifacts=[
                {
                    "kind": "checkpoint",
                    "relative_path": source_file.name,
                    "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
                    "size_bytes": source_file.stat().st_size,
                }
            ],
            image_digest="sha256:" + "a" * 64,
        )

    assert store.list_artifacts(artifact_type="checkpoint") == []
    worker_root = store.artifact_root / "worker" / job["id"]
    assert not worker_root.exists() or not any(
        path.is_file() for path in worker_root.rglob("*")
    )


def test_artifact_registration_is_atomic_and_identity_is_immutable(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    parent = store.register_artifact(artifact_type="dataset", name="Dataset")
    rejected_id = f"artifact_{uuid4()}"
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_artifact_parent BEFORE INSERT ON artifact_parents "
            "BEGIN SELECT RAISE(ABORT, 'parent rejected'); END"
        )
    with pytest.raises(WorkstationError, match="registry contract"):
        store.register_artifact(
            artifact_id=rejected_id,
            artifact_type="checkpoint",
            name="Rejected",
            parent_artifact_ids=[parent["id"]],
        )
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute(
            "SELECT 1 FROM artifacts WHERE id = ?", (rejected_id,)
        ).fetchone() is None
        connection.execute("DROP TRIGGER reject_artifact_parent")
    engine_path = store.artifact_root / "engines" / "test.plan"
    engine_path.parent.mkdir(parents=True)
    engine_path.write_bytes(b"engine")
    artifact = store.register_artifact(
        artifact_type="engine", name="Engine", local_path=engine_path
    )
    with sqlite3.connect(store.database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE artifacts SET sha256 = ? WHERE id = ?", ("2" * 64, artifact["id"])
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE artifacts SET local_path = ? WHERE id = ?",
                ("engines/other.plan", artifact["id"]),
            )


def test_artifact_lineage_cycle_is_blocked_by_database_contract(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    first = store.register_artifact(artifact_type="dataset", name="First")
    second = store.register_artifact(
        artifact_type="checkpoint", name="Second", parent_artifact_ids=[first["id"]]
    )
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError, match="cycle"):
            connection.execute(
                "INSERT INTO artifact_parents(artifact_id, parent_artifact_id, position, created_at) "
                "VALUES (?, ?, 0, '2026-08-30T00:00:00.000Z')",
                (first["id"], second["id"]),
            )


def test_schema_four_database_migrates_to_expression_bank(tmp_path: Path) -> None:
    root = tmp_path / "workstation"
    original = WorkstationStore(root)
    project = original.create_project(kind="training", name="Existing Project")
    artifact = original.register_artifact(artifact_type="checkpoint", name="Existing Checkpoint")
    with sqlite3.connect(original.database_path) as connection:
        connection.execute("DROP TABLE expression_drafts")
        connection.execute("UPDATE metadata SET value = '4' WHERE key = 'schema'")
    migrated = WorkstationStore(root)
    assert migrated.get_project(project["id"])["name"] == "Existing Project"
    assert migrated.get_artifact(artifact["id"])["name"] == "Existing Checkpoint"
    with sqlite3.connect(migrated.database_path) as connection:
        schema = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema'"
        ).fetchone()
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'expression_drafts'"
        ).fetchone()
    assert schema == ("9",)
    assert table == ("expression_drafts",)


def test_expression_draft_crud_and_filters_survive_restart(tmp_path: Path) -> None:
    root = tmp_path / "workstation"
    store = WorkstationStore(root)
    created = store.create_expression_draft(
        name="Relieved, quiet",
        profile_id="relieved.quiet",
        model_id="v2proplus-local-test",
        language="ja",
        emotion="relieved",
        intensity=0.72,
        descriptions=["Soft release of tension", "Low-energy delivery"],
        vad={"valence": 0.44, "arousal": 0.18, "dominance": 0.35},
        prosody={"f0_median_hz": 181.0, "speaking_rate": 3.7},
        qualification_status="pending",
    )
    assert created["id"].startswith("expression_")
    assert len(created["id"].partition("_")[2]) == 36
    assert created["reference"] is None
    assert created["vad"]["arousal"] == pytest.approx(0.18)
    with pytest.raises(WorkstationError, match="evaluation evidence"):
        store.update_expression_draft(
            created["id"], qualification_status="qualified"
        )
    updated = store.update_expression_draft(
        created["id"],
        name="Relieved",
        intensity=0.8,
        descriptions=["Warm and relieved"],
        vad={"valence": 0.5, "arousal": 0.2, "dominance": 0.35},
        prosody={"speaking_rate": 3.6},
        qualification_status="pending",
    )
    assert updated["profile_id"] == created["profile_id"]
    assert updated["model_id"] == created["model_id"]
    assert updated["intensity"] == pytest.approx(0.8)
    assert updated["descriptions"] == ["Warm and relieved"]
    assert store.list_expression_drafts(
        model_id="v2proplus-local-test", language="JA", emotion="relieved",
        qualification_status="pending",
    ) == [updated]
    reloaded = WorkstationStore(root)
    assert reloaded.get_expression_draft(created["id"]) == updated
    deleted = reloaded.delete_expression_draft(created["id"])
    assert deleted == updated
    with pytest.raises(WorkstationError, match="not found"):
        reloaded.get_expression_draft(created["id"])


def test_expression_reference_identity_is_contained_hashed_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_root = tmp_path / "references"
    import_root.mkdir()
    imported_reference = import_root / "happy.wav"
    imported_reference.write_bytes(b"test-reference-audio")
    monkeypatch.setenv("ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS", str(import_root))
    store = WorkstationStore(tmp_path / "workstation")
    artifact_reference = store.artifact_root / "expressions" / "neutral.wav"
    artifact_reference.parent.mkdir(parents=True)
    artifact_reference.write_bytes(b"artifact-reference-audio")

    artifact_scoped = store.create_expression_draft(
        name="Neutral", profile_id="neutral", language="ja", emotion="neutral",
        intensity=0.0, reference_path=artifact_reference,
    )
    imported = store.create_expression_draft(
        name="Happy", profile_id="happy", model_id="voice-a", language="ja",
        emotion="happy", intensity=0.9, reference_path=imported_reference,
    )
    assert artifact_scoped["reference"] == {
        "scope": "artifact",
        "path": "expressions/neutral.wav",
        "sha256": hashlib.sha256(b"artifact-reference-audio").hexdigest(),
    }
    assert imported["reference"]["scope"] == "import-root"
    assert imported["reference"]["path"] == "happy.wav"
    assert len(imported["reference"]["root_id"]) == 16
    assert imported["reference"]["sha256"] == hashlib.sha256(
        b"test-reference-audio"
    ).hexdigest()
    assert str(import_root.resolve()) not in json.dumps(imported)

    outside = tmp_path / "private.wav"
    outside.write_bytes(b"outside")
    with pytest.raises(WorkstationError, match="outside"):
        store.create_expression_draft(
            name="Outside", profile_id="outside", language="ja", emotion="neutral",
            intensity=0.2, reference_path=outside,
        )
    with sqlite3.connect(store.database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE expression_drafts SET reference_sha256 = ? WHERE id = ?",
                ("0" * 64, imported["id"]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE expression_drafts SET profile_id = 'changed' WHERE id = ?",
                (imported["id"],),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE expression_drafts SET reference_path = 'other.wav' WHERE id = ?",
                (imported["id"],),
            )
    store.delete_expression_draft(imported["id"])
    assert imported_reference.read_bytes() == b"test-reference-audio"


def test_expression_draft_rejects_invalid_ranges_shapes_and_duplicates(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    required = {
        "name": "Happy", "profile_id": "happy", "language": "ja",
        "emotion": "happy", "intensity": 0.8,
    }
    for invalid in (-0.1, 1.1, float("nan"), float("inf"), True):
        with pytest.raises(WorkstationError, match="finite number"):
            store.create_expression_draft(**{**required, "intensity": invalid})
    with pytest.raises(WorkstationError, match="profile_id"):
        store.create_expression_draft(**{**required, "profile_id": "Happy Face"})
    with pytest.raises(WorkstationError, match="language"):
        store.create_expression_draft(**{**required, "language": "Japanese"})
    with pytest.raises(WorkstationError, match="unsupported fields"):
        store.create_expression_draft(**required, vad={"joy": 0.8})
    for invalid_vad in (1.1, float("nan"), float("inf")):
        with pytest.raises(WorkstationError, match="vad.valence"):
            store.create_expression_draft(**required, vad={"valence": invalid_vad})
    with pytest.raises(WorkstationError, match="list of strings"):
        store.create_expression_draft(**required, descriptions="bright")  # type: ignore[arg-type]
    with pytest.raises(WorkstationError, match="32 entries"):
        store.create_expression_draft(**required, descriptions=["x"] * 33)
    with pytest.raises(WorkstationError, match="JSON-compatible"):
        store.create_expression_draft(**required, prosody={"energy": float("nan")})
    with pytest.raises(WorkstationError, match="qualification status"):
        store.create_expression_draft(**required, qualification_status="published")
    invalid_reference = store.artifact_root / "reference.txt"
    invalid_reference.write_text("not audio", encoding="utf-8")
    with pytest.raises(WorkstationError, match="audio extension"):
        store.create_expression_draft(**required, reference_path=invalid_reference)

    first = store.create_expression_draft(**required, model_id="voice-a")
    with pytest.raises(WorkstationError, match="already exists"):
        store.create_expression_draft(**required, model_id="voice-a")
    other_model = store.create_expression_draft(**required, model_id="voice-b")
    assert first["profile_id"] == other_model["profile_id"]
    with pytest.raises(WorkstationError, match="At least one"):
        store.update_expression_draft(first["id"])
    with pytest.raises(WorkstationError, match="malformed"):
        store.get_expression_draft("expression_deadbeef")



def test_lineage_leaf_insert_has_bounded_work_and_rejects_real_cycles(tmp_path: Path) -> None:
    from uuid import UUID

    store = WorkstationStore(tmp_path / "workstation")
    ids = [f"artifact_{UUID(int=i + 1)}" for i in range(514)]
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executemany(
            "INSERT INTO artifacts(id,type,name,status,metadata_json,created_at,updated_at) "
            "VALUES (?,'checkpoint','Node','planned','{}','2026-09-07','2026-09-07')",
            [(value,) for value in ids],
        )
        connection.executemany(
            "INSERT INTO artifact_parents(artifact_id,parent_artifact_id,position,created_at) "
            "VALUES (?,?,0,'2026-09-07')",
            [(ids[i], ids[i - 1]) for i in range(1, 513)],
        )
        steps = []
        connection.set_progress_handler(lambda: steps.append(1) or 0, 100)
        connection.execute(
            "INSERT INTO artifact_parents(artifact_id,parent_artifact_id,position,created_at) "
            "VALUES (?,?,0,'2026-09-07')", (ids[513], ids[512]),
        )
        connection.set_progress_handler(None, 0)
        assert len(steps) < 20
        with pytest.raises(sqlite3.IntegrityError, match="cycle"):
            connection.execute(
                "INSERT INTO artifact_parents(artifact_id,parent_artifact_id,position,created_at) "
                "VALUES (?,?,1,'2026-09-07')", (ids[513], ids[513]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="cycle"):
            connection.execute(
                "INSERT INTO artifact_parents(artifact_id,parent_artifact_id,position,created_at) "
                "VALUES (?,?,0,'2026-09-07')", (ids[0], ids[513]),
            )
        with pytest.raises(WorkstationError, match="cycle"):
            store._validate_artifact_parents(connection, ids[0], [ids[513]])
        with pytest.raises(WorkstationError, match="own parent"):
            store._validate_artifact_parents(connection, ids[513], [ids[513]])
        with pytest.raises(WorkstationError, match="not found"):
            store._validate_artifact_parents(connection, ids[513], ["artifact_missing"])


def test_existing_cycle_trigger_upgrades_without_changing_lineage(tmp_path: Path) -> None:
    root = tmp_path / "workstation"
    store = WorkstationStore(root)
    parent = store.register_artifact(artifact_type="dataset", name="Parent")
    child = store.register_artifact(
        artifact_type="checkpoint", name="Child", parent_artifact_ids=[parent["id"]]
    )
    with sqlite3.connect(store.database_path) as connection:
        sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='artifact_parents_no_cycle'"
        ).fetchone()[0]
        legacy = sql[:sql.index("WHEN")] + sql[sql.index("BEGIN"):]
        connection.execute("DROP TRIGGER artifact_parents_no_cycle")
        connection.execute(legacy)
    upgraded = WorkstationStore(root)
    assert upgraded.get_artifact(child["id"]) == child
    with sqlite3.connect(upgraded.database_path) as connection:
        sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='artifact_parents_no_cycle'"
        ).fetchone()[0]
        assert "WHERE parent_artifact_id = NEW.artifact_id" in sql
        with pytest.raises(sqlite3.IntegrityError, match="cycle"):
            connection.execute(
                "INSERT INTO artifact_parents(artifact_id,parent_artifact_id,position,created_at) "
                "VALUES (?,?,0,'2026-09-07')", (parent["id"], child["id"]),
            )
