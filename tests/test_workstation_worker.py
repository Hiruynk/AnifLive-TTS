from __future__ import annotations

import hashlib
import json
import signal
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from aniflive_tts import workstation_worker as workstation_worker_module
from aniflive_tts.cli import build_parser, main
from aniflive_tts.workstation import GPU_RESOURCE_KEY, WorkstationError, WorkstationStore
from aniflive_tts.workstation_adapters import AdapterCancelled, AdapterResult
from aniflive_tts.workstation_docker import DockerCleanupUncertain
from aniflive_tts.workstation_worker import WorkstationWorker


def _dataset_job(store: WorkstationStore, source: Path, *, name: str = "Dataset"):
    project = store.create_project(
        kind="dataset", name=name, config={"source": str(source)}
    )
    job = store.create_job(job_type="dataset.inventory", project_id=project["id"])
    return project, job


def _training_config(tmp_path: Path) -> dict[str, str]:
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    inventory: list[dict[str, object]] = []
    for split, count in (("train", 2), ("validation", 1), ("test", 1)):
        audio_dir = dataset / split / "wav"
        audio_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, object]] = []
        for index in range(count):
            audio = audio_dir / f"{split}-{index}.wav"
            audio.write_bytes(f"RIFF-{split}-{index}".encode())
            record = {
                "source_item_id": f"item-{split}-{index}",
                "path": f"{split}/wav/{audio.name}",
                "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "split": split,
                "transcript": f"{split} {index}",
                "language": "ja",
                "speaker": "voice",
            }
            records.append(record)
            inventory.append(record)
        (dataset / split / "manifest.json").write_text(
            json.dumps(
                {"split": split, "count": count, "items": records},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    training_list = dataset / "train" / "voice.list"
    training_list.write_text(
        "train-0.wav|voice|ja|zero\ntrain-1.wav|voice|ja|one\n",
        encoding="utf-8",
    )
    (dataset / "training-input.json").write_text(
        json.dumps(
            {
                "schema": "aniflive-v2proplus-training-input-v2",
                "dataset_id": "dataset-test",
                "frozen_manifest_sha256": "a" * 64,
                "training_list_sha256": hashlib.sha256(
                    training_list.read_bytes()
                ).hexdigest(),
                "audio_files": inventory,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(exist_ok=True)
    config = {"dataset": str(dataset), "shared_dir": str(shared_dir)}
    for key, filename in (
        ("pretrained_gpt", "pretrained-gpt.ckpt"),
        ("pretrained_sovits_g", "pretrained-sovits-g.pth"),
        ("pretrained_sovits_d", "pretrained-sovits-d.pth"),
    ):
        path = tmp_path / filename
        path.write_bytes(b"fixed-test-weight")
        config[key] = str(path)
    return config


def test_safe_error_preserves_context_and_final_cause() -> None:
    error = RuntimeError("CONTEXT:" + (" progress" * 300) + ":FINAL_CAUSE")

    message = workstation_worker_module._safe_error(error)

    assert len(message) == 1000
    assert message.startswith("CONTEXT:")
    assert "[middle omitted]" in message
    assert message.endswith(":FINAL_CAUSE")


def test_once_worker_claims_and_completes_dataset_inventory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "voice.wav").write_bytes(b"RIFF")
    store = WorkstationStore(tmp_path / "state")
    _project, job = _dataset_job(store, source)
    reports: list[dict] = []
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        reporter=lambda record: reports.append(dict(record)),
        heartbeat_seconds=0.05,
    )
    summary = worker.run(once=True)
    record = store.get_job(job["id"])
    assert summary.completed == 1
    assert summary.processed == 1
    assert record["status"] == "succeeded"
    assert record["result"]["media_files"] == 1
    assert any(report["kind"] == "job-event" for report in reports)
    assert reports[-1]["disposition"] == "completed"


def test_once_worker_reports_idle_without_claimable_jobs(tmp_path: Path) -> None:
    worker = WorkstationWorker(
        WorkstationStore(tmp_path / "state"),
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )
    summary = worker.run(once=True)
    assert summary.idle is True
    assert summary.processed == 0


def test_prepare_without_backend_is_blocked_not_completed(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(
        kind="training", name="Training", config=_training_config(tmp_path)
    )
    job = store.create_job(job_type="training.prepare", project_id=project["id"])
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )
    summary = worker.run(once=True)
    record = store.get_job(job["id"])
    assert summary.blocked == 1
    assert summary.completed == 0
    assert record["status"] == "failed"
    assert record["result"]["execution_performed"] is False
    assert "trusted backend" in record["error"]
    assert Path(record["result"]["manifest_path"]).is_file()


def test_successful_dataset_training_queues_validation_checkpoint_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = WorkstationStore(tmp_path / "state")
    config = {**_training_config(tmp_path), "auto_build_production": True}
    project = store.create_project(kind="training", name="Voice", config=config)
    training = store.create_job(
        job_type="training.prepare", project_id=project["id"], priority=7
    )

    def successful_training(*_args, **_kwargs):
        return AdapterResult(
            "training.prepare",
            "gpu-exclusive",
            "completed",
            {"execution_performed": True},
        )

    monkeypatch.setattr(
        "aniflive_tts.workstation_worker.run_adapter", successful_training
    )
    result = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("training.prepare",),
    ).run_once()

    assert result.disposition == "completed"
    jobs = store.list_jobs()
    selection = next(job for job in jobs if job["type"] == "checkpoint.select")
    assert selection["depends_on"] == [training["id"]]
    assert selection["priority"] == 7
    assert not any(job["type"] == "engine.prepare" for job in jobs)


@pytest.mark.parametrize(
    "job_type", ["checkpoint.select", "holdout.evaluate", "conversion.parity"]
)
def test_failed_quality_gate_is_a_failed_job(
    tmp_path: Path, job_type: str
) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="training", name="Voice")
    job = store.create_job(job_type=job_type, project_id=project["id"])
    claim = store.claim_job(job["id"])
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=(job_type,),
    )

    result = worker._finish_result(
        job["id"],
        job_type,
        claim.token,
        AdapterResult(
            job_type,
            "gpu-exclusive",
            "completed",
            {"execution_performed": True, "status": "failed"},
        ),
    )

    assert result.disposition == "failed"
    assert store.get_job(job["id"])["status"] == "failed"
    assert not any(
        candidate["type"] in {"engine.prepare", "model.package"}
        for candidate in store.list_jobs()
    )


def test_failed_quality_gate_marks_staged_outputs_as_diagnostic_evidence(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="training", name="Voice")
    job = store.create_job(job_type="holdout.evaluate", project_id=project["id"])
    claim = store.claim_job(job["id"])
    output = tmp_path / "output"
    output.mkdir()
    report = output / "holdout-evaluation.json"
    report.write_text('{"status":"failed"}\n', encoding="utf-8")
    staged = store.register_worker_artifacts(
        job_id=job["id"],
        claim_token=claim.token,
        job_type="holdout.evaluate",
        project_id=project["id"],
        source_root=output,
        artifacts=[
            {
                "kind": "evaluation",
                "relative_path": report.name,
                "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "size_bytes": report.stat().st_size,
            }
        ],
        image_digest="sha256:" + "b" * 64,
    )
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("holdout.evaluate",),
    )

    result = worker._finish_result(
        job["id"],
        "holdout.evaluate",
        claim.token,
        AdapterResult(
            "holdout.evaluate",
            "gpu-exclusive",
            "completed",
            {
                "execution_performed": True,
                "status": "failed",
                "registered_artifact_ids": [staged[0]["id"]],
            },
        ),
    )

    record = store.get_job(job["id"])
    assert result.disposition == "failed"
    assert record["status"] == "failed"
    assert record["result"]["diagnostic_artifact_ids"] == [staged[0]["id"]]
    assert "registered_artifact_ids" not in record["result"]
    assert store.get_artifact(staged[0]["id"])["status"] == "rejected"
    decision_id = record["result"]["final_decision_artifact_id"]
    decision = store.get_artifact(decision_id)
    assert decision["status"] == "ready"
    assert decision["parent_artifact_ids"] == [staged[0]["id"]]
    payload = json.loads(
        (store.artifact_root / decision["local_path"]).read_text(encoding="utf-8")
    )
    assert payload["decision"] == "NO-GO"
    assert payload["failed_gate"] == "holdout.evaluate"
    assert payload["downstream_production_blocked"] is True
    updated_project = store.get_project(project["id"])
    assert updated_project["config"]["final_decision"] == "NO-GO"
    assert updated_project["config"]["final_decision_artifact_id"] == decision_id


@pytest.mark.parametrize("job_type", ["holdout.evaluate", "conversion.parity"])
def test_nested_docker_quality_gate_status_is_honored(
    tmp_path: Path, job_type: str
) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="training", name="Voice")
    job = store.create_job(job_type=job_type, project_id=project["id"])
    claim = store.claim_job(job["id"])
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=(job_type,),
    )

    result = worker._finish_result(
        job["id"],
        job_type,
        claim.token,
        AdapterResult(
            job_type,
            "gpu-exclusive",
            "completed",
            {
                "execution_performed": True,
                "backend": {"payload": {"status": "passed"}},
            },
        ),
    )

    assert result.disposition == "completed"
    assert store.get_job(job["id"])["status"] == "succeeded"


def test_adapter_exception_is_a_fenced_failed_terminal_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    store = WorkstationStore(tmp_path / "state")
    _project, job = _dataset_job(store, source)
    claim_tokens: list[str | None] = []
    original_update = store.update_job

    def observed_update(job_id, **kwargs):
        claim_tokens.append(kwargs.get("claim_token"))
        return original_update(job_id, **kwargs)

    def failing_adapter(*_args, **_kwargs):
        raise RuntimeError("adapter failed safely")

    monkeypatch.setattr(store, "update_job", observed_update)
    monkeypatch.setattr(
        "aniflive_tts.workstation_worker.run_adapter", failing_adapter
    )
    result = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    ).run_once()
    record = store.get_job(job["id"])
    assert result.disposition == "failed"
    assert record["status"] == "failed"
    assert record["error"] == "adapter failed safely"
    assert claim_tokens and all(token and token.startswith("claim_") for token in claim_tokens)


def test_running_cancellation_is_observed_and_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    store = WorkstationStore(tmp_path / "state")
    _project, job = _dataset_job(store, source)

    def cancelling_adapter(_job, _project, *, cancel_requested, **_kwargs):
        store.cancel_job(job["id"])
        if cancel_requested():
            raise AdapterCancelled("cooperative cancellation")
        raise AssertionError("Cancellation callback did not observe the request")

    monkeypatch.setattr(
        "aniflive_tts.workstation_worker.run_adapter", cancelling_adapter
    )
    result = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    ).run_once()
    assert result.disposition == "cancelled"
    assert store.get_job(job["id"])["status"] == "cancelled"


def test_running_pause_is_observed_and_can_be_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    store = WorkstationStore(tmp_path / "state")
    _project, job = _dataset_job(store, source)

    def pausing_adapter(_job, _project, *, cancel_requested, **_kwargs):
        store.pause_job(job["id"])
        if cancel_requested():
            raise AdapterCancelled("cooperative pause")
        raise AssertionError("Pause callback did not observe the request")

    monkeypatch.setattr("aniflive_tts.workstation_worker.run_adapter", pausing_adapter)
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )
    result = worker.run_once()
    assert result.disposition == "paused"
    assert result.store_status == "paused"
    assert store.get_job(job["id"])["pause_requested"] is False
    assert store.resume_job(job["id"])["status"] == "queued"


def test_worker_claims_higher_priority_before_fifo_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(
        kind="dataset", name="Dataset", config={"source": str(source)}
    )
    low = store.create_job(
        job_type="dataset.inventory", project_id=project["id"], priority=-10
    )
    high = store.create_job(
        job_type="dataset.inventory", project_id=project["id"], priority=90
    )
    execution_order: list[str] = []

    def successful_adapter(job, _project, **_kwargs):
        execution_order.append(str(job["id"]))
        return AdapterResult(
            "dataset.inventory",
            "io-heavy",
            "completed",
            {"ok": True, "execution_performed": True},
        )

    monkeypatch.setattr("aniflive_tts.workstation_worker.run_adapter", successful_adapter)
    summary = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    ).run(max_jobs=2)
    assert summary.completed == 2
    assert summary.paused == 0
    assert execution_order == [high["id"], low["id"]]


def test_worker_heartbeats_during_a_long_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    store = WorkstationStore(tmp_path / "state")
    _project, job = _dataset_job(store, source)
    heartbeat_count = 0
    original_heartbeat = store.heartbeat_job

    def observed_heartbeat(job_id, claim_token):
        nonlocal heartbeat_count
        heartbeat_count += 1
        return original_heartbeat(job_id, claim_token)

    def slow_adapter(_job, _project, *, cancel_requested, **_kwargs):
        deadline = time.monotonic() + 0.18
        while time.monotonic() < deadline:
            assert cancel_requested() is False
            time.sleep(0.01)
        return AdapterResult(
            "dataset.inventory",
            "io-heavy",
            "completed",
            {"execution_performed": True},
        )

    monkeypatch.setattr(store, "heartbeat_job", observed_heartbeat)
    monkeypatch.setattr("aniflive_tts.workstation_worker.run_adapter", slow_adapter)
    result = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        heartbeat_seconds=0.05,
    ).run_once()
    assert result.disposition == "completed"
    assert heartbeat_count >= 2
    assert store.get_job(job["id"])["status"] == "succeeded"


def test_heartbeat_interval_is_clamped_below_the_claim_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANIFLIVE_TTS_WORKSTATION_LEASE_SECONDS", "1")
    source = tmp_path / "source"
    source.mkdir()
    store = WorkstationStore(tmp_path / "state")
    _project, job = _dataset_job(store, source)
    heartbeat_count = 0
    original_heartbeat = store.heartbeat_job

    def observed_heartbeat(job_id, claim_token):
        nonlocal heartbeat_count
        heartbeat_count += 1
        return original_heartbeat(job_id, claim_token)

    def slow_adapter(_job, _project, *, cancel_requested, **_kwargs):
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            assert cancel_requested() is False
            time.sleep(0.01)
        return AdapterResult(
            "dataset.inventory",
            "io-heavy",
            "completed",
            {"ok": True, "execution_performed": True},
        )

    monkeypatch.setattr(store, "heartbeat_job", observed_heartbeat)
    monkeypatch.setattr("aniflive_tts.workstation_worker.run_adapter", slow_adapter)
    result = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        heartbeat_seconds=3600.0,
    ).run_once()
    assert result.disposition == "completed"
    assert heartbeat_count >= 1
    assert store.get_job(job["id"])["status"] == "succeeded"


def test_worker_rejects_mismatched_adapter_result_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    store = WorkstationStore(tmp_path / "state")
    _project, job = _dataset_job(store, source)

    def wrong_adapter(*_args, **_kwargs):
        return AdapterResult(
            "training.prepare", "gpu-exclusive", "completed", {"fake": True}
        )

    monkeypatch.setattr("aniflive_tts.workstation_worker.run_adapter", wrong_adapter)
    result = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    ).run_once()
    assert result.disposition == "failed"
    assert "claimed job contract" in store.get_job(job["id"])["error"]


def test_worker_rejects_completed_result_without_verified_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    store = WorkstationStore(tmp_path / "state")
    _project, job = _dataset_job(store, source)

    def cosmetic_completion(*_args, **_kwargs):
        return AdapterResult(
            "dataset.inventory",
            "io-heavy",
            "completed",
            {"execution_performed": False, "readiness": "ready"},
        )

    monkeypatch.setattr(
        "aniflive_tts.workstation_worker.run_adapter", cosmetic_completion
    )
    result = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    ).run_once()
    record = store.get_job(job["id"])
    assert result.disposition == "failed"
    assert record["status"] == "failed"
    assert record["result"]["execution_performed"] is False
    assert "without verified execution" in record["error"]


def test_worker_fails_a_non_json_adapter_payload_instead_of_leaving_it_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    store = WorkstationStore(tmp_path / "state")
    _project, job = _dataset_job(store, source)

    def invalid_payload(*_args, **_kwargs):
        return AdapterResult(
            "dataset.inventory",
            "io-heavy",
            "completed",
            {"execution_performed": True, "invalid_metric": float("nan")},
        )

    monkeypatch.setattr("aniflive_tts.workstation_worker.run_adapter", invalid_payload)
    result = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    ).run_once()
    record = store.get_job(job["id"])
    assert result.disposition == "failed"
    assert record["status"] == "failed"
    assert record["error"] == "Adapter result is not JSON-compatible"


def test_worker_never_completes_a_preparation_only_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"RIFF")
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(
        kind="dataset", name="Dataset", config={"source": str(source)}
    )
    job = store.create_job(job_type="dataset.process", project_id=project["id"])

    def preparation_only(*_args, **_kwargs):
        return AdapterResult(
            "dataset.process",
            "gpu-exclusive",
            "prepared-only",
            {
                "execution_performed": False,
                "readiness": "ready",
                "missing_inputs": [],
                "backend_available": True,
            },
        )

    monkeypatch.setattr("aniflive_tts.workstation_worker.run_adapter", preparation_only)
    result = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    ).run_once()
    record = store.get_job(job["id"])
    assert result.disposition == "blocked"
    assert record["status"] == "failed"
    assert record["result"]["execution_performed"] is False
    assert record["error"] == "Preparation is blocked: execution was not performed"


def test_polling_mode_stops_after_configured_job_bound(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "voice.wav").write_bytes(b"RIFF")
    store = WorkstationStore(tmp_path / "state")
    first = _dataset_job(store, source, name="First")[1]
    second = _dataset_job(store, source, name="Second")[1]
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        poll_seconds=0.05,
    )
    summary = worker.run(max_jobs=2)
    assert summary.processed == 2
    assert summary.completed == 2
    assert store.get_job(first["id"])["status"] == "succeeded"
    assert store.get_job(second["id"])["status"] == "succeeded"


def test_shutdown_before_claim_leaves_queued_job_untouched(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    store = WorkstationStore(tmp_path / "state")
    _project, job = _dataset_job(store, source)
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )
    worker.request_stop()
    assert worker.run_once().disposition == "idle"
    assert store.get_job(job["id"])["status"] == "queued"


def test_shutdown_during_adapter_cooperatively_cancels_claimed_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    store = WorkstationStore(tmp_path / "state")
    _project, job = _dataset_job(store, source)
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )

    def interrupted_adapter(_job, _project, *, cancel_requested, **_kwargs):
        worker.request_stop()
        if cancel_requested():
            raise AdapterCancelled("worker shutdown requested")
        raise AssertionError("Worker shutdown was not observed")

    monkeypatch.setattr(
        "aniflive_tts.workstation_worker.run_adapter", interrupted_adapter
    )
    result = worker.run_once()
    assert result.disposition == "cancelled"
    assert store.get_job(job["id"])["status"] == "cancelled"


def test_signal_handler_requests_stop_and_is_restored(tmp_path: Path) -> None:
    worker = WorkstationWorker(
        WorkstationStore(tmp_path / "state"),
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )
    previous = signal.getsignal(signal.SIGINT)
    with worker.signal_handlers():
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        assert worker.stopping is True
    assert signal.getsignal(signal.SIGINT) == previous


def test_worker_cli_is_fixed_and_has_no_arbitrary_command_option(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "worker",
            "--once",
            "--workstation-dir",
            str(tmp_path / "state"),
            "--import-root",
            str(tmp_path),
        ]
    )
    assert args.command == "worker"
    assert not hasattr(args, "command_argv")
    assert main(
        [
            "worker",
            "--once",
            "--workstation-dir",
            str(tmp_path / "state"),
            "--import-root",
            str(tmp_path),
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["kind"] == "worker-summary"
    assert summary["idle"] is True
    with pytest.raises(SystemExit):
        parser.parse_args(["worker", "--command", "anything"])


def test_worker_configuration_is_bounded(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "state")
    with pytest.raises(WorkstationError, match="Unsupported worker adapter"):
        WorkstationWorker(
            store,
            manifest_root=tmp_path / "manifests",
            allowed_path_roots=(tmp_path,),
            enabled_job_types=("not.registered",),
        )
    with pytest.raises(WorkstationError, match="poll_seconds"):
        WorkstationWorker(
            store,
            manifest_root=tmp_path / "manifests",
            allowed_path_roots=(tmp_path,),
            poll_seconds=0.0,
        )


def test_configured_linux_broker_is_called_through_the_worker(tmp_path: Path) -> None:
    config = _training_config(tmp_path)
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(
        kind="training", name="Training", config=config
    )
    job = store.create_job(
        job_type="training.prepare", project_id=project["id"]
    )

    class FakeExecution:
        def as_dict(self):
            return {
                "backend": "linux-docker",
                "image_digest": "sha256:" + "a" * 64,
                "payload": {"qualified": True},
                "artifacts": [],
            }

    class FakeBroker:
        def __init__(self) -> None:
            self.reconciled: tuple[str, ...] | None = None
            self.reconciliation_calls = 0
            self.executions: list[dict] = []
            self.dataset_scope: dict | None = None

        def describe(self, job_type: str):
            return {
                "available": job_type == "training.prepare",
                "kind": "linux-docker",
                "image_digest": "sha256:" + "a" * 64,
                "network": "none",
            }

        def reconcile(self, active_job_ids):
            self.reconciled = tuple(active_job_ids)
            self.reconciliation_calls += 1
            return SimpleNamespace(retained=(), removed=())

        def execute(self, job_type: str, **kwargs):
            self.executions.append({"job_type": job_type, **kwargs})
            dataset = Path(kwargs["input_paths"]["dataset"])
            self.dataset_scope = json.loads(
                (dataset / "training-input.json").read_text(encoding="utf-8")
            )
            assert (dataset / "train").is_dir()
            assert not (dataset / "validation").exists()
            assert not (dataset / "test").exists()
            return FakeExecution()

    broker = FakeBroker()
    reports: list[dict] = []
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("training.prepare",),
        worker_broker=broker,
        reporter=lambda record: reports.append(dict(record)),
    )
    result = worker.run_once()
    assert result.disposition == "completed"
    assert broker.reconciled == ()
    assert len(broker.executions) == 1
    execution = broker.executions[0]
    assert execution["job_type"] == "training.prepare"
    assert execution["job_id"] == job["id"]
    assert execution["project_id"] == project["id"]
    assert set(execution["input_paths"]) == set(config)
    assert execution["input_paths"]["dataset"] != str(Path(config["dataset"]).resolve())
    assert execution["input_paths"] | {"dataset": str(Path(config["dataset"]).resolve())} == {
        key: str(Path(value).resolve()) for key, value in config.items()
    }
    assert broker.dataset_scope is not None
    assert broker.dataset_scope["worker_scope"]["visible_splits"] == ["train"]
    record = store.get_job(job["id"])
    assert record["status"] == "succeeded"
    assert record["result"]["execution_performed"] is True
    assert reports[0]["kind"] == "docker-reconciliation"
    worker.reconcile_broker()
    assert broker.reconciliation_calls == 1
    worker.reconcile_broker(force=True)
    assert broker.reconciliation_calls == 2


def test_gpu_worker_waits_for_inference_lease_then_runs_after_release(
    tmp_path: Path,
) -> None:
    config = _training_config(tmp_path)
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="training", name="Training", config=config)
    job = store.create_job(job_type="training.prepare", project_id=project["id"])
    inference_lease = store.acquire_resource_lease(
        GPU_RESOURCE_KEY,
        purpose="inference:resident",
        owner_id="inference-test",
    )

    class FakeExecution:
        def as_dict(self):
            return {
                "backend": "linux-docker",
                "image_digest": "sha256:" + "a" * 64,
                "payload": {"qualified": True},
                "artifacts": [],
            }

    class FakeBroker:
        def __init__(self) -> None:
            self.executions = 0

        def describe(self, job_type: str):
            return {
                "available": job_type == "training.prepare",
                "kind": "linux-docker",
                "image_digest": "sha256:" + "a" * 64,
                "network": "none",
            }

        def reconcile(self, _active_job_ids):
            return SimpleNamespace(retained=(), removed=())

        def execute(self, _job_type: str, **_kwargs):
            self.executions += 1
            return FakeExecution()

    broker = FakeBroker()
    reports: list[dict] = []
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("training.prepare",),
        worker_broker=broker,
        reporter=lambda record: reports.append(dict(record)),
    )

    assert worker.run_once().disposition == "idle"
    assert worker.run_once().disposition == "idle"
    waiting = store.get_job(job["id"])
    assert waiting["status"] == "queued"
    assert waiting["wait_reason"] == (
        "Waiting for gpu:0; TensorRT inference or another GPU job is resident"
    )
    assert broker.executions == 0
    assert sum(report.get("kind") == "job-wait" for report in reports) == 1

    store.release_resource_lease(inference_lease)
    result = worker.run_once()

    assert result.disposition == "completed"
    assert broker.executions == 1
    completed = store.get_job(job["id"])
    assert completed["status"] == "succeeded"
    assert completed["wait_reason"] is None


def test_periodic_broker_reconciliation_removes_orphan_after_lease_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="training", name="Training")
    job = store.create_job(
        job_type="training.prepare", project_id=project["id"]
    )
    store.claim_job(job["id"])
    clock = [100.0]
    monkeypatch.setattr(
        "aniflive_tts.workstation_worker.time.monotonic", lambda: clock[0]
    )

    class FakeBroker:
        def __init__(self) -> None:
            self.containers = {f"container-{job['id']}": job["id"]}
            self.calls: list[tuple[str, ...]] = []

        def reconcile(self, active_job_ids):
            active = tuple(active_job_ids)
            self.calls.append(active)
            retained = tuple(
                container_id
                for container_id, job_id in self.containers.items()
                if job_id in active
            )
            removed = tuple(
                container_id
                for container_id, job_id in self.containers.items()
                if job_id not in active
            )
            for container_id in removed:
                self.containers.pop(container_id)
            return SimpleNamespace(retained=retained, removed=removed)

    broker = FakeBroker()
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("training.prepare",),
        worker_broker=broker,
    )

    assert worker.run_once().disposition == "idle"
    assert broker.calls == [(job["id"],)]
    assert broker.containers

    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00.000000Z", job["id"]),
        )
    clock[0] += 31.0

    assert worker.run_once().disposition == "idle"
    assert broker.calls == [(job["id"],), ()]
    assert broker.containers == {}
    assert store.get_job(job["id"])["status"] == "failed"


def test_uncertain_docker_cleanup_quarantines_gpu_until_reconciliation_confirms_absence(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(
        kind="training", name="Training", config=_training_config(tmp_path)
    )
    job = store.create_job(
        job_type="training.prepare", project_id=project["id"]
    )

    class UncertainBroker:
        def __init__(self) -> None:
            self.execute_attempted = False
            self.cleanup_confirmed = False
            self.reconcile_observations: list[tuple[tuple[str, ...], str]] = []

        def describe(self, job_type: str):
            return {
                "available": job_type == "training.prepare",
                "kind": "linux-docker",
                "image_digest": "sha256:" + "a" * 64,
                "network": "none",
            }

        def execute(self, _job_type: str, **_kwargs):
            self.execute_attempted = True
            raise DockerCleanupUncertain("synthetic container still present")

        def reconcile(self, active_job_ids):
            active = tuple(active_job_ids)
            self.reconcile_observations.append(
                (active, store.get_job(job["id"])["status"])
            )
            if self.execute_attempted and not self.cleanup_confirmed:
                raise DockerCleanupUncertain("synthetic reconciliation failure")
            return SimpleNamespace(retained=(), removed=("container",))

    broker = UncertainBroker()
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("training.prepare",),
        worker_broker=broker,
    )
    result = worker.run_once()
    assert result.disposition == "failed"
    assert store.get_job(job["id"])["status"] == "running"
    with pytest.raises(WorkstationError, match="busy"):
        store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="inference:test")

    with pytest.raises(DockerCleanupUncertain, match="reconciliation failure"):
        worker.reconcile_broker(force=True)
    assert broker.reconcile_observations[-1] == ((), "running")
    assert store.get_job(job["id"])["status"] == "running"
    with pytest.raises(WorkstationError, match="busy"):
        store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="inference:test")

    broker.cleanup_confirmed = True
    worker.reconcile_broker(force=True)
    assert broker.reconcile_observations[-1] == ((), "running")
    assert store.get_job(job["id"])["status"] == "failed"
    token = store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="inference:test")
    store.release_resource_lease(token)


def test_worker_imports_docker_outputs_into_immutable_artifact_lineage(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(
        kind="training", name="Training", config=_training_config(tmp_path)
    )
    parent = store.register_artifact(artifact_type="dataset", name="Input dataset")
    job = store.create_job(
        job_type="training.prepare",
        project_id=project["id"],
        parameters={"parent_artifact_ids": [parent["id"]]},
    )
    output = tmp_path / "docker-output"
    artifact_path = output / "artifacts" / "checkpoint.bin"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"trusted checkpoint")
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    class FakeExecution:
        output_root = output
        image_digest = "sha256:" + "a" * 64
        artifacts = (
            {
                "kind": "checkpoint",
                "relative_path": "artifacts/checkpoint.bin",
                "sha256": digest,
                "size_bytes": artifact_path.stat().st_size,
            },
        )

        def as_dict(self):
            return {
                "backend": "linux-docker",
                "image_digest": self.image_digest,
                "payload": {"loss": 0.25},
                "artifacts": [dict(self.artifacts[0])],
            }

    class FakeBroker:
        def describe(self, job_type: str):
            return {
                "available": job_type == "training.prepare",
                "kind": "linux-docker",
                "image_digest": "sha256:" + "a" * 64,
                "network": "none",
            }

        def reconcile(self, _active_job_ids):
            return SimpleNamespace(retained=(), removed=())

        def execute(self, _job_type: str, **_kwargs):
            return FakeExecution()

    result = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("training.prepare",),
        worker_broker=FakeBroker(),
    ).run_once()

    assert result.disposition == "completed"
    job_record = store.get_job(job["id"])
    artifact_ids = job_record["result"]["registered_artifact_ids"]
    assert len(artifact_ids) == 1
    artifact = store.get_artifact(artifact_ids[0])
    stored_path = store.artifact_root.joinpath(*Path(artifact["local_path"]).parts)
    assert artifact["type"] == "checkpoint"
    assert artifact["status"] == "ready"
    assert artifact["sha256"] == digest
    assert artifact["parent_artifact_ids"] == [parent["id"]]
    assert stored_path.read_bytes() == b"trusted checkpoint"
    artifact_path.write_bytes(b"changed after registration")
    assert stored_path.read_bytes() == b"trusted checkpoint"


def test_worker_derives_artifact_lineage_from_completed_dependencies(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(
        kind="training", name="Training", config=_training_config(tmp_path)
    )
    first_job = store.create_job(
        job_type="training.prepare",
        project_id=project["id"],
    )
    second_job = store.create_job(
        job_type="training.prepare",
        project_id=project["id"],
        depends_on=[first_job["id"]],
    )
    output = tmp_path / "docker-output"
    artifact_path = output / "checkpoint.bin"
    output.mkdir()
    artifact_path.write_bytes(b"dependency checkpoint")
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    class FakeExecution:
        output_root = output
        image_digest = "sha256:" + "a" * 64
        artifacts = (
            {
                "kind": "checkpoint",
                "relative_path": "checkpoint.bin",
                "sha256": digest,
                "size_bytes": artifact_path.stat().st_size,
            },
        )

        def as_dict(self):
            return {
                "backend": "linux-docker",
                "image_digest": self.image_digest,
                "payload": {},
                "artifacts": [dict(self.artifacts[0])],
            }

    class FakeBroker:
        def describe(self, job_type: str):
            return {
                "available": job_type == "training.prepare",
                "kind": "linux-docker",
                "image_digest": "sha256:" + "a" * 64,
                "network": "none",
            }

        def reconcile(self, _active_job_ids):
            return SimpleNamespace(retained=(), removed=())

        def execute(self, _job_type: str, **_kwargs):
            return FakeExecution()

    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("training.prepare",),
        worker_broker=FakeBroker(),
    )
    assert worker.run_once().disposition == "completed"
    first_artifact_id = store.get_job(first_job["id"])["result"][
        "registered_artifact_ids"
    ][0]

    assert worker.run_once().disposition == "completed"
    second_artifact_id = store.get_job(second_job["id"])["result"][
        "registered_artifact_ids"
    ][0]
    second_artifact = store.get_artifact(second_artifact_id)
    assert second_artifact["parent_artifact_ids"] == [first_artifact_id]


def test_worker_compacts_large_dependency_lineage_to_verified_anchors(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="training", name="Training")
    dependency = store.create_job(
        job_type="checkpoint.select", project_id=project["id"]
    )
    artifact_ids = []
    anchor_ids = {}
    paths = [
        "checkpoint-selection-report.json",
        "selected/deployment-checkpoints.json",
        *(f"audio/joint-sweep/case-{index}.wav" for index in range(63)),
    ]
    for path in paths:
        artifact = store.register_artifact(
            artifact_type="checkpoint",
            name=path.rsplit("/", 1)[-1],
            project_id=project["id"],
            metadata={
                "job_id": dependency["id"],
                "worker_relative_path": path,
            },
        )
        artifact_ids.append(artifact["id"])
        if path in {
            "checkpoint-selection-report.json",
            "selected/deployment-checkpoints.json",
        }:
            anchor_ids[path] = artifact["id"]
    claim = store.claim_job(dependency["id"])
    store.update_job(
        dependency["id"],
        status="succeeded",
        claim_token=claim.token,
    )
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE jobs SET result_json = ? WHERE id = ?",
            (
                json.dumps({"registered_artifact_ids": artifact_ids}),
                dependency["id"],
            ),
        )
    job = store.create_job(
        job_type="reference.select",
        project_id=project["id"],
        depends_on=(dependency["id"],),
    )
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )

    assert worker._artifact_parent_ids(job, {}) == [
        anchor_ids["checkpoint-selection-report.json"],
        anchor_ids["selected/deployment-checkpoints.json"],
    ]


def _completed_conversion_package_dependency(
    store: WorkstationStore, project_id: str, tmp_path: Path
) -> tuple[dict, list[dict]]:
    conversion = store.create_job(job_type="conversion.parity", project_id=project_id)
    claim = store.claim_job(conversion["id"])
    output = tmp_path / "conversion-output"
    manifest = output / "model-package" / "manifest.json"
    onnx = output / "model-package" / "onnx" / "gpt_step.onnx"
    manifest.parent.mkdir(parents=True)
    onnx.parent.mkdir(parents=True)
    manifest.write_text('{"format":"aniflive-tts-model-package"}\n', encoding="utf-8")
    onnx.write_bytes(b"verified-engine-input")
    published = []
    for path in (manifest, onnx):
        published.append(
            {
                "kind": "evaluation",
                "relative_path": path.relative_to(output).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    artifacts = store.register_worker_artifacts(
        job_id=conversion["id"],
        claim_token=claim.token,
        job_type="conversion.parity",
        project_id=project_id,
        source_root=output,
        artifacts=published,
        image_digest="sha256:" + "a" * 64,
    )
    store.update_job(
        conversion["id"],
        status="succeeded",
        progress=1.0,
        result={"registered_artifact_ids": [artifact["id"] for artifact in artifacts]},
        claim_token=claim.token,
    )
    return conversion, artifacts


def _completed_model_package_dependency(
    store: WorkstationStore, project_id: str, tmp_path: Path
) -> tuple[dict, list[dict]]:
    package = store.create_job(job_type="model.package", project_id=project_id)
    claim = store.claim_job(package["id"])
    output = tmp_path / "package-output"
    manifest = output / "model-package" / "manifest.json"
    engine = output / "model-package" / "engines" / "gpt_step.engine"
    manifest.parent.mkdir(parents=True)
    engine.parent.mkdir(parents=True)
    manifest.write_text('{"format":"aniflive-tts-model-package"}\n', encoding="utf-8")
    engine.write_bytes(b"verified-package-input")
    published = []
    for path in (manifest, engine):
        published.append(
            {
                "kind": "package",
                "relative_path": path.relative_to(output).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    artifacts = store.register_worker_artifacts(
        job_id=package["id"],
        claim_token=claim.token,
        job_type="model.package",
        project_id=project_id,
        source_root=output,
        artifacts=published,
        image_digest="sha256:" + "b" * 64,
    )
    store.update_job(
        package["id"],
        status="succeeded",
        progress=1.0,
        result={"registered_artifact_ids": [artifact["id"] for artifact in artifacts]},
        claim_token=claim.token,
    )
    return package, artifacts


def _completed_training_checkpoint_dependency(
    store: WorkstationStore,
    project_id: str,
    tmp_path: Path,
    *,
    manifest_gpt_sha256: str | None = None,
    holdout_evidence_override: dict | None = None,
) -> tuple[dict, dict, list[dict]]:
    selection_job = store.create_job(job_type="checkpoint.select", project_id=project_id)
    claim = store.claim_job(selection_job["id"])
    output = tmp_path / "selection-output"
    selected = output / "selected"
    gpt = selected / "checkpoints" / "gpt" / "voice-e10.ckpt"
    sovits = selected / "checkpoints" / "sovits" / "voice_e8_s80.pth"
    gpt.parent.mkdir(parents=True)
    sovits.parent.mkdir(parents=True)
    gpt.write_bytes(b"gpt-checkpoint")
    sovits.write_bytes(b"sovits-checkpoint")
    deployment = selected / "deployment-checkpoints.json"
    deployment.write_text(
        json.dumps(
            {
                "schema": "aniflive-tts-v2proplus-deployment-checkpoints-v2",
                "model_family": "gsv-v2proplus",
                "selection": {
                    "method": "validation-checkpoint-selection-v1",
                    "report_sha256": "1" * 64,
                    "validation_manifest_sha256": "2" * 64,
                    "winner_reason": "passed hard gates and ranked first",
                    "test_split_accessed": False,
                },
                "gpt": {
                    "relative_path": gpt.relative_to(selected).as_posix(),
                    "sha256": manifest_gpt_sha256
                    or hashlib.sha256(gpt.read_bytes()).hexdigest(),
                    "size_bytes": gpt.stat().st_size,
                },
                "sovits": {
                    "relative_path": sovits.relative_to(selected).as_posix(),
                    "sha256": hashlib.sha256(sovits.read_bytes()).hexdigest(),
                    "size_bytes": sovits.stat().st_size,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    published = [
        {
            "kind": "checkpoint",
            "relative_path": path.relative_to(output).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in (deployment, gpt, sovits)
    ]
    artifacts = store.register_worker_artifacts(
        job_id=selection_job["id"],
        claim_token=claim.token,
        job_type="checkpoint.select",
        project_id=project_id,
        source_root=output,
        artifacts=published,
        image_digest="sha256:" + "c" * 64,
    )
    store.update_job(
        selection_job["id"],
        status="succeeded",
        progress=1.0,
        result={"registered_artifact_ids": [artifact["id"] for artifact in artifacts]},
        claim_token=claim.token,
    )
    holdout_job = store.create_job(job_type="holdout.evaluate", project_id=project_id)
    holdout_claim = store.claim_job(holdout_job["id"])
    store.update_job(
        holdout_job["id"],
        status="succeeded",
        progress=1.0,
        result={
            "execution_performed": True,
            "backend": {
                "payload": {
                    "schema": "aniflive-tts-holdout-evaluation-v1",
                    "status": "passed",
                    "winner_locked_before_test": True,
                    "test_consumed_once": True,
                    **(holdout_evidence_override or {}),
                },
            },
        },
        claim_token=holdout_claim.token,
    )
    return selection_job, holdout_job, artifacts


def _completed_dataset_dependency(
    store: WorkstationStore,
    project_id: str,
    tmp_path: Path,
    *,
    job_type: str = "dataset.target-speaker",
) -> tuple[dict, list[dict]]:
    job = store.create_job(job_type=job_type, project_id=project_id)
    claim = store.claim_job(job["id"])
    output = tmp_path / f"{job_type.replace('.', '-')}-output"
    report = output / "target-speaker-routing" / "routing-report.json"
    clip = output / "target-speaker-routing" / "clips" / "clean" / "seg_000000_abcdef" / "original.wav"
    report.parent.mkdir(parents=True)
    clip.parent.mkdir(parents=True)
    report.write_text(
        json.dumps({"schema": "aniflive-target-speaker-routing-v1", "records": []}) + "\n",
        encoding="utf-8",
    )
    clip.write_bytes(b"RIFF-verified-dataset-artifact")
    published = [
        {
            "kind": "dataset",
            "relative_path": path.relative_to(output).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in (report, clip)
    ]
    artifacts = store.register_worker_artifacts(
        job_id=job["id"],
        claim_token=claim.token,
        job_type=job_type,
        project_id=project_id,
        source_root=output,
        artifacts=published,
        image_digest="sha256:" + "d" * 64,
    )
    store.update_job(
        job["id"],
        status="succeeded",
        progress=1.0,
        result={"registered_artifact_ids": [artifact["id"] for artifact in artifacts]},
        claim_token=claim.token,
    )
    return job, artifacts


def test_worker_materializes_verified_training_checkpoint_pair_for_engine(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "state")
    training_project = store.create_project(kind="training", name="Voice training")
    evaluation_project = store.create_project(kind="evaluation", name="Qualification")
    selection_job, holdout_job, _artifacts = _completed_training_checkpoint_dependency(
        store, training_project["id"], tmp_path
    )
    engine = store.create_job(
        job_type="engine.prepare",
        project_id=evaluation_project["id"],
        depends_on=[selection_job["id"], holdout_job["id"]],
    )
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("engine.prepare",),
    )

    with worker._job_with_dependency_inputs(engine) as runnable:
        checkpoint = Path(runnable["parameters"]["checkpoint"])
        assert (checkpoint / "deployment-checkpoints.json").is_file()
        assert (checkpoint / "checkpoints/gpt/voice-e10.ckpt").read_bytes() == (
            b"gpt-checkpoint"
        )
        assert (checkpoint / "checkpoints/sovits/voice_e8_s80.pth").read_bytes() == (
            b"sovits-checkpoint"
        )
        materialized_root = checkpoint.parent.parent

    assert not materialized_root.exists()


def test_worker_materializes_verified_dataset_tree_for_next_stage(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="dataset", name="Voice acquisition")
    routing, _artifacts = _completed_dataset_dependency(
        store, project["id"], tmp_path
    )
    separation = store.create_job(
        job_type="dataset.separate",
        project_id=project["id"],
        depends_on=[routing["id"]],
    )
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("dataset.separate",),
    )

    with worker._job_with_dependency_inputs(separation) as runnable:
        dataset = Path(runnable["parameters"]["dataset"])
        assert (dataset / "target-speaker-routing/routing-report.json").is_file()
        clip = dataset / "target-speaker-routing/clips/clean/seg_000000_abcdef/original.wav"
        assert clip.read_bytes() == b"RIFF-verified-dataset-artifact"
        materialized_root = dataset.parent

    assert not materialized_root.exists()


def test_worker_rejects_cross_project_dataset_dependency(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "state")
    source_project = store.create_project(kind="dataset", name="Source")
    other_project = store.create_project(kind="dataset", name="Other")
    routing, _artifacts = _completed_dataset_dependency(
        store, source_project["id"], tmp_path
    )
    separation = store.create_job(
        job_type="dataset.separate",
        project_id=other_project["id"],
        depends_on=[routing["id"]],
    )
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("dataset.separate",),
    )

    with (
        pytest.raises(WorkstationError, match="must use one project"),
        worker._job_with_dependency_inputs(separation),
    ):
        pass


def test_worker_rejects_tampered_dataset_dependency(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="dataset", name="Voice acquisition")
    routing, artifacts = _completed_dataset_dependency(
        store, project["id"], tmp_path
    )
    separation = store.create_job(
        job_type="dataset.separate",
        project_id=project["id"],
        depends_on=[routing["id"]],
    )
    stored = store.artifact_root.joinpath(*Path(artifacts[0]["local_path"]).parts)
    stored.write_bytes(b"tampered")
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("dataset.separate",),
    )

    with (
        pytest.raises(WorkstationError, match="checksum no longer matches"),
        worker._job_with_dependency_inputs(separation),
    ):
        pass


def test_worker_rejects_training_deployment_metadata_mismatch(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "state")
    training_project = store.create_project(kind="training", name="Voice training")
    evaluation_project = store.create_project(kind="evaluation", name="Qualification")
    selection_job, holdout_job, _artifacts = _completed_training_checkpoint_dependency(
        store,
        training_project["id"],
        tmp_path,
        manifest_gpt_sha256="0" * 64,
    )
    engine = store.create_job(
        job_type="engine.prepare",
        project_id=evaluation_project["id"],
        depends_on=[selection_job["id"], holdout_job["id"]],
    )
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("engine.prepare",),
    )

    with (
        pytest.raises(WorkstationError, match="metadata does not match"),
        worker._job_with_dependency_inputs(engine),
    ):
        pass


def test_worker_materializes_verified_conversion_package_for_dependent_packaging(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "state")
    original = tmp_path / "original-package"
    original.mkdir()
    project = store.create_project(
        kind="evaluation",
        name="Release workflow",
        config={"model_package": str(original)},
    )
    conversion, _artifacts = _completed_conversion_package_dependency(
        store, project["id"], tmp_path
    )
    package = store.create_job(
        job_type="model.package",
        project_id=project["id"],
        depends_on=[conversion["id"]],
    )
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("model.package",),
    )

    with worker._job_with_dependency_inputs(package) as runnable:
        model_package = Path(runnable["parameters"]["model_package"])
        assert model_package != original
        assert (model_package / "manifest.json").is_file()
        assert (model_package / "onnx" / "gpt_step.onnx").read_bytes() == (
            b"verified-engine-input"
        )
        materialized_root = model_package.parent.parent

    assert not materialized_root.exists()


def test_worker_materializes_verified_model_package_for_evaluation(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="evaluation", name="Qualification")
    package, _artifacts = _completed_model_package_dependency(
        store, project["id"], tmp_path
    )
    evaluation = store.create_job(
        job_type="evaluation.prepare",
        project_id=project["id"],
        depends_on=[package["id"]],
    )
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("evaluation.prepare",),
    )

    with worker._job_with_dependency_inputs(evaluation) as runnable:
        model_package = Path(runnable["parameters"]["model_package"])
        assert (model_package / "manifest.json").is_file()
        assert (model_package / "engines" / "gpt_step.engine").read_bytes() == (
            b"verified-package-input"
        )
        materialized_root = model_package.parent.parent

    assert not materialized_root.exists()


def test_holdout_can_only_be_consumed_once_per_training_run(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(
        kind="training",
        name="Voice training",
    )
    first = store.create_job(
        job_type="holdout.evaluate",
        project_id=project["id"],
    )
    store.update_project(
        project["id"],
        config={"holdout_consumption_job_id": first["id"]},
    )
    second = store.create_job(
        job_type="holdout.evaluate",
        project_id=project["id"],
    )
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("holdout.evaluate",),
    )

    with (
        pytest.raises(WorkstationError, match="already consumed"),
        worker._job_with_human_reference(second),
    ):
        pass


def test_holdout_marks_test_split_accessed_before_materializing_inputs(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="training", name="Voice training")
    holdout = store.create_job(
        job_type="holdout.evaluate",
        project_id=project["id"],
    )
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("holdout.evaluate",),
    )

    with (
        pytest.raises(WorkstationError, match="human-locked deployment reference"),
        worker._job_with_human_reference(holdout),
    ):
        pass

    config = store.get_project(project["id"])["config"]
    assert config["holdout_consumption_job_id"] == holdout["id"]
    assert config["test_split_accessed"] is True


def test_automatic_production_chain_preserves_gate_order(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(
        kind="training",
        name="Voice training",
        config={"auto_build_production": True},
    )
    training = store.create_job(job_type="training.prepare", project_id=project["id"])
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )

    worker._queue_automatic_production_build(training)
    checkpoint = next(job for job in store.list_jobs() if job["type"] == "checkpoint.select")
    assert checkpoint["depends_on"] == [training["id"]]

    worker._queue_reference_selection(checkpoint)
    reference = next(job for job in store.list_jobs() if job["type"] == "reference.select")
    assert reference["depends_on"] == [checkpoint["id"]]

    worker._mark_reference_pending_human_evidence(
        reference,
        {
            "backend": {
                "payload": {
                    "status": "blocked-pending-human-evidence",
                    "candidate_count": 5,
                }
            },
        },
    )
    reference_config = store.get_project(project["id"])["config"]
    assert reference_config["reference_status"] == "pending-human-evidence"
    assert reference_config["reference_selection_job_id"] == reference["id"]
    assert reference_config["reference_candidate_count"] == 5

    holdout = store.create_job(
        job_type="holdout.evaluate",
        project_id=project["id"],
        depends_on=[checkpoint["id"], reference["id"]],
    )
    worker._queue_engine_build(holdout)
    engine = next(job for job in store.list_jobs() if job["type"] == "engine.prepare")
    assert engine["depends_on"] == [checkpoint["id"], holdout["id"]]

    worker._queue_conversion_parity(engine)
    parity = next(job for job in store.list_jobs() if job["type"] == "conversion.parity")
    assert parity["depends_on"] == [engine["id"], checkpoint["id"], holdout["id"]]

    worker._queue_model_package(parity)
    package = next(job for job in store.list_jobs() if job["type"] == "model.package")
    assert package["depends_on"] == [parity["id"]]


def test_automatic_evaluation_is_blocked_when_required_assets_are_missing(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="training", name="Voice training")
    package = store.create_job(job_type="model.package", project_id=project["id"])
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )

    worker._queue_canonical_evaluation(package)

    evaluations = store.list_projects(kind="evaluation")
    assert len(evaluations) == 1
    assert evaluations[0]["config"]["qualification_status"] == "blocked"
    assert "shared_dir" in evaluations[0]["config"]["blocked_reason"]
    assert "asr_model" in evaluations[0]["config"]["blocked_reason"]
    assert "baseline_report" not in evaluations[0]["config"]["blocked_reason"]
    assert not any(job["type"] == "evaluation.prepare" for job in store.list_jobs())


def test_worker_rejects_tampered_dependency_artifact(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "state")
    original = tmp_path / "original-package"
    original.mkdir()
    project = store.create_project(
        kind="evaluation",
        name="Release workflow",
        config={"model_package": str(original)},
    )
    conversion, artifacts = _completed_conversion_package_dependency(
        store, project["id"], tmp_path
    )
    package = store.create_job(
        job_type="model.package",
        project_id=project["id"],
        depends_on=[conversion["id"]],
    )
    stored = store.artifact_root.joinpath(*Path(artifacts[0]["local_path"]).parts)
    stored.write_bytes(b"tampered")
    worker = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("model.package",),
    )

    with (
        pytest.raises(WorkstationError, match="checksum no longer matches"),
        worker._job_with_dependency_inputs(package),
    ):
        pass


def test_worker_finalization_failure_removes_staged_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(
        kind="training", name="Training", config=_training_config(tmp_path)
    )
    job = store.create_job(job_type="training.prepare", project_id=project["id"])
    output = tmp_path / "docker-output"
    artifact_path = output / "checkpoint.bin"
    output.mkdir()
    artifact_path.write_bytes(b"trusted checkpoint")
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    class FakeExecution:
        output_root = output
        image_digest = "sha256:" + "a" * 64
        artifacts = (
            {
                "kind": "checkpoint",
                "relative_path": artifact_path.name,
                "sha256": digest,
                "size_bytes": artifact_path.stat().st_size,
            },
        )

        def as_dict(self):
            return {
                "backend": "linux-docker",
                "image_digest": self.image_digest,
                "payload": {"loss": 0.25},
                "artifacts": [dict(self.artifacts[0])],
            }

    class FakeBroker:
        def describe(self, job_type: str):
            return {
                "available": job_type == "training.prepare",
                "kind": "linux-docker",
                "image_digest": "sha256:" + "a" * 64,
                "network": "none",
            }

        def reconcile(self, _active_job_ids):
            return SimpleNamespace(retained=(), removed=())

        def execute(self, _job_type: str, **_kwargs):
            return FakeExecution()

    original_update = store.update_job

    def fail_success_transition(job_id: str, **kwargs):
        if kwargs.get("status") == "succeeded":
            raise WorkstationError("synthetic finalization failure")
        return original_update(job_id, **kwargs)

    monkeypatch.setattr(store, "update_job", fail_success_transition)
    result = WorkstationWorker(
        store,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        enabled_job_types=("training.prepare",),
        worker_broker=FakeBroker(),
    ).run_once()

    assert result.disposition == "failed"
    assert store.get_job(job["id"])["status"] == "failed"
    assert store.list_artifacts(artifact_type="checkpoint") == []
    worker_root = store.artifact_root / "worker" / job["id"]
    assert not worker_root.exists() or not any(
        path.is_file() for path in worker_root.rglob("*")
    )


def test_worker_cli_accepts_only_an_admin_config_path_not_image_or_argv(
    tmp_path: Path,
) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "worker",
            "--workstation-dir",
            str(tmp_path / "state"),
            "--import-root",
            str(tmp_path),
            "--docker-broker-config",
            str(tmp_path / "broker.json"),
        ]
    )
    assert args.docker_broker_config == tmp_path / "broker.json"
    with pytest.raises(SystemExit):
        parser.parse_args(["worker", "--docker-image", "untrusted:latest"])
    with pytest.raises(SystemExit):
        parser.parse_args(["worker", "--docker-argv", "python -c malicious"])


@pytest.mark.parametrize(
    "override",
    [
        {"status": "failed"},
        {"winner_locked_before_test": False},
        {"test_consumed_once": False},
        {"schema": "untrusted-report"},
    ],
)
def test_engine_rejects_invalid_nested_holdout_evidence(tmp_path: Path, override: dict) -> None:
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="training", name="Test")
    checkpoint, holdout, _ = _completed_training_checkpoint_dependency(
        store, project["id"], tmp_path, holdout_evidence_override=override
    )
    engine = store.create_job(
        job_type="engine.prepare", project_id=project["id"],
        depends_on=[checkpoint["id"], holdout["id"]],
    )
    worker = WorkstationWorker(store, allowed_path_roots=(tmp_path,))
    with pytest.raises(WorkstationError, match="blocked by holdout"):
        with worker._job_with_dependency_inputs(engine):
            pytest.fail("Invalid holdout evidence must not reach engine materialization")


def test_first_model_evaluation_queues_without_private_historical_baseline(tmp_path):
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="training", name="First voice", config={
        "shared_dir": str(tmp_path / "shared"), "asr_model": str(tmp_path / "asr"),
    })
    package = store.create_job(job_type="model.package", project_id=project["id"])
    worker = WorkstationWorker(
        store, manifest_root=tmp_path / "manifests", allowed_path_roots=(tmp_path,),
    )
    worker._queue_canonical_evaluation(package)
    worker._queue_canonical_evaluation(package)
    evaluations = store.list_projects(kind="evaluation")
    jobs = [j for j in store.list_jobs() if j["type"] == "evaluation.prepare"]
    assert len(evaluations) == len(jobs) == 1
    assert jobs[0]["depends_on"] == [package["id"]]
    assert evaluations[0]["config"]["comparison_status"] == "unavailable"
    assert evaluations[0]["config"]["qualification_status"] == "pending"
