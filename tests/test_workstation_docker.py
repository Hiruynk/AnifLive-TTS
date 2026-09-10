from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from aniflive_tts.workstation import WorkstationError, WorkstationStore
from aniflive_tts.workstation_adapters import AdapterCancelled, AdapterEvent, run_adapter
from aniflive_tts.workstation_docker import (
    BROKER_CONFIG_SCHEMA,
    RESULT_MANIFEST_SCHEMA,
    CommandResult,
    DockerBrokerSettings,
    DockerCleanupUncertain,
    DockerExecutionResult,
    DockerWorkerBroker,
    SubprocessDockerRunner,
    load_broker_settings,
    parse_result_manifest,
)


IMAGE = "ghcr.io/hiruynk/aniflive-tts-worker@sha256:" + "a" * 64
IMAGE_DIGEST = "sha256:" + "a" * 64
JOB_ID = "job_00000000-0000-0000-0000-000000000001"
PROJECT_ID = "training_00000000-0000-0000-0000-000000000002"
RUN_ID = "3" * 32
CONTAINER_ID = "4" * 64
MANIFEST_SHA = "b" * 64


def _mount_source(argv: tuple[str, ...], destination: str) -> Path:
    for index, value in enumerate(argv):
        if value != "--mount":
            continue
        mount = argv[index + 1]
        if f"dst={destination}" not in mount:
            continue
        source = mount.split("src=", 1)[1].split(",dst=", 1)[0]
        return Path(source)
    raise AssertionError(f"Mount {destination} was not found")


def _environment_value(argv: tuple[str, ...], name: str) -> str:
    prefix = name + "="
    for index, value in enumerate(argv):
        if value == "--env" and argv[index + 1].startswith(prefix):
            return argv[index + 1][len(prefix) :]
    raise AssertionError(f"Environment value {name} was not found")


def _write_result(
    root: Path,
    *,
    job_id: str = JOB_ID,
    job_type: str = "training.prepare",
    project_id: str = PROJECT_ID,
    run_id: str = RUN_ID,
    image_digest: str = IMAGE_DIGEST,
    payload: dict | None = None,
    artifacts: list[dict] | None = None,
    manifest_sha256: str | None = None,
) -> None:
    result = {
        "schema": RESULT_MANIFEST_SCHEMA,
        "job_id": job_id,
        "job_type": job_type,
        "project_id": project_id,
        "run_id": run_id,
        "image_digest": image_digest,
        "manifest_sha256": manifest_sha256 or MANIFEST_SHA,
        "platform": "linux/amd64",
        "outcome": "completed",
        "payload": payload or {"metric": 1.25},
        "artifacts": artifacts or [],
    }
    (root / "result.json").write_text(
        json.dumps(result, ensure_ascii=True), encoding="utf-8"
    )
class SuccessfulRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], float]] = []
        self.inspect_count = 0

    def run(self, argv, *, timeout_seconds: float) -> CommandResult:
        command = tuple(argv)
        self.calls.append((command, timeout_seconds))
        operation = command[1]
        if operation == "run":
            output = _mount_source(command, "/aniflive/output")
            artifact = output / "artifacts" / "checkpoint.bin"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"trusted artifact")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            _write_result(
                output,
                manifest_sha256=_environment_value(
                    command, "ANIFLIVE_TTS_WORKER_MANIFEST_SHA256"
                ),
                artifacts=[
                    {
                        "kind": "checkpoint",
                        "relative_path": "artifacts/checkpoint.bin",
                        "sha256": digest,
                        "size_bytes": artifact.stat().st_size,
                    }
                ],
            )
            return CommandResult(0, CONTAINER_ID + "\n", "")
        if operation == "inspect":
            self.inspect_count += 1
            state = (
                {"Running": True, "ExitCode": 0}
                if self.inspect_count == 1
                else {"Running": False, "ExitCode": 0}
            )
            return CommandResult(0, json.dumps(state), "")
        if operation in {"rm", "stop", "logs"}:
            return CommandResult(0, "", "")
        if operation == "ps":
            return CommandResult(0, "", "")
        raise AssertionError(f"Unexpected Docker operation: {operation}")


def _broker(tmp_path: Path, runner, **settings) -> DockerWorkerBroker:
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    return DockerWorkerBroker(
        DockerBrokerSettings(IMAGE, **settings),
        output_root=tmp_path / "output",
        allowed_input_roots=(inputs,),
        runner=runner,
        run_id_factory=lambda: RUN_ID,
        sleep=lambda _seconds: None,
    )


def _manifest(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "manifest.json"
    path.write_text('{"schema":"prepared"}\n', encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_broker_requires_digest_pinned_image_and_defaults_to_no_network(
    tmp_path: Path,
) -> None:
    with pytest.raises(WorkstationError, match="digest"):
        DockerWorkerBroker(
            DockerBrokerSettings("ghcr.io/hiruynk/worker:latest"),
            output_root=tmp_path / "output",
            allowed_input_roots=(tmp_path,),
            runner=SuccessfulRunner(),
        )
    broker = _broker(tmp_path, SuccessfulRunner())
    assert broker.describe("training.prepare") == {
        "available": True,
        "kind": "linux-docker",
        "image": IMAGE,
        "image_digest": IMAGE_DIGEST,
        "network": "none",
    }
    assert broker.describe("dataset.inventory")["available"] is False
    assert broker.describe("dataset.process")["available"] is True


def test_execution_result_summarizes_artifacts_outside_bounded_job_json(
    tmp_path: Path,
) -> None:
    artifacts = tuple(
        {"kind": "evaluation-audio", "relative_path": f"audio/{index}.wav"}
        for index in range(600)
    )
    result = DockerExecutionResult(
        image=IMAGE,
        image_digest=IMAGE_DIGEST,
        container_id=CONTAINER_ID,
        run_id=RUN_ID,
        elapsed_seconds=1.0,
        payload={},
        artifacts=artifacts,
        output_root=tmp_path,
    ).as_dict()

    assert "artifacts" not in result
    assert result["artifact_inventory"] == {
        "count": 600,
        "kinds": {"evaluation-audio": 600},
    }
    assert len(json.dumps(result)) < 4_096


def test_dataset_process_uses_source_owned_worker_argv_and_dataset_artifacts(
    tmp_path: Path,
) -> None:
    dataset_project_id = "dataset_00000000-0000-0000-0000-000000000002"
    source = tmp_path / "inputs" / "voice.mp4"
    source.parent.mkdir()
    source.write_bytes(b"fixed-media")
    manifest, digest = _manifest(tmp_path)
    calls: list[tuple[str, ...]] = []

    class DatasetRunner:
        def run(self, argv, *, timeout_seconds: float) -> CommandResult:
            del timeout_seconds
            command = tuple(argv)
            calls.append(command)
            if command[1] == "run":
                output = _mount_source(command, "/aniflive/output")
                artifact = output / "dataset-pipeline" / "canonical.wav"
                artifact.parent.mkdir(parents=True)
                artifact.write_bytes(b"RIFFdataset")
                artifact_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
                _write_result(
                    output,
                    job_type="dataset.process",
                    project_id=dataset_project_id,
                    manifest_sha256=_environment_value(
                        command, "ANIFLIVE_TTS_WORKER_MANIFEST_SHA256"
                    ),
                    artifacts=[{
                        "kind": "dataset",
                        "relative_path": "dataset-pipeline/canonical.wav",
                        "sha256": artifact_digest,
                        "size_bytes": artifact.stat().st_size,
                    }],
                )
                return CommandResult(0, CONTAINER_ID, "")
            if command[1] == "inspect":
                return CommandResult(0, '{"Running":false,"ExitCode":0}', "")
            return CommandResult(0, "", "")

    broker = DockerWorkerBroker(
        DockerBrokerSettings(IMAGE),
        output_root=tmp_path / "output",
        allowed_input_roots=(source.parent,),
        runner=DatasetRunner(),
        run_id_factory=lambda: RUN_ID,
        sleep=lambda _seconds: None,
    )
    result = broker.execute(
        "dataset.process",
        manifest_path=manifest,
        manifest_sha256=digest,
        input_paths={"source": str(source)},
        job_id=JOB_ID,
        project_id=dataset_project_id,
        cancel_requested=lambda: False,
        emit_event=lambda _event: None,
    )
    command = calls[0]
    image_index = command.index(IMAGE)
    assert command[image_index + 1 :] == (
        "dataset",
        "--manifest",
        "/aniflive/job/manifest.json",
        "--result",
        "/aniflive/output/result.json",
    )
    source_mount = next(
        command[index + 1]
        for index, token in enumerate(command)
        if token == "--mount" and "dst=/aniflive/input/source.mp4" in command[index + 1]
    )
    assert source_mount.endswith(",readonly")
    assert result.artifacts[0]["kind"] == "dataset"


def test_worker_progress_protocol_is_strict_and_deduplicated() -> None:
    from aniflive_tts.workstation_docker import _worker_progress_events

    seen: set[str] = set()
    line = (
        'ANIFLIVE_TTS_PROGRESS '
        '{"progress":0.5,"stage":"gpt","message":"GPT epoch 2/4"}'
    )
    events = _worker_progress_events(
        "noise\n" + line + "\nANIFLIVE_TTS_PROGRESS {\"progress\":NaN}",
        seen=seen,
    )
    assert len(events) == 1
    assert events[0].progress == pytest.approx(0.59)
    assert events[0].message == "GPT epoch 2/4"
    assert _worker_progress_events(line, seen=seen) == ()


def test_config_is_versioned_strict_and_cannot_supply_runtime_argv(tmp_path: Path) -> None:
    config = tmp_path / "broker.json"
    config.write_text(
        json.dumps(
            {
                "schema": BROKER_CONFIG_SCHEMA,
                "image": IMAGE,
                "network": "none",
                "timeout_seconds": 600,
                "poll_seconds": 0.1,
            }
        ),
        encoding="utf-8",
    )
    settings = load_broker_settings(config)
    assert settings.image == IMAGE
    assert settings.network == "none"
    config.write_text(
        json.dumps(
            {
                "schema": BROKER_CONFIG_SCHEMA,
                "image": IMAGE,
                "argv": ["malicious"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(WorkstationError, match="unsupported field"):
        load_broker_settings(config)
    config.write_text(
        '{"schema":"aniflive-tts-docker-broker-config-v1",'
        f'"image":"{IMAGE}","image":"{IMAGE}"}}',
        encoding="utf-8",
    )
    with pytest.raises(WorkstationError, match="duplicate key"):
        load_broker_settings(config)


@pytest.mark.parametrize("network", ["host", "default", "container:other"])
def test_broker_rejects_host_or_container_networks(tmp_path: Path, network: str) -> None:
    with pytest.raises(WorkstationError, match="network policy"):
        DockerWorkerBroker(
            DockerBrokerSettings(IMAGE, network=network),
            output_root=tmp_path / "output",
            allowed_input_roots=(tmp_path,),
            runner=SuccessfulRunner(),
        )


def test_run_uses_fixed_hardened_linux_argv_and_read_only_inputs(tmp_path: Path) -> None:
    runner = SuccessfulRunner()
    broker = _broker(tmp_path, runner)
    dataset = tmp_path / "inputs" / "dataset"
    dataset.mkdir()
    manifest, digest = _manifest(tmp_path)
    events: list[AdapterEvent] = []
    result = broker.execute(
        "training.prepare",
        manifest_path=manifest,
        manifest_sha256=digest,
        input_paths={"dataset": str(dataset)},
        job_id=JOB_ID,
        project_id=PROJECT_ID,
        cancel_requested=lambda: False,
        emit_event=events.append,
    )
    command = runner.calls[0][0]
    assert command[:4] == ("docker", "run", "--detach", "--pull=never")
    assert ("--platform", "linux/amd64") == (
        command[command.index("--platform")],
        command[command.index("--platform") + 1],
    )
    assert ("--network", "none") == (
        command[command.index("--network")],
        command[command.index("--network") + 1],
    )
    assert "--read-only" in command
    assert "--gpus" in command and command[command.index("--gpus") + 1] == "device=0"
    assert "no-new-privileges:true" in command
    assert "--cap-drop" in command and command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--shm-size") + 1] == "1073741824"
    assert "--privileged" not in command
    assert "--volume" not in command
    assert not any(token in {"pull", "build"} for token in command[1:])
    image_index = command.index(IMAGE)
    assert command[image_index + 1 :] == (
        "training",
        "--manifest",
        "/aniflive/job/manifest.json",
        "--result",
        "/aniflive/output/result.json",
    )
    mounts = [command[index + 1] for index, token in enumerate(command) if token == "--mount"]
    assert next(value for value in mounts if "dst=/aniflive/job/manifest.json" in value).endswith(
        ",readonly"
    )
    assert next(value for value in mounts if "dst=/aniflive/input/dataset" in value).endswith(
        ",readonly"
    )
    assert not next(value for value in mounts if "dst=/aniflive/output" in value).endswith(
        ",readonly"
    )
    labels = [command[index + 1] for index, token in enumerate(command) if token == "--label"]
    assert any(value.endswith("=" + JOB_ID) for value in labels)
    assert any(value.endswith("=training.prepare") for value in labels)
    assert result.payload == {"metric": 1.25}
    assert result.artifacts[0]["kind"] == "checkpoint"
    assert events[-1].progress == 1.0
    assert any(call[0][1:3] == ("rm", "--force") for call in runner.calls)
    assert runner.calls[-1][0][1:3] == ("ps", "--all")


def test_job_parameters_cannot_change_broker_image_network_or_argv(tmp_path: Path) -> None:
    dataset = tmp_path / "inputs" / "dataset"
    dataset.mkdir(parents=True)
    shared_dir = tmp_path / "inputs" / "shared"
    shared_dir.mkdir()
    pretrained = {}
    for key, name in (
        ("pretrained_gpt", "gpt.ckpt"),
        ("pretrained_sovits_g", "G.pth"),
        ("pretrained_sovits_d", "D.pth"),
    ):
        path = tmp_path / "inputs" / name
        path.write_bytes(b"fixed-test-weight")
        pretrained[key] = str(path)
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(
        kind="training",
        name="Training",
        config={"dataset": str(dataset), "shared_dir": str(shared_dir), **pretrained},
    )
    job = store.create_job(
        job_type="training.prepare",
        project_id=project["id"],
        parameters={"preset": "balanced"},
    )
    runner = SuccessfulRunner()
    broker = DockerWorkerBroker(
        DockerBrokerSettings(IMAGE),
        output_root=tmp_path / "output",
        allowed_input_roots=(tmp_path / "inputs",),
        runner=runner,
        run_id_factory=lambda: RUN_ID,
        sleep=lambda _seconds: None,
    )

    # The fake result must carry the IDs created by the store.
    def corrected_run(argv, *, timeout_seconds: float) -> CommandResult:
        command = tuple(argv)
        runner.calls.append((command, timeout_seconds))
        if command[1] == "run":
            output = _mount_source(command, "/aniflive/output")
            _write_result(
                output,
                job_id=job["id"],
                project_id=project["id"],
                manifest_sha256=_environment_value(
                    command, "ANIFLIVE_TTS_WORKER_MANIFEST_SHA256"
                ),
            )
            return CommandResult(0, CONTAINER_ID, "")
        if command[1] == "inspect":
            return CommandResult(0, '{"Running":false,"ExitCode":0}', "")
        return CommandResult(0, "", "")

    runner.run = corrected_run  # type: ignore[method-assign]
    result = run_adapter(
        job,
        project,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path / "inputs",),
        worker_broker=broker,
    )
    assert result.outcome == "completed"
    assert result.payload["execution_performed"] is True
    command = runner.calls[0][0]
    assert "balanced" not in command
    assert command[command.index("--network") + 1] == "none"
    assert _environment_value(command, "NUMBA_CACHE_DIR") == "/tmp/numba-cache"
    assert command.count(IMAGE) == 1


def test_cancellation_stops_and_removes_container(tmp_path: Path) -> None:
    runner = SuccessfulRunner()
    broker = _broker(tmp_path, runner)
    dataset = tmp_path / "inputs" / "dataset"
    dataset.mkdir()
    manifest, digest = _manifest(tmp_path)
    cancellation_checks = iter((False, True))
    with pytest.raises(AdapterCancelled, match="cancelled"):
        broker.execute(
            "training.prepare",
            manifest_path=manifest,
            manifest_sha256=digest,
            input_paths={"dataset": str(dataset)},
            job_id=JOB_ID,
            project_id=PROJECT_ID,
            cancel_requested=lambda: next(cancellation_checks),
            emit_event=lambda _event: None,
        )
    operations = [call[0][1] for call in runner.calls]
    assert "stop" in operations
    assert operations.count("rm") >= 1


def test_worker_failure_preserves_final_diagnostic_after_large_logs(
    tmp_path: Path,
) -> None:
    class FailedRunner(SuccessfulRunner):
        def run(self, argv, *, timeout_seconds: float) -> CommandResult:
            command = tuple(argv)
            if command[1] == "inspect":
                self.calls.append((command, timeout_seconds))
                return CommandResult(0, '{"Running":false,"ExitCode":1}', "")
            if command[1] == "logs":
                self.calls.append((command, timeout_seconds))
                stdout = "worker progress\n" * 300
                stdout += "FINAL_CAUSE: validation gate failed"
                stderr = "inference progress\n" * 400
                return CommandResult(0, stdout, stderr)
            return super().run(command, timeout_seconds=timeout_seconds)

    broker = _broker(tmp_path, FailedRunner())
    dataset = tmp_path / "inputs" / "dataset"
    dataset.mkdir()
    manifest, digest = _manifest(tmp_path)

    with pytest.raises(WorkstationError, match="FINAL_CAUSE: validation gate failed"):
        broker.execute(
            "training.prepare",
            manifest_path=manifest,
            manifest_sha256=digest,
            input_paths={"dataset": str(dataset)},
            job_id=JOB_ID,
            project_id=PROJECT_ID,
            cancel_requested=lambda: False,
            emit_event=lambda _event: None,
        )


def test_cleanup_failure_is_safe_only_when_container_absence_is_verified(
    tmp_path: Path,
) -> None:
    class CleanupRunner(SuccessfulRunner):
        def __init__(self, *, remains_present: bool) -> None:
            super().__init__()
            self.remains_present = remains_present

        def run(self, argv, *, timeout_seconds: float) -> CommandResult:
            command = tuple(argv)
            if command[1] in {"stop", "rm"}:
                self.calls.append((command, timeout_seconds))
                return CommandResult(1, "", "synthetic cleanup failure")
            if command[1] == "ps" and "--quiet" in command:
                self.calls.append((command, timeout_seconds))
                return CommandResult(
                    0,
                    CONTAINER_ID + "\n" if self.remains_present else "",
                    "",
                )
            return super().run(command, timeout_seconds=timeout_seconds)

    absent_root = tmp_path / "absent"
    absent_root.mkdir()
    absent = _broker(absent_root, CleanupRunner(remains_present=False))
    absent_input = absent_root / "inputs" / "dataset"
    absent_input.mkdir(parents=True)
    absent_manifest, absent_digest = _manifest(absent_root)
    result = absent.execute(
        "training.prepare",
        manifest_path=absent_manifest,
        manifest_sha256=absent_digest,
        input_paths={"dataset": str(absent_input)},
        job_id=JOB_ID,
        project_id=PROJECT_ID,
        cancel_requested=lambda: False,
        emit_event=lambda _event: None,
    )
    assert result.container_id == CONTAINER_ID

    present_root = tmp_path / "present"
    present_root.mkdir()
    present = _broker(present_root, CleanupRunner(remains_present=True))
    present_input = present_root / "inputs" / "dataset"
    present_input.mkdir(parents=True)
    present_manifest, present_digest = _manifest(present_root)
    with pytest.raises(DockerCleanupUncertain, match="lease remains quarantined"):
        present.execute(
            "training.prepare",
            manifest_path=present_manifest,
            manifest_sha256=present_digest,
            input_paths={"dataset": str(present_input)},
            job_id=JOB_ID,
            project_id=PROJECT_ID,
            cancel_requested=lambda: False,
            emit_event=lambda _event: None,
        )


def test_broker_rejects_changed_manifest_and_outside_input_before_start(
    tmp_path: Path,
) -> None:
    runner = SuccessfulRunner()
    broker = _broker(tmp_path, runner)
    manifest, digest = _manifest(tmp_path)
    manifest.write_text("changed", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(WorkstationError, match="checksum changed"):
        broker.execute(
            "training.prepare",
            manifest_path=manifest,
            manifest_sha256=digest,
            input_paths={"dataset": str(outside)},
            job_id=JOB_ID,
            project_id=PROJECT_ID,
            cancel_requested=lambda: False,
            emit_event=lambda _event: None,
        )
    manifest, digest = _manifest(tmp_path)
    with pytest.raises(WorkstationError, match="outside the configured input roots"):
        broker.execute(
            "training.prepare",
            manifest_path=manifest,
            manifest_sha256=digest,
            input_paths={"dataset": str(outside)},
            job_id=JOB_ID,
            project_id=PROJECT_ID,
            cancel_requested=lambda: False,
            emit_event=lambda _event: None,
        )
    assert runner.calls == []


def test_timeout_stops_and_removes_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = SuccessfulRunner()
    broker = _broker(tmp_path, runner, timeout_seconds=1.0)
    dataset = tmp_path / "inputs" / "dataset"
    dataset.mkdir()
    manifest, digest = _manifest(tmp_path)
    times = iter((0.0, 2.0))
    monkeypatch.setattr(
        "aniflive_tts.workstation_docker.time.monotonic", lambda: next(times)
    )
    with pytest.raises(WorkstationError, match="timed out"):
        broker.execute(
            "training.prepare",
            manifest_path=manifest,
            manifest_sha256=digest,
            input_paths={"dataset": str(dataset)},
            job_id=JOB_ID,
            project_id=PROJECT_ID,
            cancel_requested=lambda: False,
            emit_event=lambda _event: None,
        )
    operations = [call[0][1] for call in runner.calls]
    assert "stop" in operations and "rm" in operations


def test_result_parser_rejects_wrong_job_stale_run_and_bad_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _write_result(output, job_id="job_00000000-0000-0000-0000-000000000009")
    arguments = {
        "output_root": output,
        "expected_job_id": JOB_ID,
        "expected_job_type": "training.prepare",
        "expected_project_id": PROJECT_ID,
        "expected_run_id": RUN_ID,
        "expected_image_digest": IMAGE_DIGEST,
        "expected_manifest_sha256": MANIFEST_SHA,
    }
    with pytest.raises(WorkstationError, match="different job"):
        parse_result_manifest(output / "result.json", **arguments)
    _write_result(output, run_id="9" * 32)
    with pytest.raises(WorkstationError, match="stale run"):
        parse_result_manifest(output / "result.json", **arguments)
    artifact = output / "artifact.bin"
    artifact.write_bytes(b"content")
    _write_result(
        output,
        artifacts=[
            {
                "kind": "checkpoint",
                "relative_path": "artifact.bin",
                "sha256": "0" * 64,
                "size_bytes": len(b"content"),
            }
        ],
    )
    with pytest.raises(WorkstationError, match="wrong checksum"):
        parse_result_manifest(output / "result.json", **arguments)
    _write_result(
        output,
        artifacts=[
            {
                "kind": "checkpoint",
                "relative_path": "../outside.bin",
                "sha256": "0" * 64,
                "size_bytes": 0,
            }
        ],
    )
    with pytest.raises(WorkstationError, match="invalid path"):
        parse_result_manifest(output / "result.json", **arguments)


@pytest.mark.parametrize(
    "relative_path",
    (
        "artifact.bin:stream",
        "CON",
        "aux.txt",
        "folder./artifact.bin",
        "folder /artifact.bin",
    ),
)
def test_result_parser_rejects_windows_special_artifact_paths(
    tmp_path: Path, relative_path: str
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _write_result(
        output,
        artifacts=[
            {
                "kind": "checkpoint",
                "relative_path": relative_path,
                "sha256": "0" * 64,
                "size_bytes": 0,
            }
        ],
    )
    with pytest.raises(WorkstationError, match="invalid path"):
        parse_result_manifest(
            output / "result.json",
            output_root=output,
            expected_job_id=JOB_ID,
            expected_job_type="training.prepare",
            expected_project_id=PROJECT_ID,
            expected_run_id=RUN_ID,
            expected_image_digest=IMAGE_DIGEST,
            expected_manifest_sha256=MANIFEST_SHA,
        )


def test_result_parser_rejects_control_fields_and_duplicate_json_keys(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _write_result(output, payload={"command": "do not execute"})
    arguments = {
        "output_root": output,
        "expected_job_id": JOB_ID,
        "expected_job_type": "training.prepare",
        "expected_project_id": PROJECT_ID,
        "expected_run_id": RUN_ID,
        "expected_image_digest": IMAGE_DIGEST,
        "expected_manifest_sha256": MANIFEST_SHA,
    }
    with pytest.raises(WorkstationError, match="execution control"):
        parse_result_manifest(output / "result.json", **arguments)
    text = (output / "result.json").read_text(encoding="utf-8")
    (output / "result.json").write_text(
        text.replace('"outcome": "completed"', '"outcome": "completed", "outcome": "completed"'),
        encoding="utf-8",
    )
    with pytest.raises(WorkstationError, match="duplicate key"):
        parse_result_manifest(output / "result.json", **arguments)
    _write_result(output)
    text = (output / "result.json").read_text(encoding="utf-8")
    (output / "result.json").write_text(
        text.replace("1.25", "NaN"), encoding="utf-8"
    )
    with pytest.raises(WorkstationError, match="non-finite"):
        parse_result_manifest(output / "result.json", **arguments)


def test_restart_reconciliation_retains_live_job_and_removes_orphans(tmp_path: Path) -> None:
    live_container = "5" * 64
    stale_container = "6" * 64
    stale_job = "job_00000000-0000-0000-0000-000000000007"

    class ReconcileRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []
            self.containers = {
                live_container: JOB_ID,
                stale_container: stale_job,
            }

        def run(self, argv, *, timeout_seconds: float) -> CommandResult:
            command = tuple(argv)
            self.calls.append(command)
            if command[1] == "ps":
                if "--quiet" in command:
                    selector = command[command.index("--filter") + 1]
                    container_id = selector.removeprefix("id=")
                    present = container_id in self.containers
                    return CommandResult(
                        0, container_id + "\n" if present else "", ""
                    )
                return CommandResult(
                    0,
                    "".join(
                        f"{container_id}\t{job_id}\n"
                        for container_id, job_id in self.containers.items()
                    ),
                    "",
                )
            if command[1] == "rm":
                self.containers.pop(command[-1], None)
            return CommandResult(0, "", "")

    runner = ReconcileRunner()
    broker = _broker(tmp_path, runner)
    report = broker.reconcile((JOB_ID,))
    assert report.retained == (live_container,)
    assert report.removed == (stale_container,)
    stale_calls = [call for call in runner.calls if stale_container in call]
    assert [call[1] for call in stale_calls] == ["stop", "rm"]
    assert not any(live_container in call for call in runner.calls[1:])


def test_reconciliation_does_not_report_an_unverified_container_removal(
    tmp_path: Path,
) -> None:
    stale_container = "6" * 64
    stale_job = "job_00000000-0000-0000-0000-000000000007"

    class UncertainReconcileRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def run(self, argv, *, timeout_seconds: float) -> CommandResult:
            command = tuple(argv)
            self.calls.append(command)
            if command[1] == "ps" and "--quiet" not in command:
                return CommandResult(0, f"{stale_container}\t{stale_job}\n", "")
            if command[1] == "ps":
                return CommandResult(0, stale_container + "\n", "")
            if command[1] in {"stop", "rm"}:
                return CommandResult(1, "", "synthetic Docker failure")
            raise AssertionError(f"Unexpected Docker operation: {command[1]}")

    runner = UncertainReconcileRunner()
    broker = _broker(tmp_path, runner)
    with pytest.raises(DockerCleanupUncertain, match="lease remains quarantined"):
        broker.reconcile(())
    assert [call[1] for call in runner.calls] == ["ps", "stop", "rm", "ps"]


def test_subprocess_runner_is_never_shell_true(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed["shell"] = kwargs.get("shell")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr("aniflive_tts.workstation_docker.subprocess.run", fake_run)
    result = SubprocessDockerRunner().run(
        ("docker", "version"), timeout_seconds=1.0
    )
    assert result.returncode == 0
    assert observed == {"argv": ["docker", "version"], "shell": False}


@pytest.mark.parametrize(
    "job_type",
    ["training.prepare", "checkpoint.select", "reference.select",
     "holdout.evaluate", "conversion.parity", "evaluation.prepare"],
)
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_selection_and_quality_jobs_forward_worker_progress(
    tmp_path: Path, job_type: str, stream: str
) -> None:
    class ProgressRunner(SuccessfulRunner):
        def run(self, argv, *, timeout_seconds: float) -> CommandResult:
            command = tuple(argv)
            if command[1] == "logs":
                self.calls.append((command, timeout_seconds))
                line = 'ANIFLIVE_TTS_PROGRESS {"progress":0.5,"message":"Validation 5/10"}'
                return CommandResult(
                    0, line if stream == "stdout" else "", line if stream == "stderr" else ""
                )
            result = super().run(command, timeout_seconds=timeout_seconds)
            if command[1] == "run":
                path = _mount_source(command, "/aniflive/output") / "result.json"
                payload = json.loads(path.read_text())
                payload["job_type"] = job_type
                path.write_text(json.dumps(payload), encoding="utf-8")
            return result

    runner = ProgressRunner()
    broker = _broker(tmp_path, runner)
    dataset = tmp_path / "inputs" / "dataset"
    dataset.mkdir()
    manifest, digest = _manifest(tmp_path)
    events = []
    broker.execute(
        job_type,
        manifest_path=manifest,
        manifest_sha256=digest,
        input_paths={"dataset": str(dataset)},
        job_id=JOB_ID,
        project_id=PROJECT_ID,
        cancel_requested=lambda: False,
        emit_event=events.append,
    )
    updates = [event for event in events if event.message == "Validation 5/10"]
    assert len(updates) == 1
    assert updates[0].progress == pytest.approx(0.59)
    assert events[-1].progress == 1.0


def test_broker_reconciliation_is_scoped_to_one_workstation(tmp_path):
    class Runner:
        def __init__(self):
            self.calls = []
        def run(self, argv, *, timeout_seconds):
            self.calls.append(tuple(argv))
            return CommandResult(0, "", "")
    first_runner, second_runner = Runner(), Runner()
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    first = _broker(tmp_path / "first", first_runner)
    second = _broker(tmp_path / "second", second_runner)
    assert first.workstation_scope != second.workstation_scope
    first.reconcile(())
    command = first_runner.calls[-1]
    assert "label=io.aniflive-tts.workstation.managed=worker-v2" in command
    assert "label=io.aniflive-tts.workstation.scope=" + first.workstation_scope in command
    reopened = _broker(tmp_path / "first", Runner())
    assert reopened.workstation_scope == first.workstation_scope


def test_legacy_or_other_scope_presence_cannot_release_expired_gpu_job(tmp_path):
    present = "job_00000000-0000-4000-8000-000000000009"
    absent = "job_00000000-0000-4000-8000-000000000010"
    class Runner:
        def __init__(self):
            self.calls = []
        def run(self, argv, *, timeout_seconds):
            self.calls.append(tuple(argv))
            return CommandResult(0, "f" * 64 if any(present in str(x) for x in argv) else "", "")
    runner = Runner()
    broker = _broker(tmp_path, runner)
    assert broker.confirmed_absent_jobs((present, absent)) == (absent,)
    assert all(command[1] == "ps" for command in runner.calls)
    assert not any("workstation.scope=" in str(command) for command in runner.calls)


def test_cleanup_waits_for_confirmed_asynchronous_removal(tmp_path):
    class Runner:
        def __init__(self):
            self.polls = 0
        def run(self, argv, *, timeout_seconds):
            if argv[1] == "rm":
                return CommandResult(1, "", "removal of container is already in progress")
            assert argv[1] == "ps"
            self.polls += 1
            return CommandResult(0, "e" * 64 if self.polls == 1 else "", "")
    runner = Runner()
    broker = _broker(tmp_path, runner)
    broker._remove("e" * 64)
    assert runner.polls == 2


@pytest.mark.parametrize("valid_ack", [True, False])
def test_checkpoint_pause_waits_for_worker_ack_without_stopping_container(tmp_path, valid_ack):
    class PausingRunner(SuccessfulRunner):
        def run(self, argv, *, timeout_seconds):
            result = super().run(argv, timeout_seconds=timeout_seconds)
            if argv[1] == "run":
                self.output = _mount_source(tuple(argv), "/aniflive/output")
            if argv[1] == "inspect":
                request = json.loads((self.output / "training-control.json").read_text())
                record = json.loads((self.output / "result.json").read_text())
                record["payload"] = {"status": "paused", "pause_ack": {
                    **request, "schema": "aniflive-training-pause-ack-v1",
                    "request_id": request["request_id"] if valid_ack else "wrong",
                }}
                (self.output / "result.json").write_text(json.dumps(record))
            return result
    runner = PausingRunner()
    broker = _broker(tmp_path, runner)
    manifest, digest = _manifest(tmp_path)
    events = []
    def execute():
        return broker.execute("training.prepare", manifest_path=manifest,
                              manifest_sha256=digest, input_paths={"dataset": str(tmp_path / "inputs")},
                              job_id=JOB_ID, project_id=PROJECT_ID,
                              cancel_requested=lambda: False,
                              pause_requested=lambda: True, emit_event=events.append)
    if valid_ack:
        assert execute().payload["status"] == "paused"
        assert all(event.progress < 1 for event in events)
    else:
        with pytest.raises(WorkstationError, match="acknowledgement"):
            execute()
    assert not any(command[1] == "stop" for command, _ in runner.calls)
