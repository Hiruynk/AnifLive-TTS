from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import subprocess
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import UUID, uuid4

from .workstation import WorkstationError, validated_artifact_relative_path, worker_artifact_limit


BROKER_CONFIG_SCHEMA = "aniflive-tts-docker-broker-config-v1"
RESULT_MANIFEST_SCHEMA = "aniflive-tts-docker-worker-result-v1"
MANAGED_LABEL = "io.aniflive-tts.workstation.managed"
MANAGED_LABEL_VALUE = "worker-v2"
WORKSTATION_SCOPE_LABEL = "io.aniflive-tts.workstation.scope"
JOB_ID_LABEL = "io.aniflive-tts.workstation.job-id"
JOB_TYPE_LABEL = "io.aniflive-tts.workstation.job-type"
PROJECT_ID_LABEL = "io.aniflive-tts.workstation.project-id"
RUN_ID_LABEL = "io.aniflive-tts.workstation.run-id"
MANIFEST_SHA_LABEL = "io.aniflive-tts.workstation.manifest-sha256"

_CONTAINER_ENTRYPOINT = "/opt/aniflive-tts/bin/workstation-worker"
_CONTAINER_MANIFEST = "/aniflive/job/manifest.json"
_CONTAINER_OUTPUT = "/aniflive/output"
_CONTAINER_RESULT = f"{_CONTAINER_OUTPUT}/result.json"
_LINUX_PLATFORM = "linux/amd64"
_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_DIGEST_IMAGE = re.compile(
    r"^[a-z0-9][a-z0-9._:/-]{0,254}@sha256:(?P<digest>[0-9a-f]{64})$"
)
_SAFE_NETWORK = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}$")
_CONTAINER_ID = re.compile(r"^[0-9a-f]{12,64}$")
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESULT_LIMIT_BYTES = 1024 * 1024
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024 * 1024

# This is deliberately source-owned. Neither a project nor a job can select an
# executable, subcommand or flag inside the worker image.
_ADAPTER_ARGV: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "dataset.process": (
            "dataset",
            "--manifest",
            _CONTAINER_MANIFEST,
            "--result",
            _CONTAINER_RESULT,
        ),
        "dataset.decode": (
            "dataset-decode",
            "--manifest",
            _CONTAINER_MANIFEST,
            "--result",
            _CONTAINER_RESULT,
        ),
        "dataset.target-speaker": (
            "dataset-target-speaker",
            "--manifest",
            _CONTAINER_MANIFEST,
            "--result",
            _CONTAINER_RESULT,
        ),
        "dataset.separate": (
            "dataset-separate",
            "--manifest",
            _CONTAINER_MANIFEST,
            "--result",
            _CONTAINER_RESULT,
        ),
        "dataset.transcribe": (
            "dataset-transcribe",
            "--manifest",
            _CONTAINER_MANIFEST,
            "--result",
            _CONTAINER_RESULT,
        ),
        "dataset.finalize": (
            "dataset-finalize",
            "--manifest",
            _CONTAINER_MANIFEST,
            "--result",
            _CONTAINER_RESULT,
        ),
        "tse.prepare": (
            "tse",
            "--manifest",
            _CONTAINER_MANIFEST,
            "--result",
            _CONTAINER_RESULT,
        ),
        "training.prepare": (
            "training",
            "--manifest",
            _CONTAINER_MANIFEST,
            "--result",
            _CONTAINER_RESULT,
        ),
        "checkpoint.select": (
            "checkpoint-selection",
            "--manifest",
            _CONTAINER_MANIFEST,
            "--result",
            _CONTAINER_RESULT,
        ),
        "reference.select": (
            "reference-selection",
            "--manifest",
            _CONTAINER_MANIFEST,
            "--result",
            _CONTAINER_RESULT,
        ),
        "holdout.evaluate": (
            "holdout-evaluation",
            "--manifest",
            _CONTAINER_MANIFEST,
            "--result",
            _CONTAINER_RESULT,
        ),
        "evaluation.prepare": (
            "evaluation",
            "--manifest",
            _CONTAINER_MANIFEST,
            "--result",
            _CONTAINER_RESULT,
        ),
        "engine.prepare": (
            "engine-build",
            "--manifest",
            _CONTAINER_MANIFEST,
            "--result",
            _CONTAINER_RESULT,
        ),
        "conversion.parity": (
            "conversion-parity",
            "--manifest",
            _CONTAINER_MANIFEST,
            "--result",
            _CONTAINER_RESULT,
        ),
        "model.package": (
            "model-package",
            "--manifest",
            _CONTAINER_MANIFEST,
            "--result",
            _CONTAINER_RESULT,
        ),
    }
)


class DockerCommandRunner(Protocol):
    def run(self, argv: Sequence[str], *, timeout_seconds: float) -> "CommandResult": ...


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class DockerCleanupUncertain(WorkstationError):
    """Docker could not prove that a GPU-owning worker container is absent."""


class SubprocessDockerRunner:
    """Small shell-free boundary around the locally installed Docker CLI."""

    def run(self, argv: Sequence[str], *, timeout_seconds: float) -> CommandResult:
        try:
            completed = subprocess.run(
                list(argv),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            raise WorkstationError("Docker control command timed out") from error
        except OSError as error:
            raise WorkstationError("Docker CLI could not be started") from error
        return CommandResult(
            int(completed.returncode),
            _bounded_text(completed.stdout),
            _bounded_text(completed.stderr),
        )


@dataclass(frozen=True)
class DockerBrokerSettings:
    image: str
    network: str = "none"
    timeout_seconds: float = 24 * 60 * 60
    poll_seconds: float = 0.25

    @property
    def image_digest(self) -> str:
        match = _DIGEST_IMAGE.fullmatch(self.image)
        if match is None:  # Defensive; construction validates this.
            raise WorkstationError("Docker worker image must be pinned by sha256 digest")
        return f"sha256:{match.group('digest')}"


@dataclass(frozen=True)
class DockerExecutionResult:
    image: str
    image_digest: str
    container_id: str
    run_id: str
    elapsed_seconds: float
    payload: Mapping[str, Any]
    artifacts: tuple[Mapping[str, Any], ...]
    output_root: Path

    def as_dict(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for artifact in self.artifacts:
            kind = str(artifact.get("kind") or "artifact")
            kinds[kind] = kinds.get(kind, 0) + 1
        return {
            "backend": "linux-docker",
            "image": self.image,
            "image_digest": self.image_digest,
            "container_id": self.container_id,
            "run_id": self.run_id,
            "elapsed_seconds": self.elapsed_seconds,
            "payload": dict(self.payload),
            "artifact_inventory": {
                "count": len(self.artifacts),
                "kinds": dict(sorted(kinds.items())),
            },
        }


@dataclass(frozen=True)
class ReconciliationReport:
    retained: tuple[str, ...]
    removed: tuple[str, ...]


def load_broker_settings(path: Path) -> DockerBrokerSettings:
    config_path = _resolved_regular_file(path, "Docker broker config")
    try:
        if config_path.stat().st_size > 64 * 1024:
            raise WorkstationError("Docker broker config is too large")
        document = _strict_json(config_path.read_text(encoding="utf-8"))
    except UnicodeError as error:
        raise WorkstationError("Docker broker config must be UTF-8") from error
    except OSError as error:
        raise WorkstationError("Docker broker config could not be read") from error
    if not isinstance(document, dict):
        raise WorkstationError("Docker broker config must be a JSON object")
    allowed = {"schema", "image", "network", "timeout_seconds", "poll_seconds"}
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise WorkstationError(f"Docker broker config has an unsupported field: {unknown[0]}")
    if document.get("schema") != BROKER_CONFIG_SCHEMA:
        raise WorkstationError("Docker broker config schema is unsupported")
    image = _digest_image(document.get("image"))
    network = _network_policy(document.get("network", "none"))
    timeout_seconds = _bounded_float(
        document.get("timeout_seconds", 24 * 60 * 60),
        "timeout_seconds",
        minimum=1.0,
        maximum=7 * 24 * 60 * 60,
    )
    poll_seconds = _bounded_float(
        document.get("poll_seconds", 0.25),
        "poll_seconds",
        minimum=0.05,
        maximum=5.0,
    )
    return DockerBrokerSettings(image, network, timeout_seconds, poll_seconds)


class DockerWorkerBroker:
    supports_checkpoint_pause = True

    """Launch one fixed Linux worker image through a constrained Docker contract."""

    def __init__(
        self,
        settings: DockerBrokerSettings,
        *,
        output_root: Path,
        allowed_input_roots: Sequence[Path],
        runner: DockerCommandRunner | None = None,
        docker_executable: str = "docker",
        run_id_factory: Callable[[], str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        image = _digest_image(settings.image)
        network = _network_policy(settings.network)
        timeout_seconds = _bounded_float(
            settings.timeout_seconds,
            "timeout_seconds",
            minimum=1.0,
            maximum=7 * 24 * 60 * 60,
        )
        poll_seconds = _bounded_float(
            settings.poll_seconds,
            "poll_seconds",
            minimum=0.05,
            maximum=5.0,
        )
        if not isinstance(docker_executable, str) or docker_executable != "docker":
            raise WorkstationError("Docker executable is fixed by the broker")
        roots = tuple(_resolved_existing_directory(path, "allowed input root") for path in allowed_input_roots)
        if not roots:
            raise WorkstationError("Docker broker requires at least one allowed input root")
        root = _prepare_output_root(output_root)
        self.settings = DockerBrokerSettings(image, network, timeout_seconds, poll_seconds)
        self.output_root = root
        scope_file = root / ".workstation-scope"
        try:
            with scope_file.open("x", encoding="ascii") as stream:
                stream.write(uuid4().hex + "\n")
        except FileExistsError:
            pass
        if scope_file.is_symlink() or not scope_file.is_file():
            raise WorkstationError("Docker workstation scope file is unsafe")
        scope = ""
        for _ in range(20):
            scope = scope_file.read_text(encoding="ascii").strip()
            if re.fullmatch(r"[a-f0-9]{32}", scope):
                break
            time.sleep(0.05)
        if re.fullmatch(r"[a-f0-9]{32}", scope) is None:
            raise WorkstationError("Docker workstation scope is malformed")
        self.workstation_scope = scope
        self.allowed_input_roots = roots
        self.runner = runner or SubprocessDockerRunner()
        self.docker_executable = docker_executable
        self.run_id_factory = run_id_factory or (lambda: uuid4().hex)
        self.sleep = sleep

    @classmethod
    def from_config(
        cls,
        config_path: Path,
        *,
        output_root: Path,
        allowed_input_roots: Sequence[Path],
        runner: DockerCommandRunner | None = None,
    ) -> "DockerWorkerBroker":
        return cls(
            load_broker_settings(config_path),
            output_root=output_root,
            allowed_input_roots=allowed_input_roots,
            runner=runner,
        )

    def configured(self, job_type: str) -> bool:
        return job_type in _ADAPTER_ARGV

    def describe(self, job_type: str) -> dict[str, Any]:
        return {
            "available": self.configured(job_type),
            "kind": "linux-docker",
            "image": self.settings.image if self.configured(job_type) else None,
            "image_digest": self.settings.image_digest if self.configured(job_type) else None,
            "network": self.settings.network if self.configured(job_type) else None,
        }

    def execute(
        self,
        job_type: str,
        *,
        manifest_path: Path,
        manifest_sha256: str,
        input_paths: Mapping[str, str],
        job_id: str,
        project_id: str,
        cancel_requested: Callable[[], bool],
        emit_event: Callable[[Any], None],
        pause_requested: Callable[[], bool] | None = None,
    ) -> DockerExecutionResult:
        adapter_argv = _ADAPTER_ARGV.get(job_type)
        if adapter_argv is None:
            raise WorkstationError(f"Docker worker does not support {job_type}")
        job_id = _record_id(job_id, "job_id", "job")
        project_id = _project_id(project_id)
        manifest = _resolved_regular_file(manifest_path, "worker manifest")
        manifest_digest = _sha256_text(manifest_sha256, "manifest_sha256")
        if _sha256_file(manifest) != manifest_digest:
            raise WorkstationError("Worker manifest checksum changed before Docker launch")
        mounts = self._validated_input_mounts(input_paths)
        run_id = self.run_id_factory()
        if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
            raise WorkstationError("Docker broker generated a malformed run ID")
        run_root = self._new_run_root(job_id, run_id)
        container_name = f"aniflive-tts-job-{job_id[4:12]}-{run_id[:12]}"
        argv = self._run_argv(
            container_name=container_name,
            job_id=job_id,
            job_type=job_type,
            project_id=project_id,
            run_id=run_id,
            manifest=manifest,
            manifest_digest=manifest_digest,
            output_root=run_root,
            input_mounts=mounts,
            adapter_argv=adapter_argv,
        )
        started = time.monotonic()
        container_id: str | None = None
        seen_progress: set[str] = set()
        pause_request_id: str | None = None
        try:
            if cancel_requested():
                raise _adapter_cancelled(
                    "Linux Docker worker was cancelled before launch"
                )
            started_result = self._command(argv, timeout_seconds=30.0, operation="start")
            if started_result.returncode != 0:
                self._remove(container_name)
                raise WorkstationError(_docker_failure("Docker worker could not start", started_result))
            reported_container_id = started_result.stdout.strip()
            if _CONTAINER_ID.fullmatch(reported_container_id) is None:
                self._remove(container_name)
                raise WorkstationError("Docker returned a malformed container ID")
            container_id = reported_container_id
            emit_event(_event(0.2, "Linux Docker worker started"))
            while True:
                if cancel_requested():
                    self._stop_and_remove(container_id)
                    container_id = None
                    raise _adapter_cancelled("Linux Docker worker was cancelled")
                if time.monotonic() - started > self.settings.timeout_seconds:
                    self._stop_and_remove(container_id)
                    container_id = None
                    raise WorkstationError("Linux Docker worker timed out")
                if (job_type == "training.prepare" and pause_requested is not None
                        and pause_request_id is None and pause_requested()):
                    pause_request_id = uuid4().hex
                    request = {
                        "schema": "aniflive-training-control-v1", "action": "pause",
                        "job_id": job_id, "run_id": run_id,
                        "request_id": pause_request_id,
                    }
                    temporary = run_root / "training-control.tmp"
                    temporary.write_text(json.dumps(request), encoding="utf-8")
                    temporary.replace(run_root / "training-control.json")
                    emit_event(_event(0.2, "Pause requested; waiting for a complete training checkpoint"))
                state = self._inspect_state(container_id)
                if job_type in {
                    "training.prepare",
                    "checkpoint.select",
                    "reference.select",
                    "holdout.evaluate",
                    "conversion.parity",
                    "evaluation.prepare",
                }:
                    for event in _worker_progress_events(
                        self._logs(container_id), seen=seen_progress
                    ):
                        emit_event(event)
                if not state["running"]:
                    exit_code = state["exit_code"]
                    break
                self.sleep(self.settings.poll_seconds)
            if exit_code != 0:
                logs = self._logs(container_id)
                raise WorkstationError(
                    f"Linux Docker worker exited with code {exit_code}"
                    + (f": {logs}" if logs else "")
                )
            parsed = parse_result_manifest(
                run_root / "result.json",
                output_root=run_root,
                expected_job_id=job_id,
                expected_job_type=job_type,
                expected_project_id=project_id,
                expected_run_id=run_id,
                expected_image_digest=self.settings.image_digest,
                expected_manifest_sha256=manifest_digest,
            )
            completed_container_id = container_id
            self._remove(container_id)
            container_id = None
            if parsed["payload"].get("status") == "paused":
                ack = parsed["payload"].get("pause_ack", {})
                if (job_type != "training.prepare" or pause_request_id is None
                        or ack.get("schema") != "aniflive-training-pause-ack-v1"
                        or ack.get("job_id") != job_id or ack.get("run_id") != run_id
                        or ack.get("request_id") != pause_request_id):
                    raise WorkstationError("Training pause acknowledgement does not match this request")
                emit_event(_event(0.2, "Training paused at a complete checkpoint"))
            else:
                emit_event(_event(1.0, "Linux Docker worker completed"))
            return DockerExecutionResult(
                image=self.settings.image,
                image_digest=self.settings.image_digest,
                container_id=completed_container_id,
                run_id=run_id,
                elapsed_seconds=time.monotonic() - started,
                payload=MappingProxyType(parsed["payload"]),
                artifacts=tuple(MappingProxyType(item) for item in parsed["artifacts"]),
                output_root=run_root,
            )
        finally:
            if container_id is not None:
                self._remove(container_id)

    def reconcile(self, active_job_ids: Sequence[str]) -> ReconciliationReport:
        active = {_record_id(value, "active_job_id", "job") for value in active_job_ids}
        command = [
            self.docker_executable,
            "ps",
            "--all",
            "--filter",
            f"label={MANAGED_LABEL}={MANAGED_LABEL_VALUE}",
            "--filter",
            f"label={WORKSTATION_SCOPE_LABEL}={self.workstation_scope}",
            "--format",
            f'{{{{.ID}}}}\t{{{{.Label "{JOB_ID_LABEL}"}}}}',
        ]
        result = self._command(command, timeout_seconds=15.0, operation="list")
        if result.returncode != 0:
            raise WorkstationError(_docker_failure("Docker reconciliation failed", result))
        retained: list[str] = []
        removed: list[str] = []
        seen_jobs: set[str] = set()
        for raw_line in result.stdout.splitlines():
            if not raw_line.strip():
                continue
            fields = raw_line.split("\t")
            if len(fields) != 2 or _CONTAINER_ID.fullmatch(fields[0]) is None:
                raise WorkstationError("Docker reconciliation returned malformed container metadata")
            container_id, job_id = fields
            job_id = _record_id(job_id, "container job_id", "job")
            if job_id in active and job_id not in seen_jobs:
                retained.append(container_id)
                seen_jobs.add(job_id)
            else:
                self._stop_and_remove(container_id)
                removed.append(container_id)
        return ReconciliationReport(tuple(retained), tuple(removed))

    def confirmed_absent_jobs(self, job_ids: Sequence[str]) -> tuple[str, ...]:
        """Do not release an expired GPU job if any legacy/foreign container still bears its ID."""
        absent = []
        for value in job_ids:
            job_id = _record_id(value, "expired_job_id", "job")
            result = self._command(
                [self.docker_executable, "ps", "--all", "--quiet", "--filter",
                 f"label={JOB_ID_LABEL}={job_id}"],
                timeout_seconds=15.0, operation="verify expired worker absence",
            )
            if result.returncode:
                raise DockerCleanupUncertain("Expired GPU worker absence could not be verified")
            ids = result.stdout.split()
            if any(_CONTAINER_ID.fullmatch(ident) is None for ident in ids):
                raise DockerCleanupUncertain("Expired GPU worker lookup returned malformed identities")
            if not ids:
                absent.append(job_id)
        return tuple(absent)

    def _validated_input_mounts(self, input_paths: Mapping[str, str]) -> tuple[tuple[str, Path, str], ...]:
        if not isinstance(input_paths, Mapping):
            raise WorkstationError("Docker input_paths must be a JSON object")
        mounts: list[tuple[str, Path, str]] = []
        for key, value in sorted(input_paths.items()):
            if not isinstance(key, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key) is None:
                raise WorkstationError("Docker input path key is malformed")
            path = _resolved_input(Path(value), self.allowed_input_roots, key)
            suffix = path.suffix.lower() if path.is_file() else ""
            if re.fullmatch(r"\.[a-z0-9]{1,12}", suffix) is None:
                suffix = ""
            mounts.append((key, path, f"/aniflive/input/{key}{suffix}"))
        if not mounts:
            raise WorkstationError("Docker worker requires at least one validated input mount")
        return tuple(mounts)

    def _new_run_root(self, job_id: str, run_id: str) -> Path:
        job_root = self.output_root / job_id
        run_root = job_root / run_id
        if _contains_reparse(self.output_root, job_root):
            raise WorkstationError("Docker output path contains a symbolic link or reparse point")
        try:
            run_root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise WorkstationError("Docker run output directory already exists") from error
        except OSError as error:
            raise WorkstationError("Docker run output directory could not be created") from error
        if _contains_reparse(self.output_root, run_root):
            raise WorkstationError("Docker output path contains a symbolic link or reparse point")
        return run_root.resolve(strict=True)

    def _run_argv(
        self,
        *,
        container_name: str,
        job_id: str,
        job_type: str,
        project_id: str,
        run_id: str,
        manifest: Path,
        manifest_digest: str,
        output_root: Path,
        input_mounts: Sequence[tuple[str, Path, str]],
        adapter_argv: Sequence[str],
    ) -> list[str]:
        argv = [
            self.docker_executable,
            "run",
            "--detach",
            "--pull=never",
            "--platform",
            _LINUX_PLATFORM,
            "--name",
            container_name,
            "--read-only",
            "--network",
            self.settings.network,
            "--gpus",
            "device=0",
            "--security-opt",
            "no-new-privileges:true",
            "--cap-drop",
            "ALL",
            "--pids-limit",
            "512",
            "--ipc",
            "private",
            "--shm-size",
            "1073741824",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=2147483648",
            "--label",
            f"{MANAGED_LABEL}={MANAGED_LABEL_VALUE}",
            "--label",
            f"{WORKSTATION_SCOPE_LABEL}={self.workstation_scope}",
            "--label",
            f"{JOB_ID_LABEL}={job_id}",
            "--label",
            f"{JOB_TYPE_LABEL}={job_type}",
            "--label",
            f"{PROJECT_ID_LABEL}={project_id}",
            "--label",
            f"{RUN_ID_LABEL}={run_id}",
            "--label",
            f"{MANIFEST_SHA_LABEL}={manifest_digest}",
            "--env",
            f"ANIFLIVE_TTS_WORKER_RESULT_SCHEMA={RESULT_MANIFEST_SCHEMA}",
            "--env",
            f"ANIFLIVE_TTS_WORKER_RUN_ID={run_id}",
            "--env",
            f"ANIFLIVE_TTS_WORKER_IMAGE_DIGEST={self.settings.image_digest}",
            "--env",
            f"ANIFLIVE_TTS_WORKER_MANIFEST_SHA256={manifest_digest}",
            "--env",
            f"ANIFLIVE_TTS_WORKER_PLATFORM={_LINUX_PLATFORM}",
            "--env",
            "NUMBA_CACHE_DIR=/tmp/numba-cache",
            "--mount",
            _mount_value(manifest, _CONTAINER_MANIFEST, read_only=True),
            "--mount",
            _mount_value(output_root, _CONTAINER_OUTPUT, read_only=False),
        ]
        for _key, source, destination in input_mounts:
            argv.extend(("--mount", _mount_value(source, destination, read_only=True)))
        argv.extend(("--entrypoint", _CONTAINER_ENTRYPOINT, self.settings.image, *adapter_argv))
        return argv

    def _inspect_state(self, container_id: str) -> dict[str, Any]:
        result = self._command(
            [
                self.docker_executable,
                "inspect",
                "--format",
                "{{json .State}}",
                container_id,
            ],
            timeout_seconds=10.0,
            operation="inspect",
        )
        if result.returncode != 0:
            raise WorkstationError(_docker_failure("Docker worker state could not be inspected", result))
        try:
            state = _strict_json(result.stdout.strip())
        except WorkstationError as error:
            raise WorkstationError("Docker returned malformed container state") from error
        if not isinstance(state, dict) or not isinstance(state.get("Running"), bool):
            raise WorkstationError("Docker returned malformed container state")
        exit_code = state.get("ExitCode")
        if not isinstance(exit_code, int) or isinstance(exit_code, bool):
            raise WorkstationError("Docker returned malformed container exit code")
        return {"running": state["Running"], "exit_code": exit_code}

    def _logs(self, container_id: str) -> str:
        result = self._command(
            [self.docker_executable, "logs", "--tail", "100", container_id],
            timeout_seconds=10.0,
            operation="logs",
        )
        return _bounded_streams(result.stdout, result.stderr, maximum=3072)

    def _stop_and_remove(self, container_id: str) -> None:
        stop_result = self._command(
            [self.docker_executable, "stop", "--time", "5", container_id],
            timeout_seconds=10.0,
            operation="stop",
            allow_failure=True,
        )
        self._remove(container_id, prior_result=stop_result)

    def _remove(
        self,
        container_reference: str,
        *,
        prior_result: CommandResult | None = None,
    ) -> None:
        remove_result = self._command(
            [self.docker_executable, "rm", "--force", container_reference],
            timeout_seconds=10.0,
            operation="remove",
            allow_failure=True,
        )
        if self._container_is_absent(container_reference):
            return
        removal_pending = (
            remove_result.returncode == 0
            or "in progress" in str(remove_result.stderr).lower()
        )
        if removal_pending:
            # Docker can acknowledge asynchronous removal before ps drops the
            # container. The GPU lease remains held throughout this bounded wait.
            for _ in range(20):
                self.sleep(0.25)
                if self._container_is_absent(container_reference):
                    return
        failures = [
            result
            for result in (prior_result, remove_result)
            if result is not None and result.returncode != 0
        ]
        detail = "; ".join(
            _docker_failure("Docker cleanup command failed", result)
            for result in failures
        )
        raise DockerCleanupUncertain(
            "Docker worker cleanup is uncertain; the GPU lease remains quarantined"
            + (f": {detail}" if detail else "")
        )

    def _container_is_absent(self, container_reference: str) -> bool:
        if _CONTAINER_ID.fullmatch(container_reference) is not None:
            selector = f"id={container_reference}"
        else:
            if re.fullmatch(r"aniflive-tts-job-[a-z0-9-]{1,64}", container_reference) is None:
                raise DockerCleanupUncertain(
                    "Docker cleanup received an invalid container reference"
                )
            selector = f"name=^/{container_reference}$"
        result = self._command(
            [
                self.docker_executable,
                "ps",
                "--all",
                "--quiet",
                "--filter",
                selector,
            ],
            timeout_seconds=10.0,
            operation="verify cleanup",
            allow_failure=True,
        )
        if result.returncode != 0:
            raise DockerCleanupUncertain(
                _docker_failure("Docker worker absence could not be verified", result)
            )
        identifiers = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if any(_CONTAINER_ID.fullmatch(value) is None for value in identifiers):
            raise DockerCleanupUncertain(
                "Docker cleanup verification returned malformed container metadata"
            )
        return not identifiers

    def _command(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        operation: str,
        allow_failure: bool = False,
    ) -> CommandResult:
        if not argv or argv[0] != self.docker_executable:
            raise WorkstationError("Docker broker attempted an invalid control command")
        result = self.runner.run(tuple(argv), timeout_seconds=timeout_seconds)
        if not isinstance(result, CommandResult):
            raise WorkstationError(f"Docker {operation} runner returned an invalid result")
        if result.returncode != 0 and not allow_failure:
            return result
        return result


def parse_result_manifest(
    path: Path,
    *,
    output_root: Path,
    expected_job_id: str,
    expected_job_type: str,
    expected_project_id: str,
    expected_run_id: str,
    expected_image_digest: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    expected_job_id = _record_id(expected_job_id, "expected_job_id", "job")
    if expected_job_type not in _ADAPTER_ARGV:
        raise WorkstationError("Expected Docker worker job type is unsupported")
    expected_project_id = _project_id(expected_project_id)
    if not isinstance(expected_run_id, str) or _RUN_ID.fullmatch(expected_run_id) is None:
        raise WorkstationError("Expected Docker worker run ID is malformed")
    if (
        not isinstance(expected_image_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", expected_image_digest) is None
    ):
        raise WorkstationError("Expected Docker worker image digest is malformed")
    expected_manifest_sha256 = _sha256_text(
        expected_manifest_sha256, "expected_manifest_sha256"
    )
    root = _resolved_existing_directory(output_root, "Docker run output root")
    result_path = _resolved_regular_file(path, "Docker worker result manifest")
    if not _is_within(result_path, (root,)):
        raise WorkstationError("Docker worker result escaped its output directory")
    try:
        size = result_path.stat().st_size
        if not 1 <= size <= _RESULT_LIMIT_BYTES:
            raise WorkstationError("Docker worker result manifest has an invalid size")
        document = _strict_json(result_path.read_text(encoding="utf-8"))
    except UnicodeError as error:
        raise WorkstationError("Docker worker result manifest must be UTF-8") from error
    except OSError as error:
        raise WorkstationError("Docker worker result manifest could not be read") from error
    if not isinstance(document, dict):
        raise WorkstationError("Docker worker result manifest must be a JSON object")
    required = {
        "schema",
        "job_id",
        "job_type",
        "project_id",
        "run_id",
        "image_digest",
        "manifest_sha256",
        "platform",
        "outcome",
        "payload",
        "artifacts",
    }
    if set(document) != required:
        raise WorkstationError("Docker worker result manifest fields are invalid")
    if document["schema"] != RESULT_MANIFEST_SCHEMA:
        raise WorkstationError("Docker worker result manifest schema is unsupported")
    if document["job_id"] != expected_job_id:
        raise WorkstationError("Docker worker result belongs to a different job")
    if document["job_type"] != expected_job_type:
        raise WorkstationError("Docker worker result has the wrong job type")
    if document["project_id"] != expected_project_id:
        raise WorkstationError("Docker worker result belongs to a different project")
    if document["run_id"] != expected_run_id:
        raise WorkstationError("Docker worker result belongs to a stale run")
    if document["image_digest"] != expected_image_digest:
        raise WorkstationError("Docker worker result has the wrong image digest")
    if document["manifest_sha256"] != expected_manifest_sha256:
        raise WorkstationError("Docker worker result has the wrong manifest digest")
    if document["platform"] != _LINUX_PLATFORM:
        raise WorkstationError("Docker worker result did not come from the Linux platform")
    if document["outcome"] != "completed":
        raise WorkstationError("Docker worker result did not report completion")
    payload = _safe_json_mapping(document["payload"], "Docker worker payload")
    artifacts_value = document["artifacts"]
    if not isinstance(artifacts_value, list) or len(artifacts_value) > worker_artifact_limit(expected_job_type):
        raise WorkstationError("Docker worker artifacts must be a bounded JSON array")
    artifacts = [
        _validated_artifact(value, root, index) for index, value in enumerate(artifacts_value)
    ]
    return {"payload": payload, "artifacts": artifacts}


def _validated_artifact(value: Any, root: Path, index: int) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "kind",
        "relative_path",
        "sha256",
        "size_bytes",
    }:
        raise WorkstationError(f"Docker worker artifact {index} is malformed")
    kind = value["kind"]
    if not isinstance(kind, str) or re.fullmatch(r"[a-z][a-z0-9._-]{0,63}", kind) is None:
        raise WorkstationError(f"Docker worker artifact {index} has an invalid kind")
    relative = value["relative_path"]
    try:
        posix = validated_artifact_relative_path(
            relative, field=f"Docker worker artifact {index} path"
        )
    except WorkstationError as error:
        raise WorkstationError(
            f"Docker worker artifact {index} has an invalid path"
        ) from error
    candidate = root.joinpath(*posix.parts)
    if _contains_reparse(root, candidate):
        raise WorkstationError(f"Docker worker artifact {index} contains a link")
    artifact = _resolved_regular_file(candidate, f"Docker worker artifact {index}")
    if not _is_within(artifact, (root,)):
        raise WorkstationError(f"Docker worker artifact {index} escaped its output directory")
    size_bytes = value["size_bytes"]
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or not 0 <= size_bytes <= _MAX_ARTIFACT_BYTES
        or artifact.stat().st_size != size_bytes
    ):
        raise WorkstationError(f"Docker worker artifact {index} has the wrong size")
    digest = _sha256_text(value["sha256"], f"artifact {index} sha256")
    if _sha256_file(artifact) != digest:
        raise WorkstationError(f"Docker worker artifact {index} has the wrong checksum")
    return {
        "kind": kind,
        "relative_path": relative,
        "sha256": digest,
        "size_bytes": size_bytes,
    }


def _event(progress: float, message: str):
    # Imported lazily to keep the Docker boundary independent from adapter setup.
    from .workstation_adapters import AdapterEvent

    return AdapterEvent(progress, "info", message)


def _worker_progress_events(logs: str, *, seen: set[str]) -> tuple[Any, ...]:
    prefix = "ANIFLIVE_TTS_PROGRESS "
    events = []
    for raw_line in logs.splitlines():
        # Broker diagnostics label the first line of each captured stream.
        raw_line = raw_line.removeprefix("[stdout] ").removeprefix("[stderr] ")
        if not raw_line.startswith(prefix) or raw_line in seen:
            continue
        seen.add(raw_line)
        try:
            value = _strict_json(raw_line[len(prefix) :])
        except (ValueError, json.JSONDecodeError, WorkstationError):
            continue
        if not isinstance(value, Mapping):
            continue
        progress = value.get("progress")
        message = value.get("message")
        if (
            not isinstance(progress, (int, float))
            or isinstance(progress, bool)
            or not math.isfinite(float(progress))
            or not 0.0 <= float(progress) <= 1.0
            or not isinstance(message, str)
            or not message.strip()
        ):
            continue
        events.append(_event(min(0.98, 0.2 + 0.78 * float(progress)), message[:2000]))
    return tuple(events)


def _adapter_cancelled(message: str) -> Exception:
    from .workstation_adapters import AdapterCancelled

    return AdapterCancelled(message)


def _digest_image(value: Any) -> str:
    if not isinstance(value, str) or _DIGEST_IMAGE.fullmatch(value) is None:
        raise WorkstationError("Docker worker image must be a lowercase digest reference")
    return value


def _network_policy(value: Any) -> str:
    if not isinstance(value, str) or _SAFE_NETWORK.fullmatch(value) is None:
        raise WorkstationError("Docker worker network policy is malformed")
    if value in {"host", "default"} or value.startswith("container"):
        raise WorkstationError("Docker worker network policy is not allowed")
    return value


def _bounded_float(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkstationError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise WorkstationError(f"{field} must be between {minimum:g} and {maximum:g}")
    return parsed


def _strict_json(text: str) -> Any:
    def object_pairs(pairs):
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise WorkstationError(f"JSON contains a duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str):
        raise WorkstationError(f"JSON contains a non-finite number: {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except WorkstationError:
        raise
    except (json.JSONDecodeError, RecursionError, TypeError) as error:
        raise WorkstationError("JSON document is malformed") from error


def _safe_json_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkstationError(f"{field} must be a JSON object")
    try:
        encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise WorkstationError(f"{field} contains unsupported values") from error
    if len(encoded.encode("utf-8")) > 512 * 1024:
        raise WorkstationError(f"{field} is too large")
    _reject_control_fields(value, field)
    return json.loads(encoded)


def _reject_control_fields(value: Any, field: str, depth: int = 0) -> None:
    if depth > 64:
        raise WorkstationError(f"{field} exceeds the maximum nesting depth")
    forbidden = {"argv", "command", "entrypoint", "executable", "image", "script", "shell"}
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).strip().lower() in forbidden:
                raise WorkstationError(f"{field} cannot contain execution control fields")
            _reject_control_fields(child, field, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _reject_control_fields(child, field, depth + 1)


def _record_id(value: Any, field: str, prefix: str) -> str:
    if not isinstance(value, str):
        raise WorkstationError(f"{field} is malformed")
    actual_prefix, separator, suffix = value.partition("_")
    if not separator or actual_prefix != prefix:
        raise WorkstationError(f"{field} is malformed")
    try:
        parsed = UUID(suffix)
    except (ValueError, AttributeError) as error:
        raise WorkstationError(f"{field} is malformed") from error
    if str(parsed) != suffix.lower():
        raise WorkstationError(f"{field} is malformed")
    return value


def _project_id(value: Any) -> str:
    if not isinstance(value, str):
        raise WorkstationError("project_id is malformed")
    prefix = value.partition("_")[0]
    if prefix not in {"dataset", "tse", "training", "evaluation"}:
        raise WorkstationError("project_id is malformed")
    return _record_id(value, "project_id", prefix)


def _sha256_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise WorkstationError(f"{field} is malformed")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise WorkstationError(f"File could not be hashed: {path.name}") from error
    return digest.hexdigest()


def _prepare_output_root(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        raise WorkstationError("Docker output root must use an absolute path")
    if _is_reparse(candidate):
        raise WorkstationError("Docker output root cannot be a symbolic link or reparse point")
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise WorkstationError("Docker output root could not be created") from error
    if not resolved.is_dir() or _is_reparse(resolved):
        raise WorkstationError("Docker output root must be a real directory")
    return resolved


def _resolved_existing_directory(path: Path, field: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute() or _is_reparse(candidate):
        raise WorkstationError(f"{field} must be an absolute real directory")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise WorkstationError(f"{field} was not found") from error
    if not resolved.is_dir():
        raise WorkstationError(f"{field} must be a directory")
    return resolved


def _resolved_regular_file(path: Path, field: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute() or _is_reparse(candidate):
        raise WorkstationError(f"{field} must be an absolute regular file")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise WorkstationError(f"{field} was not found") from error
    if not resolved.is_file() or _is_reparse(resolved):
        raise WorkstationError(f"{field} must be a regular file")
    return resolved


def _resolved_input(path: Path, roots: Sequence[Path], field: str) -> Path:
    raw = path.expanduser()
    if not raw.is_absolute():
        raise WorkstationError(f"{field} must use an absolute path")
    for root in roots:
        try:
            raw.relative_to(root)
        except ValueError:
            continue
        if _contains_reparse(root, raw):
            raise WorkstationError(f"{field} contains a symbolic link or reparse point")
    resolved = _resolved_existing_directory(path, field) if path.is_dir() else _resolved_regular_file(path, field)
    if not _is_within(resolved, roots):
        raise WorkstationError(f"{field} is outside the configured input roots")
    if _unsafe_mount_path(resolved):
        raise WorkstationError(f"{field} cannot be represented as a Docker bind mount")
    return resolved


def _is_within(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _is_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE)


def _contains_reparse(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() and _is_reparse(current):
            return True
    return False


def _unsafe_mount_path(path: Path) -> bool:
    value = str(path)
    return any(character in value for character in (",", "\n", "\r", "\x00"))


def _mount_value(source: Path, destination: str, *, read_only: bool) -> str:
    if _unsafe_mount_path(source):
        raise WorkstationError("Docker bind mount source contains an unsupported character")
    value = f"type=bind,src={source},dst={destination}"
    return f"{value},readonly" if read_only else value


def _bounded_text(value: Any, maximum: int = 64 * 1024) -> str:
    if not isinstance(value, str):
        return ""
    return value[:maximum]


def _bounded_tail(value: Any, maximum: int = 64 * 1024) -> str:
    if not isinstance(value, str):
        return ""
    return value[-maximum:]


def _bounded_streams(stdout: Any, stderr: Any, *, maximum: int) -> str:
    streams = [
        (name, value.strip())
        for name, value in (("stdout", stdout), ("stderr", stderr))
        if isinstance(value, str) and value.strip()
    ]
    if not streams:
        return ""
    allowance = max(1, (maximum - (16 * len(streams))) // len(streams))
    return "\n".join(
        f"[{name}] {_bounded_tail(value, maximum=allowance)}" for name, value in streams
    ).strip()


def _docker_failure(prefix: str, result: CommandResult) -> str:
    detail = _bounded_streams(result.stdout, result.stderr, maximum=3072)
    return prefix + (f": {detail}" if detail else "")


__all__ = [
    "BROKER_CONFIG_SCHEMA",
    "RESULT_MANIFEST_SCHEMA",
    "CommandResult",
    "DockerBrokerSettings",
    "DockerCleanupUncertain",
    "DockerCommandRunner",
    "DockerExecutionResult",
    "DockerWorkerBroker",
    "ReconciliationReport",
    "SubprocessDockerRunner",
    "load_broker_settings",
    "parse_result_manifest",
]
