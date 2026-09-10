from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import UUID

from .workstation import (
    JOB_PROJECT_KINDS,
    JOB_RESOURCE_CLASSES,
    JOB_TYPES,
    MEDIA_SUFFIXES,
    WorkstationError,
)


_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_EXECUTABLE_KEYS = frozenset(
    {
        "argv",
        "binary",
        "cmd",
        "command",
        "entrypoint",
        "executable",
        "flags",
        "image",
        "mounts",
        "network",
        "options",
        "program",
        "script",
        "shell",
    }
)
_PATH_KEYS = frozenset(
    {
        "audio",
        "asr_model",
        "baseline_report",
        "checkpoint",
        "checkpoint_candidates",
        "content_review",
        "dataset",
        "dataset_path",
        "deployment_reference",
        "diarization_model",
        "engine",
        "engine_dir",
        "input",
        "model_package",
        "listening_plan",
        "pretrained_gpt",
        "pretrained_sovits_d",
        "pretrained_sovits_g",
        "reference",
        "resume_checkpoint",
        "separation_model",
        "selected_checkpoints",
        "shared_dir",
        "speaker_component",
        "speaker_engine",
        "source",
        "vad_model",
    }
)
_TRAINING_SETTING_KEYS = frozenset(
    {
        "epoch_policy",
        "experiment_name",
        "gpt_batch_size",
        "gpt_epochs",
        "gpt_learning_rate",
        "gradient_checkpointing",
        "preset",
        "save_every_epoch",
        "seed",
        "sovits_batch_size",
        "sovits_epochs",
        "sovits_learning_rate",
        "source_dataset_artifact_id",
        "source_dataset_id",
        "source_dataset_manifest_sha256",
        "stage",
        "training_bundle_sha256",
    }
)


class AdapterCancelled(RuntimeError):
    """Raised when an adapter observes cooperative cancellation."""


@dataclass(frozen=True)
class AdapterEvent:
    progress: float
    level: str
    message: str


@dataclass(frozen=True)
class ArtifactHandoff:
    source_root: Path
    artifacts: tuple[Mapping[str, Any], ...]
    image_digest: str


@dataclass(frozen=True)
class AdapterResult:
    job_type: str
    resource_class: str
    outcome: str
    payload: Mapping[str, Any]
    artifact_handoff: ArtifactHandoff | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_type": self.job_type,
            "resource_class": self.resource_class,
            "outcome": self.outcome,
            "payload": dict(self.payload),
        }


CancelCallback = Callable[[], bool]
EventCallback = Callable[[AdapterEvent], None]


class JobAdapter(Protocol):
    job_type: str
    project_kinds: frozenset[str]
    resource_class: str

    def run(self, context: "AdapterContext") -> AdapterResult: ...


class BrokerExecution(Protocol):
    def as_dict(self) -> dict[str, Any]: ...


class WorkerBroker(Protocol):
    def describe(self, job_type: str) -> dict[str, Any]: ...

    def execute(
        self,
        job_type: str,
        *,
        manifest_path: Path,
        manifest_sha256: str,
        input_paths: Mapping[str, str],
        job_id: str,
        project_id: str,
        cancel_requested: CancelCallback,
        emit_event: EventCallback,
    ) -> BrokerExecution: ...

    def reconcile(self, active_job_ids: Sequence[str]) -> Any: ...


@dataclass(frozen=True)
class AdapterContext:
    job: Mapping[str, Any]
    project: Mapping[str, Any]
    manifest_root: Path
    allowed_path_roots: tuple[Path, ...]
    cancel_requested: CancelCallback
    emit_event: EventCallback
    command_allowlist: "CommandAllowlist | None"
    worker_broker: WorkerBroker | None
    maximum_inventory_files: int
    pause_requested: CancelCallback | None = None

    def check_cancelled(self) -> None:
        if self.cancel_requested():
            raise AdapterCancelled("Job cancellation was requested")

    def emit(self, progress: float, message: str, *, level: str = "info") -> None:
        if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
            raise WorkstationError("Adapter progress must be finite and between 0 and 1")
        if level not in {"debug", "info", "warning", "error"}:
            raise WorkstationError("Adapter event level is unsupported")
        self.emit_event(AdapterEvent(progress, level, _required_text(message, "message", 2000)))


@dataclass(frozen=True)
class AllowedCommand:
    """A trusted, fully specified command. Job records cannot alter its argv."""

    command_id: str
    executable: Path
    arguments: tuple[str, ...] = ()
    working_directory: Path | None = None
    environment: tuple[tuple[str, str], ...] = ()
    timeout_seconds: float = 3600.0


class CommandAllowlist:
    """Explicit command configuration for out-of-process heavy workers."""

    def __init__(
        self,
        commands: Mapping[str, AllowedCommand],
        *,
        executable_roots: Sequence[Path],
    ) -> None:
        roots = tuple(_resolved_existing_directory(path, "executable root") for path in executable_roots)
        if not roots:
            raise WorkstationError("At least one executable root is required")
        validated: dict[str, AllowedCommand] = {}
        for job_type, command in commands.items():
            if job_type not in JOB_TYPES:
                raise WorkstationError(f"Unsupported command job type: {job_type}")
            if not isinstance(command, AllowedCommand):
                raise WorkstationError("Configured commands must be AllowedCommand records")
            executable = command.executable.expanduser()
            if not executable.is_absolute():
                raise WorkstationError("Configured executables must use absolute paths")
            executable = executable.resolve(strict=True)
            if not executable.is_file() or not _is_within(executable, roots):
                raise WorkstationError("Configured executable is outside the executable allowlist")
            if command.working_directory is not None:
                working_directory = command.working_directory.expanduser()
                if not working_directory.is_absolute():
                    raise WorkstationError("Configured working directories must use absolute paths")
                working_directory = working_directory.resolve(strict=True)
                if not working_directory.is_dir():
                    raise WorkstationError("Configured working directory was not found")
            else:
                working_directory = None
            command_id = _required_text(command.command_id, "command_id", 80)
            arguments = tuple(_literal_text(value, "argument", 4096) for value in command.arguments)
            environment = tuple(
                (
                    _required_text(key, "environment key", 128),
                    _literal_text(value, "environment value", 4096, allow_empty=True),
                )
                for key, value in command.environment
            )
            if not math.isfinite(command.timeout_seconds) or command.timeout_seconds <= 0:
                raise WorkstationError("Configured command timeout must be positive")
            validated[job_type] = AllowedCommand(
                command_id=command_id,
                executable=executable,
                arguments=arguments,
                working_directory=working_directory,
                environment=environment,
                timeout_seconds=float(command.timeout_seconds),
            )
        self._commands = MappingProxyType(validated)

    def configured(self, job_type: str) -> bool:
        return job_type in self._commands

    def describe(self, job_type: str) -> dict[str, Any]:
        command = self._commands.get(job_type)
        if command is None:
            return {"available": False, "command_id": None}
        return {"available": True, "command_id": command.command_id}

    def execute(
        self,
        job_type: str,
        *,
        manifest_path: Path,
        cancel_requested: CancelCallback,
        emit_event: EventCallback,
    ) -> dict[str, Any]:
        """Run one fixed allowlisted command without a shell or user-supplied argv."""

        command = self._commands.get(job_type)
        if command is None:
            raise WorkstationError(f"No trusted command is configured for {job_type}")
        manifest = manifest_path.resolve(strict=True)
        environment = {
            key: os.environ[key]
            for key in (
                "COMSPEC",
                "HOME",
                "LANG",
                "LC_ALL",
                "SYSTEMROOT",
                "TEMP",
                "TMP",
                "USERPROFILE",
                "WINDIR",
            )
            if key in os.environ
        }
        environment.update(dict(command.environment))
        environment["ANIFLIVE_TTS_JOB_MANIFEST"] = str(manifest)
        started = time.monotonic()
        with tempfile.TemporaryFile(mode="w+b") as output:
            process = subprocess.Popen(
                [str(command.executable), *command.arguments],
                cwd=str(command.working_directory) if command.working_directory else None,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                shell=False,
            )
            while process.poll() is None:
                if cancel_requested():
                    process.terminate()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5.0)
                    raise AdapterCancelled("Configured worker command was cancelled")
                if time.monotonic() - started > command.timeout_seconds:
                    process.kill()
                    process.wait(timeout=5.0)
                    raise WorkstationError("Configured worker command timed out")
                time.sleep(0.05)
            output.seek(0)
            captured = output.read(64 * 1024 + 1)
        if len(captured) > 64 * 1024:
            captured = captured[: 64 * 1024]
        text = captured.decode("utf-8", errors="replace")
        event = AdapterEvent(1.0, "info" if process.returncode == 0 else "error", "Worker command exited")
        emit_event(event)
        return {
            "command_id": command.command_id,
            "exit_code": int(process.returncode),
            "elapsed_seconds": time.monotonic() - started,
            "output": text,
        }


@dataclass(frozen=True)
class AdapterSpec:
    job_type: str
    project_kinds: frozenset[str]
    resource_class: str
    required_paths: tuple[str, ...] = ()
    required_any_paths: tuple[tuple[str, ...], ...] = ()
    optional_paths: tuple[str, ...] = ()
    backend_label: str | None = None
    setting_keys: frozenset[str] | None = None


def _required_text(value: Any, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkstationError(f"{field} must not be empty")
    result = " ".join(value.strip().split())
    if len(result) > maximum:
        raise WorkstationError(f"{field} is limited to {maximum} characters")
    return result


def _literal_text(value: Any, field: str, maximum: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise WorkstationError(f"{field} must not be empty")
    if "\x00" in value:
        raise WorkstationError(f"{field} cannot contain a null byte")
    if len(value) > maximum:
        raise WorkstationError(f"{field} is limited to {maximum} characters")
    return value


def _record_id(value: Any, field: str, prefixes: Sequence[str]) -> str:
    text = _required_text(value, field, 80)
    prefix, separator, suffix = text.partition("_")
    if not separator or prefix not in prefixes:
        raise WorkstationError(f"{field} is malformed")
    try:
        parsed = UUID(suffix)
    except (ValueError, AttributeError) as error:
        raise WorkstationError(f"{field} is malformed") from error
    if str(parsed) != suffix.lower():
        raise WorkstationError(f"{field} is malformed")
    return text


def _json_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkstationError(f"{field} must be a JSON object")
    try:
        encoded = json.dumps(dict(value), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise WorkstationError(f"{field} must contain JSON-compatible values") from error
    if len(encoded.encode("utf-8")) > 64 * 1024:
        raise WorkstationError(f"{field} is too large")
    decoded = json.loads(encoded)
    _reject_executable_keys(decoded, field)
    return decoded


def _reject_executable_keys(value: Any, field: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).strip().lower() in _EXECUTABLE_KEYS:
                raise WorkstationError(
                    f"{field} cannot configure executables; use the trusted command allowlist"
                )
            _reject_executable_keys(child, field)
    elif isinstance(value, list):
        for child in value:
            _reject_executable_keys(child, field)


def _resolved_existing_directory(path: Path, field: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        raise WorkstationError(f"{field} must use an absolute path")
    if _is_reparse(candidate):
        raise WorkstationError(f"{field} cannot be a symbolic link or reparse point")
    try:
        candidate = candidate.resolve(strict=True)
    except OSError as error:
        raise WorkstationError(f"{field} was not found") from error
    if not candidate.is_dir():
        raise WorkstationError(f"{field} must be a directory")
    return candidate


def _is_within(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _is_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE)


def _validated_input_path(value: Any, roots: Sequence[Path], field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WorkstationError(f"{field} must be a local path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise WorkstationError(f"{field} must use an absolute local path")
    if _is_reparse(candidate):
        raise WorkstationError(f"{field} cannot be a symbolic link or reparse point")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise WorkstationError(f"{field} was not found") from error
    if not _is_within(resolved, roots):
        raise WorkstationError(f"{field} is outside the configured local roots")
    return resolved


def _merged_value(config: Mapping[str, Any], parameters: Mapping[str, Any], key: str) -> Any:
    return parameters[key] if key in parameters else config.get(key)


def _validated_paths(
    spec: AdapterSpec,
    config: Mapping[str, Any],
    parameters: Mapping[str, Any],
    roots: Sequence[Path],
) -> tuple[dict[str, str], list[str]]:
    validated: dict[str, str] = {}
    missing: list[str] = []
    permitted = tuple(
        dict.fromkeys(
            (
                *spec.required_paths,
                *(key for group in spec.required_any_paths for key in group),
                *spec.optional_paths,
            )
        )
    )
    for key in permitted:
        value = _merged_value(config, parameters, key)
        if value is not None:
            validated[key] = str(_validated_input_path(value, roots, key))
    for key in spec.required_paths:
        if key not in validated:
            missing.append(key)
    for group in spec.required_any_paths:
        if not any(key in validated for key in group):
            missing.append(" | ".join(group))
    if spec.job_type == "engine.prepare" and "checkpoint" in validated:
        if "shared_dir" not in validated:
            missing.append("shared_dir")
        if "model_package" not in validated and "reference" not in validated:
            missing.append("model_package | reference")
    return validated, missing


def _container_input_path(key: str, source: str) -> str:
    path = Path(source)
    suffix = path.suffix.lower() if path.is_file() else ""
    if (
        not suffix
        or len(suffix) > 13
        or not suffix[1:].isascii()
        or not suffix[1:].isalnum()
    ):
        suffix = ""
    return f"/aniflive/input/{key}{suffix}"


def _validated_records(
    spec: AdapterSpec, job: Mapping[str, Any], project: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    job_record = _json_mapping(job, "job")
    project_record = _json_mapping(project, "project")
    if job_record.get("type") != spec.job_type:
        raise WorkstationError(f"Adapter {spec.job_type} cannot run job {job_record.get('type')}")
    if project_record.get("kind") not in spec.project_kinds:
        allowed = ", ".join(sorted(spec.project_kinds))
        raise WorkstationError(f"{spec.job_type} requires a project of kind: {allowed}")
    job_id = _record_id(job_record.get("id"), "job_id", ("job",))
    project_kind = str(project_record["kind"])
    project_id = _record_id(project_record.get("id"), "project_id", (project_kind,))
    linked_project_id = _record_id(
        job_record.get("project_id"), "job project_id", (project_kind,)
    )
    if linked_project_id != project_id:
        raise WorkstationError("Job and project IDs do not match")
    if job_record.get("resource_class") != spec.resource_class:
        raise WorkstationError(
            f"{spec.job_type} requires resource class {spec.resource_class}"
        )
    job_record["id"] = job_id
    job_record["project_id"] = linked_project_id
    project_record["id"] = project_id
    config = _json_mapping(project_record.get("config", {}), "project config")
    parameters = _json_mapping(job_record.get("parameters", {}), "job parameters")
    return job_record, project_record, config, parameters


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_immutable_manifest(root: Path, job_type: str, job_id: str, payload: Mapping[str, Any]) -> tuple[Path, str]:
    manifest_root = root.expanduser().resolve()
    destination = manifest_root / job_type.replace(".", "-") / f"{job_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    if destination.exists():
        try:
            existing = destination.read_bytes()
        except OSError as error:
            raise WorkstationError("Existing preparation manifest could not be read") from error
        if existing != encoded:
            raise WorkstationError("Preparation manifest is immutable and does not match")
        return destination, digest
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    except FileExistsError:
        if not destination.exists() or destination.read_bytes() != encoded:
            raise WorkstationError("Preparation manifest is immutable and does not match")
    except OSError as error:
        raise WorkstationError("Preparation manifest could not be written") from error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return destination, digest


class DatasetInventoryAdapter:
    job_type = "dataset.inventory"
    project_kinds = JOB_PROJECT_KINDS[job_type]
    resource_class = JOB_RESOURCE_CLASSES[job_type]
    spec = AdapterSpec(job_type, project_kinds, resource_class, required_paths=("source",))

    def run(self, context: AdapterContext) -> AdapterResult:
        job, _project, config, parameters = _validated_records(self.spec, context.job, context.project)
        paths, missing = _validated_paths(
            self.spec, config, parameters, context.allowed_path_roots
        )
        if missing:
            raise WorkstationError(f"Dataset inventory requires local path: {missing[0]}")
        source = Path(paths["source"])
        context.check_cancelled()
        context.emit(0.02, "Dataset inventory started")
        media_files = total_bytes = examined = 0
        for path, file_info in _walk_local_files(source, context.allowed_path_roots):
            context.check_cancelled()
            examined += 1
            if examined > context.maximum_inventory_files:
                raise WorkstationError("Dataset inventory exceeds the configured file limit")
            if path.suffix.lower() in MEDIA_SUFFIXES:
                media_files += 1
                total_bytes += file_info.st_size
            if examined % 128 == 0:
                context.emit(min(0.95, 0.02 + examined / context.maximum_inventory_files), "Dataset inventory running")
        context.emit(1.0, "Dataset inventory completed")
        payload = MappingProxyType(
            {
                "execution_performed": True,
                "media_files": media_files,
                "total_bytes": total_bytes,
                "examined_files": examined,
                "source": str(source),
                "source_kind": "file" if source.is_file() else "directory",
                "adapter": self.job_type,
                "job_id": job["id"],
            }
        )
        return AdapterResult(self.job_type, self.resource_class, "completed", payload)


def _walk_local_files(source: Path, roots: Sequence[Path]):
    if source.is_file():
        yield source, source.stat()
        return
    stack = [source]
    visited: set[tuple[int, int]] = set()
    while stack:
        directory = stack.pop()
        info = directory.stat()
        identity = (int(info.st_dev), int(info.st_ino))
        if identity in visited:
            continue
        visited.add(identity)
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise WorkstationError(f"Dataset directory could not be inspected: {directory}") from error
        for entry in entries:
            path = Path(entry.path)
            try:
                entry_info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise WorkstationError(f"Dataset entry could not be inspected: {path}") from error
            if entry.is_symlink() or bool(
                getattr(entry_info, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
            ):
                raise WorkstationError("Dataset source contains a symbolic link or reparse point")
            resolved = path.resolve(strict=True)
            if not _is_within(resolved, roots):
                raise WorkstationError("Dataset entry escaped the configured local roots")
            if stat.S_ISDIR(entry_info.st_mode):
                stack.append(resolved)
            elif stat.S_ISREG(entry_info.st_mode):
                yield resolved, entry_info


class PreparationAdapter:
    def __init__(self, spec: AdapterSpec) -> None:
        self.spec = spec
        self.job_type = spec.job_type
        self.project_kinds = spec.project_kinds
        self.resource_class = spec.resource_class

    def run(self, context: AdapterContext) -> AdapterResult:
        job, project, config, parameters = _validated_records(
            self.spec, context.job, context.project
        )
        context.check_cancelled()
        context.emit(0.1, f"Preparing {self.spec.backend_label} worker manifest")
        paths, missing = _validated_paths(
            self.spec, config, parameters, context.allowed_path_roots
        )
        if context.worker_broker is not None:
            backend = context.worker_broker.describe(self.job_type)
        elif context.command_allowlist is not None:
            backend = {
                **context.command_allowlist.describe(self.job_type),
                "kind": "trusted-command-preparation-only",
            }
        else:
            backend = {"available": False, "kind": None, "command_id": None}
        ready = not missing and bool(backend["available"])
        settings = {
            key: value
            for key, value in {**config, **parameters}.items()
            if key not in _PATH_KEYS
            and (self.spec.setting_keys is None or key in self.spec.setting_keys)
        }
        manifest = {
            "schema": "aniflive-tts-worker-preparation-v1",
            "job_id": job["id"],
            "job_type": self.job_type,
            "project_id": project["id"],
            "project_kind": project["kind"],
            "resource_class": self.resource_class,
            "created_at": job.get("created_at"),
            "input_paths": paths,
            "container_input_paths": {
                key: _container_input_path(key, paths[key]) for key in sorted(paths)
            },
            "settings": settings,
            "missing_inputs": missing,
            "backend": {
                **backend,
                "label": self.spec.backend_label,
                "execution_performed": False,
            },
            "readiness": "ready" if ready else "blocked",
            "disposition": (
                "execute-linux-docker"
                if ready and context.worker_broker is not None
                else "prepared-only"
            ),
        }
        if self.job_type == "training.prepare":
            manifest["resume_requested"] = _merged_value(config, parameters, "resume_checkpoint") is not None
            if manifest["resume_requested"] and "resume_checkpoint" not in paths:
                raise WorkstationError("Resume checkpoint was requested but was not admitted to the worker")
        path, digest = _write_immutable_manifest(
            context.manifest_root, self.job_type, str(job["id"]), manifest
        )
        if ready and context.worker_broker is not None:
            context.check_cancelled()
            context.emit(0.15, "Dispatching validated job to Linux Docker")
            execution = context.worker_broker.execute(
                self.job_type,
                manifest_path=path,
                manifest_sha256=digest,
                input_paths=paths,
                job_id=str(job["id"]),
                project_id=str(project["id"]),
                cancel_requested=context.cancel_requested,
                emit_event=context.emit_event,
                **({"pause_requested": context.pause_requested} if context.pause_requested is not None else {}),
            )
            execution_payload = execution.as_dict()
            execution_artifacts = getattr(execution, "artifacts", ())
            artifact_handoff = None
            if execution_artifacts:
                output_root = getattr(execution, "output_root", None)
                image_digest = getattr(execution, "image_digest", None)
                if not isinstance(output_root, Path) or not isinstance(image_digest, str):
                    raise WorkstationError(
                        "Docker execution artifacts are missing their trusted host handoff"
                    )
                artifact_handoff = ArtifactHandoff(
                    source_root=output_root,
                    artifacts=tuple(
                        MappingProxyType(dict(artifact))
                        for artifact in execution_artifacts
                    ),
                    image_digest=image_digest,
                )
            payload = MappingProxyType(
                {
                    "execution_performed": True,
                    "readiness": "ready",
                    "missing_inputs": (),
                    "backend_available": True,
                    "backend": execution_payload,
                    "manifest_path": str(path),
                    "manifest_sha256": digest,
                }
            )
            return AdapterResult(
                self.job_type,
                self.resource_class,
                "completed",
                payload,
                artifact_handoff,
            )

        context.emit(
            1.0,
            "Worker preparation completed" if ready else "Worker preparation recorded as blocked",
            level="info" if ready else "warning",
        )
        payload = MappingProxyType(
            {
                "execution_performed": False,
                "readiness": manifest["readiness"],
                "missing_inputs": tuple(missing),
                "backend_available": bool(backend["available"]),
                "command_id": backend.get("command_id"),
                "manifest_path": str(path),
                "manifest_sha256": digest,
            }
        )
        return AdapterResult(self.job_type, self.resource_class, "prepared-only", payload)


_SPECS = (
    AdapterSpec(
        "dataset.process",
        JOB_PROJECT_KINDS["dataset.process"],
        JOB_RESOURCE_CLASSES["dataset.process"],
        required_paths=("source",),
        optional_paths=("asr_model",),
        backend_label="Dataset Decode and Segmentation",
    ),
    AdapterSpec(
        "dataset.decode",
        JOB_PROJECT_KINDS["dataset.decode"],
        JOB_RESOURCE_CLASSES["dataset.decode"],
        required_paths=("source",),
        backend_label="Dataset Source Decode",
    ),
    AdapterSpec(
        "dataset.target-speaker",
        JOB_PROJECT_KINDS["dataset.target-speaker"],
        JOB_RESOURCE_CLASSES["dataset.target-speaker"],
        required_paths=(
            "source",
            "reference",
            "speaker_engine",
            "vad_model",
            "diarization_model",
        ),
        backend_label="Dataset Target Speaker Routing",
    ),
    AdapterSpec(
        "dataset.separate",
        JOB_PROJECT_KINDS["dataset.separate"],
        JOB_RESOURCE_CLASSES["dataset.separate"],
        required_paths=("dataset", "reference", "speaker_engine", "separation_model"),
        backend_label="Dataset Target Speaker Salvage",
    ),
    AdapterSpec(
        "dataset.transcribe",
        JOB_PROJECT_KINDS["dataset.transcribe"],
        JOB_RESOURCE_CLASSES["dataset.transcribe"],
        required_paths=("dataset", "asr_model"),
        backend_label="Dataset Speech Recognition",
    ),
    AdapterSpec(
        "dataset.finalize",
        JOB_PROJECT_KINDS["dataset.finalize"],
        JOB_RESOURCE_CLASSES["dataset.finalize"],
        required_paths=("dataset",),
        backend_label="Dataset Acquisition Finalize",
    ),
    AdapterSpec(
        "tse.prepare", JOB_PROJECT_KINDS["tse.prepare"], JOB_RESOURCE_CLASSES["tse.prepare"],
        required_paths=("source", "reference", "model_package", "separation_model"),
        backend_label="Target Speaker Extraction",
    ),
    AdapterSpec(
        "training.prepare", JOB_PROJECT_KINDS["training.prepare"], JOB_RESOURCE_CLASSES["training.prepare"],
        required_paths=(
            "dataset",
            "pretrained_gpt",
            "pretrained_sovits_g",
            "pretrained_sovits_d",
            "shared_dir",
        ),
        optional_paths=("resume_checkpoint",),
        backend_label="Model Training",
        setting_keys=_TRAINING_SETTING_KEYS,
    ),
    AdapterSpec(
        "checkpoint.select",
        JOB_PROJECT_KINDS["checkpoint.select"],
        JOB_RESOURCE_CLASSES["checkpoint.select"],
        required_paths=(
            "checkpoint_candidates",
            "dataset",
            "shared_dir",
            "asr_model",
            "speaker_component",
        ),
        backend_label="Validation Checkpoint Selection",
    ),
    AdapterSpec(
        "reference.select",
        JOB_PROJECT_KINDS["reference.select"],
        JOB_RESOURCE_CLASSES["reference.select"],
        required_paths=("selected_checkpoints", "dataset", "shared_dir"),
        backend_label="Representative Reference Selection",
    ),
    AdapterSpec(
        "holdout.evaluate",
        JOB_PROJECT_KINDS["holdout.evaluate"],
        JOB_RESOURCE_CLASSES["holdout.evaluate"],
        required_paths=(
            "selected_checkpoints",
            "deployment_reference",
            "reference",
            "dataset",
            "shared_dir",
            "asr_model",
            "speaker_component",
        ),
        backend_label="Locked Test Holdout",
    ),
    AdapterSpec(
        "evaluation.prepare", JOB_PROJECT_KINDS["evaluation.prepare"], JOB_RESOURCE_CLASSES["evaluation.prepare"],
        required_paths=("model_package", "shared_dir", "asr_model"),
        optional_paths=(
            "baseline_report", "listening_plan", "selected_checkpoints", "deployment_reference",
        ),
        setting_keys=(
            "asr_compute_type", "semantic_sampling", "benchmark_language", "benchmark_text", "benchmark_runs",
            "benchmark_sessions", "benchmark_warmups", "request_timeout_seconds",
        ),
        backend_label="Evaluation",
    ),
    AdapterSpec(
        "engine.prepare", JOB_PROJECT_KINDS["engine.prepare"], JOB_RESOURCE_CLASSES["engine.prepare"],
        required_any_paths=(("checkpoint", "model_package"),),
        optional_paths=("shared_dir", "reference"),
        backend_label="TensorRT Engine Build",
    ),
    AdapterSpec(
        "conversion.parity",
        JOB_PROJECT_KINDS["conversion.parity"],
        JOB_RESOURCE_CLASSES["conversion.parity"],
        required_paths=(
            "model_package",
            "selected_checkpoints",
            "deployment_reference",
            "shared_dir",
            "asr_model",
        ),
        optional_paths=("content_review",),
        backend_label="PyTorch ONNX TensorRT Conversion Parity",
    ),
    AdapterSpec(
        "model.package", JOB_PROJECT_KINDS["model.package"], JOB_RESOURCE_CLASSES["model.package"],
        required_paths=("model_package",), backend_label="Model Packaging",
    ),
)

_registry: dict[str, JobAdapter] = {"dataset.inventory": DatasetInventoryAdapter()}
_registry.update({spec.job_type: PreparationAdapter(spec) for spec in _SPECS})
if frozenset(_registry) != JOB_TYPES:
    raise RuntimeError("The workstation adapter registry does not cover every job type")
ADAPTER_REGISTRY: Mapping[str, JobAdapter] = MappingProxyType(_registry)


def get_adapter(job_type: str) -> JobAdapter:
    try:
        return ADAPTER_REGISTRY[job_type]
    except KeyError as error:
        raise WorkstationError(f"Unsupported job adapter: {job_type}") from error


def run_adapter(
    job: Mapping[str, Any],
    project: Mapping[str, Any],
    *,
    manifest_root: Path,
    allowed_path_roots: Sequence[Path],
    cancel_requested: CancelCallback | None = None,
    emit_event: EventCallback | None = None,
    command_allowlist: CommandAllowlist | None = None,
    worker_broker: WorkerBroker | None = None,
    maximum_inventory_files: int = 200_000,
    pause_requested: CancelCallback | None = None,
) -> AdapterResult:
    """Validate and run the explicit adapter selected by the trusted job type."""

    if not isinstance(job, Mapping):
        raise WorkstationError("job must be a JSON object")
    adapter = get_adapter(str(job.get("type", "")))
    roots = tuple(_resolved_existing_directory(Path(root), "allowed path root") for root in allowed_path_roots)
    if not roots:
        raise WorkstationError("At least one allowed local path root is required")
    if not isinstance(maximum_inventory_files, int) or not 1 <= maximum_inventory_files <= 10_000_000:
        raise WorkstationError("maximum_inventory_files must be between 1 and 10000000")
    if command_allowlist is not None and worker_broker is not None:
        raise WorkstationError("Configure either a command allowlist or a Docker worker broker")
    context = AdapterContext(
        job=MappingProxyType(dict(job)),
        project=MappingProxyType(dict(project)),
        manifest_root=manifest_root,
        allowed_path_roots=roots,
        cancel_requested=cancel_requested or (lambda: False),
        emit_event=emit_event or (lambda _event: None),
        command_allowlist=command_allowlist,
        worker_broker=worker_broker,
        maximum_inventory_files=maximum_inventory_files,
        pause_requested=pause_requested,
    )
    return adapter.run(context)


__all__ = [
    "ADAPTER_REGISTRY",
    "AdapterCancelled",
    "AdapterContext",
    "AdapterEvent",
    "AdapterResult",
    "ArtifactHandoff",
    "AllowedCommand",
    "CommandAllowlist",
    "JobAdapter",
    "WorkerBroker",
    "get_adapter",
    "run_adapter",
]
