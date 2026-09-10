from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import stat
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .workstation import (
    WorkstationError,
    WorkstationStore,
    validated_artifact_relative_path,
)
from .workstation_adapters import (
    ADAPTER_REGISTRY,
    AdapterCancelled,
    AdapterEvent,
    AdapterResult,
    CommandAllowlist,
    WorkerBroker,
    run_adapter,
)
from .workstation_docker import DockerCleanupUncertain
from .workstation_dataset_scope import (
    DatasetScopeError,
    materialize_worker_dataset_scope,
)

WorkerReporter = Callable[[Mapping[str, Any]], None]
_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_BROKER_RECONCILIATION_INTERVAL_SECONDS = 30.0
GPU_WAIT_REASON = "Waiting for gpu:0; TensorRT inference or another GPU job is resident"
_MAX_DIRECT_LINEAGE_PARENTS = 64
_DEPENDENCY_LINEAGE_ANCHORS = {
    "training.prepare": (
        "training-report.json",
        "checkpoint-candidates.json",
    ),
    "checkpoint.select": (
        "checkpoint-selection-report.json",
        "selected/deployment-checkpoints.json",
    ),
    "reference.select": (
        "reference-selection-report.json",
        "blind-reference-manifest.json",
    ),
    "holdout.evaluate": ("holdout-evaluation.json",),
    "engine.prepare": (
        "engine-build.json",
        "model-package/manifest.json",
    ),
    "conversion.parity": (
        "conversion-parity-report.json",
        "model-package/manifest.json",
    ),
    "model.package": (
        "package-validation.json",
        "model-package/manifest.json",
    ),
    "evaluation.prepare": ("evaluation-report.json",),
}


@dataclass(frozen=True)
class WorkerRunResult:
    disposition: str
    job_id: str | None
    job_type: str | None
    store_status: str | None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "job_id": self.job_id,
            "job_type": self.job_type,
            "store_status": self.store_status,
            "error": self.error,
        }


@dataclass(frozen=True)
class WorkerSummary:
    processed: int
    completed: int
    blocked: int
    failed: int
    cancelled: int
    paused: int
    idle: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "completed": self.completed,
            "blocked": self.blocked,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "paused": self.paused,
            "idle": self.idle,
        }


class _LeaseMonitor:
    def __init__(
        self,
        store: WorkstationStore,
        job_id: str,
        claim_token: str,
        interval_seconds: float,
    ) -> None:
        self._store = store
        self._job_id = job_id
        self._claim_token = claim_token
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._error_lock = threading.Lock()
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"aniflive-worker-heartbeat-{job_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self._interval_seconds + 1.0))
        if self._thread.is_alive():
            raise WorkstationError("The job heartbeat thread did not stop")

    @property
    def error(self) -> Exception | None:
        with self._error_lock:
            return self._error

    def _set_error(self, error: Exception) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = error

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                self._store.heartbeat_job(self._job_id, self._claim_token)
            except Exception as error:
                self._set_error(error)
                self._stop.set()
                return


class WorkstationWorker:
    """Bounded worker for the explicit workstation adapter registry."""

    def __init__(
        self,
        store: WorkstationStore,
        *,
        manifest_root: Path | None = None,
        allowed_path_roots: Sequence[Path] | None = None,
        enabled_job_types: Sequence[str] | None = None,
        command_allowlist: CommandAllowlist | None = None,
        worker_broker: WorkerBroker | None = None,
        poll_seconds: float = 2.0,
        heartbeat_seconds: float = 15.0,
        maximum_inventory_files: int = 200_000,
        reporter: WorkerReporter | None = None,
    ) -> None:
        self.store = store
        self.manifest_root = (manifest_root or (store.root / "worker-manifests")).resolve()
        self.dependency_input_root = (store.root / "dependency-inputs").resolve()
        self.dependency_input_root.mkdir(parents=True, exist_ok=True)
        configured_roots = tuple(
            _validated_root(Path(root))
            for root in (
                allowed_path_roots
                if allowed_path_roots is not None
                else store.allowed_import_roots()
            )
        )
        roots = tuple(
            dict.fromkeys((*configured_roots, _validated_root(self.dependency_input_root)))
        )
        if not roots:
            raise WorkstationError("At least one allowed local path root is required")
        for root in roots:
            if not root.is_dir():
                raise WorkstationError(f"Allowed local path root was not found: {root}")
        selected = tuple(enabled_job_types or ADAPTER_REGISTRY.keys())
        if not selected:
            raise WorkstationError("At least one worker adapter must be enabled")
        if len(selected) != len(set(selected)):
            raise WorkstationError("enabled_job_types contains duplicates")
        unsupported = sorted(set(selected) - set(ADAPTER_REGISTRY))
        if unsupported:
            raise WorkstationError(f"Unsupported worker adapter: {unsupported[0]}")
        if not math.isfinite(poll_seconds) or not 0.05 <= poll_seconds <= 300.0:
            raise WorkstationError("poll_seconds must be between 0.05 and 300")
        if not math.isfinite(heartbeat_seconds) or not 0.05 <= heartbeat_seconds <= 3600.0:
            raise WorkstationError("heartbeat_seconds must be between 0.05 and 3600")
        if not isinstance(maximum_inventory_files, int) or not (
            1 <= maximum_inventory_files <= 10_000_000
        ):
            raise WorkstationError(
                "maximum_inventory_files must be between 1 and 10000000"
            )
        if command_allowlist is not None and worker_broker is not None:
            raise WorkstationError(
                "Configure either a command allowlist or a Docker worker broker"
            )
        self.allowed_path_roots = roots
        self.enabled_job_types = frozenset(selected)
        self.command_allowlist = command_allowlist
        self.worker_broker = worker_broker
        self.poll_seconds = float(poll_seconds)
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.maximum_inventory_files = maximum_inventory_files
        self.reporter = reporter or (lambda _record: None)
        self._shutdown = threading.Event()
        self._next_broker_reconciliation_at = 0.0
        self.runtime_handoff = None

    def request_stop(self) -> None:
        self._shutdown.set()

    @property
    def stopping(self) -> bool:
        return self._shutdown.is_set()

    @contextmanager
    def signal_handlers(self) -> Iterator[None]:
        """Convert SIGINT/SIGTERM into cooperative worker shutdown."""

        if threading.current_thread() is not threading.main_thread():
            yield
            return
        installed: dict[signal.Signals, Any] = {}

        def stop_handler(_signum, _frame) -> None:
            self.request_stop()

        candidates = [signal.SIGINT]
        if hasattr(signal, "SIGTERM"):
            candidates.append(signal.SIGTERM)
        if hasattr(signal, "SIGBREAK"):
            candidates.append(signal.SIGBREAK)
        try:
            for candidate in candidates:
                installed[candidate] = signal.getsignal(candidate)
                signal.signal(candidate, stop_handler)
            yield
        finally:
            for candidate, previous in installed.items():
                signal.signal(candidate, previous)

    def run(self, *, once: bool = False, max_jobs: int | None = None) -> WorkerSummary:
        if max_jobs is not None and (not isinstance(max_jobs, int) or max_jobs < 1):
            raise WorkstationError("max_jobs must be a positive integer")
        counts = {
            "completed": 0,
            "blocked": 0,
            "failed": 0,
            "cancelled": 0,
            "paused": 0,
        }
        processed = 0
        observed_idle = False
        while not self.stopping:
            result = self.run_once()
            if result.disposition == "idle":
                observed_idle = True
                if once:
                    break
                self._shutdown.wait(self.poll_seconds)
                continue
            observed_idle = False
            processed += 1
            counts[result.disposition] += 1
            self.reporter({"kind": "job-result", **result.as_dict()})
            if once or (max_jobs is not None and processed >= max_jobs):
                break
        return WorkerSummary(
            processed=processed,
            completed=counts["completed"],
            blocked=counts["blocked"],
            failed=counts["failed"],
            cancelled=counts["cancelled"],
            paused=counts["paused"],
            idle=observed_idle and processed == 0,
        )

    def run_once(self) -> WorkerRunResult:
        if self.runtime_handoff is None:
            return self._run_once()
        self._reconcile_broker_if_due()
        self.runtime_handoff.recover()
        try:
            return self._run_once()
        finally:
            self.runtime_handoff.recover()

    def _run_once(self) -> WorkerRunResult:
        if self.stopping:
            return WorkerRunResult("idle", None, None, None)
        self._reconcile_broker_if_due()
        claim = self._claim_next()
        if claim is None:
            return WorkerRunResult("idle", None, None, None)
        job = claim.job
        job_id = str(job["id"])
        job_type = str(job["type"])
        token = claim.token
        try:
            heartbeat_interval = min(
                self.heartbeat_seconds,
                _safe_heartbeat_interval(claim.lease_expires_at),
            )
        except Exception as error:
            return self._finish_failed(job_id, job_type, token, error)
        monitor = _LeaseMonitor(
            self.store,
            job_id,
            token,
            heartbeat_interval,
        )
        try:
            monitor.start()
        except Exception as error:
            return self._finish_failed(job_id, job_type, token, error)

        checkpoint_pause = (
            job_type == "training.prepare"
            and getattr(self.worker_broker, "supports_checkpoint_pause", False) is True
        )

        def pause_requested() -> bool:
            return self.store.job_pause_requested(job_id, token)

        def cancel_requested() -> bool:
            heartbeat_error = monitor.error
            if heartbeat_error is not None:
                raise WorkstationError(
                    f"Job lease heartbeat failed: {_safe_error(heartbeat_error)}"
                )
            if self.stopping:
                return True
            return self.store.job_cancel_requested(
                job_id, token
            ) or (not checkpoint_pause and pause_requested())

        last_progress = float(job.get("progress", 0.0))

        def emit_event(event: AdapterEvent) -> None:
            nonlocal last_progress
            cancel_requested()
            progress = min(float(event.progress), 0.99)
            if progress > last_progress:
                self.store.update_job(job_id, progress=progress, claim_token=token)
                last_progress = progress
            self.store.append_job_log(
                job_id, event.level, event.message, claim_token=token
            )
            self.reporter(
                {
                    "kind": "job-event",
                    "job_id": job_id,
                    "job_type": job_type,
                    "progress": progress,
                    "level": event.level,
                    "message": event.message,
                }
            )

        try:
            adapter_result: AdapterResult | None = None
            adapter_error: Exception | None = None
            try:
                project_id = job.get("project_id")
                if not isinstance(project_id, str):
                    raise WorkstationError("Worker jobs require a project")
                project = self.store.get_project(project_id)
                emit_event(AdapterEvent(0.01, "info", "Validating dependency inputs before Docker dispatch"))
                with self._job_with_dependency_inputs(job) as runnable_job:
                    adapter_result = run_adapter(
                        runnable_job,
                        project,
                        manifest_root=self.manifest_root,
                        allowed_path_roots=self.allowed_path_roots,
                        cancel_requested=cancel_requested,
                        pause_requested=pause_requested if checkpoint_pause else None,
                        emit_event=emit_event,
                        command_allowlist=self.command_allowlist,
                        worker_broker=self.worker_broker,
                        maximum_inventory_files=self.maximum_inventory_files,
                    )
                if (checkpoint_pause
                        and _worker_evidence(adapter_result.payload).get("status") == "paused"):
                    adapter_result = AdapterResult(
                        adapter_result.job_type, adapter_result.resource_class,
                        "checkpoint-paused", adapter_result.payload,
                    )
                if adapter_result.artifact_handoff is not None:
                    cancel_requested()
                    parameters = job.get("parameters", {})
                    if not isinstance(parameters, Mapping):
                        raise WorkstationError("Job parameters are malformed")
                    parent_artifact_ids = self._artifact_parent_ids(job, parameters)
                    self.store.heartbeat_job(job_id, token)
                    registered = self.store.register_worker_artifacts(
                        job_id=job_id,
                        claim_token=token,
                        job_type=job_type,
                        project_id=project_id,
                        source_root=adapter_result.artifact_handoff.source_root,
                        artifacts=adapter_result.artifact_handoff.artifacts,
                        image_digest=adapter_result.artifact_handoff.image_digest,
                        parent_artifact_ids=parent_artifact_ids,
                    )
                    cancel_requested()
                    emit_event(AdapterEvent(0.98, "info", "Docker artifacts registered"))
                    adapter_result = AdapterResult(
                        adapter_result.job_type,
                        adapter_result.resource_class,
                        adapter_result.outcome,
                        {
                            **dict(adapter_result.payload),
                            "registered_artifact_ids": [
                                artifact["id"] for artifact in registered
                            ],
                        },
                    )
            except Exception as error:
                adapter_error = error
            if monitor.error is not None:
                return WorkerRunResult(
                    "failed", job_id, job_type, "running", _safe_error(monitor.error)
                )
            if isinstance(adapter_error, DockerCleanupUncertain):
                message = _safe_error(adapter_error)
                try:
                    self.store.expire_job_for_docker_cleanup(
                        job_id, token, error=message
                    )
                except WorkstationError:
                    # The original resource lease is deliberately not released. If
                    # fencing races expiry, reconciliation will still be required
                    # before a GPU job lease can be reclaimed.
                    pass
                return WorkerRunResult("failed", job_id, job_type, "running", message)
            if isinstance(adapter_error, AdapterCancelled):
                try:
                    if self.store.job_cancel_requested(job_id, token):
                        return self._finish_cancelled(
                            job_id, job_type, token, adapter_error
                        )
                    if self.store.job_pause_requested(job_id, token):
                        return self._finish_paused(job_id, job_type, token)
                except WorkstationError as state_error:
                    return WorkerRunResult(
                        "failed", job_id, job_type, "running", _safe_error(state_error)
                    )
                return self._finish_cancelled(job_id, job_type, token, adapter_error)
            if adapter_error is not None:
                return self._finish_failed(job_id, job_type, token, adapter_error)
            if adapter_result is None:
                return self._finish_failed(
                    job_id, job_type, token, WorkstationError("Adapter returned no result")
                )
            return self._finish_result(job_id, job_type, token, adapter_result)
        finally:
            monitor.stop()

    def _artifact_parent_ids(
        self, job: Mapping[str, Any], parameters: Mapping[str, Any]
    ) -> Sequence[str] | None:
        """Resolve lineage when dependency outputs do not exist at queue time."""

        if "parent_artifact_ids" in parameters:
            value = parameters["parent_artifact_ids"]
            if value is None:
                return None
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise WorkstationError("parent_artifact_ids must be a list of artifact IDs")
            return list(value)

        dependencies = job.get("depends_on", [])
        if isinstance(dependencies, (str, bytes)) or not isinstance(
            dependencies, Sequence
        ):
            raise WorkstationError("Job dependencies are malformed")

        result: list[str] = []
        seen: set[str] = set()
        for dependency_id in dependencies:
            dependency = self.store.get_job(str(dependency_id))
            if dependency.get("status") != "succeeded":
                raise WorkstationError(
                    f"Dependency {dependency_id} has not completed successfully"
                )
            dependency_result = dependency.get("result", {})
            if not isinstance(dependency_result, Mapping):
                raise WorkstationError(f"Dependency {dependency_id} result is malformed")
            artifact_ids = dependency_result.get("registered_artifact_ids", [])
            if isinstance(artifact_ids, (str, bytes)) or not isinstance(
                artifact_ids, Sequence
            ):
                raise WorkstationError(
                    f"Dependency {dependency_id} artifact lineage is malformed"
                )
            selected_ids = self._dependency_lineage_ids(dependency, artifact_ids)
            for artifact_id in selected_ids:
                if not isinstance(artifact_id, str):
                    raise WorkstationError(
                        f"Dependency {dependency_id} artifact lineage is malformed"
                    )
                if artifact_id not in seen:
                    seen.add(artifact_id)
                    result.append(artifact_id)
        return result or None

    def _dependency_lineage_ids(
        self,
        dependency: Mapping[str, Any],
        artifact_ids: Sequence[Any],
    ) -> list[str]:
        if len(artifact_ids) <= _MAX_DIRECT_LINEAGE_PARENTS:
            if any(not isinstance(artifact_id, str) for artifact_id in artifact_ids):
                raise WorkstationError(
                    f"Dependency {dependency.get('id')} artifact lineage is malformed"
                )
            return list(artifact_ids)
        job_type = str(dependency.get("type", ""))
        anchors = _DEPENDENCY_LINEAGE_ANCHORS.get(job_type)
        if anchors is None:
            raise WorkstationError(
                f"Dependency {dependency.get('id')} publishes too many direct lineage parents"
            )
        required = set(anchors)
        selected: dict[str, str] = {}
        for artifact_id in artifact_ids:
            if not isinstance(artifact_id, str):
                raise WorkstationError(
                    f"Dependency {dependency.get('id')} artifact lineage is malformed"
                )
            artifact = self.store.get_artifact(artifact_id)
            metadata = artifact.get("metadata", {})
            relative_path = (
                metadata.get("worker_relative_path")
                if isinstance(metadata, Mapping)
                else None
            )
            if relative_path in required:
                selected[str(relative_path)] = artifact_id
                if len(selected) == len(required):
                    break
        missing = [path for path in anchors if path not in selected]
        if missing:
            raise WorkstationError(
                f"Dependency {dependency.get('id')} is missing lineage anchor: {missing[0]}"
            )
        return [selected[path] for path in anchors]

    @contextmanager
    def _job_with_dependency_inputs(
        self, job: Mapping[str, Any]
    ) -> Iterator[Mapping[str, Any]]:
        """Materialize a verified package produced by the preceding workflow job."""

        job_type = str(job.get("type", ""))

        if job_type == "training.prepare":
            with self._job_with_dataset_scope(job, ("train",)) as runnable:
                yield runnable
            return

        if job_type == "checkpoint.select":
            with self._job_with_training_candidates(job) as candidates:
                with self._job_with_dataset_scope(
                    candidates, ("train", "validation")
                ) as runnable:
                    yield runnable
            return

        if job_type == "reference.select":
            with self._job_with_selected_checkpoints(job) as selected:
                with self._job_with_dataset_scope(selected, ("train",)) as runnable:
                    yield runnable
            return

        if job_type == "holdout.evaluate":
            with self._job_with_selected_checkpoints(job) as selected:
                with self._job_with_human_reference(selected) as referenced:
                    with self._job_with_dataset_scope(
                        referenced, ("train", "test")
                    ) as runnable:
                        yield runnable
            return

        if job_type == "engine.prepare":
            with self._job_with_training_checkpoints(job) as runnable:
                yield runnable
            return

        if job_type == "conversion.parity":
            with self._job_with_selected_checkpoints(job) as selected:
                with self._job_with_human_reference(selected) as referenced:
                    with self._job_with_package_dependency(
                        referenced,
                        expected_type="engine.prepare",
                        input_key="model_package",
                    ) as runnable:
                        yield runnable
            return

        dataset_dependency = {
            "dataset.separate": "dataset.target-speaker",
            "dataset.transcribe": "dataset.separate",
            "dataset.finalize": "dataset.transcribe",
        }.get(job_type)
        if dataset_dependency is not None:
            with self._job_with_dataset_artifacts(
                job, expected_type=dataset_dependency
            ) as runnable:
                yield runnable
            return

        contract = {
            "model.package": ("conversion.parity", "model_package"),
            "evaluation.prepare": ("model.package", "model_package"),
        }.get(job_type)
        if contract is None:
            yield job
            return

        expected_type, input_key = contract
        with self._job_with_package_dependency(
            job, expected_type=expected_type, input_key=input_key
        ) as runnable:
            yield runnable

    @contextmanager
    def _job_with_dataset_scope(
        self,
        job: Mapping[str, Any],
        visible_splits: Sequence[str],
    ) -> Iterator[Mapping[str, Any]]:
        project_id = job.get("project_id")
        if not isinstance(project_id, str):
            raise WorkstationError("Dataset-scoped jobs require a training project")
        project = self.store.get_project(project_id)
        config = project.get("config", {})
        parameters = job.get("parameters", {})
        if not isinstance(config, Mapping) or not isinstance(parameters, Mapping):
            raise WorkstationError("Training project configuration is malformed")
        dataset_value = parameters.get("dataset", config.get("dataset"))
        if not isinstance(dataset_value, str) or not dataset_value.strip():
            raise WorkstationError("Training bundle is not configured")
        source = Path(dataset_value).expanduser()
        if not source.is_absolute() or source.is_symlink():
            raise WorkstationError("Training bundle must be an absolute local directory")
        try:
            source = source.resolve(strict=True)
        except OSError as error:
            raise WorkstationError("Training bundle was not found") from error
        if not source.is_dir() or not any(
            source == root or root in source.parents for root in self.allowed_path_roots
        ):
            raise WorkstationError("Training bundle is outside the configured local roots")

        job_id = str(job.get("id", ""))
        staging = self.dependency_input_root / job_id / "dataset-scope"
        if staging.exists():
            shutil.rmtree(staging)
        try:
            materialize_worker_dataset_scope(
                source=source,
                destination=staging,
                visible_splits=visible_splits,
            )
            runnable = dict(job)
            runnable["parameters"] = {**dict(parameters), "dataset": str(staging)}
            yield runnable
        except DatasetScopeError as error:
            raise WorkstationError(str(error)) from error
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @contextmanager
    def _job_with_package_dependency(
        self,
        job: Mapping[str, Any],
        *,
        expected_type: str,
        input_key: str,
    ) -> Iterator[Mapping[str, Any]]:
        dependencies = job.get("depends_on", [])
        if isinstance(dependencies, (str, bytes)) or not isinstance(
            dependencies, Sequence
        ):
            raise WorkstationError("Job dependencies are malformed")
        matching: list[Mapping[str, Any]] = []
        for dependency_id in dependencies:
            dependency = self.store.get_job(str(dependency_id))
            if dependency.get("type") == expected_type:
                matching.append(dependency)
        if not matching:
            yield job
            return
        if len(matching) != 1:
            raise WorkstationError(
                f"{job.get('type')} requires exactly one {expected_type} dependency"
            )

        dependency = matching[0]
        if dependency.get("status") != "succeeded":
            raise WorkstationError(
                f"Dependency {dependency.get('id')} has not completed successfully"
            )
        result = dependency.get("result", {})
        artifact_ids = (
            result.get("registered_artifact_ids", [])
            if isinstance(result, Mapping)
            else []
        )
        if isinstance(artifact_ids, (str, bytes)) or not isinstance(
            artifact_ids, Sequence
        ) or not artifact_ids:
            raise WorkstationError(
                f"Dependency {dependency.get('id')} has no verified artifacts"
            )

        job_id = str(job.get("id", ""))
        dependency_id = str(dependency.get("id", ""))
        staging = self.dependency_input_root / job_id / dependency_id
        if staging.exists():
            shutil.rmtree(staging)
        package_root = staging / "model-package"
        package_root.mkdir(parents=True)
        copied = 0
        try:
            for artifact_id in artifact_ids:
                if not isinstance(artifact_id, str):
                    raise WorkstationError("Dependency artifact IDs are malformed")
                artifact = self.store.get_artifact(artifact_id)
                metadata = artifact.get("metadata", {})
                if (
                    artifact.get("status") != "ready"
                    or not isinstance(metadata, Mapping)
                    or metadata.get("job_id") != dependency_id
                ):
                    raise WorkstationError(
                        "Dependency artifact does not belong to the completed worker job"
                    )
                worker_path = validated_artifact_relative_path(
                    metadata.get("worker_relative_path"),
                    field="Dependency worker artifact path",
                )
                if not worker_path.parts or worker_path.parts[0] != "model-package":
                    continue
                relative = PurePosixPath(*worker_path.parts[1:])
                if not relative.parts:
                    continue
                source = self._verified_artifact_source(artifact)
                destination = package_root.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if _sha256_file(destination) != artifact.get("sha256"):
                    raise WorkstationError("Dependency artifact copy verification failed")
                copied += 1
            if copied == 0 or not (package_root / "manifest.json").is_file():
                raise WorkstationError(
                    f"Dependency {dependency_id} did not publish a complete model package"
                )
            parameters = job.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise WorkstationError("Job parameters are malformed")
            runnable = dict(job)
            runnable["parameters"] = {**dict(parameters), input_key: str(package_root)}
            yield runnable
        finally:
            shutil.rmtree(self.dependency_input_root / job_id, ignore_errors=True)

    @contextmanager
    def _job_with_human_reference(
        self, job: Mapping[str, Any]
    ) -> Iterator[Mapping[str, Any]]:
        project_id = job.get("project_id")
        if not isinstance(project_id, str):
            raise WorkstationError("Human reference jobs require a training project")
        project = self.store.get_project(project_id)
        config = project.get("config", {})
        if not isinstance(config, Mapping):
            raise WorkstationError("Training project config is malformed")
        job_id = str(job.get("id", ""))
        if job.get("type") == "holdout.evaluate":
            consumed_by = config.get("holdout_consumption_job_id")
            if isinstance(consumed_by, str) and consumed_by != job_id:
                raise WorkstationError(
                    "The test holdout was already consumed by this training run"
                )
            if consumed_by is None:
                project = self.store.update_project(
                    project_id,
                    config={
                        **dict(config),
                        "holdout_consumption_job_id": job_id,
                        "holdout_consumed_at": datetime.now(timezone.utc).isoformat(),
                        "test_split_accessed": True,
                    },
                )
                config = project["config"]
        artifact_id = config.get("deployment_reference_artifact_id")
        if not isinstance(artifact_id, str):
            raise WorkstationError("A human-locked deployment reference is required")
        artifact = self.store.get_artifact(artifact_id)
        if (
            artifact.get("type") != "reference"
            or artifact.get("status") != "ready"
            or artifact.get("project_id") != project_id
        ):
            raise WorkstationError("Deployment reference artifact is not ready")
        source_manifest = self._verified_artifact_source(artifact)
        try:
            reference = json.loads(source_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise WorkstationError("Deployment reference artifact is unreadable") from error
        if (
            not isinstance(reference, Mapping)
            or reference.get("schema")
            != "aniflive-tts-v2proplus-deployment-reference-v2"
            or reference.get("status") != "human-locked"
            or not isinstance(reference.get("item_id"), str)
            or not isinstance(reference.get("audio_sha256"), str)
        ):
            raise WorkstationError("Deployment reference artifact is unsupported")
        dataset_value = config.get("dataset")
        if not isinstance(dataset_value, str):
            raise WorkstationError("Training bundle is not configured")
        training_bundle = Path(dataset_value).expanduser().resolve(strict=True)
        train_manifest_path = training_bundle / "train" / "manifest.json"
        try:
            train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise WorkstationError("Training bundle manifest is unreadable") from error
        records = train_manifest.get("items") if isinstance(train_manifest, Mapping) else None
        if not isinstance(records, list):
            raise WorkstationError("Training bundle manifest is malformed")
        record = next(
            (
                row
                for row in records
                if isinstance(row, Mapping)
                and row.get("source_item_id") == reference["item_id"]
            ),
            None,
        )
        if record is None:
            raise WorkstationError("Deployment reference is not in the train split")
        relative = validated_artifact_relative_path(
            record.get("path"), field="Training reference path"
        )
        audio = training_bundle.joinpath(*relative.parts)
        try:
            resolved_audio = audio.resolve(strict=True)
            resolved_audio.relative_to(training_bundle)
        except (OSError, ValueError) as error:
            raise WorkstationError("Training reference path escaped its bundle") from error
        if (
            resolved_audio.is_symlink()
            or not resolved_audio.is_file()
            or _sha256_file(resolved_audio) != reference["audio_sha256"]
        ):
            raise WorkstationError("Deployment reference audio failed integrity validation")
        staging = self.dependency_input_root / job_id / "human-reference"
        staging.mkdir(parents=True, exist_ok=True)
        manifest_copy = staging / "deployment-reference.json"
        audio_copy = staging / "reference.wav"
        try:
            shutil.copy2(source_manifest, manifest_copy)
            shutil.copy2(resolved_audio, audio_copy)
            if (
                _sha256_file(manifest_copy) != artifact.get("sha256")
                or _sha256_file(audio_copy) != reference["audio_sha256"]
            ):
                raise WorkstationError("Deployment reference copy verification failed")
            parameters = job.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise WorkstationError("Job parameters are malformed")
            runnable = dict(job)
            runnable["parameters"] = {
                **dict(parameters),
                "deployment_reference": str(manifest_copy),
                "reference": str(audio_copy),
            }
            yield runnable
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @contextmanager
    def _job_with_training_candidates(
        self, job: Mapping[str, Any]
    ) -> Iterator[Mapping[str, Any]]:
        dependencies = job.get("depends_on", [])
        if isinstance(dependencies, (str, bytes)) or not isinstance(
            dependencies, Sequence
        ):
            raise WorkstationError("Job dependencies are malformed")
        matching: list[Mapping[str, Any]] = []
        for dependency_id in dependencies:
            dependency = self.store.get_job(str(dependency_id))
            if dependency.get("type") == "training.prepare":
                matching.append(dependency)
        if len(matching) != 1:
            raise WorkstationError(
                "checkpoint.select requires exactly one training.prepare dependency"
            )
        dependency = matching[0]
        dependency_id = str(dependency.get("id", ""))
        if dependency.get("status") != "succeeded":
            raise WorkstationError(
                f"Dependency {dependency_id} has not completed successfully"
            )
        result = dependency.get("result", {})
        artifact_ids = (
            result.get("registered_artifact_ids", [])
            if isinstance(result, Mapping)
            else []
        )
        if isinstance(artifact_ids, (str, bytes)) or not isinstance(
            artifact_ids, Sequence
        ):
            raise WorkstationError("Training dependency artifact lineage is malformed")
        job_id = str(job.get("id", ""))
        staging = self.dependency_input_root / job_id / dependency_id / "candidates"
        staging.mkdir(parents=True, exist_ok=True)
        copied = 0
        try:
            for artifact_id in artifact_ids:
                artifact = self.store.get_artifact(str(artifact_id))
                metadata = artifact.get("metadata", {})
                if (
                    artifact.get("status") != "ready"
                    or not isinstance(metadata, Mapping)
                    or metadata.get("job_id") != dependency_id
                ):
                    raise WorkstationError(
                        "Training candidate artifact does not belong to the dependency"
                    )
                relative = validated_artifact_relative_path(
                    metadata.get("worker_relative_path"),
                    field="Training candidate worker path",
                )
                relative_value = relative.as_posix()
                if not (
                    relative_value == "checkpoint-candidates.json"
                    or relative_value.startswith("checkpoints/")
                    or relative_value.startswith("selection-assets/")
                ):
                    continue
                source = self._verified_artifact_source(artifact)
                destination = staging.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if _sha256_file(destination) != artifact.get("sha256"):
                    raise WorkstationError("Training candidate copy verification failed")
                copied += 1
            if copied < 5 or not (staging / "checkpoint-candidates.json").is_file():
                raise WorkstationError(
                    "Training dependency did not publish complete checkpoint candidates"
                )
            parameters = job.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise WorkstationError("Job parameters are malformed")
            runnable = dict(job)
            runnable["parameters"] = {
                **dict(parameters),
                "checkpoint_candidates": str(staging),
            }
            yield runnable
        finally:
            shutil.rmtree(self.dependency_input_root / job_id, ignore_errors=True)

    @contextmanager
    def _job_with_selected_checkpoints(
        self, job: Mapping[str, Any]
    ) -> Iterator[Mapping[str, Any]]:
        dependencies = job.get("depends_on", [])
        if isinstance(dependencies, (str, bytes)) or not isinstance(
            dependencies, Sequence
        ):
            raise WorkstationError("Job dependencies are malformed")
        matching: list[Mapping[str, Any]] = []
        for dependency_id in dependencies:
            dependency = self.store.get_job(str(dependency_id))
            if dependency.get("type") == "checkpoint.select":
                matching.append(dependency)
        if len(matching) != 1:
            raise WorkstationError(
                f"{job.get('type')} requires exactly one checkpoint.select dependency"
            )
        dependency = matching[0]
        dependency_id = str(dependency.get("id", ""))
        if dependency.get("status") != "succeeded":
            raise WorkstationError(
                f"Dependency {dependency_id} has not completed successfully"
            )
        result = dependency.get("result", {})
        artifact_ids = (
            result.get("registered_artifact_ids", [])
            if isinstance(result, Mapping)
            else []
        )
        if isinstance(artifact_ids, (str, bytes)) or not isinstance(
            artifact_ids, Sequence
        ):
            raise WorkstationError("Checkpoint selection artifacts are malformed")
        job_id = str(job.get("id", ""))
        staging = self.dependency_input_root / job_id / dependency_id / "selection"
        staging.mkdir(parents=True, exist_ok=True)
        try:
            copied = 0
            for artifact_id in artifact_ids:
                artifact = self.store.get_artifact(str(artifact_id))
                metadata = artifact.get("metadata", {})
                if (
                    artifact.get("status") != "ready"
                    or not isinstance(metadata, Mapping)
                    or metadata.get("job_id") != dependency_id
                ):
                    raise WorkstationError(
                        "Checkpoint selection artifact does not belong to the dependency"
                    )
                relative = validated_artifact_relative_path(
                    metadata.get("worker_relative_path"),
                    field="Checkpoint selection worker path",
                )
                value = relative.as_posix()
                if not (
                    value == "checkpoint-selection-report.json"
                    or value.startswith("selected/")
                    or value.startswith("selection-assets/")
                ):
                    continue
                source = self._verified_artifact_source(artifact)
                destination = staging.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if _sha256_file(destination) != artifact.get("sha256"):
                    raise WorkstationError("Checkpoint selection copy verification failed")
                copied += 1
            if (
                copied < 5
                or not (staging / "selected" / "deployment-checkpoints.json").is_file()
            ):
                raise WorkstationError(
                    "Checkpoint selection dependency did not publish a locked winner"
                )
            parameters = job.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise WorkstationError("Job parameters are malformed")
            runnable = dict(job)
            runnable["parameters"] = {
                **dict(parameters),
                "selected_checkpoints": str(staging),
            }
            yield runnable
        finally:
            shutil.rmtree(self.dependency_input_root / job_id, ignore_errors=True)

    @contextmanager
    def _job_with_dataset_artifacts(
        self,
        job: Mapping[str, Any],
        *,
        expected_type: str,
    ) -> Iterator[Mapping[str, Any]]:
        """Rebuild one verified dataset artifact tree for the next worker stage."""

        dependencies = job.get("depends_on", [])
        if isinstance(dependencies, (str, bytes)) or not isinstance(
            dependencies, Sequence
        ):
            raise WorkstationError("Job dependencies are malformed")
        matching: list[Mapping[str, Any]] = []
        for dependency_id in dependencies:
            dependency = self.store.get_job(str(dependency_id))
            if dependency.get("type") == expected_type:
                matching.append(dependency)
        if not matching:
            yield job
            return
        if len(matching) != 1:
            raise WorkstationError(
                f"{job.get('type')} requires exactly one {expected_type} dependency"
            )

        dependency = matching[0]
        dependency_id = str(dependency.get("id", ""))
        if dependency.get("project_id") != job.get("project_id"):
            raise WorkstationError("Dataset workflow dependencies must use one project")
        if dependency.get("status") != "succeeded":
            raise WorkstationError(
                f"Dependency {dependency_id} has not completed successfully"
            )
        result = dependency.get("result", {})
        artifact_ids = (
            result.get("registered_artifact_ids", [])
            if isinstance(result, Mapping)
            else []
        )
        if (
            isinstance(artifact_ids, (str, bytes))
            or not isinstance(artifact_ids, Sequence)
            or not artifact_ids
        ):
            raise WorkstationError(
                f"Dependency {dependency_id} has no verified artifacts"
            )

        job_id = str(job.get("id", ""))
        staging = self.dependency_input_root / job_id / dependency_id
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        copied = 0
        try:
            for artifact_id in artifact_ids:
                if not isinstance(artifact_id, str):
                    raise WorkstationError("Dependency artifact IDs are malformed")
                artifact = self.store.get_artifact(artifact_id)
                metadata = artifact.get("metadata", {})
                if (
                    artifact.get("type") != "dataset"
                    or artifact.get("status") != "ready"
                    or artifact.get("project_id") != job.get("project_id")
                    or not isinstance(metadata, Mapping)
                    or metadata.get("job_id") != dependency_id
                ):
                    raise WorkstationError(
                        "Dependency artifact does not belong to the completed dataset job"
                    )
                relative = validated_artifact_relative_path(
                    metadata.get("worker_relative_path"),
                    field="Dependency worker artifact path",
                )
                source = self._verified_artifact_source(artifact)
                destination = staging.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if _sha256_file(destination) != artifact.get("sha256"):
                    raise WorkstationError(
                        "Dependency dataset artifact copy verification failed"
                    )
                copied += 1
            if copied == 0:
                raise WorkstationError(
                    f"Dependency {dependency_id} published no dataset artifacts"
                )
            parameters = job.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise WorkstationError("Job parameters are malformed")
            runnable = dict(job)
            runnable["parameters"] = {
                **dict(parameters),
                "dataset": str(staging),
            }
            yield runnable
        finally:
            shutil.rmtree(self.dependency_input_root / job_id, ignore_errors=True)

    @contextmanager
    def _job_with_training_checkpoints(
        self, job: Mapping[str, Any]
    ) -> Iterator[Mapping[str, Any]]:
        dependencies = job.get("depends_on", [])
        if isinstance(dependencies, (str, bytes)) or not isinstance(
            dependencies, Sequence
        ):
            raise WorkstationError("Job dependencies are malformed")
        matching: list[Mapping[str, Any]] = []
        holdout_jobs: list[Mapping[str, Any]] = []
        for dependency_id in dependencies:
            dependency = self.store.get_job(str(dependency_id))
            if dependency.get("type") == "checkpoint.select":
                matching.append(dependency)
            elif dependency.get("type") == "holdout.evaluate":
                holdout_jobs.append(dependency)
        if not matching:
            yield job
            return
        if len(matching) != 1:
            raise WorkstationError(
                "engine.prepare requires exactly one checkpoint.select dependency"
            )
        if len(holdout_jobs) != 1:
            raise WorkstationError(
                "engine.prepare requires exactly one holdout.evaluate dependency"
            )
        holdout = holdout_jobs[0]
        holdout_result = holdout.get("result", {})
        holdout_evidence = (
            _worker_evidence(holdout_result)
            if isinstance(holdout_result, Mapping)
            else {}
        )
        if (
            holdout.get("status") != "succeeded"
            or holdout_evidence.get("schema") != "aniflive-tts-holdout-evaluation-v1"
            or holdout_evidence.get("status") != "passed"
            or holdout_evidence.get("winner_locked_before_test") is not True
            or holdout_evidence.get("test_consumed_once") is not True
        ):
            raise WorkstationError("engine.prepare is blocked by holdout evaluation")

        dependency = matching[0]
        dependency_id = str(dependency.get("id", ""))
        if dependency.get("status") != "succeeded":
            raise WorkstationError(
                f"Dependency {dependency_id} has not completed successfully"
            )
        result = dependency.get("result", {})
        artifact_ids = (
            result.get("registered_artifact_ids", [])
            if isinstance(result, Mapping)
            else []
        )
        if isinstance(artifact_ids, (str, bytes)) or not isinstance(
            artifact_ids, Sequence
        ) or not artifact_ids:
            raise WorkstationError(
                f"Dependency {dependency_id} has no verified artifacts"
            )

        artifacts_by_worker_path: dict[str, Mapping[str, Any]] = {}
        for artifact_id in artifact_ids:
            if not isinstance(artifact_id, str):
                raise WorkstationError("Dependency artifact IDs are malformed")
            artifact = self.store.get_artifact(artifact_id)
            metadata = artifact.get("metadata", {})
            if (
                artifact.get("status") != "ready"
                or not isinstance(metadata, Mapping)
                or metadata.get("job_id") != dependency_id
            ):
                raise WorkstationError(
                    "Dependency artifact does not belong to the completed worker job"
                )
            worker_path = validated_artifact_relative_path(
                metadata.get("worker_relative_path"),
                field="Dependency worker artifact path",
            ).as_posix()
            if worker_path in artifacts_by_worker_path:
                raise WorkstationError("Dependency published duplicate worker artifact paths")
            normalized_path = (
                worker_path.removeprefix("selected/")
                if worker_path.startswith("selected/")
                else worker_path
            )
            artifacts_by_worker_path[normalized_path] = artifact

        manifest_artifact = artifacts_by_worker_path.get("deployment-checkpoints.json")
        if manifest_artifact is None:
            raise WorkstationError(
                "Checkpoint selection did not publish deployment-checkpoints.json"
            )
        manifest_source = self._verified_artifact_source(manifest_artifact)
        try:
            deployment = json.loads(manifest_source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise WorkstationError("Training deployment manifest is unreadable") from error
        if (
            not isinstance(deployment, Mapping)
            or deployment.get("schema")
            != "aniflive-tts-v2proplus-deployment-checkpoints-v2"
            or deployment.get("model_family") != "gsv-v2proplus"
            or not isinstance(deployment.get("selection"), Mapping)
            or deployment["selection"].get("test_split_accessed") is not False
        ):
            raise WorkstationError("Training deployment manifest is unsupported")

        selected: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
        contracts = {
            "gpt": ("checkpoints/gpt/", ".ckpt"),
            "sovits": ("checkpoints/sovits/", ".pth"),
        }
        for role, (prefix, suffix) in contracts.items():
            record = deployment.get(role)
            if not isinstance(record, Mapping):
                raise WorkstationError(f"Training deployment manifest has no {role} record")
            relative = validated_artifact_relative_path(
                record.get("relative_path"), field=f"Deployment {role} checkpoint path"
            ).as_posix()
            if not relative.startswith(prefix) or not relative.lower().endswith(suffix):
                raise WorkstationError(f"Deployment {role} checkpoint path is invalid")
            artifact = artifacts_by_worker_path.get(relative)
            if artifact is None:
                raise WorkstationError(f"Deployment {role} checkpoint artifact is missing")
            artifact_metadata = artifact.get("metadata", {})
            if (
                not isinstance(artifact_metadata, Mapping)
                or record.get("sha256") != artifact.get("sha256")
                or record.get("size_bytes") != artifact_metadata.get("size_bytes")
            ):
                raise WorkstationError(
                    f"Deployment {role} checkpoint metadata does not match its artifact"
                )
            selected[role] = (record, artifact)

        job_id = str(job.get("id", ""))
        staging = self.dependency_input_root / job_id / dependency_id
        checkpoint_root = staging / "checkpoint"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(manifest_source, checkpoint_root / "deployment-checkpoints.json")
            for record, artifact in selected.values():
                relative = PurePosixPath(str(record["relative_path"]))
                source = self._verified_artifact_source(artifact)
                destination = checkpoint_root.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if _sha256_file(destination) != record["sha256"]:
                    raise WorkstationError("Dependency checkpoint copy verification failed")
            parameters = job.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise WorkstationError("Job parameters are malformed")
            runnable = dict(job)
            runnable["parameters"] = {
                **dict(parameters),
                "checkpoint": str(checkpoint_root),
            }
            yield runnable
        finally:
            shutil.rmtree(self.dependency_input_root / job_id, ignore_errors=True)

    def _verified_artifact_source(self, artifact: Mapping[str, Any]) -> Path:
        local_path = validated_artifact_relative_path(
            artifact.get("local_path"), field="Dependency artifact local path"
        )
        root = self.store.artifact_root.resolve(strict=True)
        candidate = root.joinpath(*local_path.parts)
        current = root
        for part in local_path.parts:
            current = current / part
            try:
                info = current.lstat()
            except OSError as error:
                raise WorkstationError("Dependency artifact was not found") from error
            if current.is_symlink() or bool(
                getattr(info, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
            ):
                raise WorkstationError(
                    "Dependency artifacts cannot contain symbolic links or reparse points"
                )
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as error:
            raise WorkstationError("Dependency artifact escaped the artifact store") from error
        if not resolved.is_file():
            raise WorkstationError("Dependency artifact is not a regular file")
        before = resolved.stat()
        digest = _sha256_file(resolved)
        after = resolved.stat()
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise WorkstationError("Dependency artifact changed while it was read")
        if digest != artifact.get("sha256"):
            raise WorkstationError("Dependency artifact checksum no longer matches")
        return resolved

    def _reconcile_broker_if_due(self) -> None:
        self.reconcile_broker()

    def reconcile_broker(self, *, force: bool = False):
        """Recover expired leases and remove stale managed containers."""

        if self.worker_broker is None:
            return None
        now = time.monotonic()
        if not force and now < self._next_broker_reconciliation_at:
            return None
        expired_gpu_job_ids = self.store.expired_gpu_job_ids()
        expired_gpu = set(expired_gpu_job_ids)
        active_job_ids = tuple(
            str(job["id"])
            for job in self.store.list_jobs()
            if job.get("status") == "running" and str(job["id"]) not in expired_gpu
        )
        report = self.worker_broker.reconcile(active_job_ids)
        confirm_absence = getattr(self.worker_broker, "confirmed_absent_jobs", None)
        confirmed_ids = (
            confirm_absence(expired_gpu_job_ids) if confirm_absence is not None
            else expired_gpu_job_ids
        )
        self.store.recover_expired_jobs(
            confirmed_gpu_job_ids=confirmed_ids
        )
        self._next_broker_reconciliation_at = (
            now + _BROKER_RECONCILIATION_INTERVAL_SECONDS
        )
        for completed in self.store.list_jobs():
            self._queue_dataset_retry_dependents(completed)
        self.reporter(
            {
                "kind": "docker-reconciliation",
                "retained": len(report.retained),
                "removed": len(report.removed),
            }
        )
        return report

    def _queue_dataset_retry_dependents(self, completed: Mapping[str, Any]) -> None:
        stages = {"dataset.decode", "dataset.target-speaker", "dataset.separate", "dataset.transcribe", "dataset.finalize"}
        original_id = completed.get("retry_of")
        if (
            completed.get("status") != "succeeded" or completed.get("type") not in stages
            or not original_id or not isinstance(completed.get("project_id"), str)
        ):
            return
        project = self.store.get_project(str(completed["project_id"]))
        parameters = completed.get("parameters", {})
        if parameters.get("acquisition_mode", project.get("config", {}).get("acquisition_mode")) != "target-speaker":
            return
        token = None
        try:
            token = self.store.acquire_resource_lease(
                "graph:" + str(completed["id"]), purpose="dataset-retry-followup", lease_seconds=30,
            )
            jobs = self.store.list_jobs()
            for child in jobs:
                if (
                    child.get("project_id") != completed["project_id"]
                    or child.get("type") not in stages or child.get("status") != "failed"
                    or child.get("error") != f"Dependency {original_id} did not succeed"
                    or original_id not in child.get("depends_on", [])
                ):
                    continue
                dependencies = [
                    completed["id"] if value == original_id else value
                    for value in child["depends_on"]
                ]
                if any(
                    job.get("retry_of") == child["id"] and job.get("depends_on") == dependencies
                    for job in jobs
                ):
                    continue
                if any(
                    self.store.get_job(value)["status"] in {"failed", "cancelled"}
                    for value in dependencies
                ):
                    continue
                retried = self.store.create_job(
                    job_type=child["type"], project_id=child["project_id"],
                    parameters=child["parameters"], depends_on=dependencies,
                    priority=int(child.get("priority", 0)), retry_of=child["id"],
                    attempt=int(child.get("attempt", 1)) + 1,
                )
                jobs.append(retried)
                self.store.append_job_log(
                    child["id"], "info", f"Dependency retry succeeded; continuation queued as {retried['id']}",
                )
        except WorkstationError as error:
            if "busy" not in str(error).lower():
                self.store.append_job_log(
                    str(completed["id"]), "error", f"Dataset retry continuation needs attention: {_safe_error(error)}",
                )
        finally:
            if token is not None:
                self.store.release_resource_lease(token)

    def _claim_next(self):
        jobs = sorted(
            self.store.list_jobs(),
            key=lambda record: (
                -int(record.get("priority", 0)),
                str(record.get("created_at", "")),
                str(record.get("id", "")),
            ),
        )
        for job in jobs:
            if self.stopping:
                return None
            if job.get("status") != "queued" or job.get("type") not in self.enabled_job_types:
                continue
            try:
                if job.get("type") == "evaluation.prepare":
                    dependencies = [self.store.get_job(value) for value in job.get("depends_on", [])]
                    if all(dependency["status"] == "succeeded" for dependency in dependencies):
                        from .evaluation_preflight import preflight_workstation_evaluation
                        try:
                            preflight_workstation_evaluation(
                                self.store, job.get("project_id"), job.get("parameters", {}),
                                allowed_roots=self.allowed_path_roots,
                            )
                        except WorkstationError as error:
                            # A queued legacy/automatic job must fail before speech is drained.
                            self.store.update_job(
                                str(job["id"]), status="failed",
                                error="Evaluation preflight failed: " + _safe_error(error),
                            )
                            continue
                if self.runtime_handoff is not None and job.get("resource_class") == "gpu-exclusive":
                    dependencies = [self.store.get_job(value) for value in job.get("depends_on", [])]
                    if all(dependency["status"] == "succeeded" for dependency in dependencies):
                        if not self.runtime_handoff.prepare(job):
                            continue
                return self.store.claim_job(str(job["id"]))
            except WorkstationError as error:
                # Another worker, an incomplete dependency or a resource lease can
                # legitimately make a listed queued job unclaimable.
                job_id = str(job["id"])
                gpu_busy = (
                    job.get("resource_class") == "gpu-exclusive"
                    and "exclusive gpu resource is busy" in str(error).lower()
                )
                if gpu_busy:
                    if job.get("wait_reason") != GPU_WAIT_REASON:
                        waiting = self.store.set_job_wait_reason(job_id, GPU_WAIT_REASON)
                        if waiting.get("wait_reason") == GPU_WAIT_REASON:
                            self.reporter(
                                {
                                    "kind": "job-wait",
                                    "job_id": job_id,
                                    "job_type": job.get("type"),
                                    "reason": GPU_WAIT_REASON,
                                }
                            )
                elif job.get("wait_reason"):
                    self.store.set_job_wait_reason(job_id, None)
                continue
        return None

    def _finish_result(
        self,
        job_id: str,
        job_type: str,
        token: str,
        result: AdapterResult,
    ) -> WorkerRunResult:
        expected = ADAPTER_REGISTRY[job_type]
        if result.job_type != job_type or result.resource_class != expected.resource_class:
            return self._finish_failed(
                job_id,
                job_type,
                token,
                WorkstationError("Adapter result does not match the claimed job contract"),
            )
        try:
            payload = _json_mapping(result.payload)
        except Exception as error:
            return self._finish_failed(job_id, job_type, token, error)
        evidence = _worker_evidence(payload)
        if result.outcome == "checkpoint-paused":
            if job_type != "training.prepare" or evidence.get("status") != "paused":
                return self._finish_failed(job_id, job_type, token,
                                           WorkstationError("Invalid checkpoint pause result"))
            record = self.store.update_job(
                job_id, status="paused", result=payload, claim_token=token,
            )
            return WorkerRunResult("paused", job_id, job_type, record["status"])
        if result.outcome == "completed":
            if payload.get("execution_performed") is not True:
                return self._finish_failed(
                    job_id,
                    job_type,
                    token,
                    WorkstationError(
                        "Adapter reported completion without verified execution"
                    ),
                    result=payload,
                )
            if (
                job_type
                in {"checkpoint.select", "holdout.evaluate", "conversion.parity"}
                and evidence.get("status") != "passed"
            ):
                diagnostic_payload = dict(payload)
                registered_ids = diagnostic_payload.pop(
                    "registered_artifact_ids", []
                )
                if registered_ids:
                    diagnostic_payload["diagnostic_artifact_ids"] = registered_ids
                final_decision_id = self._record_quality_gate_no_go(
                    job_id=job_id,
                    job_type=job_type,
                    claim_token=token,
                    evidence=evidence,
                    diagnostic_artifact_ids=registered_ids,
                )
                if final_decision_id is not None:
                    diagnostic_payload["final_decision_artifact_id"] = (
                        final_decision_id
                    )
                return self._finish_failed(
                    job_id,
                    job_type,
                    token,
                    WorkstationError(
                        f"{job_type} quality gate failed; downstream production is blocked"
                    ),
                    result=diagnostic_payload,
                )
            try:
                record = self.store.update_job(
                    job_id,
                    status="succeeded",
                    progress=1.0,
                    result=payload,
                    claim_token=token,
                )
            except Exception as error:
                return self._finish_failed(job_id, job_type, token, error)
            if job_type == "training.prepare" and record["status"] == "succeeded":
                self._queue_automatic_production_build(record)
            elif job_type == "checkpoint.select" and record["status"] == "succeeded":
                self._queue_reference_selection(record)
            elif job_type == "reference.select" and record["status"] == "succeeded":
                self._mark_reference_pending_human_evidence(record, payload)
            elif (
                job_type == "holdout.evaluate"
                and record["status"] == "succeeded"
                and evidence.get("status") == "passed"
            ):
                self._queue_engine_build(record)
            elif job_type == "engine.prepare" and record["status"] == "succeeded":
                self._queue_conversion_parity(record)
            elif (
                job_type == "conversion.parity"
                and record["status"] == "succeeded"
                and evidence.get("status") == "passed"
            ):
                self._queue_model_package(record)
            elif job_type == "model.package" and record["status"] == "succeeded":
                self._queue_canonical_evaluation(record)
            if record["status"] == "succeeded":
                self._queue_dataset_retry_dependents(record)
            disposition = (
                "cancelled"
                if record["status"] == "cancelled"
                else "paused"
                if record["status"] == "paused"
                else "completed"
            )
            return WorkerRunResult(disposition, job_id, job_type, record["status"])
        if result.outcome == "prepared-only":
            missing = payload.get("missing_inputs", [])
            if payload.get("readiness") == "ready":
                error = "Preparation is blocked: execution was not performed"
            else:
                details = ", ".join(str(value) for value in missing) or "trusted backend"
                error = f"Preparation is blocked: missing {details}"
            try:
                record = self.store.update_job(
                    job_id,
                    status="failed",
                    result=payload,
                    error=error,
                    claim_token=token,
                )
            except Exception as update_error:
                return self._finish_failed(job_id, job_type, token, update_error)
            return WorkerRunResult("blocked", job_id, job_type, record["status"], error)
        return self._finish_failed(
            job_id,
            job_type,
            token,
            WorkstationError(f"Unsupported adapter outcome: {result.outcome}"),
        )

    def _record_quality_gate_no_go(
        self,
        *,
        job_id: str,
        job_type: str,
        claim_token: str,
        evidence: Mapping[str, Any],
        diagnostic_artifact_ids: Sequence[Any],
    ) -> str | None:
        try:
            job = self.store.get_job(job_id)
            project_id = job.get("project_id")
            if not isinstance(project_id, str):
                raise WorkstationError("Quality-gate job has no project")
            project = self.store.get_project(project_id)
            parent_ids = tuple(
                dict.fromkeys(str(value) for value in diagnostic_artifact_ids)
            )
            for artifact_id in parent_ids:
                self.store.get_artifact(artifact_id)
            failures = evidence.get("failures", [])
            if isinstance(failures, (str, bytes)) or not isinstance(
                failures, Sequence
            ):
                failures = []
            decision = {
                "schema": "aniflive-tts-production-final-decision-v1",
                "decision": "NO-GO",
                "project_id": project_id,
                "project_name": project.get("name"),
                "failed_gate": job_type,
                "job_id": job_id,
                "failures": [str(value) for value in failures],
                "diagnostic_artifact_ids": list(parent_ids),
                "test_consumed_once": evidence.get("test_consumed_once") is True,
                "downstream_production_blocked": True,
            }
            destination = (
                self.store.artifact_root
                / "production-decisions"
                / project_id
                / f"{job_type.replace('.', '-')}-{job_id}-no-go.json"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
            artifact = self.store.register_artifact(
                artifact_type="evaluation",
                name=f"{project['name']} {job_type} NO-GO decision",
                project_id=project_id,
                status="ready",
                local_path=destination,
                metadata={
                    "decision": "NO-GO",
                    "failed_gate": job_type,
                    "job_id": job_id,
                },
                parent_artifact_ids=parent_ids,
            )
            config = project.get("config", {})
            if not isinstance(config, Mapping):
                raise WorkstationError("Training project config is malformed")
            self.store.update_project(
                project_id,
                config={
                    **dict(config),
                    "production_status": f"no-go-{job_type}",
                    "final_decision": "NO-GO",
                    "final_decision_artifact_id": artifact["id"],
                    "final_decision_sha256": artifact["sha256"],
                },
            )
            self.store.append_job_log(
                job_id,
                "error",
                f"NO-GO decision recorded: {artifact['id']}",
                claim_token=claim_token,
            )
            return str(artifact["id"])
        except (OSError, WorkstationError) as error:
            self.store.append_job_log(
                job_id,
                "error",
                f"NO-GO decision could not be recorded: {_safe_error(error)}",
                claim_token=claim_token,
            )
            return None

    def _queue_automatic_production_build(self, training_job: Mapping[str, Any]) -> None:
        project_id = training_job.get("project_id")
        training_job_id = training_job.get("id")
        if not isinstance(project_id, str) or not isinstance(training_job_id, str):
            return
        project = self.store.get_project(project_id)
        config = project.get("config", {})
        if not isinstance(config, Mapping) or config.get("auto_build_production") is not True:
            return
        jobs = self.store.list_jobs()
        existing = next(
            (
                job
                for job in jobs
                if job.get("type") == "checkpoint.select"
                and training_job_id in job.get("depends_on", [])
            ),
            None,
        )
        if existing is not None:
            return
        try:
            selection = self.store.create_job(
                job_type="checkpoint.select",
                project_id=project_id,
                depends_on=(training_job_id,),
                priority=int(training_job.get("priority", 0)),
            )
            self.store.append_job_log(
                training_job_id,
                "info",
                f"Validation checkpoint selection queued: {selection['id']}",
            )
        except WorkstationError as error:
            self.store.append_job_log(
                training_job_id,
                "error",
                f"Production build could not be queued: {_safe_error(error)}",
            )

    def _queue_reference_selection(self, checkpoint_job: Mapping[str, Any]) -> None:
        project_id = checkpoint_job.get("project_id")
        checkpoint_job_id = checkpoint_job.get("id")
        if not isinstance(project_id, str) or not isinstance(checkpoint_job_id, str):
            return
        project = self.store.get_project(project_id)
        config = project.get("config", {})
        if not isinstance(config, Mapping) or config.get("auto_build_production") is not True:
            return
        existing = next(
            (
                job
                for job in self.store.list_jobs()
                if job.get("type") == "reference.select"
                and checkpoint_job_id in job.get("depends_on", [])
            ),
            None,
        )
        if existing is not None:
            return
        try:
            reference = self.store.create_job(
                job_type="reference.select",
                project_id=project_id,
                depends_on=(checkpoint_job_id,),
                priority=int(checkpoint_job.get("priority", 0)),
            )
            self.store.append_job_log(
                checkpoint_job_id,
                "info",
                f"Representative reference sweep queued: {reference['id']}",
            )
        except WorkstationError as error:
            self.store.append_job_log(
                checkpoint_job_id,
                "error",
                f"Reference sweep could not be queued: {_safe_error(error)}",
            )

    def _mark_reference_pending_human_evidence(
        self,
        reference_job: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> None:
        project_id = reference_job.get("project_id")
        reference_job_id = reference_job.get("id")
        if not isinstance(project_id, str) or not isinstance(reference_job_id, str):
            return
        evidence = _worker_evidence(payload)
        if evidence.get("status") != "blocked-pending-human-evidence":
            return
        try:
            project = self.store.get_project(project_id)
            config = project.get("config", {})
            if not isinstance(config, Mapping):
                raise WorkstationError("Training project config is malformed")
            if config.get("reference_status") in {"human-locked", "all-poor"}:
                return
            candidate_count = evidence.get("candidate_count")
            next_config = {
                **dict(config),
                "reference_status": "pending-human-evidence",
                "reference_selection_job_id": reference_job_id,
            }
            if isinstance(candidate_count, int) and not isinstance(
                candidate_count, bool
            ):
                next_config["reference_candidate_count"] = candidate_count
            self.store.update_project(project_id, config=next_config)
            self.store.append_job_log(
                reference_job_id,
                "info",
                "Blind reference evidence is ready for human listening",
            )
        except WorkstationError as error:
            self.store.append_job_log(
                reference_job_id,
                "error",
                f"Reference review state could not be recorded: {_safe_error(error)}",
            )

    def _queue_engine_build(self, holdout_job: Mapping[str, Any]) -> None:
        project_id = holdout_job.get("project_id")
        holdout_job_id = holdout_job.get("id")
        if not isinstance(project_id, str) or not isinstance(holdout_job_id, str):
            return
        checkpoint_ids = [
            str(dependency_id)
            for dependency_id in holdout_job.get("depends_on", [])
            if self.store.get_job(str(dependency_id)).get("type") == "checkpoint.select"
        ]
        if len(checkpoint_ids) != 1:
            self.store.append_job_log(
                holdout_job_id,
                "error",
                "Engine build was not queued: locked checkpoint dependency is missing",
            )
            return
        if any(
            job.get("type") == "engine.prepare"
            and holdout_job_id in job.get("depends_on", [])
            for job in self.store.list_jobs()
        ):
            return
        try:
            engine = self.store.create_job(
                job_type="engine.prepare",
                project_id=project_id,
                depends_on=(checkpoint_ids[0], holdout_job_id),
                priority=int(holdout_job.get("priority", 0)),
            )
            self.store.append_job_log(
                holdout_job_id,
                "info",
                f"TensorRT engine build queued after passed holdout: {engine['id']}",
            )
        except WorkstationError as error:
            self.store.append_job_log(
                holdout_job_id,
                "error",
                f"Engine build could not be queued: {_safe_error(error)}",
            )

    def _queue_conversion_parity(self, engine_job: Mapping[str, Any]) -> None:
        project_id = engine_job.get("project_id")
        engine_job_id = engine_job.get("id")
        if not isinstance(project_id, str) or not isinstance(engine_job_id, str):
            return
        checkpoint_ids = [
            str(value)
            for value in engine_job.get("depends_on", [])
            if self.store.get_job(str(value)).get("type") == "checkpoint.select"
        ]
        holdout_ids = [
            str(value)
            for value in engine_job.get("depends_on", [])
            if self.store.get_job(str(value)).get("type") == "holdout.evaluate"
        ]
        if len(checkpoint_ids) != 1 or len(holdout_ids) != 1:
            self.store.append_job_log(
                engine_job_id,
                "error",
                "Conversion parity was not queued: locked selection or holdout evidence is missing",
            )
            return
        if any(
            job.get("type") == "conversion.parity"
            and engine_job_id in job.get("depends_on", [])
            for job in self.store.list_jobs()
        ):
            return
        try:
            parity = self.store.create_job(
                job_type="conversion.parity",
                project_id=project_id,
                depends_on=(engine_job_id, checkpoint_ids[0], holdout_ids[0]),
                priority=int(engine_job.get("priority", 0)),
            )
            self.store.append_job_log(
                engine_job_id,
                "info",
                f"PyTorch to ONNX to TensorRT parity queued: {parity['id']}",
            )
        except WorkstationError as error:
            self.store.append_job_log(
                engine_job_id,
                "error",
                f"Conversion parity could not be queued: {_safe_error(error)}",
            )

    def _queue_model_package(self, parity_job: Mapping[str, Any]) -> None:
        project_id = parity_job.get("project_id")
        parity_job_id = parity_job.get("id")
        if not isinstance(project_id, str) or not isinstance(parity_job_id, str):
            return
        if any(
            job.get("type") == "model.package"
            and parity_job_id in job.get("depends_on", [])
            for job in self.store.list_jobs()
        ):
            return
        try:
            package = self.store.create_job(
                job_type="model.package",
                project_id=project_id,
                depends_on=(parity_job_id,),
                priority=int(parity_job.get("priority", 0)),
            )
            self.store.append_job_log(
                parity_job_id,
                "info",
                f"Validated production package queued: {package['id']}",
            )
        except WorkstationError as error:
            self.store.append_job_log(
                parity_job_id,
                "error",
                f"Model package could not be queued: {_safe_error(error)}",
            )

    def _queue_canonical_evaluation(self, package_job: Mapping[str, Any]) -> None:
        training_project_id = package_job.get("project_id")
        package_job_id = package_job.get("id")
        if not isinstance(training_project_id, str) or not isinstance(package_job_id, str):
            return
        training = self.store.get_project(training_project_id)
        config = training.get("config", {})
        if not isinstance(config, Mapping):
            return
        existing = next(
            (
                project
                for project in self.store.list_projects(kind="evaluation")
                if isinstance(project.get("config"), Mapping)
                and project["config"].get("source_package_job_id") == package_job_id
            ),
            None,
        )
        if existing is not None:
            return
        evaluation_config = {
            "source_training_project_id": training_project_id,
            "source_package_job_id": package_job_id,
            "shared_dir": config.get("shared_dir"),
            "asr_model": config.get("asr_model"),
            "baseline_report": config.get("evaluation_baseline"),
            "comparison_status": ("pending" if config.get("evaluation_baseline") else "unavailable"),
            "benchmark_language": config.get("reference_language", "ja"),
            "qualification_status": "pending",
        }
        missing = [
            key
            for key in ("shared_dir", "asr_model")
            if not isinstance(evaluation_config.get(key), str)
            or not str(evaluation_config[key]).strip()
        ]
        if missing:
            evaluation_config["qualification_status"] = "blocked"
            evaluation_config["blocked_reason"] = (
                "Canonical evaluation requires " + ", ".join(missing)
            )
        evaluation = self.store.create_project(
            kind="evaluation",
            name=f"{training.get('name', 'Voice')} Evaluation",
            config=evaluation_config,
        )
        if missing:
            self.store.append_job_log(
                package_job_id,
                "warning",
                "Package built; canonical evaluation is blocked by missing "
                + ", ".join(missing),
            )
            return
        evaluation_job = self.store.create_job(
            job_type="evaluation.prepare",
            project_id=evaluation["id"],
            depends_on=(package_job_id,),
            priority=int(package_job.get("priority", 0)),
        )
        self.store.append_job_log(
            package_job_id,
            "info",
            f"Canonical evaluation queued: {evaluation_job['id']}",
        )

    def _finish_cancelled(
        self, job_id: str, job_type: str, token: str, error: Exception
    ) -> WorkerRunResult:
        message = _safe_error(error)
        try:
            record = self.store.update_job(
                job_id, status="cancelled", error=message, claim_token=token
            )
            return WorkerRunResult("cancelled", job_id, job_type, record["status"], message)
        except WorkstationError as finish_error:
            return WorkerRunResult(
                "failed", job_id, job_type, "running", _safe_error(finish_error)
            )

    def _finish_paused(
        self, job_id: str, job_type: str, token: str
    ) -> WorkerRunResult:
        try:
            record = self.store.update_job(
                job_id, status="paused", claim_token=token
            )
            return WorkerRunResult("paused", job_id, job_type, record["status"])
        except WorkstationError as finish_error:
            return WorkerRunResult(
                "failed", job_id, job_type, "running", _safe_error(finish_error)
            )

    def _finish_failed(
        self,
        job_id: str,
        job_type: str,
        token: str,
        error: Exception,
        *,
        result: Mapping[str, Any] | None = None,
    ) -> WorkerRunResult:
        message = _safe_error(error)
        try:
            record = self.store.update_job(
                job_id,
                status="failed",
                result=result,
                error=message,
                claim_token=token,
            )
            return WorkerRunResult("failed", job_id, job_type, record["status"], message)
        except WorkstationError as finish_error:
            return WorkerRunResult(
                "failed", job_id, job_type, "running", _safe_error(finish_error)
            )


def _safe_error(error: BaseException) -> str:
    message = " ".join(str(error).strip().split()) or error.__class__.__name__
    if len(message) <= 1000:
        return message
    marker = " ... [middle omitted] ... "
    head_length = 240
    tail_length = 1000 - head_length - len(marker)
    return message[:head_length] + marker + message[-tail_length:]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_root(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = candidate.absolute()
    try:
        info = candidate.lstat()
    except OSError as error:
        raise WorkstationError(f"Allowed local path root was not found: {candidate}") from error
    if candidate.is_symlink() or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
    ):
        raise WorkstationError(
            "Allowed local path roots cannot be symbolic links or reparse points"
        )
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir():
        raise WorkstationError(f"Allowed local path root was not found: {resolved}")
    return resolved


def _safe_heartbeat_interval(lease_expires_at: str) -> float:
    try:
        expiry = datetime.fromisoformat(lease_expires_at.replace("Z", "+00:00"))
        remaining = (expiry - datetime.now(timezone.utc)).total_seconds()
    except (AttributeError, TypeError, ValueError) as error:
        raise WorkstationError("Claim returned a malformed lease expiry") from error
    if remaining <= 0:
        raise WorkstationError("Claim lease expired before the worker started")
    return max(0.05, remaining / 3.0)


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(dict(value), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise WorkstationError("Adapter result is not JSON-compatible") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise WorkstationError("Adapter result must be a JSON object")
    return decoded


def _worker_evidence(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    backend = payload.get("backend")
    if isinstance(backend, Mapping):
        nested = backend.get("payload")
        if isinstance(nested, Mapping):
            return nested
    return payload


__all__ = ["WorkerRunResult", "WorkerSummary", "WorkstationWorker"]
