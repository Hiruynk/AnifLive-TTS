"""Shared GPU resource lease used by the TensorRT inference process.

The workstation scheduler and inference service coordinate through the same
SQLite-backed ``gpu:0`` lease.  This module deliberately imports only the
standard-library workstation store; training and worker adapters never enter
the serving process.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import socket
import threading
from typing import Callable, Iterator, Protocol

from .workstation import GPU_RESOURCE_KEY, WorkstationError, WorkstationStore


LOGGER = logging.getLogger("aniflive_tts.inference_lease")
FAIL_STOP_EXIT_CODE = 70


def _fail_stop_process(error: BaseException) -> None:
    """Terminate before an expired lease can overlap a live CUDA context."""

    LOGGER.critical(
        "The gpu:0 residency lease was lost; terminating AnifLive-TTS so the "
        "operating system releases every CUDA allocation",
        exc_info=(type(error), error, error.__traceback__),
    )
    os._exit(FAIL_STOP_EXIT_CODE)


class GPUResourceBusy(RuntimeError):
    """Another process currently owns the exclusive GPU resource."""


class GPUResourceLeaseError(RuntimeError):
    """The shared inference lease could not be maintained."""


class _ResourceLeaseStore(Protocol):
    def acquire_resource_lease(
        self,
        resource_key: str,
        *,
        purpose: str,
        owner_id: str | None = None,
        lease_seconds: int | None = None,
    ) -> str: ...

    def heartbeat_resource_lease(
        self, token: str, *, lease_seconds: int | None = None
    ) -> str: ...

    def release_resource_lease(self, token: str) -> None: ...


class _LeaseState:
    def __init__(
        self,
        store: _ResourceLeaseStore,
        token: str,
        *,
        purpose: str,
        lease_seconds: int,
        heartbeat_seconds: float,
        heartbeat_attempts: int,
        fail_stop: Callable[[BaseException], None] | None,
    ) -> None:
        self.store = store
        self.token = token
        self.purpose = purpose
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.heartbeat_attempts = heartbeat_attempts
        self.fail_stop = fail_stop
        self.stop = threading.Event()
        self.error_lock = threading.Lock()
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._heartbeat,
            name="aniflive-tts-inference-gpu-lease",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def _heartbeat(self) -> None:
        while not self.stop.wait(self.heartbeat_seconds):
            last_error: BaseException | None = None
            for attempt in range(self.heartbeat_attempts):
                try:
                    self.store.heartbeat_resource_lease(
                        self.token, lease_seconds=self.lease_seconds
                    )
                    if attempt:
                        LOGGER.info(
                            "The gpu:0 lease heartbeat recovered after %d transient failure(s)",
                            attempt,
                        )
                    last_error = None
                    break
                except BaseException as error:
                    last_error = error
                    if attempt + 1 >= self.heartbeat_attempts:
                        break
                    retry_delay = min(
                        self.heartbeat_seconds,
                        0.1 * (2**attempt),
                    )
                    if self.stop.wait(retry_delay):
                        return
            if last_error is not None:
                with self.error_lock:
                    if self.error is None:
                        self.error = last_error
                self.stop.set()
                if self.fail_stop is not None:
                    self.fail_stop(last_error)
                return

    def failure(self) -> BaseException | None:
        with self.error_lock:
            return self.error


class InferenceGPUResourceLeaseHandle:
    """One idempotent reference to a process-owned GPU lease."""

    def __init__(
        self,
        manager: "InferenceGPUResourceLease",
        state: _LeaseState | None,
    ) -> None:
        self._manager = manager
        self._state = state
        self._closed = False

    @property
    def enabled(self) -> bool:
        return self._state is not None

    @property
    def failure(self) -> BaseException | None:
        return self._state.failure() if self._state is not None else None

    def ensure_valid(self) -> None:
        error = self.failure
        if error is not None:
            raise GPUResourceLeaseError(
                "The inference GPU resource lease was lost during TensorRT execution"
            ) from error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._manager._release(self._state)

    def __enter__(self) -> "InferenceGPUResourceLeaseHandle":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


class InferenceGPUResourceLease:
    """Acquire, heartbeat and release the workstation's exclusive GPU lease.

    Nested acquisition inside the inference process is reference counted.
    Startup, request workers and warm-retention use different threads, while
    TensorRT execution remains serialized by the service's own admission and
    inference locks.  The first reference represents GPU-resident TensorRT
    assets and is held until those assets are unloaded.
    """

    def __init__(
        self,
        store: _ResourceLeaseStore | None,
        *,
        owner_id: str | None = None,
        lease_seconds: int = 120,
        heartbeat_seconds: float = 15.0,
        heartbeat_attempts: int = 3,
        fail_stop: Callable[[BaseException], None] | None = None,
    ) -> None:
        if not isinstance(lease_seconds, int) or not 1 <= lease_seconds <= 86400:
            raise ValueError("lease_seconds must be between 1 and 86400")
        if not 0.01 <= float(heartbeat_seconds) < float(lease_seconds):
            raise ValueError("heartbeat_seconds must be positive and shorter than the lease")
        if not isinstance(heartbeat_attempts, int) or not 1 <= heartbeat_attempts <= 8:
            raise ValueError("heartbeat_attempts must be between 1 and 8")
        self._store = store
        self._owner_id = owner_id or f"inference:{socket.gethostname()}:{os.getpid()}"
        self._lease_seconds = lease_seconds
        self._heartbeat_seconds = float(heartbeat_seconds)
        self._heartbeat_attempts = heartbeat_attempts
        self._fail_stop = fail_stop
        self._lock = threading.Lock()
        self._state: _LeaseState | None = None
        self._references = 0

    @classmethod
    def from_env(cls) -> "InferenceGPUResourceLease":
        configured = os.environ.get("ANIFLIVE_TTS_WORKSTATION_DIR")
        if not configured:
            return cls(None)
        lease_seconds = _integer_env(
            "ANIFLIVE_TTS_INFERENCE_GPU_LEASE_SECONDS",
            120,
            minimum=120,
            maximum=86400,
        )
        heartbeat_default = min(15.0, max(0.25, lease_seconds / 3.0))
        heartbeat_seconds = _float_env(
            "ANIFLIVE_TTS_INFERENCE_GPU_HEARTBEAT_SECONDS",
            heartbeat_default,
            minimum=0.01,
            maximum=min(30.0, float(lease_seconds) / 4.0),
        )
        heartbeat_attempts = _integer_env(
            "ANIFLIVE_TTS_INFERENCE_GPU_HEARTBEAT_ATTEMPTS",
            3,
            minimum=1,
            maximum=8,
        )
        return cls(
            WorkstationStore(Path(configured).expanduser().resolve()),
            lease_seconds=lease_seconds,
            heartbeat_seconds=heartbeat_seconds,
            heartbeat_attempts=heartbeat_attempts,
            fail_stop=_fail_stop_process,
        )

    @property
    def enabled(self) -> bool:
        return self._store is not None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._state is not None

    def status(self) -> dict[str, object]:
        with self._lock:
            state = self._state
            references = self._references
        return {
            "enabled": self.enabled,
            "mode": "shared-workstation" if self.enabled else "disabled",
            "resource": GPU_RESOURCE_KEY,
            "active": state is not None,
            "lease_healthy": state is None or state.failure() is None,
            "references": references,
            "purpose": state.purpose if state is not None else None,
            "fail_stop": self._fail_stop is not None,
        }

    def acquire(self, purpose: str) -> InferenceGPUResourceLeaseHandle:
        if self._store is None:
            return InferenceGPUResourceLeaseHandle(self, None)
        with self._lock:
            if self._state is not None:
                failure = self._state.failure()
                if failure is not None:
                    raise GPUResourceLeaseError(
                        "The inference GPU resource lease was lost during TensorRT execution"
                    ) from failure
                self._references += 1
                return InferenceGPUResourceLeaseHandle(self, self._state)
            try:
                token = self._store.acquire_resource_lease(
                    GPU_RESOURCE_KEY,
                    purpose=purpose,
                    owner_id=self._owner_id,
                    lease_seconds=self._lease_seconds,
                )
            except WorkstationError as error:
                if "busy" in str(error).lower():
                    raise GPUResourceBusy(
                        "The exclusive GPU resource is reserved by a workstation job"
                    ) from error
                raise GPUResourceLeaseError(
                    "The inference GPU resource lease could not be acquired"
                ) from error
            except BaseException as error:
                raise GPUResourceLeaseError(
                    "The inference GPU resource lease could not be acquired"
                ) from error
            state = _LeaseState(
                self._store,
                token,
                purpose=purpose,
                lease_seconds=self._lease_seconds,
                heartbeat_seconds=self._heartbeat_seconds,
                heartbeat_attempts=self._heartbeat_attempts,
                fail_stop=self._fail_stop,
            )
            self._state = state
            self._references = 1
            try:
                state.start()
            except BaseException:
                self._state = None
                self._references = 0
                try:
                    self._store.release_resource_lease(token)
                finally:
                    raise
            return InferenceGPUResourceLeaseHandle(self, state)

    @contextmanager
    def hold(self, purpose: str) -> Iterator[InferenceGPUResourceLeaseHandle]:
        handle = self.acquire(purpose)
        try:
            yield handle
            handle.ensure_valid()
        finally:
            handle.close()

    def _release(self, state: _LeaseState | None) -> None:
        if state is None:
            return
        release = False
        with self._lock:
            if state is not self._state or self._references <= 0:
                return
            self._references -= 1
            if self._references == 0:
                self._state = None
                release = True
        if not release:
            return
        state.stop.set()
        state.thread.join(timeout=max(2.0, state.heartbeat_seconds + 1.0))
        heartbeat_error: BaseException | None = None
        if state.thread.is_alive():
            heartbeat_error = GPUResourceLeaseError(
                "The inference GPU lease heartbeat did not stop"
            )
        release_error: BaseException | None = None
        try:
            state.store.release_resource_lease(state.token)
        except BaseException as error:
            release_error = GPUResourceLeaseError(
                "The inference GPU resource lease could not be released"
            )
            release_error.__cause__ = error
        if release_error is not None:
            raise release_error
        if heartbeat_error is not None:
            raise heartbeat_error


def _integer_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


__all__ = [
    "GPUResourceBusy",
    "GPUResourceLeaseError",
    "InferenceGPUResourceLease",
    "InferenceGPUResourceLeaseHandle",
]
