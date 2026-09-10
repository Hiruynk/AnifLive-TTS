from __future__ import annotations

from pathlib import Path
import importlib
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from aniflive_tts.expression import ExpressionSegment
from aniflive_tts.inference_lease import (
    FAIL_STOP_EXIT_CODE,
    GPUResourceBusy,
    GPUResourceLeaseError,
    InferenceGPUResourceLease,
)
from aniflive_tts.workstation import GPU_RESOURCE_KEY, WorkstationError, WorkstationStore


ROOT = Path(__file__).resolve().parents[1]


def _service_module():
    return importlib.import_module("aniflive_tts.service")


def _options(text: str = "今日はいい天気ですね。") -> Any:
    module = _service_module()
    return module.SynthesisOptions(
        text=text,
        text_language="ja",
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        speed=1.0,
        pause_length=0.3,
        noise_scale=0.35,
        cut_punc="default",
        seed=7,
    )


class _SuccessfulStreamer:
    last_profile: dict[str, float] = {}

    @staticmethod
    def complete_wav_chunk_length() -> int:
        return 32

    @staticmethod
    def streaming_chunk_length() -> int:
        return 16

    @staticmethod
    def iter_audio(**_kwargs):
        yield np.full(320, 0.1, dtype=np.float32)

    @staticmethod
    def keepwarm_pulse() -> float:
        return 0.001


class _FailingStreamer(_SuccessfulStreamer):
    @staticmethod
    def iter_audio(**_kwargs):
        raise RuntimeError("synthetic TensorRT failure")
        yield  # pragma: no cover


class _CancellableStreamer(_SuccessfulStreamer):
    started = threading.Event()

    @classmethod
    def iter_audio(cls, **kwargs):
        cancelled = kwargs["cancelled"]
        cls.started.set()
        yield np.full(320, 0.1, dtype=np.float32)
        while not cancelled.wait(0.01):
            pass


def _service(
    tmp_path: Path, streamer: object
) -> tuple[Any, WorkstationStore]:
    module = _service_module()
    settings = module.RuntimeSettings(
        source_dir=tmp_path,
        engine_dir=tmp_path,
        bert_path=tmp_path,
        reference_wav=tmp_path / "reference.wav",
        profile_manifest=tmp_path / "profile.json",
    )
    service = module.TensorRTService(settings)
    store = WorkstationStore(tmp_path / "workstation")
    service._gpu_resource_lease = InferenceGPUResourceLease(
        store,
        owner_id="inference-test",
        lease_seconds=2,
        heartbeat_seconds=0.05,
    )
    service._engine = object()
    service._streamer = streamer
    service._sample_rate = 32000
    service._warm_retention = None
    service._segment_plan = lambda options: [  # type: ignore[method-assign]
        ExpressionSegment(text=options.text, enabled=False)
    ]
    service._conditioning = lambda _options: None  # type: ignore[method-assign]
    return service, store


def test_inference_lease_heartbeats_and_releases_shared_gpu(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="training", name="Training")
    job = store.create_job(job_type="training.prepare", project_id=project["id"])
    heartbeat_observed = threading.Event()
    original_heartbeat = store.heartbeat_resource_lease

    def heartbeat(token: str, *, lease_seconds: int | None = None) -> str:
        result = original_heartbeat(token, lease_seconds=lease_seconds)
        heartbeat_observed.set()
        return result

    store.heartbeat_resource_lease = heartbeat  # type: ignore[method-assign]
    lease = InferenceGPUResourceLease(
        store,
        owner_id="inference-test",
        lease_seconds=2,
        heartbeat_seconds=0.05,
    )

    with lease.hold("inference:test") as handle:
        assert handle.enabled is True
        assert heartbeat_observed.wait(1.0)
        with pytest.raises(WorkstationError, match="busy"):
            store.claim_job(job["id"])

    claim = store.claim_job(job["id"])
    store.update_job(job["id"], status="cancelled", claim_token=claim.token)
    assert lease.status()["active"] is False


def test_inference_lease_rejects_existing_worker_reservation(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    worker_token = store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="job:test")
    lease = InferenceGPUResourceLease(store, owner_id="inference-test")
    try:
        with pytest.raises(GPUResourceBusy, match="workstation job"):
            lease.acquire("inference:test")
    finally:
        store.release_resource_lease(worker_token)


def test_inference_lease_reports_heartbeat_loss_and_still_releases(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    heartbeat_failed = threading.Event()

    def fail_heartbeat(_token: str, *, lease_seconds: int | None = None) -> str:
        del lease_seconds
        heartbeat_failed.set()
        raise WorkstationError("database became unavailable")

    store.heartbeat_resource_lease = fail_heartbeat  # type: ignore[method-assign]
    lease = InferenceGPUResourceLease(
        store,
        owner_id="inference-test",
        lease_seconds=2,
        heartbeat_seconds=0.05,
        heartbeat_attempts=1,
    )
    handle = lease.acquire("inference:test")
    assert heartbeat_failed.wait(1.0)
    with pytest.raises(GPUResourceLeaseError, match="was lost"):
        handle.ensure_valid()
    handle.close()

    worker_token = store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="job:test")
    store.release_resource_lease(worker_token)


def test_inference_lease_recovers_from_a_transient_heartbeat_failure(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    heartbeat_recovered = threading.Event()
    fail_stop_called = threading.Event()
    original_heartbeat = store.heartbeat_resource_lease
    attempts = 0

    def transient_heartbeat(token: str, *, lease_seconds: int | None = None) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise WorkstationError("synthetic transient database failure")
        result = original_heartbeat(token, lease_seconds=lease_seconds)
        heartbeat_recovered.set()
        return result

    store.heartbeat_resource_lease = transient_heartbeat  # type: ignore[method-assign]
    lease = InferenceGPUResourceLease(
        store,
        owner_id="inference-test",
        lease_seconds=2,
        heartbeat_seconds=0.05,
        heartbeat_attempts=3,
        fail_stop=lambda _error: fail_stop_called.set(),
    )
    handle = lease.acquire("inference:engine-residency")
    assert heartbeat_recovered.wait(1.0)
    assert fail_stop_called.is_set() is False
    assert lease.status()["lease_healthy"] is True
    handle.ensure_valid()
    handle.close()


def test_production_lease_invokes_fail_stop_on_heartbeat_loss(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    heartbeat_failed = threading.Event()
    fail_stop_called = threading.Event()
    failures: list[BaseException] = []

    def fail_heartbeat(_token: str, *, lease_seconds: int | None = None) -> str:
        del lease_seconds
        heartbeat_failed.set()
        raise WorkstationError("synthetic heartbeat failure")

    def fail_stop(error: BaseException) -> None:
        failures.append(error)
        fail_stop_called.set()

    store.heartbeat_resource_lease = fail_heartbeat  # type: ignore[method-assign]
    lease = InferenceGPUResourceLease(
        store,
        owner_id="inference-test",
        lease_seconds=2,
        heartbeat_seconds=0.05,
        fail_stop=fail_stop,
    )
    handle = lease.acquire("inference:engine-residency")
    assert heartbeat_failed.wait(1.0)
    assert fail_stop_called.wait(1.0)
    assert isinstance(failures[0], WorkstationError)
    assert lease.status()["lease_healthy"] is False
    assert lease.status()["fail_stop"] is True
    handle.close()


def test_default_fail_stop_terminates_a_child_process() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from aniflive_tts.inference_lease import _fail_stop_process; "
                "_fail_stop_process(RuntimeError('synthetic lost lease'))"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == FAIL_STOP_EXIT_CODE


def test_environment_configures_fail_stop_and_safe_heartbeat_margin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANIFLIVE_TTS_WORKSTATION_DIR", str(tmp_path / "workstation"))
    lease = InferenceGPUResourceLease.from_env()
    assert lease.status()["enabled"] is True
    assert lease.status()["fail_stop"] is True

    monkeypatch.setenv("ANIFLIVE_TTS_INFERENCE_GPU_LEASE_SECONDS", "119")
    with pytest.raises(ValueError, match="between 120 and 86400"):
        InferenceGPUResourceLease.from_env()

    monkeypatch.setenv("ANIFLIVE_TTS_INFERENCE_GPU_LEASE_SECONDS", "120")
    monkeypatch.setenv("ANIFLIVE_TTS_INFERENCE_GPU_HEARTBEAT_SECONDS", "31")
    with pytest.raises(ValueError, match="between 0.01 and 30.0"):
        InferenceGPUResourceLease.from_env()


def test_nested_acquire_rejects_a_lost_lease_without_incrementing_references(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    heartbeat_failed = threading.Event()

    def fail_heartbeat(_token: str, *, lease_seconds: int | None = None) -> str:
        del lease_seconds
        heartbeat_failed.set()
        raise WorkstationError("synthetic heartbeat failure")

    store.heartbeat_resource_lease = fail_heartbeat  # type: ignore[method-assign]
    lease = InferenceGPUResourceLease(
        store,
        owner_id="inference-test",
        lease_seconds=2,
        heartbeat_seconds=0.05,
        heartbeat_attempts=1,
    )
    handle = lease.acquire("inference:outer")
    assert heartbeat_failed.wait(1.0)
    with pytest.raises(GPUResourceLeaseError, match="was lost"):
        lease.acquire("inference:nested")
    assert lease.status()["references"] == 1
    assert lease.status()["purpose"] == "inference:outer"
    handle.close()


def test_process_owned_lease_is_reference_counted_across_inference_threads(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    lease = InferenceGPUResourceLease(
        store,
        owner_id="inference-test",
        lease_seconds=2,
        heartbeat_seconds=0.05,
    )
    resident = lease.acquire("inference:residency:test")
    worker_finished = threading.Event()
    errors: list[BaseException] = []

    def request_thread() -> None:
        try:
            with lease.hold("inference:request") as request:
                request.ensure_valid()
        except BaseException as error:  # pragma: no cover - assertion below
            errors.append(error)
        finally:
            worker_finished.set()

    thread = threading.Thread(target=request_thread)
    thread.start()
    assert worker_finished.wait(1.0)
    thread.join(timeout=1.0)
    assert errors == []
    assert lease.status()["active"] is True
    assert lease.status()["references"] == 1
    with pytest.raises(WorkstationError, match="busy"):
        store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="job:test")
    resident.close()

    worker_token = store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="job:test")
    store.release_resource_lease(worker_token)


def test_complete_synthesis_releases_lease_after_success_and_error(tmp_path: Path) -> None:
    service, store = _service(tmp_path, _SuccessfulStreamer())
    result = service.synthesize(_options())
    assert result.output_samples == 320
    assert service._gpu_resource_lease.status()["active"] is False

    service._streamer = _FailingStreamer()
    with pytest.raises(RuntimeError, match="synthetic TensorRT failure"):
        service.synthesize(_options())

    worker_token = store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="job:test")
    store.release_resource_lease(worker_token)
    assert service._active_requests == 0


def test_request_conflict_does_not_consume_service_slot(tmp_path: Path) -> None:
    module = _service_module()
    service, store = _service(tmp_path, _SuccessfulStreamer())
    worker_token = store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="job:test")
    try:
        with pytest.raises(module.ServiceBusy, match="workstation job"):
            service.synthesize(_options())
        assert service._active_requests == 0
    finally:
        store.release_resource_lease(worker_token)

    assert service.synthesize(_options()).output_samples == 320


def test_unload_is_gpu_leased_and_conflict_preserves_loaded_pipeline(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path, _SuccessfulStreamer())
    service.unload()
    worker_token = store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="job:test")
    store.release_resource_lease(worker_token)
    assert service.ready is False

    service._engine = object()
    service._streamer = _SuccessfulStreamer()
    worker_token = store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="job:test")
    try:
        with pytest.raises(RuntimeError, match="reserved by an AnifLive-TTS workstation job"):
            service.unload()
        assert service._engine is not None
        assert service.ready is False
    finally:
        store.release_resource_lease(worker_token)


def test_resident_inference_assets_block_gpu_workers_until_unload(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path, _SuccessfulStreamer())
    service._resident_gpu_lease = service._gpu_resource_lease.acquire(
        "inference:residency:test"
    )
    assert service.ready is True
    assert service.health()["gpu_resource_lease"]["residency_reserved"] is True

    with pytest.raises(WorkstationError, match="busy"):
        store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="job:test")

    result: list[float] = []
    thread = threading.Thread(
        target=lambda: result.append(service._keepwarm_pulse_with_gpu_lease())
    )
    thread.start()
    thread.join(timeout=1.0)
    assert thread.is_alive() is False
    assert result == [0.001]
    assert service._gpu_resource_lease.status()["references"] == 1

    service.unload()
    worker_token = store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="job:test")
    store.release_resource_lease(worker_token)
    assert service.health()["gpu_resource_lease"]["residency_reserved"] is False


def test_cuda_probe_happens_only_after_startup_lease_is_acquired(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _service_module()
    service, _store = _service(tmp_path, _SuccessfulStreamer())
    service._engine = None
    service._streamer = None
    acquired = False
    closed = False

    class Handle:
        @staticmethod
        def ensure_valid() -> None:
            return None

        def close(self) -> None:
            nonlocal closed
            closed = True

    def acquire(_purpose: str, *, busy_is_request_error: bool):
        nonlocal acquired
        assert busy_is_request_error is False
        acquired = True
        return Handle()

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            assert acquired is True
            return False

    monkeypatch.setattr(service, "_validate_layout", lambda: None)
    monkeypatch.setattr(service, "_acquire_gpu_resource", acquire)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=Cuda()))
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace())

    with pytest.raises(module.TensorRTRuntimeError, match="No CUDA GPU"):
        service.load()
    assert acquired is True
    assert closed is True


def test_health_uses_gpu_metadata_cached_under_startup_lease(tmp_path: Path) -> None:
    service, _store = _service(tmp_path, _SuccessfulStreamer())

    class Cuda:
        def __getattr__(self, _name: str):
            raise AssertionError("health must not touch CUDA")

    service._torch = SimpleNamespace(cuda=Cuda())
    service._gpu_metadata = {
        "name": "Test GPU",
        "compute_capability": "12.0",
        "vram_bytes": 16 * 1024**3,
    }
    assert service.health()["gpu"] == service._gpu_metadata


def test_model_switch_conflict_keeps_previous_model_and_clears_switching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _service_module()

    class BusyService:
        ready = True

        @staticmethod
        def _is_busy() -> bool:
            return False

        @staticmethod
        def unload() -> None:
            raise module.TensorRTRuntimeError("GPU worker owns the lease")

    previous = tmp_path / "old-v2proplus"
    requested = tmp_path / "new-v2proplus"
    monkeypatch.setenv("ANIFLIVE_TTS_MODEL_PACKAGE", str(previous))
    monkeypatch.setattr(module, "MODEL_ID", "old-v2proplus")
    monkeypatch.setattr(module, "VOICE_ID", "default")
    manager = module.RuntimeServiceManager(BusyService())
    monkeypatch.setattr(
        manager,
        "_discover_packages",
        lambda: {
            "old-v2proplus": {"path": previous, "manifest": {}},
            "new-v2proplus": {"path": requested, "manifest": {}},
        },
    )
    monkeypatch.setattr(
        manager,
        "_load_package",
        lambda *_args: pytest.fail("a failed unload must not load another package"),
    )

    with pytest.raises(module.TensorRTRuntimeError, match="previous model remains active"):
        manager.activate("new-v2proplus")
    assert manager.switching is False
    assert manager.ready is True
    assert manager._service.__class__ is BusyService


def test_model_switch_retains_process_lease_without_worker_claim_gap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _service_module()
    previous, store = _service(tmp_path / "old", _SuccessfulStreamer())
    previous._resident_gpu_lease = previous._gpu_resource_lease.acquire(
        "inference:engine-residency"
    )
    manager = module.RuntimeServiceManager(previous)
    old_package = tmp_path / "old-v2proplus"
    new_package = tmp_path / "new-v2proplus"
    monkeypatch.setenv("ANIFLIVE_TTS_MODEL_PACKAGE", str(old_package))
    manager._package_dir = old_package
    monkeypatch.setattr(module, "MODEL_ID", "old-v2proplus")
    monkeypatch.setattr(module, "VOICE_ID", "default")
    monkeypatch.setattr(
        manager,
        "_discover_packages",
        lambda: {
            "old-v2proplus": {"path": old_package, "manifest": {}},
            "new-v2proplus": {"path": new_package, "manifest": {}},
        },
    )

    def replacement(_path: Path, _voice: str):
        with pytest.raises(WorkstationError, match="busy"):
            store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="job:race")
        service, _replacement_store = _service(
            tmp_path / "replacement", _SuccessfulStreamer()
        )
        service._gpu_resource_lease = previous._gpu_resource_lease
        service._resident_gpu_lease = service._gpu_resource_lease.acquire(
            "inference:engine-residency"
        )
        monkeypatch.setattr(module, "MODEL_ID", "new-v2proplus")
        return service

    monkeypatch.setattr(manager, "_load_package", replacement)
    result = manager.activate("new-v2proplus")
    assert result["changed"] is True
    assert manager.ready is True
    with pytest.raises(WorkstationError, match="busy"):
        store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="job:after-switch")
    manager.unload()
    token = store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="job:after-unload")
    store.release_resource_lease(token)


def test_stream_cancellation_releases_lease_and_rejects_second_request(
    tmp_path: Path,
) -> None:
    module = _service_module()
    _CancellableStreamer.started.clear()
    service, store = _service(tmp_path, _CancellableStreamer())
    pcm = service.stream_pcm(_options())
    assert _CancellableStreamer.started.wait(1.0)
    assert next(pcm)

    with pytest.raises(module.ServiceBusy, match="already processing"):
        service.stream_pcm(_options("別の要求です。"))

    assert service.cancel_active_stream() is True
    assert list(pcm) == []
    worker_token = store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="job:test")
    store.release_resource_lease(worker_token)
    assert service._active_requests == 0
    assert service._gpu_resource_lease.status()["active"] is False


def test_stream_cancellation_releases_request_reference_but_keeps_residency(
    tmp_path: Path,
) -> None:
    _CancellableStreamer.started.clear()
    service, store = _service(tmp_path, _CancellableStreamer())
    service._resident_gpu_lease = service._gpu_resource_lease.acquire(
        "inference:engine-residency"
    )
    pcm = service.stream_pcm(_options())
    assert _CancellableStreamer.started.wait(1.0)
    assert next(pcm)
    assert service.cancel_active_stream() is True
    assert list(pcm) == []
    assert service._gpu_resource_lease.status()["active"] is True
    assert service._gpu_resource_lease.status()["references"] == 1
    with pytest.raises(WorkstationError, match="busy"):
        store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="job:test")
    service.unload()
    token = store.acquire_resource_lease(GPU_RESOURCE_KEY, purpose="job:test")
    store.release_resource_lease(token)


def test_default_webui_and_compose_share_one_workstation_database() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    launcher = (ROOT / "run_studio.bat").read_text(encoding="utf-8")

    assert "ANIFLIVE_TTS_WORKSTATION_DIR: /data/workstation" in compose
    assert (
        "${ANIFLIVE_TTS_WORKSTATION_HOST_DIR:-./data/workstation}:/data/workstation"
        in compose
    )
    assert "ANIFLIVE_TTS_WORKSTATION_DIR=/data/workstation" in dockerfile
    assert (
        'ANIFLIVE_TTS_WORKSTATION_DIR=%CD%\\data\\workstation' in launcher
    )
    assert "aniflive-tts:1.4.0-cu128" in compose
    assert "aniflive-tts:1.3.0-cu128" not in compose
    assert "name: aniflive-tts-v14-workstation" in compose
    assert "container_name: aniflive-tts-v14-workstation-api" in compose
    assert 'org.opencontainers.image.version="1.4.0"' in dockerfile


def test_inference_lease_does_not_import_worker_or_training_stacks() -> None:
    lease_source = (ROOT / "src/aniflive_tts/inference_lease.py").read_text(
        encoding="utf-8"
    )
    service_source = (ROOT / "src/aniflive_tts/service.py").read_text(
        encoding="utf-8"
    )
    forbidden = ("workstation_adapters", "workstation_worker", "torch.nn")
    assert all(name not in lease_source for name in forbidden)
    assert all(name not in service_source for name in forbidden)
