"""Bounded GPU warm-retention controls for the TensorRT serving process."""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable


LOGGER = logging.getLogger("aniflive_tts.runtime_control")


@dataclass(frozen=True)
class GpuTelemetrySample:
    temperature_c: int
    utilization_percent: int
    memory_utilization_percent: int
    power_watts: float | None = None


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class NvmlTelemetry:
    """Read only the NVML values needed by the warm-retention safety gate."""

    def __init__(self, device_index: int = 0) -> None:
        self._library: ctypes.CDLL | None = None
        self._device = ctypes.c_void_p()
        self._initialized = False
        self._load(device_index)

    @property
    def available(self) -> bool:
        return self._initialized

    def _load(self, device_index: int) -> None:
        candidates = (
            ("nvml.dll",) if os.name == "nt" else ("libnvidia-ml.so.1", "libnvidia-ml.so")
        )
        for candidate in candidates:
            try:
                self._library = ctypes.CDLL(candidate)
                break
            except OSError:
                continue
        if self._library is None:
            return

        init = getattr(self._library, "nvmlInit_v2", None) or getattr(
            self._library, "nvmlInit", None
        )
        handle = getattr(self._library, "nvmlDeviceGetHandleByIndex_v2", None) or getattr(
            self._library, "nvmlDeviceGetHandleByIndex", None
        )
        if init is None or handle is None:
            self._library = None
            return
        init.restype = ctypes.c_int
        handle.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        handle.restype = ctypes.c_int
        if init() != 0 or handle(device_index, ctypes.byref(self._device)) != 0:
            self.close()
            return
        self._initialized = True

    def sample(self) -> GpuTelemetrySample | None:
        if not self._initialized or self._library is None:
            return None
        temperature = ctypes.c_uint()
        utilization = _NvmlUtilization()
        get_temperature = self._library.nvmlDeviceGetTemperature
        get_temperature.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
        get_temperature.restype = ctypes.c_int
        get_utilization = self._library.nvmlDeviceGetUtilizationRates
        get_utilization.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlUtilization)]
        get_utilization.restype = ctypes.c_int
        if get_temperature(self._device, 0, ctypes.byref(temperature)) != 0:
            return None
        if get_utilization(self._device, ctypes.byref(utilization)) != 0:
            return None

        power_watts: float | None = None
        get_power = getattr(self._library, "nvmlDeviceGetPowerUsage", None)
        if get_power is not None:
            milliwatts = ctypes.c_uint()
            get_power.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
            get_power.restype = ctypes.c_int
            if get_power(self._device, ctypes.byref(milliwatts)) == 0:
                power_watts = float(milliwatts.value) / 1000.0
        return GpuTelemetrySample(
            temperature_c=int(temperature.value),
            utilization_percent=int(utilization.gpu),
            memory_utilization_percent=int(utilization.memory),
            power_watts=power_watts,
        )

    def close(self) -> None:
        if self._library is not None and self._initialized:
            shutdown = getattr(self._library, "nvmlShutdown", None)
            if shutdown is not None:
                try:
                    shutdown()
                except Exception:
                    LOGGER.debug("NVML shutdown failed", exc_info=True)
        self._initialized = False
        self._library = None


class WarmRetentionController:
    """Replay bounded TensorRT work briefly after a real synthesis request."""

    def __init__(
        self,
        *,
        pulse: Callable[[], float],
        inference_lock: threading.RLock,
        is_busy: Callable[[], bool],
        telemetry: NvmlTelemetry | None = None,
        retention_seconds: float = 25.0,
        pulse_interval_seconds: float = 6.0,
        maximum_temperature_c: int = 70,
        resume_temperature_c: int = 65,
        maximum_utilization_percent: int = 20,
        maximum_pulse_seconds: float = 0.020,
    ) -> None:
        self._pulse = pulse
        self._inference_lock = inference_lock
        self._is_busy = is_busy
        self._telemetry = telemetry or NvmlTelemetry()
        self._retention_seconds = max(0.0, float(retention_seconds))
        self._pulse_interval_seconds = max(1.0, float(pulse_interval_seconds))
        self._maximum_temperature_c = int(maximum_temperature_c)
        self._resume_temperature_c = min(
            int(resume_temperature_c), self._maximum_temperature_c
        )
        self._maximum_utilization_percent = int(maximum_utilization_percent)
        self._maximum_pulse_seconds = max(0.001, float(maximum_pulse_seconds))
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_real_activity = 0.0
        self._next_pulse = 0.0
        self._state = "disabled" if self._retention_seconds <= 0 else "idle"
        self._thermal_suspended = False
        self._pulse_count = 0
        self._skipped_busy = 0
        self._skipped_safety = 0
        self._last_pulse_seconds: float | None = None
        self._last_sample: GpuTelemetrySample | None = None

    def start(self) -> None:
        if self._retention_seconds <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="aniflive-tts-warm-retention", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._telemetry.close()

    def notify_real_activity(self) -> None:
        now = time.monotonic()
        with self._state_lock:
            self._last_real_activity = now
            self._next_pulse = now + self._pulse_interval_seconds
            self._state = "hot"
        self._wake.set()

    def status(self) -> dict[str, object]:
        with self._state_lock:
            now = time.monotonic()
            remaining = max(
                0.0, self._retention_seconds - max(0.0, now - self._last_real_activity)
            )
            sample = asdict(self._last_sample) if self._last_sample is not None else None
            return {
                "enabled": self._retention_seconds > 0,
                "state": self._state,
                "retention_seconds": self._retention_seconds,
                "remaining_seconds": round(remaining, 3),
                "pulse_interval_seconds": self._pulse_interval_seconds,
                "pulse_count": self._pulse_count,
                "skipped_busy": self._skipped_busy,
                "skipped_safety": self._skipped_safety,
                "last_pulse_seconds": self._last_pulse_seconds,
                "telemetry_available": self._telemetry.available,
                "telemetry": sample,
                "clock_locking": False,
            }

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._state_lock:
                last_activity = self._last_real_activity
                next_pulse = self._next_pulse
            if last_activity <= 0:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            now = time.monotonic()
            if now - last_activity >= self._retention_seconds:
                with self._state_lock:
                    self._state = "idle"
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            if now < next_pulse:
                self._wake.wait(timeout=min(1.0, next_pulse - now))
                self._wake.clear()
                continue
            self._attempt_pulse()
            with self._state_lock:
                self._next_pulse = time.monotonic() + self._pulse_interval_seconds

    def _attempt_pulse(self) -> None:
        sample = self._telemetry.sample()
        with self._state_lock:
            self._last_sample = sample
        if sample is None:
            with self._state_lock:
                self._state = "telemetry_unavailable"
                self._skipped_safety += 1
            return
        if self._thermal_suspended:
            if sample.temperature_c > self._resume_temperature_c:
                with self._state_lock:
                    self._state = "thermal_suspended"
                    self._skipped_safety += 1
                return
            self._thermal_suspended = False
        if sample.temperature_c >= self._maximum_temperature_c:
            self._thermal_suspended = True
            with self._state_lock:
                self._state = "thermal_suspended"
                self._skipped_safety += 1
            return
        if sample.utilization_percent > self._maximum_utilization_percent:
            with self._state_lock:
                self._state = "external_gpu_busy"
                self._skipped_safety += 1
            return
        if self._is_busy() or not self._inference_lock.acquire(blocking=False):
            with self._state_lock:
                self._state = "request_busy"
                self._skipped_busy += 1
            return
        try:
            elapsed = float(self._pulse())
        except Exception:
            LOGGER.warning("TensorRT warm-retention pulse failed", exc_info=True)
            with self._state_lock:
                self._state = "pulse_failed"
            return
        finally:
            self._inference_lock.release()
        with self._state_lock:
            self._last_pulse_seconds = round(elapsed, 6)
            if elapsed > self._maximum_pulse_seconds:
                self._state = "pulse_budget_exceeded"
                self._skipped_safety += 1
            else:
                self._state = "hot"
                self._pulse_count += 1
