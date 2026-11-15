"""Single-owner client for the local macOS Metal JSONL worker.

This module owns no model state. The Rust process is the sole MLX owner and
receives explicit JSONL requests over stdin/stdout.
"""
from __future__ import annotations

import base64
import json
import math
import os
import queue
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_START_TIMEOUT_SECONDS = 55.0
_REQUEST_EVENT_TIMEOUT_SECONDS = 600.0
_ADMISSION_TIMEOUT_SECONDS = 55.0
_STREAM_DRAIN_TIMEOUT_SECONDS = 30.0
_CANCEL_RESPONSE_TIMEOUT_SECONDS = 5.0
_SHUTDOWN_TIMEOUT_SECONDS = 30.0
_PROCESS_EXIT_TIMEOUT_SECONDS = 5.0
_READER_JOIN_TIMEOUT_SECONDS = 5.0
_SAMPLE_RATE = 48000
_SAMPLE_FORMAT = "s16le"
_MAX_RETIRED_REQUESTS = 16
_EOF = object()


class NativeWorkerError(RuntimeError):
    """A worker protocol, lifecycle, or inference error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        event: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.event = dict(event) if event is not None else None


class NativeWorkerBusy(NativeWorkerError):
    """Raised when a request is submitted while another request owns the worker."""

    def __init__(self, message: str = "native worker already has an active request") -> None:
        super().__init__("busy", message)


@dataclass(frozen=True)
class PCMChunk:
    request_id: str
    pcm: bytes
    sample_rate: int
    channels: int
    sample_format: str
    frame_end: int
    before_eos: bool
    final_chunk: bool
    original_total_samples: int | None = None

    @property
    def sample_count(self) -> int:
        return len(self.pcm) // 2


@dataclass
class _ActiveRequest:
    request_id: str
    mode: str
    events: queue.Queue[Any] = field(default_factory=queue.Queue)
    control_events: queue.Queue[Any] = field(default_factory=queue.Queue)
    consume_lock: threading.Lock = field(default_factory=threading.Lock)
    terminal_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    terminal_seen: bool = False
    terminal_consumed: bool = False
    cancel_sent: bool = False
    delivery_stopped: bool = False
    iterator: StreamPCMIterator | None = None
    result: dict[str, Any] | None = None
    late_control_events: list[dict[str, Any]] = field(default_factory=list)
    deadline: float = field(
        default_factory=lambda: time.monotonic() + _REQUEST_EVENT_TIMEOUT_SECONDS
    )


class NativeWorkerClient:
    """Manage one owned Rust worker and admit one request at a time."""

    def __init__(
        self,
        binary: Path,
        models: Path,
        cache: Path,
        metal_library: Path,
        stderr_path: Path,
    ) -> None:
        self._precision = os.environ.get("ANIFLIVE_TTS_NATIVE_PRECISION", "FP32")
        if self._precision not in {"FP32", "FP16"}:
            raise ValueError("ANIFLIVE_TTS_NATIVE_PRECISION must be FP32 or FP16")
        self.binary = Path(binary).expanduser().resolve()
        self.models = Path(models).expanduser().resolve()
        self.cache = Path(cache).expanduser().resolve()
        self.metal_library = Path(metal_library).expanduser().resolve()
        self.stderr_path = Path(stderr_path).expanduser().resolve()

        self._state_lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._admission_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._control_events: queue.Queue[Any] = queue.Queue()
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_file = None
        self._reader_thread: threading.Thread | None = None
        self._reader_eof = threading.Event()
        self._active: _ActiveRequest | None = None
        self._retired: OrderedDict[str, _ActiveRequest] = OrderedDict()
        self._fatal: NativeWorkerError | None = None
        self._metadata: dict[str, Any] | None = None
        self._ready_event_seen = False
        self._ready = False
        self._closing = False
        self._closed = False
        self._shutdown_id: str | None = None

    @property
    def ready(self) -> bool:
        with self._state_lock:
            return bool(
                self._ready
                and not self._closed
                and self._fatal is None
                and self._process is not None
                and self._process.poll() is None
            )

    @property
    def metadata(self) -> dict[str, Any] | None:
        with self._state_lock:
            return None if self._metadata is None else dict(self._metadata)

    @property
    def pid(self) -> int | None:
        with self._state_lock:
            return None if self._process is None else self._process.pid

    def __enter__(self) -> NativeWorkerClient:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def start(self) -> dict[str, Any]:
        """Start the worker and return its validated ready metadata."""
        with self._lifecycle_lock:
            with self._state_lock:
                if self._closed:
                    raise NativeWorkerError("client_closed", "native worker client is closed")
                if self._process is not None:
                    if self.ready and self._metadata is not None:
                        return dict(self._metadata)
                    raise self._fatal or NativeWorkerError(
                        "worker_started", "native worker cannot be restarted by this client"
                    )
            self._validate_paths()
            self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr_file = self.stderr_path.open("ab")
            try:
                self._process = subprocess.Popen(
                    [str(self.binary), "worker", str(self.models), str(self.cache)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=self._stderr_file,
                    env=self._child_environment(),
                    bufsize=0,
                    start_new_session=True,
                )
            except Exception:
                self._stderr_file.close()
                self._stderr_file = None
                raise
            self._reader_thread = threading.Thread(
                target=self._reader_main,
                name="aniflive-macos-worker-jsonl-reader",
                daemon=True,
            )
            self._reader_thread.start()
            try:
                event = self._next_control_event(time.monotonic() + _START_TIMEOUT_SECONDS)
                if event.get("event") != "ready":
                    self._fail("protocol_error", f"expected worker ready event, got {event!r}")
                if event.get("backend") != "mlx-metal":
                    self._fail("backend_mismatch", "native worker backend is not mlx-metal", event)
                if event.get("precision") != ("float16" if self._precision == "FP16" else "float32"):
                    self._fail("precision_mismatch", "native worker precision differs from requested precision", event)
                if event.get("neural_fallback") is not False:
                    self._fail("neural_fallback", "native worker reported a neural fallback", event)
                if event.get("operational") is not True:
                    self._fail("worker_not_operational", "native worker is not operational", event)
                if not isinstance(event.get("qualified"), bool):
                    self._fail("protocol_error", "worker ready event omitted qualified status", event)
                with self._state_lock:
                    self._metadata = dict(event)
                    self._ready = True
                return dict(event)
            except Exception:
                self._terminate_owned_on_start_failure()
                raise

    def synthesize(
        self,
        input_dir: Path,
        output_dir: Path,
        seed: int,
        *,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        input_value = self._path_value(input_dir, "input_dir")
        output_value = self._path_value(output_dir, "output_dir")
        seed_value = self._seed_value(seed)
        top_k_value = self._top_k_value(top_k)
        self._sampling_value(top_p, temperature)
        with self._admission_lock:
            active = self._reserve("synthesize")
            request = {
                "op": "synthesize",
                "request_id": active.request_id,
                "input_dir": input_value,
                "output_dir": output_value,
                "seed": seed_value,
            }
            try:
                request.update(self._sampling_fields(top_k_value, top_p, temperature))
                self._send_request(request)
            except NativeWorkerError:
                self._finish_active(active)
                raise
        try:
            self._wait_admitted(active, _ADMISSION_TIMEOUT_SECONDS)
            while True:
                event = self._next_active_event(active)
                name = event.get("event")
                if name == "complete":
                    result = event.get("result")
                    if not isinstance(result, dict):
                        self._fail("protocol_error", "complete event has no result object", event)
                    self._validate_top_k_result(result, top_k_value, "complete synthesis")
                    self._validate_sampling_result(result, top_p, temperature, "complete synthesis")
                    active.result = dict(result)
                    self._finish_active(active)
                    return dict(result)
                if name == "cancel_requested":
                    continue
                if name == "cancelled":
                    # The worker stopped before writing a WAV; nothing partial
                    # is returned as a result.
                    self._finish_active(active)
                    raise NativeWorkerError("cancelled", "complete synthesis was cancelled", event=event)
                if name == "error":
                    error = self._worker_error(event)
                    if event.get("terminal") is not False:
                        self._finish_active(active)
                    raise error
                self._fail(
                    "protocol_error",
                    f"unexpected {name!r} event for complete synthesis",
                    event,
                )
        except NativeWorkerError:
            if active.terminal_seen or self._fatal is not None:
                self._finish_active(active)
            raise

    def prepare_reference(
        self,
        input_dir: Path,
        output_dir: Path,
        *,
        audio: Path,
        transcript: Path,
    ) -> dict[str, Any]:
        """Prepare and persist a precision-specific native reference in the native worker cache.

        Reference preparation owns the worker until its matching terminal event
        and cannot be cancelled through the stream-cancellation operation.
        """
        input_value = self._path_value(input_dir, "input_dir")
        output_value = self._path_value(output_dir, "output_dir")
        audio_value = self._path_value(audio, "audio")
        transcript_value = self._path_value(transcript, "transcript")
        with self._admission_lock:
            active = self._reserve("prepare_reference")
            request = {
                "op": "prepare_reference",
                "request_id": active.request_id,
                "input_dir": input_value,
                "output_dir": output_value,
                "audio": audio_value,
                "transcript": transcript_value,
            }
            try:
                self._send_request(request)
            except NativeWorkerError:
                self._finish_active(active)
                raise
        try:
            self._wait_admitted(active, _ADMISSION_TIMEOUT_SECONDS, expected_mode="prepare_reference")
            while True:
                event = self._next_active_event(active)
                name = event.get("event")
                if name == "complete":
                    result = event.get("result")
                    if not isinstance(result, dict):
                        self._fail("protocol_error", "reference complete event has no result object", event)
                    key = result.get("cache_key")
                    frames = result.get("reference_frames")
                    tokens = result.get("reference_prompt_tokens")
                    if (not isinstance(key, str) or len(key) != 64
                            or any(char not in "0123456789abcdef" for char in key)
                            or not isinstance(result.get("cache_hit"), bool)
                            or not isinstance(result.get("cache_root"), str) or not result["cache_root"]
                            or not isinstance(result.get("report_path"), str) or not result["report_path"]
                            or result.get("precision") != ("float16" if self._precision == "FP16" else "float32")
                            or isinstance(frames, bool) or not isinstance(frames, int) or not 1 <= frames <= 500
                            or isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0):
                        self._fail("protocol_error", "invalid native reference preparation result", event)
                    active.result = dict(result)
                    self._finish_active(active)
                    return dict(result)
                if name == "error":
                    error = self._worker_error(event)
                    if event.get("terminal") is not False:
                        self._finish_active(active)
                    raise error
                self._fail("protocol_error", f"unexpected {name!r} event for reference preparation", event)
        except NativeWorkerError:
            if active.terminal_seen or self._fatal is not None:
                self._finish_active(active)
            raise

    def frontend_g2pw(self, input_dir: Path, output_dir: Path) -> dict[str, Any]:
        """Run the native G2PW component on prepared input arrays."""
        return self._frontend("frontend_g2pw", input_dir, output_dir)

    def frontend_bert(self, input_dir: Path, output_dir: Path) -> dict[str, Any]:
        """Run the native Mandarin BERT component on prepared input arrays."""
        return self._frontend("frontend_bert", input_dir, output_dir)

    def _frontend(self, mode: str, input_dir: Path, output_dir: Path) -> dict[str, Any]:
        input_value = self._path_value(input_dir, "input_dir")
        output_value = self._path_value(output_dir, "output_dir")
        with self._admission_lock:
            active = self._reserve(mode)
            request = {
                "op": mode,
                "request_id": active.request_id,
                "input_dir": input_value,
                "output_dir": output_value,
            }
            try:
                self._send_request(request)
            except NativeWorkerError:
                self._finish_active(active)
                raise
        try:
            self._wait_admitted(active, _ADMISSION_TIMEOUT_SECONDS, expected_mode=mode)
            while True:
                event = self._next_active_event(active)
                name = event.get("event")
                if name == "complete":
                    result = event.get("result")
                    if not isinstance(result, dict):
                        self._fail("protocol_error", "frontend complete event has no result object", event)
                    shape = result.get("shape")
                    valid_shape = (
                        isinstance(shape, list)
                        and all(isinstance(dim, int) and not isinstance(dim, bool) and dim > 0 for dim in shape)
                        and (
                            (mode == "frontend_g2pw" and len(shape) == 2 and shape[1] == 1305)
                            or (mode == "frontend_bert" and len(shape) == 3 and shape[0] == 1
                                and shape[1] <= 512 and shape[2] == 1024)
                        )
                    )
                    if (result.get("backend") != "mlx-metal"
                            or result.get("precision") != (("float16" if self._precision == "FP16" else "float32") if mode == "frontend_bert" else "float32")
                            or (mode == "frontend_bert" and self._precision == "FP16" and result.get("transport_dtype") != "float32")
                            or not isinstance(result.get("output_path"), str)
                            or not result["output_path"]
                            or not valid_shape):
                        self._fail("protocol_error", f"invalid {mode} result", event)
                    active.result = dict(result)
                    self._finish_active(active)
                    return dict(result)
                if name == "error":
                    error = self._worker_error(event)
                    if event.get("terminal") is not False:
                        self._finish_active(active)
                    raise error
                self._fail("protocol_error", f"unexpected {name!r} event for {mode}", event)
        except NativeWorkerError:
            if active.terminal_seen or self._fatal is not None:
                self._finish_active(active)
            raise

    def _conditioning_fields(self, style, prompt) -> dict[str, str]:
        """Optional extra conditioning sources: (input_dir, audio, transcript) tuples."""
        fields: dict[str, str] = {}
        for group, source in (("style", style), ("prompt", prompt)):
            if source is None:
                continue
            if not self._metadata or "conditioning_components" not in (self._metadata.get("capabilities") or {}):
                raise NativeWorkerError("unsupported_conditioning",
                                        "native worker does not support separate conditioning components")
            input_dir, audio, transcript = source
            fields[f"{group}_input_dir"] = self._path_value(input_dir, f"{group}_input_dir")
            fields[f"{group}_audio"] = self._path_value(audio, f"{group}_audio")
            fields[f"{group}_transcript"] = self._path_value(transcript, f"{group}_transcript")
        return fields

    def synthesize_cached(
        self,
        input_dir: Path,
        output_dir: Path,
        seed: int,
        *,
        audio: Path,
        transcript: Path,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        style: tuple[Path, Path, Path] | None = None,
        prompt: tuple[Path, Path, Path] | None = None,
    ) -> dict[str, Any]:
        """Run seeded synthesis while reusing the worker-owned reference cache."""
        input_value = self._path_value(input_dir, "input_dir")
        output_value = self._path_value(output_dir, "output_dir")
        audio_value = self._path_value(audio, "audio")
        transcript_value = self._path_value(transcript, "transcript")
        seed_value = self._seed_value(seed)
        top_k_value = self._top_k_value(top_k)
        self._sampling_value(top_p, temperature)
        with self._admission_lock:
            active = self._reserve("synthesize_cached")
            request = {
                "op": "synthesize_cached",
                "request_id": active.request_id,
                "input_dir": input_value,
                "output_dir": output_value,
                "audio": audio_value,
                "transcript": transcript_value,
                "seed": seed_value,
            }
            try:
                request.update(self._sampling_fields(top_k_value, top_p, temperature))
                request.update(self._conditioning_fields(style, prompt))
                self._send_request(request)
            except NativeWorkerError:
                self._finish_active(active)
                raise
        try:
            self._wait_admitted(
                active,
                _ADMISSION_TIMEOUT_SECONDS,
                expected_mode="synthesize_cached",
            )
            while True:
                event = self._next_active_event(active)
                name = event.get("event")
                if name == "complete":
                    result = event.get("result")
                    if not isinstance(result, dict):
                        self._fail("protocol_error", "cached complete event has no result object", event)
                    self._validate_top_k_result(result, top_k_value, "cached complete synthesis")
                    self._validate_sampling_result(result, top_p, temperature, "cached complete synthesis")
                    result_path = result.get("result_path")
                    cache_key = result.get("cache_key")
                    if (
                        not isinstance(result_path, str)
                        or not result_path
                        or result.get("sample_rate") != 48000
                        or result.get("format") != "wav"
                        or not isinstance(result.get("cache_hit"), bool)
                        or not isinstance(cache_key, str)
                        or len(cache_key) != 64
                        or any(character not in "0123456789abcdef" for character in cache_key)
                        or result.get("reference_conditioning_reused") is not True
                        or result.get("stochastic_and_neural_capture") is not False
                    ):
                        self._fail("protocol_error", "invalid cached synthesis result", event)
                    active.result = dict(result)
                    self._finish_active(active)
                    return dict(result)
                if name == "cancel_requested":
                    continue
                if name == "cancelled":
                    # The worker stopped before writing a WAV; nothing partial
                    # is returned as a result.
                    self._finish_active(active)
                    raise NativeWorkerError("cancelled", "cached synthesis was cancelled", event=event)
                if name == "error":
                    error = self._worker_error(event)
                    if event.get("terminal") is not False:
                        self._finish_active(active)
                    raise error
                self._fail(
                    "protocol_error",
                    f"unexpected {name!r} event for cached synthesis",
                    event,
                )
        except NativeWorkerError:
            if active.terminal_seen or self._fatal is not None:
                self._finish_active(active)
            raise

    def stream(
        self,
        input_dir: Path,
        output_dir: Path,
        seed: int,
        audio: Path,
        transcript: Path,
        geometry: Mapping[str, int] | None = None,
        *,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        style: tuple[Path, Path, Path] | None = None,
        prompt: tuple[Path, Path, Path] | None = None,
    ) -> StreamPCMIterator:
        input_value = self._path_value(input_dir, "input_dir")
        output_value = self._path_value(output_dir, "output_dir")
        audio_value = self._path_value(audio, "audio")
        transcript_value = self._path_value(transcript, "transcript")
        seed_value = self._seed_value(seed)
        top_k_value = self._top_k_value(top_k)
        self._sampling_value(top_p, temperature)
        geometry_values = {} if geometry is None else self._geometry_values(geometry)
        with self._admission_lock:
            active = self._reserve("stream")
            iterator = StreamPCMIterator(self, active)
            active.iterator = iterator
            request: dict[str, Any] = {
                "op": "stream",
                "request_id": active.request_id,
                "input_dir": input_value,
                "output_dir": output_value,
                "audio": audio_value,
                "transcript": transcript_value,
                "seed": seed_value,
            }
            request.update(geometry_values)
            try:
                request.update(self._sampling_fields(top_k_value, top_p, temperature))
                request.update(self._conditioning_fields(style, prompt))
                self._send_request(request)
                self._wait_admitted(active, _ADMISSION_TIMEOUT_SECONDS)
            except NativeWorkerError:
                if active.terminal_seen or self._fatal is not None:
                    self._finish_active(active)
                raise
            return iterator

    def cancel_active_stream(self) -> bool:
        """Cancel an active stream, or raise the worker's explicit synth cancel error."""
        unsupported: _ActiveRequest | None = None
        with self._state_lock:
            self._raise_if_fatal_locked()
            active = self._active
            if active is None:
                return False
            if active.mode != "stream":
                if active.terminal_seen or active.terminal_consumed or active.cancel_sent:
                    return False
                capabilities = (self._metadata or {}).get("capabilities") or {}
                if active.mode in {"synthesize", "synthesize_cached"} and capabilities.get("synthesize_cancel") is True:
                    # Acknowledged as cancel_requested; the synthesis call then
                    # ends with a "cancelled" terminal event.
                    active.cancel_sent = True
                    self._send_request({"op": "cancel", "request_id": active.request_id})
                    return True
                self._send_request({"op": "cancel", "request_id": active.request_id})
                unsupported = active
            else:
                if active.terminal_seen or active.terminal_consumed or active.cancel_sent:
                    return False
                active.delivery_stopped = True
                active.cancel_sent = True
                self._send_request({"op": "cancel", "request_id": active.request_id})
                return True
        if unsupported is not None:
            event = self._next_active_control_event(
                unsupported, _CANCEL_RESPONSE_TIMEOUT_SECONDS
            )
            raise self._worker_error(event)
        return False

    def close(self) -> None:
        """Gracefully shut down this owned process, killing only it if unresponsive."""
        with self._lifecycle_lock:
            with self._admission_lock:
                with self._state_lock:
                    if self._closed:
                        return
                    self._closing = True
                    active = self._active
                stream_close_error: NativeWorkerError | None = None
                if active is not None and active.mode == "stream" and active.iterator is not None:
                    try:
                        active.iterator.close()
                    except NativeWorkerError as error:
                        stream_close_error = error

                process = self._process
                if process is None:
                    with self._state_lock:
                        self._closed = True
                        self._ready = False
                    return

                shutdown_error: NativeWorkerError | None = None
                if process.poll() is None:
                    shutdown_id = uuid.uuid4().hex
                    with self._state_lock:
                        self._shutdown_id = shutdown_id
                    try:
                        self._send_request({"op": "shutdown", "request_id": shutdown_id})
                        deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS
                        shutdown_complete = False
                        while not shutdown_complete:
                            event = self._next_control_event(deadline)
                            if event.get("request_id") != shutdown_id:
                                self._fail(
                                    "protocol_error",
                                    "shutdown event request_id does not match",
                                    event,
                                )
                            if event.get("event") == "shutdown_requested":
                                continue
                            if event.get("event") == "shutdown_complete":
                                shutdown_complete = True
                            elif event.get("event") == "error":
                                raise self._worker_error(event)
                            else:
                                self._fail(
                                    "protocol_error",
                                    "unexpected event while shutting down worker",
                                    event,
                                )
                    except NativeWorkerError as error:
                        shutdown_error = error
                try:
                    self._wait_then_terminate_owned(process)
                finally:
                    if self._reader_thread is not None:
                        self._reader_thread.join(timeout=_READER_JOIN_TIMEOUT_SECONDS)
                    if self._stderr_file is not None and not self._stderr_file.closed:
                        self._stderr_file.flush()
                        self._stderr_file.close()
                    with self._state_lock:
                        self._closed = True
                        self._ready = False
                if shutdown_error is not None:
                    raise shutdown_error
                if stream_close_error is not None:
                    raise stream_close_error

    def _validate_paths(self) -> None:
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise FileNotFoundError(f"native worker binary is not executable: {self.binary}")
        if not self.models.is_dir():
            raise FileNotFoundError(f"native model directory not found: {self.models}")
        if not self.metal_library.exists():
            raise FileNotFoundError(f"MLX Metal library path not found: {self.metal_library}")

    def _terminate_owned_on_start_failure(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=_PROCESS_EXIT_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=_PROCESS_EXIT_TIMEOUT_SECONDS)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=_READER_JOIN_TIMEOUT_SECONDS)
        if self._stderr_file is not None and not self._stderr_file.closed:
            self._stderr_file.flush()
            self._stderr_file.close()
        with self._state_lock:
            self._ready = False
            self._closed = True

    def _child_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["MLX_ENABLE_TF32"] = "0"
        environment["ANIFLIVE_TTS_NATIVE_PRECISION"] = self._precision
        environment.pop("ANIFLIVE_COMPONENT_CPU_REFERENCE", None)
        # CLI experiments use this process-level option. The API selects
        # top_k per request; keeping this variable would bypass that contract.
        environment.pop("ANIFLIVE_TTS_RESEARCH_AR_TOP_K", None)
        mlx_dir = self.metal_library if self.metal_library.is_dir() else self.metal_library.parent
        prebuilt_root = mlx_dir.parent if mlx_dir.name == "lib" else mlx_dir
        environment["MLX_SYS_PREBUILT"] = str(prebuilt_root)
        for variable in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
            existing = environment.get(variable, "")
            environment[variable] = os.pathsep.join(
                item for item in (str(mlx_dir), existing) if item
            )
        return environment

    def _reserve(self, mode: str) -> _ActiveRequest:
        with self._state_lock:
            self._raise_if_fatal_locked()
            if self._closing or self._closed or not self._ready:
                raise NativeWorkerError("worker_not_ready", "native worker is not ready")
            if self._process is None or self._process.poll() is not None:
                error = NativeWorkerError("worker_exit", "native worker process has exited")
                self._set_fatal_locked(error)
                raise error
            if self._active is not None:
                raise NativeWorkerBusy()
            active = _ActiveRequest(request_id=uuid.uuid4().hex, mode=mode)
            self._active = active
            return active

    def _send_request(self, request: dict[str, Any]) -> None:
        try:
            with self._write_lock:
                self._write_json_locked(request)
        except NativeWorkerError as error:
            self._set_fatal(error)
            raise

    def _write_json_locked(self, request: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise NativeWorkerError("worker_not_started", "native worker stdin is unavailable")
        if process.poll() is not None:
            raise NativeWorkerError("worker_exit", "native worker process has exited")
        try:
            line = json.dumps(request, separators=(",", ":"), ensure_ascii=False) + "\n"
            process.stdin.write(line.encode("utf-8"))
            process.stdin.flush()
        except (OSError, BrokenPipeError) as error:
            raise NativeWorkerError("worker_stdin", f"failed to write worker request: {error}") from error

    def _reader_main(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            self._set_fatal(NativeWorkerError("worker_stdout", "worker stdout is unavailable"))
            self._reader_eof.set()
            return
        line_number = 0
        try:
            while True:
                raw_line = process.stdout.readline()
                if raw_line == b"":
                    break
                line_number += 1
                if not raw_line.endswith(b"\n"):
                    raise NativeWorkerError(
                        "invalid_jsonl", f"worker stdout line {line_number} is not newline-terminated"
                    )
                try:
                    event = json.loads(raw_line.decode("utf-8"))
                except Exception as error:
                    excerpt = raw_line[:120].decode("utf-8", errors="replace")
                    raise NativeWorkerError(
                        "invalid_jsonl", f"worker stdout line {line_number} is not valid JSON: {excerpt!r}"
                    ) from error
                if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                    raise NativeWorkerError(
                        "invalid_jsonl", f"worker stdout line {line_number} is not a JSON event object"
                    )
                self._route_event(event)
        except NativeWorkerError as error:
            self._set_fatal(error)
        except Exception as error:
            self._set_fatal(NativeWorkerError("worker_reader", str(error)))
        finally:
            with self._state_lock:
                closing = self._closing
                active = self._active
            if not closing:
                self._set_fatal(
                    NativeWorkerError("worker_eof", "native worker stdout closed unexpectedly")
                )
            else:
                self._control_events.put(_EOF)
                if active is not None:
                    active.events.put(_EOF)
            self._reader_eof.set()

    def _route_event(self, event: dict[str, Any]) -> None:
        name = event.get("event")
        request_id = event.get("request_id")
        with self._state_lock:
            if name == "ready":
                if self._ready_event_seen or self._process is None:
                    self._set_fatal_locked(
                        NativeWorkerError(
                            "protocol_error", "unexpected duplicate ready event", event=event
                        )
                    )
                    return
                self._ready_event_seen = True
                self._control_events.put(event)
                return
            if name in {"shutdown_requested", "shutdown_complete"}:
                if request_id != self._shutdown_id:
                    self._set_fatal_locked(
                        NativeWorkerError(
                            "protocol_error",
                            "shutdown event has an unexpected request_id",
                            event=event,
                        )
                    )
                    return
                self._control_events.put(event)
                return
            active = self._active
            if active is not None and request_id == active.request_id:
                if name == "error" and (
                    (event.get("operation") == "cancel" and event.get("terminal") is False)
                    or event.get("code") == "not_active"
                ):
                    active.control_events.put(event)
                    return
                if name in {"complete", "cancelled"} or (
                    name == "error" and event.get("terminal") is not False
                ):
                    active.terminal_seen = True
                    active.terminal_event.set()
                active.events.put(event)
                return
            retired = self._retired.get(str(request_id)) if request_id is not None else None
            if retired is not None:
                if name == "error" and event.get("code") == "not_active":
                    retired.late_control_events.append(dict(event))
                    retired.events.put(event)
                    return
                self._set_fatal_locked(
                    NativeWorkerError(
                        "protocol_error",
                        "worker emitted a non-control event for a retired request_id",
                        event=event,
                    )
                )
                return
            self._set_fatal_locked(
                NativeWorkerError(
                    "unexpected_request_id",
                    f"worker emitted {name!r} for unexpected request_id {request_id!r}",
                    event=event,
                )
            )

    def _set_fatal(self, error: NativeWorkerError) -> None:
        with self._state_lock:
            self._set_fatal_locked(error)

    def _set_fatal_locked(self, error: NativeWorkerError) -> None:
        if self._fatal is not None:
            return
        self._fatal = error
        self._ready = False
        if self._active is not None:
            self._active.terminal_event.set()
            self._active.events.put(error)
            self._active.control_events.put(error)
        self._control_events.put(error)

    def _raise_if_fatal_locked(self) -> None:
        if self._fatal is not None:
            raise self._fatal

    def _next_control_event(self, deadline: float) -> dict[str, Any]:
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise NativeWorkerError("worker_timeout", "timed out waiting for worker control event")
        try:
            item = self._control_events.get(timeout=timeout)
        except queue.Empty as error:
            raise NativeWorkerError("worker_timeout", "timed out waiting for worker control event") from error
        if item is _EOF:
            raise NativeWorkerError("worker_eof", "worker stdout closed before control event")
        if isinstance(item, NativeWorkerError):
            raise item
        if not isinstance(item, dict):
            raise NativeWorkerError("protocol_error", "invalid worker control event")
        return item

    def _next_active_event(
        self, active: _ActiveRequest, timeout: float | None = None
    ) -> dict[str, Any]:
        deadline = active.deadline if timeout is None else time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise NativeWorkerError("worker_timeout", "timed out waiting for active worker request")
            try:
                item = active.events.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                if self._fatal is not None:
                    raise self._fatal
                process = self._process
                if process is not None and process.poll() is not None:
                    error = NativeWorkerError("worker_exit", "native worker process exited during request")
                    self._set_fatal(error)
                    raise error
                continue
            if item is _EOF:
                raise NativeWorkerError("worker_eof", "worker stdout closed during request")
            if isinstance(item, NativeWorkerError):
                raise item
            if not isinstance(item, dict):
                self._fail("protocol_error", "invalid event routed to active request")
            if item.get("request_id") != active.request_id:
                self._fail("unexpected_request_id", "event was routed to the wrong active request", item)
            return item

    def _next_active_control_event(
        self, active: _ActiveRequest, timeout: float
    ) -> dict[str, Any]:
        try:
            item = active.control_events.get(timeout=timeout)
        except queue.Empty as error:
            raise NativeWorkerError(
                "worker_timeout", "timed out waiting for worker control response"
            ) from error
        if isinstance(item, NativeWorkerError):
            raise item
        if not isinstance(item, dict) or item.get("request_id") != active.request_id:
            raise NativeWorkerError("protocol_error", "invalid active-request control event")
        return item

    def _wait_admitted(
        self, active: _ActiveRequest, timeout: float, *, expected_mode: str | None = None
    ) -> None:
        deadline = time.monotonic() + timeout
        accepted = False
        started = False
        while not (accepted and started):
            event = self._next_active_event(active, max(0.001, deadline - time.monotonic()))
            name = event.get("event")
            if name == "accepted":
                if expected_mode is not None and event.get("mode") != expected_mode:
                    self._fail("protocol_error", "worker accepted an unexpected request mode", event)
                accepted = True
                continue
            if name == "started":
                if not accepted:
                    self._fail("protocol_error", "worker started a request before accepting it", event)
                started = True
                continue
            if name == "cancel_requested":
                # A cancel can be acknowledged between accepted and started.
                continue
            if name == "error":
                self._raise_worker_event(active, event)
            self._fail("protocol_error", f"unexpected {name!r} before request admission", event)

    def _raise_worker_event(self, active: _ActiveRequest, event: dict[str, Any]) -> None:
        error = self._worker_error(event)
        if event.get("terminal") is not False:
            self._finish_active(active)
        raise error

    @staticmethod
    def _worker_error(event: Mapping[str, Any]) -> NativeWorkerError:
        code = str(event.get("code", "worker_error"))
        message = str(event.get("message", "native worker returned an unspecified error"))
        if code == "busy":
            return NativeWorkerBusy(message)
        return NativeWorkerError(code, message, event=event)

    def _finish_active(self, active: _ActiveRequest) -> None:
        with self._state_lock:
            if active.terminal_consumed:
                return
            active.terminal_consumed = True
            active.done_event.set()
            if self._active is active:
                self._active = None
            self._retired[active.request_id] = active
            while len(self._retired) > _MAX_RETIRED_REQUESTS:
                self._retired.popitem(last=False)

    def _cancel_active(self, active: _ActiveRequest) -> bool:
        with self._state_lock:
            if active.mode != "stream":
                raise NativeWorkerError(
                    "cancel_unsupported", "complete synthesis does not support cancellation"
                )
            active.delivery_stopped = True
            if active.terminal_consumed or active.terminal_seen or active.cancel_sent:
                return False
            if self._active is not active:
                return False
            active.cancel_sent = True
            self._send_request({"op": "cancel", "request_id": active.request_id})
            return True

    @staticmethod
    def _pcm_chunk(event: Mapping[str, Any]) -> PCMChunk:
        if event.get("event") != "pcm":
            raise NativeWorkerError("protocol_error", "expected PCM event", event=event)
        if event.get("sample_rate") != _SAMPLE_RATE:
            raise NativeWorkerError(
                "invalid_pcm", f"PCM sample rate must be {_SAMPLE_RATE}", event=event
            )
        channels = event.get("channels")
        sample_format = event.get("sample_format")
        if channels != 1 or sample_format != _SAMPLE_FORMAT:
            raise NativeWorkerError(
                "invalid_pcm", "PCM must be mono signed little-endian PCM16", event=event
            )
        encoded = event.get("pcm_base64")
        if not isinstance(encoded, str) or not encoded:
            raise NativeWorkerError("invalid_pcm", "PCM event has no base64 payload", event=event)
        try:
            pcm = base64.b64decode(encoded, validate=True)
        except Exception as error:
            raise NativeWorkerError("invalid_pcm", f"invalid PCM base64: {error}", event=event) from error
        if not pcm or len(pcm) % 2:
            raise NativeWorkerError("invalid_pcm", "PCM payload must be aligned nonempty PCM16", event=event)
        frame_end = event.get("frame_end")
        before_eos = event.get("before_eos")
        final_chunk = event.get("final_chunk")
        if isinstance(frame_end, bool) or not isinstance(frame_end, int) or frame_end < 0:
            raise NativeWorkerError("invalid_pcm", "PCM frame_end must be a nonnegative integer", event=event)
        if not isinstance(before_eos, bool) or not isinstance(final_chunk, bool):
            raise NativeWorkerError("invalid_pcm", "PCM timing flags must be boolean", event=event)
        total = event.get("original_total_samples")
        if total is not None and (isinstance(total, bool) or not isinstance(total, int)
                                  or total <= 0 or total < len(pcm) // 2 or before_eos):
            raise NativeWorkerError("invalid_pcm", "Invalid canonical original_total_samples", event=event)
        request_id = event.get("request_id")
        if not isinstance(request_id, str):
            raise NativeWorkerError("invalid_pcm", "PCM event has no request_id", event=event)
        return PCMChunk(
            request_id=request_id,
            pcm=pcm,
            sample_rate=_SAMPLE_RATE,
            channels=1,
            sample_format=_SAMPLE_FORMAT,
            frame_end=frame_end,
            before_eos=before_eos,
            final_chunk=final_chunk,
            original_total_samples=total,
        )

    def _fail(self, code: str, message: str, event: Mapping[str, Any] | None = None):
        error = NativeWorkerError(code, message, event=event)
        self._set_fatal(error)
        raise error

    @staticmethod
    def _path_value(value: Path, label: str) -> str:
        path = Path(value).expanduser()
        if not str(path):
            raise ValueError(f"{label} path must be nonempty")
        return str(path)

    @staticmethod
    def _seed_value(seed: int) -> int:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("seed must be an unsigned 64-bit integer")
        return seed

    @staticmethod
    def _top_k_value(top_k: int) -> int:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
            raise ValueError("top_k must be an integer between 1 and 50 for the native worker")
        return top_k

    @staticmethod
    def _sampling_value(top_p: float, temperature: float) -> None:
        for name, value in (("top_p", top_p), ("temperature", temperature)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

    def _sampling_fields(self, top_k: int, top_p: float, temperature: float) -> dict[str, Any]:
        """JSONL sampling fields. The default policy sends nothing (unchanged
        contract); any other Windows-legal set needs a worker that reports
        request-level sampling_parameters."""
        if top_k == 15 and top_p == 1.0 and temperature == 1.0:
            return {}
        capabilities = (self._metadata or {}).get("capabilities") or {}
        if "sampling_parameters" not in capabilities:
            if top_k == 1 and top_p == 1.0 and temperature == 1.0:
                return {"top_k": 1}
            raise NativeWorkerError("unsupported_sampling",
                                    "native worker supports only the default AR sampling parameters")
        fields: dict[str, Any] = {}
        if top_k != 15:
            fields["top_k"] = top_k
        if top_p != 1.0:
            fields["top_p"] = float(top_p)
        if temperature != 1.0:
            fields["temperature"] = float(temperature)
        return fields

    @staticmethod
    def _validate_sampling_result(result: Mapping[str, Any], top_p: float, temperature: float, label: str) -> None:
        for name, requested in (("top_p", top_p), ("temperature", temperature)):
            reported = result.get(name)
            if reported is None:
                if requested != 1.0:
                    raise NativeWorkerError("protocol_error", f"worker omitted {name} for {label}")
                continue
            # The worker carries the value as float32.
            if isinstance(reported, bool) or not isinstance(reported, (int, float)) \
                    or not math.isclose(float(reported), float(requested), rel_tol=1e-6, abs_tol=1e-7):
                raise NativeWorkerError("protocol_error", f"worker {name} does not match {label} request")

    @staticmethod
    def _top_k_request_field(top_k: int) -> dict[str, int]:
        # Omitting 15 preserves the existing JSONL request contract.
        return {"top_k": 1} if top_k == 1 else {}

    @staticmethod
    def _validate_top_k_result(
        result: Mapping[str, Any], requested_top_k: int, label: str
    ) -> None:
        reported = result.get("top_k")
        if reported is None:
            if requested_top_k != 15:
                raise NativeWorkerError("protocol_error", f"worker omitted top_k for {label}")
            return
        if isinstance(reported, bool) or not isinstance(reported, int) or reported != requested_top_k:
            raise NativeWorkerError("protocol_error", f"worker top_k does not match {label} request")

    @staticmethod
    def _geometry_values(geometry: Mapping[str, int]) -> dict[str, int]:
        if not isinstance(geometry, Mapping):
            raise TypeError("geometry must be a mapping of worker stream geometry fields")
        allowed = {"core_frames", "future_frames", "holdback_frames", "fade_frames"}
        unknown = set(geometry) - allowed
        if unknown:
            raise ValueError(f"unsupported stream geometry fields: {sorted(unknown)}")
        result: dict[str, int] = {}
        for key, value in geometry.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"stream geometry field {key} must be an integer")
            result[key] = value
        return result

    def _wait_then_terminate_owned(self, process: subprocess.Popen[bytes]) -> None:
        try:
            exit_code = process.wait(timeout=_PROCESS_EXIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                exit_code = process.wait(timeout=_PROCESS_EXIT_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = process.wait(timeout=_PROCESS_EXIT_TIMEOUT_SECONDS)
        if exit_code != 0:
            raise NativeWorkerError("worker_exit", f"native worker exited with status {exit_code}")


class StreamPCMIterator(Iterator[PCMChunk]):
    """Request-owned PCM iterator; close drains the matching stream terminal."""

    def __init__(self, client: NativeWorkerClient, active: _ActiveRequest) -> None:
        self._client = client
        self._active = active
        self._closed = False
        self._result: dict[str, Any] | None = None
        self._reference_metadata: dict[str, Any] | None = None
        self._total_samples_seen = False
        self._original_total_samples: int | None = None
        self._received_samples = 0
        self.cancelled = False

    @property
    def request_id(self) -> str:
        return self._active.request_id

    @property
    def result(self) -> dict[str, Any] | None:
        return None if self._result is None else dict(self._result)

    @property
    def reference_metadata(self) -> dict[str, Any] | None:
        return None if self._reference_metadata is None else dict(self._reference_metadata)

    def __iter__(self) -> StreamPCMIterator:
        return self

    def __next__(self) -> PCMChunk:
        active = self._active
        with active.consume_lock:
            if self._closed or active.terminal_consumed:
                raise StopIteration
            while True:
                event = self._client._next_active_event(active)
                name = event.get("event")
                if name == "reference_ready":
                    self._reference_metadata = dict(event)
                    continue
                if name in {"accepted", "started", "cancel_requested"}:
                    continue
                if name == "pcm":
                    chunk = self._validated_chunk(event)
                    with self._client._state_lock:
                        stop_delivery = active.delivery_stopped
                    if stop_delivery:
                        continue
                    return chunk
                if name in {"complete", "cancelled"}:
                    self._result = dict(event.get("result", {}))
                    self.cancelled = name == "cancelled"
                    self._closed = True
                    self._client._finish_active(active)
                    raise StopIteration
                if name == "error":
                    error = self._client._worker_error(event)
                    self._closed = True
                    if event.get("terminal") is not False:
                        self._client._finish_active(active)
                    raise error
                self._client._fail(
                    "protocol_error", f"unexpected {name!r} event for stream request", event
                )

    def _validated_chunk(self, event: Mapping[str, Any]) -> PCMChunk:
        chunk = self._client._pcm_chunk(event)
        total = chunk.original_total_samples
        if self._total_samples_seen and total != self._original_total_samples:
            self._client._fail("invalid_pcm", "PCM total sample count changed during stream", event)
        self._total_samples_seen = True
        self._original_total_samples = total
        self._received_samples += chunk.sample_count
        if total is not None and (self._received_samples > total or
                                  (chunk.final_chunk and self._received_samples != total)):
            self._client._fail("invalid_pcm", "PCM sample count exceeds or disagrees with total", event)
        return chunk

    def close(self) -> None:
        active = self._active
        if self._closed or active.terminal_consumed:
            self._closed = True
            return
        # Send before taking the consume lock so a blocked __next__ can drain
        # cancellation events and release the lock.
        self._client._cancel_active(active)
        deadline = time.monotonic() + _STREAM_DRAIN_TIMEOUT_SECONDS
        with active.consume_lock:
            if self._closed or active.terminal_consumed:
                self._closed = True
                return
            while True:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    raise NativeWorkerError(
                        "stream_close_timeout", "timed out draining stream after cancellation"
                    )
                event = self._client._next_active_event(active, timeout=timeout)
                name = event.get("event")
                if name == "pcm":
                    # Validate but intentionally drop in-flight media once the
                    # consumer has closed or requested cancellation.
                    self._validated_chunk(event)
                    continue
                if name == "reference_ready":
                    self._reference_metadata = dict(event)
                    continue
                if name in {"accepted", "started", "cancel_requested"}:
                    continue
                if name in {"complete", "cancelled"}:
                    self._result = dict(event.get("result", {}))
                    self.cancelled = name == "cancelled"
                    self._closed = True
                    self._client._finish_active(active)
                    return
                if name == "error":
                    error = self._client._worker_error(event)
                    self._closed = True
                    if event.get("terminal") is not False:
                        self._client._finish_active(active)
                    raise error
                self._client._fail(
                    "protocol_error", f"unexpected {name!r} event while closing stream", event
                )

    def __enter__(self) -> StreamPCMIterator:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "NativeWorkerBusy",
    "NativeWorkerClient",
    "NativeWorkerError",
    "PCMChunk",
    "StreamPCMIterator",
]
