"""Deterministic stdlib HTTP fixture for the audited AnifLive-TTS API.

Usage:
    with FixtureServer() as fixture:
        client = Client(fixture.base_url)
        ...

The fixture is protocol-only. It imports no server, inference, CUDA, or SDK code.
It binds an ephemeral 127.0.0.1 port and records every request as a RequestRecord.
Use set_override() for malformed JSON/metadata and selected HTTP error scenarios.
"""

from __future__ import annotations

import hashlib
import io
import json
import select
import socket
import threading
import wave
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any
from urllib.parse import unquote, urlsplit

DEFAULT_MODEL = "fixture-v2proplus"
ALTERNATE_MODEL = "fixture-v2proplus-alt"
SAMPLE_RATE = 32_000
FIXTURE_TIME = "2026-09-20T00:00:00Z"
CORE_LANGUAGES = ("zh", "yue", "en", "ja", "ko")
PCM_CHUNKS = (b"\x01\x00\x02", b"\x00\x03\x00\x04\x00")


@dataclass(frozen=True)
class RequestRecord:
    """One received HTTP request, including raw target/body and socket identity."""

    method: str
    path: str
    body: bytes
    headers: tuple[tuple[str, str], ...]
    connection_id: int

    @property
    def json_body(self) -> Any:
        """Decode the recorded body as JSON when a test needs its parsed value."""
        return json.loads(self.body) if self.body else None


@dataclass(frozen=True)
class ResponseOverride:
    """One exact-route response override for protocol failure-path tests."""

    status: int = 200
    body: bytes | Mapping[str, Any] | list[Any] | None = None
    headers: Mapping[str, str | None] = field(default_factory=dict)
    chunks: tuple[bytes, ...] | None = None
    delay_seconds: float = 0.0


@dataclass
class _Segment:
    segment_id: str
    sequence: int
    text: str
    language: str
    fingerprint: str
    paragraph_id: str | None
    pause_after_ms: int
    state: str = "queued"
    error: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "sequence": self.sequence,
            "characters": len(self.text),
            "language": self.language,
            "paragraph_id": self.paragraph_id,
            "pause_after_ms": self.pause_after_ms,
            "state": self.state,
            "created_at": FIXTURE_TIME,
            "updated_at": FIXTURE_TIME,
            "error": self.error,
        }


@dataclass
class _Session:
    session_id: str
    model: str
    voice_profile: str
    continuity_policy: str
    state: str = "open"
    segments: list[_Segment] = field(default_factory=list)
    audio_consumer_attached: bool = False
    next_stream_index: int = 0
    total_output_samples: int = 0
    previous_output_samples: int = 0
    previous_language: str | None = None
    previous_text: str = ""
    paragraph_id: str | None = None
    previous_pause_ms: int = 0
    last_plan: dict[str, Any] | None = None

    def public_dict(self) -> dict[str, Any]:
        completed = sum(segment.state == "completed" for segment in self.segments)
        uses_context = self.continuity_policy != "A" and completed > 0
        context = {
            "mode": "committed-neural-v1",
            "policy": self.continuity_policy,
            "neural_state_continuity": uses_context,
            "acoustic_latent_continuity": False,
            "committed_segments": completed,
            "previous_language": self.previous_language,
            "previous_expression": None,
            "paragraph_id": self.paragraph_id,
            "previous_pause_ms": self.previous_pause_ms,
            "previous_output_samples": self.previous_output_samples if completed else 0,
            "total_output_samples": self.total_output_samples,
            "hard_boundary": self.previous_pause_ms >= 700,
            "retained_phonemes": min(len(self.previous_text), 75) if uses_context else 0,
            "retained_semantic_tokens": 0,
            "retained_neural_bytes": 0,
            "last_plan": self.last_plan,
        }
        return {
            "id": self.session_id,
            "object": "speech.session",
            "state": self.state,
            "model": self.model,
            "voice_profile": self.voice_profile,
            "sample_rate": SAMPLE_RATE,
            "format": "pcm_s16le",
            "continuity_policy": self.continuity_policy,
            "context": context,
            "audio_consumer_attached": self.audio_consumer_attached,
            "created_at": FIXTURE_TIME,
            "updated_at": FIXTURE_TIME,
            "segments": [segment.public_dict() for segment in self.segments],
        }


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _tiny_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(b"\x00\x00\x10\x00\xf0\xff\x00\x00")
    return output.getvalue()


class FixtureServer:
    """Context-managed localhost API fixture with deterministic wire behavior.

    Public integration surface:
      - base_url: ephemeral HTTP URL while the context is active.
      - records: immutable snapshot of method/path/body/headers/connection_id.
      - pcm_chunks and stream_delay_seconds: configure incremental PCM output.
      - set_override(method, path, ...): inject malformed bodies/headers, status
        errors such as 429/Retry-After, or a delayed chunk stream.

    The default stream is deliberately split across an odd-byte HTTP chunk
    boundary. Consumers must concatenate raw PCM bytes rather than treating HTTP
    chunk boundaries as audio frames.
    """

    def __init__(
        self,
        *,
        pcm_chunks: Iterable[bytes] = PCM_CHUNKS,
        stream_delay_seconds: float = 0.0,
        expression_enabled: bool = False,
    ) -> None:
        if stream_delay_seconds < 0:
            raise ValueError("stream_delay_seconds must be non-negative")
        chunks = tuple(pcm_chunks)
        if any(not isinstance(chunk, bytes) for chunk in chunks):
            raise TypeError("pcm_chunks must contain bytes")
        self.pcm_chunks = chunks
        self.stream_delay_seconds = float(stream_delay_seconds)
        self.expression_enabled = bool(expression_enabled)
        self.wav_bytes = _tiny_wav()
        self.active_model = DEFAULT_MODEL
        self.available_models = (DEFAULT_MODEL, ALTERNATE_MODEL)

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._records: list[RequestRecord] = []
        self._connection_ids: dict[int, tuple[socket.socket, int]] = {}
        self._next_connection_id = 0
        self._next_session_id = 0
        self._sessions: dict[str, _Session] = {}
        self._overrides: dict[tuple[str, str], ResponseOverride] = {}
        self._active_stream: tuple[str, threading.Event] | None = None
        self._stream_queue: list[tuple[str, threading.Event]] = []
        self._current_stream: tuple[str, threading.Event] | None = None
        self._stop = threading.Event()
        self._closed = False

        self._httpd = _FixtureHTTPServer(("127.0.0.1", 0), self)
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        return str(self._httpd.server_address[0])

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def records(self) -> tuple[RequestRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def clear_records(self) -> None:
        with self._lock:
            self._records.clear()

    def set_override(
        self,
        method: str,
        path: str,
        *,
        status: int = 200,
        body: bytes | Mapping[str, Any] | list[Any] | None = None,
        headers: Mapping[str, str | None] | None = None,
        chunks: Iterable[bytes] | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        """Override one exact URL path (query strings are ignored).

        A header value of None omits that header. Use raw bytes for malformed
        JSON; use chunks to return a chunked body. Overrides are removed with
        clear_override() and are safe to install before a request is sent.
        """
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")
        route = urlsplit(path).path
        override = ResponseOverride(
            status=status,
            body=body,
            headers=dict(headers or {}),
            chunks=tuple(bytes(chunk) for chunk in chunks) if chunks is not None else None,
            delay_seconds=float(delay_seconds),
        )
        with self._lock:
            self._overrides[(method.upper(), route)] = override

    def clear_override(self, method: str, path: str) -> None:
        route = urlsplit(path).path
        with self._lock:
            self._overrides.pop((method.upper(), route), None)

    def __enter__(self) -> FixtureServer:  # noqa: PYI034 - Python 3.10 compatibility
        if self._closed:
            raise RuntimeError("FixtureServer is closed")
        if self._thread is not None:
            raise RuntimeError("FixtureServer is already running")
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="aniflive-tts-fixture",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        with self._condition:
            events = [event for _, event in self._stream_queue]
            if self._active_stream is not None:
                events.append(self._active_stream[1])
            for event in events:
                event.set()
            self._condition.notify_all()
        if self._thread is not None:
            self._httpd.shutdown()
            self._thread.join(timeout=2.0)
        self._httpd.server_close()

    def _connection_id(self, connection: socket.socket) -> int:
        key = id(connection)
        with self._lock:
            prior = self._connection_ids.get(key)
            if prior is not None and prior[0] is connection:
                return prior[1]
            self._next_connection_id += 1
            self._connection_ids[key] = (connection, self._next_connection_id)
            return self._next_connection_id

    def _record(self, handler: _RequestHandler, body: bytes) -> None:
        record = RequestRecord(
            method=handler.command,
            path=handler.path,
            body=body,
            headers=tuple((key, value) for key, value in handler.headers.items()),
            connection_id=self._connection_id(handler.connection),
        )
        with self._lock:
            self._records.append(record)

    def _get_override(self, method: str, route: str) -> ResponseOverride | None:
        with self._lock:
            return self._overrides.get((method, route))

    def _begin_stream(self, owner: str) -> threading.Event:
        with self._condition:
            event = threading.Event()
            stream = (owner, event)
            self._stream_queue.append(stream)
            self._current_stream = stream
            self._condition.notify_all()
            return event

    def _acquire_stream(
        self,
        owner: str,
        event: threading.Event,
        handler: _RequestHandler | None = None,
    ) -> bool:
        while not self._stop.is_set() and not event.is_set():
            if handler is not None and handler._peer_disconnected():
                event.set()
                return False
            with self._condition:
                if self._stop.is_set() or event.is_set():
                    return False
                if (
                    self._active_stream is None
                    and self._stream_queue
                    and self._stream_queue[0][1] is event
                ):
                    self._stream_queue.pop(0)
                    self._active_stream = (owner, event)
                    self._condition.notify_all()
                    return True
                self._condition.wait(timeout=0.05)
        return False

    def _end_stream(self, owner: str, event: threading.Event) -> None:
        with self._condition:
            self._stream_queue = [
                stream for stream in self._stream_queue if stream[1] is not event
            ]
            if (
                self._active_stream is not None
                and self._active_stream[0] == owner
                and self._active_stream[1] is event
            ):
                self._active_stream = None
            if (
                self._current_stream is not None
                and self._current_stream[1] is event
            ):
                self._current_stream = None
            self._condition.notify_all()

    def _cancel_session_locked(self, session: _Session) -> None:
        if session.state not in {"closed", "cancelled", "expired"}:
            session.state = "cancelled"
            for segment in session.segments:
                if segment.state in {"queued", "streaming"}:
                    segment.state = "cancelled"
        self._condition.notify_all()

    def _cancel_session(self, session_id: str) -> tuple[_Session | None, bool]:
        with self._condition:
            session = self._sessions.get(session_id)
            if session is None:
                return None, False
            self._cancel_session_locked(session)
            matching_events = [
                event
                for owner, event in self._stream_queue
                if owner == session_id
            ]
            if (
                self._active_stream is not None
                and self._active_stream[0] == session_id
            ):
                matching_events.append(self._active_stream[1])
            active_audio_cancelled = any(not event.is_set() for event in matching_events)
            for event in matching_events:
                event.set()
            self._condition.notify_all()
            return session, active_audio_cancelled

    def _cancel_current_stream(self) -> bool:
        with self._condition:
            if self._current_stream is None:
                return False
            owner, event = self._current_stream
            cancelled = not event.is_set()
            event.set()
            session = self._sessions.get(owner)
            if session is not None:
                self._cancel_session_locked(session)
            self._condition.notify_all()
            return cancelled

    def _expression_metadata(self) -> dict[str, Any]:
        if not self.expression_enabled:
            return {
                "enabled": False,
                "default": None,
                "profiles": [],
                "policies": [],
            }
        return {
            "enabled": True,
            "default": "calm",
            "preferred_policy": "semantic-style",
            "playback_policy": {
                "short_prebuffer_ms": 32,
                "long_prebuffer_ms": 64,
            },
            "runtime_policy": {"first_context_tokens": 17},
            "profiles": [
                {
                    "id": "calm",
                    "intensity_levels": [0.5, 0.8],
                    "languages": list(CORE_LANGUAGES),
                    "reference_languages": ["en"],
                    "preferred_policies": ["semantic-style"],
                }
            ],
            "policies": [
                "full-switch",
                "identity-lock",
                "semantic-style",
                "acoustic-style",
                "sv-only",
            ],
        }

    def _health(self) -> dict[str, Any]:
        expression = self._expression_metadata()
        return {
            "service": "AnifLive-TTS",
            "version": "1.4.0",
            "status": "ok",
            "ready": True,
            "backend": "TensorRT-11",
            "model": self.active_model,
            "voice": "default",
            "cuda": None,
            "tensorrt": None,
            "gpu": None,
            "engine_fingerprint": "fixture-engine",
            "engine_count": 9,
            "engines": {
                "ssl": True,
                "bert": True,
                "vq_encoder": True,
                "gpt_encoder": True,
                "gpt_step": True,
                "sovits": True,
                "spectrogram": True,
                "sv_embedding": True,
                "sovits_stream": True,
            },
            "reference": {"configured": True, "language": "en"},
            "expression": expression,
            "request_segment_char_limit": 32,
            "streaming": {
                "enabled": True,
                "transport": "pcm_s16le",
                "complete_wav": "request stream=false",
            },
            "startup_warmup": None,
            "active_requests": 0,
            "gpu_resource_lease": {
                "enabled": False,
                "policy": "engine-residency",
                "residency_reserved": False,
                "lease_healthy": True,
            },
            "warm_retention": None,
            "switching": False,
            "available_model_count": len(self.available_models),
        }

    def _capabilities(self) -> dict[str, Any]:
        expression_enabled = self.expression_enabled
        policies = {
            "A": {
                "name": "none",
                "phones_bert": False,
                "semantic_tokens": 0,
                "expression_aware": False,
                "boundary_adaptive": False,
            },
            "B": {
                "name": "text-bert",
                "phones_bert": True,
                "semantic_tokens": 0,
                "expression_aware": False,
                "boundary_adaptive": False,
            },
            "C": {
                "name": "text-bert-semantic-32",
                "phones_bert": True,
                "semantic_tokens": 32,
                "expression_aware": False,
                "boundary_adaptive": False,
            },
            "D": {
                "name": "text-bert-semantic-64",
                "phones_bert": True,
                "semantic_tokens": 64,
                "expression_aware": False,
                "boundary_adaptive": False,
            },
            "E": {
                "name": "expression-aware",
                "phones_bert": True,
                "semantic_tokens": 64,
                "expression_aware": True,
                "boundary_adaptive": False,
            },
            "F": {
                "name": "boundary-adaptive",
                "phones_bert": True,
                "semantic_tokens": 64,
                "expression_aware": True,
                "boundary_adaptive": True,
            },
        }
        return {
            "service": "AnifLive-TTS",
            "version": "1.4.0",
            "model_family": "v2proplus",
            "backend": "TensorRT-11",
            "precision": "FP16",
            "languages": list(CORE_LANGUAGES),
            "streaming": {"pcm16": True, "wav": False},
            "expression": {
                "native": True,
                "controlled_profiles": expression_enabled,
                "continuous_vector": False,
                "segmented": expression_enabled,
            },
            "pytorch_fallback": False,
            "adaptive_punctuation_segments": True,
            "warm_retention_seconds": 25,
            "speech_sessions": {
                "committed_segments": True,
                "pcm_stream": True,
                "cancel": True,
                "flush": True,
                "context_mode": "committed-neural-v1",
                "neural_state_continuity": True,
                "acoustic_latent_continuity": False,
                "default_policy": "A",
                "policies": policies,
                "qualification": "experimental-unqualified",
                "limits": {
                    "maximum_open": 8,
                    "ttl_seconds": 600.0,
                    "maximum_segments": 64,
                    "maximum_characters": 16_000,
                },
            },
        }

    def _model_list(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "aniflive-tts-local",
                    "description": "V2 Pro Plus fixture package",
                    "model_family": "v2proplus",
                    "voice_profiles": ["default"],
                    "active": model_id == self.active_model,
                }
                for model_id in self.available_models
            ],
        }

    def _voice_list(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "default",
                    "reference_language": "en",
                    "reference_configured": True,
                }
            ],
        }

    def _create_session(self, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        model = body.get("model") or self.active_model
        voice_profile = body.get("voice_profile") or "default"
        policy = str(body.get("continuity_policy", "A")).strip().upper()
        if model != self.active_model:
            return 400, {"code": 400, "message": "model must match the active model"}
        if voice_profile != "default":
            return 400, {"code": 400, "message": "voice_profile must match the active profile"}
        if policy not in {"A", "B", "C", "D", "E", "F"}:
            return 400, {
                "code": 400,
                "message": "continuity_policy must be one of A, B, C, D, E or F",
            }
        with self._condition:
            open_count = sum(
                session.state not in {"closed", "cancelled", "expired"}
                for session in self._sessions.values()
            )
            if open_count >= 8:
                return 429, {"code": 429, "message": "Too many open speech sessions"}
            self._next_session_id += 1
            session_id = f"sess_fixture_{self._next_session_id:04d}"
            session = _Session(
                session_id=session_id,
                model=str(model),
                voice_profile=str(voice_profile),
                continuity_policy=policy,
            )
            self._sessions[session_id] = session
            self._condition.notify_all()
            return 201, session.public_dict()

    @staticmethod
    def _segment_fingerprint(body: Mapping[str, Any]) -> str:
        payload = {
            "text": body.get("text"),
            "language": body.get("language"),
            "expression": body.get("expression"),
            "generation": body.get("generation"),
            "paragraph_id": body.get("paragraph_id"),
            "pause_after_ms": body.get("pause_after_ms", 0),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _append_segment(
        self, session_id: str, body: Mapping[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        with self._condition:
            session = self._sessions.get(session_id)
            if session is None:
                return 404, {"code": 404, "message": "Speech session was not found"}
            segment_id = body.get("segment_id")
            text = body.get("text")
            language = body.get("language")
            if not isinstance(segment_id, str) or not segment_id:
                return 400, {"code": 400, "message": "Missing required parameter: segment_id"}
            if not isinstance(text, str) or not text.strip():
                return 400, {"code": 400, "message": "Missing required parameter: text"}
            if not isinstance(language, str) or language not in CORE_LANGUAGES:
                return 400, {"code": 400, "message": "Unsupported language"}
            try:
                fingerprint = self._segment_fingerprint(body)
            except (TypeError, ValueError):
                return 400, {"code": 400, "message": "Invalid segment JSON"}
            for existing in session.segments:
                if existing.segment_id != segment_id:
                    continue
                if existing.fingerprint != fingerprint:
                    return 409, {
                        "code": 409,
                        "message": "segment_id was already used with different content",
                    }
                return 200, existing.public_dict()
            if session.state != "open":
                return 409, {
                    "code": 409,
                    "message": "Speech session no longer accepts segments",
                }
            if len(session.segments) >= 64:
                return 413, {
                    "code": 413,
                    "message": "Speech session is limited to 64 segments",
                }
            if sum(len(item.text) for item in session.segments) + len(text) > 16_000:
                return 413, {
                    "code": 413,
                    "message": "Speech session committed text exceeds its character limit",
                }
            pause = body.get("pause_after_ms", 0)
            if isinstance(pause, bool) or not isinstance(pause, int) or not 0 <= pause <= 10_000:
                return 400, {"code": 400, "message": "pause_after_ms must be an integer"}
            segment = _Segment(
                segment_id=segment_id,
                sequence=len(session.segments),
                text=text,
                language=language,
                fingerprint=fingerprint,
                paragraph_id=body.get("paragraph_id"),
                pause_after_ms=pause,
            )
            session.segments.append(segment)
            self._condition.notify_all()
            return 201, segment.public_dict()

    def _session_chunks(
        self, session: _Session, handler: _RequestHandler
    ) -> Iterable[bytes]:
        active_segment: _Segment | None = None
        active_event: threading.Event | None = None
        registered = False
        try:
            while not self._stop.is_set():
                if handler._peer_disconnected():
                    return
                with self._condition:
                    if session.state in {"cancelled", "expired", "closed"}:
                        return
                    if session.next_stream_index < len(session.segments):
                        active_segment = session.segments[session.next_stream_index]
                        session.next_stream_index += 1
                        active_segment.state = "streaming"
                    elif session.state == "flushing":
                        session.state = "closed"
                        self._condition.notify_all()
                        return
                    else:
                        self._condition.wait(timeout=0.05)
                        continue

                active_event = self._begin_stream(session.session_id)
                registered = True
                if not self._acquire_stream(
                    session.session_id, active_event, handler
                ):
                    with self._condition:
                        if active_segment.state == "streaming":
                            active_segment.state = "cancelled"
                            if session.state not in {"closed", "expired"}:
                                self._cancel_session_locked(session)
                    self._end_stream(session.session_id, active_event)
                    active_event = None
                    active_segment = None
                    registered = False
                    return

                emitted_bytes = 0
                try:
                    for chunk in self.pcm_chunks:
                        if (
                            self._stop.is_set()
                            or active_event.is_set()
                            or handler._peer_disconnected()
                        ):
                            break
                        yield chunk
                        emitted_bytes += len(chunk)
                        if self.stream_delay_seconds:
                            active_event.wait(self.stream_delay_seconds)
                    with self._condition:
                        if (
                            active_event.is_set()
                            or session.state in {"cancelled", "expired"}
                            or self._stop.is_set()
                            or handler._peer_disconnected()
                        ):
                            active_segment.state = "cancelled"
                            if session.state not in {"closed", "expired"}:
                                self._cancel_session_locked(session)
                        else:
                            active_segment.state = "completed"
                            session.previous_language = active_segment.language
                            session.previous_text = active_segment.text
                            session.paragraph_id = active_segment.paragraph_id
                            session.previous_pause_ms = active_segment.pause_after_ms
                            session.previous_output_samples = emitted_bytes // 2
                            session.total_output_samples += session.previous_output_samples
                            session.last_plan = {
                                "configured_policy": session.continuity_policy,
                                "effective_policy": session.continuity_policy,
                                "reason": "fixture",
                                "phones": 0,
                                "semantic_tokens": 0,
                                "acoustic_latent": False,
                            }
                        self._condition.notify_all()
                except GeneratorExit:
                    with self._condition:
                        if active_segment.state == "streaming":
                            active_segment.state = "cancelled"
                            self._cancel_session_locked(session)
                    raise
                finally:
                    if registered and active_event is not None:
                        self._end_stream(session.session_id, active_event)
                    active_event = None
                    active_segment = None
                    registered = False
        finally:
            with self._condition:
                session.audio_consumer_attached = False
                self._condition.notify_all()

    def _dispatch(
        self,
        handler: _RequestHandler,
        method: str,
        route: str,
        body: Mapping[str, Any],
    ) -> None:
        if method == "GET" and route == "/health":
            handler._send_json(200, self._health())
            return
        if method == "GET" and route == "/v1/capabilities":
            handler._send_json(200, self._capabilities())
            return
        if method == "GET" and route == "/v1/models":
            handler._send_json(200, self._model_list())
            return
        if method == "GET" and route == "/v1/voices":
            handler._send_json(200, self._voice_list())
            return
        if method == "GET" and route == "/v1/expressions":
            handler._send_json(
                200, {"object": "list", **self._expression_metadata()}
            )
            return
        if method == "POST" and route == "/v1/models/activate":
            requested = body.get("model")
            if not isinstance(requested, str) or not requested:
                handler._send_json(
                    400, {"code": 400, "message": "Missing required parameter: model"}
                )
                return
            with self._condition:
                if any(
                    session.state not in {"closed", "cancelled", "expired"}
                    for session in self._sessions.values()
                ):
                    handler._send_json(
                        409,
                        {"code": 409, "message": "Cannot switch models while a speech session is open"},
                    )
                    return
                if requested not in self.available_models:
                    handler._send_json(400, {"code": 400, "message": "Unknown local model"})
                    return
                changed = requested != self.active_model
                self.active_model = requested
                handler._send_json(
                    200,
                    {
                        "code": 0,
                        "message": "Success",
                        "changed": changed,
                        "model": self.active_model,
                        "voice": "default",
                        "backend": "TensorRT-11",
                        "pytorch_fallback": False,
                    },
                )
            return
        if method == "POST" and route == "/v1/audio/speech":
            self._dispatch_speech(handler, body)
            return
        if method == "POST" and route == "/v1/audio/cancel":
            cancelled = self._cancel_current_stream()
            handler._send_json(
                200, {"cancelled": cancelled, "model": self.active_model}
            )
            return
        if method == "POST" and route == "/v1/sessions":
            status, payload = self._create_session(body)
            handler._send_json(status, payload)
            return

        parts = [unquote(part) for part in route.split("/") if part]
        if len(parts) >= 3 and parts[:2] == ["v1", "sessions"]:
            session_id = parts[2]
            if method == "GET" and len(parts) == 3:
                with self._condition:
                    session = self._sessions.get(session_id)
                    if session is None:
                        handler._send_json(
                            404, {"code": 404, "message": "Speech session was not found"}
                        )
                    else:
                        handler._send_json(200, session.public_dict())
                return
            if method == "POST" and len(parts) == 4 and parts[3] == "segments":
                status, payload = self._append_segment(session_id, body)
                handler._send_json(status, payload)
                return
            if method == "GET" and len(parts) == 4 and parts[3] == "audio":
                self._dispatch_session_audio(handler, session_id)
                return
            if method == "POST" and len(parts) == 4 and parts[3] == "flush":
                with self._condition:
                    session = self._sessions.get(session_id)
                    if session is None:
                        handler._send_json(
                            404, {"code": 404, "message": "Speech session was not found"}
                        )
                    else:
                        if session.state == "open":
                            session.state = "flushing"
                            self._condition.notify_all()
                        handler._send_json(200, session.public_dict())
                return
            if method == "POST" and len(parts) == 4 and parts[3] == "cancel":
                self._dispatch_session_cancel(handler, session_id)
                return
            if method == "DELETE" and len(parts) == 3:
                self._dispatch_session_cancel(handler, session_id)
                return

        handler._send_json(404, {"code": 404, "message": "Route was not found"})

    def _dispatch_speech(
        self, handler: _RequestHandler, body: Mapping[str, Any]
    ) -> None:
        if body.get("model") not in (None, self.active_model):
            handler._send_json(
                400, {"code": 400, "message": "model must match the active model"}
            )
            return
        if body.get("voice_profile") not in (None, "default"):
            handler._send_json(
                400, {"code": 400, "message": "voice_profile must match the active profile"}
            )
            return
        streaming = body.get("stream", False)
        if not isinstance(streaming, bool):
            handler._send_json(400, {"code": 400, "message": "stream must be a boolean"})
            return
        response_format = str(
            body.get("response_format", "pcm" if streaming else "wav")
        ).lower()
        if streaming and response_format not in {"pcm", "pcm_s16le", "s16le"}:
            handler._send_json(
                400, {"code": 400, "message": "stream=true requires response_format='pcm'"}
            )
            return
        if not streaming and response_format != "wav":
            handler._send_json(
                400, {"code": 400, "message": "response_format must be 'wav' when stream=false"}
            )
            return
        if not streaming:
            handler._send_bytes(
                200,
                self.wav_bytes,
                {
                    "Content-Type": "audio/wav",
                    "X-TTS-Service": "AnifLive-TTS",
                    "X-TTS-Version": "1.4.0",
                    "X-TTS-Model": self.active_model,
                    "X-TTS-Sample-Rate": str(SAMPLE_RATE),
                    "X-TTS-Output-Samples": "4",
                    "X-TTS-Segments": "1",
                    "X-TTS-Seed": str(body.get("generation", {}).get("seed", -1)),
                },
            )
            return
        event = self._begin_stream("http:fixture")
        headers = {
            "Content-Type": "application/octet-stream",
            "X-TTS-Service": "AnifLive-TTS",
            "X-TTS-Version": "1.4.0",
            "X-TensorRT-Backend": "TensorRT-11",
            "X-TensorRT-Engine-Count": "9",
            "X-PyTorch-Fallback": "false",
            "X-TTS-Model": self.active_model,
            "X-TTS-Stream": "pcm_s16le",
            "X-TTS-Sample-Format": "s16le",
            "X-TTS-Sample-Rate": str(SAMPLE_RATE),
            "X-TTS-Channels": "1",
            "X-TTS-Recommended-Prebuffer-Ms": "32",
            "X-TTS-Queue-Ms": "0.000",
            "X-TTS-Pause-Mode": "adaptive",
            "X-TTS-Expression": "native",
            "X-TTS-Expression-Policy": "full-switch",
            "X-TTS-First-Context-Tokens": "17",
            "Cache-Control": "no-store",
        }
        try:
            handler._send_chunked(
                200,
                headers,
                self._direct_chunks(event, handler),
                delay_seconds=self.stream_delay_seconds,
                cancel_event=event,
            )
        finally:
            self._end_stream("http:fixture", event)

    def _direct_chunks(
        self, event: threading.Event, handler: _RequestHandler
    ) -> Iterable[bytes]:
        if not self._acquire_stream("http:fixture", event, handler):
            return
        try:
            for chunk in self.pcm_chunks:
                if (
                    self._stop.is_set()
                    or event.is_set()
                    or handler._peer_disconnected()
                ):
                    return
                yield chunk
        finally:
            self._end_stream("http:fixture", event)

    def _dispatch_session_audio(
        self, handler: _RequestHandler, session_id: str
    ) -> None:
        with self._condition:
            session = self._sessions.get(session_id)
            if session is None:
                handler._send_json(
                    404, {"code": 404, "message": "Speech session was not found"}
                )
                return
            if session.audio_consumer_attached:
                handler._send_json(
                    409, {"code": 409, "message": "Speech session already has an audio consumer"}
                )
                return
            if session.state in {"closed", "cancelled", "expired"}:
                handler._send_json(
                    409, {"code": 409, "message": "Speech session is no longer streamable"}
                )
                return
            session.audio_consumer_attached = True

        headers = {
            "Content-Type": "application/octet-stream",
            "X-TTS-Service": "AnifLive-TTS",
            "X-TTS-Version": "1.4.0",
            "X-TensorRT-Backend": "TensorRT-11",
            "X-TensorRT-Engine-Count": "9",
            "X-PyTorch-Fallback": "false",
            "X-TTS-Model": session.model,
            "X-TTS-Stream": "pcm_s16le",
            "X-TTS-Sample-Format": "s16le",
            "X-TTS-Sample-Rate": str(SAMPLE_RATE),
            "X-TTS-Channels": "1",
            "X-TTS-Session-ID": session.session_id,
            "X-TTS-Session-Context": "committed-neural-v1",
            "X-TTS-Session-Context-Policy": session.continuity_policy,
            "X-TTS-Neural-State-Continuity": "false",
            "X-TTS-Acoustic-Latent-Continuity": "false",
            "X-TTS-Continuity-Qualification": "experimental-unqualified",
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        }
        try:
            handler._send_chunked(
                200,
                headers,
                self._session_chunks(session, handler),
            )
        finally:
            with self._condition:
                session.audio_consumer_attached = False
                self._condition.notify_all()

    def _dispatch_session_cancel(
        self, handler: _RequestHandler, session_id: str
    ) -> None:
        session, active_audio_cancelled = self._cancel_session(session_id)
        if session is None:
            handler._send_json(
                404, {"code": 404, "message": "Speech session was not found"}
            )
            return
        handler._send_json(
            200,
            {
                **session.public_dict(),
                "active_audio_cancelled": active_audio_cancelled,
            },
        )


class _FixtureHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False

    def __init__(self, address: tuple[str, int], fixture: FixtureServer) -> None:
        self.fixture = fixture
        super().__init__(address, _RequestHandler)


class _RequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "AnifLive-TTS-Fixture/1.0"
    sys_version = ""

    @property
    def _fixture(self) -> FixtureServer:
        return self.server.fixture  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _peer_disconnected(self) -> bool:
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(
                readable and self.connection.recv(1, socket.MSG_PEEK) == b""
            )
        except (OSError, ValueError):
            return True

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()

    def _read_body(self) -> bytes:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return b""
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("Content-Length must be a non-negative integer") from error
        if length < 0:
            raise ValueError("Content-Length must be a non-negative integer")
        if length > 65_536:
            raise OverflowError("Request body is limited to 65536 bytes")
        return self.rfile.read(length)

    @staticmethod
    def _parse_json(body: bytes) -> Mapping[str, Any]:
        if not body:
            return {}
        try:
            value = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Request body must be valid JSON") from error
        if not isinstance(value, dict):
            raise TypeError("Request body must be a JSON object")
        return value

    def _handle(self) -> None:
        try:
            body = self._read_body()
        except OverflowError as error:
            self.close_connection = True
            self._fixture._record(self, b"")
            self._send_json(413, {"code": 413, "message": str(error)})
            return
        except ValueError as error:
            self._fixture._record(self, b"")
            self._send_json(400, {"code": 400, "message": str(error)})
            return

        self._fixture._record(self, body)
        route = urlsplit(self.path).path
        override = self._fixture._get_override(self.command, route)
        if override is not None:
            self._send_override(override)
            return

        if self.command == "POST" and route != "/v1/audio/cancel":
            try:
                parsed = self._parse_json(body)
            except (TypeError, ValueError) as error:
                self._send_json(400, {"code": 400, "message": str(error)})
                return
        else:
            parsed = {}
        self._fixture._dispatch(self, self.command, route, parsed)

    def _send_json(
        self,
        status: int,
        payload: Any,
        headers: Mapping[str, str | None] | None = None,
    ) -> None:
        response_headers = dict(headers or {})
        if status == 429 and not any(key.lower() == "retry-after" for key in response_headers):
            response_headers["Retry-After"] = "1"
        response_headers.setdefault("Content-Type", "application/json")
        self._send_bytes(status, _json_bytes(payload), response_headers)

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        headers: Mapping[str, str | None] | None = None,
    ) -> None:
        response_headers = dict(headers or {})
        if not any(key.lower() == "content-length" for key in response_headers):
            response_headers["Content-Length"] = str(len(body))
        if not any(key.lower() == "content-type" for key in response_headers):
            response_headers["Content-Type"] = "application/octet-stream"
        if status == 429 and not any(key.lower() == "retry-after" for key in response_headers):
            response_headers["Retry-After"] = "1"
        self.send_response(status)
        for name, value in response_headers.items():
            if value is not None:
                self.send_header(name, str(value))
            elif name.lower() == "content-length":
                self.close_connection = True
        self.end_headers()
        if body:
            try:
                self.wfile.write(body)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                self.close_connection = True

    def _send_chunked(
        self,
        status: int,
        headers: Mapping[str, str | None],
        chunks: Iterable[bytes],
        *,
        delay_seconds: float = 0.0,
        cancel_event: threading.Event | None = None,
    ) -> None:
        response_headers = dict(headers)
        if not any(key.lower() == "content-type" for key in response_headers):
            response_headers["Content-Type"] = "application/octet-stream"
        response_headers["Transfer-Encoding"] = "chunked"
        self.send_response(status)
        for name, value in response_headers.items():
            if value is not None:
                self.send_header(name, str(value))
        self.end_headers()
        iterator = iter(chunks)
        try:
            for chunk in iterator:
                if not chunk:
                    continue
                if self._fixture._stop.is_set():
                    break
                if cancel_event is not None and cancel_event.is_set():
                    break
                self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                if delay_seconds:
                    if cancel_event is not None:
                        cancel_event.wait(delay_seconds)
                    else:
                        self._fixture._stop.wait(delay_seconds)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def _send_override(self, override: ResponseOverride) -> None:
        if override.chunks is not None:
            headers = dict(override.headers)
            if not any(key.lower() == "content-type" for key in headers):
                headers["Content-Type"] = "application/octet-stream"
            self._send_chunked(
                override.status,
                headers,
                override.chunks,
                delay_seconds=override.delay_seconds,
            )
            return
        body = override.body
        headers = dict(override.headers)
        if isinstance(body, (Mapping, list)):
            encoded = _json_bytes(body)
            if not any(key.lower() == "content-type" for key in headers):
                headers["Content-Type"] = "application/json"
        elif isinstance(body, bytes):
            encoded = body
        elif body is None and override.status >= 400:
            encoded = _json_bytes(
                {"code": override.status, "message": "fixture override"}
            )
            if not any(key.lower() == "content-type" for key in headers):
                headers["Content-Type"] = "application/json"
        else:
            encoded = b""
        self._send_bytes(override.status, encoded, headers)


__all__ = [
    "ALTERNATE_MODEL",
    "CORE_LANGUAGES",
    "DEFAULT_MODEL",
    "PCM_CHUNKS",
    "SAMPLE_RATE",
    "FixtureServer",
    "RequestRecord",
]
