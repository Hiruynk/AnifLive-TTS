from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import time
from typing import Any, Mapping
from uuid import uuid4

from .continuity import (
    ContinuityPolicy,
    NeuralContinuationState,
    SpeechContextState,
    parse_continuity_policy,
)


SESSION_STATES = frozenset({"open", "flushing", "closed", "cancelled", "expired"})
TERMINAL_SESSION_STATES = frozenset({"closed", "cancelled", "expired"})
SEGMENT_STATES = frozenset({"queued", "streaming", "completed", "cancelled", "failed"})


class SpeechSessionError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def payload_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class CommittedSpeechSegment:
    segment_id: str
    sequence: int
    text: str
    language: str
    options: Any
    fingerprint: str
    paragraph_id: str | None = None
    pause_after_ms: int = 0
    state: str = "queued"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
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
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }


@dataclass
class SpeechSession:
    session_id: str
    model: str
    voice_profile: str
    sample_rate: int
    continuity_policy: ContinuityPolicy = ContinuityPolicy.NONE
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    state: str = "open"
    segments: list[CommittedSpeechSegment] = field(default_factory=list)
    audio_consumer_attached: bool = False
    next_stream_index: int = 0
    last_activity_monotonic: float = field(default_factory=time.monotonic)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition, repr=False)
    context: SpeechContextState = field(default_factory=SpeechContextState)

    def __post_init__(self) -> None:
        self.continuity_policy = parse_continuity_policy(self.continuity_policy)
        self.context.bind(
            model_id=self.model,
            voice_profile=self.voice_profile,
            policy=self.continuity_policy,
        )

    def touch(self) -> None:
        self.updated_at = utc_now()
        self.last_activity_monotonic = time.monotonic()

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.session_id,
            "object": "speech.session",
            "state": self.state,
            "model": self.model,
            "voice_profile": self.voice_profile,
            "sample_rate": self.sample_rate,
            "format": "pcm_s16le",
            "continuity_policy": self.continuity_policy.value,
            "context": self.context.public_dict(),
            "audio_consumer_attached": self.audio_consumer_attached,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "segments": [segment.public_dict() for segment in self.segments],
        }


class SpeechSessionManager:
    def __init__(
        self,
        *,
        maximum_open: int = 8,
        ttl_seconds: float = 600.0,
        maximum_segments: int = 64,
        maximum_characters: int = 16_000,
        maximum_retained: int = 64,
    ) -> None:
        if maximum_open < 1:
            raise ValueError("maximum_open must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if maximum_segments < 1:
            raise ValueError("maximum_segments must be positive")
        if maximum_characters < 1:
            raise ValueError("maximum_characters must be positive")
        if maximum_retained < maximum_open:
            raise ValueError("maximum_retained must be at least maximum_open")
        self.maximum_open = maximum_open
        self.ttl_seconds = ttl_seconds
        self.maximum_segments = maximum_segments
        self.maximum_characters = maximum_characters
        self.maximum_retained = maximum_retained
        self._sessions: dict[str, SpeechSession] = {}
        self._lock = asyncio.Lock()

    def limits_dict(self) -> dict[str, int | float]:
        return {
            "maximum_open": self.maximum_open,
            "ttl_seconds": self.ttl_seconds,
            "maximum_segments": self.maximum_segments,
            "maximum_characters": self.maximum_characters,
        }

    @staticmethod
    def _cancel_pending_locked(session: SpeechSession) -> None:
        for segment in session.segments:
            if segment.state in {"queued", "streaming"}:
                segment.state = "cancelled"
                segment.updated_at = utc_now()

    def _expire_session_locked(self, session: SpeechSession) -> None:
        session.state = "expired"
        self._cancel_pending_locked(session)
        session.context.clear()
        session.updated_at = utc_now()
        session.condition.notify_all()

    def _prune_terminal_locked(self) -> None:
        overflow = len(self._sessions) - self.maximum_retained
        if overflow <= 0:
            return
        candidates = sorted(
            (
                session
                for session in self._sessions.values()
                if session.state in TERMINAL_SESSION_STATES
                and not session.audio_consumer_attached
            ),
            key=lambda session: session.last_activity_monotonic,
        )
        for session in candidates[:overflow]:
            session.context.clear_neural()
            self._sessions.pop(session.session_id, None)

    async def _expire_locked(self) -> None:
        now = time.monotonic()
        for session in self._sessions.values():
            if session.state in TERMINAL_SESSION_STATES:
                continue
            if any(segment.state == "streaming" for segment in session.segments):
                continue
            if now - session.last_activity_monotonic > self.ttl_seconds:
                async with session.condition:
                    self._expire_session_locked(session)

    async def create(
        self,
        *,
        model: str,
        voice_profile: str,
        sample_rate: int,
        continuity_policy: ContinuityPolicy | str = ContinuityPolicy.NONE,
    ) -> SpeechSession:
        async with self._lock:
            await self._expire_locked()
            self._prune_terminal_locked()
            open_count = sum(
                session.state not in TERMINAL_SESSION_STATES
                for session in self._sessions.values()
            )
            if open_count >= self.maximum_open:
                raise SpeechSessionError("Too many open speech sessions", 429)
            session = SpeechSession(
                session_id=f"sess_{uuid4().hex}",
                model=model,
                voice_profile=voice_profile,
                sample_rate=sample_rate,
                continuity_policy=parse_continuity_policy(continuity_policy),
            )
            self._sessions[session.session_id] = session
            self._prune_terminal_locked()
            return session

    async def get(self, session_id: str) -> SpeechSession:
        async with self._lock:
            await self._expire_locked()
            session = self._sessions.get(session_id)
            if session is None:
                raise SpeechSessionError("Speech session was not found", 404)
            return session

    async def has_active(self) -> bool:
        async with self._lock:
            await self._expire_locked()
            return any(
                session.state not in TERMINAL_SESSION_STATES
                for session in self._sessions.values()
            )

    async def append(
        self,
        session_id: str,
        *,
        segment_id: str,
        text: str,
        language: str,
        options: Any,
        fingerprint_payload: Mapping[str, Any],
        paragraph_id: str | None = None,
        pause_after_ms: int = 0,
    ) -> tuple[CommittedSpeechSegment, bool]:
        session = await self.get(session_id)
        fingerprint = payload_fingerprint(fingerprint_payload)
        async with session.condition:
            for existing in session.segments:
                if existing.segment_id != segment_id:
                    continue
                if existing.fingerprint != fingerprint:
                    raise SpeechSessionError(
                        "segment_id was already used with different content", 409
                    )
                return existing, False
            if session.state != "open":
                raise SpeechSessionError("Speech session no longer accepts segments", 409)
            if len(session.segments) >= self.maximum_segments:
                raise SpeechSessionError(
                    f"Speech session is limited to {self.maximum_segments} segments",
                    413,
                )
            total_characters = sum(len(segment.text) for segment in session.segments)
            if total_characters + len(text) > self.maximum_characters:
                raise SpeechSessionError(
                    "Speech session committed text exceeds its character limit",
                    413,
                )
            record = CommittedSpeechSegment(
                segment_id=segment_id,
                sequence=len(session.segments),
                text=text,
                language=language,
                options=options,
                fingerprint=fingerprint,
                paragraph_id=paragraph_id,
                pause_after_ms=pause_after_ms,
            )
            session.segments.append(record)
            session.touch()
            session.condition.notify_all()
            return record, True

    async def attach_consumer(self, session_id: str) -> SpeechSession:
        session = await self.get(session_id)
        async with session.condition:
            if session.audio_consumer_attached:
                raise SpeechSessionError("Speech session already has an audio consumer", 409)
            if session.state in TERMINAL_SESSION_STATES:
                raise SpeechSessionError("Speech session is no longer streamable", 409)
            session.audio_consumer_attached = True
            session.touch()
            return session

    async def detach_consumer(self, session: SpeechSession) -> None:
        async with session.condition:
            session.audio_consumer_attached = False
            session.touch()
            session.condition.notify_all()

    async def next_segment(self, session: SpeechSession) -> CommittedSpeechSegment | None:
        async with session.condition:
            while True:
                if session.state in {"cancelled", "expired"}:
                    return None
                if session.next_stream_index < len(session.segments):
                    segment = session.segments[session.next_stream_index]
                    session.next_stream_index += 1
                    segment.state = "streaming"
                    segment.updated_at = utc_now()
                    session.touch()
                    return segment
                if session.state == "flushing":
                    session.state = "closed"
                    session.context.clear_neural()
                    session.touch()
                    return None
                remaining = self.ttl_seconds - (
                    time.monotonic() - session.last_activity_monotonic
                )
                if remaining <= 0:
                    self._expire_session_locked(session)
                    return None
                try:
                    await asyncio.wait_for(session.condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    if time.monotonic() - session.last_activity_monotonic >= self.ttl_seconds:
                        self._expire_session_locked(session)
                        return None

    async def finish_segment(
        self,
        session: SpeechSession,
        segment: CommittedSpeechSegment,
        *,
        error: str | None = None,
        output_samples: int = 0,
        neural_state: NeuralContinuationState | None = None,
    ) -> None:
        async with session.condition:
            if segment.state != "streaming":
                return
            if session.state in {"cancelled", "expired"}:
                segment.state = "cancelled"
                segment.updated_at = utc_now()
                session.condition.notify_all()
                return
            if error is not None:
                segment.state = "failed"
                segment.error = error
                segment.updated_at = utc_now()
                session.state = "cancelled"
                session.context.clear()
                self._cancel_pending_locked(session)
            else:
                expression = None
                if getattr(segment.options, "expression_enabled", False):
                    expression = getattr(segment.options, "expression_profile", None)
                if expression is None:
                    controls = getattr(segment.options, "expression_segments", ())
                    expression = next(
                        (
                            getattr(control, "profile", None)
                            for control in reversed(controls)
                            if getattr(control, "enabled", False)
                        ),
                        None,
                    )
                try:
                    session.context.commit(
                        text=segment.text,
                        language=segment.language,
                        expression=expression,
                        paragraph_id=segment.paragraph_id,
                        pause_after_ms=segment.pause_after_ms,
                        output_samples=output_samples,
                        neural_state=neural_state,
                        require_neural_state=(
                            session.continuity_policy is not ContinuityPolicy.NONE
                        ),
                    )
                except Exception:
                    segment.state = "failed"
                    segment.error = "continuity state could not be committed"
                    segment.updated_at = utc_now()
                    session.state = "cancelled"
                    session.context.clear()
                    self._cancel_pending_locked(session)
                    session.touch()
                    session.condition.notify_all()
                    raise
                segment.state = "completed"
                segment.error = None
                segment.updated_at = utc_now()
            session.touch()
            session.condition.notify_all()

    async def flush(self, session_id: str) -> SpeechSession:
        session = await self.get(session_id)
        async with session.condition:
            if session.state == "open":
                session.state = "flushing"
                session.touch()
                session.condition.notify_all()
            return session

    async def cancel(self, session_id: str) -> SpeechSession:
        session = await self.get(session_id)
        async with session.condition:
            if session.state not in TERMINAL_SESSION_STATES:
                session.state = "cancelled"
                session.context.clear()
                self._cancel_pending_locked(session)
                session.touch()
                session.condition.notify_all()
            return session

    async def clear_all_neural_contexts(self) -> int:
        """Release retained GPU tensors before shutdown or model activation."""

        async with self._lock:
            cleared = 0
            for session in self._sessions.values():
                if session.context.neural is not None:
                    cleared += 1
                session.context.clear_neural()
            return cleared


__all__ = [
    "CommittedSpeechSegment",
    "SESSION_STATES",
    "SEGMENT_STATES",
    "SpeechSession",
    "SpeechSessionError",
    "SpeechSessionManager",
]
