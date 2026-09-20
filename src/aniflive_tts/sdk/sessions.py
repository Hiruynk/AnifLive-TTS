from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .._native import NativeSpeechSession
else:
    NativeSpeechSession = Any

from .streams import AsyncAudioStream, AudioStream
from .types import (
    Expression,
    Generation,
    Language,
    SegmentRequest,
    SegmentInfo,
    SessionInfo,
)


def _segment_request(
    request: SegmentRequest | None,
    *,
    segment_id: str | None,
    text: str | None,
    language: Language | str | None,
    paragraph_id: str | None,
    pause_after_ms: int | None,
    expression: Expression | None,
    generation: Generation | None,
) -> SegmentRequest:
    if request is not None:
        if not isinstance(request, SegmentRequest):
            raise TypeError("request must be a SegmentRequest")
        if any(
            value is not None
            for value in (
                segment_id,
                text,
                language,
                paragraph_id,
                pause_after_ms,
                expression,
                generation,
            )
        ):
            raise TypeError("pass either a SegmentRequest or segment keyword fields")
        return request

    if segment_id is None or text is None or language is None:
        raise TypeError("segment_id, text, and language are required")
    return SegmentRequest(
        segment_id=segment_id,
        text=text,
        language=language,
        paragraph_id=paragraph_id,
        pause_after_ms=pause_after_ms,
        expression=expression,
        generation=generation,
    )


class SpeechSession:
    """Synchronous handle for one server-side speech session."""

    def __init__(self, native: NativeSpeechSession, *, owner: Any = None) -> None:
        self._native = native
        self._owner = owner

    @property
    def id(self) -> str:
        return self._native.id

    @property
    def initial_info(self) -> SessionInfo:
        """The immutable snapshot returned when this handle was created."""
        return self._native.initial_info

    def get(self) -> SessionInfo:
        return self._native.get()

    def refresh(self) -> SessionInfo:
        """Fetch and return current server state for this session."""
        return self.get()

    def append(
        self,
        request: SegmentRequest | None = None,
        *,
        segment_id: str | None = None,
        text: str | None = None,
        language: Language | str | None = None,
        paragraph_id: str | None = None,
        pause_after_ms: int | None = None,
        expression: Expression | None = None,
        generation: Generation | None = None,
    ) -> SegmentInfo:
        segment = _segment_request(
            request,
            segment_id=segment_id,
            text=text,
            language=language,
            paragraph_id=paragraph_id,
            pause_after_ms=pause_after_ms,
            expression=expression,
            generation=generation,
        )
        return self._native.append(segment)

    def audio(self) -> AudioStream:
        return AudioStream(self._native.audio(), owner=self)

    def flush(self) -> SessionInfo:
        return self._native.flush()

    def cancel(self) -> SessionInfo:
        return self._native.cancel()

    def close(self) -> SessionInfo:
        """Close the remote session (the service currently aliases DELETE to cancel)."""
        return self._native.close()

    def __enter__(self) -> SpeechSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return "SpeechSession(<native session>)"


class AsyncSpeechSession:
    """Async handle for one server-side speech session."""

    def __init__(
        self,
        native: NativeSpeechSession,
        *,
        loop_guard: Callable[[], None] | None = None,
    ) -> None:
        self._native = native
        self._loop_guard = loop_guard

    def _check_loop(self) -> None:
        if self._loop_guard is not None:
            self._loop_guard()

    @property
    def id(self) -> str:
        return self._native.id

    @property
    def initial_info(self) -> SessionInfo:
        """The immutable snapshot returned when this handle was created."""
        return self._native.initial_info

    async def get(self) -> SessionInfo:
        self._check_loop()
        return await self._native.get_async()

    async def refresh(self) -> SessionInfo:
        """Fetch and return current server state for this session."""
        return await self.get()

    async def append(
        self,
        request: SegmentRequest | None = None,
        *,
        segment_id: str | None = None,
        text: str | None = None,
        language: Language | str | None = None,
        paragraph_id: str | None = None,
        pause_after_ms: int | None = None,
        expression: Expression | None = None,
        generation: Generation | None = None,
    ) -> SegmentInfo:
        self._check_loop()
        segment = _segment_request(
            request,
            segment_id=segment_id,
            text=text,
            language=language,
            paragraph_id=paragraph_id,
            pause_after_ms=pause_after_ms,
            expression=expression,
            generation=generation,
        )
        return await self._native.append_async(segment)

    async def audio(self) -> AsyncAudioStream:
        self._check_loop()
        native_stream = await self._native.audio_async()
        return AsyncAudioStream(native_stream, loop_guard=self._loop_guard, owner=self)

    async def flush(self) -> SessionInfo:
        self._check_loop()
        return await self._native.flush_async()

    async def cancel(self) -> SessionInfo:
        self._check_loop()
        return await self._native.cancel_async()

    async def close(self) -> SessionInfo:
        """Close the remote session (the service currently aliases DELETE to cancel)."""
        self._check_loop()
        return await self._native.close_async()

    async def aclose(self) -> SessionInfo:
        return await self.close()

    async def __aenter__(self) -> AsyncSpeechSession:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def __repr__(self) -> str:
        return "AsyncSpeechSession(<native session>)"


__all__ = ["AsyncSpeechSession", "SpeechSession"]
