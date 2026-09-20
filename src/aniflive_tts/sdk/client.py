from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .._native import NativeClient
else:
    NativeClient = Any

from .sessions import AsyncSpeechSession, SpeechSession
from .streams import AsyncAudioStream, AudioStream
from .types import (
    ActivationResult,
    CancellationResult,
    Capabilities,
    ContinuityPolicy,
    Expression,
    ExpressionCatalogInfo,
    ExpressionSegment,
    Generation,
    Health,
    Language,
    ListResponse,
    ModelInfo,
    RequiredFeature,
    ServiceInfo,
    SessionCreate,
    SynthesisRequest,
    SynthesisResult,
    VoiceInfo,
)

DEFAULT_MAX_WAV_BYTES = 64 * 1024 * 1024
DEFAULT_CHUNK_BYTES = 16 * 1024


def _native_client(
    base_url: str,
    *,
    connect_timeout: float,
    request_timeout: float | None,
    stream_idle_timeout: float | None,
    headers: Mapping[str, str] | None,
    max_wav_bytes: int,
    chunk_bytes: int,
) -> NativeClient:
    if not isinstance(base_url, str):
        raise TypeError("base_url must be a string")
    native_headers: dict[str, str] | None = None
    if headers is not None:
        if not isinstance(headers, Mapping):
            raise TypeError("headers must be a mapping of strings")
        native_headers = dict(headers)
        if not all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in native_headers.items()
        ):
            raise TypeError("headers must contain only string names and values")

    try:
        native_module = import_module("aniflive_tts._native")
    except ImportError as exc:
        raise ImportError(
            "The AnifLive-TTS native extension is unavailable; reinstall the SDK wheel."
        ) from exc

    return cast(NativeClient, native_module.NativeClient(
        base_url,
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
        stream_idle_timeout=stream_idle_timeout,
        headers=native_headers,
        max_wav_bytes=max_wav_bytes,
        chunk_bytes=chunk_bytes,
    ))


def _required_feature_values(
    required: Iterable[RequiredFeature | str],
) -> tuple[str, ...]:
    if isinstance(required, (str, RequiredFeature)):
        required = (required,)
    values: list[str] = []
    for item in required:
        try:
            values.append(RequiredFeature(item).value)
        except (TypeError, ValueError) as exc:
            raise ValueError("required must contain supported RequiredFeature values") from exc
    return tuple(values)


def _synthesis_request(
    request: SynthesisRequest | None,
    *,
    text: str | None,
    language: Language | str | None,
    segments: Sequence[ExpressionSegment] | None,
    model: str | None,
    voice_profile: str | None,
    expression: Expression | None,
    generation: Generation | None,
) -> SynthesisRequest:
    if request is not None:
        if not isinstance(request, SynthesisRequest):
            raise TypeError("request must be a SynthesisRequest")
        if any(
            value is not None
            for value in (text, language, segments, model, voice_profile, expression, generation)
        ):
            raise TypeError("pass either a SynthesisRequest or synthesis keyword fields")
        return request
    if language is None:
        raise TypeError("language is required")
    return SynthesisRequest(
        text=text,
        language=language,
        segments=segments,
        model=model,
        voice_profile=voice_profile,
        expression=expression,
        generation=generation,
    )


def _session_create(
    request: SessionCreate | None,
    *,
    model: str | None,
    voice_profile: str | None,
    continuity_policy: ContinuityPolicy | str | None,
) -> SessionCreate:
    if request is not None:
        if not isinstance(request, SessionCreate):
            raise TypeError("request must be a SessionCreate")
        if any(value is not None for value in (model, voice_profile, continuity_policy)):
            raise TypeError("pass either a SessionCreate or session keyword fields")
        return request
    return SessionCreate(
        model=model,
        voice_profile=voice_profile,
        continuity_policy=continuity_policy,
    )


class Client:
    """Synchronous facade over the native AnifLive-TTS client."""

    def __init__(
        self,
        base_url: str,
        *,
        connect_timeout: float = 10.0,
        request_timeout: float | None = 120.0,
        stream_idle_timeout: float | None = 60.0,
        headers: Mapping[str, str] | None = None,
        max_wav_bytes: int = DEFAULT_MAX_WAV_BYTES,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> None:
        self._native: NativeClient = _native_client(
            base_url,
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
            stream_idle_timeout=stream_idle_timeout,
            headers=headers,
            max_wav_bytes=max_wav_bytes,
            chunk_bytes=chunk_bytes,
        )

    @property
    def is_closed(self) -> bool:
        return bool(self._native.is_closed())

    def close(self) -> None:
        """Close local client resources; this does not close remote sessions."""
        self._native.close()

    def health(self) -> Health:
        return self._native.health()

    def capabilities(self) -> Capabilities:
        return self._native.capabilities()

    def models(self) -> ListResponse[ModelInfo]:
        return self._native.models()

    def voices(self) -> ListResponse[VoiceInfo]:
        return self._native.voices()

    def expressions(self) -> ExpressionCatalogInfo:
        return self._native.expressions()

    def activate_model(self, model: str) -> ActivationResult:
        return self._native.activate_model(model)

    def cancel(self) -> CancellationResult:
        """Cancel the service's current synthesis, which may belong to any client or session."""
        return self._native.cancel()

    def probe(
        self,
        required: Iterable[RequiredFeature | str] = (),
    ) -> ServiceInfo:
        return self._native.probe(required_features=_required_feature_values(required))

    def synthesize(
        self,
        request: SynthesisRequest | None = None,
        *,
        text: str | None = None,
        language: Language | str | None = None,
        segments: Sequence[ExpressionSegment] | None = None,
        model: str | None = None,
        voice_profile: str | None = None,
        expression: Expression | None = None,
        generation: Generation | None = None,
    ) -> SynthesisResult:
        value = _synthesis_request(
            request,
            text=text,
            language=language,
            segments=segments,
            model=model,
            voice_profile=voice_profile,
            expression=expression,
            generation=generation,
        )
        return self._native.synthesize(value)

    def stream(
        self,
        request: SynthesisRequest | None = None,
        *,
        text: str | None = None,
        language: Language | str | None = None,
        segments: Sequence[ExpressionSegment] | None = None,
        model: str | None = None,
        voice_profile: str | None = None,
        expression: Expression | None = None,
        generation: Generation | None = None,
    ) -> AudioStream:
        value = _synthesis_request(
            request,
            text=text,
            language=language,
            segments=segments,
            model=model,
            voice_profile=voice_profile,
            expression=expression,
            generation=generation,
        )
        return AudioStream(self._native.stream(value), owner=self)

    def create_session(
        self,
        request: SessionCreate | None = None,
        *,
        model: str | None = None,
        voice_profile: str | None = None,
        continuity_policy: ContinuityPolicy | str | None = None,
    ) -> SpeechSession:
        value = _session_create(
            request,
            model=model,
            voice_profile=voice_profile,
            continuity_policy=continuity_policy,
        )
        return SpeechSession(self._native.create_session(value), owner=self)

    def get_session(self, session_id: str) -> SpeechSession:
        return SpeechSession(self._native.get_session(session_id), owner=self)

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return "Client(<native client>)"


class AsyncClient:
    """Async facade that remains bound to the event loop of its first use."""

    def __init__(
        self,
        base_url: str,
        *,
        connect_timeout: float = 10.0,
        request_timeout: float | None = 120.0,
        stream_idle_timeout: float | None = 60.0,
        headers: Mapping[str, str] | None = None,
        max_wav_bytes: int = DEFAULT_MAX_WAV_BYTES,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> None:
        self._native: NativeClient = _native_client(
            base_url,
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
            stream_idle_timeout=stream_idle_timeout,
            headers=headers,
            max_wav_bytes=max_wav_bytes,
            chunk_bytes=chunk_bytes,
        )
        self._loop: asyncio.AbstractEventLoop | None = None

    def _check_loop(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError("AsyncClient methods must be awaited inside an event loop") from exc
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("AsyncClient cannot be reused from a different event loop")

    @property
    def is_closed(self) -> bool:
        return bool(self._native.is_closed())

    def close(self) -> None:
        """Close local client resources; this does not close remote sessions."""
        self._native.close()

    async def aclose(self) -> None:
        """Async spelling for local, non-network client cleanup."""
        self.close()

    async def health(self) -> Health:
        self._check_loop()
        return await self._native.health_async()

    async def capabilities(self) -> Capabilities:
        self._check_loop()
        return await self._native.capabilities_async()

    async def models(self) -> ListResponse[ModelInfo]:
        self._check_loop()
        return await self._native.models_async()

    async def voices(self) -> ListResponse[VoiceInfo]:
        self._check_loop()
        return await self._native.voices_async()

    async def expressions(self) -> ExpressionCatalogInfo:
        self._check_loop()
        return await self._native.expressions_async()

    async def activate_model(self, model: str) -> ActivationResult:
        self._check_loop()
        return await self._native.activate_model_async(model)

    async def cancel(self) -> CancellationResult:
        self._check_loop()
        return await self._native.cancel_async()

    async def probe(
        self,
        required: Iterable[RequiredFeature | str] = (),
    ) -> ServiceInfo:
        self._check_loop()
        return await self._native.probe_async(
            required_features=_required_feature_values(required)
        )

    async def synthesize(
        self,
        request: SynthesisRequest | None = None,
        *,
        text: str | None = None,
        language: Language | str | None = None,
        segments: Sequence[ExpressionSegment] | None = None,
        model: str | None = None,
        voice_profile: str | None = None,
        expression: Expression | None = None,
        generation: Generation | None = None,
    ) -> SynthesisResult:
        self._check_loop()
        value = _synthesis_request(
            request,
            text=text,
            language=language,
            segments=segments,
            model=model,
            voice_profile=voice_profile,
            expression=expression,
            generation=generation,
        )
        return await self._native.synthesize_async(value)

    async def stream(
        self,
        request: SynthesisRequest | None = None,
        *,
        text: str | None = None,
        language: Language | str | None = None,
        segments: Sequence[ExpressionSegment] | None = None,
        model: str | None = None,
        voice_profile: str | None = None,
        expression: Expression | None = None,
        generation: Generation | None = None,
    ) -> AsyncAudioStream:
        self._check_loop()
        value = _synthesis_request(
            request,
            text=text,
            language=language,
            segments=segments,
            model=model,
            voice_profile=voice_profile,
            expression=expression,
            generation=generation,
        )
        native_stream = await self._native.stream_async(value)
        return AsyncAudioStream(native_stream, loop_guard=self._check_loop, owner=self)

    async def create_session(
        self,
        request: SessionCreate | None = None,
        *,
        model: str | None = None,
        voice_profile: str | None = None,
        continuity_policy: ContinuityPolicy | str | None = None,
    ) -> AsyncSpeechSession:
        self._check_loop()
        value = _session_create(
            request,
            model=model,
            voice_profile=voice_profile,
            continuity_policy=continuity_policy,
        )
        native_session = await self._native.create_session_async(value)
        return AsyncSpeechSession(native_session, loop_guard=self._check_loop)

    async def get_session(self, session_id: str) -> AsyncSpeechSession:
        self._check_loop()
        native_session = await self._native.get_session_async(session_id)
        return AsyncSpeechSession(native_session, loop_guard=self._check_loop)

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return "AsyncClient(<native client>)"


__all__ = ["AsyncClient", "Client"]
