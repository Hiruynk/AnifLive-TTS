from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .._native import NativeAudioStream
else:
    NativeAudioStream = Any

from .types import StreamMetadata


class AudioStream(Iterator[bytes]):
    """Synchronous, demand-driven iterator over exact PCM response bytes."""

    def __init__(self, native: NativeAudioStream, *, owner: Any = None) -> None:
        self._native = native
        self._owner = owner

    @property
    def sample_rate(self) -> int:
        return self._native.sample_rate

    @property
    def channels(self) -> int:
        return self._native.channels

    @property
    def sample_format(self) -> str:
        return self._native.sample_format

    @property
    def metadata(self) -> StreamMetadata:
        return self._native.metadata

    @property
    def is_closed(self) -> bool:
        return bool(self._native.is_closed())

    def next_chunk(self) -> bytes | None:
        chunk = self._native.next_chunk()
        if chunk is None:
            self._owner = None
            return None
        return bytes(chunk)

    def close(self) -> None:
        """Release this body reader without calling the global audio cancel endpoint."""
        try:
            self._native.close()
        finally:
            self._owner = None

    def __iter__(self) -> AudioStream:
        return self

    def __next__(self) -> bytes:
        chunk = self.next_chunk()
        if chunk is None:
            raise StopIteration
        return chunk

    def __enter__(self) -> AudioStream:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return "AudioStream(<native stream>)"


class AsyncAudioStream(AsyncIterator[bytes]):
    """Asynchronous PCM iterator tied to its creating AsyncClient's event loop."""

    def __init__(
        self,
        native: NativeAudioStream,
        *,
        loop_guard: Callable[[], None] | None = None,
        owner: Any = None,
    ) -> None:
        self._native = native
        self._loop_guard = loop_guard
        self._owner = owner

    @property
    def sample_rate(self) -> int:
        return self._native.sample_rate

    @property
    def channels(self) -> int:
        return self._native.channels

    @property
    def sample_format(self) -> str:
        return self._native.sample_format

    @property
    def metadata(self) -> StreamMetadata:
        return self._native.metadata

    @property
    def is_closed(self) -> bool:
        return bool(self._native.is_closed())

    async def next_chunk(self) -> bytes | None:
        if self._loop_guard is not None:
            self._loop_guard()
        chunk = await self._native.next_chunk_async()
        if chunk is None:
            self._owner = None
            return None
        return bytes(chunk)

    def close(self) -> None:
        """Release this body reader without calling the global audio cancel endpoint."""
        try:
            self._native.close()
        finally:
            self._owner = None

    async def aclose(self) -> None:
        self.close()

    def __aiter__(self) -> AsyncAudioStream:
        return self

    async def __anext__(self) -> bytes:
        chunk = await self.next_chunk()
        if chunk is None:
            raise StopAsyncIteration
        return chunk

    async def __aenter__(self) -> AsyncAudioStream:
        return self

    async def __aexit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return "AsyncAudioStream(<native stream>)"


__all__ = ["AsyncAudioStream", "AudioStream"]
