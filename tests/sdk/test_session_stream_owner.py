from __future__ import annotations

import asyncio
import gc

from fixture_server import FixtureServer
from aniflive_tts.sdk import AsyncClient, Client


def test_sync_session_stream_keeps_session_owner_alive() -> None:
    with FixtureServer() as server, Client(server.base_url) as client:
        session = client.create_session()
        session.append(segment_id="owner-check", text="Hello.", language="en")
        session.flush()
        stream = session.audio()
        del session
        gc.collect()

        assert not stream.is_closed
        assert b"".join(stream) == b"".join(server.pcm_chunks)


def test_async_session_stream_keeps_session_owner_alive() -> None:
    async def scenario(base_url: str, chunks: tuple[bytes, ...]) -> None:
        async with AsyncClient(base_url) as client:
            session = await client.create_session()
            await session.append(segment_id="owner-check", text="Hello.", language="en")
            await session.flush()
            stream = await session.audio()
            del session
            gc.collect()

            assert not stream.is_closed
            assert b"".join([chunk async for chunk in stream]) == b"".join(chunks)

    with FixtureServer() as server:
        asyncio.run(scenario(server.base_url, server.pcm_chunks))
