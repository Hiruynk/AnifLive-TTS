from __future__ import annotations

import asyncio
import json

from fixture_server import FixtureServer


def test_async_discovery_wav_and_incremental_pcm():
    from aniflive_tts import AsyncClient

    async def scenario(url, wav, chunks):
        async with AsyncClient(url) as client:
            assert (await client.health()).service == "AnifLive-TTS"
            assert (await client.capabilities()).streaming.pcm16
            assert (await client.models()).data
            assert (await client.voices()).data
            assert (await client.expressions()).enabled is False
            assert (await client.probe()).health.ready
            result = await client.synthesize(text="今日はいい天気ですね。", language="ja")
            assert result.audio == wav
            stream = await client.stream(text="Hello.", language="en")
            assert stream.sample_format == "s16le"
            assert b"".join([chunk async for chunk in stream]) == b"".join(chunks)

    with FixtureServer() as server:
        asyncio.run(scenario(server.base_url, server.wav_bytes, server.pcm_chunks))


def test_async_waiting_request_does_not_block_python_event_loop():
    from aniflive_tts import AsyncClient

    async def scenario(url):
        progressed = asyncio.Event()

        async def other_work():
            await asyncio.sleep(0.02)
            progressed.set()

        async with AsyncClient(url) as client:
            worker = asyncio.create_task(other_work())
            request = asyncio.create_task(client.health())
            await asyncio.wait_for(progressed.wait(), timeout=0.15)
            assert not request.done()
            assert (await request).service == "AnifLive-TTS"
            await worker

    with FixtureServer() as server:
        server.set_override(
            "GET",
            "/health",
            headers={"Content-Type": "application/json"},
            chunks=[
                json.dumps(
                    {"service": "AnifLive-TTS", "version": "1.4.0", "ready": True, "status": "ok"}
                ).encode()
            ],
            delay_seconds=0.3,
        )
        asyncio.run(scenario(server.base_url))


def test_async_finite_session_preserves_remote_delete_semantics():
    from aniflive_tts import AsyncClient

    async def scenario(url, chunks):
        async with AsyncClient(url) as client:
            session = await client.create_session()
            first = await session.append(segment_id="one", text="Hello.", language="en")
            duplicate = await session.append(segment_id="one", text="Hello.", language="en")
            assert duplicate.segment_id == first.segment_id
            await session.flush()
            audio = await session.audio()
            assert b"".join([chunk async for chunk in audio]) == b"".join(chunks)
            assert (await session.refresh()).state == "closed"
            await session.close()
            # DELETE is the server cancellation alias, not record removal.
            assert (await session.refresh()).state in {"closed", "cancelled"}

    with FixtureServer() as server:
        asyncio.run(scenario(server.base_url, server.pcm_chunks))
        assert any(item.method == "DELETE" for item in server.records)
        assert all(item.path != "/v1/audio/cancel" for item in server.records)
