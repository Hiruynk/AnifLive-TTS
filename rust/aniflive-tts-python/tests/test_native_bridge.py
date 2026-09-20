"""Tests of the actual extension, independent of the Python convenience facade."""
import asyncio
import sys
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests" / "sdk"))
from fixture_server import ALTERNATE_MODEL, FixtureServer

from aniflive_tts._native import NativeClient
from aniflive_tts.sdk import types
from aniflive_tts.sdk.errors import BusyError, ClosedError


def request():
    return types.SynthesisRequest(language=types.Language.JA, text="private test text", generation=types.Generation(seed=1234))

def test_sync_typed_discovery_wav_and_connection_reuse():
    with FixtureServer(expression_enabled=True) as fixture:
        client = NativeClient(fixture.base_url, headers={"User-Agent": "fixture-sdk"})
        assert isinstance(client.health(), types.Health)
        assert isinstance(client.capabilities().streaming, types.StreamingCapabilities)
        assert isinstance(client.models().data[0], types.ModelInfo)
        assert isinstance(client.voices().data[0], types.VoiceInfo)
        catalog = client.expressions()
        assert isinstance(catalog, types.ExpressionCatalogInfo)
        assert isinstance(catalog.playback_policy, Mapping)
        assert isinstance(client.probe(), types.ServiceInfo)
        assert client.synthesize(request()).audio == fixture.wav_bytes
        assert client.activate_model(ALTERNATE_MODEL).model == ALTERNATE_MODEL
        assert not client.cancel().cancelled
        # Hyper returns HTTP/1 connections to its pool asynchronously and may race a
        # new connection against checkout. Verify reuse, not a single-socket guarantee.
        connection_ids = [record.connection_id for record in fixture.records]
        assert len(set(connection_ids)) < len(connection_ids)
        assert {key.lower(): value for key, value in fixture.records[0].headers}["user-agent"] == "fixture-sdk"
        client.close()
        assert client.is_closed()
        with pytest.raises(ClosedError):
            client.health()

def test_errors_preserve_inspectable_attributes_and_redact_display():
    with FixtureServer() as fixture:
        fixture.set_override("GET", "/health", status=429,
             body={"code":"secret-code","message":"secret-text"},
             headers={"Retry-After":"Wed, 21 Oct 2026 07:28:00 GMT"})
        client=NativeClient(fixture.base_url)
        with pytest.raises(BusyError) as caught:
            client.health()
        error=caught.value
        assert error.status_code==429 and error.server_code=="secret-code"
        assert error.message=="secret-text"
        assert error.retry_after=="Wed, 21 Oct 2026 07:28:00 GMT"
        assert "secret" not in str(error)+repr(error)
        client.close()

def test_sync_stream_and_finite_session():
    with FixtureServer() as fixture:
        client=NativeClient(fixture.base_url,chunk_bytes=2)
        stream=client.stream(request())
        assert isinstance(stream.metadata,types.StreamMetadata)
        result=[]
        while (chunk:=stream.next_chunk()) is not None:
            assert len(chunk)<=2
            result.append(chunk)
        assert b"".join(result)==b"".join(fixture.pcm_chunks)
        session=client.create_session(types.SessionCreate())
        assert isinstance(session.initial_info.context,types.SessionContext)
        segment=types.SegmentRequest(segment_id="first",text="test",language=types.Language.JA)
        assert isinstance(session.append(segment),types.SegmentInfo)
        session.flush()
        audio=session.audio()
        while audio.next_chunk() is not None:
            pass
        assert session.get().state=="closed"
        assert session.close().state=="closed"
        assert client.get_session(session.id).id==session.id
        client.close()

def test_async_parity_loop_affinity_and_no_sync_wait_on_loop():
    with FixtureServer() as fixture:
        client=NativeClient(fixture.base_url)
        async def run():
            assert isinstance(await client.health_async(),types.Health)
            assert isinstance(await client.capabilities_async(),types.Capabilities)
            assert (await client.synthesize_async(request())).audio==fixture.wav_bytes
            with pytest.raises(RuntimeError,match="async SDK"):
                client.health()
            stream=await client.stream_async(request())
            assert isinstance(stream.metadata,types.StreamMetadata)
            chunks=[]
            while (chunk:=await stream.next_chunk_async()) is not None:
                chunks.append(chunk)
            assert b"".join(chunks)==b"".join(fixture.pcm_chunks)
            session=await client.create_session_async(types.SessionCreate())
            await session.append_async(types.SegmentRequest(segment_id="s",text="test",language=types.Language.JA))
            await session.flush_async()
            stream=await session.audio_async()
            while await stream.next_chunk_async() is not None:
                pass
            assert (await session.get_async()).state=="closed"
            await session.close_async()
        asyncio.run(run())
        async def other():
            with pytest.raises(RuntimeError,match="different event loop"):
                await client.health_async()
        asyncio.run(other())
        client.close()

def test_async_cancel_drops_pending_stream_read():
    with FixtureServer(stream_delay_seconds=.5) as fixture:
        client=NativeClient(fixture.base_url)
        async def run():
            stream=await client.stream_async(request())
            # Fixture emits first chunk immediately, then waits before the next.
            await stream.next_chunk_async()
            task=asyncio.ensure_future(stream.next_chunk_async())
            await asyncio.sleep(.02)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            for _ in range(50):
                if stream.is_closed():
                    break
                await asyncio.sleep(.01)
            assert stream.is_closed()
        asyncio.run(run())
        client.close()

def test_sync_gil_release_allows_another_thread_to_close_pending_read():
    with FixtureServer(stream_delay_seconds=.5) as fixture:
        client=NativeClient(fixture.base_url)
        stream=client.stream(request())
        stream.next_chunk()
        started=threading.Event()
        def read():
            started.set()
            return stream.next_chunk()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result=executor.submit(read)
            assert started.wait(1)
            stream.close()
            assert result.result(timeout=1) is None
        client.close()

def test_capability_policy_records_preserve_unknown_policies_and_fields():
    with FixtureServer() as fixture:
        fixture.set_override("GET", "/v1/capabilities", body={
            "service": "AnifLive-TTS", "version": "future", "languages": ["ja"],
            "streaming": {"pcm16": True, "wav": False},
            "expression": {"native": True, "controlled_profiles": False},
            "speech_sessions": {"committed_segments": True, "pcm_stream": True,
                "policies": {"A": {"name": "none", "phones_bert": False},
                             "Z": {"name": "future", "future_nested": {"value": 3}}}},
        })
        client = NativeClient(fixture.base_url)
        policies = client.capabilities().speech_sessions.policies
        assert all(isinstance(item, types.ContinuityPolicyCapability)
                   for item in policies.values())
        assert policies["Z"].extra["future_nested"] == {"value": 3}
        client.close()


def test_native_loop_affinity_does_not_keep_finished_loop_alive():
    import gc
    import weakref

    with FixtureServer() as fixture:
        client = NativeClient(fixture.base_url)

        async def first():
            await client.health_async()
            return weakref.ref(asyncio.get_running_loop())

        original = asyncio.run(first())
        gc.collect()
        assert original() is None

        async def second():
            with pytest.raises(RuntimeError, match="different event loop"):
                await client.health_async()

        asyncio.run(second())
        client.close()
