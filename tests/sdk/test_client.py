from __future__ import annotations

import pytest

from fixture_server import FixtureServer


def test_sync_discovery_reuses_connection_and_preserves_unknown_fields():
    from aniflive_tts import Client

    with FixtureServer() as server:
        server.set_override("GET", "/health", body={
            "service": "AnifLive-TTS", "version": "9.0.1", "ready": False,
            "status": "starting", "future_field": {"supported": True},
        })
        with Client(server.base_url) as client:
            health = client.health()
            assert health.ready is False
            assert health.extra["future_field"] == {"supported": True}
            assert client.capabilities().streaming.pcm16 is True
            assert client.models().data
            assert client.voices().data
            assert client.expressions().enabled is False
            assert client.probe().health.version == "9.0.1"
        # Hyper returns HTTP/1 connections to its pool asynchronously and may race a
        # new connection against checkout. Verify reuse, not a single-socket guarantee.
        connection_ids = [record.connection_id for record in server.records]
        assert len(set(connection_ids)) < len(connection_ids)


def test_sync_complete_wav_and_typed_request():
    from aniflive_tts import Client, Generation, SynthesisRequest

    with FixtureServer() as server, Client(server.base_url) as client:
        result = client.synthesize(SynthesisRequest(
            text="今日はいい天気ですね。", language="ja", generation=Generation(seed=1234),
        ))
        assert result.audio == server.wav_bytes
        assert result.content_type == "audio/wav"
        assert result.sample_rate == 32000
        body = server.records[-1].json_body
        assert body["stream"] is False
        assert body["response_format"] == "wav"
        assert body["generation"] == {"seed": 1234}


def test_sync_busy_error_retains_details_without_repr_leak():
    from aniflive_tts import BusyError, Client

    private_message = "private-fixture-message"
    with FixtureServer() as server, Client(server.base_url) as client:
        server.set_override("POST", "/v1/audio/speech", status=429,
                            body={"code": 429, "message": private_message},
                            headers={"Retry-After": "2"})
        with pytest.raises(BusyError) as caught:
            client.synthesize(text="Hello.", language="en")
        error = caught.value
        assert error.status_code == 429
        assert error.server_code == 429
        assert error.retry_after == "2"
        assert error.message == private_message
        assert private_message not in repr(error)
        assert private_message not in str(error)
        assert len(server.records) == 1


def test_sync_bad_response_is_protocol_error_and_proxy_env_is_ignored(monkeypatch):
    from aniflive_tts import Client, ProtocolError

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    with FixtureServer() as server, Client(server.base_url) as client:
        assert client.health().service == "AnifLive-TTS"
        server.set_override("GET", "/health", body=b"{broken",
                            headers={"Content-Type": "application/json"})
        with pytest.raises(ProtocolError):
            client.health()


def test_sync_stream_exact_bytes_and_local_close():
    from aniflive_tts import Client, ClientError

    with FixtureServer() as server:
        client = Client(server.base_url)
        stream = client.stream(text="Hello.", language="en")
        assert stream.sample_rate == 32000
        assert stream.sample_format == "s16le"
        assert stream.channels == 1
        assert b"".join(stream) == b"".join(server.pcm_chunks)
        stream.close()
        client.close()
        with pytest.raises(ClientError):
            client.health()
        assert all(item.path != "/v1/audio/cancel" for item in server.records)
