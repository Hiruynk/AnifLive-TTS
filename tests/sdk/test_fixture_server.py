"""HTTP-level contract tests for the local SDK fixture."""

from __future__ import annotations

import http.client
import io
import json
import threading
import time
import wave
from urllib.parse import urlsplit

from fixture_server import (
    ALTERNATE_MODEL,
    DEFAULT_MODEL,
    PCM_CHUNKS,
    SAMPLE_RATE,
    FixtureServer,
)


def _connection(fixture: FixtureServer) -> http.client.HTTPConnection:
    return http.client.HTTPConnection(fixture.host, fixture.port, timeout=4)


def _stream_response(
    connection: http.client.HTTPConnection,
) -> http.client.HTTPResponse:
    connection.request(
        "POST",
        "/v1/audio/speech",
        body=json.dumps(
            {
                "text": "streaming",
                "language": "en",
                "stream": True,
                "response_format": "pcm",
            }
        ),
        headers={"Content-Type": "application/json"},
    )
    return connection.getresponse()


def _send(
    connection: http.client.HTTPConnection,
    method: str,
    path: str,
    payload: object | None = None,
) -> tuple[http.client.HTTPResponse, bytes]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if body is not None else {}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    return response, response.read()


def _json(
    connection: http.client.HTTPConnection,
    method: str,
    path: str,
    payload: object | None = None,
) -> tuple[int, dict[str, str], object]:
    response, body = _send(connection, method, path, payload)
    return (
        response.status,
        {name.lower(): value for name, value in response.getheaders()},
        json.loads(body) if body else None,
    )


def _new_session(connection: http.client.HTTPConnection) -> tuple[str, dict[str, object]]:
    status, _, session = _json(connection, "POST", "/v1/sessions", {})
    assert status == 201
    assert isinstance(session, dict)
    return str(session["id"]), session


def _append(
    connection: http.client.HTTPConnection, session_id: str, payload: dict[str, object]
) -> tuple[int, object]:
    status, _, body = _json(
        connection, "POST", f"/v1/sessions/{session_id}/segments", payload
    )
    return status, body


def test_all_canonical_routes_discovery_shapes_and_keep_alive_records() -> None:
    with FixtureServer(expression_enabled=True) as fixture:
        connection = _connection(fixture)
        try:
            expected_gets = (
                ("/health", "AnifLive-TTS"),
                ("/v1/capabilities", "AnifLive-TTS"),
                ("/v1/models", "list"),
                ("/v1/voices", "list"),
                ("/v1/expressions", "list"),
            )
            for path, expected_object in expected_gets:
                response, body = _send(connection, "GET", path)
                assert response.status == 200
                payload = json.loads(body)
                if path == "/health":
                    assert payload["ready"] is True
                    assert payload["streaming"]["complete_wav"] == "request stream=false"
                elif path == "/v1/capabilities":
                    assert payload["streaming"] == {"pcm16": True, "wav": False}
                    assert payload["speech_sessions"]["default_policy"] == "A"
                elif path == "/v1/expressions":
                    assert payload["enabled"] is True
                    assert payload["profiles"][0]["id"] == "calm"
                else:
                    assert payload["object"] == expected_object
                    assert payload["data"]

            status, _, activated = _json(
                connection,
                "POST",
                "/v1/models/activate",
                {"model": ALTERNATE_MODEL},
            )
            assert status == 200
            assert activated["changed"] is True
            assert activated["model"] == ALTERNATE_MODEL

            response, wav_data = _send(
                connection,
                "POST",
                "/v1/audio/speech",
                {"text": "fixture", "language": "en", "stream": False},
            )
            assert response.status == 200
            assert response.getheader("Content-Type") == "audio/wav"
            assert response.getheader("X-TTS-Model") == ALTERNATE_MODEL
            with wave.open(io.BytesIO(wav_data), "rb") as audio:
                assert audio.getnchannels() == 1
                assert audio.getsampwidth() == 2
                assert audio.getframerate() == SAMPLE_RATE
                assert audio.getnframes() == 4

            status, _, cancelled = _json(
                connection, "POST", "/v1/audio/cancel"
            )
            assert status == 200
            assert cancelled == {"cancelled": False, "model": ALTERNATE_MODEL}

            session_id, session = _new_session(connection)
            assert session["state"] == "open"
            assert session["object"] == "speech.session"
            status, _, fetched = _json(
                connection, "GET", f"/v1/sessions/{session_id}"
            )
            assert status == 200
            assert fetched["id"] == session_id

            segment = {
                "segment_id": "seg-0",
                "text": "hello",
                "language": "en",
            }
            status, created = _append(connection, session_id, segment)
            assert status == 201
            assert created["sequence"] == 0

            status, _, flushed = _json(
                connection, "POST", f"/v1/sessions/{session_id}/flush"
            )
            assert status == 200
            assert flushed["state"] == "flushing"

            response, pcm = _send(
                connection, "GET", f"/v1/sessions/{session_id}/audio"
            )
            assert response.status == 200
            assert pcm == b"".join(PCM_CHUNKS)

            status, _, cancel_result = _json(
                connection, "POST", f"/v1/sessions/{session_id}/cancel"
            )
            assert status == 200
            assert cancel_result["state"] == "closed"

            delete_id, _ = _new_session(connection)
            status, _, deleted = _json(
                connection, "DELETE", f"/v1/sessions/{delete_id}"
            )
            assert status == 200
            assert deleted["state"] == "cancelled"

            expected_routes = {
                ("GET", "/health"),
                ("GET", "/v1/capabilities"),
                ("GET", "/v1/models"),
                ("GET", "/v1/voices"),
                ("GET", "/v1/expressions"),
                ("POST", "/v1/models/activate"),
                ("POST", "/v1/audio/speech"),
                ("POST", "/v1/audio/cancel"),
                ("POST", "/v1/sessions"),
                ("GET", f"/v1/sessions/{session_id}"),
                ("POST", f"/v1/sessions/{session_id}/segments"),
                ("GET", f"/v1/sessions/{session_id}/audio"),
                ("POST", f"/v1/sessions/{session_id}/flush"),
                ("POST", f"/v1/sessions/{session_id}/cancel"),
                ("DELETE", f"/v1/sessions/{delete_id}"),
            }
            observed = {
                (record.method, urlsplit(record.path).path)
                for record in fixture.records
            }
            assert expected_routes <= observed
            discovery_records = fixture.records[:5]
            assert len({record.connection_id for record in discovery_records}) == 1
            assert fixture.records[5].json_body == {"model": ALTERNATE_MODEL}
            assert fixture.records[6].json_body == {"text": "fixture", "language": "en", "stream": False}
        finally:
            connection.close()


def test_streamed_pcm_is_incremental_raw_bytes_across_chunk_boundaries() -> None:
    chunks = (b"\x01", b"\x00\x02\x00")
    with FixtureServer(pcm_chunks=chunks, stream_delay_seconds=0.005) as fixture:
        connection = _connection(fixture)
        try:
            connection.request(
                "POST",
                "/v1/audio/speech",
                body=json.dumps(
                    {
                        "text": "streaming",
                        "language": "en",
                        "stream": True,
                        "response_format": "pcm",
                    }
                ),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            assert response.status == 200
            assert response.getheader("Content-Type") == "application/octet-stream"
            assert response.getheader("Transfer-Encoding") == "chunked"
            assert response.getheader("X-TTS-Sample-Format") == "s16le"
            assert response.getheader("X-TTS-Sample-Rate") == str(SAMPLE_RATE)
            assert response.read() == b"".join(chunks)
        finally:
            connection.close()


def test_session_idempotency_flush_and_one_audio_consumer() -> None:
    with FixtureServer(
        pcm_chunks=(b"\x01\x00",) * 4, stream_delay_seconds=0.04
    ) as fixture:
        writer = _connection(fixture)
        first_consumer = _connection(fixture)
        second_consumer = _connection(fixture)
        try:
            session_id, _ = _new_session(writer)
            segment = {
                "segment_id": "stable-id",
                "text": "first paragraph",
                "language": "en",
                "paragraph_id": "p1",
                "pause_after_ms": 120,
            }
            status, created = _append(writer, session_id, segment)
            assert status == 201
            status, duplicate = _append(writer, session_id, segment)
            assert status == 200
            assert duplicate["segment_id"] == created["segment_id"]
            assert duplicate["sequence"] == 0

            changed = {**segment, "text": "different content"}
            status, conflict = _append(writer, session_id, changed)
            assert status == 409
            assert conflict["code"] == 409

            first_consumer.request("GET", f"/v1/sessions/{session_id}/audio")
            stream = first_consumer.getresponse()
            assert stream.status == 200
            first_chunk = stream.read(2)
            assert first_chunk == b"\x01\x00"

            status, _, rejected = _json(
                second_consumer, "GET", f"/v1/sessions/{session_id}/audio"
            )
            assert status == 409
            assert rejected["code"] == 409

            status, _, flushed = _json(
                writer, "POST", f"/v1/sessions/{session_id}/flush"
            )
            assert status == 200
            assert flushed["state"] == "flushing"
            tail = stream.read()
            assert first_chunk + tail == b"\x01\x00" * 4

            status, _, final_state = _json(
                writer, "GET", f"/v1/sessions/{session_id}"
            )
            assert status == 200
            assert final_state["state"] == "closed"
            assert final_state["audio_consumer_attached"] is False
            assert final_state["context"]["previous_output_samples"] == 4
            assert final_state["context"]["total_output_samples"] == 4
        finally:
            writer.close()
            first_consumer.close()
            second_consumer.close()


def test_current_owner_cancel_can_stop_session_stream() -> None:
    with FixtureServer(
        pcm_chunks=(b"\x01\x00",) * 20, stream_delay_seconds=0.04
    ) as fixture:
        control = _connection(fixture)
        consumer = _connection(fixture)
        try:
            session_id, _ = _new_session(control)
            status, _ = _append(
                control,
                session_id,
                {
                    "segment_id": "cancel-me",
                    "text": "session audio",
                    "language": "en",
                },
            )
            assert status == 201

            consumer.request("GET", f"/v1/sessions/{session_id}/audio")
            stream = consumer.getresponse()
            assert stream.status == 200
            assert stream.read(2) == b"\x01\x00"

            status, _, result = _json(
                control, "POST", "/v1/audio/cancel"
            )
            assert status == 200
            assert result == {"cancelled": True, "model": DEFAULT_MODEL}
            stream.read()

            status, _, state = _json(
                control, "GET", f"/v1/sessions/{session_id}"
            )
            assert status == 200
            assert state["state"] == "cancelled"
            assert state["segments"][0]["state"] == "cancelled"
        finally:
            control.close()
            consumer.close()


def test_overrides_cover_429_malformed_json_headers_and_delayed_streams() -> None:
    with FixtureServer() as fixture:
        connection = _connection(fixture)
        try:
            fixture.set_override(
                "GET",
                "/v1/capabilities",
                status=429,
                body={"code": 429, "message": "fixture busy"},
                headers={"Retry-After": "3"},
            )
            status, headers, payload = _json(
                connection, "GET", "/v1/capabilities"
            )
            assert status == 429
            assert headers["retry-after"] == "3"
            assert payload["message"] == "fixture busy"
            fixture.clear_override("GET", "/v1/capabilities")

            fixture.set_override(
                "GET",
                "/v1/models",
                body=b"{malformed",
                headers={"Content-Type": "application/json"},
            )
            response, malformed = _send(connection, "GET", "/v1/models")
            assert response.status == 200
            assert response.getheader("Content-Type") == "application/json"
            assert malformed == b"{malformed"
            fixture.clear_override("GET", "/v1/models")

            fixture.set_override(
                "POST",
                "/v1/audio/speech",
                chunks=(b"\x01", b"\x00"),
                headers={
                    "Content-Type": None,
                    "X-TTS-Sample-Rate": None,
                },
                delay_seconds=0.04,
            )
            connection.request(
                "POST",
                "/v1/audio/speech",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )
            start = time.monotonic()
            response = connection.getresponse()
            assert response.status == 200
            assert response.getheader("Transfer-Encoding") == "chunked"
            assert response.getheader("Content-Type") is None
            assert response.getheader("X-TTS-Sample-Rate") is None
            assert response.read() == b"\x01\x00"
            assert time.monotonic() - start >= 0.035
        finally:
            connection.close()



def test_competing_speech_streams_wait_for_serialized_stream_slot() -> None:
    chunks = (b"\x01\x00",) * 12
    with FixtureServer(pcm_chunks=chunks, stream_delay_seconds=0.04) as fixture:
        first = _connection(fixture)
        second = _connection(fixture)
        second_headers_received = threading.Event()
        second_finished = threading.Event()
        second_result: dict[str, object] = {}

        def read_second() -> None:
            response = _stream_response(second)
            second_result["status"] = response.status
            second_headers_received.set()
            second_result["body"] = response.read()
            second_finished.set()

        worker = threading.Thread(target=read_second, daemon=True)
        try:
            first_response = _stream_response(first)
            assert first_response.status == 200
            assert first_response.read(2) == chunks[0]

            worker.start()
            assert second_headers_received.wait(timeout=1)
            assert second_result["status"] == 200
            time.sleep(0.08)
            assert not second_finished.is_set()

            first_tail = first_response.read()
            assert chunks[0] + first_tail == b"".join(chunks)
            assert second_finished.wait(timeout=4)
            assert second_result["body"] == b"".join(chunks)
        finally:
            first.close()
            second.close()


def test_idle_session_reader_disconnect_detaches_without_cancelling_session() -> None:
    with FixtureServer() as fixture:
        writer = _connection(fixture)
        abandoned = _connection(fixture)
        replacement = _connection(fixture)
        try:
            session_id, _ = _new_session(writer)
            abandoned.request("GET", f"/v1/sessions/{session_id}/audio")
            response = abandoned.getresponse()
            assert response.status == 200
            abandoned.close()

            deadline = time.monotonic() + 2
            state: dict[str, object] = {}
            while time.monotonic() < deadline:
                status, _, payload = _json(
                    writer, "GET", f"/v1/sessions/{session_id}"
                )
                assert status == 200
                assert isinstance(payload, dict)
                state = payload
                if payload["audio_consumer_attached"] is False:
                    break
                time.sleep(0.02)
            assert state["audio_consumer_attached"] is False
            assert state["state"] == "open"

            status, _ = _append(
                writer,
                session_id,
                {
                    "segment_id": "after-disconnect",
                    "text": "reconnect",
                    "language": "en",
                },
            )
            assert status == 201
            status, _, flushed = _json(
                writer, "POST", f"/v1/sessions/{session_id}/flush"
            )
            assert status == 200
            assert flushed["state"] == "flushing"

            response, pcm = _send(
                replacement, "GET", f"/v1/sessions/{session_id}/audio"
            )
            assert response.status == 200
            assert pcm == b"".join(PCM_CHUNKS)
            status, _, final_state = _json(
                writer, "GET", f"/v1/sessions/{session_id}"
            )
            assert status == 200
            assert final_state["state"] == "closed"
            assert final_state["audio_consumer_attached"] is False
        finally:
            writer.close()
            abandoned.close()
            replacement.close()



def test_session_stream_waits_for_a_competing_direct_stream() -> None:
    chunks = (bytes((1, 0)),) * 12
    with FixtureServer(pcm_chunks=chunks, stream_delay_seconds=0.04) as fixture:
        control = _connection(fixture)
        first = _connection(fixture)
        second = _connection(fixture)
        second_headers_received = threading.Event()
        second_body_started = threading.Event()
        second_finished = threading.Event()
        second_result: dict[str, object] = {}

        session_id, _ = _new_session(control)
        status, _ = _append(
            control,
            session_id,
            {
                "segment_id": "queued-session-audio",
                "text": "queued",
                "language": "en",
            },
        )
        assert status == 201
        status, _, flushed = _json(
            control, "POST", f"/v1/sessions/{session_id}/flush"
        )
        assert status == 200
        assert flushed["state"] == "flushing"

        def read_second() -> None:
            second.request("GET", f"/v1/sessions/{session_id}/audio")
            response = second.getresponse()
            second_result["status"] = response.status
            second_headers_received.set()
            first_chunk = response.read(2)
            second_body_started.set()
            second_result["body"] = first_chunk + response.read()
            second_finished.set()

        worker = threading.Thread(target=read_second, daemon=True)
        try:
            first_response = _stream_response(first)
            assert first_response.status == 200
            assert first_response.read(2) == chunks[0]

            worker.start()
            assert second_headers_received.wait(timeout=1)
            assert second_result["status"] == 200
            assert not second_body_started.wait(timeout=0.12)

            assert first_response.read() == b"".join(chunks[1:])
            assert second_body_started.wait(timeout=2)
            assert second_finished.wait(timeout=2)
            assert second_result["body"] == b"".join(chunks)

            status, _, final_state = _json(
                control, "GET", f"/v1/sessions/{session_id}"
            )
            assert status == 200
            assert final_state["state"] == "closed"
        finally:
            control.close()
            first.close()
            second.close()
