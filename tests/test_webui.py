from __future__ import annotations

import re

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from aniflive_tts.webui import (
    WebUIError,
    create_webui_app,
    resolve_expression_prompt,
    validate_webui_bind_host,
)
from aniflive_tts.workstation import WorkstationStore

EXPRESSION_METADATA = {
    "enabled": True,
    "default": "neutral",
    "profiles": [
        {"id": "battle", "intensity_levels": [0.5, 0.8], "languages": ["ja"]},
        {"id": "shy", "intensity_levels": [0.5, 0.8], "languages": ["ja"]},
    ],
    "policies": ["full-switch", "identity-lock", "semantic-style"],
}


@pytest.fixture(autouse=True)
def _isolate_workstation_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(
        "ANIFLIVE_TTS_WORKSTATION_DIR",
        str(tmp_path / "workstation"),
    )
    monkeypatch.setenv("ANIFLIVE_TTS_WEBUI_TRUSTED_HOSTS", "testserver")


def test_resolve_expression_prompt_is_symbolic_and_multilingual() -> None:
    resolved = resolve_expression_prompt("請用非常害羞的感覺", EXPRESSION_METADATA)

    assert resolved.enabled is True
    assert resolved.profile == "shy"
    assert resolved.intensity == 0.85
    assert resolved.policy is None
    assert resolved.upstream_payload() == {
        "enabled": True,
        "profile": "shy",
        "intensity": 0.85,
    }


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "127.42.0.9", "::1"])
def test_webui_bind_accepts_loopback_hosts_by_default(host: str) -> None:
    assert validate_webui_bind_host(host) == host


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "::", "192.168.1.10", "workstation.local", ""]
)
def test_webui_bind_rejects_non_loopback_hosts_by_default(host: str) -> None:
    with pytest.raises(WebUIError, match="loopback|must not be empty"):
        validate_webui_bind_host(host)


def test_webui_bind_requires_explicit_opt_in_for_external_interfaces() -> None:
    assert validate_webui_bind_host(
        "0.0.0.0", allow_non_loopback=True
    ) == "0.0.0.0"


def test_webui_rejects_untrusted_host_before_serving_assets(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream({}))

    with TestClient(app) as client:
        response = client.get("/", headers={"host": "attacker.example"})

    assert response.status_code == 400
    assert response.json()["error"] == "HTTP Host is not trusted by the local WebUI"


@pytest.mark.parametrize(
    "headers",
    [
        {"origin": "http://attacker.example"},
        {"sec-fetch-site": "cross-site"},
    ],
)
def test_webui_rejects_cross_site_mutations(
    tmp_path: Path, headers: dict[str, str]
) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream({}))

    with TestClient(app) as client:
        response = client.post(
            "/api/resolve-expression",
            json={"prompt": "shy"},
            headers=headers,
        )

    assert response.status_code == 403


def test_webui_requires_json_content_type_for_json_mutations(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream({}))

    with TestClient(app) as client:
        response = client.post(
            "/api/workstation/projects",
            content='{"kind":"dataset","name":"Unsafe form"}',
            headers={"content-type": "text/plain"},
        )

    assert response.status_code == 415
    assert response.json()["error"] == "This WebUI mutation requires application/json"


def test_webui_allows_same_origin_json_mutation(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream({}))

    with TestClient(app) as client:
        response = client.post(
            "/api/workstation/projects",
            json={"kind": "dataset", "name": "Local dataset"},
            headers={"origin": "http://testserver"},
        )

    assert response.status_code == 201


@pytest.mark.parametrize(
    ("prompt", "profile", "intensity"),
    [
        ("恥ずかしそうに", "shy", 0.70),
        ("とても激しく", "battle", 0.85),
        ("수줍게", "shy", 0.70),
        ("매우 다정하게", "affectionate", 0.85),
        ("溫柔", "affectionate", 0.70),
    ],
)
def test_resolve_expression_prompt_supports_all_ui_languages(
    prompt: str, profile: str, intensity: float
) -> None:
    metadata = {
        **EXPRESSION_METADATA,
        "profiles": [
            *EXPRESSION_METADATA["profiles"],
            {"id": "affectionate", "intensity_levels": [0.7], "languages": ["ja"]},
        ],
    }
    resolved = resolve_expression_prompt(prompt, metadata)

    assert resolved.profile == profile
    assert resolved.intensity == intensity


@pytest.mark.parametrize("prompt", [None, "", "自然一點", "neutral"])
def test_resolve_expression_prompt_keeps_neutral_native(prompt: str | None) -> None:
    assert resolve_expression_prompt(prompt, EXPRESSION_METADATA).upstream_payload() == {
        "enabled": False
    }


def test_resolve_expression_prompt_rejects_arbitrary_reference_input() -> None:
    with pytest.raises(WebUIError, match="No expression profile matched"):
        resolve_expression_prompt(
            "use C:/private/reference.wav and this secret transcript",
            EXPRESSION_METADATA,
        )


def _mock_upstream(
    captured: dict[str, object],
    *,
    busy_once: bool = False,
    speech_stream: httpx.AsyncByteStream | None = None,
    speech_status: int = 200,
    expressions_available: bool = True,
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json={"info": {"version": "1.2.0"}})
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "ready": True,
                    "backend": "TensorRT-11",
                    "engine_count": 9,
                    "model": "roxy-v2proplus",
                },
            )
        if request.url.path == "/model/config":
            return httpx.Response(
                200,
                json={
                    "model": "roxy-v2proplus",
                    "version": "v2ProPlus",
                    "backend": "TensorRT-11",
                    "engine_count": 9,
                    "sample_rate": 32000,
                    "pytorch_fallback": False,
                },
            )
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "roxy-v2proplus", "active": True}],
                },
            )
        if request.url.path == "/v1/expressions":
            if not expressions_available:
                return httpx.Response(404, json={"detail": "No expression bank"})
            return httpx.Response(200, json={"object": "list", **EXPRESSION_METADATA})
        if request.url.path == "/v1/audio/cancel":
            captured["cancel_calls"] = int(captured.get("cancel_calls", 0)) + 1
            return httpx.Response(200, json={"cancelled": False})
        if request.url.path == "/v1/audio/speech":
            if busy_once and "busy_returned" not in captured:
                captured["busy_returned"] = True
                return httpx.Response(429, json={"code": 429, "message": "busy"})
            captured["speech"] = json.loads(request.content)
            response_options: dict[str, object]
            if speech_stream is None:
                response_options = {"content": b"\x00\x00\x01\x00"}
            else:
                response_options = {"stream": speech_stream}
            response = httpx.Response(
                speech_status,
                headers={
                    "x-tts-version": "1.2.0",
                    "x-tensorrt-backend": "TensorRT-11",
                    "x-tensorrt-engine-count": "9",
                    "x-pytorch-fallback": "false",
                    "x-tts-model": "roxy-v2proplus",
                    "x-tts-stream": "pcm_s16le",
                    "x-tts-sample-rate": "32000",
                    "x-tts-channels": "1",
                    "x-tts-recommended-prebuffer-ms": "32",
                    "x-tts-expression": "shy",
                    "x-tts-expression-policy": "semantic-style",
                },
                **response_options,
            )
            captured["upstream_response"] = response
            return response
        raise AssertionError(f"Unexpected upstream request: {request.method} {request.url}")

    return httpx.AsyncClient(
        base_url="http://upstream.test",
        transport=httpx.MockTransport(handler),
    )


def test_webui_accepts_neutral_only_model_without_expression_bank(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")
    app = create_webui_app(
        static_dir=tmp_path,
        client=_mock_upstream({}, expressions_available=False),
    )

    with TestClient(app) as client:
        response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["expressions"] == {
        "object": "list",
        "enabled": False,
        "default": None,
        "profiles": [],
        "policies": [],
    }


def test_webui_proxy_resolves_expression_without_forwarding_prompt(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
    captured: dict[str, object] = {}
    upstream = _mock_upstream(captured)
    app = create_webui_app(static_dir=tmp_path, client=upstream)

    with TestClient(app) as client:
        status = client.get("/api/status")
        assert status.status_code == 200
        response = client.post(
            "/api/speech",
            json={
                "text": "今日はいい天気ですね。",
                "language": "ja",
                "model": "roxy-v2proplus",
                "expression_prompt": "slightly shy, like a quiet confession",
            },
        )

    assert response.status_code == 200
    assert response.content == b"\x00\x00\x01\x00"
    assert response.headers["x-resolved-expression"] == "shy"
    payload = captured["speech"]
    assert isinstance(payload, dict)
    assert "expression_prompt" not in payload
    assert payload["expression"] == {
        "enabled": True,
        "profile": "shy",
        "intensity": 0.35,
    }
    assert payload["generation"] == {
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "seed": 1234,
        "noise_scale": 0.5,
        "speed": 1.0,
    }


def test_webui_proxy_validates_and_forwards_generation_controls(tmp_path) -> None:
    (tmp_path / "index.html").write_text(
        "<!doctype html><title>test</title>", encoding="utf-8"
    )
    captured: dict[str, object] = {}
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream(captured))

    with TestClient(app) as client:
        response = client.post(
            "/api/speech",
            json={
                "text": "今日はいい天気ですね。",
                "language": "ja",
                "model": "roxy-v2proplus",
                "generation": {
                    "top_k": 3,
                    "temperature": 0.75,
                    "noise_scale": 0.2,
                    "seed": -1,
                    "speed": 1.0,
                },
            },
        )

    assert response.status_code == 200
    assert captured["speech"]["generation"] == {
        "top_k": 3,
        "top_p": 1.0,
        "temperature": 0.75,
        "seed": -1,
        "noise_scale": 0.2,
        "speed": 1.0,
    }


@pytest.mark.parametrize(
    "generation",
    [
        {"unknown": 1},
        {"top_k": True},
        {"top_k": 0},
        {"top_k": 51},
        {"temperature": 0},
        {"temperature": float("nan")},
        {"noise_scale": -0.1},
        {"noise_scale": 10.1},
        {"seed": True},
        {"seed": -2},
        {"speed": 0.99},
        {"speed": float("inf")},
    ],
)
def test_webui_proxy_rejects_invalid_generation_before_upstream(
    tmp_path: Path, generation: dict[str, object]
) -> None:
    (tmp_path / "index.html").write_text(
        "<!doctype html><title>test</title>", encoding="utf-8"
    )
    captured: dict[str, object] = {}
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream(captured))

    async def scenario():
        endpoint = _route_endpoint(app, "/api/speech")
        return await endpoint(
            _direct_json_request(
                app,
                "/api/speech",
                {
                    "text": "test",
                    "language": "en",
                    "model": "roxy-v2proplus",
                    "generation": generation,
                },
            )
        )

    response = asyncio.run(scenario())
    assert response.status_code == 400
    assert "speech" not in captured


def test_webui_proxy_resolves_segment_prompts_and_preserves_text(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
    captured: dict[str, object] = {}
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream(captured))

    with TestClient(app) as client:
        response = client.post(
            "/api/speech",
            json={
                "segments": [
                    {"text": "今日は、 ", "expression_prompt": "slightly shy"},
                    {"text": "戦う準備ができました。", "expression_prompt": "battle"},
                ],
                "language": "ja",
                "model": "roxy-v2proplus",
            },
        )

    assert response.status_code == 200
    payload = captured["speech"]
    assert isinstance(payload, dict)
    assert "text" not in payload
    assert "expression_prompt" not in payload
    assert "segments" in payload
    segments = payload["segments"]
    assert isinstance(segments, list)
    assert "".join(segment["text"] for segment in segments) == "今日は、 戦う準備ができました。"
    assert segments == [
        {
            "text": "今日は、 ",
            "expression": {
                "enabled": True,
                "profile": "shy",
                "intensity": 0.35,
            },
        },
        {
            "text": "戦う準備ができました。",
            "expression": {
                "enabled": True,
                "profile": "battle",
                "intensity": 0.7,
            },
        },
    ]


def test_webui_proxy_rejects_expression_switch_inside_phrase(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
    captured: dict[str, object] = {}
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream(captured))

    with TestClient(app) as client:
        response = client.post(
            "/api/speech",
            json={
                "segments": [
                    {"text": "I want ", "expression_prompt": "shy"},
                    {"text": "to explain.", "expression_prompt": "battle"},
                ],
                "language": "en",
                "model": "roxy-v2proplus",
            },
        )

    assert response.status_code == 400
    assert "speech-safe punctuation" in response.json()["error"]
    assert "speech" not in captured


@pytest.mark.parametrize(
    "body",
    [
        {
            "text": "test",
            "segments": [{"text": "test"}],
            "language": "ja",
            "model": "roxy-v2proplus",
        },
        {
            "segments": [{"text": "test", "unexpected": True}],
            "language": "ja",
            "model": "roxy-v2proplus",
        },
        {
            "segments": [{"text": "test", "expression_prompt": "unknown feeling"}],
            "language": "ja",
            "model": "roxy-v2proplus",
        },
    ],
)
def test_webui_proxy_rejects_invalid_segment_contract_before_upstream(
    tmp_path, body: dict[str, object]
) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
    captured: dict[str, object] = {}
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream(captured))

    with TestClient(app) as client:
        response = client.post("/api/speech", json=body)

    assert response.status_code == 400
    assert "speech" not in captured


def test_webui_proxy_rejects_unknown_emotion_before_upstream(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
    captured: dict[str, object] = {}
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream(captured))

    with TestClient(app) as client:
        response = client.post(
            "/api/speech",
            json={
                "text": "test",
                "language": "en",
                "model": "roxy-v2proplus",
                "expression_prompt": "sound like an unsupported character",
            },
        )

    assert response.status_code == 400
    assert "speech" not in captured


def test_webui_proxy_waits_for_cancelled_upstream_to_release(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
    captured: dict[str, object] = {}
    app = create_webui_app(
        static_dir=tmp_path,
        client=_mock_upstream(captured, busy_once=True),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/speech",
            json={
                "text": "test",
                "language": "en",
                "model": "roxy-v2proplus",
                "expression_prompt": "",
            },
        )

    assert response.status_code == 200
    assert captured["busy_returned"] is True
    assert captured["cancel_calls"] == 1
    assert isinstance(captured["speech"], dict)


def test_webui_cancel_is_idempotent_without_active_stream(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream({}))

    with TestClient(app) as client:
        response = client.post("/api/cancel")

    assert response.status_code == 200
    assert response.json() == {"cancelled": False}


def _direct_json_request(app, path: str, payload: dict[str, object]) -> Request:
    encoded = json.dumps(payload).encode("utf-8")
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.request", "body": b"", "more_body": False}
        delivered = True
        return {"type": "http.request", "body": encoded, "more_body": False}

    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 9891),
            "app": app,
        },
        receive,
    )


def _route_endpoint(app, path: str):
    return next(route.endpoint for route in app.routes if getattr(route, "path", None) == path)


def _speech_body() -> dict[str, object]:
    return {
        "text": "test",
        "language": "en",
        "model": "roxy-v2proplus",
        "expression_prompt": "",
    }


def test_webui_prefetched_response_closes_without_iterating_body(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")

    class OneChunkPCM(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = asyncio.Event()

        async def __aiter__(self):
            yield b"\x01\x00"

        async def aclose(self) -> None:
            self.closed.set()

    stream = OneChunkPCM()
    captured: dict[str, object] = {}
    app = create_webui_app(
        static_dir=tmp_path,
        client=_mock_upstream(captured, speech_stream=stream),
    )

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            endpoint = _route_endpoint(app, "/api/speech")
            response = await endpoint(_direct_json_request(app, "/api/speech", _speech_body()))
            upstream_response = captured["upstream_response"]
            assert isinstance(upstream_response, httpx.Response)
            assert response.status_code == 200
            assert app.state.speech_lock.locked() is True
            assert stream.closed.is_set() is False

            # The response owns cleanup even though body_iterator was never advanced.
            await response.aclose()
            await response.aclose()

            assert app.state.speech_lock.locked() is False
            assert app.state.active_upstream is None
            assert stream.closed.is_set() is True
            assert upstream_response.is_closed is True

    asyncio.run(scenario())


def test_webui_cancelled_response_body_releases_prefetched_upstream(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")

    class BlockingPCM(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.waiting = asyncio.Event()
            self.closed = asyncio.Event()

        async def __aiter__(self):
            yield b"\x01\x00"
            self.waiting.set()
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            self.closed.set()

    stream = BlockingPCM()
    captured: dict[str, object] = {}
    app = create_webui_app(
        static_dir=tmp_path,
        client=_mock_upstream(captured, speech_stream=stream),
    )

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            endpoint = _route_endpoint(app, "/api/speech")
            response = await endpoint(_direct_json_request(app, "/api/speech", _speech_body()))

            async def receive():
                await asyncio.Event().wait()

            async def send(_message):
                return None

            task = asyncio.create_task(
                response(
                    {
                        "type": "http",
                        "asgi": {"version": "3.0", "spec_version": "2.4"},
                        "http_version": "1.1",
                        "method": "POST",
                        "scheme": "http",
                        "path": "/api/speech",
                        "raw_path": b"/api/speech",
                        "query_string": b"",
                        "headers": [],
                    },
                    receive,
                    send,
                )
            )
            await asyncio.wait_for(stream.waiting.wait(), timeout=1.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            assert stream.closed.is_set() is True
            assert app.state.speech_lock.locked() is False
            assert app.state.active_upstream is None

            # Generator-finally and response-finally may both arrive; cleanup is idempotent.
            await response.aclose()
            assert app.state.speech_lock.locked() is False

    asyncio.run(scenario())


def test_webui_cancelled_core_cancel_still_closes_active_owner(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")

    class OneChunkPCM(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = asyncio.Event()

        async def __aiter__(self):
            yield b"\x01\x00"

        async def aclose(self) -> None:
            self.closed.set()

    stream = OneChunkPCM()
    app = create_webui_app(
        static_dir=tmp_path,
        client=_mock_upstream({}, speech_stream=stream),
    )

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            speech_endpoint = _route_endpoint(app, "/api/speech")
            response = await speech_endpoint(
                _direct_json_request(app, "/api/speech", _speech_body())
            )
            cancel_started = asyncio.Event()
            original_post = app.state.client.post

            async def blocking_post(url: str, *args, **kwargs):
                if url == "/v1/audio/cancel":
                    cancel_started.set()
                    await asyncio.Event().wait()
                return await original_post(url, *args, **kwargs)

            monkeypatch.setattr(app.state.client, "post", blocking_post)
            cancel_task = asyncio.create_task(_route_endpoint(app, "/api/cancel")())
            await asyncio.wait_for(cancel_started.wait(), timeout=1.0)
            cancel_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancel_task

            assert stream.closed.is_set() is True
            assert app.state.speech_lock.locked() is False
            assert app.state.active_upstream is None

            await response.aclose()

    asyncio.run(scenario())


def test_webui_upstream_error_read_failure_releases_owner(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")

    class FailingErrorBody(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = asyncio.Event()

        async def __aiter__(self):
            raise httpx.ReadError("error body read failed")
            yield b""  # pragma: no cover - makes this an async iterator

        async def aclose(self) -> None:
            self.closed.set()

    stream = FailingErrorBody()
    app = create_webui_app(
        static_dir=tmp_path,
        client=_mock_upstream({}, speech_stream=stream, speech_status=503),
    )

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            endpoint = _route_endpoint(app, "/api/speech")
            with pytest.raises(httpx.ReadError, match="error body read failed"):
                await endpoint(_direct_json_request(app, "/api/speech", _speech_body()))

            assert stream.closed.is_set() is True
            assert app.state.speech_lock.locked() is False
            assert app.state.active_upstream is None

    asyncio.run(scenario())


def test_webui_cancelled_upstream_error_read_releases_owner(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")

    class BlockingErrorBody(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.reading = asyncio.Event()
            self.closed = asyncio.Event()

        async def __aiter__(self):
            self.reading.set()
            await asyncio.Event().wait()
            yield b""  # pragma: no cover - cancelled before a body is yielded

        async def aclose(self) -> None:
            self.closed.set()

    stream = BlockingErrorBody()
    app = create_webui_app(
        static_dir=tmp_path,
        client=_mock_upstream({}, speech_stream=stream, speech_status=503),
    )

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            endpoint = _route_endpoint(app, "/api/speech")
            task = asyncio.create_task(
                endpoint(_direct_json_request(app, "/api/speech", _speech_body()))
            )
            await asyncio.wait_for(stream.reading.wait(), timeout=1.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            assert stream.closed.is_set() is True
            assert app.state.speech_lock.locked() is False
            assert app.state.active_upstream is None

    asyncio.run(scenario())


def test_webui_upstream_error_close_failure_still_releases_lock(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")

    class FailingCloseBody(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.close_calls = 0

        async def __aiter__(self):
            yield b'{"detail":"unavailable"}'

        async def aclose(self) -> None:
            self.close_calls += 1
            raise RuntimeError("error body close failed")

    stream = FailingCloseBody()
    app = create_webui_app(
        static_dir=tmp_path,
        client=_mock_upstream({}, speech_stream=stream, speech_status=503),
    )

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            endpoint = _route_endpoint(app, "/api/speech")
            with pytest.raises(RuntimeError, match="error body close failed"):
                await endpoint(_direct_json_request(app, "/api/speech", _speech_body()))

            assert stream.close_calls >= 1
            assert app.state.speech_lock.locked() is False
            assert app.state.active_upstream is None

    asyncio.run(scenario())


def test_workstation_api_creates_local_projects_and_jobs(tmp_path) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>", encoding="utf-8")
    store = WorkstationStore(tmp_path / "workstation")
    app = create_webui_app(
        static_dir=static,
        client=_mock_upstream({}),
        workstation=store,
    )

    with TestClient(app) as client:
        project = client.post(
            "/api/workstation/projects",
            json={
                "kind": "dataset",
                "name": "Roxy Dataset",
                "config": {"source": str(tmp_path / "audio")},
            },
        )
        job = client.post(
            "/api/workstation/jobs",
            json={
                "type": "dataset.inventory",
                "project_id": project.json()["id"],
                "parameters": {},
            },
        )
        overview = client.get("/api/workstation/overview")

    assert project.status_code == 201
    assert job.status_code == 201
    assert overview.status_code == 200
    assert overview.json()["product"] == "AnifLive-TTS Studio"
    assert overview.json()["project_counts"]["dataset"] == 1
    assert overview.json()["queued_jobs"] == 1


def test_workstation_job_control_api_exposes_dependencies_priority_and_retry(
    tmp_path: Path,
) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>", encoding="utf-8")
    app = create_webui_app(
        static_dir=static,
        client=_mock_upstream({}),
        workstation=WorkstationStore(tmp_path / "workstation"),
    )

    with TestClient(app) as client:
        project = client.post(
            "/api/workstation/projects",
            json={"kind": "dataset", "name": "Dataset", "config": {}},
        ).json()
        parent = client.post(
            "/api/workstation/jobs",
            json={
                "type": "dataset.inventory",
                "project_id": project["id"],
                "parameters": {},
                "priority": 60,
            },
        ).json()
        child_response = client.post(
            "/api/workstation/jobs",
            json={
                "type": "dataset.inventory",
                "project_id": project["id"],
                "parameters": {"stage": "child"},
                "depends_on": [parent["id"]],
                "priority": -20,
            },
        )
        child = child_response.json()
        paused = client.post(f"/api/workstation/jobs/{child['id']}/pause")
        resumed = client.post(f"/api/workstation/jobs/{child['id']}/resume")
        cancelled = client.post(f"/api/workstation/jobs/{child['id']}/cancel")
        retried = client.post(f"/api/workstation/jobs/{child['id']}/retry")

    assert child_response.status_code == 201
    assert child["depends_on"] == [parent["id"]]
    assert child["priority"] == -20
    assert paused.status_code == 200 and paused.json()["status"] == "paused"
    assert resumed.status_code == 200 and resumed.json()["status"] == "queued"
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    assert retried.status_code == 201
    assert retried.json()["retry_of"] == child["id"]
    assert retried.json()["attempt"] == 2
    assert retried.json()["depends_on"] == [parent["id"]]
    assert retried.json()["priority"] == -20


def test_workstation_api_exposes_job_events_and_artifact_lineage(tmp_path) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>", encoding="utf-8")
    store = WorkstationStore(tmp_path / "workstation")
    dataset_path = store.artifact_root / "datasets" / "roxy-clean-v8.json"
    checkpoint_path = store.artifact_root / "checkpoints" / "epoch-15.ckpt"
    dataset_path.parent.mkdir(parents=True)
    checkpoint_path.parent.mkdir(parents=True)
    dataset_path.write_bytes(b"dataset")
    checkpoint_path.write_bytes(b"checkpoint")
    app = create_webui_app(
        static_dir=static,
        client=_mock_upstream({}),
        workstation=store,
    )

    with TestClient(app) as client:
        project = client.post(
            "/api/workstation/projects",
            json={"kind": "training", "name": "Roxy training", "config": {}},
        ).json()
        job = client.post(
            "/api/workstation/jobs",
            json={
                "type": "training.prepare",
                "project_id": project["id"],
                "parameters": {},
            },
        ).json()
        detail = client.get(f"/api/workstation/jobs/{job['id']}")
        dataset = client.post(
            "/api/workstation/artifacts",
            json={
                "type": "dataset", "name": "roxy-clean-v8", "status": "ready",
                "local_path": str(dataset_path),
            },
        ).json()
        checkpoint = client.post(
            "/api/workstation/artifacts",
            json={
                "type": "checkpoint",
                "name": "epoch-15",
                "status": "ready",
                "project_id": project["id"],
                "local_path": str(checkpoint_path),
                "parent_artifact_ids": [dataset["id"]],
            },
        )
        artifacts = client.get("/api/workstation/artifacts?status=ready")

    assert detail.status_code == 200
    assert detail.json()["job"]["id"] == job["id"]
    assert detail.json()["logs"][-1]["message"] == "Job queued"
    assert checkpoint.status_code == 201
    assert checkpoint.json()["parent_artifact_ids"] == [dataset["id"]]
    assert {item["type"] for item in artifacts.json()["data"]} == {
        "dataset",
        "checkpoint",
    }


def test_workstation_expression_draft_api_supports_strict_local_crud(tmp_path) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>", encoding="utf-8")
    store = WorkstationStore(tmp_path / "workstation")
    reference = store.artifact_root / "references" / "bright.wav"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    app = create_webui_app(
        static_dir=static,
        client=_mock_upstream({}),
        workstation=store,
    )

    with TestClient(app) as client:
        created_response = client.post(
            "/api/workstation/expression-drafts",
            json={
                "name": "Bright relief",
                "profile_id": "bright-relief",
                "model_id": "voice-v2proplus",
                "reference_path": str(reference),
                "language": "yue",
                "emotion": "relieved",
                "intensity": 0.72,
                "descriptions": ["Warm relief after tension"],
                "vad": {"valence": 0.7, "arousal": 0.1, "dominance": 0.2},
                "prosody": {"speaking_rate": 3.8},
            },
        )
        assert created_response.status_code == 201
        created = created_response.json()

        listed = client.get(
            "/api/workstation/expression-drafts",
            params={"model_id": "voice-v2proplus", "language": "yue"},
        )
        detail = client.get(
            f"/api/workstation/expression-drafts/{created['id']}"
        )
        immutable = client.patch(
            f"/api/workstation/expression-drafts/{created['id']}",
            json={"profile_id": "rewritten"},
        )
        updated = client.patch(
            f"/api/workstation/expression-drafts/{created['id']}",
            json={
                "name": "Bright relief, reviewed",
                "intensity": 0.81,
                "descriptions": ["Released tension", "Warm and assured"],
                "vad": {"valence": 0.8, "arousal": 0.2, "dominance": 0.3},
                "prosody": {"speaking_rate": 3.6, "energy_db": -17.2},
            },
        )
        forbidden_qualification = client.patch(
            f"/api/workstation/expression-drafts/{created['id']}",
            json={"qualification_status": "qualified"},
        )
        deleted = client.delete(
            f"/api/workstation/expression-drafts/{created['id']}"
        )
        missing = client.get(
            f"/api/workstation/expression-drafts/{created['id']}"
        )
        invalid_filter = client.get(
            "/api/workstation/expression-drafts",
            params={"qualification_status": "released"},
        )

    assert created["profile_id"] == "bright-relief"
    assert created["reference"] == {
        "scope": "artifact",
        "path": "references/bright.wav",
        "sha256": created["reference"]["sha256"],
    }
    assert str(tmp_path) not in json.dumps(created)
    assert listed.status_code == 200
    assert listed.json()["source"] == "local-workstation"
    assert [record["id"] for record in listed.json()["data"]] == [created["id"]]
    assert detail.status_code == 200
    assert immutable.status_code == 400
    assert "immutable" in immutable.json()["error"].lower()
    assert updated.status_code == 200
    assert updated.json()["qualification_status"] == "draft"
    assert updated.json()["intensity"] == 0.81
    assert updated.json()["prosody"]["energy_db"] == -17.2
    assert forbidden_qualification.status_code == 400
    assert "evaluation evidence" in forbidden_qualification.json()["error"]
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert missing.status_code == 404
    assert invalid_filter.status_code == 400


def test_workstation_qualification_api_imports_evidence_and_promotes(
    tmp_path: Path,
) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>", encoding="utf-8")
    store = WorkstationStore(tmp_path / "workstation")
    engine_path = store.artifact_root / "engines" / "candidate.plan"
    engine_path.parent.mkdir(parents=True)
    engine_path.write_bytes(b"engine")
    engine = store.register_artifact(
        artifact_type="engine",
        name="Candidate engine",
        status="ready",
        local_path=engine_path,
    )
    subject = {"kind": "artifact", "id": engine["id"]}

    def register_evidence(
        name: str,
        payload: dict[str, Any],
        *,
        parents: list[str] | None = None,
        evidence_kind: str,
    ) -> dict[str, Any]:
        path = store.artifact_root / "evaluations" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return store.register_artifact(
            artifact_type="evaluation",
            name=name,
            status="ready",
            local_path=path,
            parent_artifact_ids=parents,
            metadata={"qualification_evidence_kind": evidence_kind},
        )

    languages = {
        language: {"quality_gate": {"passed": True}}
        for language in ("zh", "yue", "en", "ja", "ko")
    }
    automated = register_evidence(
        "evaluation-report.json",
        {
            "schema": "aniflive-tts-workstation-evaluation-v1",
            "runtime": {"backend": "TensorRT-11", "pytorch_neural_fallback": False},
            "languages": languages,
            "benchmark": {"session_records": [{"session": 1}]},
            "baseline": {
                "available": True,
                "languages": {
                    language: {"content_passed": True, "speaker_passed": True}
                    for language in languages
                },
                "performance": {
                    "stream_keepalive_audible_ttfa_p50_ms": {"passed": True},
                    "stream_keepalive_audible_ttfa_p95_ms": {"passed": True},
                    "wall_rtf_p50": {"passed": True},
                },
            },
            "gates": {
                "tensor_rt_engine_contract": True,
                "five_language_enqueue": True,
                "no_pytorch_neural_fallback": True,
                "stream_complete_quality": True,
                "canonical_benchmark_completed": True,
                "baseline_regression": True,
            },
        },
        parents=[engine["id"]],
        evidence_kind="automated-evaluation",
    )

    def blind(gate: str, kind: str) -> dict[str, Any]:
        return register_evidence(
            gate,
            {
                "schema": "aniflive-tts-blind-ab-evidence-v1",
                "subject": subject,
                "gate": gate,
                "decision": "passed",
                "protocol": {
                    "blinded": True,
                    "comparison": "candidate-vs-baseline",
                    "completed_trials": 8,
                    "listener_count": 1,
                    "sample_manifest_sha256": "a" * 64,
                    "randomization_sha256": "b" * 64,
                },
                "summary": f"{gate} blind A/B passed",
                "operator": "listener-1",
                "recorded_at": "2026-08-31T12:00:00Z",
            },
            evidence_kind=kind,
        )

    long_form = blind("long-form-continuity", "long-form-blind-ab")
    expression = blind("expression-quality", "expression-blind-ab")
    security = register_evidence(
        "security-verification",
        {
            "schema": "aniflive-tts-security-verification-v1",
            "subject": subject,
            "tool": {"name": "check_release_security.py", "version": "1"},
            "release_tree_sha256": "c" * 64,
            "checks": [
                {"id": check_id, "status": "passed", "summary": f"{check_id} passed"}
                for check_id in (
                    "release-inventory",
                    "private-asset-scan",
                    "credential-scan",
                    "action-pin-review",
                    "public-webui-state",
                    "oci-source-metadata",
                )
            ],
            "completed_at": "2026-08-31T12:00:00Z",
        },
        evidence_kind="security-verification",
    )
    app = create_webui_app(
        static_dir=static,
        client=_mock_upstream({}),
        workstation=store,
    )

    with TestClient(app) as client:
        composed = client.post(
            "/api/workstation/qualifications/compose",
            json={
                "subject_kind": "artifact",
                "subject_id": engine["id"],
                "automated_evaluation_artifact_id": automated["id"],
                "long_form_evidence_artifact_id": long_form["id"],
                "expression_evidence_artifact_id": expression["id"],
                "security_evidence_artifact_id": security["id"],
            },
        )
        imported = composed.json()["qualification"]
        listed = client.get("/api/workstation/qualifications")
        promoted = client.post(
            f"/api/workstation/artifacts/{engine['id']}/promote",
            json={"qualification_id": imported["id"]},
        )
        details = client.get(f"/api/workstation/artifacts/{engine['id']}")
        malformed = client.post(
            f"/api/workstation/artifacts/{engine['id']}/promote",
            json={"qualification_id": imported["id"], "override": True},
        )

    assert composed.status_code == 201
    assert imported["overall_status"] == "passed"
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["data"]] == [imported["id"]]
    assert promoted.status_code == 200
    assert promoted.json()["promoted"] is True
    assert details.status_code == 200
    assert details.json()["artifact"]["promotion"]["qualification_id"] == imported["id"]
    assert details.json()["qualifications"][0]["gates"][0]["status"] == "passed"
    assert malformed.status_code == 409


def test_workstation_expression_draft_api_rejects_unknown_create_fields(tmp_path) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>", encoding="utf-8")
    app = create_webui_app(
        static_dir=static,
        client=_mock_upstream({}),
        workstation=WorkstationStore(tmp_path / "workstation"),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/workstation/expression-drafts",
            json={
                "name": "Unknown field",
                "profile_id": "unknown-field",
                "language": "en",
                "emotion": "neutral",
                "intensity": 0.5,
                "runtime_command": "do-not-accept",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"].endswith("runtime_command")


def test_webui_asset_uses_current_contract_without_legacy_hardcoding() -> None:
    html = (Path(__file__).parents[1] / "webui" / "synthesis.html").read_text(
        encoding="utf-8"
    )

    assert 'id="annotationEditor"' in html
    assert 'id="annotationMirror"' in html
    assert 'id="expressionMenu"' in html
    assert 'id="expressionCards"' in html
    assert 'id="expressionPrompt"' not in html
    assert "annotationEditor.buildSegments()" in html
    assert '"/api/sessions"' in html
    assert "for (const segment of plan.segments)" in html
    assert "await postSessionJson(`${basePath}/flush`, {}, controller.signal)" in html
    assert 'fetch("/api/status"' in html
    assert 'fetch("/api/models/activate"' in html
    assert '`/api/sessions/${encodeURIComponent(activeSessionId)}/cancel`' in html
    assert ': "/api/cancel"' in html
    assert "/assets/everynight_dance.gif" in html
    assert '<script src="/assets/playback_model.js?v=1.4.0-editorial20"></script>' in html
    assert '<script src="/assets/annotation_editor.js?v=1.4.0-editorial20"></script>' in html
    assert "aniflive-tts-${safeModel}-${language}-${stamp}.wav" in html
    assert "--playback-gold:" in html
    assert ".playback-active-character" in html
    assert ".playback-mirror-content" in html
    assert "startPlaybackIndicator(sourceText, state.language, context, runId)" in html
    assert "activeTextRangeAtAudioTime" in html
    assert "buildPcmActivityTimeline" in html
    assert "elapsed / duration" not in html
    assert "const timing = schedulePcm(context, chunk, metrics)" in html
    assert 'el("text").addEventListener("input", clearPlaybackIndicator)' in html
    assert "v1.1" not in html
    assert "/static/" not in html

    editor = (Path(__file__).parents[1] / "webui" / "annotation_editor.js").read_text(
        encoding="utf-8"
    )
    set_playback = editor[editor.index("setPlaybackRange"):editor.index("hasAnnotations")]
    assert "renderPlayback()" in set_playback
    assert "renderMirror()" not in set_playback


def test_workstation_shell_preserves_product_name_and_synthesis_module() -> None:
    root = Path(__file__).parents[1] / "webui"
    html = (root / "index.html").read_text(encoding="utf-8")
    style = (root / "studio.css").read_text(encoding="utf-8")
    script = (root / "studio.js").read_text(encoding="utf-8")

    assert "<title>AnifLive-TTS Studio</title>" in html
    assert '<span class="overview-title-core">AnifLive-TTS</span>' in html
    assert '<span class="overview-title-suffix">Studio</span>' in html
    assert 'data-view-panel="synthesis"' in html
    assert 'src="/synthesis?embedded=1&amp;v=' in html
    assert '<canvas id="voicePulse"' not in html
    assert 'id="refreshButton"' not in html
    assert 'id="studioLocalePicker"' in html
    assert 'data-lucide="globe-2"' in html
    assert 'data-locale="zh-Hant"' in html
    assert 'data-locale="zh-Hans"' in html
    assert '<span>01</span> CREATE' in html
    assert '<h1>Speech Synthesis</h1>' in html
    assert 'id="tseAudioPlayer"' in html
    assert 'id="tseAudioToggle"' in html
    assert 'id="tseAudioSeek"' in html
    assert 'id="tseAudioMute"' in html
    assert 'class="rail-footer"' not in html
    assert 'data-lucide="audio-waveform"' in html
    assert '/assets/lucide.min.js?v=1.8.0' in html
    assert "Dataset Factory" in html
    assert "Target Speaker Extraction" in html
    assert "Evaluation Lab" in html
    assert "Model Registry" in html
    assert 'id="expressionDraftRows"' in html
    assert 'id="expressionDraftForm"' in html
    assert 'id="expressionQualificationEvidence"' in html
    assert 'id="expressionPromoteButton"' in html
    assert 'id="evaluationGateList"' in html
    assert 'id="evaluationArtifactRows"' in html
    assert 'id="artifactLineage"' in html
    assert 'id="artifactPromotionEvidence"' in html
    assert 'id="artifactPromoteButton"' in html
    assert 'id="engineArtifactRows"' in html
    assert '<option value="qualified">' not in html
    assert 'aria-label="AnifLive-TTS Studio expression inspector"' in html
    assert "brand-mark" not in html
    # Subtle gradients are limited to actionable/elevated UI surfaces.
    # The editorial canvas and ambient background remain unchanged.
    gradient_rules = re.findall(r"([^{}]+)\{([^{}]*gradient[^{}]*)\}", style)
    assert all(
        ".studio-ux" in selector
        and ("primary-button" in selector or "glass-panel" in selector)
        for selector, _ in gradient_rules
    )
    assert "letter-spacing: -" not in style
    assert "aniflive-tts:locale-changed" in script
    assert 'api("/api/workstation/expression-drafts")' in script
    assert 'api("/api/workstation/qualifications")' in script
    assert 'api("/api/workstation/qualifications/import"' in script
    assert '}/promote`' in script
    assert "qualification_status: \"qualified\"" not in script
    assert 'definition("Source", "AnifLive-TTS Studio model package")' in script
    assert "setVoicePulseActive(running.length > 0 || state.playbackActive)" in script
    assert "innerHTML" not in script


def test_webui_launcher_discovers_only_usable_python_and_api_ports() -> None:
    launcher_path = Path(__file__).parents[1] / "run_webui.bat"
    launcher_bytes = launcher_path.read_bytes()
    launcher = launcher_bytes.decode("utf-8")

    assert launcher_bytes.count(b"\n") == launcher_bytes.count(b"\r\n")
    assert "if defined ANIFLIVE_TTS_PYTHON" in launcher
    assert r'if exist ".venv\Scripts\python.exe"' in launcher
    assert launcher.index("if defined ANIFLIVE_TTS_PYTHON") < launcher.index(
        r'if exist ".venv\Scripts\python.exe"'
    )
    assert "where python.exe" in launcher
    assert "where py.exe" in launcher
    assert "for %%V in (-3.12 -3.11 -3.10)" in launcher
    assert ":validate_python" in launcher
    assert "import fastapi, httpx, pydantic, uvicorn" in launcher
    assert "import aniflive_tts.webui" in launcher
    assert 'call "%ANIFLIVE_TTS_PYTHON_EXE%" %ANIFLIVE_TTS_PYTHON_ARGS%' in launcher
    assert "No usable WebUI Python environment was found." in launcher
    assert "No packages were installed or changed." in launcher
    assert "if not defined ANIFLIVE_TTS_WEBUI_UPSTREAM" in launcher
    assert "for %%P in (9880 9882)" in launcher
    assert "pip install" not in launcher.lower()
    assert "taskkill" not in launcher.lower()


def test_webui_playback_model_asset_is_served(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (tmp_path / "playback_model.js").write_text(
        "globalThis.testPlaybackModel = true;", encoding="utf-8"
    )
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream({}))

    with TestClient(app) as client:
        response = client.get("/assets/playback_model.js")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert response.text == "globalThis.testPlaybackModel = true;"


def test_webui_annotation_editor_asset_is_served(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (tmp_path / "annotation_editor.js").write_text(
        "globalThis.testAnnotationEditor = true;", encoding="utf-8"
    )
    app = create_webui_app(static_dir=tmp_path, client=_mock_upstream({}))

    with TestClient(app) as client:
        response = client.get("/assets/annotation_editor.js")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert response.text == "globalThis.testAnnotationEditor = true;"


def test_webui_vendored_lucide_asset_is_served_without_a_cdn(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (tmp_path / "lucide.min.js").write_text(
        "globalThis.lucide = { createIcons() {} };", encoding="utf-8"
    )
    app = create_webui_app(
        static_dir=tmp_path,
        client=_mock_upstream({}),
        workstation=WorkstationStore(tmp_path / "workstation"),
    )

    with TestClient(app) as client:
        response = client.get("/assets/lucide.min.js")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert "max-age=31536000" in response.headers["cache-control"]
    assert "createIcons" in response.text


def test_webui_job_control_policy_asset_is_served_from_fixed_route(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (tmp_path / "job_controls.js").write_text(
        "globalThis.AnifLiveTTSJobControls = {};", encoding="utf-8"
    )
    app = create_webui_app(
        static_dir=tmp_path,
        client=_mock_upstream({}),
        workstation=WorkstationStore(tmp_path / "workstation"),
    )

    with TestClient(app) as client:
        response = client.get("/assets/job_controls.js")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert response.headers["cache-control"] == "no-store"
    assert "AnifLiveTTSJobControls" in response.text


def test_webui_voice_workstation_media_is_served_from_fixed_routes(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")
    media = tmp_path / "media"
    media.mkdir()
    video_bytes = b"\x00\x00\x00\x18ftypmp42"
    poster_bytes = b"\xff\xd8\xff\xe0poster\xff\xd9"
    (media / "voice-workstation-background.mp4").write_bytes(video_bytes)
    (media / "voice-workstation-poster.jpg").write_bytes(poster_bytes)
    (media / "private.txt").write_text("not publicly routed", encoding="utf-8")
    app = create_webui_app(
        static_dir=tmp_path,
        client=_mock_upstream({}),
        workstation=WorkstationStore(tmp_path / "workstation"),
    )

    with TestClient(app) as client:
        video = client.get("/media/voice-workstation-background.mp4")
        poster = client.get("/media/voice-workstation-poster.jpg")
        unknown = client.get("/media/private.txt")

    assert video.status_code == 200
    assert video.content == video_bytes
    assert video.headers["content-type"].startswith("video/mp4")
    assert "max-age=31536000" in video.headers["cache-control"]
    assert "immutable" in video.headers["cache-control"]
    assert poster.status_code == 200
    assert poster.content == poster_bytes
    assert poster.headers["content-type"].startswith("image/jpeg")
    assert "max-age=31536000" in poster.headers["cache-control"]
    assert "immutable" in poster.headers["cache-control"]
    assert unknown.status_code == 404


def test_annotation_editor_expression_boundaries_match_api_contract() -> None:
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    source = Path(__file__).parents[1] / "webui" / "annotation_editor.js"
    script = (
        "require('vm').runInThisContext(require('fs').readFileSync(process.argv[1], 'utf8'));"
        "const b=globalThis.AnifLiveTTSExpressionBoundaries;"
        "console.log(JSON.stringify(["
        "b.isSafeExpressionRange('今日は、続けます。',0,4),"
        "b.isSafeExpressionRange('今日は、続けます。',0,3),"
        "b.isSafeExpressionRange('I want to explain.',0,7),"
        "b.isSafeExpressionRange('I want to explain.',0,18)]));"
    )
    result = subprocess.run(
        [node, "-e", script, str(source)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert json.loads(result.stdout) == [True, False, False, True]
