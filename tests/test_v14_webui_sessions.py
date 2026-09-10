from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from aniflive_tts.webui import create_webui_app
from aniflive_tts.workstation import WorkstationStore


EXPRESSION_METADATA = {
    "enabled": True,
    "default": "neutral",
    "profiles": [
        {
            "id": "shy",
            "intensity_levels": [0.35, 0.7, 0.85],
            "languages": ["ja", "zh", "yue", "en", "ko"],
        }
    ],
    "policies": ["semantic-style"],
}


@pytest.fixture(autouse=True)
def _trust_test_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANIFLIVE_TTS_WEBUI_TRUSTED_HOSTS", "testserver")


def _session_upstream(captured: dict[str, object]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/openapi.json":
            return httpx.Response(200, json={"info": {"version": "1.4.0-dev"}})
        if path == "/health":
            return httpx.Response(
                200,
                json={
                    "ready": True,
                    "backend": "TensorRT-11",
                    "engine_count": 9,
                    "model": "roxy-v2proplus",
                },
            )
        if path == "/model/config":
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
        if path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "roxy-v2proplus", "active": True}],
                },
            )
        if path == "/v1/expressions":
            return httpx.Response(200, json={"object": "list", **EXPRESSION_METADATA})
        if path == "/v1/sessions" and request.method == "POST":
            captured["create"] = json.loads(request.content)
            return httpx.Response(
                201,
                json={
                    "id": "sess_test",
                    "state": "open",
                    "model": "roxy-v2proplus",
                    "voice_profile": "default",
                },
            )
        if path == "/v1/sessions/sess_test/segments" and request.method == "POST":
            segments = captured.setdefault("segments", [])
            assert isinstance(segments, list)
            payload = json.loads(request.content)
            segments.append(payload)
            return httpx.Response(
                201,
                json={
                    "segment_id": payload["segment_id"],
                    "sequence": len(segments) - 1,
                    "state": "queued",
                },
            )
        if path == "/v1/sessions/sess_test/flush" and request.method == "POST":
            captured["flush"] = True
            return httpx.Response(200, json={"id": "sess_test", "state": "flushing"})
        if path == "/v1/sessions/sess_test/cancel" and request.method == "POST":
            captured["cancel"] = True
            return httpx.Response(200, json={"id": "sess_test", "state": "cancelled"})
        if path == "/v1/sessions/sess_test/audio" and request.method == "GET":
            captured["audio"] = True
            return httpx.Response(
                200,
                content=b"\x01\x00\x02\x00\x03\x00",
                headers={
                    "x-tensorrt-backend": "TensorRT-11",
                    "x-tensorrt-engine-count": "9",
                    "x-pytorch-fallback": "false",
                    "x-tts-model": "roxy-v2proplus",
                    "x-tts-stream": "pcm_s16le",
                    "x-tts-sample-rate": "32000",
                    "x-tts-channels": "1",
                    "x-tts-session-id": "sess_test",
                    "x-tts-session-context": "committed-neural-v1",
                    "x-tts-session-context-policy": "A",
                    "x-tts-neural-state-continuity": "false",
                    "x-tts-acoustic-latent-continuity": "false",
                    "x-tts-continuity-qualification": "experimental-unqualified",
                },
            )
        if path == "/v1/audio/cancel":
            return httpx.Response(200, json={"cancelled": False})
        raise AssertionError(f"Unexpected upstream request: {request.method} {path}")

    return httpx.AsyncClient(
        base_url="http://upstream.test",
        transport=httpx.MockTransport(handler),
    )


def _app(tmp_path: Path, captured: dict[str, object]):
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")
    return create_webui_app(
        static_dir=tmp_path,
        client=_session_upstream(captured),
        workstation=WorkstationStore(tmp_path / "workstation"),
    )


def test_webui_session_proxy_preserves_order_and_resolves_each_segment(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    app = _app(tmp_path, captured)
    generation = {
        "top_k": 7,
        "temperature": 0.8,
        "noise_scale": 0.3,
        "seed": 42,
        "speed": 1.0,
    }

    with TestClient(app) as client:
        created = client.post(
            "/api/sessions",
            json={"model": "roxy-v2proplus", "voice_profile": "default"},
        )
        first = client.post(
            "/api/sessions/sess_test/segments",
            json={
                "segment_id": "seg_01",
                "text": "你好，",
                "language": "yue",
                "expression_prompt": "shy 35%",
                "generation": generation,
                "paragraph_id": "webui_paragraph_1",
                "pause_after_ms": 120,
            },
        )
        second = client.post(
            "/api/sessions/sess_test/segments",
            json={
                "segment_id": "seg_02",
                "text": "今天很高興見到你。",
                "language": "zh",
                "expression_prompt": None,
                "generation": generation,
                "paragraph_id": "webui_paragraph_1",
                "pause_after_ms": 0,
            },
        )
        flushed = client.post("/api/sessions/sess_test/flush", json={})

    assert created.status_code == 201
    assert first.status_code == 201
    assert second.status_code == 201
    assert flushed.status_code == 200
    assert captured["create"] == {
        "model": "roxy-v2proplus",
        "voice_profile": "default",
        "continuity_policy": "A",
    }
    segments = captured["segments"]
    assert isinstance(segments, list)
    assert [segment["segment_id"] for segment in segments] == ["seg_01", "seg_02"]
    assert "".join(segment["text"] for segment in segments) == "你好，今天很高興見到你。"
    assert [segment["language"] for segment in segments] == ["yue", "zh"]
    assert [segment["pause_after_ms"] for segment in segments] == [120, 0]
    assert segments[0]["expression"] == {
        "enabled": True,
        "profile": "shy",
        "intensity": 0.35,
    }
    assert segments[1]["expression"] == {"enabled": False}
    assert all("expression_prompt" not in segment for segment in segments)


def test_webui_session_audio_is_pcm_stream_with_honest_context_header(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    app = _app(tmp_path, captured)

    with TestClient(app) as client:
        response = client.get("/api/sessions/sess_test/audio")
        assert app.state.speech_lock.locked() is False
        assert app.state.active_upstream is None

    assert response.status_code == 200
    assert response.content == b"\x01\x00\x02\x00\x03\x00"
    assert response.headers["x-tensorrt-backend"] == "TensorRT-11"
    assert response.headers["x-tts-session-context"] == "committed-neural-v1"
    assert response.headers["x-tts-session-context-policy"] == "A"
    assert response.headers["x-tts-neural-state-continuity"] == "false"
    assert response.headers["x-tts-acoustic-latent-continuity"] == "false"
    assert response.headers["x-tts-recommended-prebuffer-ms"] == "32"
    assert captured["audio"] is True


def test_webui_session_proxy_forwards_explicit_continuity_policy(tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    app = _app(tmp_path, captured)

    with TestClient(app) as client:
        response = client.post(
            "/api/sessions",
            json={
                "model": "roxy-v2proplus",
                "voice_profile": "default",
                "continuity_policy": "f",
            },
        )
        invalid = client.post(
            "/api/sessions",
            json={"model": "roxy-v2proplus", "continuity_policy": "Z"},
        )

    assert response.status_code == 201
    assert captured["create"] == {
        "model": "roxy-v2proplus",
        "voice_profile": "default",
        "continuity_policy": "F",
    }
    assert invalid.status_code == 400


def test_webui_session_cancel_forwards_to_the_bound_session(tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    app = _app(tmp_path, captured)

    with TestClient(app) as client:
        response = client.post("/api/sessions/sess_test/cancel", json={})

    assert response.status_code == 200
    assert response.json()["state"] == "cancelled"
    assert captured["cancel"] is True


def test_webui_session_proxy_rejects_invalid_ids_and_fake_segment_fields(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    app = _app(tmp_path, captured)

    with TestClient(app) as client:
        invalid_id = client.post(
            "/api/sessions/sess_test/segments",
            json={
                "segment_id": "../escape",
                "text": "hello",
                "language": "en",
            },
        )
        fake_control = client.post(
            "/api/sessions/sess_test/segments",
            json={
                "segment_id": "seg_01",
                "text": "hello",
                "language": "en",
                "speed": 1.5,
            },
        )

    assert invalid_id.status_code == 400
    assert fake_control.status_code == 400
    assert "segments" not in captured
