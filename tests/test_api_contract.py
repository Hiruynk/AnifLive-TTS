from __future__ import annotations

import asyncio
import importlib
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request


def _client(monkeypatch):
    monkeypatch.setenv("ANIFLIVE_TTS_MODEL_ID", "test-v2pp")
    monkeypatch.setenv("ANIFLIVE_TTS_VOICE_PROFILE", "default")
    monkeypatch.setenv("ANIFLIVE_TTS_REFERENCE_TEXT", "reference")
    monkeypatch.setenv("ANIFLIVE_TTS_REFERENCE_LANGUAGE", "en")
    module = importlib.import_module("aniflive_tts.service")
    monkeypatch.setattr(module.SERVICE, "load", lambda: None)
    monkeypatch.setattr(module.SERVICE, "unload", lambda: None)
    return TestClient(module.app)


def test_capabilities_reserve_expression_interface(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        response = client.get("/v1/capabilities")
    assert response.status_code == 200
    assert response.json()["expression"] == {
        "native": True,
        "controlled_profiles": False,
        "continuous_vector": False,
        "segmented": False,
    }


def test_expression_enabled_rejects_package_without_profiles(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-v2pp",
                "voice_profile": "default",
                "text": "hello",
                "language": "en",
                "stream": False,
                "expression": {"enabled": True, "profile": "calm", "intensity": 0.5},
            },
        )
    assert response.status_code == 400
    assert "unavailable" in response.json()["message"]


def test_expression_request_reaches_runtime_with_structured_controls(monkeypatch) -> None:
    module = importlib.import_module("aniflive_tts.service")
    captured = []
    monkeypatch.setattr(module.SERVICE, "_sample_rate", 32000)
    monkeypatch.setattr(module.SERVICE, "validate_expression", captured.append)
    monkeypatch.setattr(module.SERVICE, "stream_pcm", lambda options: iter((b"\x01\x00",)))

    with _client(monkeypatch) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-v2pp",
                "voice_profile": "default",
                "text": "hello",
                "language": "en",
                "stream": True,
                "expression": {
                    "enabled": True,
                    "profile": "shy",
                    "intensity": 0.7,
                    "policy": "identity-lock",
                },
            },
        )

    assert response.status_code == 200
    assert captured[-1].expression_profile == "shy"
    assert captured[-1].expression_intensity == 0.7
    assert captured[-1].expression_policy is module.ConditioningPolicy.IDENTITY_LOCK
    assert response.headers["x-tts-expression"] == "shy"
    assert response.headers["x-tts-expression-policy"] == "identity-lock"


def test_expression_request_inherits_active_package_preferred_policy(monkeypatch) -> None:
    module = importlib.import_module("aniflive_tts.service")
    captured = []
    monkeypatch.setattr(module.SERVICE, "_sample_rate", 32000)
    monkeypatch.setattr(module.SERVICE, "validate_expression", captured.append)
    monkeypatch.setattr(module.SERVICE, "stream_pcm", lambda options: iter((b"\x01\x00",)))
    monkeypatch.setattr(
        module.SERVICE,
        "preferred_expression_policy",
        lambda **_: module.ConditioningPolicy.SEMANTIC_STYLE,
    )
    monkeypatch.setattr(
        module.SERVICE,
        "expression_metadata",
        lambda: {"runtime_policy": {"first_context_tokens": 64}},
    )

    with _client(monkeypatch) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-v2pp",
                "voice_profile": "default",
                "text": "hello",
                "language": "en",
                "stream": True,
                "expression": {
                    "enabled": True,
                    "profile": "shy",
                    "intensity": 0.7,
                },
            },
        )

    assert response.status_code == 200
    assert captured[-1].expression_policy is module.ConditioningPolicy.SEMANTIC_STYLE
    assert response.headers["x-tts-expression-policy"] == "semantic-style"
    assert response.headers["x-tts-first-context-tokens"] == "64"


def test_segmented_expression_request_is_structured_and_inherits_defaults(monkeypatch) -> None:
    module = importlib.import_module("aniflive_tts.service")
    captured = []
    monkeypatch.setattr(module.SERVICE, "_sample_rate", 32000)
    monkeypatch.setattr(module.SERVICE, "validate_expression", captured.append)
    monkeypatch.setattr(module.SERVICE, "stream_pcm", lambda options: iter((b"\x01\x00",)))

    with _client(monkeypatch) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-v2pp",
                "voice_profile": "default",
                "segments": [
                    {"text": "I was worried,", "expression": {"profile": "shy"}},
                    {
                        "text": "but now I am ready.",
                        "expression": {"profile": "battle", "intensity": 0.9},
                    },
                ],
                "language": "en",
                "stream": True,
                "expression": {
                    "enabled": True,
                    "intensity": 0.6,
                    "policy": "identity-lock",
                },
            },
        )

    assert response.status_code == 200
    options = captured[-1]
    assert [segment.text for segment in options.expression_segments] == [
        "I was worried,",
        "but now I am ready.",
    ]
    assert [segment.profile for segment in options.expression_segments] == [
        "shy",
        "battle",
    ]
    assert options.expression_segments[0].intensity == 0.6
    assert all(
        segment.policy is module.ConditioningPolicy.IDENTITY_LOCK
        for segment in options.expression_segments
    )


def test_expression_segments_preserve_text_and_merge_identical_controls(monkeypatch) -> None:
    module = importlib.import_module("aniflive_tts.service")
    segments = module._expression_segments(
        [
            {"text": "Hello, ", "expression": {"enabled": False}},
            {"text": "world!", "expression": {"enabled": False}},
        ],
        default_enabled=False,
        default_profile=None,
        default_intensity=0.5,
        default_policy=module.ConditioningPolicy.SEMANTIC_STYLE,
    )

    assert len(segments) == 1
    assert segments[0].text == "Hello, world!"


def test_expression_switch_rejects_mid_phrase_segments() -> None:
    module = importlib.import_module("aniflive_tts.service")
    segments = (
        module.ExpressionSegment(text="I want ", enabled=True, profile="shy"),
        module.ExpressionSegment(text="to explain.", enabled=True, profile="battle"),
    )

    with pytest.raises(module.RequestError, match="speech-safe punctuation"):
        module._validate_expression_segment_boundaries(segments)


def test_expression_switch_accepts_complete_clauses() -> None:
    module = importlib.import_module("aniflive_tts.service")
    segments = (
        module.ExpressionSegment(text="I was worried, ", enabled=True, profile="shy"),
        module.ExpressionSegment(text="but now I am ready.", enabled=True, profile="battle"),
    )

    module._validate_expression_segment_boundaries(segments)


def test_canonical_api_rejects_text_and_segments_together(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "text": "hello",
                "segments": [{"text": "world"}],
                "language": "en",
            },
        )
    assert response.status_code == 400
    assert "Exactly one of text or segments" in response.json()["message"]


def test_segment_expression_rejects_reference_overrides(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "segments": [
                    {
                        "text": "hello",
                        "expression": {
                            "profile": "shy",
                            "reference_audio": "C:/private.wav",
                        },
                    }
                ],
                "language": "en",
            },
        )
    assert response.status_code == 400
    assert "unsupported fields" in response.json()["message"]


def test_stream_transport_declares_little_endian_pcm(monkeypatch) -> None:
    module = importlib.import_module("aniflive_tts.service")
    monkeypatch.setattr(module.SERVICE, "_sample_rate", 32000)
    monkeypatch.setattr(module.SERVICE, "stream_pcm", lambda options: iter((b"\x01\x00",)))

    with _client(monkeypatch) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-v2pp",
                "voice_profile": "default",
                "text": "hello",
                "language": "en",
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["x-tts-stream"] == "pcm_s16le"
    assert response.headers["x-tts-sample-format"] == "s16le"
    assert response.headers["x-tts-sample-rate"] == "32000"
    assert response.headers["x-tts-channels"] == "1"
    assert response.headers["x-tts-first-context-tokens"] == "17"
    assert response.content == b"\x01\x00"


def test_pcm16_encoder_is_little_endian() -> None:
    module = importlib.import_module("aniflive_tts.service")
    assert module._pcm16_bytes(np.array([1.0, -1.0], dtype=np.float32)) == b"\xff\x7f\x01\x80"


def _request(body: bytes, *, content_length: int | None = None) -> Request:
    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode("ascii")))
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.request", "body": b"", "more_body": False}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": headers,
        },
        receive,
    )


def test_json_body_rejects_declared_oversize_before_reading(monkeypatch) -> None:
    module = importlib.import_module("aniflive_tts.service")
    request = _request(b"{}", content_length=module.MAX_JSON_BODY_BYTES + 1)
    monkeypatch.setattr(request, "body", lambda: pytest.fail("oversize body should not be read"))

    with pytest.raises(module.RequestBodyTooLarge):
        asyncio.run(module._read_json_body(request))


def test_json_body_rejects_actual_oversize_without_content_length() -> None:
    module = importlib.import_module("aniflive_tts.service")
    body = json.dumps({"text": "x" * module.MAX_JSON_BODY_BYTES}).encode("utf-8")

    with pytest.raises(module.RequestBodyTooLarge):
        asyncio.run(module._read_json_body(_request(body)))


def test_json_body_rejects_oversize_when_content_length_is_underreported() -> None:
    module = importlib.import_module("aniflive_tts.service")
    body = json.dumps({"text": "x" * module.MAX_JSON_BODY_BYTES}).encode("utf-8")

    with pytest.raises(module.RequestBodyTooLarge):
        asyncio.run(module._read_json_body(_request(body, content_length=2)))


def test_oversize_json_request_returns_413(monkeypatch) -> None:
    module = importlib.import_module("aniflive_tts.service")
    body = json.dumps({"text": "x" * module.MAX_JSON_BODY_BYTES}).encode("utf-8")
    with _client(monkeypatch) as client:
        response = client.post("/", content=body, headers={"content-type": "application/json"})

    assert response.status_code == 413
    assert response.json() == {
        "code": 413,
        "message": f"Request body is limited to {module.MAX_JSON_BODY_BYTES} bytes",
    }


def test_api_hides_reference_and_runtime_paths(monkeypatch) -> None:
    module = importlib.import_module("aniflive_tts.service")
    monkeypatch.setattr(module.SERVICE, "_sample_rate", 32000)
    with _client(monkeypatch) as client:
        config = client.get("/model/config")
        voices = client.get("/v1/voices")
        health = client.get("/health")

    assert config.status_code == 200
    assert (
        not {
            "engine_dir",
            "onnx_dir",
            "reference_wav",
            "reference_text",
        }
        & config.json().keys()
    )
    assert config.json()["reference_configured"] is True
    assert voices.status_code == 200
    voice = voices.json()["data"][0]
    assert set(voice) == {"id", "reference_language", "reference_configured"}
    assert voice["reference_configured"] is True
    assert health.status_code == 200
    assert health.json()["reference"] == {
        "configured": True,
        "language": module.REFERENCE_LANGUAGE,
    }


def test_canonical_api_requires_one_of_five_core_language_codes(monkeypatch) -> None:
    module = importlib.import_module("aniflive_tts.service")
    with _client(monkeypatch) as client:
        missing = client.post(
            "/v1/audio/speech",
            json={"text": "hello", "model": module.MODEL_ID},
        )
        automatic = client.post(
            "/v1/audio/speech",
            json={
                "text": "hello",
                "model": module.MODEL_ID,
                "language": "auto",
            },
        )

    assert missing.status_code == 400
    assert "Missing required parameter: language" in missing.json()["message"]
    assert automatic.status_code == 400
    assert "canonical language codes" in automatic.json()["message"]
