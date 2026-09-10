from __future__ import annotations

import asyncio
import importlib
import json
import threading
from types import SimpleNamespace

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
    monkeypatch.setattr(module, "MODEL_ID", "test-v2pp")
    monkeypatch.setattr(module, "VOICE_ID", "default")
    monkeypatch.setattr(module, "REFERENCE_TEXT", "reference")
    monkeypatch.setattr(module, "REFERENCE_LANGUAGE", "en")
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
    speech_sessions = response.json()["speech_sessions"]
    assert speech_sessions == {
        "committed_segments": True,
        "pcm_stream": True,
        "cancel": True,
        "flush": True,
        "context_mode": "committed-neural-v1",
        "neural_state_continuity": True,
        "acoustic_latent_continuity": False,
        "default_policy": "A",
        "policies": {
            "A": {"name": "none", "phones_bert": False, "semantic_tokens": 0, "expression_aware": False, "boundary_adaptive": False},
            "B": {"name": "text-bert", "phones_bert": True, "semantic_tokens": 0, "expression_aware": False, "boundary_adaptive": False},
            "C": {"name": "text-bert-semantic-32", "phones_bert": True, "semantic_tokens": 32, "expression_aware": False, "boundary_adaptive": False},
            "D": {"name": "text-bert-semantic-64", "phones_bert": True, "semantic_tokens": 64, "expression_aware": False, "boundary_adaptive": False},
            "E": {"name": "expression-aware", "phones_bert": True, "semantic_tokens": 64, "expression_aware": True, "boundary_adaptive": False},
            "F": {"name": "boundary-adaptive", "phones_bert": True, "semantic_tokens": 64, "expression_aware": True, "boundary_adaptive": True},
        },
        "qualification": "experimental-unqualified",
        "limits": {
            "maximum_open": 8,
            "ttl_seconds": 600.0,
            "maximum_segments": 64,
            "maximum_characters": 16000,
        },
    }


def test_speech_session_streams_committed_segments_and_is_idempotent(
    monkeypatch,
) -> None:
    module = importlib.import_module("aniflive_tts.service")
    monkeypatch.setattr(
        module,
        "SESSION_MANAGER",
        module.SpeechSessionManager(maximum_open=2, ttl_seconds=60),
    )
    monkeypatch.setattr(module.SERVICE, "_sample_rate", 32000)
    monkeypatch.setattr(module.SERVICE, "validate_expression", lambda _: None)
    monkeypatch.setattr(
        module.SERVICE,
        "prepare_stream",
        lambda options, model, **_: (
            [options.text],
            iter((b"\x01\x00", b"\x02\x00")),
            32000,
            model,
        ),
    )

    with _client(monkeypatch) as client:
        created = client.post(
            "/v1/sessions",
            json={"model": "test-v2pp", "voice_profile": "default"},
        )
        session_id = created.json()["id"]
        segment_body = {
            "segment_id": "turn-001",
            "text": "Hello.",
            "language": "en",
            "expression": {"enabled": False},
            "pause_after_ms": 1,
        }
        first = client.post(f"/v1/sessions/{session_id}/segments", json=segment_body)
        duplicate = client.post(f"/v1/sessions/{session_id}/segments", json=segment_body)
        flushed = client.post(f"/v1/sessions/{session_id}/flush")
        audio = client.get(f"/v1/sessions/{session_id}/audio")
        final = client.get(f"/v1/sessions/{session_id}")

    assert created.status_code == 201
    assert created.json()["context"]["neural_state_continuity"] is False
    assert first.status_code == 201
    assert duplicate.status_code == 200
    assert flushed.json()["state"] == "flushing"
    assert audio.status_code == 200
    assert audio.headers["x-tts-session-context"] == "committed-neural-v1"
    assert audio.headers["x-tts-session-context-policy"] == "A"
    assert audio.headers["x-tts-neural-state-continuity"] == "false"
    assert audio.headers["x-tts-acoustic-latent-continuity"] == "false"
    assert audio.content[:4] == b"\x01\x00\x02\x00"
    assert len(audio.content) == 4 + 64
    assert final.json()["state"] == "closed"
    assert final.json()["segments"][0]["state"] == "completed"
    assert final.json()["context"]["committed_segments"] == 1
    assert final.json()["context"]["previous_language"] == "en"
    assert final.json()["context"]["previous_output_samples"] == 2


def test_speech_session_policy_is_explicit_and_invalid_policy_fails_closed(
    monkeypatch,
) -> None:
    module = importlib.import_module("aniflive_tts.service")
    monkeypatch.setattr(
        module,
        "SESSION_MANAGER",
        module.SpeechSessionManager(maximum_open=2, ttl_seconds=60),
    )
    monkeypatch.setattr(module.SERVICE, "_sample_rate", 32000)

    with _client(monkeypatch) as client:
        created = client.post("/v1/sessions", json={"continuity_policy": "f"})
        invalid = client.post("/v1/sessions", json={"continuity_policy": "Z"})

    assert created.status_code == 201
    assert created.json()["continuity_policy"] == "F"
    assert created.json()["context"]["policy"] == "F"
    assert created.json()["context"]["acoustic_latent_continuity"] is False
    assert invalid.status_code == 400
    assert "one of A, B, C, D, E or F" in invalid.json()["message"]


def test_model_activation_clears_terminal_session_gpu_context_first(
    monkeypatch,
) -> None:
    module = importlib.import_module("aniflive_tts.service")
    manager = module.SpeechSessionManager(maximum_open=2, ttl_seconds=60)
    monkeypatch.setattr(module, "SESSION_MANAGER", manager)
    observed = []

    async def arrange() -> None:
        session = await manager.create(
            model="test-v2pp",
            voice_profile="default",
            sample_rate=32000,
            continuity_policy="D",
        )
        session.state = "closed"
        session.context.neural = object()
        observed.append(session)

    asyncio.run(arrange())

    def activate(model_id: str):
        assert observed[0].context.neural is None
        return {"changed": True, "model": model_id, "voice": "default"}

    monkeypatch.setattr(module.SERVICE, "activate", activate)
    with _client(monkeypatch) as client:
        response = client.post("/v1/models/activate", json={"model": "new-model"})

    assert response.status_code == 200
    assert response.json()["model"] == "new-model"


def test_speech_session_rejects_conflicting_duplicate_and_model_switch(
    monkeypatch,
) -> None:
    module = importlib.import_module("aniflive_tts.service")
    monkeypatch.setattr(
        module,
        "SESSION_MANAGER",
        module.SpeechSessionManager(maximum_open=2, ttl_seconds=60),
    )
    monkeypatch.setattr(module.SERVICE, "_sample_rate", 32000)
    monkeypatch.setattr(module.SERVICE, "validate_expression", lambda _: None)
    activated = []
    monkeypatch.setattr(module.SERVICE, "activate", activated.append)

    with _client(monkeypatch) as client:
        session_id = client.post("/v1/sessions", json={}).json()["id"]
        original = {
            "segment_id": "same-id",
            "text": "First.",
            "language": "en",
        }
        assert client.post(
            f"/v1/sessions/{session_id}/segments", json=original
        ).status_code == 201
        conflict = client.post(
            f"/v1/sessions/{session_id}/segments",
            json={**original, "text": "Different."},
        )
        switch = client.post("/v1/models/activate", json={"model": "another"})
        cancelled = client.post(f"/v1/sessions/{session_id}/cancel")

    assert conflict.status_code == 409
    assert "different content" in conflict.json()["message"]
    assert switch.status_code == 409
    assert "speech session" in switch.json()["message"]
    assert activated == []
    assert cancelled.json()["state"] == "cancelled"


def test_speech_session_retry_is_idempotent_after_flush(monkeypatch) -> None:
    module = importlib.import_module("aniflive_tts.service")
    monkeypatch.setattr(
        module,
        "SESSION_MANAGER",
        module.SpeechSessionManager(maximum_open=1, ttl_seconds=60),
    )
    monkeypatch.setattr(module.SERVICE, "_sample_rate", 32000)
    monkeypatch.setattr(module.SERVICE, "validate_expression", lambda _: None)

    body = {"segment_id": "stable", "text": "Hello.", "language": "en"}
    with _client(monkeypatch) as client:
        session_id = client.post("/v1/sessions", json={}).json()["id"]
        assert client.post(
            f"/v1/sessions/{session_id}/segments", json=body
        ).status_code == 201
        assert client.post(f"/v1/sessions/{session_id}/flush").status_code == 200
        duplicate = client.post(
            f"/v1/sessions/{session_id}/segments", json=body
        )
        new_segment = client.post(
            f"/v1/sessions/{session_id}/segments",
            json={**body, "segment_id": "new"},
        )

    assert duplicate.status_code == 200
    assert new_segment.status_code == 409


def test_session_cancel_uses_owner_scoped_stream_cancellation(monkeypatch) -> None:
    module = importlib.import_module("aniflive_tts.service")
    monkeypatch.setattr(
        module,
        "SESSION_MANAGER",
        module.SpeechSessionManager(maximum_open=1, ttl_seconds=60),
    )
    monkeypatch.setattr(module.SERVICE, "_sample_rate", 32000)
    owners = []
    monkeypatch.setattr(
        module.SERVICE, "cancel_stream", lambda owner: owners.append(owner) or False
    )
    monkeypatch.setattr(
        module.SERVICE,
        "cancel_active_stream",
        lambda: pytest.fail("session cancellation must not use process-wide cancel"),
    )

    with _client(monkeypatch) as client:
        session_id = client.post("/v1/sessions", json={}).json()["id"]
        response = client.post(f"/v1/sessions/{session_id}/cancel")

    assert response.status_code == 200
    assert owners == [session_id]
    assert response.json()["active_audio_cancelled"] is False


def test_runtime_stream_owner_prevents_cross_session_cancel(monkeypatch) -> None:
    module = importlib.import_module("aniflive_tts.service")

    class FakeService:
        sample_rate = 32000

        def __init__(self) -> None:
            self.cancelled = 0

        @staticmethod
        def _segments(options):
            return [options.text]

        @staticmethod
        def stream_pcm(options):
            del options
            return iter((b"\x01\x00",))

        def cancel_active_stream(self):
            self.cancelled += 1
            return True

    fake = FakeService()
    manager = module.RuntimeServiceManager(fake)
    monkeypatch.setattr(module, "MODEL_ID", "voice")
    _, pcm, _, _ = manager.prepare_stream(
        SimpleNamespace(text="Hello."),
        "voice",
        stream_owner="session-a",
    )

    assert manager.active_stream_owner == "session-a"
    assert manager.cancel_stream("session-b") is False
    assert manager.cancel_stream("session-a") is True
    assert fake.cancelled == 1
    assert list(pcm) == [b"\x01\x00"]
    assert manager.active_stream_owner is None

    _, unopened, _, _ = manager.prepare_stream(
        SimpleNamespace(text="Unopened."),
        "voice",
        stream_owner="session-c",
    )
    assert manager.active_stream_owner == "session-c"
    unopened.close()
    assert manager.active_stream_owner is None


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


def test_direct_pcm_response_releases_request_before_body_iteration(monkeypatch) -> None:
    module = importlib.import_module("aniflive_tts.service")
    base = module.TensorRTService(
        module.RuntimeSettings(
            source_dir=module.Path("."),
            engine_dir=module.Path("."),
            bert_path=module.Path("."),
            reference_wav=module.Path("reference.wav"),
            profile_manifest=module.Path("profile.json"),
        )
    )
    manager = module.RuntimeServiceManager(base)
    started = threading.Event()
    released = threading.Event()

    class FakeLease:
        @staticmethod
        def ensure_valid() -> None:
            return None

        @staticmethod
        def close() -> None:
            released.set()

    class FakeStreamer:
        @staticmethod
        def streaming_chunk_length() -> int:
            return 1

        @staticmethod
        def iter_audio(**kwargs):
            cancelled = kwargs["cancelled"]
            started.set()
            while not cancelled.is_set():
                yield np.ones(16, dtype=np.float32)

    base._engine = object()
    base._streamer = FakeStreamer()
    base._sample_rate = 32000
    base._gpu_resource_lease = SimpleNamespace(acquire=lambda purpose: FakeLease())
    monkeypatch.setattr(
        base,
        "_segment_plan",
        lambda options: [SimpleNamespace(text=options.text)],
    )
    monkeypatch.setattr(base, "_conditioning", lambda options: None)
    monkeypatch.setattr(module, "SERVICE", manager)
    monkeypatch.setattr(module, "MODEL_ID", "test-v2pp")

    async def scenario() -> None:
        response = await module._tts_response(
            _request(b""),
            {
                "model": "test-v2pp",
                "text": "hello",
                "language": "en",
                "stream": True,
                "response_format": "pcm",
            },
            text_field="text",
            language_field="language",
        )
        assert await asyncio.to_thread(started.wait, 1.0)
        assert base._active_requests == 1

        async def receive():
            raise AssertionError("The ASGI 2.4 response must not poll receive")

        async def disconnect_on_headers(message):
            assert message["type"] == "http.response.start"
            raise RuntimeError("client disconnected before body iteration")

        scope = {
            "type": "http",
            "asgi": {"spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/audio/speech",
            "raw_path": b"/v1/audio/speech",
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
        with pytest.raises(RuntimeError, match="before body iteration"):
            await response(scope, receive, disconnect_on_headers)

        assert await asyncio.to_thread(released.wait, 1.0)
        for _ in range(100):
            if base._active_stream_cancel is None:
                break
            await asyncio.sleep(0.01)
        assert base._active_requests == 0
        assert base._active_stream_cancel is None
        assert base._request_slot.acquire(blocking=False) is True
        base._request_slot.release()
        await response.aclose()

    asyncio.run(scenario())


def test_pcm16_encoder_is_little_endian() -> None:
    module = importlib.import_module("aniflive_tts.service")
    assert module._pcm16_bytes(np.array([1.0, -1.0], dtype=np.float32)) == b"\xff\x7f\x01\x80"


def _request(
    body: bytes,
    *,
    content_length: int | None = None,
    app=None,
    path: str = "/",
    delivered_event: asyncio.Event | None = None,
) -> Request:
    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode("ascii")))
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.request", "body": b"", "more_body": False}
        delivered = True
        if delivered_event is not None:
            delivered_event.set()
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers,
    }
    if app is not None:
        scope["app"] = app
    return Request(scope, receive)


def test_model_activation_wins_atomic_race_before_session_creation(monkeypatch) -> None:
    module = importlib.import_module("aniflive_tts.service")
    manager = module.SpeechSessionManager(maximum_open=2, ttl_seconds=60)
    monkeypatch.setattr(module, "SESSION_MANAGER", manager)
    monkeypatch.setattr(module, "MODEL_ID", "old-model")
    monkeypatch.setattr(module, "VOICE_ID", "default")
    monkeypatch.setattr(module.SERVICE, "_sample_rate", 32000)
    activation_entered = threading.Event()
    allow_activation = threading.Event()

    def blocking_activate(model_id: str):
        activation_entered.set()
        assert allow_activation.wait(timeout=2.0)
        module.MODEL_ID = model_id
        return {"changed": True, "model": model_id, "voice": "default"}

    monkeypatch.setattr(module.SERVICE, "activate", blocking_activate)

    async def scenario() -> None:
        module.app.state.session_model_lock = asyncio.Lock()
        activation = asyncio.create_task(
            module.activate_model(
                _request(
                    json.dumps({"model": "new-model"}).encode(),
                    app=module.app,
                    path="/v1/models/activate",
                )
            )
        )
        assert await asyncio.to_thread(activation_entered.wait, 1.0)

        creation_body_read = asyncio.Event()
        creation = asyncio.create_task(
            module.create_speech_session(
                _request(
                    json.dumps(
                        {"model": "new-model", "voice_profile": "default"}
                    ).encode(),
                    app=module.app,
                    path="/v1/sessions",
                    delivered_event=creation_body_read,
                )
            )
        )
        await asyncio.wait_for(creation_body_read.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert creation.done() is False

        allow_activation.set()
        activated, created = await asyncio.gather(activation, creation)
        assert activated.status_code == 200
        assert created.status_code == 201
        assert json.loads(created.body)["model"] == "new-model"

    asyncio.run(scenario())


def test_session_creation_wins_atomic_race_before_model_activation(monkeypatch) -> None:
    module = importlib.import_module("aniflive_tts.service")
    manager = module.SpeechSessionManager(maximum_open=2, ttl_seconds=60)
    original_create = manager.create
    monkeypatch.setattr(module, "SESSION_MANAGER", manager)
    monkeypatch.setattr(module, "MODEL_ID", "old-model")
    monkeypatch.setattr(module, "VOICE_ID", "default")
    monkeypatch.setattr(module.SERVICE, "_sample_rate", 32000)
    activated: list[str] = []
    monkeypatch.setattr(module.SERVICE, "activate", activated.append)

    async def scenario() -> None:
        module.app.state.session_model_lock = asyncio.Lock()
        creation_entered = asyncio.Event()
        allow_creation = asyncio.Event()

        async def blocking_create(**kwargs):
            creation_entered.set()
            await allow_creation.wait()
            return await original_create(**kwargs)

        monkeypatch.setattr(manager, "create", blocking_create)
        creation = asyncio.create_task(
            module.create_speech_session(
                _request(
                    json.dumps(
                        {"model": "old-model", "voice_profile": "default"}
                    ).encode(),
                    app=module.app,
                    path="/v1/sessions",
                )
            )
        )
        await asyncio.wait_for(creation_entered.wait(), timeout=1.0)

        activation_body_read = asyncio.Event()
        activation = asyncio.create_task(
            module.activate_model(
                _request(
                    json.dumps({"model": "new-model"}).encode(),
                    app=module.app,
                    path="/v1/models/activate",
                    delivered_event=activation_body_read,
                )
            )
        )
        await asyncio.wait_for(activation_body_read.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert activation.done() is False

        allow_creation.set()
        created, switched = await asyncio.gather(creation, activation)
        assert created.status_code == 201
        assert switched.status_code == 409
        assert "speech session" in json.loads(switched.body)["message"]
        assert activated == []

    asyncio.run(scenario())


def test_speech_session_response_detaches_consumer_before_body_iteration(
    monkeypatch,
) -> None:
    module = importlib.import_module("aniflive_tts.service")
    manager = module.SpeechSessionManager(maximum_open=2, ttl_seconds=60)
    monkeypatch.setattr(module, "SESSION_MANAGER", manager)

    async def scenario() -> None:
        session = await manager.create(
            model="test-v2pp",
            voice_profile="default",
            sample_rate=32000,
        )
        response = await module.stream_speech_session_audio(session.session_id)
        assert session.audio_consumer_attached is True

        async def receive():
            raise AssertionError("The ASGI 2.4 response must not poll receive")

        async def disconnect_on_headers(message):
            assert message["type"] == "http.response.start"
            raise RuntimeError("client disconnected before body iteration")

        scope = {
            "type": "http",
            "asgi": {"spec_version": "2.4"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": f"/v1/sessions/{session.session_id}/audio",
            "raw_path": b"/v1/sessions/session/audio",
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
        with pytest.raises(RuntimeError, match="before body iteration"):
            await response(scope, receive, disconnect_on_headers)

        assert session.audio_consumer_attached is False
        await response.aclose()
        assert session.audio_consumer_attached is False

        reattached = await manager.attach_consumer(session.session_id)
        assert reattached is session
        await manager.detach_consumer(session)

    asyncio.run(scenario())


def test_speech_session_response_consumer_detach_survives_caller_cancellation(
    monkeypatch,
) -> None:
    module = importlib.import_module("aniflive_tts.service")
    manager = module.SpeechSessionManager(maximum_open=2, ttl_seconds=60)
    monkeypatch.setattr(module, "SESSION_MANAGER", manager)
    original_detach = manager.detach_consumer

    async def scenario() -> None:
        detach_entered = asyncio.Event()
        allow_detach = asyncio.Event()
        detach_calls = 0

        async def blocking_detach(session):
            nonlocal detach_calls
            detach_calls += 1
            detach_entered.set()
            await allow_detach.wait()
            await original_detach(session)

        monkeypatch.setattr(manager, "detach_consumer", blocking_detach)
        session = await manager.create(
            model="test-v2pp",
            voice_profile="default",
            sample_rate=32000,
        )
        response = await module.stream_speech_session_audio(session.session_id)

        closing = asyncio.create_task(response.aclose())
        await asyncio.wait_for(detach_entered.wait(), timeout=1.0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert session.audio_consumer_attached is True

        allow_detach.set()
        await response.aclose()
        assert session.audio_consumer_attached is False
        assert detach_calls == 1

    asyncio.run(scenario())


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
