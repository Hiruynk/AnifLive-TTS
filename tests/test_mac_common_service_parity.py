"""Shared service wiring required by the native macOS backend."""
import asyncio
import os
from types import SimpleNamespace

import pytest

from aniflive_tts import service


class _LockMustNotBeUsed:
    def __enter__(self):
        raise AssertionError("native macOS cancellation must not wait for the switch lock")

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_native_cancel_and_owner_reads_bypass_switch_lock(monkeypatch):
    runtime = SimpleNamespace(cancel_active_stream=lambda: True)
    manager = service.RuntimeServiceManager(runtime)
    manager._active_stream_owner = "session-a"
    manager._switch_lock = _LockMustNotBeUsed()
    monkeypatch.setattr(service, "_native_macos", lambda: True)

    assert manager.cancel_active_stream() is True
    assert manager.active_stream_owner == "session-a"


def test_voice_library_uses_configured_prepared_reference_capacity(monkeypatch):
    expected = int(os.environ.get("ANIFLIVE_TTS_PREPARED_REFERENCE_CAPACITY", "64"))
    assert service.VOICE_LIBRARY.maximum_prepared == expected


def test_native_lifespan_restores_registered_references_on_demand(monkeypatch):
    calls = []
    library = SimpleNamespace(
        prepare_on_demand=False,
        prepare_callback=None,
        restore_default_references=lambda: calls.append("defaults"),
        restore_registered_references=lambda: calls.append("registered"),
        close=lambda: calls.append("close-library"),
    )
    runtime = SimpleNamespace(
        load=lambda: calls.append("load"),
        unload=lambda: calls.append("unload"),
        prepare_dynamic_reference=lambda **kwargs: None,
    )

    async def clear_contexts():
        calls.append("clear-contexts")

    monkeypatch.setattr(service, "SERVICE", runtime)
    monkeypatch.setattr(service, "VOICE_LIBRARY", library)
    monkeypatch.setattr(service, "SESSION_MANAGER", SimpleNamespace(clear_all_neural_contexts=clear_contexts))
    monkeypatch.setattr(service, "_native_macos", lambda: True)

    async def scenario():
        async with service.lifespan(service.app):
            assert library.prepare_on_demand is True
            assert "registered" in calls
            assert "defaults" not in calls

    asyncio.run(scenario())

    assert calls[-3:] == ["clear-contexts", "close-library", "unload"]


def test_native_top_k_overflow_reports_shared_limit(monkeypatch):
    monkeypatch.setattr(service, "_native_macos", lambda: True)
    monkeypatch.setattr(service, "_assert_fixed_reference", lambda values: None)

    with pytest.raises(service.RequestError, match="top_k must be at most 50") as error:
        service._options_from_values(
            {"text": "hello", "text_language": "en", "top_k": 51},
            text_field="text",
            language_field="text_language",
        )

    assert "requires top_k=15" not in str(error.value)


def test_native_session_context_header_uses_shared_contract(monkeypatch):
    session = SimpleNamespace(
        session_id="session-a",
        model="voice-a",
        sample_rate=32000,
        continuity_policy=service.ContinuityPolicy.NONE,
    )

    async def attach_consumer(session_id):
        assert session_id == "session-a"
        return session

    monkeypatch.setattr(service, "SESSION_MANAGER", SimpleNamespace(attach_consumer=attach_consumer))
    monkeypatch.setattr(service, "_native_macos", lambda: True)
    monkeypatch.setattr(service, "_runtime_backend_headers", lambda: {})

    response = asyncio.run(service.stream_speech_session_audio("session-a"))

    assert response.headers["X-TTS-Session-Context"] == "committed-neural-v1"
    assert response.headers["X-TTS-Neural-State-Continuity"] == "false"
