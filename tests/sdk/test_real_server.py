"""Opt-in integration checks against an explicitly configured TTS service."""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.sdk_real_server


_LANGUAGE_SAMPLES = {
    "ja": "\u4eca\u65e5\u306f\u3044\u3044\u5929\u6c17\u3067\u3059\u306d\u3002",
    "en": "The weather is nice today.",
    "zh": "\u4eca\u5929\u5929\u6c14\u5f88\u597d\u3002",
    "yue": "\u4f60\u597d\u3002",
    "ko": "\uc624\ub298 \ub0a0\uc528\uac00 \uc88b\ub124\uc694.",
}


@pytest.fixture(scope="module")
def server_url() -> str:
    value = os.environ.get("ANIFLIVE_TTS_TEST_URL")
    if not value:
        pytest.skip("set ANIFLIVE_TTS_TEST_URL to run live SDK integration checks")
    return value.rstrip("/")


def _language_sample(client) -> tuple[str, str]:
    from aniflive_tts import Language

    advertised = set(client.capabilities().languages)
    supported = {item.value for item in Language}
    for language in ("ja", "en", "zh", "yue", "ko"):
        if language in advertised and language in supported:
            return language, _LANGUAGE_SAMPLES[language]
    pytest.skip("the configured service advertises no language supported by the SDK")


def test_real_server_discovery(server_url: str) -> None:
    from aniflive_tts import Client

    with Client(server_url) as client:
        result = client.probe()
        assert result.health.service
        assert result.health.version
        assert result.capabilities.service == result.health.service
        assert result.capabilities.languages
        assert client.models().object == "list"
        assert client.voices().object == "list"
        assert client.expressions().object == "list"


def test_real_server_complete_wav(server_url: str) -> None:
    from aniflive_tts import Client, Generation, SynthesisRequest

    with Client(server_url) as client:
        language, text = _language_sample(client)
        result = client.synthesize(
            SynthesisRequest(
                text=text,
                language=language,
                generation=Generation(seed=1234),
            )
        )
        assert result.content_type.split(";", 1)[0].lower() == "audio/wav"
        assert result.audio.startswith(b"RIFF")
        assert result.audio[8:12] == b"WAVE"
        assert len(result.audio) > 44


def test_real_server_pcm_stream_can_be_cancelled(server_url: str) -> None:
    from aniflive_tts import Client, Generation, RequiredFeature

    with Client(server_url) as client:
        client.probe(required=(RequiredFeature.PCM_STREAMING,))
        language, text = _language_sample(client)
        stream = client.stream(
            text=text,
            language=language,
            generation=Generation(seed=1234),
        )
        try:
            assert stream.sample_format == "s16le"
            assert stream.channels == 1
            assert stream.next_chunk()
            # The SDK cancellation endpoint targets the active service request.
            result = client.cancel()
            assert isinstance(result.cancelled, bool)
        finally:
            stream.close()


def test_real_server_finite_speech_session(server_url: str) -> None:
    from aniflive_tts import (
        Client,
        ConflictError,
        ContinuityPolicy,
        Generation,
        RequiredFeature,
        SegmentRequest,
        SessionCreate,
    )

    with Client(server_url) as client:
        client.probe(required=(RequiredFeature.SPEECH_SESSIONS,))
        session = client.create_session(
            SessionCreate(continuity_policy=ContinuityPolicy.A)
        )
        audio = None
        try:
            fetched = client.get_session(session.id)
            assert fetched.initial_info.id == session.id
            assert fetched.refresh().id == session.id

            language, text = _language_sample(client)
            segment_id = uuid.uuid4().hex
            request = SegmentRequest(
                segment_id=segment_id,
                text=text,
                language=language,
                generation=Generation(seed=1234),
            )
            appended = session.append(request)
            duplicate = session.append(request)
            assert appended.segment_id == segment_id
            assert duplicate.sequence == appended.sequence

            with pytest.raises(ConflictError):
                session.append(
                    SegmentRequest(
                        segment_id=segment_id,
                        text=text + text,
                        language=language,
                        generation=Generation(seed=1234),
                    )
                )

            flushed = session.flush()
            assert flushed.id == session.id

            audio = session.audio()
            assert audio.sample_rate > 0
            assert audio.channels == 1
            chunks = list(audio)
            assert chunks
            assert sum(map(len, chunks)) > 0
        finally:
            if audio is not None:
                audio.close()
            session.close()


def test_real_server_session_cancel(server_url: str) -> None:
    from aniflive_tts import (
        Client,
        ContinuityPolicy,
        Generation,
        RequiredFeature,
        SegmentRequest,
        SessionCreate,
    )

    with Client(server_url) as client:
        client.probe(required=(RequiredFeature.SPEECH_SESSIONS,))
        session = client.create_session(
            SessionCreate(continuity_policy=ContinuityPolicy.A)
        )
        audio = None
        try:
            language, text = _language_sample(client)
            session.append(
                SegmentRequest(
                    segment_id=uuid.uuid4().hex,
                    text=text,
                    language=language,
                    generation=Generation(seed=1234),
                )
            )
            # Leave the session open so cancellation targets this finite segment.
            audio = session.audio()
            assert audio.next_chunk()
            cancelled = session.cancel()
            assert cancelled.id == session.id
            assert session.get().state == "cancelled"
        finally:
            if audio is not None:
                audio.close()
            session.close()
