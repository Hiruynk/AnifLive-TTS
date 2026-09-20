from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from aniflive_tts.sdk import (
    APIError,
    ActivationResult,
    AnifLiveTTSError,
    BusyError,
    Capabilities,
    ConditioningPolicy,
    ContinuityPolicyCapability,
    Expression,
    ExpressionCapabilities,
    ExpressionCatalogInfo,
    ExpressionProfileInfo,
    ExpressionSegment,
    Generation,
    Health,
    Language,
    ListResponse,
    ModelInfo,
    SegmentInfo,
    SegmentRequest,
    SessionCapabilities,
    SessionContext,
    SessionCreate,
    SessionInfo,
    SessionLimits,
    SpeechSegmentState,
    StreamMetadata,
    StreamingCapabilities,
    SynthesisMetadata,
    SynthesisRequest,
    SynthesisResult,
    TimeoutError,
    TransportError,
)


def test_request_records_are_frozen_and_do_not_repr_synthesis_text() -> None:
    request = SynthesisRequest(
        language=Language.EN,
        text="sensitive spoken text",
        expression=Expression(
            enabled=True,
            profile="calm",
            intensity=0.5,
            policy=ConditioningPolicy.SEMANTIC_STYLE,
        ),
        generation=Generation(seed=42, speed=1.0),
    )

    assert request.language is Language.EN
    assert request.generation == Generation(seed=42, speed=1.0)
    assert "sensitive spoken text" not in repr(request)
    with pytest.raises(FrozenInstanceError):
        request.model = "changed"  # type: ignore[misc]


def test_segments_keep_order_without_repr_exposing_segment_text() -> None:
    segments = [
        ExpressionSegment("private one"),
        ExpressionSegment("private two", expression=Expression(profile="calm")),
    ]
    request = SynthesisRequest(language=Language.ZH, segments=segments)

    assert isinstance(request.segments, tuple)
    assert [segment.text for segment in request.segments or ()] == [
        "private one",
        "private two",
    ]
    assert "private one" not in repr(request)
    assert "private two" not in repr(request)


def test_response_extras_are_preserved_and_exposed_as_read_only_mappings() -> None:
    unknown = {"future_field": {"enabled": True}}
    model = ModelInfo(
        id="fixture-model",
        object="model",
        active=True,
        voice_profiles=("default",),
        owned_by="aniflive-tts-local",
        description="fixture",
        model_family="v2proplus",
        extra=unknown,
    )
    response = ListResponse(object="list", data=[model], extra={"future_list_key": 7})

    unknown["added_later"] = "not copied"
    assert "future_field" in response.data[0].extra
    assert "added_later" not in response.data[0].extra
    assert response.data == (model,)
    assert response.extra["future_list_key"] == 7
    with pytest.raises(TypeError):
        response.extra["new"] = True  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        response.object = "changed"  # type: ignore[misc]


def test_discovery_types_retain_nested_metadata_and_optional_runtime_fields() -> None:
    health = Health(
        service="AnifLive-TTS",
        version="1.4.0",
        ready=True,
        status="ok",
        engines={"ssl": True},
        reference={"configured": True, "language": "en"},
        extra={"future_health_field": 3},
    )
    profile = ExpressionProfileInfo(
        id="calm",
        intensity_levels=(0.5, 0.8),
        languages=("en",),
        reference_languages=("en",),
        preferred_policies=("semantic-style",),
    )
    catalog = ExpressionCatalogInfo(
        object="list",
        enabled=True,
        default="calm",
        profiles=[profile],
        policies=["full-switch", "future-policy"],
        preferred_policy="semantic-style",
        playback_policy={"short_prebuffer_ms": 32},
        extra={"future_expression_field": "kept"},
    )
    capabilities = Capabilities(
        service="AnifLive-TTS",
        version="1.4.0",
        languages=["zh", "yue", "en", "ja", "ko"],
        streaming=StreamingCapabilities(pcm16=True, wav=False),
        expression=ExpressionCapabilities(native=True, controlled_profiles=True),
        model_family="v2proplus",
        speech_sessions=SessionCapabilities(
            committed_segments=True,
            pcm_stream=True,
            policies={"F": ContinuityPolicyCapability(name="boundary-adaptive", extra={"future": True})},
            limits=SessionLimits(
                maximum_open=8,
                ttl_seconds=600.0,
                maximum_segments=64,
                maximum_characters=16_000,
            ),
        ),
        extra={"future_capability": {"value": 1}},
    )

    assert health.engines == {"ssl": True}
    assert health.extra["future_health_field"] == 3
    assert catalog.profiles == (profile,)
    assert catalog.policies[-1] == "future-policy"
    assert catalog.extra["future_expression_field"] == "kept"
    assert capabilities.languages == ("zh", "yue", "en", "ja", "ko")
    assert capabilities.speech_sessions is not None
    assert capabilities.speech_sessions.policies["F"].extra["future"] is True
    assert capabilities.speech_sessions.limits is not None
    assert capabilities.speech_sessions.limits.maximum_open == 8


def test_synthesis_metadata_and_audio_are_not_rendered_in_repr() -> None:
    headers = {"X-TTS-Sample-Rate": "32000"}
    metadata = SynthesisMetadata(sample_rate=32_000, headers=headers)
    result = SynthesisResult(
        audio=b"secret-audio-bytes",
        content_type="audio/wav",
        metadata=metadata,
    )

    headers["X-TTS-Extra"] = "outside mutation"
    assert metadata.headers == {"X-TTS-Sample-Rate": "32000"}
    assert "secret-audio-bytes" not in repr(result)
    assert "secret-audio-bytes" not in repr(metadata)


def test_activation_message_is_structured_but_not_in_repr() -> None:
    result = ActivationResult(
        code=0,
        message="service text may echo private data",
        changed=True,
        model="fixture-model",
        voice="default",
        backend="TensorRT-11",
    )

    assert result.message == "service text may echo private data"
    assert "private data" not in repr(result)


def test_error_hierarchy_preserves_fields_without_exposing_server_text() -> None:
    error = BusyError(
        "server echoed sensitive text",
        status_code=429,
        server_code=429,
        retry_after="1",
        response_body=b"private response body",
    )

    assert isinstance(error, APIError)
    assert isinstance(error, AnifLiveTTSError)
    assert error.status_code == 429
    assert error.server_code == 429
    assert error.retry_after == "1"
    assert error.message == "server echoed sensitive text"
    assert error.response_body == b"private response body"
    assert "sensitive text" not in str(error)
    assert "sensitive text" not in repr(error)
    assert "private response body" not in repr(error)
    assert str(error) == "BusyError (HTTP 429)"


def test_client_errors_have_the_documented_hierarchy_and_safe_repr() -> None:
    error = TimeoutError("request included sensitive content", diagnostic="private detail")

    assert isinstance(error, TransportError) is False
    assert error.message == "request included sensitive content"
    assert error.diagnostic == "private detail"
    assert "sensitive content" not in str(error)
    assert "private detail" not in repr(error)


def test_api_error_preserves_unusual_status_and_raw_retry_after() -> None:
    error = APIError("redirect response", status_code=302, retry_after="Wed, 21 Oct 2026 07:28:00 GMT")

    assert error.status_code == 302
    assert error.retry_after == "Wed, 21 Oct 2026 07:28:00 GMT"
    assert "07:28:00" not in repr(error)
    with pytest.raises(ValueError):
        APIError("invalid", status_code=600)
    with pytest.raises(TypeError):
        APIError("invalid", status_code=429, retry_after=1)  # type: ignore[arg-type]


def test_stream_and_session_values_keep_metadata_and_redact_private_values() -> None:
    stream_metadata = StreamMetadata(
        sample_rate=32_000,
        channels=1,
        sample_format="s16le",
        metadata=SynthesisMetadata(
            sample_rate=32_000,
            headers={"X-TTS-Sample-Rate": "32000"},
        ),
    )
    from aniflive_tts.sdk import ContinuityPolicy

    create = SessionCreate(continuity_policy=ContinuityPolicy.F)
    segment_request = SegmentRequest(
        segment_id="segment-private-id",
        text="private spoken words",
        language=Language.EN,
    )
    segment = SegmentInfo(
        segment_id="segment-private-id",
        sequence=1,
        characters=19,
        language="en",
        paragraph_id=None,
        pause_after_ms=0,
        state="completed",
        created_at="2026-09-20T00:00:00Z",
        updated_at="2026-09-20T00:00:00Z",
        extra={"future_segment_field": True},
    )
    context = SessionContext(
        mode="semantic",
        policy="F",
        neural_state_continuity=True,
        acoustic_latent_continuity=True,
        committed_segments=1,
        last_plan={"private_note": "hidden"},
    )
    info = SessionInfo(
        id="session-private-id",
        object="speech_session",
        state="open",
        model="fixture-model",
        voice_profile="default",
        sample_rate=32_000,
        format="pcm_s16le",
        continuity_policy="F",
        context=context,
        audio_consumer_attached=False,
        created_at="2026-09-20T00:00:00Z",
        updated_at="2026-09-20T00:00:00Z",
        segments=[segment],
    )

    assert create.continuity_policy.value == "F"
    assert segment_request.language is Language.EN
    assert stream_metadata.sample_rate == 32_000
    assert segment.extra["future_segment_field"] is True
    assert segment.state is SpeechSegmentState.COMPLETED
    future_state = SegmentInfo(
        segment_id="future-state",
        sequence=2,
        characters=4,
        language="en",
        paragraph_id=None,
        pause_after_ms=0,
        state="future-state",
        created_at="2026-09-20T00:00:00Z",
        updated_at="2026-09-20T00:00:00Z",
    )
    assert future_state.state == "future-state"
    assert info.segments == (segment,)
    assert "private spoken words" not in repr(segment_request)
    assert "segment-private-id" not in repr(segment_request)
    assert "session-private-id" not in repr(info)
    assert "private_note" not in repr(context)
    with pytest.raises(FrozenInstanceError):
        info.state = "closed"  # type: ignore[misc]
