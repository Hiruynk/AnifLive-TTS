from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from aniflive_tts.continuity import (
    MAX_CONTEXT_PHONEMES,
    MAX_CONTEXT_SEMANTIC_TOKENS,
    MAX_CONTEXT_TEXT_CHARS,
    ContinuationCapture,
    ContinuityPolicy,
    NeuralContinuationState,
    SpeechContextState,
)
from aniflive_tts.speech_session import SpeechSessionError, SpeechSessionManager


def test_committed_context_tracks_boundaries_without_claiming_neural_state() -> None:
    state = SpeechContextState()

    state.commit(
        text="A" * (MAX_CONTEXT_TEXT_CHARS + 20),
        language="yue",
        expression="relieved",
        paragraph_id="p1",
        pause_after_ms=120,
        output_samples=3200,
    )

    assert len(state.previous_text) == MAX_CONTEXT_TEXT_CHARS
    assert state.hard_boundary is False
    assert state.public_dict()["neural_state_continuity"] is False
    assert state.public_dict()["acoustic_latent_continuity"] is False
    assert state.public_dict()["mode"] == "committed-neural-v1"

    state.commit(
        text="Next paragraph.",
        language="en",
        expression=None,
        paragraph_id="p2",
        pause_after_ms=800,
        output_samples=6400,
    )

    assert state.segment_index == 2
    assert state.total_output_samples == 9600
    assert state.hard_boundary is True


def test_context_clear_removes_committed_metadata() -> None:
    state = SpeechContextState()
    state.commit(
        text="Hello.",
        language="en",
        expression=None,
        paragraph_id=None,
        pause_after_ms=0,
        output_samples=100,
    )

    state.clear()

    assert state.public_dict()["committed_segments"] == 0
    assert state.previous_text == ""
    assert state.total_output_samples == 0


class _FakeTensor:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, key):
        return _FakeTensor(self.values[key])

    def detach(self):
        return self

    def clone(self):
        return _FakeTensor(self.values.copy())

    def nelement(self):
        return self.values.size

    def element_size(self):
        return self.values.dtype.itemsize


def _neural_state(
    *,
    model: str = "voice",
    expression: str | None = "calm",
    phones: int = 100,
    tokens: int = 100,
) -> NeuralContinuationState:
    return NeuralContinuationState.capture(
        model_id=model,
        voice_profile="default",
        language="en",
        expression=expression,
        phones=range(phones),
        bert=_FakeTensor(np.arange(1024 * phones).reshape(1024, phones)),
        semantic=_FakeTensor(np.arange(tokens).reshape(1, tokens)),
    )


def _bound_context(policy: ContinuityPolicy) -> SpeechContextState:
    state = SpeechContextState()
    state.bind(model_id="voice", voice_profile="default", policy=policy)
    state.commit(
        text="Carry this,",
        language="en",
        expression="calm",
        paragraph_id="p1",
        pause_after_ms=0,
        output_samples=100,
        neural_state=_neural_state(),
        require_neural_state=True,
    )
    return state


def test_neural_capture_clones_and_caps_gpu_history() -> None:
    source_bert = _FakeTensor(np.ones((1024, 100), dtype=np.float16))
    source_semantic = _FakeTensor(np.arange(100).reshape(1, 100))
    capture = ContinuationCapture(model_id="voice", voice_profile="default")
    capture.publish(
        language="en",
        expression="calm",
        phones=range(100),
        bert=source_bert,
        semantic=source_semantic,
    )
    state = capture.take()

    assert state is not None
    assert len(state.phones) == MAX_CONTEXT_PHONEMES
    assert state.semantic_tokens == MAX_CONTEXT_SEMANTIC_TOKENS
    source_bert.values.fill(0)
    source_semantic.values.fill(0)
    assert np.all(state.bert.values == 1)
    assert state.semantic.values[0, -1] == 99


@pytest.mark.parametrize(
    ("policy", "semantic_tokens"),
    [
        (ContinuityPolicy.TEXT_BERT, 0),
        (ContinuityPolicy.SEMANTIC_32, 32),
        (ContinuityPolicy.SEMANTIC_64, 64),
    ],
)
def test_explicit_context_policies_apply_exact_limits(
    policy: ContinuityPolicy, semantic_tokens: int
) -> None:
    state = _bound_context(policy)
    plan = state.plan_for(
        text="Continue",
        language="en",
        expression="calm",
        paragraph_id="p1",
    )

    assert plan.effective_policy is policy
    assert plan.neural_input is not None
    assert len(plan.neural_input.phones) == MAX_CONTEXT_PHONEMES
    assert plan.neural_input.semantic_tokens == semantic_tokens


def test_expression_aware_policy_uses_bounded_switch_context() -> None:
    state = _bound_context(ContinuityPolicy.EXPRESSION_AWARE)
    plan = state.plan_for(
        text="Switch",
        language="en",
        expression="shy",
        paragraph_id="p1",
    )

    assert plan.reason == "expression-switch"
    assert plan.neural_input is not None
    assert len(plan.neural_input.phones) == 12
    assert plan.neural_input.semantic_tokens == 48


def test_boundary_adaptive_policy_selects_context_from_committed_boundary() -> None:
    state = _bound_context(ContinuityPolicy.BOUNDARY_ADAPTIVE)
    soft = state.plan_for(
        text="Continue",
        language="en",
        expression="calm",
        paragraph_id="p1",
    )
    switched = state.plan_for(
        text="Continue",
        language="en",
        expression="shy",
        paragraph_id="p1",
    )

    assert soft.effective_policy is ContinuityPolicy.SEMANTIC_32
    assert soft.neural_input is not None and soft.neural_input.semantic_tokens == 32
    assert switched.effective_policy is ContinuityPolicy.EXPRESSION_AWARE
    assert switched.neural_input is not None
    assert len(switched.neural_input.phones) == 12

    state.previous_text = "Complete."
    hard = state.plan_for(
        text="New sentence",
        language="en",
        expression="calm",
        paragraph_id="p1",
    )
    assert hard.effective_policy is ContinuityPolicy.NONE
    assert hard.neural_input is None
    assert hard.reason == "hard-natural-boundary"


def test_context_runtime_identity_mismatch_clears_neural_state() -> None:
    state = _bound_context(ContinuityPolicy.SEMANTIC_64)
    state.model_id = "different"

    plan = state.plan_for(
        text="Continue",
        language="en",
        expression="calm",
        paragraph_id="p1",
    )

    assert plan.reason == "runtime-identity-mismatch"
    assert state.neural is None


def test_context_treats_entering_or_leaving_a_named_paragraph_as_a_boundary() -> None:
    state = SpeechContextState()
    state.commit(
        text="First.",
        language="en",
        expression=None,
        paragraph_id="p1",
        pause_after_ms=0,
        output_samples=10,
    )
    state.commit(
        text="Unlabelled.",
        language="en",
        expression=None,
        paragraph_id=None,
        pause_after_ms=0,
        output_samples=10,
    )

    assert state.hard_boundary is True


async def _append(
    manager: SpeechSessionManager,
    session_id: str,
    *,
    segment_id: str,
    text: str = "Hello.",
) -> tuple[object, bool]:
    return await manager.append(
        session_id,
        segment_id=segment_id,
        text=text,
        language="en",
        options=SimpleNamespace(expression_profile=None, expression_segments=()),
        fingerprint_payload={"text": text, "language": "en"},
    )


def test_session_duplicate_remains_idempotent_after_flush() -> None:
    async def scenario() -> None:
        manager = SpeechSessionManager(maximum_open=1, ttl_seconds=60)
        session = await manager.create(
            model="voice", voice_profile="default", sample_rate=32000
        )
        original, created = await _append(manager, session.session_id, segment_id="s1")
        assert created is True
        await manager.flush(session.session_id)
        duplicate, duplicate_created = await _append(
            manager, session.session_id, segment_id="s1"
        )
        assert duplicate is original
        assert duplicate_created is False
        with pytest.raises(SpeechSessionError, match="no longer accepts"):
            await _append(manager, session.session_id, segment_id="s2")

    asyncio.run(scenario())


def test_session_cancelled_stream_never_commits_context() -> None:
    async def scenario() -> None:
        manager = SpeechSessionManager(maximum_open=1, ttl_seconds=60)
        session = await manager.create(
            model="voice", voice_profile="default", sample_rate=32000
        )
        await _append(manager, session.session_id, segment_id="s1")
        segment = await manager.next_segment(session)
        assert segment is not None and segment.state == "streaming"

        await manager.cancel(session.session_id)
        await manager.finish_segment(session, segment, output_samples=3200)

        assert segment.state == "cancelled"
        assert session.context.segment_index == 0
        assert session.context.total_output_samples == 0

    asyncio.run(scenario())


def test_nonbaseline_session_fails_closed_without_neural_capture() -> None:
    async def scenario() -> None:
        manager = SpeechSessionManager(maximum_open=1, ttl_seconds=60)
        session = await manager.create(
            model="voice",
            voice_profile="default",
            sample_rate=32000,
            continuity_policy="D",
        )
        await _append(manager, session.session_id, segment_id="s1")
        segment = await manager.next_segment(session)
        assert segment is not None

        with pytest.raises(ValueError, match="without committed continuation"):
            await manager.finish_segment(session, segment, output_samples=100)

        assert segment.state == "failed"
        assert segment.error == "continuity state could not be committed"
        assert session.state == "cancelled"
        assert session.context.segment_index == 0
        assert session.context.neural is None

    asyncio.run(scenario())


def test_committed_neural_state_survives_segments_and_is_released_on_close() -> None:
    async def scenario() -> None:
        manager = SpeechSessionManager(maximum_open=1, ttl_seconds=60)
        session = await manager.create(
            model="voice",
            voice_profile="default",
            sample_rate=32000,
            continuity_policy="D",
        )
        await _append(manager, session.session_id, segment_id="s1", text="Carry this,")
        segment = await manager.next_segment(session)
        assert segment is not None
        await manager.finish_segment(
            session,
            segment,
            output_samples=100,
            neural_state=_neural_state(),
        )

        plan = session.context.plan_for(
            text="into this segment",
            language="en",
            expression=None,
            paragraph_id=None,
        )
        assert plan.neural_input is not None
        assert plan.neural_input.semantic_tokens == 64
        assert session.context.neural is not None

        await manager.flush(session.session_id)
        assert await manager.next_segment(session) is None
        assert session.state == "closed"
        assert session.context.neural is None
        assert session.context.segment_index == 1
        assert session.context.last_plan is plan

    asyncio.run(scenario())


def test_cancel_and_expiry_release_retained_neural_tensors() -> None:
    async def scenario() -> None:
        manager = SpeechSessionManager(maximum_open=2, ttl_seconds=0.02)
        cancelled = await manager.create(
            model="voice",
            voice_profile="default",
            sample_rate=32000,
            continuity_policy="D",
        )
        cancelled.context.neural = _neural_state()
        await manager.cancel(cancelled.session_id)
        assert cancelled.context.neural is None

        expired = await manager.create(
            model="voice",
            voice_profile="default",
            sample_rate=32000,
            continuity_policy="D",
        )
        expired.context.neural = _neural_state()
        await asyncio.sleep(0.03)
        with pytest.raises(SpeechSessionError, match="not found"):
            await manager.get("missing")
        assert expired.state == "expired"
        assert expired.context.neural is None

    asyncio.run(scenario())


def test_model_switch_cleanup_releases_all_retained_neural_state() -> None:
    async def scenario() -> None:
        manager = SpeechSessionManager(maximum_open=2, ttl_seconds=60)
        session = await manager.create(
            model="voice",
            voice_profile="default",
            sample_rate=32000,
            continuity_policy="D",
        )
        session.context.neural = _neural_state()

        assert await manager.clear_all_neural_contexts() == 1
        assert session.context.neural is None
        assert session.context.segment_index == 0

    asyncio.run(scenario())


def test_waiting_audio_consumer_expires_without_an_external_cleanup_request() -> None:
    async def scenario() -> None:
        manager = SpeechSessionManager(maximum_open=1, ttl_seconds=0.02)
        session = await manager.create(
            model="voice", voice_profile="default", sample_rate=32000
        )
        await manager.attach_consumer(session.session_id)

        assert await manager.next_segment(session) is None
        assert session.state == "expired"

    asyncio.run(scenario())


def test_streaming_segment_is_not_expired_mid_inference() -> None:
    async def scenario() -> None:
        manager = SpeechSessionManager(maximum_open=1, ttl_seconds=0.01)
        session = await manager.create(
            model="voice", voice_profile="default", sample_rate=32000
        )
        await _append(manager, session.session_id, segment_id="s1")
        segment = await manager.next_segment(session)
        assert segment is not None
        await asyncio.sleep(0.02)

        assert (await manager.get(session.session_id)).state == "open"
        await manager.finish_segment(session, segment, output_samples=10)

    asyncio.run(scenario())


def test_session_enforces_segment_and_total_text_limits() -> None:
    async def scenario() -> None:
        segment_limited = SpeechSessionManager(
            maximum_open=1,
            ttl_seconds=60,
            maximum_segments=1,
            maximum_characters=100,
        )
        first = await segment_limited.create(
            model="voice", voice_profile="default", sample_rate=32000
        )
        await _append(segment_limited, first.session_id, segment_id="s1")
        with pytest.raises(SpeechSessionError, match="limited to 1 segments"):
            await _append(segment_limited, first.session_id, segment_id="s2")

        character_limited = SpeechSessionManager(
            maximum_open=1,
            ttl_seconds=60,
            maximum_segments=2,
            maximum_characters=6,
        )
        second = await character_limited.create(
            model="voice", voice_profile="default", sample_rate=32000
        )
        await _append(
            character_limited,
            second.session_id,
            segment_id="s1",
            text="1234",
        )
        with pytest.raises(SpeechSessionError, match="character limit"):
            await _append(
                character_limited,
                second.session_id,
                segment_id="s2",
                text="567",
            )

    asyncio.run(scenario())
