from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from aniflive_tts import service
from aniflive_tts.expression import (
    ConditioningPolicy, ExpressionCatalog, ExpressionSegment, PreparedReference,
    PreparedReferenceBank, ReferenceDescriptor,
)
from aniflive_tts.macos_service import MacOSV5Service, NativeReferenceHandle
from aniflive_tts.voice_library import VoiceBinding
from aniflive_tts.voices import VoiceError


def options(binding):
    return service.SynthesisOptions(
        text="Hello.", text_language="en", top_k=15, top_p=1.0,
        temperature=1.0, speed=1.0, pause_length=0.0, noise_scale=0.5,
        cut_punc="", seed=1, voice_profile="registered", voice_binding=binding,
    )


def prepared_binding():
    identity = NativeReferenceHandle(Path('/reference/default'), Path('/audio/default'), Path('/text/default'), 'ja')
    alternate = NativeReferenceHandle(Path('/reference/alternate'), Path('/audio/alternate'), Path('/text/alternate'), 'en')
    bank = PreparedReferenceBank(identity_id='default')
    bank.add(PreparedReference(identity, identity, identity, [1], None, id='default'))
    bank.add(PreparedReference(alternate, alternate, alternate, [2], None, id='alternate'))
    descriptors = tuple(ReferenceDescriptor(id=ref, emotion=emotion, intensity=0.5,
        reference_audio=Path('/unused'), reference_text='text', reference_language=language,
        manual_verified=True) for ref, emotion, language in
        [('default', 'neutral', 'ja'), ('alternate', 'warm', 'en')])
    return VoiceBinding('registered', bank, ExpressionCatalog('default', descriptors), SimpleNamespace(close=lambda: None)), identity, alternate


def native_service():
    instance = MacOSV5Service.__new__(MacOSV5Service)
    instance._client = SimpleNamespace(ready=True, metadata={"capabilities": {
        "conditioning_components": ["semantic", "style", "acoustic_prompt"]}})
    instance._supported_top_k = (15,)
    instance._worker_supported_top_k = (15,)
    instance._request_sampling = False
    instance._mandarin_enabled = False
    instance._persistent_frontend = object()
    instance._segment_char_limit = service.PROFILE_SEGMENT_CHAR_LIMITS["small"]
    instance._expression_catalog = None
    instance._package_bank = None
    instance._reference_inputs = None
    return instance


def components(native, value):
    plan = native._segment_conditionings(value)
    return [(text, refs, meta["expression_id"], meta["policy"]) for text, refs, meta in plan]


def test_default_and_explicit_full_switch_use_entire_prepared_reference():
    binding, identity, alternate = prepared_binding()
    native = native_service()
    default = options(binding)
    assert native._validate_native_options(default) == 'en'
    [(_, refs, _, _)] = components(native, default)
    assert refs == (identity, identity, identity)
    selected = replace(default, expression_enabled=True, expression_profile='alternate',
                       expression_policy=ConditioningPolicy.FULL_SWITCH)
    assert native._validate_native_options(selected) == 'en'
    [(_, refs, chosen, policy)] = components(native, selected)
    assert refs == (alternate, alternate, alternate) and chosen == 'alternate' and policy == 'full-switch'
    [(_, refs, chosen, _)] = components(native, replace(selected, expression_profile='warm'))
    assert refs == (alternate, alternate, alternate) and chosen == 'alternate'


def test_policies_compose_semantic_style_and_acoustic_prompt_per_contract():
    """semantic -> AR prompt, spectrogram -> ge style, speaker_embedding -> CFM prompt."""
    binding, identity, alternate = prepared_binding()
    native = native_service()
    base = replace(options(binding), expression_enabled=True, expression_profile='alternate')
    expected = {
        ConditioningPolicy.IDENTITY_LOCK: (alternate, alternate, identity),
        ConditioningPolicy.SEMANTIC_STYLE: (alternate, identity, identity),
        ConditioningPolicy.ACOUSTIC_STYLE: (identity, alternate, identity),
        ConditioningPolicy.SV_ONLY: (identity, identity, alternate),
    }
    for policy, refs in expected.items():
        [(_, got, _, reported)] = components(native, replace(base, expression_policy=policy))
        assert got == refs and reported == policy.value


def test_segmented_expressions_keep_order_and_reject_unknown_or_unprepared_profiles():
    binding, identity, alternate = prepared_binding()
    native = native_service()
    base = options(binding)
    segmented = replace(base, text="Hello. Goodbye.", expression_segments=(
        ExpressionSegment(text='Hello.'),
        ExpressionSegment(text='Goodbye.', enabled=True, profile='warm', intensity=0.5,
                          policy=ConditioningPolicy.FULL_SWITCH)))
    plan = components(native, segmented)
    assert [item[2] for item in plan] == ['default', 'alternate']
    assert plan[0][1] == (identity, identity, identity) and plan[1][1] == (alternate, alternate, alternate)
    with pytest.raises(VoiceError) as missing:
        native._validate_native_options(replace(base, expression_enabled=True, expression_profile='missing'))
    assert missing.value.code in {'reference_not_ready', 'unknown_expression', 'expression_not_found'} or missing.value.status_code in {404, 409}


@pytest.mark.parametrize("components,supports_mixed", [
    (None, False),
    (["semantic", "style", "acoustic_prompt"], True),
])
def test_worker_conditioning_capabilities_and_rejection(components, supports_mixed):
    binding, _, _ = prepared_binding()
    native = native_service()
    native._client.metadata = {"capabilities": {}}
    if components is not None:
        native._client.metadata["capabilities"]["conditioning_components"] = components
    native._expression_catalog = binding.catalog
    metadata = native.expression_metadata()
    assert metadata["policies"] == ([policy.value for policy in ConditioningPolicy]
                                    if supports_mixed else ["full-switch"])
    assert metadata["preferred_policy_supported"] is supports_mixed
    selected = replace(options(binding), expression_enabled=True, expression_profile="alternate",
                       expression_policy=ConditioningPolicy.IDENTITY_LOCK)
    if supports_mixed:
        native.validate_expression(selected)
        assert components_for_policy(native, selected) == "identity-lock"
        native._complete_mode = "captured"
        with pytest.raises(service.RequestError, match="requires cached") as error:
            native.synthesize(selected)
        assert error.value.status_code == 400
    else:
        with pytest.raises(service.RequestError, match="only full-switch") as error:
            native.validate_expression(selected)
        assert error.value.status_code == 400
        with pytest.raises(service.RequestError, match="only full-switch"):
            native._segment_conditionings(selected)
        full_switch = replace(selected, expression_policy=ConditioningPolicy.FULL_SWITCH)
        native.validate_expression(full_switch)
        assert components_for_policy(native, full_switch) == "full-switch"


@pytest.mark.parametrize("components,expected", [
    (None, ["full-switch"]),
    (["semantic", "style", "acoustic_prompt"], [policy.value for policy in ConditioningPolicy]),
])
def test_shared_base_without_package_catalog_still_reports_worker_policies(components, expected):
    native = native_service()
    native._client.metadata = {"capabilities": {}}
    if components is not None:
        native._client.metadata["capabilities"]["conditioning_components"] = components
    native._expression_catalog = None
    metadata = native.expression_metadata()
    assert metadata["enabled"] is False and metadata["profiles"] == []
    assert metadata["policies"] == expected


def components_for_policy(native, value):
    return components(native, value)[0][3]
