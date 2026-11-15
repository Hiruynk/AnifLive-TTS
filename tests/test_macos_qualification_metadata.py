from types import SimpleNamespace

import pytest

from aniflive_tts import service
from aniflive_tts.macos_service import MacOSV5Service, _CANONICAL_VOCODER_STREAM_CONTRACT


def native(monkeypatch, *, canonical, research=False, ready=True, precision="FP16"):
    monkeypatch.setenv("ANIFLIVE_TTS_NATIVE_PRECISION", precision)
    monkeypatch.setenv("ANIFLIVE_TTS_NATIVE_CANONICAL_STREAM", "1" if canonical else "0")
    monkeypatch.setenv("ANIFLIVE_TTS_ALLOW_UNQUALIFIED_STREAM", "1" if research else "0")
    native_service = MacOSV5Service(service.RuntimeSettings.from_env())
    native_service._client = SimpleNamespace(ready=ready, metadata={"prototype": True, "qualified": False} if ready else None)
    if ready:
        native_service._canonical_stream_worker_capability = {
            **_CANONICAL_VOCODER_STREAM_CONTRACT, "enabled": canonical,
        }
    return native_service


def test_macos_metadata_separates_availability_and_historical_evidence(monkeypatch):
    native_service = native(monkeypatch, canonical=True)
    caps = native_service.capabilities({})
    health = native_service.health()
    assert caps["quality_qualified"] is False and caps["prototype"] is False
    assert caps["streaming"]["quality_qualified"] is False
    assert caps["streaming"]["pcm16"] is True
    assert native_service._expression_catalog is None
    assert caps["expression"]["native"] is True
    assert caps["qualification"] == health["qualification"]
    qualification = caps["qualification"]
    assert qualification["current_runtime"] == "not_assessed"
    assert qualification["reference_validation"] == "accepted_with_documented_exceptions"
    assert qualification["worker_protocol"]["prototype"] is True
    assert qualification["worker_protocol"]["qualified"] is False
    assert qualification["complete_wav"]["available"] is True
    assert qualification["pcm_stream"]["available"] is True
    assert qualification["pcm_stream"]["worker_quality_qualified"] is False
    assert qualification["speech_sessions"]["policy_a"] == "supported"
    assert qualification["speech_sessions"]["policies_b_to_f"] == "unsupported"
    quality = qualification["objective_quality"]
    assert (quality["original_gate_passed"], quality["case_count"]) == (14, 17)
    assert len(quality["accepted_tolerances"]) == 2
    assert len(quality["official_model_limitations"]) == 1
    assert quality["all_original_gates_passed"] is False
    assert "full matrix not rerun" in quality["p1_evidence"]
    assert qualification["identity_style_equivalence"] == "passed_by_explicit_owner_acceptance"
    assert caps["speech_sessions"]["qualification"] == "qualified-policy-a"
    assert qualification["host"]["current_host"] == "not_assessed"


@pytest.mark.parametrize("canonical,research,ready,expected", [
    (False, False, True, False),
    (False, True, True, True),
    (True, False, False, False),
])
def test_availability_follows_mode_and_readiness(monkeypatch, canonical, research, ready, expected):
    native_service = native(monkeypatch, canonical=canonical, research=research, ready=ready, precision="FP32")
    metadata = native_service.capabilities({})["qualification"]
    assert metadata["pcm_stream"]["available"] is expected
    assert metadata["complete_wav"]["available"] is ready
    assert metadata["speech_sessions"]["policy_a_stream_available"] is expected
    assert "historical FP16" in metadata["objective_quality"]["scope"]
    assert metadata["current_runtime"] == "not_assessed"


def test_worker_flag_is_contract_data_not_admission_qualification():
    contract = {**_CANONICAL_VOCODER_STREAM_CONTRACT, "enabled": True}
    assert MacOSV5Service._validate_canonical_stream_contract(
        {"canonical_vocoder_stream": contract}, True
    )["quality_qualified"] is False
    with pytest.raises(RuntimeError, match="unqualified"):
        MacOSV5Service._validate_canonical_stream_contract(
            {"canonical_vocoder_stream": {**contract, "quality_qualified": True}}, True
        )


@pytest.mark.parametrize("components,supports_mixed", [
    (None, False),
    (["semantic", "style", "acoustic_prompt"], True),
])
def test_expression_capabilities_follow_native_worker(monkeypatch, components, supports_mixed):
    monkeypatch.setenv("ANIFLIVE_TTS_NATIVE_COMPLETE_MODE", "cached")
    native_service = native(monkeypatch, canonical=True)
    native_service._client.metadata["capabilities"] = {}
    if components is not None:
        native_service._client.metadata["capabilities"]["conditioning_components"] = components
    caps = native_service.capabilities({})
    assert caps["native_worker"]["separate_conditioning_components"] is supports_mixed
    assert len(caps["expression"]["policies"]) == (5 if supports_mixed else 1)
    assert caps["expression"]["complete_policies"] == caps["expression"]["policies"]
    assert "full-switch" in caps["expression"]["policies"]
