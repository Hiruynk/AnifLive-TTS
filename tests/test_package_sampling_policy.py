import pytest
from aniflive_tts.sampling_policy import (
    LEGACY, NATIVE, package_sampling_contract, effective_sampling_contract,
)


def test_new_and_legacy_packages_select_their_own_contract(monkeypatch):
    monkeypatch.delenv("ANIFLIVE_TTS_SEMANTIC_SAMPLING", raising=False)
    assert effective_sampling_contract({"semantic_sampling": NATIVE}) == NATIVE
    assert effective_sampling_contract({}) == LEGACY


def test_switching_package_default_does_not_leave_native_enabled(monkeypatch):
    monkeypatch.delenv("ANIFLIVE_TTS_SEMANTIC_SAMPLING", raising=False)
    monkeypatch.setenv("ANIFLIVE_TTS_PACKAGE_SEMANTIC_SAMPLING", NATIVE)
    assert effective_sampling_contract() == NATIVE
    monkeypatch.setenv("ANIFLIVE_TTS_PACKAGE_SEMANTIC_SAMPLING", LEGACY)
    assert effective_sampling_contract() == LEGACY


def test_explicit_diagnostic_override_is_preserved(monkeypatch):
    monkeypatch.setenv("ANIFLIVE_TTS_SEMANTIC_SAMPLING", NATIVE)
    assert effective_sampling_contract({}) == NATIVE


@pytest.mark.parametrize("value", [None, [], "unknown"])
def test_invalid_package_policy_cannot_silently_fall_back(value):
    with pytest.raises(ValueError, match="Unsupported"):
        package_sampling_contract({"semantic_sampling": value})
