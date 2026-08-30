from __future__ import annotations

import json
from pathlib import Path

import pytest

from aniflive_tts.errors import PackageValidationError
from aniflive_tts.expression import (
    BoundaryKind,
    ConditioningPolicy,
    PreparedReference,
    PreparedReferenceBank,
    classify_boundary,
    has_safe_expression_boundary,
    load_expression_catalog,
    should_bridge_expression_context,
)


@pytest.mark.parametrize(
    ("text", "switched", "expected"),
    [
        ("Finished.", False, BoundaryKind.HARD_NATURAL),
        ("続けて、", False, BoundaryKind.SOFT_NATURAL),
        ("profile-safe split", False, BoundaryKind.TECHNICAL),
        ("same words", True, BoundaryKind.EXPLICIT_EXPRESSION),
    ],
)
def test_boundary_classification(
    text: str, switched: bool, expected: BoundaryKind
) -> None:
    assert classify_boundary(text, expression_switch=switched) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("話を", True),
        ("keep speaking", True),
        ("続けて、", False),
        ("Finished.", False),
    ],
)
def test_expression_context_bridge_only_crosses_mid_utterance_switches(
    text: str, expected: bool
) -> None:
    assert (
        should_bridge_expression_context(text, expression_switch=True) is expected
    )
    assert not should_bridge_expression_context(text, expression_switch=False)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("今日は、", True),
        ("Ready! ”", True),
        ("First clause;", True),
        ("not finished", False),
        ("label:", False),
    ],
)
def test_expression_switch_boundary_only_accepts_speech_safe_punctuation(
    text: str, expected: bool
) -> None:
    assert has_safe_expression_boundary(text) is expected


def _prepared(reference_id: str) -> PreparedReference:
    return PreparedReference(
        prompt_semantic=f"semantic:{reference_id}",
        spectrogram=f"spec:{reference_id}",
        speaker_embedding=f"speaker:{reference_id}",
        phones=[1, 2],
        bert=f"bert:{reference_id}",
        id=reference_id,
    )


def _profile(tmp_path: Path) -> dict:
    audio = tmp_path / "expressions" / "happy-ja" / "reference.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"RIFF")
    return {
        "id": "default",
        "expression": {
            "schema": 1,
            "default_profile": "happy",
            "profiles": [
                {
                    "id": "happy-ja",
                    "emotion": "happy",
                    "intensity": 0.7,
                    "reference_audio": "expressions/happy-ja/reference.wav",
                    "reference_text": "うれしいです。",
                    "reference_language": "ja",
                    "manual_verified": True,
                    "preferred_policy": "acoustic-style",
                }
            ],
        },
    }


def test_additive_schema_one_catalog_and_public_metadata(tmp_path: Path) -> None:
    catalog = load_expression_catalog(profile_dir=tmp_path, profile_manifest=_profile(tmp_path))
    assert catalog is not None
    selected = catalog.select(profile="happy", intensity=0.8, language="ja")
    assert selected.id == "happy-ja"
    assert selected.preferred_policy is ConditioningPolicy.ACOUSTIC_STYLE
    assert catalog.policy_for(
        profile="happy", intensity=0.8, language="ja"
    ) is ConditioningPolicy.ACOUSTIC_STYLE
    assert catalog.preferred_policy is ConditioningPolicy.SEMANTIC_STYLE
    assert catalog.output_gain == 1.0
    assert catalog.public_metadata() == {
        "default": "happy",
        "preferred_policy": "semantic-style",
        "playback_policy": {
            "short_prebuffer_ms": 32,
            "long_prebuffer_ms": 64,
        },
        "runtime_policy": {
            "preview_publish_seconds": 0.112,
            "minimum_initial_preview_seconds": 0.112,
            "progressive_refill_tokens": 8,
            "progressive_refill_seconds": 0.24,
            "progressive_refill_count": 1,
            "progressive_refill_min_segments": 2,
            "progressive_refill_min_phonemes": 80,
            "progressive_refill_short_tokens": 16,
            "progressive_refill_short_seconds": 0.112,
            "transition": "hard-natural",
            "transition_ms": 4.0,
            "sigmoid_k": 12.0,
            "first_context_tokens": 17,
            "onset_hold_ms": 0,
            "normalize_reference_semantic_onset": False,
            "reference_semantic_onset_threshold_dbfs": -45.0,
            "reference_semantic_onset_retain_ms": 40,
        },
        "profiles": [
            {
                "id": "happy",
                "intensity_levels": [0.7],
                "languages": ["en", "ja", "ko", "yue", "zh"],
                "reference_languages": ["ja"],
                "preferred_policies": ["acoustic-style"],
            }
        ],
    }


def test_expression_catalog_loads_reproducible_output_contract(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    profile["expression"]["preferred_policy"] = "semantic-style"
    profile["expression"]["output_gain"] = 1.75
    profile["expression"]["playback_policy"] = {
        "short_prebuffer_ms": 32,
        "long_prebuffer_ms": 208,
    }
    catalog = load_expression_catalog(profile_dir=tmp_path, profile_manifest=profile)
    assert catalog is not None
    assert catalog.preferred_policy is ConditioningPolicy.SEMANTIC_STYLE
    assert catalog.output_gain == 1.75
    assert catalog.playback_policy.short_prebuffer_ms == 32
    assert catalog.playback_policy.long_prebuffer_ms == 208
    assert catalog.runtime_policy.preview_publish_seconds == 0.112
    assert catalog.runtime_policy.minimum_initial_preview_seconds == 0.112
    assert catalog.runtime_policy.progressive_refill_tokens == 8
    assert catalog.runtime_policy.normalize_reference_semantic_onset is False
    assert catalog.runtime_policy.reference_semantic_onset_threshold_dbfs == -45.0
    assert catalog.runtime_policy.reference_semantic_onset_retain_ms == 40


@pytest.mark.parametrize("gain", [0, 8.1, float("inf"), True, "1.0"])
def test_expression_catalog_rejects_invalid_output_gain(tmp_path: Path, gain: object) -> None:
    profile = _profile(tmp_path)
    profile["expression"]["output_gain"] = gain
    with pytest.raises(PackageValidationError, match="output_gain"):
        load_expression_catalog(profile_dir=tmp_path, profile_manifest=profile)


@pytest.mark.parametrize(
    "policy",
    [
        "208",
        {"short_prebuffer_ms": True},
        {"long_prebuffer_ms": 501},
        {"short_prebuffer_ms": 64, "long_prebuffer_ms": 32},
    ],
)
def test_expression_catalog_rejects_invalid_playback_policy(
    tmp_path: Path, policy: object
) -> None:
    profile = _profile(tmp_path)
    profile["expression"]["playback_policy"] = policy
    with pytest.raises(PackageValidationError, match="playback_policy"):
        load_expression_catalog(profile_dir=tmp_path, profile_manifest=profile)


@pytest.mark.parametrize(
    "policy",
    [
        "invalid",
        {"preview_publish_seconds": 1.1},
        {"minimum_initial_preview_seconds": -0.1},
        {"progressive_refill_tokens": True},
        {"progressive_refill_count": 0},
        {"transition": "linear"},
        {"first_context_tokens": 0},
    ],
)
def test_expression_catalog_rejects_invalid_runtime_policy(
    tmp_path: Path, policy: object
) -> None:
    profile = _profile(tmp_path)
    profile["expression"]["runtime_policy"] = policy
    with pytest.raises(PackageValidationError, match="runtime_policy"):
        load_expression_catalog(profile_dir=tmp_path, profile_manifest=profile)


def test_reference_bank_separates_identity_and_expression_channels() -> None:
    bank = PreparedReferenceBank(identity_id="default")
    bank.add(_prepared("default"))
    bank.add(_prepared("happy-ja"))

    identity_lock = bank.conditioning(
        reference_id="happy-ja", policy=ConditioningPolicy.IDENTITY_LOCK
    )
    assert identity_lock.semantic.id == "happy-ja"
    assert identity_lock.spectrogram == "spec:happy-ja"
    assert identity_lock.speaker_embedding == "speaker:default"

    acoustic = bank.conditioning(
        reference_id="happy-ja", policy=ConditioningPolicy.ACOUSTIC_STYLE
    )
    assert acoustic.semantic.id == "default"
    assert acoustic.spectrogram == "spec:happy-ja"
    assert acoustic.speaker_embedding == "speaker:default"


@pytest.mark.parametrize(
    "path",
    ["../private.wav", "C:/private.wav", "/private.wav", "nested\\private.wav"],
)
def test_expression_reference_path_must_be_package_contained(
    tmp_path: Path, path: str
) -> None:
    profile = _profile(tmp_path)
    profile["expression"]["profiles"][0]["reference_audio"] = path
    with pytest.raises(PackageValidationError):
        load_expression_catalog(profile_dir=tmp_path, profile_manifest=profile)


def test_expression_catalog_rejects_unverified_reference(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    profile["expression"]["profiles"][0]["manual_verified"] = False
    with pytest.raises(PackageValidationError, match="manual_verified"):
        load_expression_catalog(profile_dir=tmp_path, profile_manifest=profile)


def test_expression_public_metadata_does_not_leak_paths_or_transcript(tmp_path: Path) -> None:
    catalog = load_expression_catalog(profile_dir=tmp_path, profile_manifest=_profile(tmp_path))
    assert catalog is not None
    payload = json.dumps(catalog.public_metadata())
    assert "reference.wav" not in payload
    assert "うれしい" not in payload
