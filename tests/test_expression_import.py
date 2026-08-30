from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from aniflive_tts.errors import PackageValidationError
from aniflive_tts.expression import load_expression_catalog
from aniflive_tts.expression_import import import_expression_profiles
from aniflive_tts.model_package import validate_checksums, write_checksums


def _wav(path: Path, *, frames: int = 16000) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * frames)


def _package(
    tmp_path: Path,
    *,
    model_id: str = "third-party-v2proplus",
    model_family: str = "gsv-v2proplus",
) -> Path:
    package = tmp_path / "source"
    profile = package / "voices" / "default"
    profile.mkdir(parents=True)
    _wav(profile / "reference.wav")
    (profile / "profile.json").write_text(
        json.dumps(
            {
                "id": "default",
                "reference_audio": "reference.wav",
                "reference_text": "基準です。",
                "reference_language": "ja",
            }
        ),
        encoding="utf-8",
    )
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "format": "aniflive-tts-model-package",
                "model_id": model_id,
                "model_family": model_family,
                "voice_profiles": ["default"],
                "default_voice_profile": "default",
            }
        ),
        encoding="utf-8",
    )
    write_checksums(package)
    return package


def _spec(path: Path, *, reference_audio: str = "shy.wav", verified: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "default_profile": "neutral",
                "preferred_policy": "semantic-style",
                "output_gain": 1.25,
                "playback_policy": {
                    "short_prebuffer_ms": 32,
                    "long_prebuffer_ms": 208,
                },
                "runtime_policy": {
                    "preview_publish_seconds": 0.112,
                    "minimum_initial_preview_seconds": 0.075,
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
                        "id": "shy-1",
                        "emotion": "shy",
                        "intensity": 0.7,
                        "reference_audio": reference_audio,
                        "reference_text": "ありがとうございます。",
                        "reference_language": "ja",
                        "manual_verified": verified,
                        "preferred_policy": "acoustic-style",
                        "vad": {"valence": 0.6, "arousal": 0.4, "dominance": 0.3},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_import_publishes_valid_package_without_mutating_source(tmp_path: Path) -> None:
    source = _package(tmp_path)
    source_profile = (source / "voices" / "default" / "profile.json").read_bytes()
    assets = tmp_path / "assets"
    assets.mkdir()
    _wav(assets / "shy.wav")
    spec = tmp_path / "spec.json"
    _spec(spec)

    output = import_expression_profiles(
        model_package=source,
        voice_profile="default",
        spec_file=spec,
        asset_root=assets,
        output=tmp_path / "output",
    )

    validate_checksums(output)
    assert (source / "voices" / "default" / "profile.json").read_bytes() == source_profile
    profile_dir = output / "voices" / "default"
    profile = json.loads((profile_dir / "profile.json").read_text(encoding="utf-8"))
    catalog = load_expression_catalog(profile_dir=profile_dir, profile_manifest=profile)
    assert catalog is not None
    assert catalog.select(profile="shy", intensity=0.8, language="ja").id == "shy-1"
    assert catalog.policy_for(
        profile="shy", intensity=0.8, language="ja"
    ).value == "acoustic-style"
    assert catalog.preferred_policy.value == "semantic-style"
    assert catalog.output_gain == 1.25
    assert catalog.playback_policy.short_prebuffer_ms == 32
    assert catalog.playback_policy.long_prebuffer_ms == 208
    assert catalog.runtime_policy.preview_publish_seconds == 0.112
    assert catalog.runtime_policy.minimum_initial_preview_seconds == 0.075
    assert catalog.runtime_policy.progressive_refill_tokens == 8
    assert catalog.runtime_policy.onset_hold_ms == 0
    assert catalog.runtime_policy.normalize_reference_semantic_onset is False
    assert catalog.runtime_policy.reference_semantic_onset_retain_ms == 40
    assert (profile_dir / "expressions" / "shy-1.wav").is_file()


@pytest.mark.parametrize(
    "model_id",
    ["third-party-v2proplus", "voice-a", "custom-japanese-v2pp"],
)
def test_import_is_generic_for_v2proplus_model_ids(
    tmp_path: Path, model_id: str
) -> None:
    source = _package(tmp_path, model_id=model_id)
    assets = tmp_path / "assets"
    assets.mkdir()
    _wav(assets / "shy.wav")
    spec = tmp_path / "spec.json"
    _spec(spec)

    output = import_expression_profiles(
        model_package=source,
        voice_profile="default",
        spec_file=spec,
        asset_root=assets,
        output=tmp_path / "output",
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == model_id
    assert manifest["model_family"] == "gsv-v2proplus"


def test_reimport_replaces_reference_without_mutating_source_package(
    tmp_path: Path,
) -> None:
    source = _package(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    _wav(assets / "shy.wav", frames=16000)
    spec = tmp_path / "spec.json"
    _spec(spec)
    first = import_expression_profiles(
        model_package=source,
        voice_profile="default",
        spec_file=spec,
        asset_root=assets,
        output=tmp_path / "first",
    )
    source_expression = first / "voices" / "default" / "expressions" / "shy-1.wav"
    original_reference = source_expression.read_bytes()

    _wav(assets / "shy.wav", frames=24000)
    second = import_expression_profiles(
        model_package=first,
        voice_profile="default",
        spec_file=spec,
        asset_root=assets,
        output=tmp_path / "second",
    )

    validate_checksums(first)
    validate_checksums(second)
    assert source_expression.read_bytes() == original_reference
    assert (
        second / "voices" / "default" / "expressions" / "shy-1.wav"
    ).read_bytes() != original_reference


@pytest.mark.parametrize("reference_audio", ["../shy.wav", "/tmp/shy.wav", "C:/shy.wav"])
def test_import_rejects_assets_outside_root(tmp_path: Path, reference_audio: str) -> None:
    source = _package(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    _wav(tmp_path / "shy.wav")
    spec = tmp_path / "spec.json"
    _spec(spec, reference_audio=reference_audio)
    with pytest.raises(PackageValidationError):
        import_expression_profiles(
            model_package=source,
            voice_profile="default",
            spec_file=spec,
            asset_root=assets,
            output=tmp_path / "output",
        )


def test_import_requires_manual_verification(tmp_path: Path) -> None:
    source = _package(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    _wav(assets / "shy.wav")
    spec = tmp_path / "spec.json"
    _spec(spec, verified=False)
    with pytest.raises(PackageValidationError, match="manual_verified"):
        import_expression_profiles(
            model_package=source,
            voice_profile="default",
            spec_file=spec,
            asset_root=assets,
            output=tmp_path / "output",
        )
