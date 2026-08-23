from __future__ import annotations

import json
from pathlib import Path

import pytest

from aniflive_tts import api, model_package
from aniflive_tts.errors import EngineRebuildRequired, PackageValidationError
from aniflive_tts.inspector import CheckpointInspection
from aniflive_tts.settings import RuntimeSettings


RUNTIME_FINGERPRINT = {
    "tensorrt": "11.2.1.2",
    "cuda_runtime": "12.8",
    "compute_capability": "12.0",
    "gpu": "NVIDIA GeForce RTX 5070 Ti",
    "gpu_sm_count": "70",
    "gpu_total_memory_bytes": "17163091968",
    "platform_system": "linux",
    "platform_machine": "x86_64",
    "platform": "Linux-test",
}


def _engine_package(tmp_path: Path, runtime: dict[str, str]) -> tuple[Path, dict[str, str]]:
    fingerprint = "abcdef0123456789abcdef01"
    package_dir = tmp_path / "model"
    engine_dir = package_dir / "engines" / fingerprint
    engine_dir.mkdir(parents=True)
    (engine_dir / "engine-manifest.json").write_text(
        json.dumps({"fingerprint": fingerprint, "runtime": runtime}), encoding="utf-8"
    )
    for stage in model_package.STAGE_ORDER:
        (engine_dir / f"{stage}.engine").write_bytes(b"engine")
    return package_dir, {"active_engine_fingerprint": fingerprint}


def test_engine_bundle_accepts_matching_platform_and_gpu(tmp_path, monkeypatch) -> None:
    package_dir, manifest = _engine_package(tmp_path, dict(RUNTIME_FINGERPRINT))
    monkeypatch.setattr(model_package, "runtime_fingerprint", lambda: dict(RUNTIME_FINGERPRINT))

    selected = model_package.select_engine_dir(package_dir, manifest)

    assert selected == (package_dir / "engines" / manifest["active_engine_fingerprint"]).resolve()


@pytest.mark.parametrize(
    ("field", "package_value"),
    (("platform_system", "windows"), ("gpu", "NVIDIA GeForce RTX 4090")),
)
def test_engine_mismatch_requires_target_linux_rebuild(
    tmp_path, monkeypatch, field, package_value
) -> None:
    recorded = dict(RUNTIME_FINGERPRINT)
    recorded[field] = package_value
    package_dir, manifest = _engine_package(tmp_path, recorded)
    monkeypatch.setattr(model_package, "runtime_fingerprint", lambda: dict(RUNTIME_FINGERPRINT))

    with pytest.raises(EngineRebuildRequired) as captured:
        model_package.select_engine_dir(package_dir, manifest)

    message = str(captured.value)
    assert "ENGINE_REBUILD_REQUIRED" in message
    assert field in message
    assert "target Linux container" in message
    assert "aniflive-tts model rebuild-engines" in message


def test_legacy_engine_manifest_without_platform_identity_requires_rebuild(
    tmp_path, monkeypatch
) -> None:
    recorded = {
        key: value
        for key, value in RUNTIME_FINGERPRINT.items()
        if key not in {"platform_system", "platform_machine"}
    }
    package_dir, manifest = _engine_package(tmp_path, recorded)
    monkeypatch.setattr(model_package, "runtime_fingerprint", lambda: dict(RUNTIME_FINGERPRINT))

    with pytest.raises(EngineRebuildRequired, match="platform_system mismatch"):
        model_package.select_engine_dir(package_dir, manifest)


def test_engine_fingerprint_cannot_escape_engine_directory(tmp_path) -> None:
    with pytest.raises(EngineRebuildRequired, match="ENGINE_REBUILD_REQUIRED"):
        model_package.select_engine_dir(
            tmp_path,
            {"active_engine_fingerprint": "../escape"},
        )


@pytest.mark.parametrize(
    "value",
    (
        "../escape",
        "voice/profile",
        r"voice\profile",
        ".hidden",
        "trailing.",
        "bad profile",
        "C:escape",
        "NUL",
    ),
)
def test_voice_profile_identifier_rejects_paths(value) -> None:
    with pytest.raises(PackageValidationError):
        model_package.validate_safe_identifier(value, "voice_profile")


def test_voice_profile_identifier_accepts_safe_id() -> None:
    assert (
        model_package.validate_safe_identifier("miku-v2_pro.plus", "voice_profile")
        == "miku-v2_pro.plus"
    )


@pytest.mark.parametrize("value", ("../reference.wav", r"..\reference.wav", "/tmp/a.wav"))
def test_package_path_rejects_traversal(tmp_path, value) -> None:
    with pytest.raises(PackageValidationError):
        model_package.resolve_contained_path(tmp_path, value, "reference audio path")


def test_checksum_validation_rejects_unlisted_files(tmp_path: Path) -> None:
    package_dir = tmp_path / "model"
    package_dir.mkdir()
    (package_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (package_dir / "unlisted.engine").write_bytes(b"engine")
    (package_dir / "checksums.json").write_text(
        json.dumps({"manifest.json": model_package.sha256_file(package_dir / "manifest.json")}),
        encoding="utf-8",
    )

    with pytest.raises(PackageValidationError, match="unlisted files"):
        model_package.validate_checksums(package_dir)


def test_checksum_validation_requires_sha256_values(tmp_path: Path) -> None:
    package_dir = tmp_path / "model"
    package_dir.mkdir()
    (package_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (package_dir / "checksums.json").write_text(
        json.dumps({"manifest.json": "not-a-sha256"}), encoding="utf-8"
    )

    with pytest.raises(PackageValidationError, match="Invalid SHA-256"):
        model_package.validate_checksums(package_dir)


def test_api_rejects_voice_profile_traversal_before_path_access(tmp_path, monkeypatch) -> None:
    package_dir = tmp_path / "model"
    package_dir.mkdir()
    (package_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "aniflive-tts-model-package",
                "model_id": "test-model",
                "model_family": "gsv-v2proplus",
                "precision": "FP16",
                "default_voice_profile": "../escape",
                "voice_profiles": ["../escape"],
            }
        ),
        encoding="utf-8",
    )
    settings = RuntimeSettings(package_dir, tmp_path / "shared", tmp_path / "cache")
    monkeypatch.delenv("ANIFLIVE_TTS_VOICE_PROFILE", raising=False)
    monkeypatch.setattr(api, "validate_checksums", lambda path: {})
    monkeypatch.setattr(api, "select_engine_dir", lambda package, manifest: tmp_path / "engines")

    with pytest.raises(PackageValidationError, match="voice_profile"):
        api.configure_runtime(settings)


def test_api_rejects_reference_audio_traversal(tmp_path, monkeypatch) -> None:
    package_dir = tmp_path / "model"
    profile_dir = package_dir / "voices" / "default"
    profile_dir.mkdir(parents=True)
    (package_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "aniflive-tts-model-package",
                "model_id": "test-model",
                "model_family": "gsv-v2proplus",
                "precision": "FP16",
                "default_voice_profile": "default",
                "voice_profiles": ["default"],
            }
        ),
        encoding="utf-8",
    )
    (profile_dir / "profile.json").write_text(
        json.dumps(
            {
                "id": "default",
                "reference_audio": "../outside.wav",
                "reference_text": "reference",
                "reference_language": "en",
            }
        ),
        encoding="utf-8",
    )
    settings = RuntimeSettings(package_dir, tmp_path / "shared", tmp_path / "cache")
    monkeypatch.delenv("ANIFLIVE_TTS_VOICE_PROFILE", raising=False)
    monkeypatch.setattr(api, "validate_checksums", lambda path: {})
    monkeypatch.setattr(api, "select_engine_dir", lambda package, manifest: tmp_path / "engines")

    with pytest.raises(PackageValidationError, match="reference audio path"):
        api.configure_runtime(settings)


def test_api_rejects_unsafe_model_id_before_package_access(tmp_path, monkeypatch) -> None:
    package_dir = tmp_path / "model"
    package_dir.mkdir()
    (package_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "aniflive-tts-model-package",
                "model_id": "../escape",
                "model_family": "gsv-v2proplus",
                "precision": "FP16",
            }
        ),
        encoding="utf-8",
    )
    settings = RuntimeSettings(package_dir, tmp_path / "shared", tmp_path / "cache")
    monkeypatch.setattr(api, "validate_checksums", lambda path: pytest.fail("must reject first"))

    with pytest.raises(PackageValidationError, match="model_id"):
        api.configure_runtime(settings)


def test_checkpoint_inspection_public_dict_excludes_private_paths_and_config(
    tmp_path: Path,
) -> None:
    inspection = CheckpointInspection(
        path=tmp_path / "private" / "voice.ckpt",
        sha256="a" * 64,
        kind="gpt",
        model_version="gsv-v2proplus",
        config={"output_dir": str(tmp_path / "private" / "training")},
        state_key_count=1,
        evidence=("shape match",),
    )
    payload = inspection.to_dict()
    assert payload["filename"] == "voice.ckpt"
    assert "path" not in payload
    assert "config" not in payload


def test_converter_applies_safe_voice_profile_and_complete_runtime_fingerprint() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "src" / "aniflive_tts" / "converter.py"
    ).read_text(encoding="utf-8")
    assert 'validate_safe_identifier(model_id, "model_id")' in source
    assert 'validate_safe_identifier(voice_profile, "voice_profile")' in source
    assert "for key in ENGINE_RUNTIME_KEYS" in source


def test_exported_runtime_config_does_not_publish_full_training_hparams() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "minimal_inference"
        / "export_onnx.py"
    ).read_text(encoding="utf-8")
    assert "config_dict = hparams_to_dict(hps)" not in source
    assert '"semantic_frame_rate": "25hz"' in source
    assert '"sampling_rate": hps_obj.data.sampling_rate' in source


def test_validated_bundle_import_applies_safe_ids_and_complete_runtime_fingerprint() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "import_validated_bundle.py"
    ).read_text(encoding="utf-8")
    assert 'validate_safe_identifier(args.model_id, "model_id")' in source
    assert 'validate_safe_identifier(args.voice_profile, "voice_profile")' in source
    assert "for key in ENGINE_RUNTIME_KEYS" in source
    assert "resolve_contained_path(" in source
    assert "validate_onnx_bundle(onnx_dir)" in source
    assert "validate_engine_bundle(engine_dir)" in source
    assert "runtime_fingerprint()" in source


def test_audio_quality_uses_project_root_and_safe_checkpoint_loading() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "aniflive_tts"
        / "backend"
        / "audio_quality.py"
    ).read_text(encoding="utf-8")
    assert 'parents[3] / "minimal_inference"' in source
    assert 'torch.load(model_path, map_location="cpu", weights_only=True)' in source
    assert "weights_only=False" not in source


def test_legacy_builder_cli_is_blocked() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "aniflive_tts"
        / "backend"
        / "legacy_converter.py"
    ).read_text(encoding="utf-8")
    assert "asset-lock.json" not in source
    assert "export_v2proplus_onnx.py" not in source
    assert "legacy_converter is an internal TensorRT builder" in source
