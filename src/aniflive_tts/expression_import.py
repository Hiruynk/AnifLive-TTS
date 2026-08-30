from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import PackageValidationError
from .expression import (
    CANONICAL_EXPRESSION_LANGUAGES,
    ConditioningPolicy,
    parse_expression_runtime_policy,
    parse_playback_policy,
)
from .model_package import (
    resolve_contained_path,
    sha256_file,
    validate_checksums,
    validate_safe_identifier,
    write_checksums,
)


@dataclass(frozen=True)
class WaveMetadata:
    channels: int
    sample_rate: int
    sample_width_bytes: int
    frames: int
    duration_seconds: float


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackageValidationError(f"Unable to read {label}: {path}") from error
    if not isinstance(value, dict):
        raise PackageValidationError(f"{label} must contain a JSON object")
    return value


def _finite_unit_interval(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PackageValidationError(f"{label} must be a number from 0 to 1")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise PackageValidationError(f"{label} must be a number from 0 to 1")
    return result


def _validate_wav(path: Path) -> WaveMetadata:
    try:
        with wave.open(str(path), "rb") as source:
            if source.getcomptype() != "NONE":
                raise PackageValidationError("Expression reference WAV must be uncompressed PCM")
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            sample_width = source.getsampwidth()
            frames = source.getnframes()
    except (OSError, EOFError, wave.Error) as error:
        raise PackageValidationError(f"Invalid expression reference WAV: {path}") from error
    if channels not in {1, 2}:
        raise PackageValidationError("Expression reference WAV must be mono or stereo")
    if sample_rate < 8000 or sample_rate > 192000:
        raise PackageValidationError("Expression reference WAV has an unsupported sample rate")
    if sample_width not in {2, 3, 4}:
        raise PackageValidationError("Expression reference WAV must use 16/24/32-bit PCM")
    if frames <= 0:
        raise PackageValidationError("Expression reference WAV is empty")
    duration = frames / sample_rate
    if duration < 0.5 or duration > 30.0:
        raise PackageValidationError(
            "Expression reference WAV duration must be between 0.5 and 30 seconds"
        )
    return WaveMetadata(
        channels=channels,
        sample_rate=sample_rate,
        sample_width_bytes=sample_width,
        frames=frames,
        duration_seconds=duration,
    )


def _copy_package(source: Path, destination: Path) -> None:
    def copy_file(path: str, target: str) -> str:
        try:
            os.link(path, target)
            return target
        except OSError:
            return shutil.copy2(path, target)

    shutil.copytree(source, destination, copy_function=copy_file)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _parse_vad(item: Mapping[str, Any], label: str) -> dict[str, float] | None:
    raw = item.get("vad")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise PackageValidationError(f"{label}.vad must be an object")
    return {
        name: _finite_unit_interval(raw.get(name), f"{label}.vad.{name}")
        for name in ("valence", "arousal", "dominance")
    }


def import_expression_profiles(
    *,
    model_package: Path,
    voice_profile: str,
    spec_file: Path,
    asset_root: Path,
    output: Path,
) -> Path:
    """Publish an expression-enabled package without mutating the source package."""

    model_package = model_package.expanduser().resolve()
    asset_root = asset_root.expanduser().resolve()
    spec_file = spec_file.expanduser().resolve()
    output = output.expanduser().resolve()
    if output == model_package:
        raise PackageValidationError("Expression import output must differ from the source package")
    validate_checksums(model_package)
    manifest = _read_object(model_package / "manifest.json", "model package manifest")
    if manifest.get("format") != "aniflive-tts-model-package":
        raise PackageValidationError("Unsupported model package format")
    voice_profile = validate_safe_identifier(voice_profile, "voice_profile")
    voice_profiles = manifest.get("voice_profiles")
    if not isinstance(voice_profiles, list) or voice_profile not in voice_profiles:
        raise PackageValidationError(f"Unknown voice profile: {voice_profile}")
    if not asset_root.is_dir():
        raise PackageValidationError(f"Expression asset root does not exist: {asset_root}")

    spec = _read_object(spec_file, "expression import specification")
    if spec.get("schema") != 1:
        raise PackageValidationError("Expression import specification must use schema 1")
    default_profile = validate_safe_identifier(
        spec.get("default_profile", "neutral"), "default_profile"
    )
    try:
        preferred_policy = ConditioningPolicy(
            spec.get("preferred_policy", ConditioningPolicy.SEMANTIC_STYLE.value)
        )
    except (TypeError, ValueError) as error:
        raise PackageValidationError("preferred_policy is unsupported") from error
    raw_output_gain = spec.get("output_gain", 1.0)
    if isinstance(raw_output_gain, bool) or not isinstance(raw_output_gain, (int, float)):
        raise PackageValidationError("output_gain must be a number from 0.1 to 8")
    output_gain = float(raw_output_gain)
    if not math.isfinite(output_gain) or not 0.1 <= output_gain <= 8.0:
        raise PackageValidationError("output_gain must be a number from 0.1 to 8")
    playback_policy = parse_playback_policy(spec.get("playback_policy"))
    runtime_policy = parse_expression_runtime_policy(spec.get("runtime_policy"))
    raw_profiles = spec.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise PackageValidationError("Expression import profiles must be a non-empty array")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    shutil.rmtree(staging)
    try:
        _copy_package(model_package, staging)
        profile_dir = resolve_contained_path(
            staging / "voices", voice_profile, "voice profile path"
        )
        profile_manifest_path = profile_dir / "profile.json"
        profile_manifest = _read_object(profile_manifest_path, "voice profile manifest")
        identity_audio = resolve_contained_path(
            profile_dir, profile_manifest.get("reference_audio"), "identity reference audio"
        )
        identity_meta = _validate_wav(identity_audio)
        identity_text = profile_manifest.get("reference_text")
        identity_language = profile_manifest.get("reference_language")
        if not isinstance(identity_text, str) or not identity_text.strip():
            raise PackageValidationError("Identity reference text must not be empty")
        if identity_language not in CANONICAL_EXPRESSION_LANGUAGES:
            raise PackageValidationError("Identity reference language is unsupported")

        packaged_profiles: list[dict[str, Any]] = [
            {
                "id": "neutral",
                "emotion": "neutral",
                "intensity": 0.0,
                "reference_audio": str(profile_manifest["reference_audio"]),
                "reference_text": identity_text.strip(),
                "reference_language": identity_language,
                "manual_verified": True,
                "sha256": sha256_file(identity_audio),
                "audio": {
                    "channels": identity_meta.channels,
                    "sample_rate": identity_meta.sample_rate,
                    "sample_width_bytes": identity_meta.sample_width_bytes,
                    "frames": identity_meta.frames,
                    "duration_seconds": round(identity_meta.duration_seconds, 6),
                },
            }
        ]
        seen = {"neutral"}
        expression_dir = profile_dir / "expressions"
        expression_dir.mkdir(exist_ok=True)
        for index, raw in enumerate(raw_profiles):
            label = f"profiles[{index}]"
            if not isinstance(raw, Mapping):
                raise PackageValidationError(f"{label} must be an object")
            if raw.get("manual_verified") is not True:
                raise PackageValidationError(f"{label}.manual_verified must be true")
            reference_id = validate_safe_identifier(raw.get("id"), f"{label}.id")
            emotion = validate_safe_identifier(raw.get("emotion"), f"{label}.emotion")
            if reference_id in seen:
                raise PackageValidationError(f"Duplicate expression profile id: {reference_id}")
            seen.add(reference_id)
            language = raw.get("reference_language")
            if language not in CANONICAL_EXPRESSION_LANGUAGES:
                raise PackageValidationError(f"{label}.reference_language is unsupported")
            text = raw.get("reference_text")
            if not isinstance(text, str) or not text.strip():
                raise PackageValidationError(f"{label}.reference_text must not be empty")
            source_audio = resolve_contained_path(
                asset_root, raw.get("reference_audio"), f"{label}.reference_audio"
            )
            if not source_audio.is_file():
                raise PackageValidationError(f"Missing expression reference WAV: {source_audio}")
            audio_meta = _validate_wav(source_audio)
            relative_audio = f"expressions/{reference_id}.wav"
            destination_audio = profile_dir / relative_audio
            # Package copies may use hard links for large immutable files. An
            # expression import can replace an existing reference with the same
            # id, so unlink the staging path before copying to keep the source
            # package inode immutable.
            destination_audio.unlink(missing_ok=True)
            shutil.copy2(source_audio, destination_audio)
            packaged = {
                "id": reference_id,
                "emotion": emotion,
                "intensity": _finite_unit_interval(raw.get("intensity"), f"{label}.intensity"),
                "reference_audio": relative_audio,
                "reference_text": text.strip(),
                "reference_language": language,
                "manual_verified": True,
                "sha256": sha256_file(destination_audio),
                "audio": {
                    "channels": audio_meta.channels,
                    "sample_rate": audio_meta.sample_rate,
                    "sample_width_bytes": audio_meta.sample_width_bytes,
                    "frames": audio_meta.frames,
                    "duration_seconds": round(audio_meta.duration_seconds, 6),
                },
            }
            if raw.get("preferred_policy") is not None:
                try:
                    profile_policy = ConditioningPolicy(raw.get("preferred_policy"))
                except (TypeError, ValueError) as error:
                    raise PackageValidationError(
                        f"{label}.preferred_policy is unsupported"
                    ) from error
                packaged["preferred_policy"] = profile_policy.value
            vad = _parse_vad(raw, label)
            if vad is not None:
                packaged["vad"] = vad
            packaged_profiles.append(packaged)

        available = {item["id"] for item in packaged_profiles} | {
            item["emotion"] for item in packaged_profiles
        }
        if default_profile not in available:
            raise PackageValidationError("default_profile is unavailable")
        profile_manifest["expression"] = {
            "schema": 1,
            "default_profile": default_profile,
            "preferred_policy": preferred_policy.value,
            "output_gain": output_gain,
            "playback_policy": playback_policy.public_metadata(),
            "runtime_policy": runtime_policy.public_metadata(),
            "profiles": packaged_profiles,
        }
        # The package copier may use hard links for large immutable engines.
        # Replace mutable manifests atomically so the source inode is untouched.
        _atomic_write_json(profile_manifest_path, profile_manifest)
        manifest["expression"] = {
            "schema": 1,
            "voice_profiles": [voice_profile],
            "controlled_profiles": True,
            "continuous_vector": False,
            "preferred_policy": preferred_policy.value,
            "output_gain": output_gain,
            "playback_policy": playback_policy.public_metadata(),
            "runtime_policy": runtime_policy.public_metadata(),
        }
        _atomic_write_json(staging / "manifest.json", manifest)
        write_checksums(staging)
        validate_checksums(staging)
        if output.exists():
            raise PackageValidationError(f"Expression import output already exists: {output}")
        os.replace(staging, output)
        return output
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
