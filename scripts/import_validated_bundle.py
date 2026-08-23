#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import tempfile
from pathlib import Path

from aniflive_tts.backend.contracts import STAGE_ORDER
from aniflive_tts.model_package import (
    ENGINE_RUNTIME_KEYS,
    engine_fingerprint,
    resolve_contained_path,
    runtime_fingerprint,
    sha256_file,
    validate_safe_identifier,
    write_checksums,
)


def require(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def validate_onnx_bundle(onnx_dir: Path) -> None:
    import onnx

    for stage in STAGE_ORDER:
        path = require(onnx_dir / f"{stage}.onnx", f"{stage} ONNX model")
        onnx.checker.check_model(str(path), full_check=True)


def validate_engine_bundle(engine_dir: Path) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    for stage in STAGE_ORDER:
        path = require(engine_dir / f"{stage}.engine", f"{stage} TensorRT engine")
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import already validated portable ONNX and same-machine TRT11 engines"
    )
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text-file", type=Path, required=True)
    parser.add_argument("--reference-language", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--voice-profile", default="default")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_id = validate_safe_identifier(args.model_id, "model_id")
    voice_profile = validate_safe_identifier(args.voice_profile, "voice_profile")

    onnx_dir = args.onnx_dir.resolve()
    engine_dir = args.engine_dir.resolve()
    reference_audio = require(args.reference_audio, "reference audio")
    reference_text_file = require(args.reference_text_file, "reference text")
    onnx_hashes = {
        path.name: sha256_file(path)
        for path in sorted(onnx_dir.iterdir())
        if path.is_file() and (path.suffix == ".onnx" or path.name == "config.json")
    }
    validate_onnx_bundle(onnx_dir)
    validate_engine_bundle(engine_dir)
    source_engine_manifest = json.loads(
        (engine_dir / "engine-manifest.json").read_text(encoding="utf-8")
    )
    recorded_runtime = source_engine_manifest.get("runtime")
    current_runtime = runtime_fingerprint()
    if not isinstance(recorded_runtime, dict) or any(
        str(recorded_runtime.get(key)) != str(current_runtime.get(key))
        for key in ENGINE_RUNTIME_KEYS
    ):
        raise RuntimeError(
            "Imported TensorRT engines do not match the current runtime fingerprint"
        )
    profiles = source_engine_manifest.get("payload", {}).get("settings", {"profile": "fitted"})
    build_config = {
        "precision": "FP16",
        "workspace_mib": source_engine_manifest.get("workspace_mib", 4096),
        "optimization_level": source_engine_manifest.get("builder_optimization_level", 5),
        "tf32": bool(source_engine_manifest.get("tf32_allowed", False)),
        "imported_validated_bundle": True,
    }
    fingerprint, fingerprint_payload = engine_fingerprint(
        onnx_hashes=onnx_hashes, profiles=profiles, build_config=build_config
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        shutil.copytree(onnx_dir, staging / "onnx", dirs_exist_ok=True)
        target_engines = staging / "engines" / fingerprint
        shutil.copytree(engine_dir, target_engines, dirs_exist_ok=True)
        source_engine_manifest.update(
            {
                "kind": "aniflive-tts-gsv-v2proplus-tensorrt11-engines",
                "fingerprint": fingerprint,
                "runtime": {key: fingerprint_payload[key] for key in ENGINE_RUNTIME_KEYS},
                "fingerprint_payload": fingerprint_payload,
            }
        )
        (target_engines / "engine-manifest.json").write_text(
            json.dumps(source_engine_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        voice_dir = resolve_contained_path(staging / "voices", voice_profile, "voice profile path")
        voice_dir.mkdir(parents=True)
        shutil.copy2(reference_audio, voice_dir / "reference.wav")
        reference_text = reference_text_file.read_text(encoding="utf-8-sig").strip()
        (voice_dir / "reference.txt").write_text(reference_text + "\n", encoding="utf-8")
        (voice_dir / "profile.json").write_text(
            json.dumps(
                {
                    "id": voice_profile,
                    "reference_audio": "reference.wav",
                    "reference_text": reference_text,
                    "reference_language": args.reference_language,
                    "conditioning_cache": "conditioning.pt",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema": 1,
            "format": "aniflive-tts-model-package",
            "model_id": model_id,
            "model_family": "gsv-v2proplus",
            "precision": "FP16",
            "backend": "TensorRT-11",
            "active_engine_fingerprint": fingerprint,
            "voice_profiles": [voice_profile],
            "default_voice_profile": voice_profile,
            "languages": ["zh", "yue", "en", "ja", "ko"],
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "build_config": build_config,
            "reference_conditioning_cached": False,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "source.json").write_text(
            json.dumps(
                {
                    "mode": "import-validated-bundle",
                    "onnx_sha256": onnx_hashes,
                    "reference_audio_sha256": sha256_file(reference_audio),
                    "original_checkpoints_included": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        write_checksums(staging)
        if output.exists():
            backup = output.parent / f"{output.name}.previous-{dt.datetime.now():%Y%m%dT%H%M%S}"
            os.replace(output, backup)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
