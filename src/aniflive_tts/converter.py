from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .backend.contracts import ModelPaths, STAGE_ORDER
from .backend.legacy_converter import build_engines, validate_onnx_bundle
from .backend.profiles import FITTED_PROFILE
from .errors import PackageValidationError
from .inspector import inspect_pair
from .model_package import (
    ENGINE_RUNTIME_KEYS,
    engine_fingerprint,
    resolve_contained_path,
    sha256_file,
    validate_safe_identifier,
    write_checksums,
)


class ConversionError(RuntimeError):
    pass


def _require(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    printable = " ".join(command)
    print(f"[aniflive-tts-convert] {printable}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def export_onnx(
    *,
    gpt: Path,
    sovits: Path,
    shared_dir: Path,
    source_dir: Path,
    output_dir: Path,
    max_len: int,
    allow_unsafe_pickle: bool = False,
) -> dict[str, str]:
    hubert = _require(shared_dir / "chinese-hubert-base" / "pytorch_model.bin", "HuBERT weights").parent
    bert = _require(
        shared_dir / "chinese-roberta-wwm-ext-large" / "pytorch_model.bin", "BERT weights"
    ).parent
    sv = _require(shared_dir / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt", "speaker model")
    exporter = _require(source_dir / "export_onnx.py", "minimal inference exporter")
    fp16_converter = _require(source_dir / "onnx_to_fp16.py", "FP16 converter")
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32 = output_dir.parent / f".{output_dir.name}.fp32-{uuid.uuid4().hex[:8]}"
    fp16 = output_dir.parent / f".{output_dir.name}.fp16-{uuid.uuid4().hex[:8]}"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(source_dir), str(source_dir / "GPT_SoVITS"))),
        "SV_MODEL_PATH": str(sv),
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    try:
        export_command = [
            sys.executable,
            str(exporter),
            "--gpt_path",
            str(gpt),
            "--sovits_path",
            str(sovits),
            "--cnhubert_base_path",
            str(hubert),
            "--bert_path",
            str(bert),
            "--sv_path",
            str(sv),
            "--max_len",
            str(max_len),
            "--output_dir",
            str(fp32),
        ]
        if allow_unsafe_pickle:
            export_command.append("--allow_unsafe_pickle")
        _run(export_command, cwd=source_dir, env=env)
        _run(
            [sys.executable, str(fp16_converter), "--input_dir", str(fp32), "--output_dir", str(fp16)],
            cwd=source_dir,
            env=env,
        )
        hashes = validate_onnx_bundle(fp16)
        if output_dir.exists():
            backup = output_dir.parent / f"{output_dir.name}.previous-{dt.datetime.now():%Y%m%dT%H%M%S}"
            os.replace(output_dir, backup)
        os.replace(fp16, output_dir)
        return hashes
    finally:
        shutil.rmtree(fp32, ignore_errors=True)
        shutil.rmtree(fp16, ignore_errors=True)


def _jsonable_profiles() -> dict[str, Any]:
    return {
        stage: {
            name: {
                "min": list(shape.min_shape),
                "opt": list(shape.opt_shape),
                "max": list(shape.max_shape),
            }
            for name, shape in profile.items()
        }
        for stage, profile in FITTED_PROFILE.items()
    }


def rebuild_engines(
    *,
    model_package: Path,
    workspace_mib: int = 4096,
    optimization_level: int = 5,
    force: bool = False,
) -> Path:
    """Build a hardware-specific engine bundle from a package's portable ONNX.

    Checkpoints are intentionally not needed here.  A model package can be
    moved between supported hosts, while serialized engines are selected by a
    fingerprint containing the exact TensorRT/CUDA/GPU build environment.
    """

    model_package = model_package.expanduser().resolve()
    manifest_path = _require(model_package / "manifest.json", "model package manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "aniflive-tts-model-package":
        raise ConversionError("Unsupported model package format")
    if manifest.get("model_family") != "gsv-v2proplus":
        raise ConversionError("AnifLive-TTS v1 currently rebuilds gsv-v2proplus packages")

    onnx_dir = model_package / "onnx"
    onnx_hashes = validate_onnx_bundle(onnx_dir)
    build_config = {
        "precision": "FP16",
        "workspace_mib": int(workspace_mib),
        "optimization_level": int(optimization_level),
        "tf32": False,
    }
    fingerprint, fingerprint_payload = engine_fingerprint(
        onnx_hashes=onnx_hashes,
        profiles=_jsonable_profiles(),
        build_config=build_config,
    )
    engine_dir = model_package / "engines" / fingerprint
    if engine_dir.is_dir() and not force:
        missing = [stage for stage in STAGE_ORDER if not (engine_dir / f"{stage}.engine").is_file()]
        if not missing:
            manifest["active_engine_fingerprint"] = fingerprint
            manifest["build_config"] = build_config
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            write_checksums(model_package)
            return engine_dir

    placeholder = model_package / "manifest.json"
    paths = ModelPaths(
        gpt_checkpoint=placeholder,
        sovits_checkpoint=placeholder,
        reference_audio=placeholder,
        onnx_dir=onnx_dir,
        engine_dir=engine_dir,
        tokenizer_dir=model_package,
    )
    build_engines(
        paths,
        workspace_mib=workspace_mib,
        optimization_level=optimization_level,
    )
    engine_manifest_path = engine_dir / "engine-manifest.json"
    engine_manifest = json.loads(engine_manifest_path.read_text(encoding="utf-8"))
    engine_manifest.update(
        {
            "kind": "aniflive-tts-gsv-v2proplus-tensorrt11-engines",
            "fingerprint": fingerprint,
            "runtime": {key: fingerprint_payload[key] for key in ENGINE_RUNTIME_KEYS},
            "fingerprint_payload": fingerprint_payload,
        }
    )
    engine_manifest_path.write_text(
        json.dumps(engine_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["active_engine_fingerprint"] = fingerprint
    manifest["build_config"] = build_config
    manifest["engines_rebuilt_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_checksums(model_package)
    return engine_dir


def convert_model(
    *,
    gpt: Path,
    sovits: Path,
    reference_audio: Path,
    reference_text_file: Path,
    reference_language: str,
    model_id: str,
    voice_profile: str,
    output: Path,
    shared_dir: Path,
    source_dir: Path,
    allow_unsafe_pickle: bool = False,
    max_len: int = 1000,
    workspace_mib: int = 4096,
    optimization_level: int = 5,
) -> Path:
    if reference_language not in {"zh", "yue", "en", "ja", "ko"}:
        raise ConversionError(f"Unsupported reference language: {reference_language}")
    try:
        model_id = validate_safe_identifier(model_id, "model_id")
        voice_profile = validate_safe_identifier(voice_profile, "voice_profile")
    except PackageValidationError as error:
        raise ConversionError(str(error)) from error
    gpt = _require(gpt, "GPT checkpoint")
    sovits = _require(sovits, "SoVITS checkpoint")
    reference_audio = _require(reference_audio, "reference audio")
    reference_text_file = _require(reference_text_file, "reference text")
    source_dir = source_dir.resolve()
    shared_dir = shared_dir.resolve()
    inspection = inspect_pair(gpt, sovits, allow_unsafe_pickle=allow_unsafe_pickle)
    reference_text = reference_text_file.read_text(encoding="utf-8-sig").strip()
    if not reference_text:
        raise ConversionError("Reference text is empty")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        onnx_dir = staging / "onnx"
        onnx_hashes = export_onnx(
            gpt=gpt,
            sovits=sovits,
            shared_dir=shared_dir,
            source_dir=source_dir,
            output_dir=onnx_dir,
            max_len=max_len,
            allow_unsafe_pickle=allow_unsafe_pickle,
        )
        build_config = {
            "precision": "FP16",
            "workspace_mib": workspace_mib,
            "optimization_level": optimization_level,
            "tf32": False,
        }
        fingerprint, fingerprint_payload = engine_fingerprint(
            onnx_hashes=onnx_hashes,
            profiles=_jsonable_profiles(),
            build_config=build_config,
        )
        engines_dir = staging / "engines" / fingerprint
        paths = ModelPaths(
            gpt_checkpoint=gpt,
            sovits_checkpoint=sovits,
            reference_audio=reference_audio,
            onnx_dir=onnx_dir,
            engine_dir=engines_dir,
            tokenizer_dir=shared_dir / "chinese-roberta-wwm-ext-large",
        )
        build_engines(
            paths,
            workspace_mib=workspace_mib,
            optimization_level=optimization_level,
        )
        engine_manifest_path = engines_dir / "engine-manifest.json"
        engine_manifest = json.loads(engine_manifest_path.read_text(encoding="utf-8"))
        engine_manifest.update(
            {
                "kind": "aniflive-tts-gsv-v2proplus-tensorrt11-engines",
                "fingerprint": fingerprint,
                "runtime": {key: fingerprint_payload[key] for key in ENGINE_RUNTIME_KEYS},
                "fingerprint_payload": fingerprint_payload,
            }
        )
        engine_manifest_path.write_text(
            json.dumps(engine_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        profile_dir = resolve_contained_path(
            staging / "voices", voice_profile, "voice profile path"
        )
        profile_dir.mkdir(parents=True)
        shutil.copy2(reference_audio, profile_dir / "reference.wav")
        (profile_dir / "reference.txt").write_text(reference_text + "\n", encoding="utf-8")
        (profile_dir / "profile.json").write_text(
            json.dumps(
                {
                    "id": voice_profile,
                    "reference_audio": "reference.wav",
                    "reference_text": reference_text,
                    "reference_language": reference_language,
                    "conditioning_cache": "conditioning.pt",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        source_manifest = {
            "gpt": {"sha256": sha256_file(gpt), "filename": gpt.name},
            "sovits": {"sha256": sha256_file(sovits), "filename": sovits.name},
            "reference_audio": {"sha256": sha256_file(reference_audio), "filename": reference_audio.name},
            "inspection": inspection,
            "unsafe_pickle_allowed": allow_unsafe_pickle,
        }
        (staging / "source.json").write_text(
            json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
        write_checksums(staging)
        if output.exists():
            backup = output.parent / f"{output.name}.previous-{dt.datetime.now():%Y%m%dT%H%M%S}"
            os.replace(output, backup)
        os.replace(staging, output)
        return output
    except Exception:
        failed = output.parent / f"{output.name}.failed-{dt.datetime.now():%Y%m%dT%H%M%S}"
        if staging.exists():
            os.replace(staging, failed)
        raise
