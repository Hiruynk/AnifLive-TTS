from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .errors import EngineRebuildRequired
from .model_package import (
    resolve_contained_path,
    select_engine_dir,
    validate_checksums,
    validate_safe_identifier,
)
from .settings import RuntimeSettings


def configure_runtime(settings: RuntimeSettings | None = None) -> dict[str, Any]:
    settings = settings or RuntimeSettings.from_env()
    package_dir = settings.model_package.resolve()
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model package is missing manifest.json: {package_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "aniflive-tts-model-package":
        raise RuntimeError("Unsupported model package format")
    if manifest.get("model_family") != "gsv-v2proplus":
        raise RuntimeError("AnifLive-TTS v1 currently supports gsv-v2proplus")
    if manifest.get("precision") != "FP16":
        raise RuntimeError("AnifLive-TTS v1 currently supports FP16 packages")
    model_id = validate_safe_identifier(manifest.get("model_id"), "model_id")
    validate_checksums(package_dir)
    engine_dir = select_engine_dir(package_dir, manifest)
    voice = validate_safe_identifier(
        os.environ.get(
            "ANIFLIVE_TTS_VOICE_PROFILE", str(manifest.get("default_voice_profile", "default"))
        ),
        "voice_profile",
    )
    voice_profiles = manifest.get("voice_profiles")
    if not isinstance(voice_profiles, list) or voice not in voice_profiles:
        raise RuntimeError(f"Unknown voice profile: {voice}")
    profile_dir = resolve_contained_path(package_dir / "voices", voice, "voice profile path")
    profile_path = resolve_contained_path(
        profile_dir, "profile.json", "voice profile manifest path"
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise RuntimeError("Voice profile must contain a JSON object")
    if profile.get("id") != voice:
        raise RuntimeError("Voice profile id does not match the selected voice_profile")
    source_dir = Path(os.environ.get("ANIFLIVE_TTS_SOURCE_DIR", "/app/minimal_inference")).resolve()
    bert_path = settings.shared_dir / "chinese-roberta-wwm-ext-large"
    required = {
        "source": source_dir / "run_trt_inference.py",
        "BERT": bert_path / "config.json",
        "reference": resolve_contained_path(
            profile_dir, profile.get("reference_audio"), "reference audio path"
        ),
    }
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} runtime resource: {path}")
    values = {
        "ANIFLIVE_TTS_MODEL_ID": model_id,
        "ANIFLIVE_TTS_VOICE_PROFILE": voice,
        "ANIFLIVE_TTS_SOURCE_DIR": str(source_dir),
        "ANIFLIVE_TTS_ENGINE_DIR": str(engine_dir),
        "ANIFLIVE_TTS_ONNX_DIR": str(package_dir / "onnx"),
        "ANIFLIVE_TTS_BERT_PATH": str(bert_path),
        "ANIFLIVE_TTS_REFERENCE_WAV": str(required["reference"]),
        "ANIFLIVE_TTS_REFERENCE_TEXT": str(profile["reference_text"]),
        "ANIFLIVE_TTS_REFERENCE_LANGUAGE": str(profile["reference_language"]),
        "ANIFLIVE_TTS_JA_USERDIC_DIR": str(source_dir / "GPT_SoVITS" / "text" / "ja_userdic"),
        "NLTK_DATA": os.environ.get(
            "ANIFLIVE_TTS_NLTK_DATA",
            str(settings.shared_dir / "nltk_data")
            if (settings.shared_dir / "nltk_data").is_dir()
            else "/opt/aniflive-tts/nltk_data",
        ),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    configured_fasttext = os.environ.get("ANIFLIVE_TTS_FAST_LANGDETECT_MODEL")
    shared_fasttext = settings.shared_dir / "fast_langdetect" / "lid.176.bin"
    bundled_fasttext = source_dir.parent / "pretrained_models" / "fast_langdetect" / "lid.176.bin"
    if configured_fasttext:
        selected_fasttext = Path(configured_fasttext).expanduser().resolve()
    elif shared_fasttext.is_file():
        selected_fasttext = shared_fasttext.resolve()
    elif bundled_fasttext.is_file():
        selected_fasttext = bundled_fasttext.resolve()
    else:
        raise FileNotFoundError(
            "Missing pinned fastText language model. Run setup_shared_assets.py "
            "before starting the API; runtime downloads are disabled."
        )
    if not selected_fasttext.is_file():
        raise FileNotFoundError(
            f"Configured fastText language model does not exist: {selected_fasttext}"
        )
    os.environ["ANIFLIVE_TTS_FAST_LANGDETECT_MODEL"] = str(selected_fasttext)
    os.environ["ANIFLIVE_TTS_FAST_LANGDETECT_CACHE"] = str(selected_fasttext.parent)
    os.environ.update(values)
    return {"manifest": manifest, "profile": profile, "engine_dir": str(engine_dir)}


def create_app():
    try:
        configure_runtime()
    except EngineRebuildRequired:
        raise
    from .service import app

    return app
