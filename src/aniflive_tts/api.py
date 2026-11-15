from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .errors import EngineRebuildRequired
from .model_backend import V5TurboMacOSBackend, model_backend_for_manifest
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
    backend = model_backend_for_manifest(manifest)
    if backend is None:
        raise RuntimeError("Unsupported model family in model package")
    if manifest.get("precision") != backend.precision:
        raise RuntimeError("Model-package precision does not match the selected backend")
    model_id = validate_safe_identifier(manifest.get("model_id"), "model_id")
    validate_checksums(package_dir)
    if isinstance(backend, V5TurboMacOSBackend):
        requested = os.environ.get("ANIFLIVE_TTS_NATIVE_PRECISION", backend.precision)
        if requested != backend.precision:
            raise RuntimeError("Native precision setting differs from model-package precision")
        os.environ["ANIFLIVE_TTS_NATIVE_PRECISION"] = backend.precision
        return _configure_macos_runtime(settings, manifest, model_id)
    engine_dir = select_engine_dir(package_dir, manifest)
    if backend.model_family == "gsv-v5turbo":
        return _configure_v5(package_dir, manifest, model_id, engine_dir)
    dynamic_voices = manifest.get("reference_mode") == "dynamic"
    if dynamic_voices:
        if manifest.get("voice_profiles") != [] or manifest.get("default_voice_profile") is not None:
            raise RuntimeError("A shared-base package must have no fixed voice profiles")
        voice = ""
        profile = {}
        profile_dir = package_dir / "voices"
        profile_path = profile_dir / "profile.json"
    else:
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
    }
    if not dynamic_voices:
        required["reference"] = resolve_contained_path(
            profile_dir, profile.get("reference_audio"), "reference audio path"
        )
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} runtime resource: {path}")
    from .sampling_policy import package_sampling_contract

    values = {
        "ANIFLIVE_TTS_MODEL_FAMILY": backend.model_family,
        "ANIFLIVE_TTS_PACKAGE_SEMANTIC_SAMPLING": package_sampling_contract(manifest),
        "ANIFLIVE_TTS_MODEL_ID": model_id,
        "ANIFLIVE_TTS_VOICE_PROFILE": voice,
        "ANIFLIVE_TTS_SOURCE_DIR": str(source_dir),
        "ANIFLIVE_TTS_ENGINE_DIR": str(engine_dir),
        "ANIFLIVE_TTS_ONNX_DIR": str(package_dir / "onnx"),
        "ANIFLIVE_TTS_BERT_PATH": str(bert_path),
        "ANIFLIVE_TTS_REFERENCE_MODE": "dynamic" if dynamic_voices else "package",
        "ANIFLIVE_TTS_REFERENCE_WAV": "" if dynamic_voices else str(required["reference"]),
        "ANIFLIVE_TTS_REFERENCE_TEXT": str(profile.get("reference_text", "")),
        "ANIFLIVE_TTS_REFERENCE_LANGUAGE": str(profile.get("reference_language", "ja")),
        "ANIFLIVE_TTS_PROFILE_MANIFEST": str(profile_path),
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


def _configure_v5(package_dir: Path, manifest: dict[str, Any], model_id: str, engine_dir: Path) -> dict[str, Any]:
    if (manifest.get("reference_mode") != "dynamic" or manifest.get("voice_profiles") != []
            or manifest.get("default_voice_profile") is not None):
        raise RuntimeError("V5 Turbo packages require dynamic voice references")
    resources = manifest.get("resources")
    if not isinstance(resources, dict):
        raise RuntimeError("V5 Turbo package resources are missing")
    source = resolve_contained_path(package_dir, resources.get("source_dir"), "v5 source")
    models = resolve_contained_path(package_dir, resources.get("model_dir"), "v5 models")
    if models != source / "GPT_SoVITS" / "pretrained_models":
        raise RuntimeError("V5 model resources must use the upstream source layout")
    for path in (source / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py",
                 models / "s1v3.ckpt", models / "gsv-v5-pretrained" / "s2Gv5turbo.pth",
                 models / "gsv-v5-pretrained" / "vocoder.pth",
                 models / "chinese-roberta-wwm-ext-large" / "config.json",
                 models / "chinese-hubert-base" / "config.json",
                 models / "fast_langdetect" / "lid.176.bin",
                 source / "GPT_SoVITS" / "text" / "G2PWModel" / "g2pW.onnx",
                 source / "GPT_SoVITS" / "text" / "ja_userdic" / "userdict.csv",
                 source / "GPT_SoVITS" / "text" / "ja_userdic" / "user.dict",
                 source / "GPT_SoVITS" / "text" / "ja_userdic" / "userdict.md5",
                 source / "GPT_SoVITS" / "text" / "g2pw" / "polyphonic.pickle",
                 source / "GPT_SoVITS" / "text" / "g2pw" / "polyphonic.md5"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing V5 Turbo runtime resource: {path}")
    import hashlib
    text_dir = source / "GPT_SoVITS" / "text"
    def md5(path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()
    ja = text_dir / "ja_userdic"
    g2pw = text_dir / "g2pw"
    if md5(ja / "userdict.csv") != (ja / "userdict.md5").read_text(encoding="utf-8").strip():
        raise RuntimeError("V5 Japanese dictionary cache does not match its source")
    if (md5(g2pw / "polyphonic.rep") + md5(g2pw / "polyphonic-fix.rep")
            != (g2pw / "polyphonic.md5").read_text(encoding="utf-8").strip()):
        raise RuntimeError("V5 G2PW cache does not match its source")
    os.environ.update({"ANIFLIVE_TTS_MODEL_FAMILY": "gsv-v5turbo",
        "ANIFLIVE_TTS_MODEL_ID": model_id, "ANIFLIVE_TTS_VOICE_PROFILE": "",
        "ANIFLIVE_TTS_REFERENCE_MODE": "dynamic", "ANIFLIVE_TTS_REFERENCE_WAV": "",
        "ANIFLIVE_TTS_REFERENCE_TEXT": "", "ANIFLIVE_TTS_REFERENCE_LANGUAGE": "ja",
        "ANIFLIVE_TTS_V5_SOURCE_DIR": str(source), "ANIFLIVE_TTS_V5_MODEL_DIR": str(models),
        "ANIFLIVE_TTS_ENGINE_DIR": str(engine_dir), "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    return {"manifest": manifest, "profile": {}, "engine_dir": str(engine_dir)}


def _configure_macos_runtime(settings: RuntimeSettings, manifest: dict[str,Any], model_id:str)->dict[str,Any]:
    if sys.platform!="darwin":
        raise RuntimeError("The MLX/Metal model package requires native macOS")
    root=settings.model_package.resolve()
    native=manifest.get("native_runtime")
    if not isinstance(native,dict):
        raise RuntimeError("Native model package lacks native_runtime resources")
    dynamic=manifest.get("reference_mode")=="dynamic"
    keys=["worker_binary","models_dir","metal_library"]+([] if dynamic else ["reference_inputs"])
    resources={key:resolve_contained_path(root,native.get(key),"native "+key)for key in keys}
    for key,path in resources.items():
        if (key=="worker_binary" and not path.is_file())or(key!="worker_binary" and not path.is_dir()):
            raise FileNotFoundError("Missing native runtime resource: "+key)
    if dynamic:
        if manifest.get("voice_profiles")!=[] or manifest.get("default_voice_profile") is not None:
            raise RuntimeError("A shared-base package must have no fixed voice profiles")
        voice=""
        profile={"reference_text":"","reference_language":"ja"}
        profile_path=root/"voices"/"profile.json"
        reference=Path("")
    else:
        voice=validate_safe_identifier(os.environ.get("ANIFLIVE_TTS_VOICE_PROFILE",manifest.get("default_voice_profile")),"voice_profile")
        if manifest.get("reference_mode")!="package" or voice not in manifest.get("voice_profiles",[]):
            raise RuntimeError("Native package requires an explicit fixed voice profile")
        profile_dir=resolve_contained_path(root/"voices",voice,"voice profile path")
        profile_path=profile_dir/"profile.json"
        profile=json.loads(profile_path.read_text())
        if profile.get("id")!=voice:
            raise RuntimeError("Native voice profile identity mismatch")
        reference=resolve_contained_path(profile_dir,profile.get("reference_audio"),"reference audio")
        if not reference.is_file():
            raise FileNotFoundError("Native reference audio is missing")
    source=Path(os.environ.get("ANIFLIVE_TTS_SOURCE_DIR",str(settings.shared_dir/"official"))).expanduser().resolve()
    if not(source/"GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py").is_file():
        raise FileNotFoundError("Pinned official frontend source is unavailable")
    frontend=Path(os.environ.get("ANIFLIVE_TTS_NATIVE_FRONTEND_SCRIPT",str(Path(__file__).resolve().parents[2]/"scripts/macos-v5turbo/prepare_frontend.py"))).resolve()
    if not frontend.is_file():
        raise FileNotFoundError("Native frontend adapter script is unavailable")
    values={"ANIFLIVE_TTS_MODEL_FAMILY":"gsv-v5turbo","ANIFLIVE_TTS_MODEL_ID":model_id,"ANIFLIVE_TTS_VOICE_PROFILE":voice,"ANIFLIVE_TTS_SOURCE_DIR":str(source),"ANIFLIVE_TTS_ENGINE_DIR":str(root/"native"),"ANIFLIVE_TTS_PROFILE_MANIFEST":str(profile_path),"ANIFLIVE_TTS_REFERENCE_MODE":"dynamic" if dynamic else "package","ANIFLIVE_TTS_REFERENCE_WAV":"" if dynamic else str(reference),"ANIFLIVE_TTS_REFERENCE_TEXT":profile["reference_text"],"ANIFLIVE_TTS_REFERENCE_LANGUAGE":profile["reference_language"],"ANIFLIVE_TTS_NATIVE_WORKER":str(resources["worker_binary"]),"ANIFLIVE_TTS_NATIVE_MODEL_DIR":str(resources["models_dir"]),"ANIFLIVE_TTS_NATIVE_METAL_LIB":str(resources["metal_library"]),"ANIFLIVE_TTS_NATIVE_REFERENCE_INPUTS":"" if dynamic else str(resources["reference_inputs"]),"ANIFLIVE_TTS_NATIVE_FRONTEND_SCRIPT":str(frontend),"ANIFLIVE_TTS_NATIVE_CACHE_DIR":str(settings.cache_dir)}
    enable_mandarin = manifest.get("native_mandarin", False)
    if not isinstance(enable_mandarin, bool):
        raise RuntimeError("native_mandarin must be a Boolean")
    values["ANIFLIVE_TTS_NATIVE_ENABLE_MANDARIN"] = "1" if enable_mandarin else "0"
    values["ANIFLIVE_TTS_VOICE_DIR"]=os.environ.get("ANIFLIVE_TTS_VOICE_DIR",str(settings.cache_dir/"voices"))
    os.environ.update(values)
    return {"manifest":manifest,"profile":profile,"engine_dir":str(root/"native")}


def create_app():
    try:
        configure_runtime()
    except EngineRebuildRequired:
        raise
    from .service import app

    return app
