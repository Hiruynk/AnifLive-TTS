"""Assemble a local V5 Turbo shared-base package from source, assets and plans.

Copyright 2026 Hiruynk. Licensed under the repository LICENSE.
The target directory must not exist. No source, model, reference, or engine input
is modified. Built plans require target-specific execution and audio validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from aniflive_tts.model_backend import V5TURBO_BACKEND
from aniflive_tts.model_package import (ENGINE_RUNTIME_KEYS, canonical_hash, sha256_file,
                                       validate_safe_identifier, write_checksums)

SOURCE_LOCK = {
    "GPT_SoVITS/TTS_infer_pack/TTS.py": "1dc9b027987911c55f5bc8633ec2e1fab1895a8efefc7e0fcc89efa395d851b1",
    "GPT_SoVITS/TTS_infer_pack/streaming.py": "ef76a53462df53edc44d744d8700216d742ffeab683044b45ca99f9abc70cdf5",
    "GPT_SoVITS/f5_tts/model/modules.py": "d54f49402172959ee8a73f804735561d9bcbaf4d18c18f2f4138acf854dbccf1",
    "GPT_SoVITS/Accel/PyTorch/__init__.py": "2870d5422c49f61f2fbf112791774870bd26ccd3873d64052bb893efd3ef4f2d",
}
G2P_SOURCE = Path(__file__).resolve().parents[1] / "minimal_inference/GPT_SoVITS/text/g2p_en"
G2P_PROVENANCE = Path(__file__).resolve().parents[1] / "minimal_inference/ANIFLIVE_TTS_PROVENANCE.json"
G2P_FILES = {
    "__init__.py": "36ffae3e8f2cad9d285131c78963f690327f5046b5273dfecc52707659be949d",
    "checkpoint20.npz": "b8af35e4596d8dd5836dfd3fe9b2ba4f97b9c311efe8879544cbcfcbd566d8c6",
    "expand.py": "b31f9f2173280f4479de1272d57540382d64496662ef127cf4d05af5a49d70db",
    "g2p.py": "8960343d15ad72dc21525275417bca19ba6ab1d73b752e7b797ea16f8440490d",
    "homographs.en": "26cfafdf52f8585ff309b46f02ee6818e66ec165484c0dd3324dea7a2b14861e",
    "LICENSE.txt": "b40930bbcf80744c86c46a12bc9da056641d722716c378f5659b9e555ef833e1",
}
G2P_PROVENANCE_SHA256 = "52f317e50b81d5079a6d6e32e9c04c77df0d0358263107076d2062997d356d72"
WEIGHTS = {
    "s1v3.ckpt": "87133414860ea14ff6620c483a3db5ed07b44be42e2c3fcdad65523a729a745a",
    "gsv-v5-pretrained/s2Gv5turbo.pth": "287b32a06b7d486f5303ec520468b0728fb01aac8b0f5d86fd5455c9a1d3e022",
    "gsv-v5-pretrained/vocoder.pth": "b7543c83222aa2105bb4efb56845086bc5f2eeba61243d723ffa062052b3221f",
}


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def validate_inputs(source: Path, models: Path, g2pw_model: Path, ja: Path, g2pw: Path) -> None:
    for relative, digest in SOURCE_LOCK.items():
        if sha256_file(source / relative) != digest:
            raise ValueError(f"Upstream source does not match pinned commit: {relative}")
    if not (source / "LICENSE").is_file():
        raise ValueError("Upstream source license is missing")
    for relative, digest in WEIGHTS.items():
        if sha256_file(models / relative) != digest:
            raise ValueError(f"Model weights do not match the pinned checkpoint: {relative}")
    for path in (models / "chinese-roberta-wwm-ext-large/config.json",
                 models / "chinese-hubert-base/config.json",
                 models / "fast_langdetect/lid.176.bin", g2pw_model / "g2pW.onnx",
                 ja / "userdict.csv", ja / "user.dict", ja / "userdict.md5",
                 g2pw / "polyphonic.pickle", g2pw / "polyphonic.md5"):
        if not path.is_file():
            raise FileNotFoundError(f"Required V5 resource is missing: {path}")
    if file_md5(ja / "userdict.csv") != (ja / "userdict.md5").read_text(encoding="utf-8").strip():
        raise ValueError("Japanese dictionary cache does not match its source")
    dictionaries = source / "GPT_SoVITS/text/g2pw"
    expected = file_md5(dictionaries / "polyphonic.rep") + file_md5(dictionaries / "polyphonic-fix.rep")
    if expected != (g2pw / "polyphonic.md5").read_text(encoding="utf-8").strip():
        raise ValueError("G2PW cache does not match its source dictionaries")


WEIGHT_SUFFIXES = frozenset({".ckpt", ".pth", ".pt", ".bin", ".safetensors", ".onnx"})


def copy_model_file(source, destination, *, link_models: bool):
    source, destination = Path(source), Path(destination)
    if link_models and source.suffix.lower() in WEIGHT_SUFFIXES:
        try:
            os.link(source, destination)
        except OSError as error:
            raise OSError(f"Cannot hardlink immutable model file {source}; --link-models requires the same filesystem") from error
        return str(destination)
    return shutil.copy2(source, destination)

def validate_g2p_source(source: Path, provenance: Path) -> None:
    for name, expected in G2P_FILES.items():
        if sha256_file(source / name) != expected:
            raise ValueError(f"English G2P source does not match the pinned vendored version: {name}")
    if sha256_file(provenance) != G2P_PROVENANCE_SHA256:
        raise ValueError("English G2P provenance does not match the pinned vendored version")

def copy_g2p_source(source: Path, provenance: Path, target: Path) -> None:
    if target.exists():
        raise ValueError("Upstream source already contains a top-level g2p_en directory")
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(provenance, target / "ANIFLIVE_TTS_PROVENANCE.json")
    (target / "NOTICE.txt").write_text(
        "g2p-en 2.1.0, Apache-2.0; modified by AnifLive-TTS in 2026. "
        "See LICENSE.txt and ANIFLIVE_TTS_PROVENANCE.json.\n", encoding="utf-8")

def copy_source(source: Path, destination: Path) -> None:
    def ignore(directory, names):
        relative = Path(directory).relative_to(source).as_posix()
        result = {name for name in names if name in {".git", "__pycache__"} or name.endswith((".pyc", ".pyo"))}
        if relative == "GPT_SoVITS":
            result.add("pretrained_models")
        elif relative == "GPT_SoVITS/text":
            result.update({"G2PWModel", "ja_userdic"})
        elif relative == "GPT_SoVITS/text/g2pw":
            result.update({"polyphonic.pickle", "polyphonic.md5"})
        return result
    shutil.copytree(source, destination, ignore=ignore)


def assemble(args) -> Path:
    model_id = validate_safe_identifier(args.model_id, "model_id")
    paths = {name: getattr(args, name).resolve(strict=True) for name in
             ("source", "models", "g2pw_model", "ja_cache", "g2pw_cache", "ar", "encoder", "dit", "vocoder", "native", "runtime_fingerprint")}
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if any(output.is_relative_to(path) for path in paths.values() if path.is_dir()):
        raise ValueError("Output must be outside all input directories")
    validate_inputs(paths["source"], paths["models"], paths["g2pw_model"], paths["ja_cache"], paths["g2pw_cache"])
    g2p_source = args.g2p_en_source.resolve(strict=True)
    g2p_provenance = args.g2p_en_provenance.resolve(strict=True)
    if output.is_relative_to(g2p_source):
        raise ValueError("Output must be outside the English G2P source")
    validate_g2p_source(g2p_source, g2p_provenance)
    runtime = json.loads(paths["runtime_fingerprint"].read_text(encoding="utf-8"))
    if (not isinstance(runtime, dict) or any(not isinstance(runtime.get(k), str) or not runtime[k] for k in ENGINE_RUNTIME_KEYS)
            or runtime["tensorrt"] != "11.2.1.2" or runtime["cuda_runtime"] != "12.8" or runtime["platform_system"].lower() != "linux"):
        raise ValueError("Runtime fingerprint must describe the target Linux TensorRT 11.2/CUDA 12.8 runtime")
    native = json.loads((paths["native"] / "native-build.json").read_text(encoding="utf-8"))
    if native.get("status") != "built_cpu_only" or native.get("checkpoint_sha256") != WEIGHTS["s1v3.ckpt"]:
        raise ValueError("Native build metadata does not match the pinned AR checkpoint")
    report = json.loads((paths["ar"] / "report.json").read_text(encoding="utf-8"))
    artifacts = {"ar_neural": paths["ar"] / "ar_neural.plan", "encoder": paths["encoder"] / "encoder.plan",
                 "dit": paths["dit"] / "dit.plan", "dit_579": paths["dit"] / "dit_579.plan",
                 "vocoder": paths["vocoder"] / "vocoder.plan"}
    for key in ("attention_guard", "gemv_guard", "layernorm_plugin", "weight_l2", "joint_graph"):
        info = native["files"][key]
        if Path(info["file"]).name != info["file"]:
            raise ValueError("Invalid native helper file name")
        artifacts[key] = paths["native"] / info["file"]
        if sha256_file(artifacts[key]) != info["sha256"]:
            raise ValueError(f"Native helper checksum mismatch: {key}")
    hashes = {key: sha256_file(path) for key, path in artifacts.items()}
    if (report.get("build", {}).get("sha256") != hashes["ar_neural"]
            or report.get("native_k8_alignment_guard_sha256") != hashes["attention_guard"]
            or report.get("gemv_alignment_guard_sha256") != hashes["gemv_guard"]):
        raise ValueError("AR plan and guard metadata do not match the build outputs")
    fingerprint = canonical_hash({"runtime": runtime, "files": hashes,
        "source_commit": "04a44eb977586645f33594b72af13a035a2056c6", "sample_steps": 4, "cfg_rate": 0.0})[:24]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v5-package-", dir=output.parent) as temp:
        package = Path(temp) / "package"
        package.mkdir()
        source = package / "source"
        copy_source(paths["source"], source)
        copy_g2p_source(g2p_source, g2p_provenance, source / "g2p_en")
        model_dir = source / "GPT_SoVITS/pretrained_models"
        model_dir.mkdir(parents=True)
        from functools import partial
        copy_model = partial(copy_model_file, link_models=args.link_models)
        copy_model(paths["models"] / "s1v3.ckpt", model_dir / "s1v3.ckpt")
        v5_models = model_dir / "gsv-v5-pretrained"
        v5_models.mkdir()
        for name in ("s2Gv5turbo.pth", "vocoder.pth"):
            copy_model(paths["models"] / "gsv-v5-pretrained" / name, v5_models / name)
        for name in ("chinese-roberta-wwm-ext-large", "chinese-hubert-base", "fast_langdetect", "sv"):
            shutil.copytree(paths["models"] / name, model_dir / name, copy_function=copy_model)
        text = source / "GPT_SoVITS/text"
        shutil.copytree(paths["g2pw_model"], text / "G2PWModel", copy_function=copy_model,
                        ignore=shutil.ignore_patterns("*.log"))
        shutil.copytree(paths["ja_cache"], text / "ja_userdic")
        for name in ("polyphonic.pickle", "polyphonic.md5"):
            shutil.copy2(paths["g2pw_cache"] / name, text / "g2pw" / name)
        engine_dir = package / "engines" / fingerprint
        engine_dir.mkdir(parents=True)
        file_paths = {"ar_neural": "ar/ar_neural.plan", "encoder": "encoder/encoder.plan",
                      "dit": "dit/dit.plan", "dit_579": "dit/dit_579.plan", "vocoder": "vocoder/vocoder.plan",
                      "attention_guard": "plugins/attention_guard.so", "gemv_guard": "plugins/gemv_guard.so",
                      "layernorm_plugin": "plugins/layernorm_plugin.so", "weight_l2": "plugins/weight_l2.so",
                      "joint_graph": "plugins/joint_graph.so"}
        for key, relative in file_paths.items():
            target = engine_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifacts[key], target)
            if sha256_file(target) != hashes[key]:
                raise RuntimeError(f"Engine resource copy mismatch: {key}")
        (engine_dir / "ar/report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        engine_manifest = {"schema": 1, "kind": V5TURBO_BACKEND.engine_manifest_kind,
            "fingerprint": fingerprint, "runtime": runtime, "precision": "FP16", "sample_steps": 4,
            "cfg_rate": 0.0, "files": file_paths, "qualification": "unqualified"}
        (engine_dir / "engine-manifest.json").write_text(json.dumps(engine_manifest, indent=2) + "\n", encoding="utf-8")
        manifest = {"schema": 1, **V5TURBO_BACKEND.manifest_fields(), "model_id": model_id,
            "active_engine_fingerprint": fingerprint, "reference_mode": "dynamic", "voice_profiles": [],
            "default_voice_profile": None, "languages": ["zh", "yue", "en", "ja", "ko"],
            "resources": {"source_dir": "source", "model_dir": "source/GPT_SoVITS/pretrained_models"}}
        (package / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        source_files = {}
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source).as_posix()
            if (relative.startswith("GPT_SoVITS/pretrained_models/") or
                relative.startswith("GPT_SoVITS/text/G2PWModel/") or
                relative.startswith("GPT_SoVITS/text/ja_userdic/") or
                relative in {"GPT_SoVITS/text/g2pw/polyphonic.pickle", "GPT_SoVITS/text/g2pw/polyphonic.md5"}):
                continue
            source_files[relative] = sha256_file(path)
        source_lock = {"upstream_commit": "04a44eb977586645f33594b72af13a035a2056c6",
                       "verified_key_files": SOURCE_LOCK, "files": source_files,
                       "augmentations": {"g2p_en": {"version": "2.1.0", "license": "Apache-2.0",
                           "source": "minimal_inference/GPT_SoVITS/text/g2p_en", "files": G2P_FILES,
                           "provenance_sha256": G2P_PROVENANCE_SHA256}}}
        (package / "source-lock.json").write_text(json.dumps(source_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_checksums(package)
        os.replace(package, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "models", "g2pw-model", "ja-cache", "g2pw-cache", "ar", "encoder", "dit", "vocoder", "native", "runtime-fingerprint", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--g2p-en-source", type=Path, default=G2P_SOURCE,
                        help="Pinned Apache English G2P source directory")
    parser.add_argument("--g2p-en-provenance", type=Path, default=G2P_PROVENANCE,
                        help="Vendored source provenance JSON")
    parser.add_argument("--link-models", action="store_true", help="Hardlink immutable model weights; inputs and output must share a filesystem")
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    print(assemble(args))


if __name__ == "__main__":
    main()
