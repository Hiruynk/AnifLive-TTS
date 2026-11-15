from __future__ import annotations

import json

import pytest

from aniflive_tts import model_package
from aniflive_tts.errors import EngineRebuildRequired
from aniflive_tts.model_backend import V5TURBO_BACKEND


FILES = ("ar_neural", "encoder", "dit", "dit_579", "vocoder", "attention_guard",
         "gemv_guard", "layernorm_plugin", "weight_l2", "joint_graph")
RUNTIME = {key: "1" for key in model_package.ENGINE_RUNTIME_KEYS}


def test_v5_package_contract():
    assert V5TURBO_BACKEND.supports_manifest({"format": "aniflive-tts-model-package",
        "model_family": "gsv-v5turbo", "precision": "FP16"})
    assert not V5TURBO_BACKEND.supports_manifest({"format": "aniflive-tts-model-package",
        "model_family": "gsv-v5turbo", "precision": "FP32"})


def test_v5_engine_inventory_is_contained_and_complete(tmp_path, monkeypatch):
    engine = tmp_path / "engines" / "abc"
    engine.mkdir(parents=True)
    files = {name: name + ".bin" for name in FILES}
    for path in files.values():
        (engine / path).write_bytes(b"engine")
    data = {"fingerprint": "abc", "runtime": RUNTIME,
            "kind": V5TURBO_BACKEND.engine_manifest_kind, "files": files}
    import hashlib
    digest = hashlib.sha256(b"engine").hexdigest()
    (engine / "report.json").write_text(json.dumps({
        "build": {"sha256": digest}, "projection_aot_weights": [
            {"index": i, "input_features": 2048, "output_features": 512, "native_sha256": digest}
            for i in range(3, 96, 4)],
        "requires_native_k8_pointer_alignment_bytes": 16,
        "native_k8_alignment_guard_version": 2,
        "requires_gemv_pointer_alignment_bytes": 16,
        "requires_fused_native_prefix_mask": True,
        "native_k8_alignment_guard_sha256": digest,
        "gemv_alignment_guard_sha256": digest,
    }))
    path = engine / "engine-manifest.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(model_package, "runtime_fingerprint", lambda: RUNTIME)
    manifest = {"active_engine_fingerprint": "abc", "model_family": "gsv-v5turbo"}
    assert model_package.select_engine_dir(tmp_path, manifest) == engine.resolve()
    data["files"]["dit"] = "../outside"
    path.write_text(json.dumps(data))
    with pytest.raises(EngineRebuildRequired):
        model_package.select_engine_dir(tmp_path, manifest)
    data["files"].pop("dit")
    path.write_text(json.dumps(data))
    with pytest.raises(EngineRebuildRequired):
        model_package.select_engine_dir(tmp_path, manifest)
