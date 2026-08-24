from __future__ import annotations

import json
from pathlib import Path

from aniflive_tts.model_package import validate_checksums, write_checksums
from scripts.promote_model_package import promote


def test_promote_copies_only_active_engine_and_rewrites_model_id(tmp_path: Path) -> None:
    source = tmp_path / "candidate"
    active = source / "engines" / "active123"
    inactive = source / "engines" / "inactive456"
    active.mkdir(parents=True)
    inactive.mkdir(parents=True)
    (active / "engine-manifest.json").write_text("{}", encoding="utf-8")
    (inactive / "engine-manifest.json").write_text("{}", encoding="utf-8")
    (source / "onnx").mkdir()
    (source / "onnx" / "model.onnx").write_bytes(b"onnx")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "format": "aniflive-tts-model-package",
                "model_id": "candidate",
                "active_engine_fingerprint": "active123",
            }
        ),
        encoding="utf-8",
    )
    write_checksums(source)

    output = promote(source, tmp_path / "release" / "voice", "voice-v2proplus")

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == "voice-v2proplus"
    assert (output / "engines" / "active123").is_dir()
    assert not (output / "engines" / "inactive456").exists()
    validate_checksums(output)
