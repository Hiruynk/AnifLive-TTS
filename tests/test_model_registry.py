import json
from pathlib import Path
import pytest
from aniflive_tts.model_registry import select_startup_package

def package(root, name, qualified=False):
    p = root / name
    p.mkdir()
    (p / "manifest.json").write_text(json.dumps({
        "model_id": name, "model_family": "gsv-v2proplus",
        "format": "aniflive-tts-model-package", "precision": "FP16",
        **({"qualification": {"id": "reviewed"}} if qualified else {}),
    }))
    (p / "checksums.json").write_text("{}")
    return p

def test_named_model_first_start_prefers_qualified_and_preserves_explicit_choice(tmp_path):
    original = package(tmp_path, "a-original")
    qualified = package(tmp_path, "b-qualified", True)
    assert select_startup_package(tmp_path / "active") == qualified
    assert select_startup_package(original) == original
    with pytest.raises(FileNotFoundError):
        select_startup_package(tmp_path / "missing-explicit")

def test_first_start_ignores_staging_and_malformed_packages(tmp_path):
    package(tmp_path, ".aniflive-staging", True)
    p = package(tmp_path, "broken")
    (p / "manifest.json").write_text("[]")
    p = package(tmp_path, "wrong-format")
    (p / "manifest.json").write_text(json.dumps({"model_family": "gsv-v2proplus"}))
    with pytest.raises(FileNotFoundError):
        select_startup_package(tmp_path / "active")
