"""Exercise the real 4.57.6 LightGlue constructors without network access."""

import importlib.metadata
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "v5_transformers_security.py"
_SPEC = importlib.util.spec_from_file_location("v5_transformers_security", _SCRIPT)
patcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(patcher)


@pytest.fixture
def package_copy(tmp_path):
    dist = importlib.metadata.distribution("transformers")
    assert dist.version == patcher.VERSION
    source = Path(dist.locate_file("transformers/models/lightglue"))
    root = tmp_path / "transformers"
    dest = root / "models" / "lightglue"
    dest.mkdir(parents=True)
    for name in patcher.PRE_HASHES:
        shutil.copyfile(source / name, dest / name)
    return root


def load_file(path, name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_configuration_never_loads_nested_remote_code(package_copy, monkeypatch):
    from transformers.models.auto import AutoConfig

    calls = []
    class Blocked(Exception):
        pass
    def intercepted(*args, **kwargs):
        calls.append((args, kwargs))
        raise Blocked("network and payload blocked")
    monkeypatch.setattr(AutoConfig, "from_pretrained", intercepted)
    config_file = package_copy / "models/lightglue/configuration_lightglue.py"
    module_name = "transformers.models.lightglue.configuration_v5_security_test"
    unsafe = load_file(config_file, module_name, monkeypatch)
    payload = {"trust_remote_code": True, "keypoint_detector_config": {
        "model_type": "unregistered_detector", "_name_or_path": "never-fetch"}}

    with pytest.raises(Blocked):
        unsafe.LightGlueConfig.from_dict(payload)
    assert calls[0][1]["trust_remote_code"] is True
    calls.clear()

    result = patcher.patch_package(package_copy, "4.57.6")
    assert result["status"] == "patched"
    safe = load_file(config_file, module_name, monkeypatch)
    with pytest.raises(KeyError, match="unregistered_detector"):
        safe.LightGlueConfig.from_dict(payload)
    assert calls == []
    # The supported built-in detector continues to construct successfully,
    # even if an older serialized config contains the trust flag.
    known = safe.LightGlueConfig.from_dict({"trust_remote_code": True,
                                            "keypoint_detector_config": {"model_type": "superpoint"}})
    assert known.keypoint_detector_config.model_type == "superpoint"


def test_model_constructor_does_not_forward_serialized_trust(package_copy, monkeypatch):
    from transformers.models.auto import AutoModelForKeypointDetection

    patcher.patch_package(package_copy, "4.57.6")
    base = package_copy / "models/lightglue"
    config_name = "transformers.models.lightglue.configuration_lightglue"
    config = load_file(base / "configuration_lightglue.py", config_name, monkeypatch)
    modeling = load_file(base / "modeling_lightglue.py",
                         "transformers.models.lightglue.modeling_v5_security_test", monkeypatch)
    calls = []
    class Stopped(Exception):
        pass
    def intercepted(*args, **kwargs):
        calls.append((args, kwargs))
        raise Stopped("model creation stopped before inference")
    monkeypatch.setattr(AutoModelForKeypointDetection, "from_config", intercepted)
    value = config.LightGlueConfig.from_dict({"trust_remote_code": True})
    with pytest.raises(Stopped):
        modeling.LightGlueForKeypointMatching(value)
    assert len(calls) == 1 and calls[0][1] == {}


def test_rejects_unexpected_version_and_modified_or_partial_sources(package_copy):
    target = package_copy / "models/lightglue/configuration_lightglue.py"
    original = target.read_bytes()
    with pytest.raises(RuntimeError, match="Expected transformers"):
        patcher.patch_package(package_copy, "5.0.0")
    target.write_bytes(original + b"\n# modified\n")
    with pytest.raises(RuntimeError, match="Unexpected Transformers LightGlue source hashes"):
        patcher.patch_package(package_copy, "4.57.6")
    target.write_bytes(original)
    patcher.patch_package(package_copy, "4.57.6")
    assert patcher.patch_package(package_copy, "4.57.6")["status"] == "already_patched"
    target.write_bytes(original)
    with pytest.raises(RuntimeError, match="Unexpected Transformers LightGlue source hashes"):
        patcher.patch_package(package_copy, "4.57.6")
