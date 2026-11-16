#!/usr/bin/env python3
"""Exercise real 4.57.6 vulnerable and backported paths without network access."""

import importlib.metadata
import importlib.util
import shutil
import sys
from pathlib import Path
from types import MethodType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "v5_transformers_security.py"
_SPEC = importlib.util.spec_from_file_location("v5_transformers_security", _SCRIPT)
patcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(patcher)


@pytest.fixture
def package_copy(tmp_path):
    dist = importlib.metadata.distribution("transformers")
    assert dist.version == patcher.VERSION
    source = Path(dist.locate_file("transformers"))
    root = tmp_path / "transformers"
    for path in patcher.PACKAGE_FILES.values():
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / path, dest)
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
    with pytest.raises(RuntimeError, match="Unexpected Transformers source hashes"):
        patcher.patch_package(package_copy, "4.57.6")
    target.write_bytes(original)
    patcher.patch_package(package_copy, "4.57.6")
    assert patcher.patch_package(package_copy, "4.57.6")["status"] == "already_patched"
    target.write_bytes(original)
    with pytest.raises(RuntimeError, match="Unexpected Transformers source hashes"):
        patcher.patch_package(package_copy, "4.57.6")


def test_internal_attention_fields_cannot_override_public_choice(package_copy, monkeypatch):
    path = package_copy / "configuration_utils.py"
    name = "transformers.configuration_v5_security_test"
    payload = {"_attn_implementation_internal": "attacker/payload",
               "_experts_implementation_internal": "attacker/experts"}
    unsafe = load_file(path, name, monkeypatch)
    assert unsafe.PretrainedConfig.from_dict(payload)._attn_implementation == "attacker/payload"
    patcher.patch_package(package_copy, patcher.VERSION)
    safe = load_file(path, name, monkeypatch)
    assert safe.PretrainedConfig.from_dict(payload)._attn_implementation is None
    assert not hasattr(safe.PretrainedConfig.from_dict(payload), "_experts_implementation_internal")
    for attention in ("eager", "sdpa"):
        assert safe.PretrainedConfig.from_dict(payload, attn_implementation=attention)._attn_implementation == attention


def test_kernel_loader_rejects_untrusted_repositories(package_copy, monkeypatch):
    path = package_copy / "integrations/hub_kernels.py"
    name = "transformers.integrations.hub_kernels_v5_security_test"
    unsafe = load_file(path, name, monkeypatch)
    assert unsafe.is_kernel("attacker/payload")
    patcher.patch_package(package_copy, patcher.VERSION)
    safe = load_file(path, name, monkeypatch)
    calls = []
    monkeypatch.setattr(safe, "_kernels_available", True)
    monkeypatch.setattr(safe, "get_kernel", lambda *a, **kw: calls.append((a, kw)), raising=False)
    for value in (None, "eager", "sdpa", "attacker/payload", "paged|attacker/payload@rev:fn"):
        assert not safe.is_kernel(value)
        safe.load_and_register_kernel(value)
    assert calls == []
    for value in ("kernels-community/flash-attn", "kernels-community/flash-attn@rev:fn",
                  "paged|kernels-community/flash-attn"):
        assert safe.is_kernel(value)


@pytest.mark.parametrize("kind", ["tokenizer", "processor"])
def test_save_pretrained_rejects_template_path_traversal(package_copy, monkeypatch, tmp_path, kind):
    import transformers
    from transformers import BertTokenizer, BlipImageProcessor, BlipProcessor

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\n", encoding="utf-8")
    tokenizer = BertTokenizer(vocab_file=str(vocab))
    if kind == "tokenizer":
        obj = tokenizer
        filename = "tokenization_utils_base.py"
        classname = "PreTrainedTokenizerBase"
    else:
        obj = BlipProcessor(image_processor=BlipImageProcessor(), tokenizer=tokenizer)
        filename = "processing_utils.py"
        classname = "ProcessorMixin"
    path = package_copy / filename
    module_name = f"transformers.{kind}_v5_security_test"
    # The processor source resolves classes through its package directory;
    # retain the installed real package when loading the copied source file.
    monkeypatch.setattr(transformers.utils, "direct_transformers_import", lambda _: transformers)
    unsafe = load_file(path, module_name, monkeypatch)
    obj.chat_template = {"../../escaped": "payload"}
    getattr(unsafe, classname).save_pretrained(obj, tmp_path / "unsafe")
    escaped = tmp_path / "escaped.jinja"
    assert escaped.read_text() == "payload"
    escaped.unlink()

    patcher.patch_package(package_copy, patcher.VERSION)
    safe = load_file(path, module_name, monkeypatch)
    if kind == "tokenizer":
        monkeypatch.setattr(obj, "save_chat_templates", MethodType(safe.PreTrainedTokenizerBase.save_chat_templates, obj))
    save = getattr(safe, classname).save_pretrained
    for key in ("../../escaped", str(tmp_path / "absolute")):
        obj.chat_template = {key: "payload"}
        with pytest.raises(ValueError, match="Invalid chat template name"):
            save(obj, tmp_path / "safe")
    assert not escaped.exists()
    assert not (tmp_path / "absolute.jinja").exists()
    obj.chat_template = {"default": "default text", "tool_use": "tool text"}
    save(obj, tmp_path / "safe")
    assert (tmp_path / "safe/chat_template.jinja").read_text() == "default text"
    assert (tmp_path / "safe" / safe.CHAT_TEMPLATE_DIR / "tool_use.jinja").read_text() == "tool text"


def test_verify_only_requires_all_exact_fixed_files(package_copy):
    with pytest.raises(RuntimeError, match="security source verification failed"):
        patcher.verify_package(package_copy, patcher.VERSION)
    patcher.patch_package(package_copy, patcher.VERSION)
    before = {path: (package_copy / path).read_bytes() for path in patcher.PACKAGE_FILES.values()}
    result = patcher.verify_package(package_copy, patcher.VERSION)
    assert result["status"] == "verified"
    assert result["sha256"] == patcher.POST_HASHES
    assert set(result["upstream_fixes"]) == {"CVE-2026-4372", "CVE-2026-5241", "CVE-2026-9856"}
    assert before == {path: (package_copy / path).read_bytes() for path in before}
    with pytest.raises(RuntimeError, match="Expected transformers"):
        patcher.verify_package(package_copy, "5.0.0")
    for path, data in before.items():
        (package_copy / path).write_bytes(data + b"\n# tampered\n")
        with pytest.raises(RuntimeError, match="security source verification failed"):
            patcher.verify_package(package_copy, patcher.VERSION)
        (package_copy / path).write_bytes(data)
