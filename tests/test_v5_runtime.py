from dataclasses import replace
from pathlib import Path

import pytest
import threading

from aniflive_tts import service
from aniflive_tts.backend.v5.runtime import V5Service
from aniflive_tts.expression import (
    ConditioningPolicy, ExpressionCatalog, PreparedReference, PreparedReferenceBank,
    ReferenceDescriptor,
)
from aniflive_tts.voice_library import VoiceBinding


def test_cancelled_complete_request_never_returns_partial_wav(monkeypatch):
    runtime=V5Service()
    stopped=threading.Event()
    runtime._stop=stopped
    def pieces():
        yield b'\x00\x00'*480
        stopped.set()
    def release():runtime._stop=None
    monkeypatch.setattr(runtime,'prepare_stream',lambda *args,**kwargs:(
        ['first','second'],service._OwnedPCMIterator(pieces(),release),48000,'v5'))
    with pytest.raises(service.SynthesisCancelled,match='cancelled'):
        runtime.synthesize_request(None,None)


def test_v5_options_and_capabilities_use_active_family(monkeypatch):
    manager = service.RuntimeServiceManager(V5Service())
    monkeypatch.setattr(service, "SERVICE", manager)
    monkeypatch.setattr(service, "VOICE_ID", "")
    values = {"text": "Hello", "language": "en", "voice_profile": "voice"}
    options = service._options_from_values(values, text_field="text", language_field="language")
    assert options.expression_policy is ConditioningPolicy.FULL_SWITCH
    with pytest.raises(service.RequestError, match="sample_steps must remain 4"):
        service._options_from_values({**values, "sample_steps": 32}, text_field="text", language_field="language")
    with pytest.raises(service.RequestError, match="noise_scale"):
        service._options_from_values({**values, "noise_scale": 0.7}, text_field="text", language_field="language")


def test_v5_reference_selection_by_id_or_discrete_emotion():
    bank = PreparedReferenceBank(identity_id="base")
    for identifier in ("base", "low", "high"):
        bank.add(PreparedReference(None, object(), object(), [],
            {"prompt": {"owner": identifier}, "reference": tuple(f"{identifier}:{i}" for i in range(7)),
             "path": identifier, "text": "text", "language": "en"}, id=identifier))
    descriptors = tuple(ReferenceDescriptor(id=identifier, emotion="happy", intensity=intensity,
        reference_audio=Path("ref.wav"), reference_text="text", reference_language="en",
        manual_verified=True) for identifier, intensity in (("low", 0.2), ("high", 0.8)))
    binding = VoiceBinding("voice", bank, ExpressionCatalog("low", descriptors), None)
    options = service.SynthesisOptions("Hello", "all_en", 15, 1, 1, 1, 0.3, 0.5, "", 1234,
        expression_enabled=True, expression_profile="high", expression_intensity=0.2,
        expression_policy=ConditioningPolicy.FULL_SWITCH, voice_binding=binding)
    runtime = V5Service()
    assert runtime.validate_expression(options)[0][1].id == "high"
    assert runtime.validate_expression(replace(options, expression_profile="happy", expression_intensity=0.2))[0][1].id == "low"
    styled = runtime.validate_expression(replace(options,
        expression_policy=ConditioningPolicy.SEMANTIC_STYLE))[0][1]
    assert styled.bert['_conditioning_components'] == {
        'semantic': 'high', 'style': 'base', 'voice_anchor': 'base', 'policy': 'semantic-style'}


def test_v5_structured_request_preserves_sampler_validation(monkeypatch):
    manager = service.RuntimeServiceManager(V5Service())
    monkeypatch.setattr(service, "SERVICE", manager)
    monkeypatch.setattr(service, "VOICE_ID", "")
    monkeypatch.setattr(service.VOICE_LIBRARY.store, "get_voice", lambda _: {})
    values = service._structured_speech_values({"text": "Hi", "language": "en",
        "voice_profile": "voice", "generation": {"sample_steps": 32}})
    with pytest.raises(service.RequestError, match="sample_steps must remain 4"):
        service._options_from_values(values, text_field="text", language_field="language")


def test_source_banks_isolate_overlapping_modules(monkeypatch, tmp_path):
    import sys
    from types import ModuleType
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_module, second_module = ModuleType("v5_source_test"), ModuleType("v5_source_test")
    first_module.__file__ = str(first / "module.py")
    second_module.__file__ = str(second / "module.py")
    manager = service.RuntimeServiceManager(V5Service())
    monkeypatch.setitem(sys.modules, "v5_source_test", first_module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    manager._save_source_bank(first)
    assert "v5_source_test" not in sys.modules
    sys.modules["v5_source_test"] = second_module
    manager._save_source_bank(second)
    manager._restore_source_bank(first)
    assert sys.modules["v5_source_test"] is first_module


def test_v5_cancel_does_not_wait_for_active_synthesis_lock():
    import threading
    runtime = V5Service()
    runtime._owner = "session"
    runtime._stop = threading.Event()
    manager = service.RuntimeServiceManager(runtime)
    held, release = threading.Event(), threading.Event()
    def hold_lock():
        with manager._switch_lock:
            held.set()
            release.wait(2)
    worker = threading.Thread(target=hold_lock)
    worker.start()
    assert held.wait(1)
    try:
        assert manager.cancel_stream("session")
        assert runtime._stop.is_set()
    finally:
        release.set()
        worker.join()


def test_v5_repeated_interop_setting_must_match_existing(monkeypatch):
    from types import SimpleNamespace
    from aniflive_tts.backend.v5.engine import _compatible_interop_setting
    def already_set(_):
        raise RuntimeError("Error: cannot set number of interop threads after parallel work has started")
    torch = SimpleNamespace(set_num_interop_threads=already_set, get_num_interop_threads=lambda: 4)
    with _compatible_interop_setting(torch):
        torch.set_num_interop_threads(4)
        with pytest.raises(RuntimeError, match="cannot set number of interop threads"):
            torch.set_num_interop_threads(2)
    assert torch.set_num_interop_threads is already_set


def test_source_bank_removes_partial_namespace_after_failed_import(monkeypatch, tmp_path):
    import sys
    from importlib._bootstrap_external import _NamespacePath
    from types import ModuleType
    root = tmp_path / "source"
    root.mkdir()
    parent = ModuleType("v5_orphan.parent")
    parent.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    module = ModuleType("v5_orphan.parent.child")
    module.__path__ = _NamespacePath("v5_orphan.parent.child", [str(root / "child")], None)
    monkeypatch.delitem(sys.modules, parent.__name__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    manager = service.RuntimeServiceManager(V5Service())
    manager._save_source_bank(root)
    assert module.__name__ not in sys.modules
    assert manager._source_banks[str(root)][module.__name__] is module


def test_v5_plugin_reuses_only_identical_loaded_library(tmp_path, monkeypatch):
    import hashlib
    import sys
    from types import SimpleNamespace
    from aniflive_tts.backend.v5 import trt_cpp_native_layernorm_direct as plugin
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.10.0+cu128"))
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__="11.2.1.2"))
    first, second, other = (tmp_path / name for name in ("first.so", "second.so", "other.so"))
    first.write_bytes(b"same library")
    second.write_bytes(b"same library")
    other.write_bytes(b"different library")
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    monkeypatch.setattr(plugin, "_loaded", {str(first.resolve()): (digest, object())})
    assert plugin.load(second) == str(second.resolve())
    with pytest.raises(RuntimeError, match="different"):
        plugin.load(other)
    first.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        plugin.load(first)


@pytest.mark.parametrize("torch_version,trt_version", [
    ("2.10.0+cpu", "11.2.1.2"),
    ("2.10.0+cu128", "11.0.0"),
])
def test_v5_plugin_rejects_mismatched_abi(tmp_path, monkeypatch, torch_version, trt_version):
    import sys
    from types import SimpleNamespace
    from aniflive_tts.backend.v5 import trt_cpp_native_layernorm_direct as plugin

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__=torch_version))
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__=trt_version))
    with pytest.raises(RuntimeError, match="matching TensorRT/PyTorch ABI"):
        plugin.load(tmp_path / "not-loaded.so")


def test_v5_unload_releases_target_encoder_without_graph():
    from types import SimpleNamespace
    from aniflive_tts.backend.v5.engine import shared_gpu_lease
    class Handle:
        def __enter__(self): return self
        def __exit__(self, *_): return None
    class Lease:
        def acquire(self, _): return Handle()
    class Encoder:
        def __init__(self): self.disabled = False
        def disable(self): self.disabled = True
    adapter = Encoder()
    engine = SimpleNamespace(performance_encoder=adapter, prompt_cache={})
    with shared_gpu_lease(Lease()) as (_, engines):
        engines.append(engine)
    assert adapter.disabled
    assert engine.performance_encoder is None


def test_source_bank_restores_backend_environment_and_paths(monkeypatch, tmp_path):
    import os
    import sys
    root = tmp_path / "source"
    root.mkdir()
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("ANIFLIVE_TTS_BERT_PATH", "v5-assets")
    monkeypatch.setenv("UNRELATED_V5_TEST", "keep")
    manager = service.RuntimeServiceManager(V5Service())
    original_path = list(sys.path)
    manager._save_source_bank(root)
    monkeypatch.setenv("ANIFLIVE_TTS_BERT_PATH", "v2-assets")
    monkeypatch.setenv("ANIFLIVE_TTS_ONNX_DIR", "v2-only")
    sys.path.insert(0, str(tmp_path / "foreign"))
    manager._restore_source_bank(root)
    assert os.environ["ANIFLIVE_TTS_BERT_PATH"] == "v5-assets"
    assert "ANIFLIVE_TTS_ONNX_DIR" not in os.environ
    assert os.environ["UNRELATED_V5_TEST"] == "keep"
    assert sys.path == original_path


def test_v5_switch_applies_target_package_after_restoring_source_environment(monkeypatch, tmp_path):
    import json
    import os
    import sys
    from aniflive_tts import api
    source = tmp_path / "target" / "source"
    source.mkdir(parents=True)
    package = source.parent
    (package / "manifest.json").write_text(json.dumps({"model_family": "gsv-v5turbo", "resources": {"source_dir": "source"}}))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("ANIFLIVE_TTS_MODEL_PACKAGE", "/old/model")
    monkeypatch.setenv("ANIFLIVE_TTS_V5_SOURCE_DIR", "/old/source")
    manager = service.RuntimeServiceManager(V5Service())
    manager._save_source_bank(source)
    seen = {}
    def configure(settings):
        seen["package"] = str(settings.model_package)
        monkeypatch.setenv("ANIFLIVE_TTS_MODEL_ID", "target")
        monkeypatch.setenv("ANIFLIVE_TTS_VOICE_PROFILE", "")
        monkeypatch.setenv("ANIFLIVE_TTS_V5_SOURCE_DIR", str(source))
        return {"manifest": {"model_family": "gsv-v5turbo"}}
    monkeypatch.setattr(api, "configure_runtime", configure)
    monkeypatch.setattr(service, "_sync_runtime_identity_from_env", lambda: None)
    monkeypatch.setattr(V5Service, "load", lambda self: None)
    manager._load_package(package, "")
    assert seen["package"] == str(package)
    assert os.environ["ANIFLIVE_TTS_V5_SOURCE_DIR"] == str(source)


def test_v5_restored_reference_script_is_recreated_from_same_source(monkeypatch, tmp_path):
    import importlib
    import sys
    from types import ModuleType
    import torch
    from aniflive_tts.backend.v5.engine import _refresh_reference_activation
    directory = tmp_path / "GPT_SoVITS" / "module"
    directory.mkdir(parents=True)
    (directory / "commons.py").write_text(
        "import torch\n@torch.jit.script\ndef fused_add_tanh_sigmoid_multiply(x: torch.Tensor) -> torch.Tensor:\n    return torch.tanh(x)\n")
    package = ModuleType("module")
    package.__path__ = [str(directory)]
    monkeypatch.setitem(sys.modules, "module", package)
    original = sys.modules.pop("module.commons", None)
    try:
        module = importlib.import_module("module.commons")
        first = module.fused_add_tanh_sigmoid_multiply
        expected = first(torch.tensor([0.25]))
        _refresh_reference_activation(tmp_path)
        second = module.fused_add_tanh_sigmoid_multiply
        assert first is not second
        assert torch.equal(second(torch.tensor([0.25])), expected)
        with pytest.raises(RuntimeError, match="source"):
            _refresh_reference_activation(tmp_path / "other")
    finally:
        sys.modules.pop("module.commons", None)
        if original is not None:
            sys.modules["module.commons"] = original


def test_v5_activation_reports_active_backend(monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace
    from fastapi import Request
    manager = SimpleNamespace(activate=lambda model: {"changed": True, "model": model, "voice": ""},
                              _is_v5=lambda: True)
    monkeypatch.setattr(service, "SERVICE", manager)
    monkeypatch.setattr(service, "SESSION_MANAGER", service.SpeechSessionManager(maximum_open=2, ttl_seconds=60))
    async def run():
        service.app.state.session_model_lock = asyncio.Lock()
        delivered = False
        async def receive():
            nonlocal delivered
            if delivered:
                return {"type": "http.request", "body": b"", "more_body": False}
            delivered = True
            return {"type": "http.request", "body": b'{"model":"v5-model"}', "more_body": False}
        request = Request({"type": "http", "method": "POST", "path": "/v1/models/activate",
                           "headers": [], "app": service.app}, receive)
        return await service.activate_model(request)
    response = asyncio.run(run())
    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["backend"] == "hybrid-cuda"
    assert "pytorch_fallback" not in payload
