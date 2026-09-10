from __future__ import annotations

import json

import pytest

from aniflive_tts import workstation_conversion_worker as worker
from aniflive_tts import workstation_evaluation as evaluation


def _plan(tmp_path, cases):
    path = tmp_path / "listening.json"
    path.write_text(json.dumps({"schema": "aniflive-stage-listening-plan-v1", "cases": cases}))
    return path


def test_listening_plan_preserves_distinct_target_languages(tmp_path):
    cases = [{"language": "zh", "text": "測試語音"},
             {"language": "yue", "text": "我哋測試"}]
    assert worker._listening_cases(_plan(tmp_path, cases)) == [
        ("zh", "測試語音"), ("yue", "我哋測試"),
    ]


@pytest.mark.parametrize("cases", [
    [], [{"language": "xx", "text": "text"}], [{"language": [], "text": "text"}],
    [{"language": "ja", "text": ""}], [{"language": "ja", "text": "x" * 513}],
    [{"language": "ja", "text": "a\x00b"}], [{"language": "ja", "text": "a"}] * 11,
])
def test_listening_plan_rejects_unbounded_or_invalid_cases(tmp_path, cases):
    with pytest.raises(worker.ConversionParityWorkerError):
        worker._listening_cases(_plan(tmp_path, cases))


def test_evaluation_dispatches_explicit_listening_plan_without_benchmark(tmp_path, monkeypatch):
    seen = []
    def listen(manifest, output, *, listening_only):
        seen.append(listening_only)
        return {"schema": "aniflive-stage-listening-v1", "status": "measured"}, []
    monkeypatch.setattr(worker, "run_conversion_parity", listen)
    report, _ = evaluation.run_evaluation(
        {"container_input_paths": {"listening_plan": "/plan.json"}}, tmp_path,
    )
    assert seen == [True]
    assert report["status"] == "measured"


@pytest.mark.parametrize("native_only", [False, True])
def test_listening_result_cannot_become_a_qualified_package(tmp_path, monkeypatch, native_only):
    dirs = {}
    for name in ("model_package", "selected_checkpoints", "shared_dir", "asr_model"):
        path = tmp_path / name
        path.mkdir()
        dirs[name] = str(path)
    selected = tmp_path / "selected_checkpoints" / "selected"
    selected.mkdir()
    (selected / "deployment-checkpoints.json").write_text("{}")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"status": "human-locked"}))
    dirs["deployment_reference"] = str(reference)
    plan = _plan(tmp_path, [{"language": "zh", "text": "測試"}])
    if native_only:
        value = json.loads(plan.read_text())
        value.update(native_only=True, include_native_pytorch=True)
        plan.write_text(json.dumps(value))
    dirs["listening_plan"] = str(plan)
    package = tmp_path / "model_package"
    (package / "manifest.json").write_text("{}")
    engine = package / "engine"
    engine.mkdir()
    (engine / "engine-manifest.json").write_text("{}")
    monkeypatch.setenv("ANIFLIVE_TTS_SOURCE_DIR", str(tmp_path))
    monkeypatch.setattr(worker, "_pytorch_onnx_evidence", lambda _: (1.0, []))
    monkeypatch.setattr(worker, "_onnx_trt_semantic_parity", lambda _: {})
    monkeypatch.setattr(worker, "select_engine_dir", lambda *_: engine)
    row = {
        "pytorch_onnx_logits_cosine": 1.0, "onnx_trt_logits_cosine": 1.0,
        "greedy_sequence_agreement": 1.0, "log_mel_cosine": 1.0,
        "speaker_cosine": 1.0, "duration_difference_ratio": 0.0,
        "content_regression": False, "new_artifacts": False,
    }
    monkeypatch.setattr(worker, "_complete_output_cases", lambda **_: ([row], []))
    output = tmp_path / "output"
    report, artifacts = worker.run_conversion_parity(
        {"container_input_paths": dirs}, output, listening_only=True,
    )
    assert report["schema"] == "aniflive-stage-listening-v1"
    assert report["status"] == "measured"
    assert report["numerical_comparison_status"] == ("not-applicable" if native_only else "passed")
    if native_only:
        assert report["backend_chain"] == ["PyTorch"]
        assert report["conversion_parity_evaluated"] is False
    assert report["qualification"]["release_qualified"] is False
    assert not (output / "model-package").exists()
    assert output / "listening-plan.json" in artifacts


def test_production_diagnostic_is_bounded_and_always_stops_servers(tmp_path, monkeypatch):
    import json
    import subprocess

    from aniflive_tts import workstation_conversion_worker as conversion
    from aniflive_tts import workstation_evaluation as evaluation

    package, shared = tmp_path / "package", tmp_path / "shared"
    package.mkdir()
    shared.mkdir()
    (package / "manifest.json").write_text(json.dumps({"model_id": "qa"}))
    launched, stopped = [], []
    def start(argv, **kwargs):
        assert kwargs["shell"] is False
        launched.append(kwargs["env"]["ANIFLIVE_TTS_REPETITION_PENALTY"])
        return object()
    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(evaluation, "_wait_for_service",
                        lambda *args: {"engine_fingerprint": "fixture"})
    monkeypatch.setattr(evaluation, "_terminate", lambda server: stopped.append(server))
    monkeypatch.setattr(evaluation, "_request", lambda *args, **kwargs: (
        200, {"x-tensorrt-backend": "TensorRT-11", "x-pytorch-fallback": "false",
              "x-tts-model": "qa"}, evaluation._pcm_wav(b"\x01\x00" * 100, 32000),
    ))
    records, artifacts = conversion._production_runtime_variants(
        package, shared, [("zh", "今天天氣很好。")], tmp_path / "output",
    )
    assert launched == ["1.0", "1.35"]
    assert len(stopped) == 2
    assert len(records) == 2
    assert all(row["qualification_status"] == "diagnostic-only" for row in records)
    assert len([path for path in artifacts if path.suffix == ".wav"]) == 2

    monkeypatch.setattr(evaluation, "_request", lambda *args, **kwargs: (500, {}, b"failed"))
    import pytest
    with pytest.raises(conversion.ConversionParityWorkerError, match="HTTP 500"):
        conversion._production_runtime_variants(
            package, shared, [("zh", "今天天氣很好。")], tmp_path / "failed",
        )
    assert len(stopped) == 3


def test_native_decoder_replay_uses_captured_noise_without_new_random_draws(monkeypatch):
    import torch
    from types import SimpleNamespace
    from aniflive_tts import workstation_conversion_worker as conversion

    captured = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    seen = {}
    class Module:
        engine = SimpleNamespace(get_tensor_shape=lambda name: (1, 192, 12))
        input_names = ("pred_semantic", "text_seq", "refer_spec", "sv_emb",
                       "acoustic_noise", "result_length", "overlap_frames", "overlap_enabled")
        tensor_dtype = {name: torch.float32 for name in
                        ("audio", "latent", "latent_mask", "overlap_frames")}
        def __call__(self, inputs, **kwargs):
            seen.update(inputs)
            return kwargs["outputs"]
    def forbidden(*args):
        raise AssertionError("Replay must not generate another acoustic-noise sample")
    monkeypatch.setattr(conversion, "_shared_acoustic_noise", forbidden)
    decoder = conversion._FullSpanStreamDecoder(Module(), None, captured_noise=captured)
    tokens = torch.tensor([[[7, 8, 9]]])
    decoder({"pred_semantic": tokens, "text_seq": torch.tensor([[1, 2]]),
             "refer_spec": torch.ones(1, 2, 3), "sv_emb": torch.ones(1, 4)})
    assert torch.equal(seen["pred_semantic"], tokens)
    assert torch.equal(seen["acoustic_noise"], captured)
