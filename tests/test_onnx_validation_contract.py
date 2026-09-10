
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from onnx_validation import ONNXValidator
from aniflive_tts import converter


def test_onnx_tolerance_combines_absolute_and_relative_error(tmp_path: Path) -> None:
    validator = ONNXValidator(str(tmp_path))
    reference = torch.tensor([0.0, 1e-10, 100.0], dtype=torch.float64)
    actual = np.array([0.0, 1e-7, 100.05], dtype=np.float64)
    passed, metrics = validator._compare_tensors(reference, actual, "cache", 1e-3, 1e-5)
    assert passed
    assert metrics["comparison"] == "elementwise-atol-plus-rtol"
    assert metrics["max_rel_diff"] > 0.9


@pytest.mark.parametrize(
    "reference,actual",
    [
        ([0.0], [1e-4]),
        ([100.0], [100.2]),
        ([float("nan")], [float("nan")]),
        ([float("inf")], [float("inf")]),
        ([], []),
    ],
)
def test_onnx_tolerance_rejects_real_errors_and_invalid_outputs(
    tmp_path: Path, reference: list, actual: list
) -> None:
    passed, _ = ONNXValidator(str(tmp_path))._compare_tensors(
        torch.tensor(reference), np.array(actual), "cache", 1e-3, 1e-5
    )
    assert not passed


def test_discrete_outputs_never_use_float_tolerances(tmp_path: Path) -> None:
    validator = ONNXValidator(str(tmp_path))
    passed, _ = validator._compare_tensors(
        torch.tensor([1000]), np.array([1001]), "token_ids", 1.0, 2.0
    )
    assert not passed
    passed, metrics = validator._compare_tensors(
        torch.tensor([50]), np.array([50]), "length", 1e-3, 1e-5
    )
    assert passed and metrics["cosine_similarity"] == pytest.approx(1.0)


def test_equal_zero_outputs_have_finite_similarity(tmp_path: Path) -> None:
    passed, metrics = ONNXValidator(str(tmp_path))._compare_tensors(
        torch.zeros(3), np.zeros(3), "cache", 1e-3, 1e-5
    )
    assert passed and metrics["cosine_similarity"] == 1.0


def test_failed_onnx_stage_persists_its_report(tmp_path: Path, monkeypatch) -> None:
    import onnx_validation

    class Model(torch.nn.Module):
        def forward(self, value):
            return value * 2

    class Session:
        def get_inputs(self):
            from types import SimpleNamespace
            return [SimpleNamespace(name="value")]

        def run(self, output_names, inputs):
            return [inputs["value"] * 2 + 1]

    graph = tmp_path / "probe.onnx"
    graph.write_bytes(b"fixture")
    monkeypatch.setattr(onnx_validation.onnxruntime, "InferenceSession", lambda *a, **kw: Session())
    validator = ONNXValidator(str(tmp_path))
    assert not validator.validate_model(
        "Probe", str(graph), Model(), {"value": torch.tensor([1.0])}, ["output"]
    )
    rows = json.loads((tmp_path / "validation_report.json").read_text())
    assert rows[0]["passed"] is False


def test_conversion_subprocess_retains_failure_output(tmp_path: Path) -> None:
    import os
    import subprocess

    log = tmp_path / "conversion.log"
    env = {**os.environ, "ANIFLIVE_TTS_CONVERSION_LOG": str(log)}
    with pytest.raises(subprocess.CalledProcessError) as failure:
        converter._run(
            [sys.executable, "-c", "import sys; print('stage output'); print('root cause', file=sys.stderr); sys.exit(7)"],
            cwd=tmp_path, env=env,
        )
    assert failure.value.returncode == 7
    assert "stage output" in log.read_text()
    assert "root cause" in log.read_text()


@pytest.mark.parametrize("missing", [False, True])
def test_validator_uses_declared_graph_inputs(tmp_path: Path, monkeypatch, missing: bool) -> None:
    import onnx_validation
    from types import SimpleNamespace

    class Model(torch.nn.Module):
        def forward(self, value, ignored):
            return value * 2

    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="missing" if missing else "value")]

        def run(self, names, inputs):
            assert set(inputs) == {"value"}
            return [inputs["value"] * 2]

    graph = tmp_path / "probe.onnx"
    graph.write_bytes(b"fixture")
    monkeypatch.setattr(onnx_validation.onnxruntime, "InferenceSession", lambda *a, **kw: Session())
    validator = ONNXValidator(str(tmp_path))
    passed = validator.validate_model(
        "Pruned", str(graph), Model(),
        {"value": torch.tensor([1.0]), "ignored": torch.tensor([1.0])}, ["output"]
    )
    assert passed is (not missing)
    if not missing:
        report = json.loads((tmp_path / "validation_report.json").read_text())
        assert report[0]["pruned_input_names"] == ["ignored"]


@pytest.mark.parametrize("wrong_graph", [False, True])
def test_shared_random_draws_preserve_graph_and_detect_neural_errors(
    tmp_path: Path, wrong_graph: bool
) -> None:
    import hashlib

    class Noisy(torch.nn.Module):
        def __init__(self, gain):
            super().__init__()
            self.gain = gain

        def forward(self, value, scale):
            return value * self.gain + torch.randn_like(value) * scale

    values = torch.arange(1, 33, dtype=torch.float32).reshape(1, 32)
    scale = torch.tensor([0.5])
    graph = tmp_path / "noisy.onnx"
    torch.onnx.export(
        Noisy(3.0 if wrong_graph else 2.0).eval(), (values, scale), graph,
        input_names=["value", "scale"], output_names=["audio"], opset_version=18,
        dynamo=False,
    )
    before = hashlib.sha256(graph.read_bytes()).hexdigest()
    validator = ONNXValidator(str(tmp_path))
    passed = validator.validate_model(
        "Noisy", str(graph), Noisy(2.0).eval(), {"value": values, "scale": scale}, ["audio"]
    )
    assert passed is (not wrong_graph)
    assert hashlib.sha256(graph.read_bytes()).hexdigest() == before
    row = json.loads((tmp_path / "validation_report.json").read_text())[0]
    assert row["random_alignment"]["production_graph_modified"] is False
    assert len(row["random_alignment"]["controls"]) == 1
    with np.load(tmp_path / row["random_alignment"]["inputs_file"], allow_pickle=False) as data:
        assert np.any(data["noise_0"] != 0)
        assert data["input_1"].item() == 0.5


def test_random_control_count_mismatch_fails_closed(tmp_path: Path) -> None:
    class Noisy(torch.nn.Module):
        def forward(self, value):
            return value + torch.randn_like(value)

    class Deterministic(torch.nn.Module):
        def forward(self, value):
            return value + 1

    value = torch.ones(1, 8)
    graph = tmp_path / "missing-rng.onnx"
    torch.onnx.export(
        Deterministic(), (value,), graph, input_names=["value"],
        output_names=["audio"], opset_version=18, dynamo=False,
    )
    validator = ONNXValidator(str(tmp_path))
    assert not validator.validate_model("MissingRNG", str(graph), Noisy(), {"value": value}, ["audio"])
    report = json.loads((tmp_path / "validation_report.json").read_text())
    assert "draw counts differ" in report[-1]["metrics"]["error"]


def test_pytorch_inplace_update_cannot_mutate_onnx_inputs(tmp_path: Path, monkeypatch) -> None:
    import onnx_validation
    from types import SimpleNamespace

    class Model(torch.nn.Module):
        def forward(self, value):
            value.add_(1)
            return value * 2

    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="value")]

        def run(self, names, inputs):
            np.testing.assert_array_equal(inputs["value"], np.array([1.0]))
            return [(inputs["value"] + 1) * 2]

    graph = tmp_path / "inplace.onnx"
    graph.write_bytes(b"fixture")
    monkeypatch.setattr(onnx_validation.onnxruntime, "InferenceSession", lambda *a, **kw: Session())
    source = torch.tensor([1.0])
    assert ONNXValidator(str(tmp_path)).validate_model(
        "Inplace", str(graph), Model(), {"value": source}, ["output"]
    )
    assert source.item() == 1.0

@pytest.mark.parametrize("stage", ["SoVITS", "SoVITSStreaming", "SSL"])
@pytest.mark.parametrize("fail_forward", [False, True])
def test_waveform_reference_kernel_is_scoped_and_restored(
    tmp_path, stage: str, fail_forward: bool,
) -> None:
    import torch
    from onnx_validation import ONNXValidator

    native = stage in {"SoVITS", "SoVITSStreaming"}

    class Reference(torch.nn.Module):
        def forward(self, value):
            assert torch.backends.mkldnn.enabled is (not native)
            if fail_forward:
                raise RuntimeError("intentional forward failure")
            return value

    path = tmp_path / "identity.onnx"
    probe = torch.ones((1, 4))
    torch.onnx.export(
        torch.nn.Identity(), (probe,), str(path), input_names=["input"],
        output_names=["audio"], opset_version=20, dynamo=False,
    )
    validator = ONNXValidator(str(tmp_path / "validation"))
    with torch.backends.mkldnn.flags(enabled=True):
        passed = validator.validate_model(
            stage, str(path), Reference(), {"input": probe}, ["audio"],
            rtol=1e-3, atol=1e-5,
        )
        assert torch.backends.mkldnn.enabled is True
    assert passed is (not fail_forward)
    if passed:
        assert validator.validation_results[0]["reference_execution"][
            "native_cpu_waveform_reference"
        ] is native
