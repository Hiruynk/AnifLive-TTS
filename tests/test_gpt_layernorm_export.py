from __future__ import annotations

import hashlib
import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from export_onnx import stabilize_gpt_layer_norm


def _model(path, opset=20, axis=-1):
    scale = np.linspace(.8, 1.2, 16, dtype=np.float32)
    bias = np.linspace(-.1, .1, 16, dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("LayerNormalization", ["x", "scale", "bias"], ["y"], axis=axis, epsilon=1e-5)],
        "norm",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 16])],
        [numpy_helper.from_array(scale, "scale"), numpy_helper.from_array(bias, "bias")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=9)
    onnx.save(model, path)
    return scale, bias


@pytest.mark.parametrize("opset", [17, 20])
def test_centered_export_matches_native_layernorm_and_retains_weights(tmp_path, opset):
    path = tmp_path / "norm.onnx"
    scale, bias = _model(path, opset)
    assert stabilize_gpt_layer_norm(path) == 1
    model = onnx.load(path)
    assert not any(node.op_type == "LayerNormalization" for node in model.graph.node)
    weights = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    assert np.array_equal(weights["scale"], scale)
    assert np.array_equal(weights["bias"], bias)
    metadata = {value.key: value.value for value in model.metadata_props}
    assert json.loads(metadata["aniflive.layer_norm"])["native_pytorch_reference"] == "unchanged"
    x = np.random.default_rng(42).normal(size=(2, 3, 16)).astype(np.float32)
    expected = torch.nn.functional.layer_norm(
        torch.from_numpy(x), [16], torch.from_numpy(scale), torch.from_numpy(bias), 1e-5,
    ).numpy()
    actual = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-5)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    assert stabilize_gpt_layer_norm(path) == 0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum


def test_unsupported_axis_does_not_replace_original_graph(tmp_path):
    path = tmp_path / "norm.onnx"
    _model(path, axis=0)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="last-axis"):
        stabilize_gpt_layer_norm(path)
    assert path.read_bytes() == before
