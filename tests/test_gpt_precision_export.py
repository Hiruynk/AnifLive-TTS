from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from onnx_to_fp16 import optimize_single_model


@pytest.mark.parametrize(
    ("stage", "expected"),
    [("ssl", TensorProto.FLOAT), ("bert", TensorProto.FLOAT16),
     ("gpt_encoder", TensorProto.FLOAT), ("gpt_step", TensorProto.FLOAT),
     ("gpt_block", TensorProto.FLOAT), ("sovits", TensorProto.FLOAT16)],
)
def test_conversion_preserves_gpt_state_precision(
    tmp_path: Path, stage: str, expected: int,
) -> None:
    weights = np.array([[1.0003, 0.1251], [0.2001, 1.0007]], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weights"], ["output"])],
        stage,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(weights, "weights")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 9
    source = tmp_path / f"{stage}.onnx"
    output = tmp_path / f"{stage}-converted.onnx"
    onnx.save(model, source)
    optimize_single_model(str(source), str(output))
    result = onnx.load(output)
    onnx.checker.check_model(result)
    assert result.graph.input[0].type.tensor_type.elem_type == expected
    assert result.graph.output[0].type.tensor_type.elem_type == expected
    tensor = next(item for item in result.graph.initializer if item.name == "weights")
    assert tensor.data_type == expected
    if expected == TensorProto.FLOAT:
        np.testing.assert_array_equal(numpy_helper.to_array(tensor), weights)
