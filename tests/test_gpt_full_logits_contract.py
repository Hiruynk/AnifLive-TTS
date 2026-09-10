from types import SimpleNamespace

import pytest
import torch

from aniflive_tts.backend.contracts import (
    STAGE_IO_CONTRACTS,
    stage_outputs_supported,
    validate_full_logits_output,
)
from export_onnx import GPTEncoder, GPTStep


class Model:
    def infer_first_stage(self, phones, prompts, bert):
        logits = torch.arange(1025, dtype=torch.float32).reshape(1, -1)
        cache = [torch.zeros(1, 4, 8)]
        return logits, cache, cache, 2, 2

    def infer_next_stage(self, samples, keys, values, x_length, y_length, index):
        return torch.arange(1025, dtype=torch.float32).reshape(1, -1), keys, values


def test_full_logits_extend_outputs_without_changing_legacy_values():
    model = SimpleNamespace(model=Model())
    inputs = (torch.zeros(1, 2, dtype=torch.long),
              torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 1024, 2))
    legacy = GPTEncoder(model, max_len=16)(*inputs)
    extended = GPTEncoder(model, max_len=16, full_logits=True)(*inputs)
    assert len(legacy) == 6 and len(extended) == 7
    for first, second in zip(legacy, extended[:6], strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert extended[-1].shape == (1, 1025)
    step_inputs = (torch.zeros(1, 1, dtype=torch.long), *legacy[2:], torch.tensor([0]))
    legacy_step = GPTStep(model)(*step_inputs)
    extended_step = GPTStep(model, full_logits=True)(*step_inputs)
    assert len(extended_step) == 5
    for first, second in zip(legacy_step, extended_step[:4], strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert extended_step[-1].shape == (1, 1025)


@pytest.mark.parametrize("stage", list(STAGE_IO_CONTRACTS))
def test_only_gpt_stages_accept_the_explicit_logits_extension(stage):
    outputs = STAGE_IO_CONTRACTS[stage][1]
    assert stage_outputs_supported(stage, outputs)
    assert stage_outputs_supported(stage, (*outputs, "logits")) == (stage in {"gpt_encoder", "gpt_step"})
    assert not stage_outputs_supported(stage, (*outputs, "unknown"))
    assert not stage_outputs_supported(stage, outputs[:-1])
    assert not stage_outputs_supported(stage, (*outputs, outputs[0]))


def test_native_logits_shape_dtype_and_location_are_validated():
    trt = SimpleNamespace(float32="fp32", TensorLocation=SimpleNamespace(DEVICE="device"))
    engine = SimpleNamespace(
        get_tensor_shape=lambda name: (1, 1025),
        get_tensor_dtype=lambda name: "fp32",
        get_tensor_location=lambda name: "device",
    )
    assert validate_full_logits_output("gpt_step", ["logits"], engine, trt)
    engine.get_tensor_shape = lambda name: (1, 50)
    with pytest.raises(RuntimeError, match="1025"):
        validate_full_logits_output("gpt_step", ["logits"], engine, trt)
