
import torch
from export_onnx import GPT_VALIDATION_SEEDS, gpt_validation_inputs


def test_gpt_probe_bank_is_fixed_and_independent_of_global_rng():
    assert {1234, 2026, 7}.issubset(GPT_VALIDATION_SEEDS)
    assert len(GPT_VALIDATION_SEEDS) == len(set(GPT_VALIDATION_SEEDS)) == 8
    torch.manual_seed(11)
    state = torch.get_rng_state().clone()
    first = gpt_validation_inputs(1234)
    assert torch.equal(state, torch.get_rng_state())
    torch.randn(100)
    second = gpt_validation_inputs(1234)
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert first["phoneme_ids"].shape == (1, 50)
    assert first["prompts"].shape == (1, 20)
    assert first["bert_feature"].shape == (1, 1024, 50)
