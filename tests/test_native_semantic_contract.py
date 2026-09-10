from types import SimpleNamespace

import pytest
import torch

from AR.models.utils import sample as original_sample
from aniflive_tts.backend.semantic_runtime import TransformerSemanticRuntime


@pytest.mark.parametrize("seed", [0, 7, 1234])
@pytest.mark.parametrize("suppress_eos", [False, True])
@pytest.mark.parametrize("penalty", [1.0, 1.35])
def test_native_sampling_matches_upstream_full_vocab_history_and_random_draws(
    monkeypatch, seed, suppress_eos, penalty,
):
    monkeypatch.setenv("ANIFLIVE_TTS_SEMANTIC_SAMPLING", "native-v2proplus-v1")
    runtime = TransformerSemanticRuntime(
        engine=SimpleNamespace(device=torch.device("cpu")),
        sample_topk=None, torch=torch, trt=None,
    )
    logits = torch.linspace(-2, 2, 1025).reshape(1, -1)
    history = torch.tensor([[1000, 1001, 1000, 1019, 1023]])
    reference_logits = logits[:, :-1].clone() if suppress_eos else logits.clone()
    torch.manual_seed(seed)
    sampled, _ = original_sample(
        reference_logits, history, top_k=15, top_p=0.9,
        repetition_penalty=penalty, temperature=0.8,
    )
    expected = torch.where(
        (sampled == 1024) | (reference_logits.argmax(-1, keepdim=True) == 1024),
        torch.full_like(sampled, 1024), sampled,
    ).long()
    torch.manual_seed(seed)
    actual = runtime._sample_native(
        logits, history, temperature=0.8, top_k=15, top_p=0.9,
        repetition_penalty=penalty, suppress_eos=suppress_eos,
    )
    assert torch.equal(actual, expected)


def test_native_policy_rejects_packed_topk_instead_of_silent_approximation(monkeypatch):
    monkeypatch.setenv("ANIFLIVE_TTS_SEMANTIC_SAMPLING", "native-v2proplus-v1")
    runtime = TransformerSemanticRuntime(
        engine=SimpleNamespace(device=torch.device("cpu")),
        sample_topk=None, torch=torch, trt=None,
    )
    with pytest.raises(RuntimeError, match="full 1025-token logits"):
        runtime._sample_native(
            torch.ones(1, 50), torch.tensor([[1]]),
            temperature=1, top_k=15, top_p=1, repetition_penalty=1.35, suppress_eos=False,
        )


def test_sampling_contract_does_not_select_by_voice_name(monkeypatch):
    monkeypatch.setenv("ANIFLIVE_TTS_SEMANTIC_SAMPLING", "native-v2proplus-v1")
    for voice in ("voice-a", "voice-b"):
        runtime = TransformerSemanticRuntime(
            engine=SimpleNamespace(device=torch.device("cpu"), model_id=voice),
            sample_topk=None, torch=torch, trt=None,
        )
        assert runtime.sampling_contract == "native-v2proplus-v1"


def test_persistent_context_binds_full_logits_for_both_cache_directions():
    from aniflive_tts.backend.semantic_runtime import _PersistentGPTStepContexts

    contexts = []
    class Context:
        def __init__(self):
            self.bindings = {}
        def set_tensor_address(self, name, pointer):
            self.bindings[name] = pointer
            return True
        def execute_async_v3(self, **kwargs):
            return True
    def context():
        value = Context()
        contexts.append(value)
        return value
    shapes = {"topk_values": (1, 50), "topk_indices": (1, 50), "logits": (1, 1025)}
    model = SimpleNamespace(
        device=torch.device("cpu"),
        tensor_location={},
        tensor_dtype={"topk_values": torch.float32, "topk_indices": torch.int64,
                      "logits": torch.float32},
        output_names=("topk_values", "topk_indices", "logits", "k_cache_new", "v_cache_new"),
        engine=SimpleNamespace(get_tensor_shape=lambda name: shapes[name],
                               create_execution_context=context),
    )
    result = _PersistentGPTStepContexts(
        torch=torch, trt=SimpleNamespace(TensorLocation=SimpleNamespace(DEVICE="device", HOST="host")),
        model=model, stream=SimpleNamespace(cuda_stream=0),
        cache_pair=[(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)) for _ in range(2)],
        x_length=torch.tensor([1]), y_length=torch.tensor([1]),
    )
    assert all(c.bindings["logits"] == result.outputs["logits"].data_ptr() for c in contexts)
    assert result.execute(step=0, current=torch.tensor([[1]]), index=torch.tensor([0]))["logits"].shape == (1, 1025)
