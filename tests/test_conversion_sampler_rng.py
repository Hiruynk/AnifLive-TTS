import torch

from run_trt_inference import sample_topk


def test_explicit_sampler_state_is_independent_of_ambient_random_draws():
    values = torch.tensor([[1.0, 0.8, 0.4, 0.1]])
    indices = torch.tensor([[10, 20, 30, 40]])
    first = torch.Generator().manual_seed(1234)
    second = torch.Generator().manual_seed(1234)
    for _ in range(20):
        left = sample_topk(values, indices, top_k=3, generator=first)
        torch.rand(997)
        right = sample_topk(values, indices, top_k=3, generator=second)
        assert torch.equal(left, right)
    assert torch.equal(first.get_state(), second.get_state())


def test_default_sampler_preserves_global_rng_behavior():
    values = torch.tensor([[1.0, 0.8, 0.4, 0.1]])
    indices = torch.tensor([[10, 20, 30, 40]])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        actual = sample_topk(values, indices, top_k=3)
        state = torch.get_rng_state()
        torch.manual_seed(1234)
        choice = torch.multinomial(torch.softmax(values[:, :3], dim=-1), 1)
        expected = torch.gather(indices[:, :3], -1, choice)
        assert torch.equal(actual, expected)
        assert torch.equal(state, torch.get_rng_state())

def test_shared_acoustic_controls_keep_nonzero_noise_and_full_span():
    from types import SimpleNamespace
    from aniflive_tts.workstation_conversion_worker import (
        _FullSpanStreamDecoder, _shared_acoustic_noise,
    )

    tokens = torch.ones((1, 1, 24), dtype=torch.int64)
    first = torch.Generator().manual_seed(1234)
    second = torch.Generator().manual_seed(1234)
    expected = _shared_acoustic_noise(tokens, first)

    class Module:
        input_names = (
            "pred_semantic", "text_seq", "refer_spec", "sv_emb", "noise_scale",
            "acoustic_noise", "result_length", "overlap_frames", "overlap_enabled",
        )
        tensor_dtype = {name: torch.float16 for name in ("overlap_frames", "audio", "latent", "latent_mask")}
        engine = SimpleNamespace(get_tensor_shape=lambda name: (1, 192, 32))

        def __call__(self, inputs, *, outputs):
            assert torch.equal(inputs["acoustic_noise"], expected)
            assert torch.count_nonzero(expected) > 0
            assert inputs["noise_scale"].item() == 0.5
            assert inputs["result_length"].item() == 24
            assert inputs["overlap_enabled"].item() == 0
            assert inputs["overlap_frames"].shape == (1, 192, 32)
            assert outputs["audio"].shape == (1, 1, 24 * 1280)
            outputs["audio"].fill_(1)
            return outputs

    decoder = _FullSpanStreamDecoder(Module(), second)
    result = decoder({
        "pred_semantic": tokens, "text_seq": torch.ones(1, 10),
        "refer_spec": torch.ones(1, 1025, 200), "sv_emb": torch.ones(1, 20480),
        "noise_scale": torch.tensor([0.5]), "speed": torch.tensor([1.0]),
    })
    assert result["audio"].shape[-1] == 24 * 1280
    assert decoder.calls[0]["acoustic_noise"].shape == (1, 192, 48)
