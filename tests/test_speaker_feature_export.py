from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from export_onnx import SVEmbeddingWrapper
from GPT_SoVITS.eres2net import kaldi


class _FeatureIdentity(torch.nn.Module):
    def forward3(self, features: torch.Tensor) -> torch.Tensor:
        return features


@pytest.mark.parametrize("kind", ["noise", "silence", "dc", "tone"])
def test_exported_speaker_features_match_native_kaldi(kind: str) -> None:
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(1234)
    audio = torch.randn((1, 32000), generator=generator) * 0.1
    if kind == "silence":
        audio.zero_()
    elif kind == "dc":
        audio.fill_(0.2)
    elif kind == "tone":
        audio = torch.sin(torch.arange(32000) * 2 * torch.pi * 440 / 16000)[None] * 0.2
    wrapper = SVEmbeddingWrapper(_FeatureIdentity())
    actual = wrapper(audio)
    expected = kaldi.fbank(audio, num_mel_bins=80, sample_frequency=16000, dither=0)[None]
    assert actual.shape == expected.shape == (1, 198, 80)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_speaker_feature_onnx_preserves_dynamic_kaldi_framing(tmp_path: Path) -> None:
    import onnxruntime as ort

    torch.set_num_threads(4)
    wrapper = SVEmbeddingWrapper(_FeatureIdentity()).eval()
    path = tmp_path / "features.onnx"
    generator = torch.Generator().manual_seed(7)
    dummy = torch.randn((1, 16000), generator=generator) * 0.1
    torch.onnx.export(
        wrapper, (dummy,), str(path), opset_version=20, dynamo=False,
        input_names=["audio"], output_names=["features"],
        dynamic_axes={"audio": {1: "samples"}, "features": {1: "frames"}},
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    for length in (400, 512, 16000, 32001):
        audio = torch.randn((1, length), generator=generator) * 0.1
        expected = kaldi.fbank(audio, num_mel_bins=80, sample_frequency=16000, dither=0)[None]
        actual = session.run(None, {"audio": audio.numpy()})[0]
        assert actual.shape == (1, 1 + (length - 400) // 160, 80)
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-5)
