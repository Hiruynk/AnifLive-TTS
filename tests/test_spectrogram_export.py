
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from export_onnx import SpectrogramWrapper


@pytest.mark.parametrize("fft,hop,window", [(512, 160, 400), (2048, 640, 2048)])
def test_portable_spectrogram_matches_native_stft(tmp_path: Path, fft: int, hop: int, window: int) -> None:
    generator = torch.Generator().manual_seed(1234)
    model = SpectrogramWrapper(fft, hop, window, 32000).eval()
    noise = torch.randn(1, 16000, generator=generator)
    tone = torch.sin(torch.arange(16000) * (2 * torch.pi * 440 / 32000)).unsqueeze(0)
    impulse = torch.zeros_like(noise)
    impulse[0, 8000] = 0.95
    for audio in (noise, tone, impulse, torch.zeros_like(noise)):
        padding = (fft - hop) // 2
        padded = torch.nn.functional.pad(audio.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
        native = torch.sqrt(
            torch.stft(padded, fft, hop_length=hop, win_length=window,
                       window=model.hann_window, center=False, return_complex=True).abs().square()
            + 1e-8
        )
        actual = model(audio)
        assert actual.shape == native.shape
        relative_error = torch.linalg.vector_norm((actual - native).double()) / torch.linalg.vector_norm(native.double()).clamp_min(1e-20)
        assert relative_error < 8 * torch.finfo(torch.float32).eps * math.log2(fft)


def test_portable_spectrogram_onnx_uses_original_strict_tolerance(tmp_path: Path) -> None:
    model = SpectrogramWrapper(2048, 640, 2048, 32000).eval()
    graph = tmp_path / "spectrogram.onnx"
    torch.onnx.export(
        model, (torch.zeros(1, 16000),), graph,
        input_names=["audio"], output_names=["spectrogram"],
        dynamic_axes={"audio": {1: "time"}, "spectrogram": {2: "time"}},
        opset_version=20, dynamo=False,
    )
    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(2026)
    for length in (16000, 48000, 96000):
        noise = torch.randn(1, length, generator=generator)
        tone = torch.sin(torch.arange(length) * (2 * torch.pi * 440 / 32000)).unsqueeze(0) * 0.95
        for audio in (noise, tone, torch.zeros_like(noise)):
            expected = model(audio).numpy()
            actual = session.run(["spectrogram"], {"audio": audio.numpy()})[0]
            assert actual.dtype == np.float32
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("fft,hop,window", [(1000, 320, 1000), (2048, 0, 2048), (2048, 640, 4096)])
def test_portable_spectrogram_rejects_unsupported_configuration(fft, hop, window) -> None:
    with pytest.raises(ValueError):
        SpectrogramWrapper(fft, hop, window, 32000)
