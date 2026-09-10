import json

import numpy as np
import pytest
import soundfile as sf
import torch

from aniflive_tts import model_package
from aniflive_tts import workstation_evaluation as evaluation


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_speaker_evaluator_respects_engine_input_dtype_and_stream(tmp_path, monkeypatch, dtype):
    import run_trt_inference

    (tmp_path / "manifest.json").write_text(json.dumps({}))
    audio = (np.sin(np.arange(4000) * 0.013) * 0.2).astype(np.float32)
    path = tmp_path / "audio.wav"
    sf.write(path, audio, 16000, subtype="FLOAT")
    expected = torch.from_numpy(evaluation._trim_and_normalize(audio))[None].to(dtype)
    stream = object()
    original_to = torch.Tensor.to

    def cpu_transfer(tensor, *args, **kwargs):
        if args and args[0] == "cuda":
            args = args[1:]
        return original_to(tensor, *args, **kwargs)

    class Module:
        tensor_dtype = {"audio": dtype}

        def __init__(self, path, *, device, stream):
            assert stream is expected_stream
            assert device == "cuda"

        def __call__(self, inputs):
            assert inputs["audio"].dtype == dtype
            assert torch.equal(inputs["audio"], expected)
            return {"sv_embedding": torch.ones(1, 4)}

    expected_stream = stream
    monkeypatch.setattr(torch.Tensor, "to", cpu_transfer)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(model_package, "select_engine_dir", lambda *args: tmp_path)
    monkeypatch.setattr(run_trt_inference, "TRTModule", Module)
    actual = evaluation._TensorRTSpeakerEmbedder(tmp_path)(path)
    np.testing.assert_array_equal(actual, np.ones(4))
