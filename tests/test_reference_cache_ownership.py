"""A second Reference must not overwrite features retained for the first."""
from contextlib import nullcontext
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from aniflive_tts.streaming import TensorRTFixedReferenceStreamer


def test_prepared_references_own_reusable_engine_outputs(monkeypatch):
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    buffers = {
        "codes": torch.zeros((1, 1, 8), dtype=torch.int64),
        "spectrogram": torch.zeros((1, 1025, 8)),
        "sv_embedding": torch.zeros((1, 20480)),
        "bert": torch.zeros((1024, 2)),
    }
    value = 1

    def output(name):
        def run(_):
            buffers[name].fill_(value)
            return {name: buffers[name]}
        return run

    def frontend(*_):
        buffers["bert"].fill_(value)
        return [1, 2], buffers["bert"], "reference"

    streamer = TensorRTFixedReferenceStreamer.__new__(TensorRTFixedReferenceStreamer)
    streamer.torch = torch
    streamer.logger = logging.getLogger(__name__)
    streamer._load_audio = lambda *_: np.zeros(48000, dtype=np.float32)
    streamer.engine = SimpleNamespace(
        device="cpu", precision=torch.float32, stream=None, version="v2ProPlus",
        hps={"data": {"sampling_rate": 32000}},
        model_ssl=lambda _: {"last_hidden_state": torch.zeros((1, 768, 8))},
        model_vq=output("codes"), model_spectrogram=output("spectrogram"),
        model_sv_embedding=output("sv_embedding"), get_phones_and_bert=frontend,
    )
    first = streamer._prepare_reference(reference_id="a", reference_wav=Path("a.wav"),
                                       reference_text="first", reference_language="en")
    value = 2
    second = streamer._prepare_reference(reference_id="b", reference_wav=Path("b.wav"),
                                        reference_text="second", reference_language="en")
    for field in ("prompt_semantic", "spectrogram", "speaker_embedding", "bert"):
        assert torch.all(getattr(first, field) == 1), field
        assert torch.all(getattr(second, field) == 2), field
