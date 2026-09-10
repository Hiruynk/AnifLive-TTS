import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from aniflive_tts import workstation_conversion_worker as worker


def test_conversion_releases_generation_models_before_evaluation(tmp_path: Path, monkeypatch):
    import faster_whisper
    import run_inference
    import run_trt_inference
    from scipy.io import wavfile

    events, refs = [], {}
    reference = tmp_path / "voices/default/reference.wav"
    reference.parent.mkdir(parents=True)
    wavfile.write(reference, 32000, np.zeros(3200, np.float32))

    class Module:
        input_names = ("pred_semantic", "text_seq", "refer_spec", "sv_emb", "noise_scale", "acoustic_noise", "result_length", "overlap_frames", "overlap_enabled")
        tensor_dtype = {name: torch.float16 for name in ("overlap_frames", "audio", "latent", "latent_mask")}
        engine = SimpleNamespace(get_tensor_shape=lambda name: (1, 192, 32))
        def to(self, device):
            assert device == "cpu"
            return self

        def decode(self, *args, **kwargs):
            return torch.ones(1)

        def decode_streaming(self, *args, **kwargs):
            return torch.ones(1), None, None

        def __call__(self, inputs, **kwargs):
            return {}

    class PT:
        def __init__(self, *args):
            events.append("pytorch")
            refs["pytorch"] = weakref.ref(self)
            self.ssl_model = Module()
            self.bert_model = Module()
            self.vq_model = Module()
            self.t2s_model = Module()
            self.t2s_model.model = SimpleNamespace(infer_panel=lambda *a, **k: None)
            self.sv_model = SimpleNamespace(embedding_model=Module())

        def infer(self, *args, **kwargs):
            self.t2s_model.model.infer_panel(
                torch.ones((1, 2), dtype=torch.int64), torch.tensor([2]),
                torch.ones((1, 3), dtype=torch.int64), torch.zeros(1, 4, 2),
                **kwargs,
            )
            self.vq_model.decode(
                torch.ones((1, 1, 2), dtype=torch.int64), torch.ones((1, 2), dtype=torch.int64),
                [torch.ones(1, 2, 3)], sv_emb=[torch.ones(1, 4)],
            )
            return np.full(3200, 0.1, np.float32), 32000

    class TRT:
        def __init__(self, *args, **kwargs):
            assert refs["pytorch"]() is None
            events.append("tensorrt")
            refs["tensorrt"] = weakref.ref(self)
            self.model_gpt_enc = Module()
            self.model_sovits = Module()
            self.model_sovits_stream = Module()
            self.hps = {"data": {"hop_length": 640}}

        def infer(self, *args, output_path, **kwargs):
            self.model_gpt_enc({
                "phoneme_ids": torch.ones((1, 2), dtype=torch.int64),
                "prompts": torch.ones((1, 3), dtype=torch.int64),
                "bert_feature": torch.zeros(1, 4, 2),
            })
            self.model_sovits({
                "pred_semantic": torch.ones((1, 1, 2), dtype=torch.int64),
                "text_seq": torch.ones((1, 2), dtype=torch.int64),
                "refer_spec": torch.ones(1, 2, 3), "sv_emb": torch.ones(1, 4),
            })
            wavfile.write(output_path, 32000, np.full(3200, 0.1, np.float32))

    def asr(*args, **kwargs):
        assert refs["pytorch"]() is None
        assert refs["tensorrt"]() is None
        events.append("asr")
        return object()

    def object_data(path, label):
        return {
            "deployment checkpoints": {"gpt": {"relative_path": "gpt"}, "sovits": {"relative_path": "sovits"}},
            "model manifest": {"default_voice_profile": "default"},
            "voice profile": {"reference_audio": "reference.wav", "reference_text": "reference", "reference_language": "ja"},
            "export configuration": {"data": {"max_len": 1000}, "streaming": {"overlap_frames": 32}},
        }[label]

    monkeypatch.setattr(worker, "_object", object_data)
    monkeypatch.setattr(worker, "_CASES", {"ja": ("one", "two")})
    monkeypatch.setattr(worker, "select_engine_dir", lambda *a: tmp_path)
    monkeypatch.setattr(worker, "_release_unused_cuda", gc.collect)
    monkeypatch.setattr(worker, "_deployment_semantic_reference", lambda *a, **k: lambda *a, **k: None)
    monkeypatch.setattr(run_inference, "GPTSoVITSInference", PT)
    monkeypatch.setattr(run_trt_inference, "GPTSoVITS_TRT_Inference", TRT)
    monkeypatch.setattr(faster_whisper, "WhisperModel", asr)
    monkeypatch.setattr(worker, "_TensorRTSpeakerEmbedder", lambda *a: lambda *a: np.ones(4))
    monkeypatch.setattr(worker, "_transcribe", lambda *a: "text")
    monkeypatch.setattr(worker, "spoken_content_error_rate", lambda *a: 0.0)
    monkeypatch.setattr(worker, "_spectral_quality", lambda *a: {"log_mel_cosine": 1.0, "duration_difference_ratio": 0.0})
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a: None)
    output = tmp_path / "output"
    output.mkdir()
    rows, paths = worker._complete_output_cases(
        package=tmp_path, selected=tmp_path, shared=tmp_path, asr_model=tmp_path,
        output=output, semantic={"minimum_logits_cosine": 1.0, "top1_agreement": 1.0, "greedy_sequence_agreement": 1.0},
        pytorch_onnx_cosine=1.0,
    )
    assert events == ["pytorch", "tensorrt", "asr"]
    assert len(rows) == 2
    assert all(path.is_file() for path in paths)
    assert (output / "partial-audio-cases.json").is_file()
