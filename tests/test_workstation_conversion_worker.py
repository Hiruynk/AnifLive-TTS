from __future__ import annotations

import json
from pathlib import Path

import pytest

from aniflive_tts import workstation_conversion_worker as worker


def _validation_row(model: str, output: str, cosine: float = 1.0) -> dict:
    return {
        "model_name": model,
        "output_name": output,
        "passed": True,
        "metrics": {"cosine_similarity": cosine},
    }


def test_pytorch_onnx_evidence_requires_every_neural_stage(tmp_path: Path) -> None:
    onnx = tmp_path / "onnx"
    onnx.mkdir()
    rows = [
        _validation_row("GPTEncoder", "topk_values", 0.9999),
        _validation_row("GPTStep", "topk_values", 0.9998),
        _validation_row("SoVITS", "audio"),
        _validation_row("SVEmbedding", "sv_embedding"),
    ]
    (onnx / "pytorch-onnx-validation.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )

    cosine, evidence = worker._pytorch_onnx_evidence(tmp_path)

    assert cosine == pytest.approx(0.9998)
    assert evidence == rows


def test_pytorch_onnx_evidence_fails_closed_when_gpt_step_is_missing(
    tmp_path: Path,
) -> None:
    onnx = tmp_path / "onnx"
    onnx.mkdir()
    rows = [
        _validation_row("GPTEncoder", "topk_values"),
        _validation_row("SoVITS", "audio"),
        _validation_row("SVEmbedding", "sv_embedding"),
    ]
    (onnx / "pytorch-onnx-validation.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )

    with pytest.raises(worker.ConversionParityWorkerError, match="every stage"):
        worker._pytorch_onnx_evidence(tmp_path)


def test_conversion_worker_has_no_model_name_special_case() -> None:
    source = Path(worker.__file__).read_text(encoding="utf-8").casefold()
    assert 'model_id == "odette"' not in source
    assert "odette-v2" not in source


def test_exporter_validates_every_packaged_neural_stage() -> None:
    exporter = (
        Path(__file__).resolve().parents[1] / "minimal_inference" / "export_onnx.py"
    ).read_text(encoding="utf-8")
    for stage in (
        "SSL",
        "BERT",
        "VQEncoder",
        "GPTEncoder",
        "GPTStep",
        "SoVITS",
        "SoVITSStreaming",
        "Spectrogram",
        "SVEmbedding",
    ):
        assert f'model_name="{stage}"' in exporter

def _multicase_evidence(tmp_path: Path, rows: list[dict]) -> tuple[float, list[dict]]:
    (tmp_path / "onnx").mkdir(exist_ok=True)
    (tmp_path / "onnx" / "pytorch-onnx-validation.json").write_text(json.dumps(rows))
    return worker._pytorch_onnx_evidence(tmp_path)


def _multicase_rows() -> list[dict]:
    return [
        dict(_validation_row(stage, "topk_values", 0.999 - index * 0.001), case_id=f"seed-{index}")
        for stage in ("GPTEncoder", "GPTStep")
        for index in range(8)
    ] + [_validation_row("SoVITS", "audio"), _validation_row("SVEmbedding", "sv_embedding")]


def test_all_gpt_probe_cases_contribute_to_parity(tmp_path: Path) -> None:
    rows = _multicase_rows()
    cosine, evidence = _multicase_evidence(tmp_path, rows)
    assert cosine == pytest.approx(0.992)
    assert evidence == rows


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nan", "failed", "missing_logits"])
def test_multicase_evidence_rejects_incomplete_or_invalid_probes(
    tmp_path: Path, defect: str
) -> None:
    rows = _multicase_rows()
    if defect == "missing":
        rows.pop(0)
    elif defect == "duplicate":
        rows.append(dict(rows[0]))
    elif defect == "nan":
        rows[0]["metrics"]["cosine_similarity"] = float("nan")
    elif defect == "failed":
        rows[0]["passed"] = False
    else:
        rows[0]["output_name"] = "k_cache"
    with pytest.raises(worker.ConversionParityWorkerError):
        _multicase_evidence(tmp_path, rows)

def test_decode_recorder_observes_without_replacing_tensors() -> None:
    import torch

    class Module:
        input_names = ["pred_semantic"]

        def __call__(self, inputs: dict, *, sync: bool) -> object:
            assert inputs["pred_semantic"] is original
            assert sync is False
            return sentinel

    original = torch.tensor([[[1, 2, 3]]])
    sentinel = object()
    recorder = worker._DecodeInputRecorder(Module())
    assert recorder.input_names == ["pred_semantic"]
    assert recorder({"pred_semantic": original}, sync=False) is sentinel
    original[0, 0, 0] = 9
    assert recorder.calls[0]["pred_semantic"].tolist() == [[[1, 2, 3]]]


def test_decode_evidence_retains_different_semantics_without_claiming_alignment(
    tmp_path: Path,
) -> None:
    import numpy as np

    first = {
        "pred_semantic": np.array([[[1, 2]]]),
        "text_seq": np.array([[3, 4]]),
        "refer_spec": np.ones((1, 2, 3)),
        "sv_emb": np.ones((1, 4)),
    }
    second = dict(first, pred_semantic=np.array([[[1, 2, 3]]]))
    path = tmp_path / "inputs.npz"
    result = worker._decode_input_evidence([first], [second], path)
    assert result["segments"][0]["pred_semantic"]["exact"] is False
    assert result["segments"][0]["refer_spec"]["exact"] is True
    assert result["decoder_random_draws_aligned"] is False
    with np.load(path) as arrays:
        np.testing.assert_array_equal(arrays["tensorrt_0_pred_semantic"], second["pred_semantic"])

@pytest.mark.parametrize("stop_token", [1024, 44])
def test_deployment_reference_matches_sampling_and_cache_boundaries(
    monkeypatch: pytest.MonkeyPatch, stop_token: int,
) -> None:
    import torch
    import export_onnx
    import run_trt_inference

    seen = []

    def encoder_factory(model: object, *, max_len: int):
        assert max_len == 8

        def encoder(phones: object, prompts: object, bert: object):
            return (
                torch.ones(1, 1), torch.tensor([[42]]),
                torch.zeros(1, 1, 8, 1), torch.zeros(1, 1, 8, 1),
                torch.tensor([3]), torch.tensor([1]),
            )
        return encoder

    def step_factory(model: object):
        def step(samples, keys, values, x_len, y_len, index):
            seen.append((int(samples.item()), int(index.item())))
            token = 43 if len(seen) == 1 else stop_token
            return torch.ones(1, 1), torch.tensor([[token]]), keys, values
        return step

    def sampler(values, indices, *, temperature, top_k, top_p):
        assert values.device.type == indices.device.type == "cpu"
        assert (temperature, top_k, top_p) == (1.0, 15, 1.0)
        return indices[:, :1]

    monkeypatch.setattr(export_onnx, "GPTEncoder", encoder_factory)
    monkeypatch.setattr(export_onnx, "GPTStep", step_factory)
    monkeypatch.setattr(run_trt_inference, "sample_topk", sampler)
    infer = worker._deployment_semantic_reference(object(), 8)
    result, count = infer(
        torch.tensor([[1, 2, 3]]), torch.tensor([3]), torch.tensor([[9]]),
        torch.zeros(1, 1024, 3), top_k=15, top_p=1.0, temperature=1.0,
        early_stop_num=1500,
    )
    assert result.tolist() == ([[9, 42, 43]] if stop_token == 1024 else [[9, 42, 43, 44, 44]])
    assert count == (2 if stop_token == 1024 else 3)
    assert seen[:2] == [(42, 0), (43, 1)]

def test_semantic_comparison_orders_input_casts_on_the_trt_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types
    import numpy as np
    import torch
    import onnxruntime
    import run_trt_inference

    current_stream = object()
    original_to = torch.Tensor.to

    def cpu_cuda_stub(tensor, *args, **kwargs):
        if args == ("cuda",):
            return tensor
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_cuda_stub)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current_stream)
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CUDAExecutionProvider"])
    monkeypatch.setattr(worker, "_object", lambda *args: {})
    monkeypatch.setattr(worker, "select_engine_dir", lambda *args: tmp_path)

    def names(is_step):
        return (
            ["samples", "k_cache", "v_cache", "x_len", "y_len", "idx"] if is_step
            else ["phoneme_ids", "prompts", "bert_feature"]
        )

    def outputs(is_step):
        data = {
            "topk_values": np.array([[1.0]], dtype=np.float32),
            "topk_indices": np.array([[43]], dtype=np.int64),
            "k_cache_new" if is_step else "k_cache": np.zeros((1, 1, 80, 2), np.float32),
            "v_cache_new" if is_step else "v_cache": np.zeros((1, 1, 80, 2), np.float32),
        }
        if not is_step:
            data.update(x_len=np.array([50]), y_len=np.array([20]))
        return data

    class Session:
        def __init__(self, path, **kwargs):
            self.is_step = "gpt_step" in path

        def get_inputs(self):
            return [
                types.SimpleNamespace(name=n, type="tensor(float)" if n == "bert_feature"
                                      or "cache" in n else "tensor(int64)")
                for n in names(self.is_step)
            ]

        def get_provider_options(self):
            return {"CUDAExecutionProvider": {"use_tf32": "0"}}

        def get_outputs(self):
            return [types.SimpleNamespace(name=n) for n in outputs(self.is_step)]

        def run(self, output_names, inputs):
            data = outputs(self.is_step)
            return [data[n] for n in output_names]

    class TRT:
        def __init__(self, path, *, device, stream):
            assert stream is current_stream
            self.is_step = "gpt_step" in path
            self.input_names = names(self.is_step)

        def __call__(self, inputs):
            return {k: torch.from_numpy(v) for k, v in outputs(self.is_step).items()}

    monkeypatch.setattr(onnxruntime, "InferenceSession", Session)
    monkeypatch.setattr(run_trt_inference, "TRTModule", TRT)
    report = worker._onnx_trt_semantic_parity(tmp_path)
    assert report["greedy_sequence_agreement"] == 1.0
    assert report["top1_agreement"] == 1.0
    assert len(report["steps"]) == 32
    assert report["stream_contract"] == "torch-current-stream-inputs-and-tensorrt"

def test_encoder_evidence_records_prompt_mismatch(tmp_path: Path) -> None:
    import numpy as np

    first = {
        "phoneme_ids": np.array([[1, 2]]),
        "prompts": np.array([[3, 4, 5]]),
        "bert_feature": np.zeros((1, 4, 2)),
    }
    second = dict(first, prompts=np.array([[3, 9, 5]]))
    path = tmp_path / "encoder-inputs.npz"
    evidence = worker._paired_input_evidence(
        [first], [second], path, ("phoneme_ids", "prompts", "bert_feature")
    )
    assert evidence["segments"][0]["phoneme_ids"]["exact"] is True
    assert evidence["segments"][0]["bert_feature"]["exact"] is True
    assert evidence["segments"][0]["prompts"]["exact"] is False
    assert "decoder_random_draws_aligned" not in evidence
    with np.load(path) as inputs:
        np.testing.assert_array_equal(inputs["tensorrt_0_prompts"], second["prompts"])

@pytest.mark.parametrize("rates", [(32000, 16000), (44100, 16000), (16000, 32000)])
def test_qa_reference_resampler_matches_deployment(rates) -> None:
    import librosa
    import numpy as np
    import torch

    generator = torch.Generator().manual_seed(1234)
    waveform = torch.randn((2, 1001), generator=generator)
    expected = np.stack([
        librosa.resample(row, orig_sr=rates[0], target_sr=rates[1], res_type="soxr_hq")
        for row in waveform.numpy()
    ])
    actual = worker._deployment_reference_resampler(*rates)(waveform)
    assert actual.dtype == waveform.dtype
    assert actual.device == waveform.device
    np.testing.assert_array_equal(actual.numpy(), expected)
