from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .model_package import select_engine_dir
from .workstation_evaluation import (
    _TensorRTSpeakerEmbedder,
    _spectral_quality,
    spoken_content_error_rate,
)
from .workstation_production_gates import conversion_parity_report


class ConversionParityWorkerError(RuntimeError):
    pass


_CASES = {
    "ja": (
        "今日はいい天気ですね。",
        "この方法で本当に大丈夫ですか？",
        "ゆっくり息をして、もう一度始めましょう。",
        "明日の予定を確認したあと、必要な資料を準備します。",
        "長い言葉でも音を省かず、最後まではっきり話してください。",
    ),
    "zh": (
        "今天的天氣很好。",
        "這個方法真的沒有問題嗎？",
        "請慢慢呼吸，然後再開始一次。",
        "確認明天的行程後，我們會準備所需資料。",
        "即使句子較長，也要清楚完整地說到最後。",
    ),
    "yue": (
        "今日天氣真係幾好。",
        "呢個方法真係冇問題咩？",
        "慢慢呼吸，然後再開始一次。",
        "確認完聽日行程，我哋就準備需要嘅資料。",
        "就算句子比較長，都要清楚完整咁講到最後。",
    ),
    "en": (
        "The weather is pleasant today.",
        "Are you sure this method is reliable?",
        "Take a slow breath, and then begin again.",
        "After checking tomorrow's schedule, we will prepare the required material.",
        "Please pronounce every sound clearly, even when the sentence is unusually long.",
    ),
    "ko": (
        "오늘은 날씨가 정말 좋네요.",
        "이 방법이 정말 괜찮은가요?",
        "천천히 숨을 쉬고 다시 시작해 봅시다.",
        "내일 일정을 확인한 뒤 필요한 자료를 준비하겠습니다.",
        "문장이 길어도 모든 소리를 끝까지 분명하게 말해 주세요.",
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ConversionParityWorkerError(f"{label} is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConversionParityWorkerError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise ConversionParityWorkerError(f"{label} is malformed")
    return value


def _directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ConversionParityWorkerError(f"{label} directory is missing")
    return path.resolve(strict=True)


def _cosine(left: Any, right: Any) -> float:
    first = np.asarray(left, dtype=np.float64).reshape(-1)
    second = np.asarray(right, dtype=np.float64).reshape(-1)
    if first.shape != second.shape or not first.size:
        raise ConversionParityWorkerError("parity tensors have incompatible shapes")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 1.0 if np.array_equal(first, second) else 0.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def _pytorch_onnx_evidence(package: Path) -> tuple[float, list[dict[str, Any]]]:
    path = package / "onnx" / "pytorch-onnx-validation.json"
    if path.is_symlink() or not path.is_file():
        raise ConversionParityWorkerError("PyTorch to ONNX validation evidence is missing")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConversionParityWorkerError("PyTorch to ONNX evidence is unreadable") from error
    if not isinstance(rows, list) or not rows:
        raise ConversionParityWorkerError("PyTorch to ONNX evidence is malformed")
    required = {"GPTEncoder", "GPTStep", "SoVITS", "SVEmbedding"}
    observed = {
        str(row.get("model_name"))
        for row in rows
        if isinstance(row, Mapping) and row.get("passed") is True
    }
    if not required.issubset(observed) or any(
        not isinstance(row, Mapping) or row.get("passed") is not True for row in rows
    ):
        raise ConversionParityWorkerError("PyTorch to ONNX evidence did not pass every stage")
    logits: list[float] = []
    stage_cases: dict[str, set[str | None]] = {}
    for stage in ("GPTEncoder", "GPTStep"):
        stage_rows = [row for row in rows if row.get("model_name") == stage]
        case_values = [row.get("case_id") for row in stage_rows]
        if any(case is not None and not isinstance(case, str) for case in case_values):
            raise ConversionParityWorkerError("GPT validation case identifier is invalid")
        cases = set(case_values)
        if None in cases and len(cases) != 1:
            raise ConversionParityWorkerError("GPT validation cases mix legacy and named probes")
        observed: set[str | None] = set()
        for row in stage_rows:
            if row.get("output_name") != "topk_values":
                continue
            case = row.get("case_id")
            if case in observed:
                raise ConversionParityWorkerError("GPT logits evidence contains duplicate cases")
            observed.add(case)
            metrics = row.get("metrics")
            value = metrics.get("cosine_similarity") if isinstance(metrics, Mapping) else None
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not -1.0 <= value <= 1.0
            ):
                raise ConversionParityWorkerError("GPT logits cosine evidence is invalid")
            logits.append(float(value))
        if not observed or observed != cases:
            raise ConversionParityWorkerError("GPT PyTorch to ONNX logits evidence is incomplete")
        stage_cases[stage] = observed
    if stage_cases["GPTEncoder"] != stage_cases["GPTStep"]:
        raise ConversionParityWorkerError("GPT validation cases do not match between stages")
    return min(logits), [dict(row) for row in rows]


def _ort_array(meta: Any, values: np.ndarray) -> np.ndarray:
    kind = str(meta.type)
    if "float16" in kind:
        return values.astype(np.float16)
    if "float" in kind:
        return values.astype(np.float32)
    if "int64" in kind:
        return values.astype(np.int64)
    if "int32" in kind:
        return values.astype(np.int32)
    raise ConversionParityWorkerError(f"unsupported ONNX tensor type: {kind}")


def _onnx_trt_semantic_parity(package: Path) -> dict[str, Any]:
    try:
        import onnxruntime
        import torch
        from run_trt_inference import TRTModule
    except ImportError as error:
        raise ConversionParityWorkerError("ONNX/TensorRT parity dependencies are missing") from error
    providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise ConversionParityWorkerError("ONNX Runtime CUDA provider is unavailable")
    manifest = _object(package / "manifest.json", "model manifest")
    engine_dir = select_engine_dir(package, manifest)
    encoder_ort = onnxruntime.InferenceSession(
        str(package / "onnx" / "gpt_encoder.onnx"),
        providers=[("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"],
    )
    step_ort = onnxruntime.InferenceSession(
        str(package / "onnx" / "gpt_step.onnx"),
        providers=[("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"],
    )
    for session in (encoder_ort, step_ort):
        options = session.get_provider_options().get("CUDAExecutionProvider", {})
        if str(options.get("use_tf32")) != "0":
            raise ConversionParityWorkerError("ONNX CUDA parity requires TF32 disabled")
    # Input transfers and automatic dtype casts happen on PyTorch's current
    # stream. Execute TensorRT there too so these producer operations are ordered
    # before inference, as they are in the production inference stream context.
    comparison_stream = torch.cuda.current_stream()
    encoder_trt = TRTModule(
        str(engine_dir / "gpt_encoder.engine"), device="cuda", stream=comparison_stream
    )
    step_trt = TRTModule(
        str(engine_dir / "gpt_step.engine"), device="cuda", stream=comparison_stream
    )
    rng = np.random.default_rng(1234)
    base = {
        "phoneme_ids": rng.integers(0, 512, (1, 50), dtype=np.int64),
        "prompts": rng.integers(0, 1024, (1, 20), dtype=np.int64),
        "bert_feature": rng.standard_normal((1, 1024, 50)).astype(np.float32),
        "phoneme_ids_len": np.asarray([50], dtype=np.int64),
    }
    ort_inputs = {
        meta.name: _ort_array(meta, base[meta.name]) for meta in encoder_ort.get_inputs()
    }
    ort_names = [meta.name for meta in encoder_ort.get_outputs()]
    ort_values = encoder_ort.run(ort_names, ort_inputs)
    ort_state = dict(zip(ort_names, ort_values, strict=True))
    trt_inputs = {
        name: torch.from_numpy(base[name].copy()).to("cuda")
        for name in encoder_trt.input_names
    }
    trt_state_raw = encoder_trt(trt_inputs)
    trt_state = {
        name: value.detach().cpu().numpy() for name, value in trt_state_raw.items()
    }
    encoder_cosine = _cosine(ort_state["topk_values"], trt_state["topk_values"])
    top1_matches = int(
        np.asarray(ort_state["topk_indices"])[..., 0].item()
        == np.asarray(trt_state["topk_indices"])[..., 0].item()
    )
    ort_sequence: list[int] = []
    trt_sequence: list[int] = []
    logits_cosines = [encoder_cosine]
    step_evidence: list[dict[str, Any]] = []
    for index in range(32):
        ort_sample = np.asarray(ort_state["topk_indices"])[..., :1].astype(np.int64)
        trt_sample = np.asarray(trt_state["topk_indices"])[..., :1].astype(np.int64)
        ort_sequence.append(int(ort_sample.reshape(-1)[0]))
        trt_sequence.append(int(trt_sample.reshape(-1)[0]))
        ort_step_base = {
            "samples": ort_sample,
            "k_cache": np.asarray(ort_state["k_cache"]),
            "v_cache": np.asarray(ort_state["v_cache"]),
            "x_len": np.asarray(ort_state["x_len"]),
            "y_len": np.asarray(ort_state["y_len"]),
            "idx": np.asarray([index], dtype=np.int64),
        }
        ort_step_inputs = {
            meta.name: _ort_array(meta, ort_step_base[meta.name])
            for meta in step_ort.get_inputs()
        }
        ort_step_names = [meta.name for meta in step_ort.get_outputs()]
        ort_step_values = step_ort.run(ort_step_names, ort_step_inputs)
        ort_step = dict(zip(ort_step_names, ort_step_values, strict=True))
        trt_step_base = {
            "samples": trt_sample,
            "k_cache": np.asarray(trt_state["k_cache"]),
            "v_cache": np.asarray(trt_state["v_cache"]),
            "x_len": np.asarray(trt_state["x_len"]),
            "y_len": np.asarray(trt_state["y_len"]),
            "idx": np.asarray([index], dtype=np.int64),
        }
        trt_step_inputs = {
            name: torch.from_numpy(trt_step_base[name].copy()).to("cuda")
            for name in step_trt.input_names
        }
        trt_step_raw = step_trt(trt_step_inputs)
        trt_step = {
            name: value.detach().cpu().numpy() for name, value in trt_step_raw.items()
        }
        logits_cosines.append(
            _cosine(ort_step["topk_values"], trt_step["topk_values"])
        )
        step_evidence.append({
            "step": index,
            "logits_cosine": logits_cosines[-1],
            "onnx_input_token": int(ort_sample.reshape(-1)[0]),
            "tensorrt_input_token": int(trt_sample.reshape(-1)[0]),
            "onnx_top1": int(np.asarray(ort_step["topk_indices"])[..., 0].item()),
            "tensorrt_top1": int(np.asarray(trt_step["topk_indices"])[..., 0].item()),
            "onnx_top5_ids": np.asarray(ort_step["topk_indices"]).reshape(-1)[:5].tolist(),
            "tensorrt_top5_ids": np.asarray(trt_step["topk_indices"]).reshape(-1)[:5].tolist(),
            "onnx_top5_values": np.asarray(ort_step["topk_values"]).reshape(-1)[:5].tolist(),
            "tensorrt_top5_values": np.asarray(trt_step["topk_values"]).reshape(-1)[:5].tolist(),
        })
        # Isolate conversion error from accumulated autoregressive differences:
        # both backends receive the identical ONNX-prefix token and caches.
        same_inputs = {
            name: torch.from_numpy(ort_step_inputs[name].copy()).to("cuda")
            for name in step_trt.input_names
        }
        same_raw = step_trt(same_inputs)
        same = {
            name: value.detach().cpu().numpy().copy()
            for name, value in same_raw.items()
        }
        step_evidence[-1]["same_input_logits_cosine"] = _cosine(
            ort_step["topk_values"], same["topk_values"]
        )
        step_evidence[-1]["same_input_top5_ids"] = np.asarray(
            same["topk_indices"]
        ).reshape(-1)[:5].tolist()
        step_evidence[-1]["same_input_top5_values"] = np.asarray(
            same["topk_values"]
        ).reshape(-1)[:5].tolist()
        top1_matches += int(
            np.asarray(ort_step["topk_indices"])[..., 0].item()
            == np.asarray(trt_step["topk_indices"])[..., 0].item()
        )
        ort_state = {
            "topk_values": ort_step["topk_values"],
            "topk_indices": ort_step["topk_indices"],
            "k_cache": ort_step["k_cache_new"],
            "v_cache": ort_step["v_cache_new"],
            "x_len": ort_state["x_len"],
            "y_len": ort_state["y_len"],
        }
        trt_state = {
            "topk_values": trt_step["topk_values"],
            "topk_indices": trt_step["topk_indices"],
            "k_cache": trt_step["k_cache_new"],
            "v_cache": trt_step["v_cache_new"],
            "x_len": trt_state["x_len"],
            "y_len": trt_state["y_len"],
        }
    return {
        "onnx_cuda_use_tf32": False,
        "stream_contract": "torch-current-stream-inputs-and-tensorrt",
        "steps": step_evidence,
        "encoder_logits_cosine": encoder_cosine,
        "minimum_logits_cosine": min(logits_cosines),
        "top1_agreement": top1_matches / 33.0,
        "greedy_sequence_agreement": float(ort_sequence == trt_sequence),
        "onnx_sequence": ort_sequence,
        "tensorrt_sequence": trt_sequence,
    }


def _transcribe(model: Any, path: Path, language: str) -> str:
    segments, _ = model.transcribe(
        str(path),
        language="zh" if language == "yue" else language,
        beam_size=5,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    return "".join(segment.text for segment in segments).strip()


def _write_audio(path: Path, audio: Any, sample_rate: int) -> np.ndarray:
    try:
        from scipy.io import wavfile
    except ImportError as error:
        raise ConversionParityWorkerError(
            "conversion parity audio evidence requires scipy"
        ) from error
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ConversionParityWorkerError("conversion parity generated invalid audio")
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, np.clip(values, -1.0, 1.0))
    return values


def _deployment_semantic_reference(
    t2s_model: Any, max_len: int, observer: Any = None, generator: Any = None,
) -> Any:
    """Use original PyTorch weights with the exported deployment sampling loop."""
    import torch
    from export_onnx import GPTEncoder, GPTStep
    from run_trt_inference import sample_topk

    encoder, step = GPTEncoder(t2s_model, max_len=max_len), GPTStep(t2s_model)

    @torch.no_grad()
    def infer(
        phoneme_ids: Any, phoneme_lengths: Any, prompts: Any, bert: Any,
        *, top_k: int, top_p: float, temperature: float, early_stop_num: int,
    ) -> tuple[Any, int]:
        del phoneme_lengths, early_stop_num
        values, indices, k_cache, v_cache, x_len, y_len = encoder(
            phoneme_ids, prompts, bert
        )
        device = phoneme_ids.device

        def draw(logits: Any, tokens: Any) -> Any:
            values_cpu, tokens_cpu = logits.detach().cpu(), tokens.detach().cpu()
            rng_state = (generator.get_state() if generator is not None else torch.get_rng_state()) if observer is not None else None
            controls = {"generator": generator} if generator is not None else {}
            sampled = sample_topk(
                values_cpu, tokens_cpu, temperature=temperature, top_k=top_k, top_p=top_p,
                **controls,
            )
            if observer is not None:
                observer(values_cpu, tokens_cpu, rng_state, sampled)
            return sampled.to(device)

        current = draw(values, indices)
        generated = [current]
        available = k_cache.shape[2] - int(x_len.item() + y_len.item()) - 1
        maximum_steps = min(1000, available) if available > 0 else 1
        steps = 0
        for index in range(maximum_steps):
            values, indices, k_cache, v_cache = step(
                current.to(torch.int64), k_cache, v_cache,
                x_len.to(device), y_len.to(device),
                torch.tensor([index], dtype=torch.int64, device=device),
            )
            current = draw(values, indices)
            generated.append(current)
            steps += 1
            if int(current[0, 0]) == 1024:
                break
        tokens = torch.cat(generated, dim=1)
        if int(tokens[0, -1]) == 1024:
            tokens = tokens[:, :-1]
        return torch.cat((prompts, tokens), dim=1), steps

    return infer


def _capture_decode_inputs(values: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy().copy()
        for name, value in values.items()
    }


class _DecodeInputRecorder:
    """Observe a TensorRT module without changing its inputs or outputs."""

    def __init__(self, module: Any) -> None:
        self.module = module
        self.calls: list[dict[str, np.ndarray]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module, name)

    def __call__(self, inputs: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
        self.calls.append(_capture_decode_inputs(inputs))
        return self.module(inputs, *args, **kwargs)


def _paired_input_evidence(
    pytorch: list[dict[str, np.ndarray]],
    tensorrt: list[dict[str, np.ndarray]],
    path: Path,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    for backend, calls in (("pytorch", pytorch), ("tensorrt", tensorrt)):
        for index, call in enumerate(calls):
            for name, value in call.items():
                arrays[f"{backend}_{index}_{name}"] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    segments = []
    for first, second in zip(pytorch, tensorrt):
        row: dict[str, Any] = {}
        for name in fields:
            left, right = first[name], second[name]
            entry: dict[str, Any] = {
                "pytorch_shape": list(left.shape),
                "tensorrt_shape": list(right.shape),
                "exact": bool(np.array_equal(left, right)),
            }
            if left.shape == right.shape and left.size:
                entry["cosine"] = _cosine(left, right)
                entry["max_abs_difference"] = float(
                    np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
                )
            row[name] = entry
        segments.append(row)
    return {
        "pytorch_segments": len(pytorch),
        "tensorrt_segments": len(tensorrt),
        "segments": segments,
        "inputs_sha256": _sha256_file(path),
    }


def _decode_input_evidence(pytorch, tensorrt, path):
    fields = ("pred_semantic", "text_seq", "refer_spec", "sv_emb")
    shared_noise = bool(pytorch and tensorrt) and all(
        "acoustic_noise" in row for row in [*pytorch, *tensorrt]
    )
    if shared_noise:
        fields += ("acoustic_noise",)
    evidence = _paired_input_evidence(pytorch, tensorrt, path, fields)
    evidence["decoder_random_draws_aligned"] = shared_noise and len(pytorch) == len(tensorrt) and all(
        np.array_equal(left["acoustic_noise"], right["acoustic_noise"])
        for left, right in zip(pytorch, tensorrt)
    )
    return evidence


def _deployment_reference_resampler(orig_freq: int, new_freq: int) -> Any:
    """Match deployment's librosa/soxr waveform input in conversion QA only."""
    import librosa
    import torch

    class Resampler(torch.nn.Module):
        def forward(self, waveform: Any) -> Any:
            values = waveform.detach().cpu().float().numpy()
            rows = values.reshape(-1, values.shape[-1])
            resampled = np.stack([
                librosa.resample(
                    row, orig_sr=orig_freq, target_sr=new_freq, res_type="soxr_hq"
                )
                for row in rows
            ])
            shape = (*values.shape[:-1], resampled.shape[-1])
            return torch.from_numpy(resampled.reshape(shape)).to(
                device=waveform.device, dtype=waveform.dtype
            )

    return Resampler()



def _shared_acoustic_noise(tokens, generator):
    import torch

    return torch.randn(
        (1, 192, 2 * int(tokens.shape[-1])), dtype=torch.float32, generator=generator
    ).to(torch.float16)


class _FullSpanStreamDecoder:
    """Use the production native decoder with explicit nonzero noise for QA."""

    input_names = ("pred_semantic", "text_seq", "refer_spec", "sv_emb", "noise_scale", "speed")

    def __init__(self, module, generator, hop_length=640, captured_noise=None):
        self.hop_length = int(hop_length)
        self.captured_noise = captured_noise
        self.module = module
        self.generator = generator
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.module, name)

    def __call__(self, inputs):
        import torch

        tokens = inputs["pred_semantic"]
        device = tokens.device
        overlap = int(self.module.engine.get_tensor_shape("overlap_frames")[-1])
        values = dict(inputs)
        values.update({
            "acoustic_noise": (self.captured_noise if self.captured_noise is not None
                               else _shared_acoustic_noise(tokens, self.generator)).to(device),
            "result_length": torch.tensor([tokens.shape[-1]], dtype=torch.int64, device=device),
            "overlap_frames": torch.zeros(
                (1, 192, overlap), dtype=self.module.tensor_dtype["overlap_frames"], device=device
            ),
            "overlap_enabled": torch.zeros(1, dtype=torch.float32, device=device),
        })
        values = {name: value for name, value in values.items() if name in self.module.input_names}
        self.calls.append(_capture_decode_inputs(values))
        frames = 2 * int(tokens.shape[-1])
        outputs = {
            "audio": torch.empty(
                (1, 1, frames * self.hop_length),
                dtype=self.module.tensor_dtype["audio"], device=device,
            ),
            "latent": torch.empty(
                (1, 192, frames), dtype=self.module.tensor_dtype["latent"], device=device,
            ),
            "latent_mask": torch.empty(
                (1, 1, frames), dtype=self.module.tensor_dtype["latent_mask"], device=device,
            ),
        }
        return self.module(values, outputs=outputs)


def _record_sample(trace, values, indices, rng_state, sampled):
    trace.append({
        "values": values.detach().cpu().numpy().copy(),
        "indices": indices.detach().cpu().numpy().copy(),
        "rng_state": rng_state.detach().cpu().numpy().copy(),
        "sampled": sampled.detach().cpu().numpy().copy(),
    })


def _sampling_evidence(first, second, path):
    arrays = {}
    for backend, rows in (("pytorch", first), ("tensorrt", second)):
        for index, row in enumerate(rows):
            for name, values in row.items():
                arrays[f"{backend}_{index}_{name}"] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    first_difference = None
    for index, (left, right) in enumerate(zip(first, second)):
        if not np.array_equal(left["sampled"], right["sampled"]):
            first_difference = {
                "step": index,
                "same_rng_state": bool(np.array_equal(left["rng_state"], right["rng_state"])),
                "pytorch_token": int(left["sampled"].reshape(-1)[0]),
                "tensorrt_token": int(right["sampled"].reshape(-1)[0]),
                "pytorch_top5_ids": left["indices"].reshape(-1)[:5].tolist(),
                "tensorrt_top5_ids": right["indices"].reshape(-1)[:5].tolist(),
                "pytorch_top5_values": left["values"].reshape(-1)[:5].tolist(),
                "tensorrt_top5_values": right["values"].reshape(-1)[:5].tolist(),
            }
            break
    return {
        "pytorch_draws": len(first), "tensorrt_draws": len(second),
        "first_difference": first_difference, "trace_sha256": _sha256_file(path),
    }


def _release_unused_cuda():
    import gc
    import torch

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

def _complete_output_cases(
    *,
    package: Path,
    selected: Path,
    shared: Path,
    asr_model: Path,
    output: Path,
    semantic: Mapping[str, Any],
    pytorch_onnx_cosine: float,
    case_specs: list[tuple[str, str]] | None = None,
    include_native_pytorch: bool = False,
    native_only: bool = False,
    replay_native_decoder: bool = False,
) -> tuple[list[dict[str, Any]], list[Path]]:
    try:
        import torch
        from faster_whisper import WhisperModel
        from run_inference import GPTSoVITSInference
        from run_trt_inference import GPTSoVITS_TRT_Inference
        from scipy.io import wavfile
    except ImportError as error:
        raise ConversionParityWorkerError("complete conversion QA dependencies are missing") from error
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    deployment = _object(
        selected / "selected" / "deployment-checkpoints.json", "deployment checkpoints"
    )
    manifest = _object(package / "manifest.json", "model manifest")
    profile_id = str(manifest.get("default_voice_profile") or "")
    profile_root = package / "voices" / profile_id
    profile = _object(profile_root / "profile.json", "voice profile")
    reference = profile_root / str(profile.get("reference_audio") or "")
    if reference.is_symlink() or not reference.is_file():
        raise ConversionParityWorkerError("deployment reference audio is missing")
    reference_text = str(profile.get("reference_text") or "").strip()
    language = str(profile.get("reference_language") or "").casefold()
    if not reference_text or language not in _CASES:
        raise ConversionParityWorkerError("deployment reference metadata is invalid")
    reference_language = language
    target_cases = case_specs if case_specs is not None else [
        (reference_language, text) for text in _CASES[reference_language]
    ]

    import run_inference
    import run_trt_inference
    from unittest.mock import patch

    progress_path = output / "conversion-progress.json"
    paths: list[Path] = [progress_path]

    def progress(phase, count):
        progress_path.write_text(json.dumps({
            "schema": "aniflive-conversion-progress-v1",
            "phase": phase, "completed_cases": count,
            "qualification_complete": False,
        }, indent=2))

    def pytorch_phase():
        with patch.object(run_inference, "is_half", False):
            engine = GPTSoVITSInference(
                str(selected / "selected" / str(deployment["gpt"]["relative_path"])),
                str(selected / "selected" / str(deployment["sovits"]["relative_path"])),
                str(shared / "chinese-hubert-base"),
                str(shared / "chinese-roberta-wwm-ext-large"),
                str(shared / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"),
            )
        encoder_inputs, decode_inputs, samples = [], [], []
        sample_generator = torch.Generator(device="cpu")
        acoustic_generator = torch.Generator(device="cpu")
        config = _object(package / "onnx" / "config.json", "export configuration")

        def observe(values, indices, rng, sampled):
            _record_sample(samples, values, indices, rng, sampled)

        semantic_reference = _deployment_semantic_reference(
            engine.t2s_model, int(config["data"]["max_len"]), observe, sample_generator
        )
        original_panel = engine.t2s_model.model.infer_panel
        original_decode = engine.vq_model.decode

        def encoder(*args, **kwargs):
            encoder_inputs.append(_capture_decode_inputs({
                "phoneme_ids": args[0], "prompts": args[2], "bert_feature": args[3],
            }))
            return semantic_reference(*args, **kwargs)

        def decode(*args, **kwargs):
            tokens, reference_spec = args[0], args[2][0]
            noise = _shared_acoustic_noise(tokens, acoustic_generator).to(
                device=tokens.device, dtype=reference_spec.dtype
            )
            decode_inputs.append(_capture_decode_inputs({
                "pred_semantic": tokens, "text_seq": args[1],
                "refer_spec": reference_spec, "sv_emb": kwargs["sv_emb"][0],
                "acoustic_noise": noise,
            }))
            kwargs.setdefault("noise_scale", 0.5)
            return engine.vq_model.decode_streaming(
                *args, **kwargs, result_length=int(tokens.shape[-1]),
                overlap_frames=torch.zeros(
                    (1, 192, int(config["streaming"]["overlap_frames"])),
                    device=tokens.device, dtype=reference_spec.dtype,
                ),
                overlap_enabled=torch.zeros(1, device=tokens.device),
                acoustic_noise=noise,
            )[0]

        engine.t2s_model.model.infer_panel = encoder
        engine.vq_model.decode = decode
        cases = [
            {"index": index, "language": language, "text": text}
            for index, (language, text) in enumerate(target_cases, 1)
        ] if native_only else []
        try:
            for index, (language, text) in enumerate([] if native_only else target_cases, 1):
                encoder_inputs.clear()
                decode_inputs.clear()
                samples.clear()
                sample_generator.manual_seed(1234)
                acoustic_generator.manual_seed(1234 ^ 0x41C64E6D)
                np.random.seed(1234)
                torch.manual_seed(1234)
                torch.cuda.manual_seed_all(1234)
                with patch.object(
                    run_inference.torchaudio.transforms, "Resample",
                    _deployment_reference_resampler,
                ):
                    audio, rate = engine.infer(
                        str(reference), reference_text, reference_language, text, language,
                        top_k=15, top_p=1.0, temperature=1.0,
                    )
                path = output / "audio" / f"case-{index}-pytorch.wav"
                values = _write_audio(path, audio, int(rate))
                paths.append(path)
                cases.append({
                    "index": index, "text": text, "language": language, "pytorch_path": path,
                    "pytorch_values": values, "pytorch_encoder": list(encoder_inputs),
                    "pytorch_decoder": list(decode_inputs), "pytorch_samples": list(samples),
                })
                progress("pytorch-reference", index)
            if include_native_pytorch:
                engine.t2s_model.model.infer_panel = original_panel
                engine.vq_model.decode = original_decode
                native_calls = []
                def capture_native_decode(*args, **kwargs):
                    record = _capture_decode_inputs({
                        "pred_semantic": args[0], "text_seq": args[1],
                        "refer_spec": args[2][0], "sv_emb": kwargs["sv_emb"][0],
                        "noise_scale": torch.tensor([float(kwargs.get("noise_scale", 0.5))]),
                        "speed": torch.tensor([float(kwargs.get("speed", 1.0))]),
                    })
                    draws = []
                    original_randn = torch.randn_like
                    def draw(*draw_args, **draw_kwargs):
                        value = original_randn(*draw_args, **draw_kwargs)
                        if value.ndim == 3 and value.shape[1] == 192:
                            draws.append(value.detach().cpu().numpy().copy())
                        return value
                    with patch.object(torch, "randn_like", draw):
                        result = original_decode(*args, **kwargs)
                    if len(draws) != 1:
                        raise ConversionParityWorkerError("Native acoustic-noise capture contract changed")
                    record["acoustic_noise"] = draws[0]
                    native_calls.append(record)
                    return result
                if replay_native_decoder:
                    engine.vq_model.decode = capture_native_decode
                for data in cases:
                    native_calls.clear()
                    np.random.seed(1234)
                    torch.manual_seed(1234)
                    torch.cuda.manual_seed_all(1234)
                    audio, rate = engine.infer(
                        str(reference), reference_text, reference_language,
                        data["text"], data["language"],
                        top_k=15, top_p=1.0, temperature=1.0,
                    )
                    path = output / "audio" / f"case-{data['index']}-pytorch-native.wav"
                    _write_audio(path, audio, int(rate))
                    paths.append(path)
                    data["native_pytorch_path"] = path
                    if replay_native_decoder:
                        data["native_decoder_calls"] = list(native_calls)
                        capture_path = output / "diagnostics" / f"case-{data['index']}-native-decoder.npz"
                        capture_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez(capture_path, **{
                            f"segment_{number}_{name}": value
                            for number, record in enumerate(native_calls)
                            for name, value in record.items()
                        })
                        paths.append(capture_path)
                        data["native_decoder_capture_path"] = capture_path
                    progress("pytorch-native-listening", data["index"])
        finally:
            engine.t2s_model.model.infer_panel = original_panel
            engine.vq_model.decode = original_decode
            for name in ("ssl_model", "bert_model", "t2s_model", "vq_model"):
                getattr(engine, name).to("cpu")
            engine.sv_model.embedding_model.to("cpu")
        return cases

    def tensorrt_phase(cases):
        engine = GPTSoVITS_TRT_Inference(
            str(select_engine_dir(package, manifest)),
            str(shared / "chinese-roberta-wwm-ext-large"), device="cuda",
        )
        encoder = _DecodeInputRecorder(engine.model_gpt_enc)
        acoustic_generator = torch.Generator(device="cpu")
        decoder = _FullSpanStreamDecoder(
            engine.model_sovits_stream, acoustic_generator, engine.hps["data"]["hop_length"]
        )
        engine.model_gpt_enc, engine.model_sovits = encoder, decoder
        samples = []
        original_sample = run_trt_inference.sample_topk
        sample_generator = torch.Generator(device="cpu")

        def sample(values, indices, *args, **kwargs):
            state = sample_generator.get_state()
            result = original_sample(values, indices, *args, generator=sample_generator, **kwargs)
            _record_sample(samples, values, indices, state, result)
            return result

        for data in cases:
            index, text, language = data["index"], data["text"], data["language"]
            encoder.calls.clear()
            decoder.calls.clear()
            samples.clear()
            sample_generator.manual_seed(1234)
            acoustic_generator.manual_seed(1234 ^ 0x41C64E6D)
            np.random.seed(1234)
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            path = output / "audio" / f"case-{index}-tensorrt.wav"
            with patch.object(run_trt_inference, "sample_topk", sample):
                engine.infer(
                    str(reference), reference_text, reference_language, text, language,
                    top_k=15, top_p=1.0, temperature=1.0, output_path=str(path),
                )
            rate, audio = wavfile.read(path)
            values = np.asarray(audio, dtype=np.float32).reshape(-1)
            if np.issubdtype(np.asarray(audio).dtype, np.integer):
                values /= 32768.0
            data.update(trt_path=path, trt_rate=rate, trt_values=values)
            paths.append(path)
            decoder_path = output / "diagnostics" / f"case-{index}-decode-inputs.npz"
            encoder_path = output / "diagnostics" / f"case-{index}-encoder-inputs.npz"
            sample_path = output / "diagnostics" / f"case-{index}-sampling.npz"
            data["decode_evidence"] = _decode_input_evidence(
                data["pytorch_decoder"], decoder.calls, decoder_path
            )
            data["encoder_evidence"] = _paired_input_evidence(
                data["pytorch_encoder"], encoder.calls, encoder_path,
                ("phoneme_ids", "prompts", "bert_feature"),
            )
            data["sampling_evidence"] = _sampling_evidence(
                data["pytorch_samples"], samples, sample_path
            )
            paths.extend((decoder_path, encoder_path, sample_path))
            progress("tensorrt-audio", index)

    _release_unused_cuda()
    cases = pytorch_phase()
    _release_unused_cuda()
    if replay_native_decoder:
        replay_engine = GPTSoVITS_TRT_Inference(
            str(select_engine_dir(package, manifest)),
            str(shared / "chinese-roberta-wwm-ext-large"), device="cuda",
        )
        for data in cases:
            wave_parts = []
            for record in data["native_decoder_calls"]:
                values = {
                    key: torch.from_numpy(value.copy()).to(replay_engine.device)
                    for key, value in record.items()
                }
                noise = values.pop("acoustic_noise")
                decoder = _FullSpanStreamDecoder(
                    replay_engine.model_sovits_stream, None,
                    replay_engine.hps["data"]["hop_length"], captured_noise=noise,
                )
                if noise.shape[-1] != 2 * values["pred_semantic"].shape[-1]:
                    raise ConversionParityWorkerError("Native noise and semantic lengths disagree")
                result = decoder(values)["audio"].detach().cpu().numpy().reshape(-1)
                peak = float(np.max(np.abs(result)))
                if peak > 1.0:
                    result = result / peak
                wave_parts.append(result)
            if not wave_parts:
                raise ConversionParityWorkerError("Native replay captured no decoder inputs")
            path = output / "audio" / f"case-{data['index']}-native-tokens-trt.wav"
            _write_audio(path, np.concatenate(wave_parts), int(replay_engine.hps["data"]["sampling_rate"]))
            data["native_replay_path"] = path
            paths.append(path)
        del replay_engine
        _release_unused_cuda()
    if native_only:
        return [
            {
                "case": data["index"], "language": data["language"], "text": data["text"],
                "paths": {
                    "pytorch_native": data["native_pytorch_path"].relative_to(output).as_posix(),
                    **({"native_tokens_tensorrt": data["native_replay_path"].relative_to(output).as_posix()}
                       if data.get("native_replay_path") else {}),
                },
                "audio_sha256": _sha256_file(data["native_pytorch_path"]),
                **({"decoder_replay": {
                    "semantic_source": "same native PyTorch decode invocation",
                    "acoustic_noise_source": "same native PyTorch decode invocation",
                    "new_semantic_generation": False,
                    "captured_inputs": data["native_decoder_capture_path"].relative_to(output).as_posix(),
                    "captured_inputs_sha256": _sha256_file(data["native_decoder_capture_path"]),
                    "segments": len(data["native_decoder_calls"]),
                }} if data.get("native_replay_path") else {}),
                "native_pytorch_contract": (
                    "original-checkpoint-float32-native-sampler-resampler-decoder"
                ),
                "seed": 1234,
            }
            for data in cases
        ], paths
    tensorrt_phase(cases)
    _release_unused_cuda()
    asr = WhisperModel(
        str(asr_model), device="cuda", device_index=0, compute_type="float16",
        local_files_only=True,
    )
    speaker = _TensorRTSpeakerEmbedder(package)
    rows: list[dict[str, Any]] = []
    for data in cases:
        index, text, language = data["index"], data["text"], data["language"]
        pytorch_path, trt_path = data["pytorch_path"], data["trt_path"]
        pytorch_values, trt_values = data["pytorch_values"], data["trt_values"]
        trt_rate = data["trt_rate"]
        quality = _spectral_quality(pytorch_path, trt_path)
        pytorch_hypothesis = _transcribe(asr, pytorch_path, language)
        trt_hypothesis = _transcribe(asr, trt_path, language)
        pytorch_content = spoken_content_error_rate(
            text, pytorch_hypothesis, language
        )
        trt_content = spoken_content_error_rate(text, trt_hypothesis, language)
        speaker_cosine = _cosine(speaker(pytorch_path), speaker(trt_path))
        clipping = float(np.mean(np.abs(trt_values) >= 0.999))
        rows.append(
            {
                "case": index,
                "decode_input_evidence": data["decode_evidence"],
                "encoder_input_evidence": data["encoder_evidence"],
                "sampling_evidence": data["sampling_evidence"],
                "reference_resampler": "librosa-soxr-hq-both-backends",
                "decoder_contract": "native-stream-full-span-shared-noise-v1",
                "text": text,
                "language": language,
                "seed": 1234,
                "sampling": {
                    "top_k": 15, "top_p": 1.0, "temperature": 1.0,
                    "repetition_penalty": 1.0,
                    "reference_contract": "exported-stage-loop-cpu-topk-v1",
                    "pytorch_reference_precision": "float32",
                    "generator": "independent-cpu-generator-per-case",
                },
                "pytorch_onnx_logits_cosine": pytorch_onnx_cosine,
                "onnx_trt_logits_cosine": float(semantic["minimum_logits_cosine"]),
                "onnx_trt_top1_agreement": float(semantic["top1_agreement"]),
                "greedy_sequence_agreement": float(
                    semantic["greedy_sequence_agreement"]
                ),
                "log_mel_cosine": float(quality["log_mel_cosine"]),
                "speaker_cosine": speaker_cosine,
                "duration_difference_ratio": float(
                    quality["duration_difference_ratio"]
                ),
                "pytorch_hypothesis": pytorch_hypothesis,
                "tensorrt_hypothesis": trt_hypothesis,
                "pytorch_content_error": pytorch_content,
                "tensorrt_content_error": trt_content,
                "content_regression": trt_content > pytorch_content + 0.02,
                "new_artifacts": bool(
                    not np.isfinite(trt_values).all()
                    or not trt_values.size
                    or clipping > 0.001
                    or trt_rate <= 0
                    or not pytorch_values.size
                ),
                "paths": {
                    "pytorch": pytorch_path.relative_to(output).as_posix(),
                    "tensorrt": trt_path.relative_to(output).as_posix(),
                },
            }
        )
        if data.get("native_pytorch_path") is not None:
            rows[-1]["paths"]["pytorch_native"] = (
                data["native_pytorch_path"].relative_to(output).as_posix()
            )
            rows[-1]["native_pytorch_contract"] = (
                "original-checkpoint-float32-native-sampler-resampler-decoder"
            )
        progress("quality-evaluation", index)
        (output / "partial-audio-cases.json").write_text(json.dumps(rows, indent=2))
    paths.append(output / "partial-audio-cases.json")
    return rows, paths


def _apply_content_review(
    cases: list[dict[str, Any]], review_path: Path, output: Path
) -> dict[str, Any]:
    """Adjudicate content only for the exact audio and prompt heard by a human."""
    if review_path.stat().st_size > 65536:
        raise ConversionParityWorkerError("content review exceeds 64 KiB")
    review = _object(review_path, "content review")
    entries = review.get("reviews")
    if (
        review.get("schema") != "aniflive-conversion-content-review-v1"
        or not isinstance(entries, list)
        or not entries
        or len(entries) > 100
    ):
        raise ConversionParityWorkerError("content review schema is invalid")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ConversionParityWorkerError("content review entry is invalid")
        digest = entry.get("audio_sha256")
        if (
            not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(entry.get("text"), str) or not entry["text"]
        ):
            raise ConversionParityWorkerError("content review audio identity is invalid")
    review_sha256 = _sha256_file(review_path)
    applied = []
    for case in cases:
        case["automated_content_regression"] = case["content_regression"]
        audio_path = (output / case["paths"]["tensorrt"]).resolve(strict=True)
        if not audio_path.is_relative_to(output.resolve()):
            raise ConversionParityWorkerError("reviewed audio escapes output")
        audio_sha256 = _sha256_file(audio_path)
        matches = [
            entry for entry in entries
            if entry["audio_sha256"] == audio_sha256 and entry["text"] == case["text"]
        ]
        if len(matches) != 1:
            continue
        entry = matches[0]
        if (
            entry.get("decision") != "content-complete"
            or not isinstance(entry.get("user_response"), str)
            or not entry["user_response"].strip()
            or not isinstance(entry.get("origin"), str)
            or not entry["origin"].strip()
        ):
            continue
        case["human_content_review"] = {
            **entry, "review_sha256": review_sha256,
            "scope": "content completeness only",
        }
        case["content_regression"] = False
        applied.append(case["case"])
    return {"sha256": review_sha256, "applied_cases": applied,
            "scope": "content completeness only"}


def _listening_cases(path: Path) -> list[tuple[str, str]]:
    if path.stat().st_size > 16384:
        raise ConversionParityWorkerError("listening plan exceeds 16 KiB")
    plan = _object(path, "listening plan")
    cases = plan.get("cases")
    if (
        plan.get("schema") != "aniflive-stage-listening-plan-v1"
        or not isinstance(cases, list) or not 1 <= len(cases) <= 10
    ):
        raise ConversionParityWorkerError("listening plan schema is invalid")
    result = []
    for case in cases:
        if not isinstance(case, dict):
            raise ConversionParityWorkerError("listening case is invalid")
        language, text = case.get("language"), case.get("text")
        if (
            not isinstance(language, str) or language not in _CASES or not isinstance(text, str)
            or not text.strip() or len(text) > 512 or "\x00" in text
        ):
            raise ConversionParityWorkerError("listening case language or text is invalid")
        result.append((language, text))
    return result


def _production_runtime_variants(
    package: Path, shared: Path, cases: list[tuple[str, str]], output: Path,
    sampling_contract: str | None = None,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Bounded diagnostic of the real API, never a change to production defaults."""
    import subprocess
    import tempfile

    from . import workstation_evaluation as evaluation

    model_id = str(_object(package / "manifest.json", "model manifest")["model_id"])
    records, artifacts = [], []
    if sampling_contract not in {None, "native-v2proplus-v1"}:
        raise ConversionParityWorkerError("Unsupported production sampling diagnostic")
    variants = (
        (("native-sampling", 1.35),) if sampling_contract
        else (("deployment-default", 1.0), ("native-repetition", 1.35))
    )
    for name, penalty in variants:
        root = output / "production-runtime" / name
        root.mkdir(parents=True)
        log_path = root / "api.log"
        with tempfile.TemporaryDirectory(prefix="aniflive-content-diagnostic-") as cache:
            environment = dict(os.environ)
            environment.update({
                "ANIFLIVE_TTS_REPETITION_PENALTY": str(penalty),
                "ANIFLIVE_TTS_SEMANTIC_SAMPLING": sampling_contract or "legacy-topk-v1",
                "ANIFLIVE_TTS_CACHE_DIR": cache,
                "ANIFLIVE_TTS_SOURCE_DIR": "/app/minimal_inference",
                "CUDA_VISIBLE_DEVICES": "0",
                "HF_HOME": cache + "/huggingface",
                "TORCH_HOME": cache + "/torch",
                "XDG_CACHE_HOME": cache,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            })
            server = None
            with log_path.open("w", encoding="utf-8") as log:
                try:
                    server = subprocess.Popen(
                        (sys.executable, "-s", "-m", "aniflive_tts", "serve",
                         "--model-package", str(package), "--shared-dir", str(shared),
                         "--host", "127.0.0.1", "--port", str(evaluation._SERVER_PORT)),
                        cwd="/app", env=environment, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, text=True,
                        encoding="utf-8", errors="replace", shell=False,
                    )
                    health = evaluation._wait_for_service(server, 180)
                    row = {"variant": name, "repetition_penalty": penalty,
                           "sampling_contract": sampling_contract or "legacy-topk-v1",
                           "model_id": model_id, "engine_fingerprint": health.get("engine_fingerprint"),
                           "qualification_status": "diagnostic-only", "cases": []}
                    for index, (language, text) in enumerate(cases, 1):
                        status, headers, audio = evaluation._request(
                            "POST", "/v1/audio/speech", timeout=180,
                            body={"model": model_id, "text": text, "language": language,
                                  "stream": False,
                                  "generation": {"seed": 1234, "top_k": 15,
                                                 "top_p": 1.0, "temperature": 1.0}},
                        )
                        if status != 200:
                            raise ConversionParityWorkerError(
                                f"Production content diagnostic returned HTTP {status}"
                            )
                        evaluation._validate_headers(headers, language)
                        rate, values = evaluation._decode_wav(audio)
                        path = root / f"case-{index}.wav"
                        path.write_bytes(audio)
                        artifacts.append(path)
                        row["cases"].append({
                            "language": language, "text": text,
                            "path": str(path.relative_to(output)),
                            "sha256": _sha256_file(path), "samples": int(values.size),
                            "sample_rate": rate,
                        })
                    records.append(row)
                finally:
                    evaluation._terminate(server)
        artifacts.append(log_path)
    return records, artifacts


def run_conversion_parity(
    manifest: Mapping[str, Any], output: Path, *, listening_only: bool = False,
) -> tuple[dict[str, Any], list[Path]]:
    if not sys.platform.startswith("linux"):
        raise ConversionParityWorkerError("conversion parity runs only in Linux worker")
    inputs = manifest.get("container_input_paths")
    if not isinstance(inputs, Mapping):
        raise ConversionParityWorkerError("worker manifest has no container inputs")
    required = ("model_package", "selected_checkpoints", "deployment_reference", "shared_dir", "asr_model")
    if any(not isinstance(inputs.get(key), str) for key in required):
        raise ConversionParityWorkerError("conversion parity inputs are incomplete")
    case_specs = None
    listening_path = None
    native_only = False
    include_production = False
    replay_native_decoder = False
    production_sampling_contract = None
    if listening_only:
        if not isinstance(inputs.get("listening_plan"), str):
            raise ConversionParityWorkerError("stage listening requires a listening plan")
        listening_path = Path(inputs["listening_plan"])
        case_specs = _listening_cases(listening_path)
        production_sampling_contract = _object(listening_path, "listening plan").get("production_sampling_contract")
        if production_sampling_contract not in {None, "native-v2proplus-v1"}:
            raise ConversionParityWorkerError("Unsupported production sampling contract")
        replay_native_decoder = _object(listening_path, "listening plan").get("replay_native_decoder", False)
        if not isinstance(replay_native_decoder, bool):
            raise ConversionParityWorkerError("replay_native_decoder must be boolean")
        include_production = _object(listening_path, "listening plan").get("include_production_runtime", False)
        if not isinstance(include_production, bool):
            raise ConversionParityWorkerError("include_production_runtime must be boolean")
        native_flag = _object(listening_path, "listening plan").get("include_native_pytorch", False)
        if not isinstance(native_flag, bool):
            raise ConversionParityWorkerError("include_native_pytorch must be boolean")
        native_only = _object(listening_path, "listening plan").get("native_only", False)
        if not isinstance(native_only, bool) or (native_only and not native_flag):
            raise ConversionParityWorkerError("native_only requires include_native_pytorch")
    if replay_native_decoder and not (native_only and native_flag):
        raise ConversionParityWorkerError("Native decoder replay requires native-only capture")
    package = _directory(Path(str(inputs["model_package"])), "model package")
    selected = _directory(Path(str(inputs["selected_checkpoints"])), "selected checkpoints")
    shared = _directory(Path(str(inputs["shared_dir"])), "shared models")
    asr_model = _directory(Path(str(inputs["asr_model"])), "ASR model")
    reference_lock = _object(
        Path(str(inputs["deployment_reference"])), "deployment reference"
    )
    if reference_lock.get("status") != "human-locked":
        raise ConversionParityWorkerError("conversion parity requires a human-locked reference")
    source_dir = Path(
        os.environ.get("ANIFLIVE_TTS_SOURCE_DIR", "/app/minimal_inference")
    ).resolve(strict=True)
    for value in (source_dir, source_dir / "GPT_SoVITS"):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))
    output.mkdir(parents=True, exist_ok=True)
    pytorch_onnx_cosine, export_evidence = (
        (0.0, []) if native_only else _pytorch_onnx_evidence(package)
    )
    semantic = {} if native_only else _onnx_trt_semantic_parity(package)
    cases, audio = _complete_output_cases(
        package=package,
        selected=selected,
        shared=shared,
        asr_model=asr_model,
        output=output,
        semantic=semantic,
        pytorch_onnx_cosine=pytorch_onnx_cosine,
        case_specs=case_specs,
        include_native_pytorch=native_flag if listening_only else False,
        native_only=native_only,
        replay_native_decoder=replay_native_decoder,
    )
    content_review = None
    if inputs.get("content_review") and not native_only:
        review_path = Path(str(inputs["content_review"]))
        content_review = _apply_content_review(cases, review_path, output)
        retained_review = output / "human-content-review.json"
        shutil.copyfile(review_path, retained_review)
        audio.append(retained_review)
    deployment_path = selected / "selected" / "deployment-checkpoints.json"
    engine_manifest = select_engine_dir(
        package, _object(package / "manifest.json", "model manifest")
    ) / "engine-manifest.json"
    report = (
        {
            "status": "not-applicable",
            "backend_chain": ["PyTorch"],
            "cases": cases,
            "checkpoint_manifest_sha256": _sha256_file(deployment_path),
            "conversion_parity_evaluated": False,
        }
        if native_only
        else conversion_parity_report(
            cases,
            checkpoint_manifest_sha256=_sha256_file(deployment_path),
            engine_build_sha256=_sha256_file(engine_manifest),
        )
    )
    if content_review is not None:
        report["human_content_review"] = content_review
    report["pytorch_onnx"] = export_evidence
    report["onnx_tensorrt_semantic"] = semantic
    report["deployment_reference_sha256"] = _sha256_file(
        Path(str(inputs["deployment_reference"]))
    )
    if listening_only:
        report["numerical_comparison_status"] = report["status"]
        report["schema"] = "aniflive-stage-listening-v1"
        report["status"] = "measured"
        report["qualification"] = {"status": "diagnostic-only", "release_qualified": False}
        report["listening_plan_sha256"] = _sha256_file(listening_path)
        retained_plan = output / "listening-plan.json"
        shutil.copyfile(listening_path, retained_plan)
        audio.append(retained_plan)
    if include_production:
        variants, variant_artifacts = _production_runtime_variants(
            package, shared, case_specs, output, sampling_contract=production_sampling_contract,
        )
        report["production_runtime_variants"] = variants
        audio.extend(variant_artifacts)
    report_path = output / (
        "evaluation-report.json" if listening_only else "conversion-parity-report.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts: list[Path] = [report_path, *audio]
    if report["status"] == "passed":
        destination = output / "model-package"
        shutil.copytree(package, destination, symlinks=False)
        artifacts.extend(path for path in destination.rglob("*") if path.is_file())
    return report, artifacts


__all__ = ["ConversionParityWorkerError", "run_conversion_parity"]
