# Modified by AnifLive-TTS in 2026.
# Changes: restricted checkpoint loading and local-only export hardening.
# See ANIFLIVE_TTS_PROVENANCE.json for the upstream revision and details.

import argparse
import os
import sys
import torch
import json
from torch import nn
from torch.nn import functional as F
from GPT_SoVITS.process_ckpt import load_sovits_new, get_sovits_version_from_path_fast
from GPT_SoVITS.feature_extractor import cnhubert
from GPT_SoVITS.text import _symbol_to_id_v2
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.module.models import SynthesizerTrn
from utils import HParams as LegacyHParams
from transformers import AutoModelForMaskedLM
import logging

# ONNX校验
import onnx_validation

logging.getLogger("torch.onnx").setLevel(logging.WARN)
logging.getLogger("onnx").setLevel(logging.WARN)
logging.getLogger("onnx_ir").setLevel(logging.WARN)
logging.getLogger("onnxscript").setLevel(logging.WARN)

# V2ProPlus training serializes the configuration under its historical fully
# qualified name.  Pinning that exact name keeps weights-only loading enabled
# without falling back to executable pickle deserialization.
SAFE_LEGACY_HPARAMS = [(LegacyHParams, "utils.HParams")]

# Wrappers for ONNX Export

class T2SEncoder(nn.Module):
    def __init__(self, t2s_model):
        super().__init__()
        self.t2s_model = t2s_model

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        pass

class GPTEncoder(nn.Module):
    def __init__(self, t2s_model, max_len=2000, full_logits=False):
        super().__init__()
        self.t2s_model = t2s_model
        self.max_len = max_len
        self.full_logits = bool(full_logits)

    def forward(self, phoneme_ids, prompts, bert_feature):
        # Wrapper for infer_first_stage
        # Returns: logits, k_cache (stacked), v_cache (stacked), x_len, y_len
        
        logits, k_cache, v_cache, x_len, y_len = self.t2s_model.model.infer_first_stage(
            phoneme_ids, prompts, bert_feature
        )
        
        # Stack caches: List[Tensor] -> Tensor [Layers, B, T, D]
        k_cache_stacked = torch.stack(k_cache, dim=0)
        v_cache_stacked = torch.stack(v_cache, dim=0)
        
        # Pad to max length for pre-allocation
        k_cache_padded = F.pad(k_cache_stacked, (0, 0, 0, self.max_len - k_cache_stacked.shape[2]))
        v_cache_padded = F.pad(v_cache_stacked, (0, 0, 0, self.max_len - v_cache_stacked.shape[2]))
        
        # Optimization: Return Top-K instead of full logits
        topk_values, topk_indices = torch.topk(logits, k=50, dim=-1)
        
        # Ensure x_len and y_len are rank-1 tensors for ONNX export dynamic axes
        if not isinstance(x_len, torch.Tensor):
            x_len = torch.tensor([x_len], dtype=torch.long)
        else:
            x_len = x_len.reshape(1)
        if not isinstance(y_len, torch.Tensor):
            y_len = torch.tensor([y_len], dtype=torch.long)
        else:
            y_len = y_len.reshape(1)
        
        outputs = (topk_values, topk_indices, k_cache_padded, v_cache_padded, x_len, y_len)
        return (*outputs, logits) if self.full_logits else outputs

class GPTStep(nn.Module):
    def __init__(self, t2s_model, full_logits=False):
        super().__init__()
        self.t2s_model = t2s_model
        self.full_logits = bool(full_logits)

    def forward(self, samples, k_cache, v_cache, x_len, y_len, idx):
        # Wrapper for infer_next_stage
        # k_cache, v_cache are stacked [Layers, B, T_max, D]
        
        # Ensure x_len, y_len, idx are scalars for the underlying model if they come as rank-1
        x_len_s = x_len[0] if x_len.ndim > 0 else x_len
        y_len_s = y_len[0] if y_len.ndim > 0 else y_len
        idx_s = idx[0] if idx.ndim > 0 else idx

        # Unstack to list
        k_cache_list = [t for t in k_cache]
        v_cache_list = [t for t in v_cache]
        
        logits, k_cache_new, v_cache_new = self.t2s_model.model.infer_next_stage(
            samples, k_cache_list, v_cache_list, x_len_s, y_len_s, idx_s
        )
        
        # Stack again (they should still be the same tensors if updated in-place)
        k_cache_stacked = torch.stack(k_cache_new, dim=0)
        v_cache_stacked = torch.stack(v_cache_new, dim=0)
        
        # Optimization: Return Top-K instead of full logits to reduce GPU->CPU transfer
        # Keep legacy top-50 outputs; native sampling additionally needs the full vocabulary.
        topk_values, topk_indices = torch.topk(logits, k=50, dim=-1)
        
        outputs = (topk_values, topk_indices, k_cache_stacked, v_cache_stacked)
        return (*outputs, logits) if self.full_logits else outputs

class SoVITS(nn.Module):
    def __init__(self, vq_model, version):
        super().__init__()
        self.vq_model = vq_model
        self.version = version

    def forward(self, pred_semantic, text_seq, refer_spec, sv_emb=None, noise_scale=0.5, speed=1.0):
        # Reconstruct list for decode
        refer_list = [refer_spec]
        sv_emb_list = [sv_emb] if sv_emb is not None else None
        
        return self.vq_model.decode(
            pred_semantic, text_seq, refer_list, sv_emb=sv_emb_list, noise_scale=noise_scale, speed=speed
        )


class SoVITSStreaming(nn.Module):
    """Export the native V2ProPlus latent-overlap streaming decoder."""

    def __init__(self, vq_model, version):
        super().__init__()
        self.vq_model = vq_model
        self.version = version

    def forward(
        self,
        pred_semantic,
        text_seq,
        refer_spec,
        sv_emb,
        noise_scale,
        speed,
        result_length,
        overlap_frames,
        overlap_enabled,
        acoustic_noise,
    ):
        refer_list = [refer_spec]
        sv_emb_list = [sv_emb] if sv_emb is not None else None
        return self.vq_model.decode_streaming(
            pred_semantic,
            text_seq,
            refer_list,
            sv_emb=sv_emb_list,
            noise_scale=noise_scale,
            speed=speed,
            result_length=result_length[0],
            overlap_frames=overlap_frames,
            padding_length=None,
            overlap_enabled=overlap_enabled,
            acoustic_noise=acoustic_noise,
        )

class VQEncoder(nn.Module):
    def __init__(self, vq_model):
        super().__init__()
        self.vq_model = vq_model
    
    def forward(self, ssl_content):
        # ssl_content: [1, 768, T]
        codes = self.vq_model.extract_latent(ssl_content)
        # codes: [1, 1, T] (indices)
        return codes

class SpectrogramWrapper(nn.Module):
    """Portable FP32 STFT with explicit radix-2 butterflies.

    Native PyTorch and ONNX FFT libraries round quiet bins differently. Keeping
    the same operation graph gives deterministic export parity without changing
    the Hann window, reflection padding, hop, magnitude floor or public I/O.
    """

    def __init__(self, filter_length, hop_length, win_length, sampling_rate):
        super().__init__()
        if filter_length < 2 or filter_length & (filter_length - 1):
            raise ValueError("Spectrogram filter_length must be a power of two")
        if not 0 < hop_length <= filter_length or not 0 < win_length <= filter_length:
            raise ValueError("Spectrogram hop/window lengths are invalid")
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.register_buffer("hann_window", torch.hann_window(win_length))
        padding = filter_length - win_length
        self.register_buffer(
            "analysis_window",
            F.pad(self.hann_window, (padding // 2, padding - padding // 2)),
        )
        bits = filter_length.bit_length() - 1
        reverse = [
            int(format(index, f"0{bits}b")[::-1], 2)
            for index in range(filter_length)
        ]
        self.register_buffer("bit_reverse", torch.tensor(reverse, dtype=torch.long))
        self.fft_widths = tuple(2 ** stage for stage in range(1, bits + 1))
        for width in self.fft_widths:
            angle = torch.arange(width // 2, dtype=torch.float64) * (-2 * torch.pi / width)
            self.register_buffer(f"fft_cos_{width}", angle.cos().float())
            self.register_buffer(f"fft_sin_{width}", angle.sin().float())

    def forward(self, y):
        n_fft = self.filter_length
        padding = (n_fft - self.hop_length) // 2
        audio = F.pad(y.float().unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
        starts = torch.arange(
            0, audio.shape[-1] - n_fft + 1, self.hop_length, device=y.device
        )
        indexes = starts.unsqueeze(1) + torch.arange(n_fft, device=y.device).unsqueeze(0)
        real, imaginary = self.fft_components(audio[:, indexes] * self.analysis_window)
        magnitude = torch.sqrt(real.square() + imaginary.square() + 1e-8)
        return magnitude.transpose(1, 2)

    def fft_components(self, frames):
        n_fft = self.filter_length
        real = frames.index_select(-1, self.bit_reverse)
        imaginary = torch.zeros_like(real)
        for width in self.fft_widths:
            r = real.reshape(real.shape[0], real.shape[1], n_fft // width, width)
            i = imaginary.reshape(imaginary.shape[0], imaginary.shape[1], n_fft // width, width)
            half = width // 2
            even_r, even_i = r[..., :half], i[..., :half]
            odd_r, odd_i = r[..., half:], i[..., half:]
            cosine = getattr(self, f"fft_cos_{width}")
            sine = getattr(self, f"fft_sin_{width}")
            rotated_r = odd_r * cosine - odd_i * sine
            rotated_i = odd_r * sine + odd_i * cosine
            real = torch.cat((even_r + rotated_r, even_r - rotated_r), dim=-1).reshape_as(real)
            imaginary = torch.cat((even_i + rotated_i, even_i - rotated_i), dim=-1).reshape_as(imaginary)
        bins = n_fft // 2 + 1
        return real[..., :bins], imaginary[..., :bins]


class SVEmbeddingWrapper(nn.Module):
    """Export the native dither-free Kaldi fbank contract before ERes2NetV2."""

    def __init__(self, sv_model):
        super().__init__()
        from GPT_SoVITS.eres2net import kaldi

        self.embedding_model = sv_model
        self.window_size = 400
        self.window_shift = 160
        self.padded_window_size = 512
        self.register_buffer(
            "window", torch.hann_window(self.window_size, periodic=False).pow(0.85)
        )
        banks = kaldi.get_mel_banks(
            80, self.padded_window_size, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0
        )
        self.register_buffer("mel_filterbank", F.pad(banks, (0, 1)))
        self.fft = SpectrogramWrapper(512, 160, 400, 16000)

    def features(self, wav):
        audio = wav.float()
        starts = torch.arange(
            0, audio.shape[-1] - self.window_size + 1,
            self.window_shift, device=audio.device,
        )
        indexes = starts.unsqueeze(1) + torch.arange(
            self.window_size, device=audio.device
        ).unsqueeze(0)
        frames = audio[:, indexes]
        frames = frames - frames.mean(dim=-1, keepdim=True)
        previous = torch.cat((frames[..., :1], frames[..., :-1]), dim=-1)
        frames = (frames - 0.97 * previous) * self.window
        frames = F.pad(frames, (0, self.padded_window_size - self.window_size))
        real, imaginary = self.fft.fft_components(frames)
        power = real.square() + imaginary.square()
        mel = torch.matmul(power, self.mel_filterbank.T)
        return torch.clamp_min(mel, torch.finfo(torch.float32).eps).log()

    def forward(self, wav):
        return self.embedding_model.forward3(self.features(wav))


class NativeSVEmbeddingReference(nn.Module):
    """Independent native feature oracle used only by export validation."""

    def __init__(self, sv_model):
        super().__init__()
        self.embedding_model = sv_model

    def forward(self, wav):
        from GPT_SoVITS.eres2net import kaldi

        features = torch.stack([
            kaldi.fbank(
                row.float().unsqueeze(0), num_mel_bins=80,
                sample_frequency=16000, dither=0,
            )
            for row in wav
        ])
        return self.embedding_model.forward3(features)


def hparams_to_dict(hp):
    if hasattr(hp, "__dict__"):
        return {k: hparams_to_dict(v) for k, v in hp.__dict__.items()}
    elif isinstance(hp, dict):
        return {k: hparams_to_dict(v) for k, v in hp.items()}
    elif isinstance(hp, (list, tuple)):
        return [hparams_to_dict(v) for v in hp]
    else:
        return hp

def stabilize_gpt_layer_norm(onnx_path):
    """Express FP32 layer norm with centered variance, retaining the native oracle.

    ORT's LayerNormalization CPU kernel can accumulate enough cancellation
    error to fail cache checks on some GPT weights. This changes the exported
    arithmetic, not its mathematical contract, dtype, weights or tolerances.
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    from pathlib import Path

    path = Path(onnx_path)
    graph = onnx.load(path)
    candidates = [node for node in graph.graph.node if node.op_type == "LayerNormalization"]
    if not candidates:
        return 0
    opset = next(value.version for value in graph.opset_import if not value.domain)
    prefix = "aniflive_centered_layer_norm"
    occupied = {
        name for node in graph.graph.node for name in (*node.input, *node.output)
    } | {value.name for value in graph.graph.initializer}
    while any(name.startswith(prefix) for name in occupied):
        prefix += "_"
    axes = prefix + "_axes"
    if opset >= 18:
        graph.graph.initializer.append(
            numpy_helper.from_array(np.array([-1], dtype=np.int64), axes)
        )
    nodes = []
    rewritten = 0
    for node in graph.graph.node:
        if node.op_type != "LayerNormalization":
            nodes.append(node)
            continue
        attrs = {value.name: helper.get_attribute_value(value) for value in node.attribute}
        if attrs.get("axis", -1) != -1 or len(node.output) != 1 or len(node.input) not in {2, 3}:
            raise ValueError("GPT layer norm stabilization requires one last-axis output")
        stem = f"{prefix}_{rewritten}"
        rewritten += 1
        x, scale = node.input[:2]
        epsilon = stem + "_epsilon"
        graph.graph.initializer.append(numpy_helper.from_array(
            np.array(attrs.get("epsilon", 1e-5), dtype=np.float32), epsilon
        ))
        mean, centered, square, variance, adjusted, std, inverse, normalized, scaled = (
            stem + "_" + name for name in
            ("mean", "centered", "square", "variance", "adjusted", "std", "inverse", "normalized", "scaled")
        )
        def reduce_mean(value, output):
            return helper.make_node(
                "ReduceMean", [value, axes] if opset >= 18 else [value], [output],
                **({"keepdims": 1} if opset >= 18 else {"axes": [-1], "keepdims": 1}),
            )
        nodes.extend((
            reduce_mean(x, mean),
            helper.make_node("Sub", [x, mean], [centered]),
            helper.make_node("Mul", [centered, centered], [square]),
            reduce_mean(square, variance),
            helper.make_node("Add", [variance, epsilon], [adjusted]),
            helper.make_node("Sqrt", [adjusted], [std]),
            helper.make_node("Reciprocal", [std], [inverse]),
            helper.make_node("Mul", [centered, inverse], [normalized]),
            helper.make_node("Mul", [normalized, scale], [scaled]),
            helper.make_node("Add", [scaled, node.input[2]], list(node.output))
            if len(node.input) == 3 and node.input[2]
            else helper.make_node("Identity", [scaled], list(node.output)),
        ))
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    property_value = graph.metadata_props.add()
    property_value.key = "aniflive.layer_norm"
    property_value.value = json.dumps({
        "contract": "centered-variance-fp32-v1", "nodes": rewritten,
        "native_pytorch_reference": "unchanged",
    }, sort_keys=True)
    onnx.checker.check_model(graph)
    temporary = path.with_suffix(".stable.tmp.onnx")
    try:
        onnx.save(graph, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return rewritten


GPT_VALIDATION_SEEDS = (1234, 0, 1, 7, 42, 99, 2026, 31415)


def gpt_validation_inputs(seed):
    """Model-independent, reproducible encoder probes."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return {
        "phoneme_ids": torch.randint(0, 512, (1, 50), generator=generator),
        "prompts": torch.randint(0, 1024, (1, 20), generator=generator),
        "bert_feature": torch.randn(1, 1024, 50, generator=generator),
    }


def export_onnx(args):
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    device = "cpu" # Export on CPU usually safer for dynamic axes

    # 初始化校验器（如果启用）
    validator = None
    if args.validate:
        print(f"\n{'='*60}")
        print("启用ONNX导出精度校验")
        print(f"{'='*60}\n")
        validator = onnx_validation.ONNXValidator(
            output_dir=args.output_dir,
            onnx_device=args.validation_device
        )

    print("Loading models...")
    # SSL
    cnhubert.cnhubert_base_path = args.cnhubert_base_path
    ssl_model = cnhubert.get_model()
    ssl_model = ssl_model.to(device)
    ssl_model.eval()
    
    # BERT
    bert_model = AutoModelForMaskedLM.from_pretrained(
        args.bert_path, local_files_only=True, trust_remote_code=False
    )
    bert_model = bert_model.to(device)
    bert_model.eval()

    # GPT
    try:
        dict_s1 = torch.load(args.gpt_path, map_location="cpu", weights_only=True)
    except Exception as error:
        if not args.allow_unsafe_pickle:
            raise RuntimeError(
                "GPT checkpoint cannot be loaded with weights_only=True. "
                "Use --allow_unsafe_pickle only for a trusted local checkpoint."
            ) from error
        dict_s1 = torch.load(args.gpt_path, map_location="cpu", weights_only=False)
    config = dict_s1["config"]
    t2s_model = Text2SemanticLightningModule(config, "output", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    t2s_model.eval()
    
    # SoVITS
    try:
        with torch.serialization.safe_globals(SAFE_LEGACY_HPARAMS):
            dict_s2 = load_sovits_new(args.sovits_path, weights_only=True)
    except Exception as error:
        if not args.allow_unsafe_pickle:
            raise RuntimeError(
                "SoVITS checkpoint cannot be loaded with weights_only=True. "
                "Use --allow_unsafe_pickle only for a trusted local checkpoint."
            ) from error
        dict_s2 = load_sovits_new(args.sovits_path, weights_only=False)
    hps = dict_s2["config"]
    # Handle DictToAttrRecursive logic manually or using the class if available. 
    class AttrDict(object):
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, AttrDict(v))
                else:
                    setattr(self, k, v)
    
    hps_obj = AttrDict(hps)
    hps_obj.model.semantic_frame_rate = "25hz"
    with torch.serialization.safe_globals(SAFE_LEGACY_HPARAMS):
        _, model_version, _ = get_sovits_version_from_path_fast(
            args.sovits_path,
            weights_only=not args.allow_unsafe_pickle,
        )
    hps_obj.model.version = model_version
    
    # Update the original hps dict as well to ensure SynthesizerTrn gets the right values
    hps["model"]["version"] = model_version
    hps["model"]["semantic_frame_rate"] = "25hz"
    
    vq_model = SynthesizerTrn(
        hps_obj.data.filter_length // 2 + 1,
        hps_obj.train.segment_size // hps_obj.data.hop_length,
        n_speakers=hps_obj.data.n_speakers,
        **hps["model"]
    )
    vq_model.eval()
    vq_model.load_state_dict(dict_s2["weight"], strict=False)
    
    # Patch EuclideanCodebook.init_embed_ to avoid export error
    for name, module in vq_model.named_modules():
        if "EuclideanCodebook" in module.__class__.__name__:
            import types
            module.init_embed_ = types.MethodType(lambda self, data: None, module)

    # SV Model (for speaker embedding)
    sv_path = os.environ.get("SV_MODEL_PATH", "pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt")
    sys.path.append(os.path.join(os.path.dirname(__file__), "GPT_SoVITS", "eres2net"))
    from ERes2NetV2 import ERes2NetV2
    pretrained_state = torch.load(sv_path, map_location="cpu", weights_only=True)
    sv_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
    sv_model.load_state_dict(pretrained_state)
    sv_model.eval()
    sv_model = sv_model.to(device)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Exporting to {output_dir}...")
    print("Exporting SSL...")
    class SSLWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, audio):
            # HubertModel returns [B, T, C], we need [B, C, T] for VQEncoder
            return self.model(audio).last_hidden_state.transpose(1, 2)

    ssl_wrapper = SSLWrapper(ssl_model.model)
    # Input: [1, T] audio 16k
    dummy_audio = torch.randn(1, 16000 * 2)
    torch.onnx.export(
        ssl_wrapper,
        (dummy_audio,),
        f"{output_dir}/ssl.onnx",
        input_names=["audio"],
        output_names=["last_hidden_state"],
        dynamic_axes={"audio": {1: "time"}, "last_hidden_state": {2: "time"}},
        opset_version=18,
        dynamo=False
    )

    # 校验SSL模型
    if validator:
        validator.validate_model(
            model_name="SSL",
            onnx_path=f"{output_dir}/ssl.onnx",
            pytorch_model=ssl_wrapper,
            dummy_inputs={"audio": dummy_audio},
            output_names=["last_hidden_state"],
            rtol=1e-3,
            atol=1e-5
        )
    
    print("Exporting BERT...")
    # Input: input_ids [1, T], attention_mask [1, T], token_type_ids [1, T]
    dummy_input_ids = torch.randint(0, 100, (1, 20), dtype=torch.long)
    dummy_attn_mask = torch.ones(1, 20, dtype=torch.long)
    dummy_token_type = torch.zeros(1, 20, dtype=torch.long)
    # Wrapper for BERT to return only what we need
    class BERTWrapper(nn.Module):
        def __init__(self, bert):
            super().__init__()
            self.bert = bert
        def forward(self, input_ids, attention_mask, token_type_ids):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            return torch.cat(outputs.hidden_states[-3:-2], -1)

    bert_wrapper = BERTWrapper(bert_model)
    torch.onnx.export(
        bert_wrapper,
        (dummy_input_ids, dummy_attn_mask, dummy_token_type),
        f"{output_dir}/bert.onnx",
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["hidden_states"],
        dynamic_axes={"input_ids": {1: "seq_len"}, "attention_mask": {1: "seq_len"}, "token_type_ids": {1: "seq_len"}, "hidden_states": {1: "seq_len"}},
        opset_version=20,
        dynamo=False
    )

    # 校验BERT模型
    if validator:
        validator.validate_model(
            model_name="BERT",
            onnx_path=f"{output_dir}/bert.onnx",
            pytorch_model=bert_wrapper,
            dummy_inputs={
                "input_ids": dummy_input_ids,
                "attention_mask": dummy_attn_mask,
                "token_type_ids": dummy_token_type
            },
            output_names=["hidden_states"],
            rtol=1e-3,
            atol=1e-5
        )
    
    print("Exporting VQEncoder...")
    vq_enc = VQEncoder(vq_model)
    # ssl_content: [1, 768, T]
    dummy_ssl = torch.randn(1, 768, 100)
    torch.onnx.export(
        vq_enc,
        (dummy_ssl,),
        f"{output_dir}/vq_encoder.onnx",
        input_names=["ssl_content"],
        output_names=["codes"],
        dynamic_axes={"ssl_content": {2: "time"}, "codes": {2: "time"}},
        opset_version=20,
        dynamo=False
    )

    # 校验VQEncoder模型
    if validator:
        validator.validate_model(
            model_name="VQEncoder",
            onnx_path=f"{output_dir}/vq_encoder.onnx",
            pytorch_model=vq_enc,
            dummy_inputs={"ssl_content": dummy_ssl},
            output_names=["codes"],
            rtol=1e-3,
            atol=1e-5
        )

    print("Exporting GPT Encoder...")
    gpt_enc = GPTEncoder(t2s_model, max_len=args.max_len, full_logits=True)
    # Dummies
    trace_inputs = gpt_validation_inputs(GPT_VALIDATION_SEEDS[0])
    phoneme_ids = trace_inputs["phoneme_ids"]
    prompts = trace_inputs["prompts"]
    bert_feature = trace_inputs["bert_feature"]
    
    dynamic_axes_gpt = {
        "phoneme_ids": {1: "text_len"},
        "prompts": {1: "prompt_len"},
        "bert_feature": {2: "text_len"},
        "k_cache": {1: "batch_size"},
        "v_cache": {1: "batch_size"},
        "x_len": {0: "one"},
        "y_len": {0: "one"},
    }
    
    torch.onnx.export(
        gpt_enc,
        (phoneme_ids, prompts, bert_feature),
        f"{output_dir}/gpt_encoder.onnx",
        input_names=["phoneme_ids", "prompts", "bert_feature"],
        output_names=["topk_values", "topk_indices", "k_cache", "v_cache", "x_len", "y_len", "logits"],
        dynamic_axes=dynamic_axes_gpt,
        opset_version=20,
        dynamo=False
    )

    stabilize_gpt_layer_norm(f"{output_dir}/gpt_encoder.onnx")

    if validator:
        for probe_seed in GPT_VALIDATION_SEEDS:
            if not validator.validate_model(
                model_name="GPTEncoder",
                onnx_path=f"{output_dir}/gpt_encoder.onnx",
                pytorch_model=gpt_enc,
                dummy_inputs=gpt_validation_inputs(probe_seed),
                output_names=["topk_values", "topk_indices", "k_cache", "v_cache", "x_len", "y_len", "logits"],
                rtol=1e-3,
                atol=1e-5,
                case_id=f"seed-{probe_seed}",
            ):
                raise RuntimeError("GPT encoder PyTorch to ONNX validation failed")

    print("Exporting GPT Step...")
    # Get outputs from encoder to feed to step
    with torch.no_grad():
        topk_v_dummy, topk_i_dummy, k_cache, v_cache, x_len, y_len = gpt_enc(phoneme_ids, prompts, bert_feature)[:6]
    
    gpt_step = GPTStep(t2s_model, full_logits=True)
    idx = torch.tensor([0], dtype=torch.long)
    # samples input for step is indices [B, 1]
    samples = torch.randint(0, 1024, (1, 1), dtype=torch.long)
    
    dynamic_axes_step = {
        "k_cache": {1: "batch_size"},
        "v_cache": {1: "batch_size"},
        "x_len": {0: "one"},
        "y_len": {0: "one"},
        "idx": {0: "one"},
    }
    
    torch.onnx.export(
        gpt_step,
        (samples, k_cache, v_cache, x_len, y_len, idx),
        f"{output_dir}/gpt_step.onnx",
        input_names=["samples", "k_cache", "v_cache", "x_len", "y_len", "idx"],
        output_names=["topk_values", "topk_indices", "k_cache_new", "v_cache_new", "logits"],
        dynamic_axes=dynamic_axes_step,
        opset_version=20,
        dynamo=False
    )

    stabilize_gpt_layer_norm(f"{output_dir}/gpt_step.onnx")

    if validator:
        for probe_seed in GPT_VALIDATION_SEEDS:
            probe = gpt_validation_inputs(probe_seed)
            with torch.no_grad():
                _, _, probe_k, probe_v, probe_x, probe_y = gpt_enc(*probe.values())[:6]
            probe_sample = torch.randint(
                0, 1024, (1, 1),
                generator=torch.Generator(device="cpu").manual_seed(probe_seed),
            )
            if not validator.validate_model(
                model_name="GPTStep",
                onnx_path=f"{output_dir}/gpt_step.onnx",
                pytorch_model=gpt_step,
                dummy_inputs={
                    "samples": probe_sample, "k_cache": probe_k, "v_cache": probe_v,
                    "x_len": probe_x, "y_len": probe_y, "idx": idx,
                },
                output_names=["topk_values", "topk_indices", "k_cache_new", "v_cache_new", "logits"],
                rtol=1e-3,
                atol=1e-5,
                case_id=f"seed-{probe_seed}",
            ):
                raise RuntimeError("GPT step PyTorch to ONNX validation failed")

    print("Exporting SoVITS...")
    sovits_wrapper = SoVITS(vq_model, model_version)
    # Dummies
    # pred_semantic: [1, 1, T_sem] -> [1, 1, 150]
    pred_semantic = torch.randint(0, 1024, (1, 1, 150), dtype=torch.long)
    text_seq = torch.randint(0, 512, (1, 50), dtype=torch.long)
    # refer_spec: [1, C, T_ref] -> [1, 1025, 200]
    refer_spec = torch.randn(1, 1025, 200)
    noise_scale = torch.tensor([0.5], dtype=torch.float32)
    speed = torch.tensor([1.0], dtype=torch.float32)
    
    args_sovits = [pred_semantic, text_seq, refer_spec]
    input_names = ["pred_semantic", "text_seq", "refer_spec"]
    
    if "Pro" in model_version:
        sv_emb = torch.randn(1, 20480)
        args_sovits.append(sv_emb)
        input_names.append("sv_emb")

    args_sovits.extend([noise_scale, speed])
    input_names.extend(["noise_scale", "speed"])
    
    dynamic_axes_sovits = {
        "pred_semantic": {2: "sem_len"},
        "text_seq": {1: "text_len"},
        "refer_spec": {2: "ref_len"},
    }
    
    torch.onnx.export(
        sovits_wrapper,
        tuple(args_sovits),
        f"{output_dir}/sovits.onnx",
        input_names=input_names,
        output_names=["audio"],
        dynamic_axes=dynamic_axes_sovits,
        opset_version=20,
        dynamo=False
    )

    if validator and not validator.validate_model(
        model_name="SoVITS",
        onnx_path=f"{output_dir}/sovits.onnx",
        pytorch_model=sovits_wrapper,
        dummy_inputs=dict(zip(input_names, args_sovits)),
        output_names=["audio"],
        rtol=1e-3,
        atol=1e-5,
    ):
        raise RuntimeError("SoVITS PyTorch to ONNX validation failed")

    if "Pro" not in model_version:
        raise RuntimeError(
            "AnifLive-TTS streaming export currently requires a V2ProPlus model"
        )
    print("Exporting SoVITS streaming decoder...")
    sovits_streaming_wrapper = SoVITSStreaming(vq_model, model_version)
    streaming_result_length = torch.tensor([18], dtype=torch.int64)
    streaming_overlap = torch.zeros(
        1, 192, args.stream_overlap_frames, dtype=torch.float32
    )
    streaming_overlap_enabled = torch.ones(1, dtype=torch.float32)
    streaming_acoustic_noise = torch.randn(1, 192, 36, dtype=torch.float32)
    torch.onnx.export(
        sovits_streaming_wrapper,
        (
            pred_semantic,
            text_seq,
            refer_spec,
            sv_emb,
            noise_scale,
            speed,
            streaming_result_length,
            streaming_overlap,
            streaming_overlap_enabled,
            streaming_acoustic_noise,
        ),
        f"{output_dir}/sovits_stream.onnx",
        input_names=[
            "pred_semantic",
            "text_seq",
            "refer_spec",
            "sv_emb",
            "noise_scale",
            "speed",
            "result_length",
            "overlap_frames",
            "overlap_enabled",
            "acoustic_noise",
        ],
        output_names=["audio", "latent", "latent_mask"],
        dynamic_axes={
            "pred_semantic": {2: "sem_len"},
            "text_seq": {1: "text_len"},
            "refer_spec": {2: "ref_len"},
            "result_length": {0: "one"},
            "acoustic_noise": {2: "result_frames"},
        },
        opset_version=20,
        dynamo=False,
    )

    if validator and not validator.validate_model(
        model_name="SoVITSStreaming",
        onnx_path=f"{output_dir}/sovits_stream.onnx",
        pytorch_model=sovits_streaming_wrapper,
        dummy_inputs={
            "pred_semantic": pred_semantic,
            "text_seq": text_seq,
            "refer_spec": refer_spec,
            "sv_emb": sv_emb,
            "noise_scale": noise_scale,
            "speed": speed,
            "result_length": streaming_result_length,
            "overlap_frames": streaming_overlap,
            "overlap_enabled": streaming_overlap_enabled,
            "acoustic_noise": streaming_acoustic_noise,
        },
        output_names=["audio", "latent", "latent_mask"],
        rtol=1e-3,
        atol=1e-5,
    ):
        raise RuntimeError("SoVITS streaming PyTorch to ONNX validation failed")

    # Export SpectrogramWrapper
    print("Exporting Spectrogram...")
    spec_wrapper = SpectrogramWrapper(
        filter_length=hps_obj.data.filter_length,
        hop_length=hps_obj.data.hop_length,
        win_length=hps_obj.data.win_length,
        sampling_rate=hps_obj.data.sampling_rate
    )
    # Input: [1, T] audio waveform at sampling_rate
    dummy_wav = torch.randn(1, 48000)
    torch.onnx.export(
        spec_wrapper,
        (dummy_wav,),
        f"{output_dir}/spectrogram.onnx",
        input_names=["audio"],
        output_names=["spectrogram"],
        dynamic_axes={"audio": {1: "time"}, "spectrogram": {2: "time"}},
        opset_version=20,
        dynamo=False
    )

    # 校验Spectrogram模型
    if validator:
        validator.validate_model(
            model_name="Spectrogram",
            onnx_path=f"{output_dir}/spectrogram.onnx",
            pytorch_model=spec_wrapper,
            dummy_inputs={"audio": dummy_wav},
            output_names=["spectrogram"],
            rtol=1e-4,
            atol=1e-6
        )

    # Export SVEmbeddingWrapper
    print("Exporting SV Embedding...")
    sv_wrapper = SVEmbeddingWrapper(sv_model)
    # Input: [B, T] audio waveform at 16kHz
    dummy_wav_16k = torch.randn(1, 16000 * 3)
    torch.onnx.export(
        sv_wrapper,
        (dummy_wav_16k,),
        f"{output_dir}/sv_embedding.onnx",
        input_names=["audio"],
        output_names=["sv_embedding"],
        dynamic_axes={"audio": {1: "time"}},
        opset_version=20,
        dynamo=False
    )

    # 校验SVEmbedding模型
    if validator:
        validator.validate_model(
            model_name="SVEmbedding",
            onnx_path=f"{output_dir}/sv_embedding.onnx",
            pytorch_model=NativeSVEmbeddingReference(sv_model),
            dummy_inputs={"audio": dummy_wav_16k},
            output_names=["sv_embedding"],
            rtol=1e-3,
            atol=1e-5
        )

    # Publish only inference data. Training configs may contain private paths.
    config_dict = {
        "schema": 1,
        "model": {
            "version": model_version,
            "semantic_frame_rate": "25hz",
        },
        "data": {
            "filter_length": hps_obj.data.filter_length,
            "hop_length": hps_obj.data.hop_length,
            "win_length": hps_obj.data.win_length,
            "sampling_rate": hps_obj.data.sampling_rate,
            "max_len": args.max_len,
        },
        "symbol_to_id": _symbol_to_id_v2,
        "spectrogram": {
        "filter_length": hps_obj.data.filter_length,
        "hop_length": hps_obj.data.hop_length,
        "win_length": hps_obj.data.win_length,
            "sampling_rate": hps_obj.data.sampling_rate,
        },
        "sv_embedding": {
            "embedding_size": 20480 if "Pro" in model_version else 512,
            "model_version": model_version,
        },
        "streaming": {
            "overlap_frames": args.stream_overlap_frames,
        },
    }
    with open(f"{output_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)

    print(f"Export complete! Config saved to {output_dir}/config.json")

    # 校验摘要
    if validator:
        validator.print_summary()
        validator.save_report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS ONNX Export")
    parser.add_argument("--gpt_path", required=True)
    parser.add_argument("--sovits_path", required=True)
    parser.add_argument("--cnhubert_base_path", default="pretrained_models/chinese-hubert-base")
    parser.add_argument("--bert_path", default="pretrained_models/chinese-roberta-wwm-ext-large")
    parser.add_argument("--sv_path", default=None, help="Path to SV model (default: pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt)")
    parser.add_argument("--max_len", type=int, default=2000, help="Pre-allocated KV cache length")
    parser.add_argument(
        "--stream_overlap_frames",
        type=int,
        default=12,
        help="Static latent overlap exported into the V2ProPlus streaming decoder",
    )
    parser.add_argument("--output_dir", default="onnx_export", help="Output directory for ONNX models")
    parser.add_argument("--validate", action="store_true", help="Enable ONNX export accuracy validation")
    parser.add_argument("--validation_device", default="cpu", choices=["cpu", "cuda"], help="Device for ONNX validation (default: cpu)")
    parser.add_argument(
        "--allow_unsafe_pickle",
        action="store_true",
        help="Allow weights_only=False for trusted GPT/SoVITS checkpoints",
    )

    args = parser.parse_args()

    if args.stream_overlap_frames <= 0 or args.stream_overlap_frames % 2:
        parser.error("--stream_overlap_frames must be a positive even integer")

    # Set SV model path if provided
    if args.sv_path:
        os.environ["SV_MODEL_PATH"] = args.sv_path

    export_onnx(args)
