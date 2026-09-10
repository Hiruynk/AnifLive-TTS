from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

STAGE_ORDER = (
    "ssl",
    "bert",
    "vq_encoder",
    "gpt_encoder",
    "gpt_step",
    "spectrogram",
    "sv_embedding",
    "sovits",
    "sovits_stream",
)

STAGE_IO_CONTRACTS = {
    "ssl": (("audio",), ("last_hidden_state",)),
    "bert": (("input_ids", "attention_mask", "token_type_ids"), ("hidden_states",)),
    "vq_encoder": (("ssl_content",), ("codes",)),
    "gpt_encoder": (
        ("phoneme_ids", "prompts", "bert_feature"),
        ("topk_values", "topk_indices", "k_cache", "v_cache", "x_len", "y_len"),
    ),
    "gpt_step": (
        ("samples", "k_cache", "v_cache", "x_len", "y_len", "idx"),
        ("topk_values", "topk_indices", "k_cache_new", "v_cache_new"),
    ),
    "spectrogram": (("audio",), ("spectrogram",)),
    "sv_embedding": (("audio",), ("sv_embedding",)),
    "sovits": (
        ("pred_semantic", "text_seq", "refer_spec", "sv_emb", "noise_scale"),
        ("audio",),
    ),
    "sovits_stream": (
        (
            "pred_semantic",
            "text_seq",
            "refer_spec",
            "sv_emb",
            "noise_scale",
            "result_length",
            "overlap_frames",
            "overlap_enabled",
            "acoustic_noise",
        ),
        ("audio", "latent", "latent_mask"),
    ),
}

# v1.0 streaming engines generated acoustic noise internally. v1.1 accepts an
# explicit buffer so full and streaming decode can share the same noise field.
# Keep the input optional only when loading an existing v1.0 package; newly
# converted engines are still validated against the complete contract above.
OPTIONAL_LEGACY_STAGE_INPUTS = {
    "sovits_stream": ("acoustic_noise",),
}

SUPPORTED_LANGUAGES = ("zh", "yue", "en", "ja", "ko")
LEGACY_LANGUAGE_ALIASES = ("auto", "auto_yue")


@dataclass(frozen=True)
class ModelPaths:
    gpt_checkpoint: Path
    sovits_checkpoint: Path
    reference_audio: Path
    onnx_dir: Path
    engine_dir: Path
    tokenizer_dir: Path

    def validated(self) -> "ModelPaths":
        missing = [
            str(path)
            for path in (self.gpt_checkpoint, self.sovits_checkpoint, self.reference_audio)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError("Missing model assets: " + ", ".join(missing))
        return self


def stage_outputs_supported(stage: str, outputs) -> bool:
    """Accept legacy outputs or the explicit full-logit extension, never arbitrary extras."""
    actual = set(outputs)
    if len(actual) != len(outputs):
        return False
    required = set(STAGE_IO_CONTRACTS[stage][1])
    return actual == required or (
        stage in {"gpt_encoder", "gpt_step"} and actual == required | {"logits"}
    )


def validate_full_logits_output(stage: str, outputs, engine, trt) -> bool:
    if stage not in {"gpt_encoder", "gpt_step"} or "logits" not in outputs:
        return False
    shape = tuple(engine.get_tensor_shape("logits"))
    if (len(shape) != 2 or shape[0] not in {1, -1} or shape[1] != 1025
            or engine.get_tensor_dtype("logits") != trt.float32
            or engine.get_tensor_location("logits") != trt.TensorLocation.DEVICE):
        raise RuntimeError(f"{stage} full logits require FP32 device output [batch, 1025]")
    return True
