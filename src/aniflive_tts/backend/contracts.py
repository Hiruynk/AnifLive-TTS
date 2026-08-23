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
