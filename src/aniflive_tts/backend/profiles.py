from __future__ import annotations

from .trt_builder import ShapeRange


FITTED_PROFILE: dict[str, dict[str, ShapeRange]] = {
    "bert": {
        name: ShapeRange((1, 1), (1, 100), (1, 256))
        for name in ("input_ids", "attention_mask", "token_type_ids")
    },
    "gpt_encoder": {
        "phoneme_ids": ShapeRange((1, 1), (1, 100), (1, 256)),
        "prompts": ShapeRange((1, 1), (1, 150), (1, 300)),
        "bert_feature": ShapeRange((1, 1024, 1), (1, 1024, 100), (1, 1024, 256)),
    },
    "gpt_step": {
        "samples": ShapeRange((1, 1), (1, 1), (1, 1)),
        "k_cache": ShapeRange(
            (24, 1, 1000, 512), (24, 1, 1000, 512), (24, 1, 1000, 512)
        ),
        "v_cache": ShapeRange(
            (24, 1, 1000, 512), (24, 1, 1000, 512), (24, 1, 1000, 512)
        ),
        "x_len": ShapeRange((1,), (1,), (1,)),
        "y_len": ShapeRange((1,), (1,), (1,)),
        "idx": ShapeRange((1,), (1,), (1,)),
    },
    "sovits": {
        "pred_semantic": ShapeRange((1, 1, 1), (1, 1, 120), (1, 1, 250)),
        "text_seq": ShapeRange((1, 1), (1, 50), (1, 100)),
        "refer_spec": ShapeRange((1, 1025, 1), (1, 1025, 280), (1, 1025, 512)),
    },
    "ssl": {"audio": ShapeRange((1, 16000), (1, 96000), (1, 200000))},
    "vq_encoder": {
        "ssl_content": ShapeRange((1, 768, 50), (1, 768, 300), (1, 768, 700))
    },
    "spectrogram": {"audio": ShapeRange((1, 1), (1, 180000), (1, 400000))},
    "sv_embedding": {"audio": ShapeRange((1, 16000), (1, 90000), (1, 180000))},
}


def profiles_for(stage: str) -> tuple[dict[str, ShapeRange], ...]:
    try:
        return (FITTED_PROFILE[stage],)
    except KeyError as error:
        raise KeyError(f"No fitted TensorRT profile for stage {stage!r}") from error
