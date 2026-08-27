#!/usr/bin/env python
"""Compare the v1.2 Hybrid student with the frozen Transformer in AR mode.

This is a research gate.  It never becomes a production runtime fallback and
does not claim acoustic quality parity; it rejects obviously divergent
semantic students before TensorRT integration is attempted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

from safetensors import safe_open
from safetensors.torch import load_file
import torch
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aniflive_tts.backend.semantic_sampling import (  # noqa: E402
    SemanticSamplingConfig,
    logits_to_probabilities,
)
from research.v12_semantic.hybrid_model import (  # noqa: E402
    HybridLayerState,
    HybridT2SBackbone,
)
from research.v12_semantic.train_hybrid import (  # noqa: E402
    load_model,
    read_record,
    sha256,
    split_records,
)


EOS = 1024


def edit_distance(left: list[int], right: list[int]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def sequence_similarity(left: list[int], right: list[int]) -> float:
    denominator = max(len(left), len(right), 1)
    return 1.0 - edit_distance(left, right) / denominator


def positional_agreement(left: list[int], right: list[int], count: int) -> float:
    compared = min(len(left), len(right), count)
    if compared == 0:
        return 0.0
    return sum(left[index] == right[index] for index in range(compared)) / compared


def adjacent_repetition_rate(tokens: list[int]) -> float:
    if len(tokens) < 2:
        return 0.0
    return sum(left == right for left, right in zip(tokens, tokens[1:])) / (
        len(tokens) - 1
    )


def _prompt_inputs(
    core: nn.Module,
    phonemes: Tensor,
    prompts: Tensor,
    bert: Tensor,
) -> tuple[Tensor, Tensor, int]:
    text = core.ar_text_position(
        core.ar_text_embedding(phonemes) + core.bert_proj(bert.transpose(1, 2))
    )
    audio = core.ar_audio_position(core.ar_audio_embedding(prompts))
    x_len = int(text.shape[1])
    y_len = int(audio.shape[1])
    text_mask = torch.cat(
        (
            torch.zeros((x_len, x_len), dtype=torch.bool, device=text.device),
            torch.ones((x_len, y_len), dtype=torch.bool, device=text.device),
        ),
        dim=1,
    )
    audio_mask = torch.cat(
        (
            torch.zeros((y_len, x_len), dtype=torch.bool, device=text.device),
            torch.triu(
                torch.ones((y_len, y_len), dtype=torch.bool, device=text.device),
                diagonal=1,
            ),
        ),
        dim=1,
    )
    return torch.cat((text, audio), dim=1), torch.cat(
        (text_mask, audio_mask), dim=0
    ), y_len


def _positioned_token(core: nn.Module, token: Tensor, *, y_len: int, index: int) -> Tensor:
    embedding = core.ar_audio_embedding(token.reshape(1, 1))
    position = torch.tensor([y_len + index], dtype=torch.int64, device=token.device)
    positional = core.ar_audio_position.pe.index_select(1, position)
    return (
        embedding * core.ar_audio_position.x_scale
        + core.ar_audio_position.alpha
        * positional.to(dtype=embedding.dtype, device=embedding.device)
    )


def _sample(
    logits: Tensor,
    history: Tensor,
    config: SemanticSamplingConfig,
    generator: torch.Generator,
) -> Tensor:
    probabilities = logits_to_probabilities(logits, history, config)
    noise = torch.empty_like(probabilities).exponential_(1, generator=generator)
    return torch.argmax(probabilities / noise, dim=-1, keepdim=True).to(torch.long)


def _generate(
    core: nn.Module,
    phonemes: Tensor,
    prompts: Tensor,
    bert: Tensor,
    *,
    seed: int,
    max_tokens: int,
    prompt_forward: Callable[[Tensor, Tensor], tuple[Tensor, Any]],
    step_forward: Callable[[Tensor, Any], tuple[Tensor, Any]],
) -> tuple[list[int], bool, float]:
    config = SemanticSamplingConfig()
    inputs, mask, y_len = _prompt_inputs(core, phonemes, prompts, bert)
    if inputs.device.type == "cuda":
        torch.cuda.synchronize(inputs.device)
    started = time.perf_counter()
    hidden, state = prompt_forward(inputs, mask)
    logits = core.ar_predict_layer(hidden[:, -1])
    history = prompts.reshape(1, -1)
    tokens: list[int] = []
    generator = torch.Generator(device=inputs.device).manual_seed(seed)
    reached_eos = False
    for index in range(max_tokens):
        sampling_logits = logits[:, :-1] if index == 0 else logits
        token = _sample(sampling_logits, history, config, generator)
        greedy_eos = index > 0 and int(logits.argmax(dim=-1).item()) == EOS
        token_value = int(token.item())
        if token_value == EOS or greedy_eos:
            reached_eos = True
            break
        tokens.append(token_value)
        history = torch.cat((history, token), dim=1)
        positioned = _positioned_token(core, token, y_len=y_len, index=index)
        hidden, state = step_forward(positioned, state)
        logits = core.ar_predict_layer(hidden[:, -1])
    if inputs.device.type == "cuda":
        torch.cuda.synchronize(inputs.device)
    return tokens, reached_eos, time.perf_counter() - started


def _transformer_functions(core: nn.Module) -> tuple[Callable[..., Any], Callable[..., Any]]:
    def prompt(hidden: Tensor, mask: Tensor) -> tuple[Tensor, tuple[Any, Any]]:
        output, k_cache, v_cache = core.t2s_transformer.process_prompt(hidden, mask, None)
        return output, (k_cache, v_cache)

    def step(hidden: Tensor, state: tuple[Any, Any]) -> tuple[Tensor, tuple[Any, Any]]:
        output, k_cache, v_cache = core.t2s_transformer.decode_next_token(
            hidden, state[0], state[1], None
        )
        return output, (k_cache, v_cache)

    return prompt, step


def _hybrid_functions(
    hybrid: HybridT2SBackbone,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    def prompt(hidden: Tensor, mask: Tensor) -> tuple[Tensor, list[HybridLayerState]]:
        return hybrid.process_prompt(hidden, mask)

    def step(
        hidden: Tensor, state: list[HybridLayerState]
    ) -> tuple[Tensor, list[HybridLayerState]]:
        return hybrid.decode_next_token(hidden, state)

    return prompt, step


def _sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_adapter(
    hybrid: HybridT2SBackbone,
    adapter_path: Path,
    checkpoint_path: Path,
) -> None:
    with safe_open(str(adapter_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if metadata.get("format") != "aniflive-tts-v1.2-hybrid-adapter-v1":
        raise ValueError("Unsupported Hybrid adapter format")
    if metadata.get("base_gpt_sha256") != _sha256_bytes(checkpoint_path):
        raise ValueError("Hybrid adapter does not match the base GPT checkpoint")
    adapter = load_file(str(adapter_path), device="cpu")
    incompatible = hybrid.mamba_layers.load_state_dict(adapter, strict=False)
    missing_trainable = [
        key for key in incompatible.missing_keys if ".mamba." in f".{key}"
    ]
    if missing_trainable or incompatible.unexpected_keys:
        raise ValueError(
            "Hybrid adapter state is incomplete or incompatible: "
            f"missing={missing_trainable}, unexpected={incompatible.unexpected_keys}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("capture_sha256") != sha256(args.capture):
        raise ValueError("Capture SHA256 does not match its manifest")
    device = torch.device(args.device)
    dtype = torch.float16 if args.precision == "float16" else torch.float32
    base = load_model(args.gpt.resolve(), args.source_dir.resolve())
    hybrid = HybridT2SBackbone(base.model)
    _load_adapter(hybrid, args.adapter.resolve(), args.gpt.resolve())
    hybrid = hybrid.to(device=device, dtype=dtype).eval()
    _, validation = split_records(
        manifest["records"], args.validation_fraction, args.split_seed
    )
    if args.limit:
        validation = validation[: args.limit]
    transformer_prompt, transformer_step = _transformer_functions(hybrid.core)
    hybrid_prompt, hybrid_step = _hybrid_functions(hybrid)
    rows: list[dict[str, Any]] = []
    with safe_open(str(args.capture), framework="pt", device="cpu") as handle:
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=device.type == "cuda" and dtype == torch.float16,
        ):
            for index, record in enumerate(validation, 1):
                phonemes, prompts, bert, captured = read_record(
                    handle, record, device=device, dtype=dtype
                )
                captured_tokens = [
                    int(value) for value in captured.tolist() if int(value) != EOS
                ]
                max_tokens = min(400, max(32, len(captured_tokens) + 64))
                seed = int(record.get("seed", args.split_seed + index))
                teacher, teacher_eos, teacher_seconds = _generate(
                    hybrid.core,
                    phonemes,
                    prompts,
                    bert,
                    seed=seed,
                    max_tokens=max_tokens,
                    prompt_forward=transformer_prompt,
                    step_forward=transformer_step,
                )
                student, student_eos, student_seconds = _generate(
                    hybrid.core,
                    phonemes,
                    prompts,
                    bert,
                    seed=seed,
                    max_tokens=max_tokens,
                    prompt_forward=hybrid_prompt,
                    step_forward=hybrid_step,
                )
                row = {
                    "prefix": record["prefix"],
                    "language": record["language"],
                    "seed": seed,
                    "captured_tokens": len(captured_tokens),
                    "teacher_tokens": len(teacher),
                    "student_tokens": len(student),
                    "teacher_eos": teacher_eos,
                    "student_eos": student_eos,
                    "teacher_student_similarity": sequence_similarity(teacher, student),
                    "teacher_student_first17_agreement": positional_agreement(
                        teacher, student, 17
                    ),
                    "teacher_capture_similarity": sequence_similarity(
                        teacher, captured_tokens
                    ),
                    "student_capture_similarity": sequence_similarity(
                        student, captured_tokens
                    ),
                    "teacher_repetition_rate": adjacent_repetition_rate(teacher),
                    "student_repetition_rate": adjacent_repetition_rate(student),
                    "teacher_seconds": teacher_seconds,
                    "student_seconds": student_seconds,
                }
                rows.append(row)
                print(json.dumps({"completed": index, "total": len(validation), **row}), flush=True)

    mean = lambda key: statistics.fmean(float(row[key]) for row in rows)
    summary = {
        "records": len(rows),
        "teacher_student_similarity_mean": mean("teacher_student_similarity"),
        "teacher_student_first17_agreement_mean": mean(
            "teacher_student_first17_agreement"
        ),
        "teacher_capture_similarity_mean": mean("teacher_capture_similarity"),
        "student_capture_similarity_mean": mean("student_capture_similarity"),
        "teacher_eos_rate": mean("teacher_eos"),
        "student_eos_rate": mean("student_eos"),
        "teacher_repetition_rate_mean": mean("teacher_repetition_rate"),
        "student_repetition_rate_mean": mean("student_repetition_rate"),
        "student_teacher_length_ratio_mean": statistics.fmean(
            row["student_tokens"] / max(row["teacher_tokens"], 1) for row in rows
        ),
        "teacher_seconds_mean": mean("teacher_seconds"),
        "student_seconds_mean": mean("student_seconds"),
    }
    proxy_pass = (
        summary["teacher_student_similarity_mean"] >= 0.85
        and summary["teacher_student_first17_agreement_mean"] >= 0.80
        and summary["student_eos_rate"] >= 0.95
        and 0.90 <= summary["student_teacher_length_ratio_mean"] <= 1.10
        and summary["student_repetition_rate_mean"]
        <= summary["teacher_repetition_rate_mean"] + 0.02
    )
    result = {
        "schema": 1,
        "kind": "aniflive-tts-v1.2-hybrid-ar-evaluation",
        "status": "pass" if proxy_pass else "fail",
        "scope": "semantic-proxy-only",
        "production_quality_gate": "not_evaluated",
        "base_gpt_sha256": _sha256_bytes(args.gpt.resolve()),
        "adapter_sha256": _sha256_bytes(args.adapter.resolve()),
        "capture_sha256": _sha256_bytes(args.capture.resolve()),
        "precision": args.precision,
        "device": str(device),
        "summary": summary,
        "rows": rows,
    }
    target = args.report.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(target), "status": result["status"], **summary}, indent=2))
    return 0 if proxy_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
