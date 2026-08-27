#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.v12_semantic.mtp_adapter import prompt_conditioned_targets  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(checkpoint: Path, source_dir: Path) -> torch.nn.Module:
    sys.path.insert(0, str(source_dir.resolve()))
    from GPT_SoVITS.AR.models.t2s_lightning_module import (
        Text2SemanticLightningModule,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Text2SemanticLightningModule(payload["config"], "output", is_train=False)
    model.load_state_dict(payload["weight"])
    return model.eval()


def _prompt_hidden(
    core: torch.nn.Module,
    phonemes: torch.Tensor,
    prompts: torch.Tensor,
    bert: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], int]:
    text = core.ar_text_embedding(phonemes)
    text = text + core.bert_proj(bert.transpose(1, 2))
    text = core.ar_text_position(text)
    prompt_embedding = core.ar_audio_position(core.ar_audio_embedding(prompts))
    x_len = int(text.shape[1])
    y_len = int(prompts.shape[1])
    combined = torch.cat((text, prompt_embedding), dim=1)
    x_mask = torch.cat(
        (
            text.new_zeros((x_len, x_len), dtype=torch.bool),
            text.new_ones((x_len, y_len), dtype=torch.bool),
        ),
        dim=1,
    )
    y_mask = torch.cat(
        (
            prompts.new_zeros((y_len, x_len), dtype=torch.bool),
            torch.triu(
                prompts.new_ones((y_len, y_len), dtype=torch.bool), diagonal=1
            ),
        ),
        dim=1,
    )
    hidden, k_cache, v_cache = core.t2s_transformer.process_prompt(
        combined, torch.cat((x_mask, y_mask), dim=0), None
    )
    return hidden[:, -1], k_cache, v_cache, y_len


def _consume_hidden(
    core: torch.nn.Module,
    token: torch.Tensor,
    k_cache: list[torch.Tensor],
    v_cache: list[torch.Tensor],
    *,
    y_len: int,
    index: int,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    embedding = core.ar_audio_embedding(token.reshape(1, 1))
    position = torch.tensor([y_len + index], dtype=torch.int64, device=token.device)
    pe_slice = core.ar_audio_position.pe.index_select(1, position)
    positioned = (
        embedding * core.ar_audio_position.x_scale
        + core.ar_audio_position.alpha
        * pe_slice.to(dtype=embedding.dtype, device=embedding.device)
    )
    hidden, k_cache, v_cache = core.t2s_transformer.decode_next_token(
        positioned, k_cache, v_cache, idx=None
    )
    return hidden[:, -1], k_cache, v_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--heads", type=int, choices=(2, 4), default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    checkpoint = args.gpt.resolve()
    checkpoint_sha = _sha256(checkpoint)
    manifest = json.loads(args.capture_manifest.read_text(encoding="utf-8"))
    capture = args.capture.resolve()
    if manifest.get("capture_sha256") != _sha256(capture):
        raise ValueError("Semantic capture SHA256 does not match its manifest")
    device = torch.device(args.device)
    dtype = torch.float16 if args.precision == "float16" else torch.float32
    model = _load_model(checkpoint, args.source_dir.resolve()).to(device=device, dtype=dtype)
    core = model.model
    hidden_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    record_id_rows: list[torch.Tensor] = []
    used_records = 0
    skipped_short = 0
    language_counts: Counter[str] = Counter()
    with safe_open(str(capture), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        if metadata.get("format") != "aniflive-tts-v1.2-semantic-capture-v1":
            raise ValueError("Unsupported semantic capture format")
        with torch.inference_mode():
            for record in manifest["records"]:
                prefix = record["prefix"]
                tokens = handle.get_tensor(f"{prefix}.tokens").long().to(device)
                if int(tokens.numel()) < args.heads:
                    skipped_short += 1
                    continue
                phonemes = handle.get_tensor(f"{prefix}.phoneme_ids").long().to(device)
                prompts = handle.get_tensor(f"{prefix}.prompts").long().to(device)
                bert = handle.get_tensor(f"{prefix}.bert_feature").to(
                    device=device, dtype=dtype
                )
                prompt_hidden, k_cache, v_cache, y_len = _prompt_hidden(
                    core, phonemes, prompts, bert
                )
                targets = prompt_conditioned_targets(tokens, args.heads)
                sequence_hidden = [prompt_hidden]
                for index in range(int(targets.shape[0]) - 1):
                    consumed, k_cache, v_cache = _consume_hidden(
                        core,
                        tokens[index],
                        k_cache,
                        v_cache,
                        y_len=y_len,
                        index=index,
                    )
                    sequence_hidden.append(consumed)
                hidden_rows.append(torch.cat(sequence_hidden, dim=0).cpu())
                target_rows.append(targets.cpu())
                record_id_rows.append(
                    torch.full(
                        (int(targets.shape[0]),),
                        used_records,
                        dtype=torch.int64,
                    )
                )
                used_records += 1
                language_counts[record["language"]] += 1
                if used_records % 10 == 0:
                    print(json.dumps({"processed_records": used_records}))
    if not hidden_rows:
        raise RuntimeError("No semantic capture records were long enough for MTP")
    hidden = torch.cat(hidden_rows, dim=0).contiguous()
    targets = torch.cat(target_rows, dim=0).contiguous()
    record_ids = torch.cat(record_id_rows, dim=0).contiguous()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "hidden": hidden,
            "future_targets": targets,
            "record_ids": record_ids,
        },
        str(output),
        metadata={
            "format": "aniflive-tts-v1.2-mtp-dataset-v1",
            "base_gpt_sha256": checkpoint_sha,
            "capture_sha256": _sha256(capture),
            "corpus_sha256": manifest["corpus_sha256"],
            "heads": str(args.heads),
            "precision": args.precision,
            "records": str(used_records),
            "rows": str(hidden.shape[0]),
        },
    )
    report = {
        "schema": 1,
        "kind": "aniflive-tts-v1.2-mtp-dataset",
        "dataset": str(output),
        "dataset_sha256": _sha256(output),
        "base_gpt_sha256": checkpoint_sha,
        "capture_sha256": _sha256(capture),
        "heads": args.heads,
        "precision": args.precision,
        "records": used_records,
        "skipped_short_records": skipped_short,
        "rows": int(hidden.shape[0]),
        "hidden_dim": int(hidden.shape[1]),
        "language_records": dict(sorted(language_counts.items())),
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
