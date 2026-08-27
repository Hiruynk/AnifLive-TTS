#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_corpus(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        text = str(row.get("text", "")).strip()
        language = str(row.get("language", "")).strip()
        if not text or language not in {"zh", "yue", "en", "ja", "ko"}:
            raise ValueError(f"Invalid corpus row at line {line_number}")
        rows.append(
            {
                "id": str(row.get("id", f"row-{line_number:06d}")),
                "text": text,
                "language": language,
                "seed": int(row.get("seed", 1234 + line_number)),
            }
        )
    if not rows:
        raise ValueError("Calibration corpus is empty")
    return rows


def _cpu_tensor(value: Any) -> torch.Tensor:
    return value.detach().cpu().contiguous()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")

    repo = args.repo.resolve()
    corpus_path = args.corpus.resolve()
    corpus = _load_corpus(corpus_path)
    if args.limit:
        corpus = corpus[: args.limit]
    sys.path.insert(0, str(repo / "src"))
    os.environ.update(
        {
            "ANIFLIVE_TTS_MODEL_PACKAGE": str(args.model_package.resolve()),
            "ANIFLIVE_TTS_SHARED_DIR": str(args.shared_dir.resolve()),
            "ANIFLIVE_TTS_SOURCE_DIR": str((repo / "minimal_inference").resolve()),
            "ANIFLIVE_TTS_WARM_RETENTION_SECONDS": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    from aniflive_tts.api import configure_runtime

    configure_runtime()
    from aniflive_tts import service as service_module

    runtime = service_module.TensorRTService(service_module.RuntimeSettings.from_env())
    runtime.load()
    semantic_runtime = runtime._streamer.semantic_runtime
    original_prepare = semantic_runtime.prepare
    original_iter_batches = semantic_runtime.iter_batches
    active_item: dict[str, Any] = {}
    state_records: dict[int, dict[str, Any]] = {}
    captured: list[dict[str, Any]] = []

    def capture_prepare(
        inputs: Mapping[str, Any], **options: Any
    ) -> Any:
        state = original_prepare(inputs, **options)
        record = {
            "item_id": active_item["id"],
            "language": active_item["language"],
            "seed": active_item["seed"],
            "segment": sum(
                row["item_id"] == active_item["id"] for row in captured
            ),
            "inputs": {
                "phoneme_ids": _cpu_tensor(inputs["phoneme_ids"]),
                "prompts": _cpu_tensor(inputs["prompts"]),
                "bert_feature": _cpu_tensor(inputs["bert_feature"]),
            },
            "parts": [_cpu_tensor(state.first_token).to(torch.int64)],
            "eos": False,
        }
        captured.append(record)
        state_records[id(state)] = record
        return state

    def capture_iter_batches(
        state: Any,
        *,
        sync_policy: Any,
        cancelled: Any | None = None,
    ) -> Iterator[Any]:
        record = state_records[id(state)]
        for batch in original_iter_batches(
            state, sync_policy=sync_policy, cancelled=cancelled
        ):
            if batch.accepted_tokens:
                record["parts"].append(
                    _cpu_tensor(batch.tokens[:, : batch.accepted_tokens]).to(torch.int64)
                )
            record["eos"] = bool(record["eos"] or batch.eos)
            yield batch

    semantic_runtime.prepare = capture_prepare
    semantic_runtime.iter_batches = capture_iter_batches
    try:
        for index, item in enumerate(corpus, 1):
            active_item.clear()
            active_item.update(item)
            runtime.synthesize(
                service_module.SynthesisOptions(
                    text=item["text"],
                    text_language=item["language"],
                    top_k=15,
                    top_p=1.0,
                    temperature=1.0,
                    speed=1.0,
                    pause_length=0.0,
                    noise_scale=0.5,
                    cut_punc="",
                    seed=item["seed"],
                )
            )
            if index % 10 == 0 or index == len(corpus):
                print(json.dumps({"captured_items": index, "total": len(corpus)}))
    finally:
        semantic_runtime.prepare = original_prepare
        semantic_runtime.iter_batches = original_iter_batches
        runtime.unload()

    tensors: dict[str, torch.Tensor] = {}
    manifest_rows: list[dict[str, Any]] = []
    for index, record in enumerate(captured):
        prefix = f"record_{index:06d}"
        tokens = torch.cat(record["parts"], dim=1).reshape(-1)
        if record["eos"]:
            tokens = torch.cat((tokens, torch.tensor([1024], dtype=torch.int64)))
        tensors[f"{prefix}.phoneme_ids"] = record["inputs"]["phoneme_ids"]
        tensors[f"{prefix}.prompts"] = record["inputs"]["prompts"]
        tensors[f"{prefix}.bert_feature"] = record["inputs"]["bert_feature"]
        tensors[f"{prefix}.tokens"] = tokens.contiguous()
        manifest_rows.append(
            {
                "prefix": prefix,
                "item_id": record["item_id"],
                "language": record["language"],
                "seed": record["seed"],
                "segment": record["segment"],
                "semantic_tokens": int(tokens.numel()),
                "eos": bool(record["eos"]),
            }
        )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(output),
        metadata={
            "format": "aniflive-tts-v1.2-semantic-capture-v1",
            "corpus_sha256": _sha256(corpus_path),
            "records": str(len(manifest_rows)),
        },
    )
    manifest = (args.manifest or output.with_suffix(".json")).resolve()
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "aniflive-tts-v1.2-semantic-capture",
                "capture": str(output),
                "capture_sha256": _sha256(output),
                "corpus": str(corpus_path),
                "corpus_sha256": _sha256(corpus_path),
                "model_package": str(args.model_package.resolve()),
                "records": manifest_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
