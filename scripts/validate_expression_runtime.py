from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", default="shy")
    parser.add_argument("--intensity", type=float, default=0.7)
    parser.add_argument("--policy", default="identity-lock")
    parser.add_argument("--text", default="今日はいい天気ですね。")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    return parser


def main() -> int:
    args = _parser().parse_args()
    repo = args.repo.resolve()
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
    startup_started = time.perf_counter()
    runtime.load()
    startup_seconds = time.perf_counter() - startup_started
    options = service_module.SynthesisOptions(
        text=args.text,
        text_language=args.language,
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        speed=1.0,
        pause_length=0.0,
        noise_scale=0.5,
        cut_punc="",
        seed=1234,
        expression_enabled=True,
        expression_profile=args.profile,
        expression_intensity=args.intensity,
        expression_policy=service_module.ConditioningPolicy(args.policy),
    )
    try:
        runtime.validate_expression(options)
        catalog = runtime.expression_metadata()
        prepared = sorted(runtime._streamer.reference_bank.references)
        expected_ids = {runtime._streamer.reference_bank.identity_id}
        expected_ids.update(item.id for item in runtime._expression_catalog.profiles)
        if set(prepared) != expected_ids:
            raise RuntimeError("Not every expression reference was prepared")
        for _ in range(args.warmup):
            runtime.synthesize(options)
            list(runtime.stream_pcm(options))

        ttfa_rows: list[float] = []
        wall_rows: list[float] = []
        rtf_rows: list[float] = []
        pcm_hashes: list[str] = []
        pcm_payload = b""
        for _ in range(args.runs):
            started = time.perf_counter()
            iterator = iter(runtime.stream_pcm(options))
            first = next(iterator)
            ttfa_rows.append(time.perf_counter() - started)
            pcm_payload = b"".join((first, *iterator))
            elapsed = time.perf_counter() - started
            duration = len(pcm_payload) / 2.0 / runtime.sample_rate
            wall_rows.append(elapsed)
            rtf_rows.append(elapsed / duration)
            pcm_hashes.append(hashlib.sha256(pcm_payload).hexdigest())
        full = runtime.synthesize(options)
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{args.profile}-{args.policy}.wav").write_bytes(full.wav)
        report: dict[str, Any] = {
            "schema": 1,
            "model": os.environ["ANIFLIVE_TTS_MODEL_ID"],
            "profile": args.profile,
            "intensity": args.intensity,
            "policy": args.policy,
            "startup_seconds": startup_seconds,
            "prepared_reference_ids": prepared,
            "catalog": catalog,
            "stream_ttfa_seconds": _summary(ttfa_rows),
            "stream_wall_seconds": _summary(wall_rows),
            "stream_rtf": _summary(rtf_rows),
            "stream_deterministic": len(set(pcm_hashes)) == 1,
            "stream_pcm_sha256": pcm_hashes[0],
            "complete_wav_sha256": hashlib.sha256(full.wav).hexdigest(),
            "complete_profile": full.profile,
        }
        report_path = output_dir / f"{args.profile}-{args.policy}.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(report_path)
        return 0
    finally:
        runtime.unload()


if __name__ == "__main__":
    raise SystemExit(main())
