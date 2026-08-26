#!/usr/bin/env python
from __future__ import annotations

import argparse
import http.client
import io
import json
import statistics
import time
import wave
from pathlib import Path
from typing import Any


def _float_header(headers: dict[str, str], name: str, default: float = 0.0) -> float:
    value = headers.get(name.lower())
    return default if value is None else float(value)


def _int_header(headers: dict[str, str], name: str, default: int = 0) -> int:
    value = headers.get(name.lower())
    return default if value is None else int(value)


def _p95(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty sample")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.95 + 0.999999) - 1))]


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "mean": 0.0}
    return {
        "count": len(values),
        "p50": statistics.median(values),
        "p95": _p95(values),
        "mean": statistics.fmean(values),
    }


def _activate(host: str, port: int, model: str, timeout: float) -> None:
    body = json.dumps({"model": model}).encode("utf-8")
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request(
            "POST",
            "/v1/models/activate",
            body,
            {"Content-Type": "application/json", "Connection": "close"},
        )
        response = connection.getresponse()
        payload = response.read()
    finally:
        connection.close()
    if response.status != 200:
        raise RuntimeError(f"Model activation failed with HTTP {response.status}: {payload[:500]!r}")


def _request(
    host: str,
    port: int,
    model: str,
    text: str,
    language: str,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "voice_profile": "default",
            "text": text,
            "language": language,
            "stream": False,
            "generation": {
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "seed": seed,
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    started = time.perf_counter()
    try:
        connection.request(
            "POST",
            "/v1/audio/speech",
            body,
            {"Content-Type": "application/json; charset=utf-8", "Connection": "close"},
        )
        response = connection.getresponse()
        payload = response.read()
        elapsed = time.perf_counter() - started
        headers = {key.lower(): value for key, value in response.getheaders()}
    finally:
        connection.close()
    if response.status != 200:
        raise RuntimeError(f"Synthesis failed with HTTP {response.status}: {payload[:500]!r}")
    with wave.open(io.BytesIO(payload), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
    tokens = _int_header(headers, "X-TTS-Semantic-Tokens")
    nfe = _int_header(headers, "X-TTS-Semantic-NFE", _int_header(headers, "X-TTS-GPT-Steps"))
    decode_seconds = _float_header(headers, "X-TTS-Stage-GPT-Decode-Seconds")
    return {
        "wall_ms": elapsed * 1000.0,
        "audio_seconds": duration,
        "server_ms": _float_header(headers, "X-TTS-Inference-Seconds") * 1000.0,
        "gpt_encoder_ms": _float_header(headers, "X-TTS-Stage-GPT-Encoder-Seconds") * 1000.0,
        "gpt_decode_ms": decode_seconds * 1000.0,
        "first_preview_semantic_ms": _float_header(
            headers, "X-TTS-First-Preview-Semantic-Seconds"
        )
        * 1000.0,
        "semantic_tokens": tokens,
        "semantic_nfe": nfe,
        "semantic_tokens_per_nfe": (tokens / nfe) if nfe else 0.0,
        "semantic_tokens_per_second": (tokens / decode_seconds) if decode_seconds else 0.0,
        "host_sync_count": _int_header(headers, "X-TTS-Host-Sync-Count"),
        "host_sync_ms": _float_header(headers, "X-TTS-Host-Sync-Seconds") * 1000.0,
        "attention_kv_bytes": _int_header(headers, "X-TTS-Attention-KV-Bytes"),
        "mamba_state_bytes": _int_header(headers, "X-TTS-Mamba-State-Bytes"),
        "semantic_backend": headers.get("x-tts-semantic-backend", "transformer"),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "wall_ms",
        "server_ms",
        "gpt_encoder_ms",
        "gpt_decode_ms",
        "first_preview_semantic_ms",
        "semantic_tokens",
        "semantic_nfe",
        "semantic_tokens_per_nfe",
        "semantic_tokens_per_second",
        "host_sync_count",
        "host_sync_ms",
        "attention_kv_bytes",
        "mamba_state_bytes",
    )
    return {
        "semantic_backends": sorted({str(row["semantic_backend"]) for row in rows}),
        "metrics": {name: _summary([float(row[name]) for row in rows]) for name in metrics},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure the AnifLive-TTS semantic hot path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9881)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--text", default="今日はいい天気ですね。")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.warmups < 0 or args.runs < 1:
        raise ValueError("warmups must be >= 0 and runs must be >= 1")
    model_reports: dict[str, Any] = {}
    for model in args.models:
        _activate(args.host, args.port, model, args.timeout)
        for index in range(args.warmups):
            _request(
                args.host,
                args.port,
                model,
                args.text,
                args.language,
                args.seed + index,
                args.timeout,
            )
        rows = [
            _request(
                args.host,
                args.port,
                model,
                args.text,
                args.language,
                args.seed + args.warmups + index,
                args.timeout,
            )
            for index in range(args.runs)
        ]
        model_reports[model] = {"summary": _summarize(rows), "runs": rows}
    report = {
        "schema": "aniflive-tts-semantic-microbenchmark-v1",
        "concurrency": 1,
        "connection_reuse": False,
        "warmups_per_model": args.warmups,
        "runs_per_model": args.runs,
        "text": args.text,
        "language": args.language,
        "models": model_reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

