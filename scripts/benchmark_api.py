#!/usr/bin/env python
from __future__ import annotations

import argparse
import http.client
import io
import json
import statistics
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Any

from benchmark_audio import (
    AUDIBLE_FRAME_MS,
    AUDIBLE_THRESHOLD_DBFS,
    analyze_pcm16_stream,
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * fraction) - 1))]


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


class GPUSampler:
    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=2,
                ).strip().splitlines()[0]
                util, memory, power = (float(value.strip()) for value in output.split(","))
                self.samples.append({"utilization_percent": util, "memory_mib": memory, "power_w": power})
            except Exception:
                pass
            self.stop.wait(self.interval)

    def __enter__(self) -> "GPUSampler":
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=3)


def request(host: str, port: int, body: bytes, *, stream: bool) -> dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=60)
    started = time.perf_counter()
    connection.request(
        "POST",
        "/v1/audio/speech",
        body,
        {"Content-Type": "application/json; charset=utf-8"},
    )
    response = connection.getresponse()
    stream_chunks: list[tuple[float, bytes]] = []
    if stream:
        while True:
            chunk = response.read1(64 * 1024)
            if not chunk:
                break
            stream_chunks.append((time.perf_counter() - started, chunk))
        payload = b"".join(chunk for _, chunk in stream_chunks)
    else:
        payload = response.read()
    completed = time.perf_counter()
    headers = {key.lower(): value for key, value in response.getheaders()}
    connection.close()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {payload[:500]!r}")
    if stream:
        sample_rate = int(headers["x-tts-sample-rate"])
        audio_duration = len(payload) / 2 / sample_rate
        audibility = analyze_pcm16_stream(stream_chunks, sample_rate)
    else:
        with wave.open(io.BytesIO(payload), "rb") as wav:
            sample_rate = wav.getframerate()
            audio_duration = wav.getnframes() / sample_rate
        audibility = None
    stage_headers = {
        "text_processing": "x-tts-stage-text-seconds",
        "gpt_encoder": "x-tts-stage-gpt-encoder-seconds",
        "gpt_decode": "x-tts-stage-gpt-decode-seconds",
        "sovits": "x-tts-stage-sovits-seconds",
        "audio_postprocess": "x-tts-stage-postprocess-seconds",
    }
    wall_seconds = completed - started
    server_seconds = (
        float(headers["x-tts-inference-seconds"])
        if "x-tts-inference-seconds" in headers
        else None
    )
    return {
        "wall_seconds": wall_seconds,
        "ttfp_seconds": audibility.ttfp_seconds if audibility else None,
        "audible_ttfa_seconds": (
            audibility.audible_ttfa_seconds if audibility else None
        ),
        "leading_silence_seconds": (
            audibility.leading_silence_seconds if audibility else None
        ),
        "first_active_sample": (
            audibility.first_active_sample if audibility else None
        ),
        "first_active_chunk_index": (
            audibility.first_active_chunk_index if audibility else None
        ),
        "audio_duration_seconds": audio_duration,
        "wall_rtf": wall_seconds / audio_duration,
        "server_rtf": server_seconds / audio_duration if server_seconds is not None else None,
        "bytes": len(payload),
        "backend": headers.get("x-tensorrt-backend"),
        "pytorch_fallback": headers.get("x-pytorch-fallback"),
        "server_inference_seconds": server_seconds,
        "stage_seconds": {
            stage: float(headers[header])
            for stage, header in stage_headers.items()
            if header in headers
        },
        "semantic_tokens": int(headers["x-tts-semantic-tokens"])
        if "x-tts-semantic-tokens" in headers
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9882)
    parser.add_argument("--model", required=True)
    parser.add_argument("--voice-profile", default="default")
    parser.add_argument("--text", default="今日はいい天気ですね。")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    base = {
        "model": args.model,
        "voice_profile": args.voice_profile,
        "text": args.text,
        "language": args.language,
        "generation": {"top_k": 15, "top_p": 1.0, "temperature": 1.0, "seed": 1234},
    }
    full_body = json.dumps({**base, "stream": False}, ensure_ascii=False).encode("utf-8")
    stream_body = json.dumps({**base, "stream": True}, ensure_ascii=False).encode("utf-8")
    cold = request(args.host, args.port, full_body, stream=False)
    for _ in range(args.warmup):
        request(args.host, args.port, full_body, stream=False)
    with GPUSampler() as gpu:
        full = [request(args.host, args.port, full_body, stream=False) for _ in range(args.runs)]
        streaming = [request(args.host, args.port, stream_body, stream=True) for _ in range(args.runs)]
    full_wall = [row["wall_seconds"] for row in full]
    full_server = [row["server_inference_seconds"] for row in full]
    full_wall_rtf = [row["wall_rtf"] for row in full]
    full_server_rtf = [row["server_rtf"] for row in full if row["server_rtf"] is not None]
    ttfp = [row["ttfp_seconds"] for row in streaming]
    audible_ttfa = [row["audible_ttfa_seconds"] for row in streaming]
    leading_silence = [row["leading_silence_seconds"] for row in streaming]
    stream_wall = [row["wall_seconds"] for row in streaming]
    stage_names = sorted(
        set().union(*(row["stage_seconds"].keys() for row in full))
    )
    report = {
        "schema": 1,
        "model": args.model,
        "text": args.text,
        "language": args.language,
        "warmup_runs": args.warmup,
        "formal_runs": args.runs,
        "cold": cold,
        "full_wall_seconds": summary(full_wall),
        "full_server_seconds": summary(full_server),
        "full_stage_seconds": {
            stage: summary(
                [row["stage_seconds"][stage] for row in full if stage in row["stage_seconds"]]
            )
            for stage in stage_names
        },
        "semantic_tokens": full[-1]["semantic_tokens"],
        "full_wall_rtf": summary(full_wall_rtf),
        "full_server_rtf": summary(full_server_rtf),
        "stream_ttfp_seconds": summary(ttfp),
        "stream_audible_ttfa_seconds": summary(audible_ttfa),
        "stream_leading_silence_seconds": summary(leading_silence),
        "audibility": {
            "threshold_dbfs": AUDIBLE_THRESHOLD_DBFS,
            "frame_ms": AUDIBLE_FRAME_MS,
            "device_output_latency_included": False,
        },
        "stream_full_wall_seconds": summary(stream_wall),
        "audio_duration_seconds": full[-1]["audio_duration_seconds"],
        "tensorrt_execution_proof": {
            "backend": full[-1]["backend"],
            "pytorch_fallback": full[-1]["pytorch_fallback"],
        },
        "gpu": {
            "samples": len(gpu.samples),
            "utilization_percent": summary([row["utilization_percent"] for row in gpu.samples]),
            "memory_mib": summary([row["memory_mib"] for row in gpu.samples]),
            "power_w": summary([row["power_w"] for row in gpu.samples]),
        }
        if gpu.samples
        else {"samples": 0},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
