#!/usr/bin/env python
from __future__ import annotations

import argparse
import http.client
import io
import json
import math
import statistics
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from benchmark_audio import (
    AUDIBLE_FRAME_MS,
    AUDIBLE_THRESHOLD_DBFS,
    analyze_pcm16_stream,
)


METRIC_KEYS = (
    "complete_wav_wall_p50_ms",
    "complete_wav_wall_p95_ms",
    "server_inference_p50_ms",
    "wall_rtf_p50",
    "stream_ttfp_p50_ms",
    "stream_ttfp_p95_ms",
    "stream_keepalive_ttfp_p50_ms",
    "stream_keepalive_ttfp_p95_ms",
    "stream_audible_ttfa_p50_ms",
    "stream_audible_ttfa_p95_ms",
    "stream_keepalive_audible_ttfa_p50_ms",
    "stream_keepalive_audible_ttfa_p95_ms",
    "gpu_busy_p50_percent",
    "gpu_busy_p95_percent",
)

METRIC_LABELS = {
    "en": (
        "Complete REST WAV wall P50",
        "Complete REST WAV wall P95",
        "Server inference P50",
        "RTF P50",
        "Streaming first-packet latency P50",
        "Streaming first-packet latency P95",
        "Keep-alive streaming first-packet latency P50",
        "Keep-alive streaming first-packet latency P95",
        "Audible streaming TTFA P50",
        "Audible streaming TTFA P95",
        "Keep-alive audible streaming TTFA P50",
        "Keep-alive audible streaming TTFA P95",
        "GPU busy-time P50",
        "GPU busy-time P95",
    ),
    "zh-HK": (
        "完整 REST WAV 端到端 P50",
        "完整 REST WAV 端到端 P95",
        "伺服器推理 P50",
        "RTF P50",
        "串流首包延遲 P50",
        "串流首包延遲 P95",
        "持續連線串流首包延遲 P50",
        "持續連線串流首包延遲 P95",
        "串流有效音訊 TTFA P50",
        "串流有效音訊 TTFA P95",
        "持續連線串流有效音訊 TTFA P50",
        "持續連線串流有效音訊 TTFA P95",
        "GPU 佔用率 P50",
        "GPU 佔用率 P95",
    ),
    "zh-CN": (
        "完整 REST WAV 端到端 P50",
        "完整 REST WAV 端到端 P95",
        "服务器推理 P50",
        "RTF P50",
        "流式首包延迟 P50",
        "流式首包延迟 P95",
        "持久连接流式首包延迟 P50",
        "持久连接流式首包延迟 P95",
        "流式有效音频 TTFA P50",
        "流式有效音频 TTFA P95",
        "持久连接流式有效音频 TTFA P50",
        "持久连接流式有效音频 TTFA P95",
        "GPU 占用率 P50",
        "GPU 占用率 P95",
    ),
}

TABLE_HEADERS = {
    "en": ("Metric", "Median across {sessions} session-level statistics", "Session range"),
    "zh-HK": ("指標", "{sessions} 組統計值的中位數", "各組範圍"),
    "zh-CN": ("指标", "{sessions} 组统计值的中位数", "各组范围"),
}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty sample")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def p50(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty sample")
    return statistics.median(values)


@dataclass(frozen=True)
class Aggregate:
    median: float
    range_min: float
    range_max: float


class GPUSampler:
    def __init__(self, gpu_index: int, interval: float) -> None:
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=2,
                ).strip()
                lines = output.splitlines()
                if self.gpu_index >= len(lines):
                    raise RuntimeError(
                        f"GPU index {self.gpu_index} is unavailable; nvidia-smi returned "
                        f"{len(lines)} device(s)"
                    )
                utilization = float(lines[self.gpu_index].strip())
                self.samples.append({"utilization_percent": utilization})
            except Exception as error:
                self.last_error = str(error)
            self._stop.wait(self.interval)

    def __enter__(self) -> "GPUSampler":
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=3)


def health_check(host: str, port: int, timeout: float) -> dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", "/health", headers={"Connection": "close"})
        response = connection.getresponse()
        payload = response.read()
    finally:
        connection.close()
    if response.status != 200:
        raise RuntimeError(f"Health check returned HTTP {response.status}: {payload[:500]!r}")
    health = json.loads(payload.decode("utf-8"))
    if not health.get("ready") or health.get("backend") != "TensorRT-11":
        raise RuntimeError(f"API is not ready on TensorRT 11: {health!r}")
    return health


def wait_until_idle(host: str, port: int, timeout: float) -> dict[str, Any]:
    """Wait outside measured requests until the single-request runtime is idle."""

    deadline = time.monotonic() + timeout
    while True:
        health = health_check(host, port, timeout)
        if int(health.get("active_requests", 0)) == 0 and not health.get(
            "switching", False
        ):
            return health
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "AnifLive-TTS did not become idle before the benchmark session"
            )
        time.sleep(0.01)


def activate_model(host: str, port: int, model: str, timeout: float) -> dict[str, Any]:
    body = json.dumps({"model": model}).encode("utf-8")
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request(
            "POST",
            "/v1/models/activate",
            body,
            {
                "Content-Type": "application/json",
                "Connection": "close",
            },
        )
        response = connection.getresponse()
        payload = response.read()
    finally:
        connection.close()
    if response.status != 200:
        raise RuntimeError(
            f"Model activation returned HTTP {response.status}: {payload[:500]!r}"
        )
    health = health_check(host, port, timeout)
    if health.get("model") != model:
        raise RuntimeError(
            f"Model activation did not select {model!r}: {health.get('model')!r}"
        )
    return health


def request(
    host: str,
    port: int,
    path: str,
    body: bytes,
    *,
    stream: bool,
    timeout: float,
    connection: http.client.HTTPConnection | None = None,
) -> dict[str, Any]:
    owns_connection = connection is None
    if connection is None:
        connection = http.client.HTTPConnection(host, port, timeout=timeout)
    started = time.perf_counter()
    try:
        connection.request(
            "POST",
            path,
            body,
            {
                "Content-Type": "application/json; charset=utf-8",
                "Connection": "close" if owns_connection else "keep-alive",
            },
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
        status = response.status
    finally:
        if owns_connection:
            connection.close()

    if status != 200:
        raise RuntimeError(f"HTTP {status}: {payload[:500]!r}")
    backend = headers.get("x-tensorrt-backend")
    fallback = headers.get("x-pytorch-fallback")
    if backend != "TensorRT-11" or fallback != "false":
        raise RuntimeError(
            f"Request did not prove strict TensorRT execution: "
            f"backend={backend!r}, pytorch_fallback={fallback!r}"
        )

    if stream:
        sample_rate = int(headers["x-tts-sample-rate"])
        audio_duration = len(payload) / 2 / sample_rate
        audibility = analyze_pcm16_stream(stream_chunks, sample_rate)
    else:
        with wave.open(io.BytesIO(payload), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise RuntimeError("Benchmark requires mono PCM16 WAV output")
            sample_rate = wav.getframerate()
            audio_duration = wav.getnframes() / sample_rate
        audibility = None

    server_header = headers.get("x-tts-inference-seconds")
    server_seconds = float(server_header) if server_header is not None else None
    if not stream and server_seconds is None:
        raise RuntimeError("Response is missing X-TTS-Inference-Seconds")
    wall_seconds = completed - started
    return {
        "wall_seconds": wall_seconds,
        "ttfp_seconds": audibility.ttfp_seconds if audibility else None,
        "audible_ttfa_seconds": (
            audibility.audible_ttfa_seconds if audibility else None
        ),
        "leading_silence_seconds": (
            audibility.leading_silence_seconds if audibility else None
        ),
        "audio_duration_seconds": audio_duration,
        "wall_rtf": wall_seconds / audio_duration,
        "server_inference_seconds": server_seconds,
        "backend": backend,
        "pytorch_fallback": fallback,
    }


def run_session(
    args: argparse.Namespace, full_body: bytes, stream_body: bytes
) -> tuple[dict[str, float], int]:
    for _ in range(args.warmup):
        request(
            args.host,
            args.port,
            args.path,
            full_body,
            stream=False,
            timeout=args.timeout,
        )

    with GPUSampler(args.gpu_index, args.gpu_sample_interval) as gpu:
        full = [
            request(
                args.host,
                args.port,
                args.path,
                full_body,
                stream=False,
                timeout=args.timeout,
            )
            for _ in range(args.runs)
        ]
        streaming = [
            request(
                args.host,
                args.port,
                args.path,
                stream_body,
                stream=True,
                timeout=args.timeout,
            )
            for _ in range(args.runs)
        ]
        keepalive_connection = http.client.HTTPConnection(
            args.host, args.port, timeout=args.timeout
        )
        try:
            request(
                args.host,
                args.port,
                args.path,
                stream_body,
                stream=True,
                timeout=args.timeout,
                connection=keepalive_connection,
            )
            keepalive_streaming = [
                request(
                    args.host,
                    args.port,
                    args.path,
                    stream_body,
                    stream=True,
                    timeout=args.timeout,
                    connection=keepalive_connection,
                )
                for _ in range(args.runs)
            ]
        finally:
            keepalive_connection.close()

    if not gpu.samples:
        detail = f": {gpu.last_error}" if gpu.last_error else ""
        raise RuntimeError(f"nvidia-smi produced no GPU samples{detail}")

    full_wall = [row["wall_seconds"] for row in full]
    full_server = [float(row["server_inference_seconds"]) for row in full]
    full_rtf = [row["wall_rtf"] for row in full]
    stream_ttfp = [float(row["ttfp_seconds"]) for row in streaming]
    keepalive_stream_ttfp = [
        float(row["ttfp_seconds"]) for row in keepalive_streaming
    ]
    stream_audible_ttfa = [
        float(row["audible_ttfa_seconds"]) for row in streaming
    ]
    keepalive_stream_audible_ttfa = [
        float(row["audible_ttfa_seconds"]) for row in keepalive_streaming
    ]
    utilization = [row["utilization_percent"] for row in gpu.samples]
    metrics = {
        "complete_wav_wall_p50_ms": p50(full_wall) * 1000,
        "complete_wav_wall_p95_ms": percentile(full_wall, 0.95) * 1000,
        "server_inference_p50_ms": p50(full_server) * 1000,
        "wall_rtf_p50": p50(full_rtf),
        "stream_ttfp_p50_ms": p50(stream_ttfp) * 1000,
        "stream_ttfp_p95_ms": percentile(stream_ttfp, 0.95) * 1000,
        "stream_keepalive_ttfp_p50_ms": p50(keepalive_stream_ttfp) * 1000,
        "stream_keepalive_ttfp_p95_ms": (
            percentile(keepalive_stream_ttfp, 0.95) * 1000
        ),
        "stream_audible_ttfa_p50_ms": p50(stream_audible_ttfa) * 1000,
        "stream_audible_ttfa_p95_ms": percentile(stream_audible_ttfa, 0.95) * 1000,
        "stream_keepalive_audible_ttfa_p50_ms": (
            p50(keepalive_stream_audible_ttfa) * 1000
        ),
        "stream_keepalive_audible_ttfa_p95_ms": (
            percentile(keepalive_stream_audible_ttfa, 0.95) * 1000
        ),
        "gpu_busy_p50_percent": p50(utilization),
        "gpu_busy_p95_percent": percentile(utilization, 0.95),
    }
    if tuple(metrics) != METRIC_KEYS:
        raise AssertionError("Benchmark table metric contract changed")
    return metrics, len(gpu.samples)


def aggregate_sessions(sessions: list[dict[str, float]]) -> dict[str, Aggregate]:
    return {
        key: Aggregate(
            median=p50([session[key] for session in sessions]),
            range_min=min(session[key] for session in sessions),
            range_max=max(session[key] for session in sessions),
        )
        for key in METRIC_KEYS
    }


def format_value(key: str, value: float, *, headline: bool) -> str:
    if key.endswith("_ms"):
        return f"{value:.3f} ms"
    if key == "wall_rtf_p50":
        return f"{value:.6f}"
    if key.startswith("gpu_busy_"):
        return f"{value:.1f}%" if headline else f"{value:g}%"
    raise KeyError(key)


def render_markdown(
    aggregates: dict[str, Aggregate], *, locale: str, sessions: int
) -> str:
    labels = METRIC_LABELS[locale]
    headers = TABLE_HEADERS[locale]
    lines = [
        f"| {headers[0]} | {headers[1].format(sessions=sessions)} | {headers[2]} |",
        "|---|---:|---:|",
    ]
    for key, label in zip(METRIC_KEYS, labels, strict=True):
        metric = aggregates[key]
        headline = format_value(key, metric.median, headline=True)
        range_text = (
            f"{format_value(key, metric.range_min, headline=False)}"
            f"\u2013{format_value(key, metric.range_max, headline=False)}"
        )
        if key.endswith("_ms"):
            range_text = range_text.replace(" ms\u2013", "\u2013")
        elif key.startswith("gpu_busy_"):
            range_text = range_text.replace("%\u2013", "\u2013")
        lines.append(f"| {label} | **{headline}** | {range_text} |")
    return "\n".join(lines) + "\n"


def build_report(
    args: argparse.Namespace,
    health: dict[str, Any],
    aggregates: dict[str, Aggregate],
    per_model: dict[str, dict[str, Aggregate]],
    gpu_samples: int,
) -> dict[str, Any]:
    models = list(per_model)
    return {
        "schema": 3,
        "measured_at": date.today().isoformat(),
        "project": "AnifLive-TTS",
        "release": "1.1.0",
        "models": models,
        "voice_profile": args.voice_profile,
        "workload": {
            "text": args.text,
            "language": args.language,
            "seed": 1234,
            "top_k": 15,
            "top_p": 1.0,
            "temperature": 1.0,
        },
        "methodology": {
            "sessions": args.sessions,
            "models": len(models),
            "total_model_sessions": args.sessions * len(models),
            "warmup_requests_per_session": args.warmup,
            "full_wav_requests_per_session": args.runs,
            "stream_requests_per_session": args.runs,
            "keepalive_stream_requests_per_session": args.runs,
            "concurrency": 1,
            "transport": (
                "new HTTP/1.1 connection per request plus a separately reported "
                "persistent HTTP/1.1 connection workload"
            ),
            "connection_reuse": "reported separately",
            "keepalive_connection_warmup_requests_per_session": 1,
            "session_idle_barrier": (
                "health active_requests=0 and switching=false before each session; "
                "barrier time is outside every measured request"
            ),
            "ttfp_definition": (
                "client wall-clock time from sending the HTTP request until reading "
                "the first server-emitted PCM chunk"
            ),
            "audible_ttfa_definition": (
                "device-independent earliest audible playback time at the first PCM "
                "sample above threshold within the earliest active RMS frame, "
                "constrained by that PCM chunk's arrival time"
            ),
            "audible_threshold_dbfs": AUDIBLE_THRESHOLD_DBFS,
            "audible_frame_ms": AUDIBLE_FRAME_MS,
            "headline_statistic": "median of all model-session-level statistics",
        },
        "environment": {
            "gpu": health.get("gpu"),
            "cuda": health.get("cuda"),
            "tensorrt": health.get("tensorrt"),
            "engine_count": health.get("engine_count"),
        },
        "execution_proof": {
            "backend": "TensorRT-11",
            "pytorch_fallback": False,
            "formal_full_wav_requests": args.sessions * args.runs * len(models),
            "formal_stream_requests": args.sessions * args.runs * len(models),
            "formal_keepalive_stream_requests": (
                args.sessions * args.runs * len(models)
            ),
            "gpu_samples": gpu_samples,
        },
        "table": [
            {
                "key": key,
                "label": METRIC_LABELS[args.locale][index],
                "median": aggregates[key].median,
                "range_min": aggregates[key].range_min,
                "range_max": aggregates[key].range_max,
            }
            for index, key in enumerate(METRIC_KEYS)
        ],
        "session_distribution": {
            key: {
                "session_median": aggregates[key].median,
                "best": aggregates[key].range_min,
                "worst": aggregates[key].range_max,
            }
            for key in METRIC_KEYS
        },
        "per_model": {
            model: {
                key: {
                    "session_median": values[key].median,
                    "best": values[key].range_min,
                    "worst": values[key].range_max,
                }
                for key in METRIC_KEYS
            }
            for model, values in per_model.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the fourteen-metric AnifLive-TTS README benchmark table"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9882)
    parser.add_argument("--path", default="/v1/audio/speech")
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Model id to benchmark; repeat to aggregate multiple voice packages.",
    )
    parser.add_argument("--voice-profile")
    parser.add_argument("--text", default="今日はいい天気ですね。")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--locale", choices=tuple(METRIC_LABELS), default="en")
    parser.add_argument("--report", type=Path, default=Path("reports/benchmark.json"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/benchmark.md"))
    args = parser.parse_args()
    if args.sessions < 1 or args.runs < 1 or args.warmup < 0:
        parser.error("sessions and runs must be positive; warmup must be non-negative")
    if args.gpu_index < 0 or args.gpu_sample_interval <= 0 or args.timeout <= 0:
        parser.error("GPU index must be non-negative and intervals/timeouts must be positive")
    if not args.path.startswith("/"):
        parser.error("path must start with '/'")
    return args


def main() -> int:
    args = parse_args()
    health = health_check(args.host, args.port, args.timeout)
    models = args.models or [health.get("model")]
    args.voice_profile = args.voice_profile or health.get("voice") or "default"
    if not all(models):
        raise RuntimeError("Model id is missing; pass --model explicitly")

    sessions: list[dict[str, float]] = []
    per_model: dict[str, dict[str, Aggregate]] = {}
    gpu_samples = 0
    for model in models:
        if health.get("model") != model:
            print(f"[benchmark] activating model {model}", file=sys.stderr, flush=True)
            health = activate_model(args.host, args.port, model, args.timeout)
        base = {
            "model": model,
            "voice_profile": args.voice_profile,
            "text": args.text,
            "language": args.language,
            "generation": {
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "seed": 1234,
            },
        }
        full_body = json.dumps(
            {**base, "stream": False}, ensure_ascii=False
        ).encode("utf-8")
        stream_body = json.dumps(
            {**base, "stream": True}, ensure_ascii=False
        ).encode("utf-8")
        model_sessions: list[dict[str, float]] = []
        for index in range(args.sessions):
            print(
                f"[benchmark] {model} session {index + 1}/{args.sessions}: "
                f"warmup={args.warmup}, full={args.runs}, stream={args.runs}, "
                f"keepalive_stream={args.runs}",
                file=sys.stderr,
                flush=True,
            )
            health = wait_until_idle(args.host, args.port, args.timeout)
            metrics, samples = run_session(args, full_body, stream_body)
            model_sessions.append(metrics)
            sessions.append(metrics)
            gpu_samples += samples
        per_model[model] = aggregate_sessions(model_sessions)

    aggregates = aggregate_sessions(sessions)
    total_sessions = args.sessions * len(models)
    markdown = render_markdown(aggregates, locale=args.locale, sessions=total_sessions)
    report = build_report(args, health, aggregates, per_model, gpu_samples)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"[benchmark] JSON: {args.report}", file=sys.stderr)
    print(f"[benchmark] Markdown: {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
