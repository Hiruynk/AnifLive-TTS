#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
import statistics
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


STAGE_HEADERS = {
    "text": "X-TTS-Stage-Text-Seconds",
    "gpt_encoder": "X-TTS-Stage-GPT-Encoder-Seconds",
    "gpt_decode": "X-TTS-Stage-GPT-Decode-Seconds",
    "sovits": "X-TTS-Stage-SoVITS-Seconds",
    "postprocess": "X-TTS-Stage-Postprocess-Seconds",
}


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95 = ordered[max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))]
    return {
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": p95,
        "min": ordered[0],
        "max": ordered[-1],
    }


@dataclass(frozen=True)
class Target:
    name: str
    url: str


class GPUSampler:
    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
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
                self.samples.append({"utilization": util, "memory_mib": memory, "power_w": power})
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "GPUSampler":
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=3)


class HTTPClient:
    """One persistent HTTP/1.1 connection per backend for warm inference."""

    def __init__(self, target: Target) -> None:
        parsed = urlsplit(target.url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError(f"Unsupported benchmark URL: {target.url}")
        self.target = target
        self.path = parsed.path or "/"
        self.connection = http.client.HTTPConnection(
            parsed.hostname, parsed.port or 80, timeout=180
        )

    def close(self) -> None:
        self.connection.close()

    def request(self, payload: bytes) -> tuple[dict[str, Any], bytes]:
        started = time.perf_counter()
        self.connection.request(
            "POST",
            self.path,
            body=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        response = self.connection.getresponse()
        body = response.read()
        headers = response.headers
        status = response.status
        wall = time.perf_counter() - started
        return _parse_response(self.target, status, headers, body, wall)


def _parse_response(
    target: Target,
    status: int,
    headers: Any,
    body: bytes,
    wall: float,
) -> tuple[dict[str, Any], bytes]:
    if status != 200:
        raise RuntimeError(f"{target.name} returned HTTP {status}")
    if headers.get("X-TensorRT-Backend") != "TensorRT-11":
        raise RuntimeError(f"{target.name} did not prove TensorRT 11 execution")
    if headers.get("X-PyTorch-Fallback") != "false":
        raise RuntimeError(f"{target.name} reported a PyTorch fallback")
    with wave.open(io.BytesIO(body), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise RuntimeError(f"{target.name} returned an unexpected WAV format")
        duration = wav.getnframes() / wav.getframerate()
        sample_rate = wav.getframerate()
    server = float(headers["X-TTS-Inference-Seconds"])
    return (
        {
            "wall_seconds": wall,
            "server_seconds": server,
            "wall_rtf": wall / duration,
            "server_rtf": server / duration,
            "audio_duration_seconds": duration,
            "sample_rate": sample_rate,
            "semantic_tokens": int(headers["X-TTS-Semantic-Tokens"]),
            "gpt_steps": int(headers["X-TTS-GPT-Steps"]),
            "sample_cuda_graph": headers.get("X-TTS-Sample-CUDA-Graph") == "true",
            "sha256": hashlib.sha256(body).hexdigest(),
            "stages": {
                name: float(headers[header])
                for name, header in STAGE_HEADERS.items()
                if headers.get(header) is not None
            },
        },
        body,
    )


def _target_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_names = sorted(set().union(*(row["stages"].keys() for row in rows)))
    return {
        "wall_seconds": _summary([row["wall_seconds"] for row in rows]),
        "server_seconds": _summary([row["server_seconds"] for row in rows]),
        "wall_rtf": _summary([row["wall_rtf"] for row in rows]),
        "server_rtf": _summary([row["server_rtf"] for row in rows]),
        "stages": {
            name: _summary([row["stages"][name] for row in rows]) for name in stage_names
        },
        "semantic_tokens": sorted({row["semantic_tokens"] for row in rows}),
        "gpt_steps": sorted({row["gpt_steps"] for row in rows}),
        "sample_cuda_graph": sorted({row["sample_cuda_graph"] for row in rows}),
        "audio_duration_seconds": sorted({row["audio_duration_seconds"] for row in rows}),
        "sha256": sorted({row["sha256"] for row in rows}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Matched AnifLive-TTS A/B API benchmark")
    parser.add_argument("--baseline-url", default="http://127.0.0.1:9880/")
    parser.add_argument("--candidate-url", default="http://127.0.0.1:9882/")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--text", default="今日はいい天気ですね。")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("warmup must be non-negative and runs must be positive")

    targets = (
        Target("baseline", args.baseline_url),
        Target("aniflive_tts", args.candidate_url),
    )
    payload = json.dumps(
        {
            "text": args.text,
            "text_language": "ja",
            "top_k": 15,
            "top_p": 1.0,
            "temperature": 1.0,
            "speed": 1.0,
            "noise_scale": 0.5,
            "pause_length": 0.0,
            "seed": 1234,
            "stream": False,
            "response_format": "wav",
        },
        ensure_ascii=False,
    ).encode("utf-8")

    cold: dict[str, Any] = {}
    rows: dict[str, list[dict[str, Any]]] = {target.name: [] for target in targets}
    audio: dict[str, bytes] = {}
    for target in targets:
        cold_client = HTTPClient(target)
        try:
            cold[target.name], audio[target.name] = cold_client.request(payload)
        finally:
            cold_client.close()
    clients = {target.name: HTTPClient(target) for target in targets}
    try:
        for _ in range(args.warmup):
            for target in targets:
                clients[target.name].request(payload)

        with GPUSampler() as gpu:
            for index in range(args.runs):
                order = targets if index % 2 == 0 else tuple(reversed(targets))
                for target in order:
                    row, body = clients[target.name].request(payload)
                    rows[target.name].append(row)
                    audio[target.name] = body
    finally:
        for client in clients.values():
            client.close()

    summaries = {name: _target_summary(values) for name, values in rows.items()}
    baseline = summaries[targets[0].name]
    candidate = summaries[targets[1].name]
    matched_output = (
        baseline["semantic_tokens"] == candidate["semantic_tokens"]
        and baseline["audio_duration_seconds"] == candidate["audio_duration_seconds"]
        and baseline["sha256"] == candidate["sha256"]
    )
    p50_gain = 1.0 - candidate["wall_seconds"]["p50"] / baseline["wall_seconds"]["p50"]
    wall_rtf_gain = (
        1.0 - candidate["wall_rtf"]["p50"] / baseline["wall_rtf"]["p50"]
    )
    server_rtf_gain = (
        1.0 - candidate["server_rtf"]["p50"] / baseline["server_rtf"]["p50"]
    )
    report = {
        "schema": 1,
        "workload": {"text": args.text, "language": "ja", "seed": 1234, "top_k": 15},
        "warmup_runs": args.warmup,
        "formal_runs_per_target": args.runs,
        "transport": "HTTP/1.1 persistent connections after cold request",
        "cold": cold,
        "targets": summaries,
        "comparison": {
            "matched_output": matched_output,
            "complete_wav_p50_gain": p50_gain,
            "wall_rtf_p50_gain": wall_rtf_gain,
            "server_rtf_p50_gain": server_rtf_gain,
            "release_gate": matched_output and p50_gain >= 0.10 and server_rtf_gain >= 0.10,
        },
        "gpu_combined": {
            "samples": len(gpu.samples),
            "utilization": _summary([row["utilization"] for row in gpu.samples]),
            "memory_mib": _summary([row["memory_mib"] for row in gpu.samples]),
            "power_w": _summary([row["power_w"] for row in gpu.samples]),
        }
        if gpu.samples
        else {"samples": 0},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.audio_dir:
        args.audio_dir.mkdir(parents=True, exist_ok=True)
        for name, body in audio.items():
            (args.audio_dir / f"{name}.wav").write_bytes(body)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
