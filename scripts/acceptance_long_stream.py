#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np


CASES = {
    "punctuated": (
        "今日はとてもいい天気ですね、少し散歩をしてから、温かい飲み物を飲みましょう。"
        "それから今日の予定を一緒に確認して、焦らず自然に進めていきましょう。"
    ),
    "unpunctuated": (
        "今日はとてもいい天気なので少し散歩をしてから温かい飲み物を飲んで"
        "そのあと今日の予定を一緒に確認しながら焦らず自然に進めていきましょう"
    ),
}


def _internal_silence(pcm: bytes, sample_rate: int) -> dict[str, float | int]:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    frame_samples = max(1, int(round(sample_rate * 0.010)))
    threshold = 10.0 ** (-45.0 / 20.0)
    active = []
    for start in range(0, samples.size, frame_samples):
        frame = samples[start : start + frame_samples]
        rms = math.sqrt(float(np.mean(np.square(frame, dtype=np.float64))))
        active.append(rms > threshold)
    indices = [index for index, value in enumerate(active) if value]
    if not indices:
        raise RuntimeError("Streaming response contains no audible PCM")
    longest = current = 0
    for value in active[indices[0] : indices[-1] + 1]:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return {
        "first_active_seconds": indices[0] * 0.010,
        "last_active_seconds": min(samples.size / sample_rate, (indices[-1] + 1) * 0.010),
        "max_internal_silence_seconds": longest * 0.010,
    }


def _request(host: str, port: int, model: str, text: str) -> tuple[bytes, dict[str, Any]]:
    body = json.dumps(
        {
            "model": model,
            "voice_profile": "default",
            "text": text,
            "language": "ja",
            "stream": True,
            "generation": {
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "seed": 1234,
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")
    connection = http.client.HTTPConnection(host, port, timeout=120)
    started = time.perf_counter()
    connection.request(
        "POST",
        "/v1/audio/speech",
        body,
        {"Content-Type": "application/json; charset=utf-8"},
    )
    response = connection.getresponse()
    headers = {key.lower(): value for key, value in response.getheaders()}
    chunks: list[bytes] = []
    arrivals: list[float] = []
    while chunk := response.read1(65536):
        chunks.append(chunk)
        arrivals.append(time.perf_counter() - started)
    status = response.status
    connection.close()
    payload = b"".join(chunks)
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {payload[:500]!r}")
    if headers.get("x-tensorrt-backend") != "TensorRT-11":
        raise RuntimeError("Streaming request did not report TensorRT-11")
    if headers.get("x-pytorch-fallback") != "false":
        raise RuntimeError("Streaming request reported PyTorch fallback")
    if not payload or len(payload) % 2:
        raise RuntimeError(f"Invalid PCM response length: {len(payload)}")
    sample_rate = int(headers["x-tts-sample-rate"])
    gaps = [later - earlier for earlier, later in zip(arrivals, arrivals[1:])]
    return payload, {
        "status": status,
        "backend": headers["x-tensorrt-backend"],
        "pytorch_fallback": False,
        "sample_rate": sample_rate,
        "pcm_bytes": len(payload),
        "duration_seconds": len(payload) / 2 / sample_rate,
        "wall_seconds": time.perf_counter() - started,
        "read1_calls": len(chunks),
        "first_payload_seconds": arrivals[0],
        "max_payload_gap_seconds": max(gaps, default=0.0),
        "sha256": hashlib.sha256(payload).hexdigest(),
        **_internal_silence(payload, sample_rate),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9881)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schema": 1, "model": args.model, "cases": {}}
    for name, text in CASES.items():
        pcm, result = _request(args.host, args.port, args.model, text)
        path = args.output_dir / f"{name}.wav"
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(int(result["sample_rate"]))
            audio.writeframes(pcm)
        report["cases"][name] = {"text": text, "path": str(path), **result}
    report["status"] = "passed"
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
