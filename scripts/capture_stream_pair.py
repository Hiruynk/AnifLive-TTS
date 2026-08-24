#!/usr/bin/env python
from __future__ import annotations

import argparse
import http.client
import json
import time
import wave
from pathlib import Path


def request(
    host: str,
    port: int,
    model: str,
    voice_profile: str,
    stream: bool,
    *,
    text: str,
    language: str,
    seed: int,
    noise_scale: float,
) -> tuple[bytes, dict[str, str], float]:
    body = json.dumps(
        {
            "model": model,
            "voice_profile": voice_profile,
            "text": text,
            "language": language,
            "stream": stream,
            "generation": {
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "seed": seed,
                "noise_scale": noise_scale,
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")
    connection = http.client.HTTPConnection(host, port, timeout=120)
    started = time.perf_counter()
    connection.request("POST", "/v1/audio/speech", body, {"Content-Type": "application/json"})
    response = connection.getresponse()
    first = response.read(1024) if stream else b""
    ttfa = time.perf_counter() - started if stream else 0.0
    payload = first + response.read()
    headers = {key.lower(): value for key, value in response.getheaders()}
    status = response.status
    connection.close()
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {payload[:500]!r}")
    return payload, headers, ttfa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9882)
    parser.add_argument("--model", required=True)
    parser.add_argument("--voice-profile", default="default")
    parser.add_argument("--text", default="今日はいい天気ですね。")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_options = {
        "text": args.text,
        "language": args.language,
        "seed": args.seed,
        "noise_scale": args.noise_scale,
    }
    full, _, _ = request(
        args.host,
        args.port,
        args.model,
        args.voice_profile,
        False,
        **request_options,
    )
    stream, headers, ttfa = request(
        args.host,
        args.port,
        args.model,
        args.voice_profile,
        True,
        **request_options,
    )
    (args.output_dir / "full.wav").write_bytes(full)
    with wave.open(str(args.output_dir / "stream.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(int(headers["x-tts-sample-rate"]))
        audio.writeframes(stream)
    result = {
        "ttfa_seconds": ttfa,
        "stream_pcm_bytes": len(stream),
        "seed": args.seed,
        "noise_scale": args.noise_scale,
    }
    (args.output_dir / "capture.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
