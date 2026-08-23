#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
import wave
from pathlib import Path
from typing import Any


LANGUAGE_CASES = {
    "zh": "今天天氣很好。",
    "yue": "今日天氣好好。",
    "en": "The weather is nice today.",
    "ja": "今日はいい天気ですね。",
    "ko": "오늘 날씨가 좋네요.",
}


def call(
    host: str,
    port: int,
    path: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection(host, port, timeout=120)
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"} if payload else {}
    connection.request(method, path, payload, headers)
    response = connection.getresponse()
    response_body = response.read()
    response_headers = {key.lower(): value for key, value in response.getheaders()}
    status = response.status
    connection.close()
    return status, response_headers, response_body


def validate_wav(payload: bytes) -> dict[str, Any]:
    with wave.open(io.BytesIO(payload), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise RuntimeError("Expected mono PCM16 WAV")
        frames = audio.getnframes()
        sample_rate = audio.getframerate()
    if frames <= 0:
        raise RuntimeError("TTS returned an empty WAV")
    return {
        "sample_rate": sample_rate,
        "frames": frames,
        "duration_seconds": frames / sample_rate,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_pcm_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9882)
    parser.add_argument("--model", required=True)
    parser.add_argument("--voice-profile", default="default")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"schema": 1, "languages": {}, "adapters": {}}
    for language, text in LANGUAGE_CASES.items():
        request = {
            "model": args.model,
            "voice_profile": args.voice_profile,
            "text": text,
            "language": language,
            "stream": False,
            "generation": {"top_k": 15, "top_p": 1.0, "temperature": 1.0, "seed": 1234},
        }
        status, headers, payload = call(
            args.host, args.port, "/v1/audio/speech", method="POST", body=request
        )
        if status != 200:
            raise RuntimeError(f"{language} TTS failed with HTTP {status}: {payload[:500]!r}")
        if headers.get("x-tensorrt-backend") != "TensorRT-11":
            raise RuntimeError(f"{language} did not report TensorRT-11")
        if headers.get("x-pytorch-fallback") != "false":
            raise RuntimeError(f"{language} reported a fallback")
        metadata = validate_wav(payload)
        path = args.output_dir / f"canonical-{language}.wav"
        path.write_bytes(payload)
        report["languages"][language] = {
            "status": status,
            "path": str(path),
            "backend": headers["x-tensorrt-backend"],
            "pytorch_fallback": False,
            **metadata,
        }

    ja_request = {
        "model": args.model,
        "voice_profile": args.voice_profile,
        "text": LANGUAGE_CASES["ja"],
        "language": "ja",
        "stream": True,
        "generation": {"top_k": 15, "top_p": 1.0, "temperature": 1.0, "seed": 1234},
    }
    status, headers, pcm = call(
        args.host, args.port, "/v1/audio/speech", method="POST", body=ja_request
    )
    if status != 200 or not pcm or len(pcm) % 2:
        raise RuntimeError(f"Streaming PCM validation failed: HTTP {status}, bytes={len(pcm)}")
    stream_path = args.output_dir / "canonical-ja-stream.wav"
    stream_rate = int(headers["x-tts-sample-rate"])
    write_pcm_wav(stream_path, pcm, stream_rate)
    report["stream"] = {
        "status": status,
        "path": str(stream_path),
        "transport": headers.get("x-tts-stream"),
        "sample_rate": stream_rate,
        "pcm_bytes": len(pcm),
        "sha256": hashlib.sha256(pcm).hexdigest(),
    }

    legacy = {
        "text": LANGUAGE_CASES["ja"],
        "text_language": "ja",
        "seed": 1234,
    }
    status, headers, payload = call(args.host, args.port, "/", method="POST", body=legacy)
    report["adapters"]["legacy_flat"] = {"status": status, **validate_wav(payload)}

    openai = {
        "model": args.model,
        "voice": args.voice_profile,
        "input": LANGUAGE_CASES["en"],
        "text_lang": "en",
        "seed": 1234,
    }
    status, headers, payload = call(
        args.host, args.port, "/v1/audio/speech", method="POST", body=openai
    )
    report["adapters"]["openai_input_voice"] = {"status": status, **validate_wav(payload)}

    expression = {
        "model": args.model,
        "voice_profile": args.voice_profile,
        "text": LANGUAGE_CASES["ja"],
        "language": "ja",
        "expression": {"enabled": True, "profile": "happy", "intensity": 0.7},
    }
    status, _, payload = call(
        args.host, args.port, "/v1/audio/speech", method="POST", body=expression
    )
    error = json.loads(payload)
    if status != 501 or error.get("error", {}).get("code") != "expression_not_implemented":
        raise RuntimeError("Expression reservation contract failed")
    report["expression"] = {"status": status, "code": error["error"]["code"]}

    status, _, payload = call(
        args.host,
        args.port,
        "/v1/audio/speech",
        method="POST",
        body={"model": "wrong", "voice_profile": "default", "text": "x", "language": "ja"},
    )
    if status != 400:
        raise RuntimeError(f"Invalid model should return HTTP 400, got {status}")
    report["invalid_input"] = {"status": status, "body": json.loads(payload)}
    report["status"] = "passed"
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
