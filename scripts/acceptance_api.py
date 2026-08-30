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
    parser.add_argument("--expression-profile", default="languid-1")
    parser.add_argument("--expected-first-context-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema": 2,
        "languages": {},
        "expression_languages": {},
        "adapters": {},
    }

    status, _, payload = call(args.host, args.port, "/health")
    health = json.loads(payload)
    if status != 200 or not health.get("ready") or health.get("model") != args.model:
        raise RuntimeError(f"Health contract failed: HTTP {status}: {health!r}")
    status, _, payload = call(args.host, args.port, "/model/config")
    config = json.loads(payload)
    expected_config = {
        "model": args.model,
        "version": "v2ProPlus",
        "backend": "TensorRT-11",
        "engine_count": 9,
        "pytorch_fallback": False,
    }
    for field, expected in expected_config.items():
        if config.get(field) != expected:
            raise RuntimeError(
                f"Model config {field} mismatch: {config.get(field)!r} != {expected!r}"
            )
    status, _, payload = call(args.host, args.port, "/v1/expressions")
    expressions = json.loads(payload)
    expression_ids = {item["id"] for item in expressions.get("profiles", [])}
    if status != 200 or args.expression_profile not in expression_ids:
        raise RuntimeError("Expression catalog does not expose the requested profile")
    if expressions.get("preferred_policy") != "semantic-style":
        raise RuntimeError("Expression catalog preferred policy is not semantic-style")
    report["runtime"] = {
        "health": health,
        "config": config,
        "expression_profile": args.expression_profile,
        "preferred_policy": expressions["preferred_policy"],
        "first_context_tokens": expressions["runtime_policy"]["first_context_tokens"],
    }
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
        stream_request = {**request, "stream": True}
        status, headers, pcm = call(
            args.host,
            args.port,
            "/v1/audio/speech",
            method="POST",
            body=stream_request,
        )
        if status != 200 or not pcm or len(pcm) % 2:
            raise RuntimeError(
                f"{language} streaming PCM failed: HTTP {status}, bytes={len(pcm)}"
            )
        if headers.get("x-tensorrt-backend") != "TensorRT-11":
            raise RuntimeError(f"{language} stream did not report TensorRT-11")
        if headers.get("x-pytorch-fallback") != "false":
            raise RuntimeError(f"{language} stream reported a fallback")
        if headers.get("x-tts-first-context-tokens") != "17":
            raise RuntimeError(f"{language} neutral stream changed the v1.2 context")
        stream_path = args.output_dir / f"canonical-{language}-stream.wav"
        stream_rate = int(headers["x-tts-sample-rate"])
        write_pcm_wav(stream_path, pcm, stream_rate)
        report["languages"][language]["stream"] = {
            "path": str(stream_path),
            "transport": headers.get("x-tts-stream"),
            "sample_rate": stream_rate,
            "pcm_bytes": len(pcm),
            "sha256": hashlib.sha256(pcm).hexdigest(),
            "first_context_tokens": 17,
        }

        controlled_request = {
            **stream_request,
            "expression": {
                "enabled": True,
                "profile": args.expression_profile,
                "intensity": 1.0,
            },
        }
        status, headers, controlled_pcm = call(
            args.host,
            args.port,
            "/v1/audio/speech",
            method="POST",
            body=controlled_request,
        )
        if status != 200 or not controlled_pcm or len(controlled_pcm) % 2:
            raise RuntimeError(
                f"{language} controlled stream failed: HTTP {status}, "
                f"bytes={len(controlled_pcm)}"
            )
        if headers.get("x-tts-expression-policy") != "semantic-style":
            raise RuntimeError(f"{language} did not inherit semantic-style")
        if headers.get("x-tts-expression") != args.expression_profile:
            raise RuntimeError(f"{language} expression profile header mismatch")
        if headers.get("x-tts-first-context-tokens") != str(
            args.expected_first_context_tokens
        ):
            raise RuntimeError(f"{language} first-context policy header mismatch")
        if headers.get("x-tensorrt-backend") != "TensorRT-11":
            raise RuntimeError(f"{language} controlled stream is not TensorRT-11")
        if headers.get("x-pytorch-fallback") != "false":
            raise RuntimeError(f"{language} controlled stream reported a fallback")
        report["expression_languages"][language] = {
            "status": status,
            "profile": headers["x-tts-expression"],
            "policy": headers["x-tts-expression-policy"],
            "first_context_tokens": int(headers["x-tts-first-context-tokens"]),
            "pcm_bytes": len(controlled_pcm),
            "sha256": hashlib.sha256(controlled_pcm).hexdigest(),
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
