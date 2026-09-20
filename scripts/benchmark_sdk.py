"""Matched raw HTTP versus SDK measurements; no audio is written to disk."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import statistics
import time
import wave
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DEFAULT_TEXT = "今日はいい天気ですね。"
GENERATION = {"top_k": 15, "top_p": 1.0, "temperature": 1.0,
              "seed": 1234, "noise_scale": 0.5, "speed": 1.0}
METRICS = ("wall_seconds", "headers_seconds", "first_packet_seconds",
           "audible_ttfa_seconds", "steady_bytes_per_second")


def summary(values: list[float]) -> dict[str, float]:
    """Median and nearest-rank p95; never silently accept invalid measurements."""
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Measurements must be finite, nonnegative and nonempty")
    ordered = sorted(values)
    return {"p50": statistics.median(ordered),
            "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "min": ordered[0], "max": ordered[-1]}


def validate_url(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("Use an HTTP(S) base URL without credentials, query or fragment")
    return value.rstrip("/")


def _audibility(chunks, sample_rate):
    # Reuse the repository's device-independent threshold/frame algorithm.
    if __package__:
        from .benchmark_audio import analyze_pcm16_stream
    else:
        from benchmark_audio import analyze_pcm16_stream
    return analyze_pcm16_stream(chunks, sample_rate)


def _wav_duration(audio: bytes) -> float:
    with wave.open(io.BytesIO(audio), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("Expected mono PCM16 WAV")
        if wav.getframerate() <= 0 or wav.getnframes() == 0:
            raise ValueError("Expected nonempty WAV")
        return wav.getnframes() / wav.getframerate()


def measure(client, request: dict[str, Any], *, streaming: bool,
            max_audio_bytes: int = 16 * 1024 * 1024, clock=time.perf_counter):
    """Measure the public call boundary, including transport and server work."""
    if max_audio_bytes <= 0:
        raise ValueError("Audio memory limit must be positive")
    started = clock()
    if not streaming:
        result = client.synthesize(**request)
        completed = clock()
        audio = result.audio
        if len(audio) > max_audio_bytes:
            raise ValueError("Audio exceeds benchmark memory limit")
        return {"wall_seconds": completed - started, "bytes": len(audio),
                "audio_duration_seconds": _wav_duration(audio)}
    stream = client.stream(**request)
    headers_elapsed = clock() - started
    chunks = []
    size = 0
    try:
        if (stream.sample_format != "s16le" or stream.channels != 1
                or stream.sample_rate <= 0):
            raise ValueError("Expected mono PCM16 stream metadata")
        for chunk in stream:
            if not chunk:
                continue
            elapsed = clock() - started
            size += len(chunk)
            if size > max_audio_bytes:
                raise ValueError("Audio exceeds benchmark memory limit")
            chunks.append((elapsed, bytes(chunk)))
        completed = clock()
        audibility = _audibility(chunks, stream.sample_rate)
        duration = audibility.total_samples / stream.sample_rate
    finally:
        stream.close()
    steady_seconds = chunks[-1][0] - chunks[0][0]
    steady_bytes = size - len(chunks[0][1])
    return {"wall_seconds": completed - started, "headers_seconds": headers_elapsed,
            "first_packet_seconds": audibility.ttfp_seconds,
            "audible_ttfa_seconds": audibility.audible_ttfa_seconds,
            "steady_bytes_per_second": (
                steady_bytes / steady_seconds if steady_seconds > 0 else None),
            "bytes": size, "audio_duration_seconds": duration}


class RawHTTP:
    """Benchmark baseline only; this is not an official Python SDK transport."""

    def __init__(self, base_url: str, *, timeout: float, max_audio_bytes: int):
        import httpx
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0),
            follow_redirects=False, trust_env=False,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        )
        self._url = validate_url(base_url) + "/v1/audio/speech"
        self._maximum = max_audio_bytes

    def close(self):
        self._client.close()

    def synthesize(self, **request):
        from types import SimpleNamespace
        with self._client.stream("POST", self._url,
                                 json={**request, "stream": False,
                                       "response_format": "wav"}) as response:
            self._check_status(response)
            if response.headers.get("content-type", "").split(";")[0] != "audio/wav":
                raise ValueError("Expected WAV content type")
            body = bytearray()
            for chunk in response.iter_bytes():
                if len(body) + len(chunk) > self._maximum:
                    raise ValueError("Audio exceeds benchmark memory limit")
                body.extend(chunk)
        return SimpleNamespace(audio=bytes(body))

    @staticmethod
    def _check_status(response):
        if response.status_code != 200:
            # Never include URL, text, response bodies or headers in diagnostics.
            raise RuntimeError("Benchmark endpoint returned non-success status")

    def stream(self, **request):
        manager = self._client.stream("POST", self._url,
                                      json={**request, "stream": True,
                                            "response_format": "pcm"})
        response = manager.__enter__()
        try:
            self._check_status(response)
            if response.headers.get("content-type", "").split(";")[0] != "application/octet-stream":
                raise ValueError("Expected PCM content type")
            return RawStream(manager, response)
        except BaseException:
            manager.__exit__(None, None, None)
            raise


class RawStream:
    def __init__(self, manager, response):
        self._manager = manager
        self._response = response
        self.sample_rate = int(response.headers["x-tts-sample-rate"])
        self.sample_format = response.headers["x-tts-sample-format"]
        self.channels = int(response.headers["x-tts-channels"])

    def __iter__(self):
        return self._response.iter_bytes()

    def close(self):
        self._manager.__exit__(None, None, None)


class SDKAdapter:
    def __init__(self, base_url: str, *, timeout: float):
        # Lazy import lets helper tests run before native implementation exists.
        from aniflive_tts import Client, Generation
        self._generation = Generation
        self._client = Client(base_url, connect_timeout=10.0,
                              request_timeout=timeout, stream_idle_timeout=timeout)

    def _request(self, request):
        return {**request, "generation": self._generation(**request["generation"])}

    def synthesize(self, **request):
        return self._client.synthesize(**self._request(request))

    def stream(self, **request):
        return self._client.stream(**self._request(request))

    def close(self):
        self._client.close()


def aggregate(rows):
    result = {}
    flags = []
    for mode in ("complete", "stream"):
        result[mode] = {}
        for metric in METRICS:
            sets = {}
            for name in ("raw", "sdk"):
                values = [r["measurement"].get(metric) for r in rows
                          if r["mode"] == mode and r["client"] == name]
                values = [v for v in values if v is not None]
                if values:
                    sets[name] = summary(values)
            if len(sets) != 2:
                continue
            deltas = {}
            for quantile in ("p50", "p95"):
                raw = sets["raw"][quantile]
                sdk = sets["sdk"][quantile]
                percent = 100 * (sdk - raw) / raw if raw else None
                deltas[quantile] = {"absolute": sdk - raw, "percent": percent}
                if (metric in {"first_packet_seconds", "audible_ttfa_seconds"}
                        and percent is not None and percent > 10):
                    flags.append({"mode": mode, "metric": metric, "quantile": quantile,
                                  "delta_percent": percent, "action": "investigate"})
            result[mode][metric] = {**sets, "sdk_minus_raw": deltas}
    return result, flags


def run_pairs(raw, sdk, request, *, pairs=30, warmups=3, max_audio_bytes=16 * 1024 * 1024,
              measure_fn=measure):
    """Warm both paths and alternate path/mode order; fail instead of biasing samples."""
    if pairs < 1 or warmups < 0:
        raise ValueError("Pairs must be positive and warmups nonnegative")
    clients = {"raw": raw, "sdk": sdk}
    rows = []
    for index in range(warmups + pairs):
        formal = index >= warmups
        pair = index - warmups if formal else index
        order = ("raw", "sdk") if pair % 2 == 0 else ("sdk", "raw")
        modes = ("complete", "stream") if pair % 2 == 0 else ("stream", "complete")
        for mode in modes:
            for name in order:
                measurement = measure_fn(clients[name], request, streaming=mode == "stream",
                                         max_audio_bytes=max_audio_bytes)
                if formal:
                    rows.append({"pair": pair, "mode": mode, "client": name,
                                 "order_in_pair": order.index(name),
                                 "measurement": measurement})
    aggregates, flags = aggregate(rows)
    return {
        "schema": "aniflive-tts-sdk-benchmark-v1",
        "qualification": "measurements-only; real-server qualification pending",
        "pairs_per_mode": pairs, "warmup_pairs_per_mode": warmups,
        "percentile_method": "median p50; nearest-rank p95",
        "request": {"language": request["language"], "text_characters": len(request["text"]),
                    "text_sha256": hashlib.sha256(request["text"].encode("utf-8")).hexdigest(),
                    "model": request.get("model"), "voice_profile": request.get("voice_profile"),
                    "generation": dict(request["generation"])},
        "connection_reuse": {"mode": "persistent client instances, sequential requests",
                             "verified": False, "evidence": "not instrumented; pending"},
        "audibility": {"threshold_dbfs": -45.0, "frame_ms": 10.0,
                       "device_output_latency_included": False},
        "interpretation": [
            "Deltas include server scheduling/inference and transport noise, not pure client overhead.",
            "Repeated >10% first-packet/TTFA regression needs matched-run investigation.",
            "No automatic performance pass/fail or connection-reuse qualification.",
            "Headers time includes server response preparation; no pure pre-body client-overhead claim.",
            "Steady throughput excludes first chunk; single-chunk streams report no steady throughput.",
            "Raw timeout is per network operation; SDK complete request timeout is a total deadline.",
        ],
        "summary": aggregates, "investigation_flags": flags, "samples": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9880")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--language", choices=("zh", "yue", "en", "ja", "ko"), default="ja")
    parser.add_argument("--model")
    parser.add_argument("--voice-profile")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-audio-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--output", type=Path, default=Path(".local-sdk/benchmark-sdk.json"))
    args = parser.parse_args(argv)
    try:
        base_url = validate_url(args.base_url)
        if (not args.text.strip() or len(args.text) > 1000 or args.seed < 0
                or args.pairs < 1 or args.warmups < 0 or args.max_audio_bytes <= 0
                or not math.isfinite(args.timeout) or args.timeout <= 0):
            raise ValueError("Invalid benchmark configuration")
        request = {"text": args.text, "language": args.language,
                   "generation": {**GENERATION, "seed": args.seed}}
        if args.model:
            request["model"] = args.model
        if args.voice_profile:
            request["voice_profile"] = args.voice_profile
        with ExitStack() as resources:
            raw = RawHTTP(base_url, timeout=args.timeout, max_audio_bytes=args.max_audio_bytes)
            resources.callback(raw.close)
            sdk = SDKAdapter(base_url, timeout=args.timeout)
            resources.callback(sdk.close)
            report = run_pairs(raw, sdk, request, pairs=args.pairs, warmups=args.warmups,
                               max_audio_bytes=args.max_audio_bytes)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                         allow_nan=False) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 - redact arbitrary SDK/transport exception payloads
        # SDK/HTTP exception strings may contain service bodies or credentials.
        print("Benchmark failed; no new report written. Check configuration, SDK availability and service.")
        return 1
    print("Benchmark report written; inspect investigation flags. Qualification remains pending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
