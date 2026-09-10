from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPORT_SCHEMA = "aniflive-tts-long-form-context-ab-v1"
POLICIES = ("A", "B", "C", "D", "E", "F")


class BenchmarkError(RuntimeError):
    pass


def _json_request(
    base_url: str, method: str, path: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    encoded = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        base_url.rstrip("/") + path,
        data=encoded,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise BenchmarkError(f"{method} {path} failed ({error.code}): {detail}") from error
    except (URLError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"{method} {path} failed: {error}") from error
    if not isinstance(result, dict):
        raise BenchmarkError(f"{method} {path} returned a non-object JSON body")
    return result


def _pcm_has_audible_sample(payload: bytes, threshold: int) -> bool:
    usable = len(payload) - len(payload) % 2
    if usable <= 0:
        return False
    return any(
        abs(sample[0]) >= threshold
        for sample in struct.iter_unpack("<h", payload[:usable])
    )


def _stream_audio(
    base_url: str, session_id: str, *, audible_threshold: int
) -> tuple[bytes, dict[str, str], float, float, float]:
    request = Request(
        base_url.rstrip("/") + f"/v1/sessions/{session_id}/audio",
        method="GET",
    )
    started = time.perf_counter()
    first_packet_ms: float | None = None
    audible_ttfa_ms: float | None = None
    pieces: list[bytes] = []
    try:
        with urlopen(request, timeout=300) as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            while True:
                chunk = response.read(4096)
                if not chunk:
                    break
                now = time.perf_counter()
                if first_packet_ms is None:
                    first_packet_ms = (now - started) * 1000.0
                if audible_ttfa_ms is None and _pcm_has_audible_sample(
                    chunk, audible_threshold
                ):
                    audible_ttfa_ms = (now - started) * 1000.0
                pieces.append(chunk)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise BenchmarkError(f"session audio failed ({error.code}): {detail}") from error
    except (URLError, OSError) as error:
        raise BenchmarkError(f"session audio failed: {error}") from error
    if first_packet_ms is None:
        raise BenchmarkError("session audio returned no PCM")
    if audible_ttfa_ms is None:
        raise BenchmarkError("session audio contained no audible PCM")
    return (
        b"".join(pieces),
        headers,
        first_packet_ms,
        audible_ttfa_ms,
        (time.perf_counter() - started) * 1000.0,
    )


def run_once(
    *,
    base_url: str,
    policy: str,
    segments: Sequence[str],
    language: str,
    seed: int,
    model: str | None,
    voice_profile: str,
    audible_threshold: int,
) -> dict[str, Any]:
    create_payload: dict[str, Any] = {
        "voice_profile": voice_profile,
        "continuity_policy": policy,
    }
    if model is not None:
        create_payload["model"] = model
    session = _json_request(base_url, "POST", "/v1/sessions", create_payload)
    session_id = session.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise BenchmarkError("session creation did not return an id")
    try:
        for index, text in enumerate(segments):
            _json_request(
                base_url,
                "POST",
                f"/v1/sessions/{session_id}/segments",
                {
                    "segment_id": f"context-{policy}-{index:03d}",
                    "text": text,
                    "language": language,
                    "paragraph_id": "context-ab",
                    "pause_after_ms": 0,
                    "expression": {"enabled": False},
                    "generation": {"seed": seed},
                },
            )
        _json_request(base_url, "POST", f"/v1/sessions/{session_id}/flush", {})
        pcm, headers, first_packet_ms, audible_ttfa_ms, total_ms = _stream_audio(
            base_url, session_id, audible_threshold=audible_threshold
        )
        final = _json_request(base_url, "GET", f"/v1/sessions/{session_id}")
    except BaseException:
        try:
            _json_request(base_url, "POST", f"/v1/sessions/{session_id}/cancel", {})
        except BenchmarkError:
            pass
        raise
    return {
        "session_id": session_id,
        "pcm_bytes": len(pcm),
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "first_packet_ms": first_packet_ms,
        "audible_ttfa_ms": audible_ttfa_ms,
        "complete_ms": total_ms,
        "context_header": headers.get("x-tts-session-context"),
        "policy_header": headers.get("x-tts-session-context-policy"),
        "neural_state_header": headers.get("x-tts-neural-state-continuity"),
        "acoustic_latent_header": headers.get("x-tts-acoustic-latent-continuity"),
        "qualification_header": headers.get("x-tts-continuity-qualification"),
        "final_context": final.get("context"),
    }


def build_report(
    *,
    policies: Sequence[str],
    repetitions: int,
    runner: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for policy in policies:
        runs = [runner(policy) for _ in range(repetitions)]
        hashes = {run["pcm_sha256"] for run in runs}
        results[policy] = {
            "runs": runs,
            "fixed_seed_deterministic": len(hashes) == 1,
            "first_packet_ms_min": min(run["first_packet_ms"] for run in runs),
            "audible_ttfa_ms_min": min(run["audible_ttfa_ms"] for run in runs),
        }
    return {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "release_qualified": False,
        "acoustic_latent_continuity": False,
        "repetitions": repetitions,
        "policies": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-seed AnifLive-TTS Speech Session A-F context diagnostics. "
            "The report is experimental evidence, not a release qualification."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:9880")
    parser.add_argument("--model")
    parser.add_argument("--voice-profile", default="default")
    parser.add_argument("--language", choices=("zh", "yue", "en", "ja", "ko"), required=True)
    parser.add_argument("--segment", action="append", required=True)
    parser.add_argument("--policy", action="append", choices=POLICIES)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--audible-threshold", type=int, default=256)
    parser.add_argument("--output", type=Path)
    return parser


def _write_report(path: Path, encoded: str) -> None:
    destination = path.expanduser().resolve()
    if not destination.parent.is_dir():
        raise BenchmarkError(f"output directory does not exist: {destination.parent}")
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8")
    temporary.replace(destination)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.repetitions <= 20:
        print("repetitions must be between 1 and 20", file=sys.stderr)
        return 2
    if not 1 <= args.audible_threshold <= 32767:
        print("audible-threshold must be between 1 and 32767", file=sys.stderr)
        return 2
    if any(not isinstance(segment, str) or not segment.strip() for segment in args.segment):
        print("segments must not be empty", file=sys.stderr)
        return 2
    policies = tuple(dict.fromkeys(args.policy or POLICIES))
    try:
        report = build_report(
            policies=policies,
            repetitions=args.repetitions,
            runner=lambda policy: run_once(
                base_url=args.base_url,
                policy=policy,
                segments=args.segment,
                language=args.language,
                seed=args.seed,
                model=args.model,
                voice_profile=args.voice_profile,
                audible_threshold=args.audible_threshold,
            ),
        )
        report.update(
            {
                "base_url": args.base_url,
                "language": args.language,
                "segment_count": len(args.segment),
                "seed": args.seed,
            }
        )
        encoded = json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        if args.output is not None:
            _write_report(args.output, encoded)
        print(encoded)
        return 0
    except (BenchmarkError, OSError, ValueError) as error:
        print(
            json.dumps(
                {"schema": REPORT_SCHEMA, "diagnostic_only": True, "error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
