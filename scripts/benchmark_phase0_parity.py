#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))
    return {
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "p95": ordered[p95_index],
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
    }


def _options(service_module: Any, text: str) -> Any:
    return service_module.SynthesisOptions(
        text=text,
        text_language="ja",
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        speed=1.0,
        pause_length=0.0,
        noise_scale=0.5,
        cut_punc="",
        seed=1234,
    )


def _capture_semantics(streamer: Any) -> tuple[list[list[int]], Callable[[], None]]:
    original = streamer._decode_native_stream_chunk
    captured: list[list[int]] = []

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        cumulative = kwargs.get("cumulative_tokens")
        if cumulative is not None:
            tokens = [int(value) for value in cumulative.detach().cpu().flatten().tolist()]
            if not captured or len(tokens) >= len(captured[-1]):
                captured.append(tokens)
        return original(*args, **kwargs)

    streamer._decode_native_stream_chunk = wrapped

    def restore() -> None:
        streamer._decode_native_stream_chunk = original

    return captured, restore


def _stream_once(service: Any, options: Any) -> tuple[bytes, float, float, float]:
    started = time.perf_counter()
    iterator = iter(service.stream_pcm(options))
    try:
        first = next(iterator)
    except StopIteration as error:
        raise RuntimeError("Streaming inference returned no PCM") from error
    ttfa = time.perf_counter() - started
    chunks = [first, *iterator]
    wall = time.perf_counter() - started
    pcm = b"".join(chunks)
    duration = len(pcm) / 2.0 / float(service.sample_rate)
    return pcm, ttfa, wall, wall / duration


def _capture_runtime(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    import os

    os.environ.update(
        {
            "ANIFLIVE_TTS_MODEL_PACKAGE": str(args.model_package.resolve()),
            "ANIFLIVE_TTS_SHARED_DIR": str(args.shared_dir.resolve()),
            "ANIFLIVE_TTS_SOURCE_DIR": str((repo / "minimal_inference").resolve()),
            "ANIFLIVE_TTS_WARM_RETENTION_SECONDS": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    from aniflive_tts.api import configure_runtime

    configure_runtime()
    from aniflive_tts import service as service_module

    runtime = service_module.TensorRTService(service_module.RuntimeSettings.from_env())
    runtime.load()
    options = _options(service_module, args.text)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        full_tokens, restore = _capture_semantics(runtime._streamer)
        full_result = runtime.synthesize(options)
        restore()
        if not full_tokens:
            raise RuntimeError("No complete-WAV semantic sequence was captured")

        stream_tokens, restore = _capture_semantics(runtime._streamer)
        stream_pcm, _, _, _ = _stream_once(runtime, options)
        restore()
        if not stream_tokens:
            raise RuntimeError("No streaming semantic sequence was captured")

        (output_dir / "full.wav").write_bytes(full_result.wav)
        (output_dir / "stream.pcm").write_bytes(stream_pcm)

        for _ in range(args.warmup):
            runtime.synthesize(options)
            _stream_once(runtime, options)

        full_rows: list[dict[str, float]] = []
        stream_rows: list[dict[str, float]] = []
        full_hashes: set[str] = set()
        stream_hashes: set[str] = set()
        for _ in range(args.runs):
            started = time.perf_counter()
            result = runtime.synthesize(options)
            full_wall = time.perf_counter() - started
            full_hashes.add(_sha256(result.wav))
            full_rows.append(
                {
                    "wall_seconds": full_wall,
                    "server_seconds": float(result.elapsed_seconds),
                    "rtf": full_wall / (result.output_samples / result.sample_rate),
                    "gpt_decode_seconds": float(
                        result.profile.get("gpt_decode_seconds", 0.0)
                    ),
                }
            )
            pcm, ttfa, wall, rtf = _stream_once(runtime, options)
            stream_hashes.add(_sha256(pcm))
            stream_rows.append(
                {
                    "ttfa_seconds": ttfa,
                    "wall_seconds": wall,
                    "rtf": rtf,
                }
            )

        report = {
            "schema": 1,
            "source_commit": subprocess.check_output(
                ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "repo": str(repo),
            "warmup_runs": args.warmup,
            "formal_runs": args.runs,
            "workload": {"text": args.text, "language": "ja", "seed": 1234},
            "exact": {
                "full_semantic_tokens": full_tokens[-1],
                "stream_semantic_tokens": stream_tokens[-1],
                "full_wav_sha256": _sha256(full_result.wav),
                "stream_pcm_sha256": _sha256(stream_pcm),
                "full_wav_hashes_during_formal_runs": sorted(full_hashes),
                "stream_pcm_hashes_during_formal_runs": sorted(stream_hashes),
            },
            "performance": {
                "full_wall_seconds": _summary(
                    [row["wall_seconds"] for row in full_rows]
                ),
                "full_server_seconds": _summary(
                    [row["server_seconds"] for row in full_rows]
                ),
                "full_rtf": _summary([row["rtf"] for row in full_rows]),
                "gpt_decode_seconds": _summary(
                    [row["gpt_decode_seconds"] for row in full_rows]
                ),
                "stream_ttfa_seconds": _summary(
                    [row["ttfa_seconds"] for row in stream_rows]
                ),
                "stream_wall_seconds": _summary(
                    [row["wall_seconds"] for row in stream_rows]
                ),
                "stream_rtf": _summary([row["rtf"] for row in stream_rows]),
            },
        }
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(args.report)
        return 0
    finally:
        runtime.unload()


def _regression(candidate: float, baseline: float) -> float:
    return candidate / baseline - 1.0


def _compare(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, Any]] = {}
    for name, repo in (("baseline", args.baseline_repo), ("candidate", args.candidate_repo)):
        target = output / name
        target.mkdir(parents=True, exist_ok=True)
        report_path = target / "capture.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "capture",
            "--repo",
            str(repo.resolve()),
            "--model-package",
            str(args.model_package.resolve()),
            "--shared-dir",
            str(args.shared_dir.resolve()),
            "--output-dir",
            str(target),
            "--report",
            str(report_path),
            "--text",
            args.text,
            "--warmup",
            str(args.warmup),
            "--runs",
            str(args.runs),
        ]
        subprocess.run(command, check=True, cwd=repo.resolve())
        reports[name] = json.loads(report_path.read_text(encoding="utf-8"))

    baseline = reports["baseline"]
    candidate = reports["candidate"]
    baseline_exact = baseline["exact"]
    candidate_exact = candidate["exact"]
    exact = {
        "complete_semantic_tokens": (
            baseline_exact["full_semantic_tokens"]
            == candidate_exact["full_semantic_tokens"]
        ),
        "stream_semantic_tokens": (
            baseline_exact["stream_semantic_tokens"]
            == candidate_exact["stream_semantic_tokens"]
        ),
        "complete_wav_sha256": (
            baseline_exact["full_wav_sha256"]
            == candidate_exact["full_wav_sha256"]
        ),
        "stream_pcm_sha256": (
            baseline_exact["stream_pcm_sha256"]
            == candidate_exact["stream_pcm_sha256"]
        ),
        "baseline_deterministic": (
            len(baseline_exact["full_wav_hashes_during_formal_runs"]) == 1
            and len(baseline_exact["stream_pcm_hashes_during_formal_runs"]) == 1
        ),
        "candidate_deterministic": (
            len(candidate_exact["full_wav_hashes_during_formal_runs"]) == 1
            and len(candidate_exact["stream_pcm_hashes_during_formal_runs"]) == 1
        ),
    }
    regression = {
        metric: _regression(
            candidate["performance"][metric]["p50"],
            baseline["performance"][metric]["p50"],
        )
        for metric in (
            "full_wall_seconds",
            "full_rtf",
            "gpt_decode_seconds",
            "stream_ttfa_seconds",
            "stream_rtf",
        )
    }
    report = {
        "schema": 1,
        "phase": "v1.2-phase0-semantic-runtime-extraction",
        "baseline": baseline,
        "candidate": candidate,
        "exact_parity": exact,
        "p50_regression": regression,
        "gate": {
            "exact_parity_passed": all(exact.values()),
            "performance_passed": (
                regression["stream_ttfa_seconds"] <= 0.015
                and regression["full_rtf"] <= 0.015
            ),
        },
    }
    report["gate"]["passed"] = all(report["gate"].values())
    report_path = output / "phase0-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["gate"], indent=2))
    print(report_path)
    return 0 if report["gate"]["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--repo", type=Path, required=True)
    capture.add_argument("--model-package", type=Path, required=True)
    capture.add_argument("--shared-dir", type=Path, required=True)
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument("--report", type=Path, required=True)
    capture.add_argument("--text", required=True)
    capture.add_argument("--warmup", type=int, default=3)
    capture.add_argument("--runs", type=int, default=10)
    capture.set_defaults(handler=_capture_runtime)

    compare = commands.add_parser("compare")
    compare.add_argument("--baseline-repo", type=Path, required=True)
    compare.add_argument("--candidate-repo", type=Path, required=True)
    compare.add_argument("--model-package", type=Path, required=True)
    compare.add_argument("--shared-dir", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument("--text", default="今日はいい天気ですね。")
    compare.add_argument("--warmup", type=int, default=3)
    compare.add_argument("--runs", type=int, default=10)
    compare.set_defaults(handler=_compare)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("warmup must be non-negative and runs must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
