from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SEGMENTS = (
    ("今日は静かに話したいことがあります。", "neutral", 0.5),
    ("あなたに伝えるのは、少し恥ずかしいです。", "shy", 0.7),
    ("でも、守るべきもののためなら絶対に退きません！", "battle", 0.9),
    ("もう大丈夫です。無事で本当によかった。", "relieved", 0.7),
)


def _wav(sample_rate: int, pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(pcm)
    return output.getvalue()


def _boundary_metrics(
    pcm: bytes, sample_rate: int, segment_profiles: list[dict[str, Any]]
) -> list[dict[str, float | int | str]]:
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    window = max(1, int(round(sample_rate * 0.010)))
    rows: list[dict[str, float | int | str]] = []
    boundary = 0
    for index, profile in enumerate(segment_profiles[:-1]):
        boundary += int(profile.get("published_samples", 0))
        if boundary <= 0 or boundary >= int(audio.size):
            continue
        before = audio[max(0, boundary - window) : boundary]
        after = audio[boundary : min(int(audio.size), boundary + window)]
        local = audio[max(0, boundary - window) : min(int(audio.size), boundary + window)]
        derivative = np.abs(np.diff(local.astype(np.float64, copy=False)))
        rows.append(
            {
                "after_segment": index,
                "sample": boundary,
                "seconds": boundary / sample_rate,
                "kind": str(profile.get("boundary_kind", "unknown")),
                "sample_jump": float(abs(float(audio[boundary]) - float(audio[boundary - 1]))),
                "rms_before_10ms": float(
                    np.sqrt(np.mean(np.square(before, dtype=np.float64)))
                ),
                "rms_after_10ms": float(
                    np.sqrt(np.mean(np.square(after, dtype=np.float64)))
                ),
                "max_derivative_20ms": float(np.max(derivative)) if derivative.size else 0.0,
            }
        )
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", default="identity-lock")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    return parser


def main() -> int:
    args = _parser().parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
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
    from aniflive_tts.expression import ExpressionSegment

    segments = tuple(
        ExpressionSegment(
            text=text,
            enabled=profile != "neutral",
            profile=None if profile == "neutral" else profile,
            intensity=intensity,
            policy=service_module.ConditioningPolicy(args.policy),
        )
        for text, profile, intensity in DEFAULT_SEGMENTS
    )
    options = service_module.SynthesisOptions(
        text="".join(segment.text for segment in segments),
        text_language="ja",
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        speed=1.0,
        pause_length=0.44,
        noise_scale=0.5,
        cut_punc="",
        seed=1234,
        expression_segments=segments,
    )
    runtime = service_module.TensorRTService(service_module.RuntimeSettings.from_env())
    runtime.load()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        runtime.validate_expression(options)
        for _ in range(args.warmup):
            list(runtime.stream_pcm(options))
        rows: list[dict[str, Any]] = []
        payload = b""
        profile: dict[str, Any] = {}
        for _ in range(args.runs):
            started = time.perf_counter()
            iterator = iter(runtime.stream_pcm(options))
            first = next(iterator)
            ttfa = time.perf_counter() - started
            payload = b"".join((first, *iterator))
            wall = time.perf_counter() - started
            duration = len(payload) / 2.0 / runtime.sample_rate
            profile = dict(runtime._streamer.last_profile or {})
            rows.append(
                {
                    "ttfa_seconds": ttfa,
                    "wall_seconds": wall,
                    "audio_seconds": duration,
                    "rtf": wall / duration,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        (output_dir / "segmented-stream.wav").write_bytes(
            _wav(runtime.sample_rate, payload)
        )
        full = runtime.synthesize(options)
        (output_dir / "segmented-complete.wav").write_bytes(full.wav)
        report = {
            "schema": 1,
            "purpose": "Roxy-first segmented expression hard-switch baseline",
            "policy": args.policy,
            "seed": 1234,
            "segments": [
                {
                    "text": text,
                    "expression": profile,
                    "intensity": intensity,
                }
                for text, profile, intensity in DEFAULT_SEGMENTS
            ],
            "stream_runs": rows,
            "stream_deterministic": len({row["sha256"] for row in rows}) == 1,
            "stream_profile": profile,
            "boundary_metrics": _boundary_metrics(
                payload,
                runtime.sample_rate,
                list(profile.get("segment_profiles", [])),
            ),
            "complete_profile": full.profile,
            "complete_sha256": hashlib.sha256(full.wav).hexdigest(),
        }
        report_path = output_dir / "segmented-expression.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(report_path)
        return 0
    finally:
        runtime.unload()


if __name__ == "__main__":
    raise SystemExit(main())
