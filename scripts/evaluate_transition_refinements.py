from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from evaluate_segmented_expression import DEFAULT_SEGMENTS, _boundary_metrics, _wav


VARIANTS = (
    ("hann-2ms", "hann", 2.0, 12.0),
    ("hann-6ms", "hann", 6.0, 12.0),
    ("hann-8ms", "hann", 8.0, 12.0),
    ("sigmoid-2ms-k12", "sigmoid", 2.0, 12.0),
    ("sigmoid-6ms-k12", "sigmoid", 6.0, 12.0),
    ("sigmoid-8ms-k12", "sigmoid", 8.0, 12.0),
    ("sigmoid-4ms-k8", "sigmoid", 4.0, 8.0),
    ("sigmoid-4ms-k20", "sigmoid", 4.0, 20.0),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", default="identity-lock")
    parser.add_argument("--runs", type=int, default=3)
    return parser


def _run_variant(
    runtime: Any,
    options: Any,
    output_dir: Path,
    *,
    name: str,
    curve: str,
    milliseconds: float,
    sigmoid_k: float,
    runs: int,
) -> dict[str, Any]:
    os.environ["ANIFLIVE_TTS_EXPRESSION_TRANSITION"] = curve
    os.environ["ANIFLIVE_TTS_EXPRESSION_TRANSITION_MS"] = str(milliseconds)
    os.environ["ANIFLIVE_TTS_EXPRESSION_SIGMOID_K"] = str(sigmoid_k)
    list(runtime.stream_pcm(options))
    rows: list[dict[str, Any]] = []
    payload = b""
    profile: dict[str, Any] = {}
    for _ in range(runs):
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
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "segmented-stream.wav").write_bytes(
        _wav(runtime.sample_rate, payload)
    )
    report = {
        "name": name,
        "curve": curve,
        "milliseconds": milliseconds,
        "sigmoid_k": sigmoid_k,
        "runs": rows,
        "deterministic": len({row["sha256"] for row in rows}) == 1,
        "profile": profile,
        "boundary_metrics": _boundary_metrics(
            payload,
            runtime.sample_rate,
            list(profile.get("segment_profiles", [])),
        ),
    }
    (variant_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


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
        reports = [
            _run_variant(
                runtime,
                options,
                output_dir,
                name=name,
                curve=curve,
                milliseconds=milliseconds,
                sigmoid_k=sigmoid_k,
                runs=args.runs,
            )
            for name, curve, milliseconds, sigmoid_k in VARIANTS
        ]
        summary = {
            "schema": 1,
            "purpose": "Roxy expression transition top-two refinements",
            "policy": args.policy,
            "reports": reports,
        }
        path = output_dir / "transition-refinements.json"
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(path)
        return 0
    finally:
        runtime.unload()


if __name__ == "__main__":
    raise SystemExit(main())
