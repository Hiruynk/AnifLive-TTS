#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete offline target-speaker component chain on one "
            "real audio case inside Linux Docker."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--speaker-component", type=Path, required=True)
    parser.add_argument("--vad-component", type=Path, required=True)
    parser.add_argument("--diarization-component", type=Path, required=True)
    parser.add_argument("--separation-component", type=Path, required=True)
    parser.add_argument("--asr-component", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--language", choices=("yue", "zh", "ja", "en", "ko"))
    parser.add_argument("--minimum-source-seconds", type=float, default=0.0)
    parser.add_argument(
        "--fixture-report",
        type=Path,
        help="Optional labelled fixture manifest used for safety/confusion gates.",
    )
    return parser


def _load_json_object(path: Path, *, maximum_bytes: int = 32 * 1024 * 1024) -> dict:
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum_bytes:
        raise SystemExit("fixture report must be a bounded regular JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit("fixture report is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise SystemExit("fixture report must contain one JSON object")
    return value


def main() -> int:
    arguments = _parser().parse_args()
    if not sys.platform.startswith("linux"):
        raise SystemExit("voice-acquisition qualification must run in Linux Docker")
    if (
        not math.isfinite(arguments.minimum_source_seconds)
        or arguments.minimum_source_seconds < 0
    ):
        raise SystemExit("minimum-source-seconds must be a finite non-negative number")
    output = arguments.output.resolve()
    if output.exists():
        raise SystemExit("qualification output must not already exist")
    output.mkdir(parents=True)

    from aniflive_tts.dataset_acquisition_worker import (
        run_target_speaker_finalize,
        run_target_speaker_routing,
        run_target_speaker_separation,
        run_target_speaker_transcription,
    )

    settings = {
        "target_threshold": 0.72,
        "review_margin": 0.08,
        "ambiguity_margin": 0.03,
        "minimum_speech_ms": 160.0,
        "maximum_segment_ms": 30_000.0,
        "context_ms": 40.0,
        "asr_backend": "sensevoice-small",
        "declared_language": arguments.language,
    }
    routing, _ = run_target_speaker_routing(
        source=arguments.source.resolve(strict=True),
        reference=arguments.reference.resolve(strict=True),
        speaker_component=arguments.speaker_component.resolve(strict=True),
        vad_component=arguments.vad_component.resolve(strict=True),
        diarization_component=arguments.diarization_component.resolve(strict=True),
        output=output / "01-routing",
        settings=settings,
    )
    separation, _ = run_target_speaker_separation(
        dependency=output / "01-routing",
        reference=arguments.reference.resolve(strict=True),
        speaker_component=arguments.speaker_component.resolve(strict=True),
        separation_model=arguments.separation_component.resolve(strict=True),
        output=output / "02-separation",
        settings=settings,
    )
    transcription, _ = run_target_speaker_transcription(
        dependency=output / "02-separation",
        asr_model=arguments.asr_component.resolve(strict=True),
        output=output / "03-transcription",
        settings=settings,
    )
    finalized, _ = run_target_speaker_finalize(
        dependency=output / "03-transcription",
        output=output / "04-final",
    )
    transcripts = [
        record.get("annotations", {}).get("transcript")
        for record in finalized["records"]
        if isinstance(record.get("annotations"), dict)
    ]
    record_counts = {
        "routing": len(routing["records"]),
        "separation": len(separation["records"]),
        "transcription": len(transcription["records"]),
        "final": len(finalized["records"]),
    }
    source_seconds = float(routing.get("source_seconds") or 0.0)
    gates = {
        "source_duration": source_seconds >= arguments.minimum_source_seconds,
        "records_present": record_counts["routing"] > 0,
        "record_lineage_complete": len(set(record_counts.values())) == 1,
        "speaker_backend": routing.get("speaker_backend")
        == "workstation-eres2netv2-tensorrt11",
        "vad_backend": str(routing.get("vad", {}).get("backend", "")).startswith(
            "fsmn-vad-"
        ),
        "diarization_backend": routing.get("diarization", {}).get("backend")
        == "sortformer-v2-nemo-speech-cpp-cuda-offline",
        "separation_backend": separation.get("separator") == "MossFormer2_SS_16K",
        "transcription_backend": bool(transcription["asr"].get("backend")),
        "nonempty_transcript": any(
            isinstance(text, str) and text.strip() for text in transcripts
        ),
        "human_review_required": transcription.get("human_review_required") is True,
        "final_waiting_for_review": finalized.get("status") == "waiting_for_review",
    }
    fixture_evaluation = None
    if arguments.fixture_report is not None:
        from aniflive_tts.voice_acquisition_fixture import (
            VoiceAcquisitionFixtureError,
            evaluate_voice_acquisition_fixture,
        )

        try:
            fixture_evaluation = evaluate_voice_acquisition_fixture(
                _load_json_object(arguments.fixture_report), routing, finalized
            )
        except VoiceAcquisitionFixtureError as error:
            raise SystemExit(f"labelled fixture evaluation failed closed: {error}") from error
        gates["labelled_fixture"] = fixture_evaluation["passed"] is True
    passed = all(gates.values())
    try:
        import torch
        import tensorrt as trt
    except ImportError as error:
        raise SystemExit(f"qualification runtime dependency is missing: {error}") from error
    report = {
        "schema": "aniflive-voice-acquisition-components-qualification-v1",
        "passed": passed,
        "neural_fallback": False,
        "network_required": False,
        "gates": gates,
        "minimum_source_seconds": arguments.minimum_source_seconds,
        "source_seconds": source_seconds,
        "routing": {
            "counts": routing["counts"],
            "speaker_backend": routing["speaker_backend"],
            "vad_backend": routing["vad"]["backend"],
            "diarization_backend": routing["diarization"]["backend"],
            "diarization_mode": routing["diarization"]["mode"],
            "diarization_postprocess": routing["diarization"].get("postprocess"),
            "diarized_clip_count": routing["diarization"]["clip_count"],
            "hypothesis_multi_speaker_clip_count": routing["diarization"].get(
                "hypothesis_multi_speaker_clip_count",
                routing["diarization"].get("multi_speaker_clip_count", 0),
            ),
            "forced_split_count": routing["vad"]["forced_split_count"],
            "reference_prototype": routing.get("reference_prototype"),
            "routing_policy": routing.get("routing_policy"),
            "sensitive_diarization": routing.get("sensitive_diarization"),
        },
        "separation": {
            "backend": separation["separator"],
            "counts": separation["counts"],
        },
        "transcription": {
            "backend": transcription["asr"]["backend"],
            "review_required": transcription["human_review_required"],
            "nonempty_transcripts": sum(
                isinstance(text, str) and bool(text.strip()) for text in transcripts
            ),
        },
        "final": {
            "status": finalized["status"],
            "counts": finalized["counts"],
            "record_count": len(finalized["records"]),
            "record_counts_by_stage": record_counts,
        },
        "runtime": {
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tensorrt": trt.__version__,
        },
        "labelled_fixture": fixture_evaluation,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
