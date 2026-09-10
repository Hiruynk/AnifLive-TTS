from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Sequence

from aniflive_tts.continuity_diagnostics import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_TOTAL_BYTES,
    DEFAULT_MAX_SEGMENTS,
    REPORT_SCHEMA,
    ContinuityDiagnosticError,
    evaluate_ordered_segments,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure waveform boundary continuity across ordered AnifLive-TTS mono PCM16/WAV "
            "segments. This is a diagnostic, not a speech-quality qualification."
        )
    )
    parser.add_argument("segments", nargs="+", type=Path, help="Ordered .wav, .pcm, or .raw files")
    parser.add_argument(
        "--input-format", choices=("auto", "wav", "pcm16le"), default="auto",
        help="Input format for every segment (default: infer from suffix)",
    )
    parser.add_argument(
        "--raw-sample-rate", type=int,
        help="Required sample rate for raw .pcm/.raw inputs",
    )
    parser.add_argument(
        "--expected-sample-rate", type=int,
        help="Reject all inputs unless they use this sample rate",
    )
    parser.add_argument("--window-ms", type=float, default=10.0)
    parser.add_argument("--silence-threshold-dbfs", type=float, default=-50.0)
    parser.add_argument(
        "--max-file-mib", type=float, default=DEFAULT_MAX_FILE_BYTES / (1024 * 1024)
    )
    parser.add_argument(
        "--max-total-mib", type=float, default=DEFAULT_MAX_TOTAL_BYTES / (1024 * 1024)
    )
    parser.add_argument("--max-segments", type=int, default=DEFAULT_MAX_SEGMENTS)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser


def _mib_bytes(value: float, *, field: str) -> int:
    if not math.isfinite(value) or value <= 0:
        raise ContinuityDiagnosticError(f"{field} must be a positive finite number")
    result = int(value * 1024 * 1024)
    if result < 1:
        raise ContinuityDiagnosticError(f"{field} is too small")
    return result


def _write_report(path: Path, encoded: str) -> None:
    destination = path.expanduser().resolve()
    if not destination.parent.is_dir():
        raise ContinuityDiagnosticError(f"Report directory does not exist: {destination.parent}")
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(encoded + "\n", encoding="utf-8")
        temporary.replace(destination)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ContinuityDiagnosticError(f"Report could not be written: {destination}") from error


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_ordered_segments(
            args.segments,
            input_format=args.input_format,
            raw_sample_rate=args.raw_sample_rate,
            expected_sample_rate=args.expected_sample_rate,
            window_ms=args.window_ms,
            silence_threshold_dbfs=args.silence_threshold_dbfs,
            max_file_bytes=_mib_bytes(args.max_file_mib, field="max_file_mib"),
            max_total_bytes=_mib_bytes(args.max_total_mib, field="max_total_mib"),
            max_segments=args.max_segments,
        )
        encoded = json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        if args.output is not None:
            _write_report(args.output, encoded)
        print(encoded)
        return 0
    except ContinuityDiagnosticError as error:
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
