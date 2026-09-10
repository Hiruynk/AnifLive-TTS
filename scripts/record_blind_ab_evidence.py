#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "aniflive-tts-blind-ab-evidence-v1"
ID_PATTERN = re.compile(r"^(?:artifact|expression)_[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256(value: str, *, field: str) -> str:
    result = value.strip().lower()
    if SHA256_PATTERN.fullmatch(result) is None:
        raise SystemExit(f"{field} must contain exactly 64 hexadecimal characters")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record an explicit human AnifLive-TTS blind A/B decision"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--subject-kind", required=True, choices=("artifact", "expression"))
    parser.add_argument("--subject-id", required=True)
    parser.add_argument(
        "--gate",
        required=True,
        choices=("long-form-continuity", "expression-quality"),
    )
    parser.add_argument("--decision", required=True, choices=("passed", "failed"))
    parser.add_argument("--completed-trials", required=True, type=int)
    parser.add_argument("--listener-count", required=True, type=int)
    parser.add_argument("--sample-manifest-sha256", required=True)
    parser.add_argument("--randomization-sha256", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    if ID_PATTERN.fullmatch(args.subject_id) is None or not args.subject_id.startswith(
        args.subject_kind + "_"
    ):
        raise SystemExit("subject-id does not match subject-kind")
    if not 1 <= args.completed_trials <= 100_000:
        raise SystemExit("completed-trials must be between 1 and 100000")
    if not 1 <= args.listener_count <= 10_000:
        raise SystemExit("listener-count must be between 1 and 10000")
    operator = " ".join(args.operator.strip().split())
    summary = " ".join(args.summary.strip().split())
    if not operator or len(operator) > 120:
        raise SystemExit("operator must contain 1 to 120 characters")
    if not summary or len(summary) > 1000:
        raise SystemExit("summary must contain 1 to 1000 characters")

    payload = {
        "schema": SCHEMA,
        "subject": {"kind": args.subject_kind, "id": args.subject_id},
        "gate": args.gate,
        "decision": args.decision,
        "protocol": {
            "blinded": True,
            "comparison": "candidate-vs-baseline",
            "completed_trials": args.completed_trials,
            "listener_count": args.listener_count,
            "sample_manifest_sha256": _sha256(
                args.sample_manifest_sha256, field="sample-manifest-sha256"
            ),
            "randomization_sha256": _sha256(
                args.randomization_sha256, field="randomization-sha256"
            ),
        },
        "summary": summary,
        "operator": operator,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8", newline="\n") as destination:
            json.dump(payload, destination, ensure_ascii=False, indent=2, sort_keys=True)
            destination.write("\n")
    except FileExistsError as error:
        raise SystemExit("Refusing to replace an existing blind A/B evidence artifact") from error
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
