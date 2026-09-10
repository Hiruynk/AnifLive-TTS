from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .workstation_production_gates import (
    ProductionGateError,
    SPEAKER_GENERATED_MEDIAN_FLOOR,
    SPEAKER_GENERATED_P10_FLOOR,
    SPEAKER_MEDIAN_RETENTION_LIMIT,
    SPEAKER_P10_RETENTION_LIMIT,
    SPEAKER_SOURCE_P10_LIMIT,
    speaker_identity_gate,
)


CHECKPOINT_CANDIDATES_SCHEMA = "aniflive-tts-v2proplus-checkpoint-candidates-v1"
CHECKPOINT_SELECTION_SCHEMA = "aniflive-tts-checkpoint-selection-v1"
CHECKPOINT_SELECTION_METHOD = "validation-checkpoint-selection-v2"
DEPLOYMENT_CHECKPOINTS_SCHEMA = "aniflive-tts-v2proplus-deployment-checkpoints-v2"
SELECTION_SEEDS = (1234, 2026, 7)


class CheckpointSelectionError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CheckpointSelectionError(f"{label} is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointSelectionError(f"{label} is unreadable") from error
    if not isinstance(payload, dict):
        raise CheckpointSelectionError(f"{label} must be a JSON object")
    return payload


def validation_records(training_bundle: Path) -> list[dict[str, Any]]:
    """Read only the validation manifest; selection must never inspect test/."""

    descriptor = _read_object(training_bundle / "training-input.json", "training descriptor")
    if descriptor.get("schema") != "aniflive-v2proplus-training-input-v2":
        raise CheckpointSelectionError("checkpoint selection requires training input v2")
    manifest = _read_object(
        training_bundle / "validation" / "manifest.json", "validation manifest"
    )
    if manifest.get("split") != "validation":
        raise CheckpointSelectionError("checkpoint selection may only read validation data")
    records = manifest.get("items")
    if not isinstance(records, list) or not records:
        raise CheckpointSelectionError("validation manifest has no items")
    if any(not isinstance(record, dict) for record in records):
        raise CheckpointSelectionError("validation manifest items are malformed")
    return records


def build_selection_plan(
    candidate_manifest: Mapping[str, Any],
    *,
    validation_item_ids: Sequence[str],
    top_n: int = 3,
) -> dict[str, Any]:
    if candidate_manifest.get("schema") != CHECKPOINT_CANDIDATES_SCHEMA:
        raise CheckpointSelectionError("checkpoint candidate manifest is unsupported")
    gpt = candidate_manifest.get("gpt")
    sovits = candidate_manifest.get("sovits")
    if not isinstance(gpt, list) or not gpt or not isinstance(sovits, list) or not sovits:
        raise CheckpointSelectionError("checkpoint candidate manifest is incomplete")
    item_ids = tuple(str(value) for value in validation_item_ids if str(value))
    if not item_ids:
        raise CheckpointSelectionError("checkpoint selection requires validation items")
    if not isinstance(top_n, int) or isinstance(top_n, bool) or not 1 <= top_n <= 5:
        raise CheckpointSelectionError("top_n must be between 1 and 5")
    latest_gpt = max(gpt, key=lambda row: int(row["epoch"]))
    latest_sovits = max(sovits, key=lambda row: int(row["epoch"]))
    return {
        "schema": "aniflive-tts-checkpoint-selection-plan-v1",
        "validation_item_ids": list(item_ids),
        "test_access_allowed": False,
        "passes": {
            "gpt_sweep": [
                {
                    "gpt_epoch": int(row["epoch"]),
                    "sovits_epoch": int(latest_sovits["epoch"]),
                    "seeds": [1234],
                }
                for row in gpt
            ],
            "sovits_sweep": {
                "provisional_gpt_epoch": None,
                "sovits_epochs": [int(row["epoch"]) for row in sovits],
                "seeds": [1234],
            },
            "joint_sweep": {
                "top_n": top_n,
                "maximum_pairs": top_n * top_n,
                "seeds": list(SELECTION_SEEDS),
            },
        },
        "probe_dependencies": {
            "latest_gpt_epoch": int(latest_gpt["epoch"]),
            "latest_sovits_epoch": int(latest_sovits["epoch"]),
            "declared_best": False,
        },
    }


def _finite(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CheckpointSelectionError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CheckpointSelectionError(f"{field} must be finite")
    return result


def _pair_key(row: Mapping[str, Any]) -> tuple[int, int]:
    try:
        gpt_epoch = int(row["gpt_epoch"])
        sovits_epoch = int(row["sovits_epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointSelectionError("checkpoint evidence has invalid epochs") from error
    if gpt_epoch < 1 or sovits_epoch < 1:
        raise CheckpointSelectionError("checkpoint evidence epochs must be positive")
    return gpt_epoch, sovits_epoch


def _passed_hard_gates(
    row: Mapping[str, Any],
    *,
    median_error_limit: float,
    p95_error_limit: float,
    speaker_cosine_limit: float,
    enforce_speaker_identity: bool,
) -> tuple[bool, list[str], dict[str, Any]]:
    content = row.get("content")
    speaker = row.get("speaker")
    audio = row.get("audio")
    if not isinstance(content, Mapping) or not isinstance(speaker, Mapping) or not isinstance(
        audio, Mapping
    ):
        raise CheckpointSelectionError("checkpoint evidence sections are malformed")
    failures: list[str] = []
    success = _finite(row.get("generation_success_rate"), "generation_success_rate")
    if success != 1.0:
        failures.append("generation-success")
    for field in (
        "repetition_failures",
        "omission_failures",
    ):
        if int(content.get(field, -1)) != 0:
            failures.append(field.replace("_", "-"))
    for field in ("invalid", "nan", "empty", "clipped", "duration_outliers"):
        if int(audio.get(field, -1)) != 0:
            failures.append(field.replace("_", "-"))
    median_error = _finite(content.get("median_error"), "content.median_error")
    p95_error = _finite(content.get("p95_error"), "content.p95_error")
    if median_error > median_error_limit:
        failures.append("median-content-error")
    if p95_error > p95_error_limit:
        failures.append("p95-content-error")
    _finite(speaker.get("centroid_cosine_median"), "speaker.centroid_cosine_median")
    _finite(speaker.get("centroid_cosine_p10"), "speaker.centroid_cosine_p10")
    try:
        speaker_gate = speaker_identity_gate(
            speaker, absolute_median_limit=speaker_cosine_limit
        )
    except ProductionGateError as error:
        raise CheckpointSelectionError(str(error)) from error
    if enforce_speaker_identity and not speaker_gate["passed"]:
        failures.append("speaker-identity")
    _finite(row.get("stability", 0.0), "stability")
    return not failures, failures, speaker_gate


def evaluate_checkpoint_evidence(
    evidence: Sequence[Mapping[str, Any]],
    *,
    median_error_limit: float = 0.08,
    p95_error_limit: float = 0.18,
    speaker_cosine_limit: float = 0.80,
    enforce_speaker_identity: bool = True,
) -> list[dict[str, Any]]:
    if not evidence:
        raise CheckpointSelectionError("checkpoint selection has no evidence")
    evaluated: list[dict[str, Any]] = []
    for raw in evidence:
        row = dict(raw)
        gpt_epoch, sovits_epoch = _pair_key(row)
        passed, failures, speaker_gate = _passed_hard_gates(
            row,
            median_error_limit=median_error_limit,
            p95_error_limit=p95_error_limit,
            speaker_cosine_limit=speaker_cosine_limit,
            enforce_speaker_identity=enforce_speaker_identity,
        )
        row["gpt_epoch"] = gpt_epoch
        row["sovits_epoch"] = sovits_epoch
        row["hard_gate"] = {"passed": passed, "failures": failures}
        row["speaker_identity_gate"] = speaker_gate
        row["status"] = "passed" if passed else "failed"
        evaluated.append(row)
    return evaluated


def consolidate_reference_evaluations(
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Collapse per-reference evidence without averaging away a valid reference.

    Production locks one deployment reference after checkpoint selection.  A
    checkpoint pair therefore qualifies when at least one independently gated
    train-only reference qualifies.  Every reference result remains attached
    to the consolidated row for audit and downstream human selection.
    """

    if not evidence:
        raise CheckpointSelectionError("reference evaluation has no evidence")
    evaluated = evaluate_checkpoint_evidence(evidence)
    pairs = {_pair_key(row) for row in evaluated}
    if len(pairs) != 1:
        raise CheckpointSelectionError(
            "reference evaluations must describe one checkpoint pair"
        )
    item_ids: list[str] = []
    for row in evaluated:
        probe = row.get("reference_probe")
        probe_ids = probe.get("item_ids") if isinstance(probe, Mapping) else None
        if not isinstance(probe_ids, list) or len(probe_ids) != 1:
            raise CheckpointSelectionError(
                "per-reference evidence must identify exactly one reference"
            )
        item_id = probe_ids[0]
        if not isinstance(item_id, str) or not item_id or item_id in item_ids:
            raise CheckpointSelectionError(
                "per-reference evidence has an invalid or duplicate reference"
            )
        item_ids.append(item_id)

    def ranking(row: Mapping[str, Any]) -> tuple[int, float, float, float, float]:
        return (
            len(row["hard_gate"]["failures"]),
            -float(row["speaker"]["centroid_cosine_median"]),
            float(row["content"]["median_error"]),
            float(row["content"]["p95_error"]),
            -float(row.get("stability", 0.0)),
        )

    passing = [row for row in evaluated if row["hard_gate"]["passed"]]
    selected = sorted(passing or evaluated, key=ranking)[0]
    consolidated = dict(selected)
    selected_id = str(selected["reference_probe"]["item_ids"][0])
    qualified_ids = [
        str(row["reference_probe"]["item_ids"][0])
        for row in evaluated
        if row["hard_gate"]["passed"]
    ]
    consolidated["reference_probe"] = {
        "mode": "checkpoint-conditioned-reference-evaluation-v1",
        "count": len(evaluated),
        "item_ids": item_ids,
        "qualified_item_ids": qualified_ids,
        "selected_item_id": selected_id,
    }
    consolidated["reference_robustness"] = {
        "qualified_count": len(qualified_ids),
        "evaluated_count": len(evaluated),
    }
    consolidated["reference_evaluations"] = evaluated
    return consolidated


def rank_checkpoint_sweep(
    rows: Sequence[Mapping[str, Any]], role: str
) -> list[int]:
    if role not in {"gpt", "sovits"}:
        raise CheckpointSelectionError("checkpoint sweep role is invalid")
    passing = [row for row in rows if row.get("status") == "passed"]
    if not passing:
        failures = "; ".join(
            f"e{int(row[f'{role}_epoch'])}="
            + "+".join(str(value) for value in row["hard_gate"]["failures"])
            for row in rows
        )
        raise CheckpointSelectionError(
            f"{role} sweep has no passing candidate ({failures})"
        )
    if role == "gpt":
        passing.sort(
            key=lambda row: (
                float(row["content"]["median_error"]),
                float(row["content"]["p95_error"]),
                -float(row.get("stability", 0.0)),
                int(row["gpt_epoch"]),
            )
        )
    else:
        passing.sort(
            key=lambda row: (
                -float(row["speaker"]["centroid_cosine_median"]),
                float(row["content"]["median_error"]),
                float(row["content"]["p95_error"]),
                int(row["sovits_epoch"]),
            )
        )
    result: list[int] = []
    for row in passing:
        epoch = int(row[f"{role}_epoch"])
        if epoch not in result:
            result.append(epoch)
    return result[:3]


def select_checkpoint_pair(
    evidence: Sequence[Mapping[str, Any]],
    *,
    validation_manifest_sha256: str,
    median_error_limit: float = 0.08,
    p95_error_limit: float = 0.18,
    speaker_cosine_limit: float = 0.80,
) -> dict[str, Any]:
    if not isinstance(validation_manifest_sha256, str) or len(validation_manifest_sha256) != 64:
        raise CheckpointSelectionError("validation manifest SHA256 is invalid")
    evaluated = evaluate_checkpoint_evidence(
        evidence,
        median_error_limit=median_error_limit,
        p95_error_limit=p95_error_limit,
        speaker_cosine_limit=speaker_cosine_limit,
    )
    passed_rows = [row for row in evaluated if row["hard_gate"]["passed"]]
    if not passed_rows:
        raise CheckpointSelectionError("no checkpoint pair passed the hard gates")

    def ranking(row: Mapping[str, Any]) -> tuple[float, float, float, float, int, int]:
        speaker = row["speaker"]
        content = row["content"]
        return (
            -float(speaker["centroid_cosine_median"]),
            float(content["median_error"]),
            float(content["p95_error"]),
            -float(row.get("stability", 0.0)),
            int(row["gpt_epoch"]),
            int(row["sovits_epoch"]),
        )

    ranked = sorted(passed_rows, key=ranking)
    winner = ranked[0]
    alternate_rows = [
        row
        for row in evaluated
        if (row["gpt_epoch"], row["sovits_epoch"])
        != (winner["gpt_epoch"], winner["sovits_epoch"])
    ]
    if not alternate_rows:
        raise CheckpointSelectionError(
            "checkpoint selection needs a distinct runner-up candidate"
        )
    runner_up = sorted(
        alternate_rows,
        key=lambda row: (len(row["hard_gate"]["failures"]), ranking(row)),
    )[0]
    return {
        "schema": CHECKPOINT_SELECTION_SCHEMA,
        "status": "passed",
        "method": CHECKPOINT_SELECTION_METHOD,
        "test_split_accessed": False,
        "validation_manifest_sha256": validation_manifest_sha256,
        "hard_gate_limits": {
            "generation_success_rate": 1.0,
            "median_content_error": median_error_limit,
            "p95_content_error": p95_error_limit,
            "speaker_centroid_cosine_median": speaker_cosine_limit,
            "source_calibrated_retention": {
                "source_p10_minimum": SPEAKER_SOURCE_P10_LIMIT,
                "generated_median_floor": SPEAKER_GENERATED_MEDIAN_FLOOR,
                "generated_p10_floor": SPEAKER_GENERATED_P10_FLOOR,
                "median_retention_ratio": SPEAKER_MEDIAN_RETENTION_LIMIT,
                "p10_retention_ratio": SPEAKER_P10_RETENTION_LIMIT,
                "calibration": "qualified-model-source-control-retention-v1",
            },
            "repetition_failures": 0,
            "omission_failures": 0,
            "invalid_audio": 0,
            "nan_audio": 0,
            "empty_audio": 0,
            "clipped_audio": 0,
            "duration_outliers": 0,
        },
        "winner": {
            "gpt_epoch": winner["gpt_epoch"],
            "sovits_epoch": winner["sovits_epoch"],
        },
        "runner_up": {
            "gpt_epoch": runner_up["gpt_epoch"],
            "sovits_epoch": runner_up["sovits_epoch"],
            "qualified": bool(runner_up["hard_gate"]["passed"]),
            "hard_gate_failures": list(runner_up["hard_gate"]["failures"]),
        },
        "winner_reason": (
            "passed all hard gates; ranked by speaker identity, median content error, "
            "P95 content error, stochastic stability, then earlier epoch"
        ),
        "evaluations": evaluated,
    }


def _candidate_by_epoch(
    candidates: Sequence[Mapping[str, Any]], epoch: int, label: str
) -> Mapping[str, Any]:
    matching = [row for row in candidates if int(row.get("epoch", -1)) == epoch]
    if len(matching) != 1:
        raise CheckpointSelectionError(f"selection did not resolve one {label} epoch {epoch}")
    return matching[0]


def materialize_deployment_checkpoints(
    *,
    candidate_root: Path,
    selection_report_path: Path,
    output: Path,
) -> tuple[Path, dict[str, Any], list[Path]]:
    candidates = _read_object(candidate_root / "checkpoint-candidates.json", "candidate manifest")
    if candidates.get("schema") != CHECKPOINT_CANDIDATES_SCHEMA:
        raise CheckpointSelectionError("checkpoint candidate manifest is unsupported")
    selection = _read_object(selection_report_path, "checkpoint selection report")
    if (
        selection.get("schema") != CHECKPOINT_SELECTION_SCHEMA
        or selection.get("status") != "passed"
        or selection.get("test_split_accessed") is not False
    ):
        raise CheckpointSelectionError("checkpoint selection report did not pass")
    winner = selection.get("winner")
    runner_up = selection.get("runner_up")
    selection_method = selection.get("method")
    if not isinstance(winner, Mapping) or not isinstance(runner_up, Mapping):
        raise CheckpointSelectionError("checkpoint selection winners are malformed")
    if selection_method not in {
        "validation-checkpoint-selection-v1",
        CHECKPOINT_SELECTION_METHOD,
    }:
        raise CheckpointSelectionError("checkpoint selection method is unsupported")
    output.mkdir(parents=True, exist_ok=False)
    copied: list[Path] = []

    def copy_record(record: Mapping[str, Any], role: str) -> dict[str, Any]:
        relative = Path(str(record.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise CheckpointSelectionError(f"{role} checkpoint path is unsafe")
        source = candidate_root / relative
        if source.is_symlink() or not source.is_file():
            raise CheckpointSelectionError(f"{role} checkpoint is missing")
        if _sha256_file(source) != record.get("sha256"):
            raise CheckpointSelectionError(f"{role} checkpoint checksum changed")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
        return {
            "epoch": int(record["epoch"]),
            "relative_path": relative.as_posix(),
            "sha256": str(record["sha256"]),
            "size_bytes": int(record["size_bytes"]),
        }

    gpt_candidates = candidates.get("gpt")
    sovits_candidates = candidates.get("sovits")
    if not isinstance(gpt_candidates, list) or not isinstance(sovits_candidates, list):
        raise CheckpointSelectionError("checkpoint candidate roles are malformed")
    winner_gpt = _candidate_by_epoch(gpt_candidates, int(winner["gpt_epoch"]), "GPT")
    winner_sovits = _candidate_by_epoch(
        sovits_candidates, int(winner["sovits_epoch"]), "SoVITS"
    )
    runner_gpt = _candidate_by_epoch(
        gpt_candidates, int(runner_up["gpt_epoch"]), "runner-up GPT"
    )
    runner_sovits = _candidate_by_epoch(
        sovits_candidates, int(runner_up["sovits_epoch"]), "runner-up SoVITS"
    )
    deployment = {
        "schema": DEPLOYMENT_CHECKPOINTS_SCHEMA,
        "model_family": "gsv-v2proplus",
        "gpt": copy_record(winner_gpt, "winner GPT"),
        "sovits": copy_record(winner_sovits, "winner SoVITS"),
        "runner_up": {
            "gpt": copy_record(runner_gpt, "runner-up GPT"),
            "sovits": copy_record(runner_sovits, "runner-up SoVITS"),
            "qualified": bool(runner_up.get("qualified", False)),
            "hard_gate_failures": list(runner_up.get("hard_gate_failures", [])),
        },
        "selection": {
            "method": selection_method,
            "report_sha256": _sha256_file(selection_report_path),
            "winner_reason": str(selection["winner_reason"]),
            "validation_manifest_sha256": str(selection["validation_manifest_sha256"]),
            "test_split_accessed": False,
        },
    }
    path = output / "deployment-checkpoints.json"
    path.write_text(
        json.dumps(deployment, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    copied.append(path)
    return path, deployment, copied


__all__ = [
    "CHECKPOINT_CANDIDATES_SCHEMA",
    "CHECKPOINT_SELECTION_SCHEMA",
    "DEPLOYMENT_CHECKPOINTS_SCHEMA",
    "CheckpointSelectionError",
    "build_selection_plan",
    "consolidate_reference_evaluations",
    "materialize_deployment_checkpoints",
    "rank_checkpoint_sweep",
    "select_checkpoint_pair",
    "validation_records",
]
