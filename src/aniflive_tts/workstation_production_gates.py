from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


HOLDOUT_EVALUATION_SCHEMA = "aniflive-tts-holdout-evaluation-v1"
CONVERSION_PARITY_SCHEMA = "aniflive-tts-conversion-parity-v1"
PRODUCTION_CHAIN = (
    "training.prepare",
    "checkpoint.select",
    "reference.select",
    "holdout.evaluate",
    "engine.prepare",
    "conversion.parity",
    "model.package",
    "evaluation.prepare",
)

SPEAKER_ABSOLUTE_MEDIAN_LIMIT = 0.80
SPEAKER_SOURCE_P10_LIMIT = 0.72
SPEAKER_GENERATED_MEDIAN_FLOOR = 0.72
SPEAKER_GENERATED_P10_FLOOR = 0.64
# Production-qualified source/control calibration shows that synthetic speech and
# studio recordings occupy measurably different verifier domains. These retain a
# safety margin below the qualified baseline while the absolute floors remain hard.
SPEAKER_MEDIAN_RETENTION_LIMIT = 0.84
SPEAKER_P10_RETENTION_LIMIT = 0.82


class ProductionGateError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ProductionGateError(f"{label} is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionGateError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise ProductionGateError(f"{label} must be a JSON object")
    return value


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProductionGateError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProductionGateError(f"{label} must be finite")
    return result


def speaker_identity_gate(
    speaker: Mapping[str, Any],
    *,
    absolute_median_limit: float = SPEAKER_ABSOLUTE_MEDIAN_LIMIT,
) -> dict[str, Any]:
    """Evaluate identity against absolute and source-preservation evidence.

    A qualified source calibration is authoritative: generated speech must preserve
    that source identity distribution and cannot pass merely by crossing the lower
    absolute floor. A source qualifies either by the absolute source thresholds or
    by immutable human speaker verification plus the lower safety floors. The
    absolute path remains the fallback when no qualified source baseline is available.
    """

    generated_median = _finite(
        speaker.get("centroid_cosine_median"), "speaker cosine median"
    )
    generated_p10 = _finite(
        speaker.get("centroid_cosine_p10"), "speaker cosine P10"
    )
    absolute_passed = generated_median >= absolute_median_limit
    source_fields = (
        "source_centroid_cosine_median",
        "source_centroid_cosine_p10",
    )
    present = [field in speaker for field in source_fields]
    if any(present) and not all(present):
        raise ProductionGateError("speaker source calibration is incomplete")

    calibration: dict[str, Any] = {"available": all(present)}
    calibrated_passed = False
    baseline_qualified = False
    if all(present):
        source_median = _finite(
            speaker.get("source_centroid_cosine_median"),
            "source speaker cosine median",
        )
        source_p10 = _finite(
            speaker.get("source_centroid_cosine_p10"),
            "source speaker cosine P10",
        )
        if source_median <= 0.0 or source_p10 <= 0.0:
            raise ProductionGateError("speaker source calibration must be positive")
        median_retention = generated_median / source_median
        p10_retention = generated_p10 / source_p10
        absolute_source_qualified = (
            source_median >= absolute_median_limit
            and source_p10 >= SPEAKER_SOURCE_P10_LIMIT
        )
        human_verified_source_qualified = (
            speaker.get("source_human_verified") is True
            and source_median >= SPEAKER_GENERATED_MEDIAN_FLOOR
            and source_p10 >= SPEAKER_GENERATED_P10_FLOOR
        )
        baseline_qualified = (
            absolute_source_qualified or human_verified_source_qualified
        )
        calibrated_passed = (
            baseline_qualified
            and generated_median >= SPEAKER_GENERATED_MEDIAN_FLOOR
            and generated_p10 >= SPEAKER_GENERATED_P10_FLOOR
            and median_retention >= SPEAKER_MEDIAN_RETENTION_LIMIT
            and p10_retention >= SPEAKER_P10_RETENTION_LIMIT
        )
        calibration.update(
            {
                "baseline_qualified": baseline_qualified,
                "authoritative": baseline_qualified,
                "qualification": (
                    "absolute-source"
                    if absolute_source_qualified
                    else "human-verified-source"
                    if human_verified_source_qualified
                    else "unqualified"
                ),
                "source_human_verified": speaker.get("source_human_verified") is True,
                "source_centroid_cosine_median": source_median,
                "source_centroid_cosine_p10": source_p10,
                "median_retention_ratio": median_retention,
                "p10_retention_ratio": p10_retention,
            }
        )

    passed = calibrated_passed if baseline_qualified else absolute_passed
    return {
        "passed": passed,
        "mode": (
            "source-calibrated-retention"
            if calibrated_passed
            else "absolute"
            if absolute_passed and not baseline_qualified
            else "failed"
        ),
        "absolute_passed": absolute_passed,
        "calibrated_passed": calibrated_passed,
        "calibration": calibration,
    }


def open_locked_test_holdout(
    *,
    training_bundle: Path,
    deployment_checkpoints: Path,
    deployment_reference: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    descriptor = _read_object(training_bundle / "training-input.json", "training descriptor")
    if descriptor.get("schema") != "aniflive-v2proplus-training-input-v2":
        raise ProductionGateError("holdout evaluation requires training input v2")
    checkpoints = _read_object(deployment_checkpoints, "deployment checkpoints")
    selection = checkpoints.get("selection")
    if (
        checkpoints.get("schema")
        != "aniflive-tts-v2proplus-deployment-checkpoints-v2"
        or not isinstance(selection, Mapping)
        or selection.get("test_split_accessed") is not False
    ):
        raise ProductionGateError("checkpoint winner is not locked from validation evidence")
    reference = _read_object(deployment_reference, "deployment reference")
    if (
        reference.get("schema") != "aniflive-tts-v2proplus-deployment-reference-v2"
        or reference.get("status") != "human-locked"
        or reference.get("human_decision")
        not in {"preferred", "confirm-auto-winner", "no-preference"}
    ):
        raise ProductionGateError("deployment reference is not human locked")
    test_manifest_path = training_bundle / "test" / "manifest.json"
    test_manifest = _read_object(test_manifest_path, "test manifest")
    records = test_manifest.get("items")
    if test_manifest.get("split") != "test" or not isinstance(records, list) or not records:
        raise ProductionGateError("test holdout is empty or malformed")
    if any(not isinstance(record, dict) for record in records):
        raise ProductionGateError("test holdout items are malformed")
    lock = {
        "test_manifest_sha256": _sha256_file(test_manifest_path),
        "checkpoint_manifest_sha256": _sha256_file(deployment_checkpoints),
        "reference_manifest_sha256": _sha256_file(deployment_reference),
    }
    return records, lock


def holdout_evaluation_report(
    metrics: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    required_lock = {
        "test_manifest_sha256",
        "checkpoint_manifest_sha256",
        "reference_manifest_sha256",
    }
    if set(lock) != required_lock or any(
        not isinstance(lock[key], str) or len(lock[key]) != 64 for key in required_lock
    ):
        raise ProductionGateError("holdout evidence lock is invalid")
    content = metrics.get("content")
    speaker = metrics.get("speaker")
    audio = metrics.get("audio")
    if not isinstance(content, Mapping) or not isinstance(speaker, Mapping) or not isinstance(
        audio, Mapping
    ):
        raise ProductionGateError("holdout metrics are incomplete")
    failures: list[str] = []
    if _finite(metrics.get("generation_success_rate"), "generation success") != 1.0:
        failures.append("generation-success")
    if _finite(content.get("median_error"), "median content error") > 0.08:
        failures.append("median-content-error")
    if _finite(content.get("p95_error"), "P95 content error") > 0.18:
        failures.append("p95-content-error")
    speaker_gate = speaker_identity_gate(speaker)
    if not speaker_gate["passed"]:
        failures.append("speaker-identity")
    for field in (
        "repetition_failures",
        "omission_failures",
    ):
        if int(content.get(field, -1)) != 0:
            failures.append(field.replace("_", "-"))
    for field in ("invalid", "nan", "empty", "clipped", "duration_outliers"):
        if int(audio.get(field, -1)) != 0:
            failures.append(field.replace("_", "-"))
    report = {
        "schema": HOLDOUT_EVALUATION_SCHEMA,
        "status": "passed" if not failures else "failed",
        "winner_locked_before_test": True,
        "test_consumed_once": True,
        "lock": dict(lock),
        "metrics": dict(metrics),
        "speaker_identity_gate": speaker_gate,
        "failures": failures,
    }
    if cases is not None:
        if not cases or any(not isinstance(case, Mapping) for case in cases):
            raise ProductionGateError("holdout case evidence is malformed")
        report["cases"] = [dict(case) for case in cases]
    return report


def require_passed_holdout(path: Path) -> dict[str, Any]:
    report = _read_object(path, "holdout evaluation")
    if (
        report.get("schema") != HOLDOUT_EVALUATION_SCHEMA
        or report.get("status") != "passed"
        or report.get("winner_locked_before_test") is not True
        or report.get("test_consumed_once") is not True
    ):
        raise ProductionGateError("engine build is blocked by holdout evaluation")
    return report


def conversion_parity_report(
    cases: Sequence[Mapping[str, Any]],
    *,
    checkpoint_manifest_sha256: str,
    engine_build_sha256: str,
) -> dict[str, Any]:
    if not cases:
        raise ProductionGateError("conversion parity requires test cases")
    failures: list[str] = []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(cases):
        row = dict(raw)
        required = (
            "pytorch_onnx_logits_cosine",
            "onnx_trt_logits_cosine",
            "greedy_sequence_agreement",
            "log_mel_cosine",
            "speaker_cosine",
            "duration_difference_ratio",
        )
        values = {field: _finite(row.get(field), f"case {index} {field}") for field in required}
        case_failures: list[str] = []
        if values["pytorch_onnx_logits_cosine"] < 0.999:
            case_failures.append("pytorch-onnx-logits")
        if values["onnx_trt_logits_cosine"] < 0.999:
            case_failures.append("onnx-trt-logits")
        if values["greedy_sequence_agreement"] < 1.0:
            case_failures.append("greedy-semantic-sequence")
        if values["log_mel_cosine"] < 0.99:
            case_failures.append("log-mel")
        if values["speaker_cosine"] < 0.98:
            case_failures.append("speaker")
        if values["duration_difference_ratio"] > 0.03:
            case_failures.append("duration")
        if row.get("content_regression") is not False:
            case_failures.append("content")
        if row.get("new_artifacts") is not False:
            case_failures.append("artifacts")
        row["failures"] = case_failures
        row["passed"] = not case_failures
        normalized.append(row)
        failures.extend(f"case-{index}:{failure}" for failure in case_failures)
    return {
        "schema": CONVERSION_PARITY_SCHEMA,
        "status": "passed" if not failures else "failed",
        "backend_chain": ["PyTorch", "ONNX", "TensorRT-11"],
        "production_pytorch_fallback": False,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "engine_build_sha256": engine_build_sha256,
        "cases": normalized,
        "failures": failures,
    }


def require_passed_conversion(path: Path) -> dict[str, Any]:
    report = _read_object(path, "conversion parity report")
    if (
        report.get("schema") != CONVERSION_PARITY_SCHEMA
        or report.get("status") != "passed"
        or report.get("production_pytorch_fallback") is not False
    ):
        raise ProductionGateError("model packaging is blocked by conversion parity")
    return report


__all__ = [
    "CONVERSION_PARITY_SCHEMA",
    "HOLDOUT_EVALUATION_SCHEMA",
    "PRODUCTION_CHAIN",
    "ProductionGateError",
    "conversion_parity_report",
    "holdout_evaluation_report",
    "open_locked_test_holdout",
    "require_passed_conversion",
    "require_passed_holdout",
    "speaker_identity_gate",
]
