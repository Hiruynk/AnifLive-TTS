from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aniflive_tts import workstation_production_gates as gates


def _holdout_metrics() -> dict:
    return {
        "generation_success_rate": 1.0,
        "content": {
            "median_error": 0.0,
            "p95_error": 0.0,
            "repetition_failures": 0,
            "omission_failures": 0,
        },
        "speaker": {
            "centroid_cosine_median": 0.95,
            "centroid_cosine_p10": 0.93,
        },
        "audio": {
            "invalid": 0,
            "nan": 0,
            "empty": 0,
            "clipped": 0,
            "duration_outliers": 0,
        },
    }


def _lock() -> dict:
    return {
        "test_manifest_sha256": "a" * 64,
        "checkpoint_manifest_sha256": "b" * 64,
        "reference_manifest_sha256": "c" * 64,
    }


def test_failed_holdout_blocks_engine_build(tmp_path: Path) -> None:
    metrics = _holdout_metrics()
    metrics["content"]["omission_failures"] = 1
    report = gates.holdout_evaluation_report(metrics, lock=_lock())
    path = tmp_path / "holdout-evaluation.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(gates.ProductionGateError, match="engine build is blocked"):
        gates.require_passed_holdout(path)


def test_holdout_report_retains_per_case_diagnostics() -> None:
    report = gates.holdout_evaluation_report(
        _holdout_metrics(),
        lock=_lock(),
        cases=[
            {
                "index": 0,
                "source_item_id": "item_" + "a" * 36,
                "transcript": "今日はいい天気ですね。",
                "hypothesis": "今日はいい天気ですね。",
                "content_error": 0.0,
                "speaker_centroid_cosine": 0.95,
                "status": "passed",
            }
        ],
    )

    assert report["status"] == "passed"
    assert report["cases"][0]["content_error"] == 0.0


def test_source_calibrated_speaker_retention_preserves_absolute_gate() -> None:
    metrics = _holdout_metrics()
    metrics["speaker"] = {
        "centroid_cosine_median": 0.77,
        "centroid_cosine_p10": 0.71,
        "source_centroid_cosine_median": 0.802,
        "source_centroid_cosine_p10": 0.783,
    }

    report = gates.holdout_evaluation_report(metrics, lock=_lock())

    assert report["status"] == "passed"
    assert report["speaker_identity_gate"]["absolute_passed"] is False
    assert report["speaker_identity_gate"]["mode"] == "source-calibrated-retention"


def test_source_calibration_cannot_hide_material_identity_loss() -> None:
    metrics = _holdout_metrics()
    metrics["speaker"] = {
        "centroid_cosine_median": 0.74,
        "centroid_cosine_p10": 0.67,
        "source_centroid_cosine_median": 0.90,
        "source_centroid_cosine_p10": 0.82,
    }

    report = gates.holdout_evaluation_report(metrics, lock=_lock())

    assert report["status"] == "failed"
    assert report["failures"] == ["speaker-identity"]


def test_human_verified_source_can_calibrate_below_absolute_threshold() -> None:
    metrics = _holdout_metrics()
    metrics["speaker"] = {
        "centroid_cosine_median": 0.787,
        "centroid_cosine_p10": 0.693,
        "source_centroid_cosine_median": 0.781,
        "source_centroid_cosine_p10": 0.716,
        "source_human_verified": True,
        "source_case_count": 10,
    }

    report = gates.holdout_evaluation_report(metrics, lock=_lock())

    gate = report["speaker_identity_gate"]
    assert report["status"] == "passed"
    assert gate["absolute_passed"] is False
    assert gate["calibration"]["qualification"] == "human-verified-source"
    assert gate["mode"] == "source-calibrated-retention"


def test_human_verification_cannot_calibrate_source_below_safety_floor() -> None:
    metrics = _holdout_metrics()
    metrics["speaker"] = {
        "centroid_cosine_median": 0.70,
        "centroid_cosine_p10": 0.60,
        "source_centroid_cosine_median": 0.70,
        "source_centroid_cosine_p10": 0.60,
        "source_human_verified": True,
        "source_case_count": 10,
    }

    report = gates.holdout_evaluation_report(metrics, lock=_lock())

    gate = report["speaker_identity_gate"]
    assert report["status"] == "failed"
    assert gate["calibration"]["qualification"] == "unqualified"


def test_qualified_source_still_fails_when_identity_retention_is_too_low() -> None:
    metrics = _holdout_metrics()
    metrics["speaker"] = {
        "centroid_cosine_median": 0.72,
        "centroid_cosine_p10": 0.64,
        "source_centroid_cosine_median": 0.8174435911704798,
        "source_centroid_cosine_p10": 0.7812102535653145,
        "source_human_verified": True,
        "source_case_count": 10,
    }

    report = gates.holdout_evaluation_report(metrics, lock=_lock())

    gate = report["speaker_identity_gate"]
    assert report["status"] == "failed"
    assert gate["calibration"]["qualification"] == "absolute-source"
    assert (
        gate["calibration"]["p10_retention_ratio"]
        < gates.SPEAKER_P10_RETENTION_LIMIT
    )


def test_qualified_production_source_control_calibration_passes() -> None:
    metrics = _holdout_metrics()
    metrics["speaker"] = {
        "centroid_cosine_median": 0.7392109064245624,
        "centroid_cosine_p10": 0.6722333277379766,
        "source_centroid_cosine_median": 0.8666911176106498,
        "source_centroid_cosine_p10": 0.8047435106556673,
        "source_human_verified": True,
        "source_case_count": 24,
    }

    report = gates.holdout_evaluation_report(metrics, lock=_lock())

    gate = report["speaker_identity_gate"]
    assert report["status"] == "passed"
    assert gate["mode"] == "source-calibrated-retention"
    assert gate["calibration"]["median_retention_ratio"] >= 0.84
    assert gate["calibration"]["p10_retention_ratio"] >= 0.82


def test_absolute_floor_cannot_override_qualified_source_identity_loss() -> None:
    metrics = _holdout_metrics()
    metrics["speaker"] = {
        "centroid_cosine_median": 0.801,
        "centroid_cosine_p10": 0.76,
        "source_centroid_cosine_median": 0.98,
        "source_centroid_cosine_p10": 0.96,
    }

    report = gates.holdout_evaluation_report(metrics, lock=_lock())

    gate = report["speaker_identity_gate"]
    assert report["status"] == "failed"
    assert gate["absolute_passed"] is True
    assert gate["calibration"]["authoritative"] is True
    assert gate["mode"] == "failed"


def test_failed_conversion_parity_blocks_package(tmp_path: Path) -> None:
    report = gates.conversion_parity_report(
        [
            {
                "pytorch_onnx_logits_cosine": 1.0,
                "onnx_trt_logits_cosine": 1.0,
                "greedy_sequence_agreement": 1.0,
                "log_mel_cosine": 0.8,
                "speaker_cosine": 0.99,
                "duration_difference_ratio": 0.0,
                "content_regression": False,
                "new_artifacts": False,
            }
        ],
        checkpoint_manifest_sha256="d" * 64,
        engine_build_sha256="e" * 64,
    )
    path = tmp_path / "conversion-parity-report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(gates.ProductionGateError, match="packaging is blocked"):
        gates.require_passed_conversion(path)


@pytest.mark.parametrize(
    "human_decision", ["preferred", "confirm-auto-winner", "no-preference"]
)
def test_test_holdout_is_opened_only_after_checkpoint_and_human_reference_lock(
    tmp_path: Path,
    human_decision: str,
) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "test").mkdir(parents=True)
    (bundle / "training-input.json").write_text(
        json.dumps({"schema": "aniflive-v2proplus-training-input-v2"}),
        encoding="utf-8",
    )
    test_path = bundle / "test" / "manifest.json"
    test_path.write_text(
        json.dumps({"split": "test", "items": [{"id": "held-out"}]}),
        encoding="utf-8",
    )
    checkpoints = tmp_path / "deployment-checkpoints.json"
    checkpoints.write_text(
        json.dumps(
            {
                "schema": "aniflive-tts-v2proplus-deployment-checkpoints-v2",
                "selection": {"test_split_accessed": False},
            }
        ),
        encoding="utf-8",
    )
    reference = tmp_path / "deployment-reference.json"
    reference.write_text(
        json.dumps(
            {
                "schema": "aniflive-tts-v2proplus-deployment-reference-v2",
                "status": "human-locked",
                "human_decision": human_decision,
            }
        ),
        encoding="utf-8",
    )
    records, lock = gates.open_locked_test_holdout(
        training_bundle=bundle,
        deployment_checkpoints=checkpoints,
        deployment_reference=reference,
    )
    assert records == [{"id": "held-out"}]
    assert lock["test_manifest_sha256"] == hashlib.sha256(test_path.read_bytes()).hexdigest()


def test_automatic_production_chain_order_is_fail_closed() -> None:
    assert gates.PRODUCTION_CHAIN == (
        "training.prepare",
        "checkpoint.select",
        "reference.select",
        "holdout.evaluate",
        "engine.prepare",
        "conversion.parity",
        "model.package",
        "evaluation.prepare",
    )
