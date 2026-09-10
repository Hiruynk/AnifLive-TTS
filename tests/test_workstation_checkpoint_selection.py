from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aniflive_tts import workstation_checkpoint_selection as selection


def _evidence(gpt: int, sovits: int, *, speaker: float, error: float) -> dict:
    return {
        "gpt_epoch": gpt,
        "sovits_epoch": sovits,
        "generation_success_rate": 1.0,
        "content": {
            "median_error": error,
            "p95_error": error,
            "repetition_failures": 0,
            "omission_failures": 0,
        },
        "speaker": {
            "centroid_cosine_median": speaker,
            "centroid_cosine_p10": speaker - 0.01,
        },
        "audio": {
            "invalid": 0,
            "nan": 0,
            "empty": 0,
            "clipped": 0,
            "duration_outliers": 0,
        },
        "stability": 1.0,
    }


def test_validation_selector_never_reads_test_manifest(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "validation").mkdir(parents=True)
    (bundle / "test").mkdir()
    (bundle / "training-input.json").write_text(
        json.dumps({"schema": "aniflive-v2proplus-training-input-v2"}),
        encoding="utf-8",
    )
    (bundle / "validation" / "manifest.json").write_text(
        json.dumps({"split": "validation", "items": [{"id": "validation-item"}]}),
        encoding="utf-8",
    )
    (bundle / "test" / "manifest.json").write_text("not-json", encoding="utf-8")

    assert selection.validation_records(bundle) == [{"id": "validation-item"}]


def test_last_epoch_is_not_automatic_and_earlier_epoch_can_win() -> None:
    report = selection.select_checkpoint_pair(
        [
            _evidence(12, 8, speaker=0.91, error=0.01),
            _evidence(8, 5, speaker=0.97, error=0.01),
        ],
        validation_manifest_sha256="a" * 64,
    )
    assert report["winner"] == {"gpt_epoch": 8, "sovits_epoch": 5}
    assert report["test_split_accessed"] is False


def test_catastrophic_content_failure_cannot_be_offset_by_speaker_score() -> None:
    bad = _evidence(12, 8, speaker=0.999, error=0.0)
    bad["content"]["repetition_failures"] = 1
    report = selection.select_checkpoint_pair(
        [bad, _evidence(8, 5, speaker=0.90, error=0.02)],
        validation_manifest_sha256="b" * 64,
    )
    assert report["winner"] == {"gpt_epoch": 8, "sovits_epoch": 5}
    assert report["evaluations"][0]["status"] == "failed"


def test_deployment_manifest_is_bound_to_selection_report(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidates"
    for role, suffix in (("gpt", ".ckpt"), ("sovits", ".pth")):
        (candidate_root / "checkpoints" / role).mkdir(parents=True, exist_ok=True)
        for epoch in (5, 8):
            path = candidate_root / "checkpoints" / role / f"voice-e{epoch}{suffix}"
            path.write_bytes(f"{role}-{epoch}".encode())
    candidates = {
        "schema": selection.CHECKPOINT_CANDIDATES_SCHEMA,
        "model_family": "gsv-v2proplus",
    }
    for role, suffix in (("gpt", ".ckpt"), ("sovits", ".pth")):
        candidates[role] = []
        for epoch in (5, 8):
            relative = f"checkpoints/{role}/voice-e{epoch}{suffix}"
            path = candidate_root / relative
            candidates[role].append(
                {
                    "epoch": epoch,
                    "relative_path": relative,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
    (candidate_root / "checkpoint-candidates.json").write_text(
        json.dumps(candidates), encoding="utf-8"
    )
    report = selection.select_checkpoint_pair(
        [
            _evidence(8, 8, speaker=0.9, error=0.01),
            _evidence(5, 5, speaker=0.95, error=0.01),
        ],
        validation_manifest_sha256="c" * 64,
    )
    report_path = tmp_path / "checkpoint-selection-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    _, deployment, copied = selection.materialize_deployment_checkpoints(
        candidate_root=candidate_root,
        selection_report_path=report_path,
        output=tmp_path / "selected",
    )

    assert deployment["gpt"]["epoch"] == 5
    assert deployment["selection"]["report_sha256"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    assert len(copied) == 5


def test_selection_fails_closed_when_every_pair_fails() -> None:
    bad = _evidence(1, 1, speaker=0.99, error=0.0)
    bad["audio"]["empty"] = 1
    with pytest.raises(selection.CheckpointSelectionError, match="no checkpoint pair"):
        selection.select_checkpoint_pair(
            [bad], validation_manifest_sha256="d" * 64
        )


def test_speaker_identity_failure_blocks_checkpoint_before_holdout() -> None:
    weak_voice = _evidence(3, 7, speaker=0.79, error=0.0)
    strong_voice = _evidence(5, 6, speaker=0.81, error=0.01)

    report = selection.select_checkpoint_pair(
        [weak_voice, strong_voice], validation_manifest_sha256="e" * 64
    )

    assert report["winner"] == {"gpt_epoch": 5, "sovits_epoch": 6}
    assert report["evaluations"][0]["hard_gate"]["failures"] == [
        "speaker-identity"
    ]
    assert report["hard_gate_limits"]["speaker_centroid_cosine_median"] == 0.80
    assert report["hard_gate_limits"]["source_calibrated_retention"] == {
        "source_p10_minimum": 0.72,
        "generated_median_floor": 0.72,
        "generated_p10_floor": 0.64,
        "median_retention_ratio": 0.84,
        "p10_retention_ratio": 0.82,
        "calibration": "qualified-model-source-control-retention-v1",
    }
    assert report["runner_up"] == {
        "gpt_epoch": 3,
        "sovits_epoch": 7,
        "qualified": False,
        "hard_gate_failures": ["speaker-identity"],
    }


def test_checkpoint_can_pass_calibrated_source_retention() -> None:
    calibrated = _evidence(12, 5, speaker=0.77, error=0.01)
    calibrated["speaker"].update(
        {
            "centroid_cosine_p10": 0.71,
            "source_centroid_cosine_median": 0.802,
            "source_centroid_cosine_p10": 0.783,
        }
    )
    weak = _evidence(3, 5, speaker=0.74, error=0.01)
    weak["speaker"].update(
        {
            "centroid_cosine_p10": 0.67,
            "source_centroid_cosine_median": 0.802,
            "source_centroid_cosine_p10": 0.783,
        }
    )

    report = selection.select_checkpoint_pair(
        [calibrated, weak], validation_manifest_sha256="1" * 64
    )

    assert report["winner"] == {"gpt_epoch": 12, "sovits_epoch": 5}
    gate = report["evaluations"][0]["speaker_identity_gate"]
    assert gate["mode"] == "source-calibrated-retention"
    assert gate["absolute_passed"] is False


def test_runner_up_must_be_distinct_when_only_one_pair_passes() -> None:
    winner = _evidence(10, 3, speaker=0.81, error=0.01)
    alternate = _evidence(10, 2, speaker=0.79, error=0.01)

    report = selection.select_checkpoint_pair(
        [winner, alternate], validation_manifest_sha256="f" * 64
    )

    assert report["winner"] == {"gpt_epoch": 10, "sovits_epoch": 3}
    assert report["runner_up"] == {
        "gpt_epoch": 10,
        "sovits_epoch": 2,
        "qualified": False,
        "hard_gate_failures": ["speaker-identity"],
    }


def test_gpt_probe_does_not_judge_the_temporary_sovits_voice() -> None:
    weak_temporary_voice = _evidence(3, 12, speaker=0.70, error=0.01)

    probe = selection.evaluate_checkpoint_evidence(
        [weak_temporary_voice], enforce_speaker_identity=False
    )
    final = selection.evaluate_checkpoint_evidence([weak_temporary_voice])

    assert probe[0]["status"] == "passed"
    assert "speaker-identity" not in probe[0]["hard_gate"]["failures"]
    assert final[0]["hard_gate"]["failures"] == ["speaker-identity"]


def test_reduced_sovits_probe_can_rank_below_final_speaker_gate() -> None:
    weak_probe_voice = _evidence(8, 2, speaker=0.77, error=0.01)

    probe = selection.evaluate_checkpoint_evidence(
        [weak_probe_voice], enforce_speaker_identity=False
    )
    final = selection.evaluate_checkpoint_evidence([weak_probe_voice])

    assert selection.rank_checkpoint_sweep(probe, "sovits") == [2]
    assert probe[0]["status"] == "passed"
    assert final[0]["status"] == "failed"
    assert final[0]["hard_gate"]["failures"] == ["speaker-identity"]


def test_reference_evaluations_keep_independent_hard_gates() -> None:
    weak = _evidence(8, 5, speaker=0.79, error=0.01)
    weak["reference_probe"] = {"count": 1, "item_ids": ["reference_weak"]}
    qualified = _evidence(8, 5, speaker=0.83, error=0.02)
    qualified["reference_probe"] = {
        "count": 1,
        "item_ids": ["reference_qualified"],
    }

    consolidated = selection.consolidate_reference_evaluations(
        [weak, qualified]
    )

    assert consolidated["hard_gate"]["passed"] is True
    assert consolidated["reference_probe"] == {
        "mode": "checkpoint-conditioned-reference-evaluation-v1",
        "count": 2,
        "item_ids": ["reference_weak", "reference_qualified"],
        "qualified_item_ids": ["reference_qualified"],
        "selected_item_id": "reference_qualified",
    }
    assert consolidated["reference_robustness"] == {
        "qualified_count": 1,
        "evaluated_count": 2,
    }
    assert len(consolidated["reference_evaluations"]) == 2


def test_reference_evaluations_cannot_mix_checkpoint_pairs() -> None:
    first = _evidence(8, 5, speaker=0.83, error=0.01)
    first["reference_probe"] = {"count": 1, "item_ids": ["reference_a"]}
    second = _evidence(9, 5, speaker=0.84, error=0.01)
    second["reference_probe"] = {"count": 1, "item_ids": ["reference_b"]}

    with pytest.raises(selection.CheckpointSelectionError, match="one checkpoint pair"):
        selection.consolidate_reference_evaluations([first, second])


def test_sweep_ranking_uses_role_specific_evidence() -> None:
    better_content = _evidence(3, 12, speaker=0.82, error=0.01)
    better_voice = _evidence(8, 6, speaker=0.92, error=0.02)
    rows = selection.evaluate_checkpoint_evidence([better_voice, better_content])

    assert selection.rank_checkpoint_sweep(rows, "gpt") == [3, 8]
    assert selection.rank_checkpoint_sweep(rows, "sovits") == [6, 12]


def test_failed_evidence_retains_machine_readable_gate_reasons() -> None:
    bad = _evidence(3, 8, speaker=0.99, error=0.25)
    bad["audio"]["duration_outliers"] = 1

    evaluated = selection.evaluate_checkpoint_evidence([bad])

    assert evaluated[0]["status"] == "failed"
    assert evaluated[0]["hard_gate"] == {
        "passed": False,
        "failures": [
            "duration-outliers",
            "median-content-error",
            "p95-content-error",
        ],
    }
