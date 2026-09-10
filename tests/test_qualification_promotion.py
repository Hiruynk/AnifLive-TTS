from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from aniflive_tts.qualification import (
    AUTOMATED_EVALUATION_SCHEMA,
    BLIND_AB_EVIDENCE_SCHEMA,
    QUALIFICATION_REPORT_SCHEMA,
    REQUIRED_QUALIFICATION_GATES,
    REQUIRED_SECURITY_CHECKS,
    SECURITY_EVIDENCE_SCHEMA,
    QualificationReportError,
    parse_qualification_report,
)
from aniflive_tts.workstation import WorkstationError, WorkstationStore


def _write_artifact(store: WorkstationStore, relative: str, content: bytes) -> Path:
    path = store.artifact_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _register_ready(
    store: WorkstationStore,
    *,
    artifact_type: str,
    name: str,
    relative: str,
    content: bytes,
    parents: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return store.register_artifact(
        artifact_type=artifact_type,
        name=name,
        status="ready",
        local_path=_write_artifact(store, relative, content),
        parent_artifact_ids=parents,
        metadata=metadata,
    )


def _report(
    *,
    subject_kind: str,
    subject_id: str,
    failed_gate: str | None = None,
) -> dict[str, object]:
    return {
        "schema": QUALIFICATION_REPORT_SCHEMA,
        "subject": {"kind": subject_kind, "id": subject_id},
        "gates": [
            {
                "id": gate_id,
                "status": "failed" if gate_id == failed_gate else "passed",
                "summary": f"{gate_id} measured",
                "metrics": {"score": 0.98, "requests": 1000},
                "evidence": {"report": f"reports/{gate_id}.json"},
            }
            for gate_id in REQUIRED_QUALIFICATION_GATES
        ],
        "run_metadata": {
            "product": "AnifLive-TTS",
            "runtime": "TensorRT-11",
            "hardware": "RTX 5070 Ti",
        },
    }


def _lineage(store: WorkstationStore) -> dict[str, dict[str, object]]:
    dataset = _register_ready(
        store,
        artifact_type="dataset",
        name="Dataset",
        relative="datasets/manifest.json",
        content=b"{}",
    )
    checkpoint = _register_ready(
        store,
        artifact_type="checkpoint",
        name="Checkpoint",
        relative="checkpoints/model.ckpt",
        content=b"checkpoint",
        parents=[dataset["id"]],
    )
    expression_bank = _register_ready(
        store,
        artifact_type="expression-bank",
        name="Expression bank",
        relative="expressions/bank.json",
        content=b"{}",
        parents=[checkpoint["id"]],
    )
    engine = _register_ready(
        store,
        artifact_type="engine",
        name="TensorRT engine",
        relative="engines/model.plan",
        content=b"engine",
        parents=[expression_bank["id"]],
        metadata={"backend": "TensorRT-11"},
    )
    return {
        "dataset": dataset,
        "checkpoint": checkpoint,
        "expression-bank": expression_bank,
        "engine": engine,
    }


def _automated_report(*, failed_gate: str | None = None) -> dict[str, object]:
    languages = {
        language: {
            "quality_gate": {"passed": failed_gate != "streaming-parity"},
        }
        for language in ("zh", "yue", "en", "ja", "ko")
    }
    baseline_languages = {
        language: {
            "content_passed": failed_gate != "multilingual",
            "speaker_passed": failed_gate != "speaker-identity",
        }
        for language in languages
    }
    return {
        "schema": AUTOMATED_EVALUATION_SCHEMA,
        "runtime": {
            "backend": "TensorRT-11",
            "pytorch_neural_fallback": failed_gate == "tensorrt-runtime",
        },
        "languages": languages,
        "benchmark": {"session_records": [{"session": 1}]},
        "baseline": {
            "available": True,
            "languages": baseline_languages,
            "performance": {
                "stream_keepalive_audible_ttfa_p50_ms": {
                    "passed": failed_gate != "latency-performance"
                },
                "stream_keepalive_audible_ttfa_p95_ms": {
                    "passed": failed_gate != "latency-performance"
                },
                "wall_rtf_p50": {"passed": failed_gate != "latency-performance"},
            },
        },
        "gates": {
            "tensor_rt_engine_contract": failed_gate != "tensorrt-runtime",
            "five_language_enqueue": True,
            "no_pytorch_neural_fallback": failed_gate != "tensorrt-runtime",
            "stream_complete_quality": failed_gate != "streaming-parity",
            "canonical_benchmark_completed": True,
            "baseline_regression": failed_gate != "latency-performance",
        },
    }


def _blind_report(
    *, subject_kind: str, subject_id: str, gate: str, decision: str = "passed"
) -> dict[str, object]:
    return {
        "schema": BLIND_AB_EVIDENCE_SCHEMA,
        "subject": {"kind": subject_kind, "id": subject_id},
        "gate": gate,
        "decision": decision,
        "protocol": {
            "blinded": True,
            "comparison": "candidate-vs-baseline",
            "completed_trials": 12,
            "listener_count": 2,
            "sample_manifest_sha256": "a" * 64,
            "randomization_sha256": "b" * 64,
        },
        "summary": f"Explicit human decision for {gate}",
        "operator": "listener-panel-1",
        "recorded_at": "2026-08-31T12:00:00Z",
    }


def _security_report(*, subject_kind: str, subject_id: str) -> dict[str, object]:
    return {
        "schema": SECURITY_EVIDENCE_SCHEMA,
        "subject": {"kind": subject_kind, "id": subject_id},
        "tool": {"name": "check_release_security.py", "version": "1"},
        "release_tree_sha256": "c" * 64,
        "checks": [
            {"id": check_id, "status": "passed", "summary": f"{check_id} passed"}
            for check_id in REQUIRED_SECURITY_CHECKS
        ],
        "completed_at": "2026-08-31T12:00:00Z",
    }


def _json_evidence(
    store: WorkstationStore,
    *,
    name: str,
    payload: dict[str, object],
    parents: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return _register_ready(
        store,
        artifact_type="evaluation",
        name=name,
        relative=f"evaluations/{name}.json",
        content=json.dumps(payload, sort_keys=True).encode("utf-8"),
        parents=parents,
        metadata=metadata,
    )


def _compose(
    store: WorkstationStore,
    *,
    subject_kind: str,
    subject_id: str,
    failed_gate: str | None = None,
) -> dict[str, object]:
    variant = failed_gate or "passed"
    automated = _json_evidence(
        store,
        name=f"automated-{variant}-{subject_id}",
        payload=_automated_report(failed_gate=failed_gate),
        parents=[subject_id] if subject_kind == "artifact" else None,
        metadata={"expression_id": subject_id} if subject_kind == "expression" else None,
    )
    long_form = _json_evidence(
        store,
        name=f"long-form-{variant}-{subject_id}",
        payload=_blind_report(
            subject_kind=subject_kind,
            subject_id=subject_id,
            gate="long-form-continuity",
            decision="failed" if failed_gate == "long-form-continuity" else "passed",
        ),
    )
    expression = _json_evidence(
        store,
        name=f"expression-{variant}-{subject_id}",
        payload=_blind_report(
            subject_kind=subject_kind,
            subject_id=subject_id,
            gate="expression-quality",
            decision="failed" if failed_gate == "expression-quality" else "passed",
        ),
    )
    security = _json_evidence(
        store,
        name=f"security-{variant}-{subject_id}",
        payload=_security_report(subject_kind=subject_kind, subject_id=subject_id),
    )
    return store.compose_qualification(
        subject_kind=subject_kind,
        subject_id=subject_id,
        automated_evaluation_artifact_id=automated["id"],
        long_form_evidence_artifact_id=long_form["id"],
        expression_evidence_artifact_id=expression["id"],
        security_evidence_artifact_id=security["id"],
    )


def test_artifact_promotion_requires_verified_complete_evidence(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    lineage = _lineage(store)
    engine = lineage["engine"]
    composed = _compose(
        store, subject_kind="artifact", subject_id=engine["id"]
    )
    qualification = composed["qualification"]
    assert qualification["overall_status"] == "passed"
    assert {gate["id"] for gate in qualification["gates"]} == set(
        REQUIRED_QUALIFICATION_GATES
    )
    assert "streaming-parity" in REQUIRED_QUALIFICATION_GATES
    assert "security" in REQUIRED_QUALIFICATION_GATES

    promoted = store.promote_artifact(
        engine["id"], qualification_id=qualification["id"]
    )
    assert promoted["promoted"] is True
    assert promoted["promotion"]["qualification_id"] == qualification["id"]

    details = store.artifact_details(engine["id"])
    assert [node["type"] for node in details["lineage"]["nodes"]][:4] == [
        "dataset", "checkpoint", "expression-bank", "engine"
    ]
    assert details["qualifications"] == [qualification]
    assert WorkstationStore(tmp_path / "workstation").get_artifact(engine["id"])[
        "promoted"
    ] is True


def test_failed_or_changed_evidence_never_promotes(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    engine = _lineage(store)["engine"]
    failed = _compose(
        store,
        subject_kind="artifact",
        subject_id=engine["id"],
        failed_gate="speaker-identity",
    )["qualification"]
    assert failed["overall_status"] == "failed"
    with pytest.raises(WorkstationError, match="did not pass"):
        store.promote_artifact(engine["id"], qualification_id=failed["id"])
    assert store.get_artifact(engine["id"])["promoted"] is False


def test_manual_gate_never_auto_passes_and_nested_evidence_is_reverified(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    engine = _lineage(store)["engine"]
    failed = _compose(
        store,
        subject_kind="artifact",
        subject_id=engine["id"],
        failed_gate="long-form-continuity",
    )["qualification"]
    assert failed["overall_status"] == "failed"
    assert next(
        gate for gate in failed["gates"] if gate["id"] == "long-form-continuity"
    )["status"] == "failed"
    with pytest.raises(WorkstationError, match="did not pass"):
        store.promote_artifact(engine["id"], qualification_id=failed["id"])

    composed = _compose(store, subject_kind="artifact", subject_id=engine["id"])
    qualification = composed["qualification"]
    report_path = store.artifact_root / composed["evaluation_artifact"]["local_path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source_id = report["run_metadata"]["composition"]["sources"]["long_form"][
        "artifact_id"
    ]
    source = store.get_artifact(source_id)
    source_path = store.artifact_root / source["local_path"]
    source_path.write_text("{}", encoding="utf-8")
    with pytest.raises(WorkstationError, match="changed after composition|changed after registration"):
        store.promote_artifact(engine["id"], qualification_id=qualification["id"])


def test_self_asserted_legacy_report_is_not_importable(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    engine = _lineage(store)["engine"]
    evaluation = _json_evidence(
        store,
        name="self-asserted",
        payload=_report(subject_kind="artifact", subject_id=engine["id"]),
        parents=[engine["id"]],
    )
    with pytest.raises(WorkstationError, match="verified composition"):
        store.record_qualification(
            evaluation_artifact_id=evaluation["id"],
            subject_kind="artifact",
            subject_id=engine["id"],
        )

    composed = _compose(store, subject_kind="artifact", subject_id=engine["id"])
    passed = composed["qualification"]
    report_path = store.artifact_root / composed["evaluation_artifact"]["local_path"]
    report_path.write_text("{}", encoding="utf-8")
    with pytest.raises(WorkstationError, match="changed after registration"):
        store.promote_artifact(engine["id"], qualification_id=passed["id"])
    assert store.get_artifact(engine["id"])["promoted"] is False


def test_expression_qualification_is_evidence_backed_and_fail_closed(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    with pytest.raises(WorkstationError, match="evaluation evidence"):
        store.create_expression_draft(
            name="Invalid",
            profile_id="invalid",
            language="ja",
            emotion="neutral",
            intensity=0.4,
            qualification_status="qualified",
        )
    expression = store.create_expression_draft(
        name="Relieved",
        profile_id="relieved",
        model_id="voice-v2proplus",
        language="ja",
        emotion="relieved",
        intensity=0.7,
    )
    with pytest.raises(WorkstationError, match="evaluation evidence"):
        store.update_expression_draft(
            expression["id"], qualification_status="qualified"
        )

    qualification = _compose(
        store, subject_kind="expression", subject_id=expression["id"]
    )["qualification"]
    promoted = store.promote_expression(
        expression["id"], qualification_id=qualification["id"]
    )
    assert promoted["qualification_status"] == "qualified"
    assert promoted["qualification"]["qualification_id"] == qualification["id"]

    unproven = store.create_expression_draft(
        name="Unproven",
        profile_id="unproven",
        language="ja",
        emotion="neutral",
        intensity=0.5,
    )

    with sqlite3.connect(store.database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError, match="evaluation evidence"):
            connection.execute(
                "UPDATE expression_drafts SET qualification_status = 'qualified' WHERE id = ?",
                (unproven["id"],),
            )


def test_schema_six_unproven_expression_qualification_is_revoked(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workstation"
    store = WorkstationStore(root)
    expression = store.create_expression_draft(
        name="Legacy",
        profile_id="legacy",
        language="ja",
        emotion="neutral",
        intensity=0.5,
    )
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DROP TRIGGER expression_qualified_update_guard")
        connection.execute(
            "UPDATE expression_drafts SET qualification_status = 'qualified' WHERE id = ?",
            (expression["id"],),
        )
        connection.execute("UPDATE metadata SET value = '6' WHERE key = 'schema'")

    migrated = WorkstationStore(root)
    assert migrated.get_expression_draft(expression["id"])["qualification_status"] == "pending"


def test_qualification_report_requires_every_gate_and_strict_json() -> None:
    report = _report(subject_kind="artifact", subject_id="artifact_" + "a" * 36)
    parsed = parse_qualification_report(report)
    assert parsed["overall_status"] == "passed"

    missing = dict(report)
    missing["gates"] = report["gates"][:-1]
    with pytest.raises(QualificationReportError, match="missing required gates"):
        parse_qualification_report(missing)

    unknown = dict(report)
    unknown["operator_override"] = True
    with pytest.raises(QualificationReportError, match="unsupported fields"):
        parse_qualification_report(unknown)


def test_content_failure_does_not_change_passed_performance_measurements():
    from aniflive_tts.qualification import automated_evaluation_gates
    report = _automated_report(failed_gate="multilingual")
    report["gates"]["baseline_regression"] = False
    gates = {row["id"]: row["status"] for row in automated_evaluation_gates(report)}
    assert gates["multilingual"] == "failed"
    assert gates["latency-performance"] == "passed"


def test_exact_audio_review_composes_without_rewriting_automated_report(tmp_path):
    from aniflive_tts.qualification import CONTENT_REVIEW_SCHEMA
    store = WorkstationStore(tmp_path / "state")
    subject_id = _lineage(store)["engine"]["id"]
    report = _automated_report()
    report["baseline"]["languages"]["ko"]["content_passed"] = False
    report["gates"]["baseline_regression"] = False
    report["languages"]["ko"]["complete_path"] = "audio/ko-complete.wav"
    automated = _json_evidence(store, name="raw-content-failure", payload=report,
                               parents=[subject_id], metadata={"job_id": "job_fixture"})
    audio = _register_ready(
        store, artifact_type="evaluation", name="reviewed-audio",
        relative="evaluations/ko-complete.wav", content=b"reviewed audio bytes",
        metadata={"job_id": "job_fixture", "worker_relative_path": "audio/ko-complete.wav"},
    )
    review = {
        "schema": CONTENT_REVIEW_SCHEMA,
        "subject": {"kind": "artifact", "id": subject_id},
        "automated_evaluation_sha256": automated["sha256"],
        "reviews": [{"language": "ko", "audio_artifact_id": audio["id"],
                     "audio_sha256": audio["sha256"], "decision": "passed",
                     "summary": "Listener confirmed complete content"}],
        "operator": "listener", "recorded_at": "2026-09-10T00:00:00Z",
    }
    review_artifact = _json_evidence(store, name="human-content", payload=review)
    evidence = {}
    for key, gate in [("long_form", "long-form-continuity"), ("expression", "expression-quality")]:
        evidence[key] = _json_evidence(store, name=key, payload=_blind_report(
            subject_kind="artifact", subject_id=subject_id, gate=gate))
    evidence["security"] = _json_evidence(store, name="security", payload=_security_report(
        subject_kind="artifact", subject_id=subject_id))
    composed = store.compose_qualification(
        subject_kind="artifact", subject_id=subject_id,
        automated_evaluation_artifact_id=automated["id"],
        content_evidence_artifact_id=review_artifact["id"],
        long_form_evidence_artifact_id=evidence["long_form"]["id"],
        expression_evidence_artifact_id=evidence["expression"]["id"],
        security_evidence_artifact_id=evidence["security"]["id"],
    )
    assert composed["qualification"]["overall_status"] == "passed"
    original = json.loads((store.artifact_root / automated["local_path"]).read_text())
    assert original["baseline"]["languages"]["ko"]["content_passed"] is False
    review["automated_evaluation_sha256"] = "0" * 64
    with pytest.raises(WorkstationError, match="another automated evaluation"):
        store._verify_content_review(review, automated, review["subject"])
    (store.artifact_root / audio["local_path"]).write_bytes(b"changed")
    with pytest.raises(WorkstationError, match="audio changed"):
        store.promote_artifact(subject_id,
                               qualification_id=composed["qualification"]["id"])


@pytest.mark.parametrize("expression_state", [
    {"enabled": True, "profiles": ["happy"]},
    {"enabled": False, "profiles": ["happy"]},
    None,
])
def test_expression_evidence_cannot_be_omitted_without_explicit_empty_catalog(expression_state):
    from aniflive_tts.qualification import compose_qualification_report
    report = _automated_report()
    report["runtime"]["health"] = {"expression": expression_state}
    with pytest.raises(QualificationReportError, match="Expression listening evidence"):
        compose_qualification_report(
            subject={"kind": "artifact", "id": "artifact_subject"},
            automated_evaluation=report,
            long_form_evidence=_blind_report(subject_kind="artifact", subject_id="artifact_subject",
                                            gate="long-form-continuity"),
            expression_evidence=None,
            security_evidence=_security_report(subject_kind="artifact", subject_id="artifact_subject"),
            sources={},
        )


def test_empty_expression_catalog_is_explicitly_not_applicable():
    from aniflive_tts.qualification import compose_qualification_report, parse_composed_qualification_report
    report = _automated_report()
    report["runtime"]["health"] = {"expression": {"enabled": False, "profiles": []}}
    sources = {
        name: {"artifact_id": "artifact_" + name, "sha256": "a" * 64, "schema": schema}
        for name, schema in [
            ("automated", AUTOMATED_EVALUATION_SCHEMA),
            ("long_form", BLIND_AB_EVIDENCE_SCHEMA),
            ("security", SECURITY_EVIDENCE_SCHEMA),
        ]
    }
    result = compose_qualification_report(
        subject={"kind": "artifact", "id": "artifact_subject"},
        automated_evaluation=report,
        long_form_evidence=_blind_report(subject_kind="artifact", subject_id="artifact_subject",
                                        gate="long-form-continuity"),
        expression_evidence=None,
        security_evidence=_security_report(subject_kind="artifact", subject_id="artifact_subject"),
        sources=sources,
    )
    parsed = parse_composed_qualification_report(result)
    gate = next(row for row in parsed["gates"] if row["id"] == "expression-quality")
    assert gate["metrics"]["applicable"] is False
    assert gate["metrics"]["completed_trials"] == 0
    assert parsed["run_metadata"]["composition"]["manual_gates"] == ["long-form-continuity"]


def _first_use_report():
    report = _automated_report()
    report["status"] = "measured"
    report["baseline"] = {"available": False, "passed": False}
    report["gates"]["baseline_regression"] = False
    for row in report["languages"].values():
        row["asr"] = {"error_rate": 0.0}
        row["quality"] = {"speaker_cosine_stream_vs_identity": 0.9}
    report["benchmark"]["session_distribution"] = {
        name: {"session_median": value} for name, value in {
            "stream_keepalive_audible_ttfa_p50_ms": 100,
            "stream_keepalive_audible_ttfa_p95_ms": 130,
            "wall_rtf_p50": 0.12,
        }.items()
    }
    report["benchmark"]["execution_proof"] = {
        "backend": "TensorRT-11", "pytorch_fallback": False,
        "formal_full_wav_requests": 1, "formal_stream_requests": 1,
        "formal_keepalive_stream_requests": 1,
    }
    return report


def test_first_use_preserves_measurement_only_and_requires_real_quality():
    from aniflive_tts.qualification import automated_evaluation_gates
    report = _first_use_report()
    gates = {g["id"]: g for g in automated_evaluation_gates(report)}
    assert all(g["status"] == "passed" for g in gates.values())
    assert gates["latency-performance"]["metrics"]["measurement_only"] is True
    assert report["baseline"]["available"] is False
    assert report["gates"]["baseline_regression"] is False
    report["languages"]["ko"]["asr"]["error_rate"] = 0.1
    report["languages"]["en"]["quality"]["speaker_cosine_stream_vs_identity"] = 0.2
    gates = {g["id"]: g for g in automated_evaluation_gates(report)}
    assert gates["multilingual"]["status"] == "failed"
    assert gates["speaker-identity"]["status"] == "failed"
    report["benchmark"]["session_distribution"]["wall_rtf_p50"]["session_median"] = None
    gates = {g["id"]: g for g in automated_evaluation_gates(report)}
    assert gates["latency-performance"]["status"] == "failed"


def test_first_use_model_can_qualify_without_claiming_historical_regression(tmp_path, monkeypatch):
    store = WorkstationStore(tmp_path / "state")
    subject = _lineage(store)["engine"]["id"]
    report = _first_use_report()
    monkeypatch.setattr(__import__(__name__), "_automated_report", lambda **kwargs: report)
    result = _compose(store, subject_kind="artifact", subject_id=subject)
    assert result["qualification"]["overall_status"] == "passed"
    gate = next(g for g in result["qualification"]["gates"] if g["id"] == "latency-performance")
    assert gate["metrics"]["comparison_status"] == "unavailable"
    assert store.promote_artifact(subject, qualification_id=result["qualification"]["id"])["promoted"]

