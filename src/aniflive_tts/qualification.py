from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime
from typing import Any

QUALIFICATION_REPORT_SCHEMA = "aniflive-tts-qualification-v1"
QUALIFICATION_COMPOSITION_SCHEMA = "aniflive-tts-qualification-composition-v1"
AUTOMATED_EVALUATION_SCHEMA = "aniflive-tts-workstation-evaluation-v1"
CONTENT_REVIEW_SCHEMA = "aniflive-tts-content-review-v1"
QUALIFICATION_COMPOSITION_V2 = "aniflive-tts-qualification-composition-v2"
QUALIFICATION_COMPOSITION_V3 = "aniflive-tts-qualification-composition-v3"
BLIND_AB_EVIDENCE_SCHEMA = "aniflive-tts-blind-ab-evidence-v1"
SECURITY_EVIDENCE_SCHEMA = "aniflive-tts-security-verification-v1"
BLIND_AB_GATES = frozenset({"long-form-continuity", "expression-quality"})
REQUIRED_SECURITY_CHECKS = (
    "release-inventory",
    "private-asset-scan",
    "credential-scan",
    "action-pin-review",
    "public-webui-state",
    "oci-source-metadata",
)
QUALIFICATION_SUBJECT_KINDS = frozenset({"artifact", "expression"})
REQUIRED_QUALIFICATION_GATES = (
    "multilingual",
    "speaker-identity",
    "streaming-parity",
    "long-form-continuity",
    "expression-quality",
    "tensorrt-runtime",
    "latency-performance",
    "security",
)
QUALIFICATION_GATE_LABELS = {
    "multilingual": "Five languages",
    "speaker-identity": "Speaker identity",
    "streaming-parity": "Streaming parity",
    "long-form-continuity": "Long-form continuity",
    "expression-quality": "Expression quality",
    "tensorrt-runtime": "TensorRT runtime",
    "latency-performance": "TTFA and RTF",
    "security": "Security",
}


class QualificationReportError(ValueError):
    pass


def _text(value: Any, *, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QualificationReportError(f"{field} must not be empty")
    result = " ".join(value.strip().split())
    if len(result) > maximum:
        raise QualificationReportError(f"{field} is limited to {maximum} characters")
    return result


def _json_object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationReportError(f"{field} must be a JSON object")
    try:
        encoded = json.dumps(dict(value), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise QualificationReportError(f"{field} must contain finite JSON values") from error
    if len(encoded.encode("utf-8")) > 64 * 1024:
        raise QualificationReportError(f"{field} is too large")
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise QualificationReportError(f"{field} must be a JSON object")
    return decoded


def _subject(value: Any, *, field: str = "subject") -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"kind", "id"}:
        raise QualificationReportError(f"{field} is malformed")
    kind = value.get("kind")
    if kind not in QUALIFICATION_SUBJECT_KINDS:
        raise QualificationReportError(f"{field} kind is unsupported")
    return {
        "kind": kind,
        "id": _text(value.get("id"), field=f"{field}.id", maximum=80),
    }


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise QualificationReportError(f"{field} must be a SHA-256 digest")
    result = value.strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise QualificationReportError(f"{field} must be a SHA-256 digest")
    return result


def _positive_integer(value: Any, *, field: str, maximum: int = 100_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise QualificationReportError(
            f"{field} must be an integer between 1 and {maximum}"
        )
    return value


def _timestamp(value: Any, *, field: str) -> str:
    text = _text(value, field=field, maximum=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise QualificationReportError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise QualificationReportError(f"{field} must include a timezone")
    return text


def _source_reference(value: Any, *, field: str, schema: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"artifact_id", "sha256", "schema"}:
        raise QualificationReportError(f"{field} is malformed")
    artifact_id = _text(value.get("artifact_id"), field=f"{field}.artifact_id", maximum=80)
    if not artifact_id.startswith("artifact_"):
        raise QualificationReportError(f"{field}.artifact_id is malformed")
    if value.get("schema") != schema:
        raise QualificationReportError(f"{field}.schema is unsupported")
    return {
        "artifact_id": artifact_id,
        "sha256": _sha256(value.get("sha256"), field=f"{field}.sha256"),
        "schema": schema,
    }


def parse_blind_ab_evidence(payload: Any) -> dict[str, Any]:
    """Parse an explicit human blind-listening decision.

    This parser deliberately never derives a pass from automated metrics. The
    human decision is required, and the protocol identity is retained as a
    separately checksummed artifact by the workstation composer.
    """

    if not isinstance(payload, Mapping):
        raise QualificationReportError("Blind A/B evidence must be a JSON object")
    required = {
        "schema", "subject", "gate", "decision", "protocol", "summary",
        "operator", "recorded_at",
    }
    if set(payload) != required:
        raise QualificationReportError("Blind A/B evidence fields are malformed")
    if payload.get("schema") != BLIND_AB_EVIDENCE_SCHEMA:
        raise QualificationReportError("Blind A/B evidence schema is unsupported")
    gate = payload.get("gate")
    if gate not in BLIND_AB_GATES:
        raise QualificationReportError("Blind A/B evidence gate is unsupported")
    decision = payload.get("decision")
    if decision not in {"passed", "failed"}:
        raise QualificationReportError("Blind A/B decision must be passed or failed")
    protocol = payload.get("protocol")
    protocol_fields = {
        "blinded", "comparison", "completed_trials", "listener_count",
        "sample_manifest_sha256", "randomization_sha256",
    }
    if not isinstance(protocol, Mapping) or set(protocol) != protocol_fields:
        raise QualificationReportError("Blind A/B protocol is malformed")
    if protocol.get("blinded") is not True:
        raise QualificationReportError("Blind A/B protocol must be blinded")
    if protocol.get("comparison") != "candidate-vs-baseline":
        raise QualificationReportError("Blind A/B comparison must be candidate-vs-baseline")
    parsed_protocol = {
        "blinded": True,
        "comparison": "candidate-vs-baseline",
        "completed_trials": _positive_integer(
            protocol.get("completed_trials"), field="protocol.completed_trials"
        ),
        "listener_count": _positive_integer(
            protocol.get("listener_count"), field="protocol.listener_count", maximum=10_000
        ),
        "sample_manifest_sha256": _sha256(
            protocol.get("sample_manifest_sha256"),
            field="protocol.sample_manifest_sha256",
        ),
        "randomization_sha256": _sha256(
            protocol.get("randomization_sha256"),
            field="protocol.randomization_sha256",
        ),
    }
    return {
        "schema": BLIND_AB_EVIDENCE_SCHEMA,
        "subject": _subject(payload.get("subject")),
        "gate": gate,
        "decision": decision,
        "protocol": parsed_protocol,
        "summary": _text(payload.get("summary"), field="summary", maximum=1000),
        "operator": _text(payload.get("operator"), field="operator", maximum=120),
        "recorded_at": _timestamp(payload.get("recorded_at"), field="recorded_at"),
    }


def parse_content_review(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema", "subject", "automated_evaluation_sha256", "reviews", "operator", "recorded_at",
    } or payload.get("schema") != CONTENT_REVIEW_SCHEMA:
        raise QualificationReportError("Content review schema or fields are invalid")
    rows = payload.get("reviews")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 5:
        raise QualificationReportError("Content review requires one to five language decisions")
    reviews = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "language", "audio_artifact_id", "audio_sha256", "decision", "summary",
        }:
            raise QualificationReportError("Content review decision is malformed")
        language = row["language"]
        if language not in {"zh", "yue", "en", "ja", "ko"} or language in seen:
            raise QualificationReportError("Content review language is invalid or duplicated")
        seen.add(language)
        if row["decision"] not in {"passed", "failed"}:
            raise QualificationReportError("Content review requires an explicit human decision")
        artifact_id = _text(row["audio_artifact_id"], field="audio_artifact_id", maximum=80)
        if not artifact_id.startswith("artifact_"):
            raise QualificationReportError("Content review audio artifact is invalid")
        reviews.append({
            "language": language, "audio_artifact_id": artifact_id,
            "audio_sha256": _sha256(row["audio_sha256"], field="audio_sha256"),
            "decision": row["decision"],
            "summary": _text(row["summary"], field="summary", maximum=1000),
        })
    return {
        "schema": CONTENT_REVIEW_SCHEMA, "subject": _subject(payload["subject"]),
        "automated_evaluation_sha256": _sha256(payload["automated_evaluation_sha256"],
                                               field="automated_evaluation_sha256"),
        "reviews": reviews,
        "operator": _text(payload["operator"], field="operator", maximum=120),
        "recorded_at": _timestamp(payload["recorded_at"], field="recorded_at"),
    }


def parse_security_evidence(payload: Any) -> dict[str, Any]:
    """Parse security evidence and derive its result from every required check."""

    if not isinstance(payload, Mapping):
        raise QualificationReportError("Security evidence must be a JSON object")
    required = {
        "schema", "subject", "tool", "release_tree_sha256", "checks", "completed_at"
    }
    if set(payload) != required:
        raise QualificationReportError("Security evidence fields are malformed")
    if payload.get("schema") != SECURITY_EVIDENCE_SCHEMA:
        raise QualificationReportError("Security evidence schema is unsupported")
    tool = payload.get("tool")
    if not isinstance(tool, Mapping) or set(tool) != {"name", "version"}:
        raise QualificationReportError("Security evidence tool is malformed")
    raw_checks = payload.get("checks")
    if not isinstance(raw_checks, list):
        raise QualificationReportError("Security evidence checks must be a JSON array")
    checks: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_checks):
        if not isinstance(raw, Mapping) or set(raw) != {"id", "status", "summary"}:
            raise QualificationReportError(f"checks[{index}] is malformed")
        check_id = _text(raw.get("id"), field=f"checks[{index}].id", maximum=80)
        if check_id in seen:
            raise QualificationReportError(f"Security check is duplicated: {check_id}")
        seen.add(check_id)
        status = raw.get("status")
        if status not in {"passed", "failed"}:
            raise QualificationReportError(
                f"checks[{index}].status must be passed or failed"
            )
        checks.append(
            {
                "id": check_id,
                "status": status,
                "summary": _text(
                    raw.get("summary"), field=f"checks[{index}].summary", maximum=1000
                ),
            }
        )
    missing = sorted(set(REQUIRED_SECURITY_CHECKS) - seen)
    unknown = sorted(seen - set(REQUIRED_SECURITY_CHECKS))
    if missing or unknown:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unknown:
            detail.append("unsupported " + ", ".join(unknown))
        raise QualificationReportError("Security evidence checks are incomplete: " + "; ".join(detail))
    checks.sort(key=lambda row: REQUIRED_SECURITY_CHECKS.index(row["id"]))
    return {
        "schema": SECURITY_EVIDENCE_SCHEMA,
        "subject": _subject(payload.get("subject")),
        "tool": {
            "name": _text(tool.get("name"), field="tool.name", maximum=160),
            "version": _text(tool.get("version"), field="tool.version", maximum=80),
        },
        "release_tree_sha256": _sha256(
            payload.get("release_tree_sha256"), field="release_tree_sha256"
        ),
        "checks": checks,
        "completed_at": _timestamp(payload.get("completed_at"), field="completed_at"),
        "status": (
            "passed" if all(check["status"] == "passed" for check in checks) else "failed"
        ),
    }


def _content_decisions(payload: Mapping[str, Any]) -> dict[str, bool]:
    if payload.get("baseline", {}).get("available") is True:
        return {language: isinstance(row, Mapping) and row.get("content_passed") is True
                for language, row in payload["baseline"]["languages"].items()}
    decisions = {}
    for language, row in payload["languages"].items():
        asr = row.get("asr", {})
        normalized = asr.get("orthographic_normalized", asr) if isinstance(asr, Mapping) else {}
        rate = normalized.get("error_rate") if isinstance(normalized, Mapping) else None
        decisions[language] = type(rate) in {int, float} and math.isfinite(rate) and rate == 0
    return decisions


def _first_use_evaluation_gates(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    from statistics import median
    from .workstation_production_gates import (
        SPEAKER_ABSOLUTE_MEDIAN_LIMIT, SPEAKER_GENERATED_P10_FLOOR, speaker_identity_gate,
    )
    languages, gates, runtime, benchmark = (
        payload.get(key) for key in ("languages", "gates", "runtime", "benchmark")
    )
    if (payload.get("status") != "measured"
            or not all(isinstance(value, Mapping) for value in (languages, gates, runtime, benchmark))
            or set(languages) != {"zh", "yue", "en", "ja", "ko"}
            or not all(isinstance(row, Mapping) for row in languages.values())):
        raise QualificationReportError("First-use evaluation evidence is incomplete")
    content = gates.get("five_language_enqueue") is True and all(_content_decisions(payload).values())
    identity = {}
    for language, row in languages.items():
        quality = row.get("quality")
        value = quality.get("speaker_cosine_stream_vs_identity") if isinstance(quality, Mapping) else None
        identity[language] = value
    valid_identity = all(type(value) in {int, float} and math.isfinite(value)
                         for value in identity.values())
    identity_median = identity_p10 = None
    speaker = False
    if valid_identity:
        ordered = sorted(identity.values())
        identity_median = median(ordered)
        position = (len(ordered) - 1) * 0.1
        lower = int(position)
        identity_p10 = ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)
        identity_gate = speaker_identity_gate({
            "centroid_cosine_median": identity_median,
            "centroid_cosine_p10": identity_p10,
        })
        speaker = identity_gate["passed"] and identity_p10 >= SPEAKER_GENERATED_P10_FLOOR
    streaming = gates.get("stream_complete_quality") is True and all(
        isinstance(row.get("quality_gate"), Mapping) and row["quality_gate"].get("passed") is True
        for row in languages.values())
    tensorrt = (
        gates.get("tensor_rt_engine_contract") is True
        and gates.get("five_language_enqueue") is True
        and gates.get("no_pytorch_neural_fallback") is True
        and runtime.get("backend") == "TensorRT-11"
        and runtime.get("pytorch_neural_fallback") is False
    )
    names = ("stream_keepalive_audible_ttfa_p50_ms",
             "stream_keepalive_audible_ttfa_p95_ms", "wall_rtf_p50")
    distribution = benchmark.get("session_distribution", {})
    if not isinstance(distribution, Mapping):
        distribution = {}
    values = {name: (distribution[name].get("session_median")
                     if isinstance(distribution.get(name), Mapping) else None) for name in names}
    proof = benchmark.get("execution_proof", {})
    if not isinstance(proof, Mapping):
        proof = {}
    measured = (
        gates.get("canonical_benchmark_completed") is True
        and isinstance(benchmark.get("session_records"), list) and bool(benchmark["session_records"])
        and proof.get("backend") == "TensorRT-11" and proof.get("pytorch_fallback") is False
        and all(type(proof.get(name)) is int and proof[name] > 0 for name in (
            "formal_full_wav_requests", "formal_stream_requests", "formal_keepalive_stream_requests"))
        and all(type(value) in {int, float} and math.isfinite(value) and value > 0
                for value in values.values())
    )
    decisions = (
        ("multilingual", content, {"languages": sorted(languages), "policy": "normalized-exact-content"}),
        ("speaker-identity", speaker, {"reference_cosines": identity,
                                      "reference_median": identity_median, "reference_p10": identity_p10,
                                      "median_limit": SPEAKER_ABSOLUTE_MEDIAN_LIMIT,
                                      "p10_floor": SPEAKER_GENERATED_P10_FLOOR}),
        ("streaming-parity", streaming, {"languages": sorted(languages)}),
        ("tensorrt-runtime", tensorrt, {"backend": runtime.get("backend")}),
        ("latency-performance", measured, {"measurement_only": True,
                                          "comparison_status": "unavailable", "values": values}),
    )
    return [{
        "id": name, "status": "passed" if passed else "failed",
        "summary": ("Performance measured; no historical regression comparison was made"
                    if name == "latency-performance" and passed else
                    f"First-use {QUALIFICATION_GATE_LABELS[name].lower()} check "
                    + ("passed" if passed else "requires attention")),
        "metrics": metrics,
    } for name, passed, metrics in decisions]


def automated_evaluation_gates(payload: Any) -> list[dict[str, Any]]:
    """Map the measured evaluation report to five automated production gates."""

    if not isinstance(payload, Mapping) or payload.get("schema") != AUTOMATED_EVALUATION_SCHEMA:
        raise QualificationReportError("Automated evaluation schema is unsupported")
    if isinstance(payload.get("baseline"), Mapping) and payload["baseline"].get("available") is False:
        return _first_use_evaluation_gates(payload)
    gates = payload.get("gates")
    languages = payload.get("languages")
    baseline = payload.get("baseline")
    benchmark = payload.get("benchmark")
    runtime = payload.get("runtime")
    if not all(isinstance(value, Mapping) for value in (gates, languages, baseline, benchmark, runtime)):
        raise QualificationReportError("Automated evaluation evidence is incomplete")
    expected_languages = {"zh", "yue", "en", "ja", "ko"}
    if set(languages) != expected_languages:
        raise QualificationReportError("Automated evaluation must contain exactly five languages")
    language_baseline = baseline.get("languages")
    performance = baseline.get("performance")
    if not isinstance(language_baseline, Mapping) or set(language_baseline) != expected_languages:
        raise QualificationReportError("Automated evaluation baseline language evidence is incomplete")
    if not isinstance(performance, Mapping):
        raise QualificationReportError("Automated evaluation performance evidence is incomplete")
    required_performance = {
        "stream_keepalive_audible_ttfa_p50_ms",
        "stream_keepalive_audible_ttfa_p95_ms",
        "wall_rtf_p50",
    }
    if set(performance) != required_performance:
        raise QualificationReportError(
            "Automated evaluation performance evidence has the wrong metrics"
        )
    session_records = benchmark.get("session_records")
    if not isinstance(session_records, list) or not session_records:
        raise QualificationReportError(
            "Automated evaluation canonical benchmark has no session records"
        )
    multilingual = bool(gates.get("five_language_enqueue")) and all(
        isinstance(language_baseline[language], Mapping)
        and language_baseline[language].get("content_passed") is True
        for language in expected_languages
    )
    speaker = all(
        isinstance(language_baseline[language], Mapping)
        and language_baseline[language].get("speaker_passed") is True
        for language in expected_languages
    )
    streaming = bool(gates.get("stream_complete_quality")) and all(
        isinstance(languages[language], Mapping)
        and isinstance(languages[language].get("quality_gate"), Mapping)
        and languages[language]["quality_gate"].get("passed") is True
        for language in expected_languages
    )
    tensorrt = bool(
        gates.get("tensor_rt_engine_contract")
        and gates.get("five_language_enqueue")
        and gates.get("no_pytorch_neural_fallback")
        and runtime.get("backend") == "TensorRT-11"
        and runtime.get("pytorch_neural_fallback") is False
    )
    latency = bool(
        gates.get("canonical_benchmark_completed")
        and baseline.get("available") is True
    ) and all(
        isinstance(row, Mapping) and row.get("passed") is True
        for row in performance.values()
    )
    values = (
        ("multilingual", multilingual, {"languages": sorted(expected_languages)}),
        ("speaker-identity", speaker, {"languages": sorted(expected_languages)}),
        ("streaming-parity", streaming, {"languages": sorted(expected_languages)}),
        ("tensorrt-runtime", tensorrt, {"backend": runtime.get("backend")}),
        ("latency-performance", latency, {"metrics": sorted(performance)}),
    )
    return [
        {
            "id": gate_id,
            "status": "passed" if passed else "failed",
            "summary": (
                f"Verified automated {QUALIFICATION_GATE_LABELS[gate_id].lower()} gate "
                + ("passed" if passed else "failed")
            ),
            "metrics": metrics,
        }
        for gate_id, passed, metrics in values
    ]


def compose_qualification_report(
    *,
    subject: Mapping[str, Any],
    automated_evaluation: Mapping[str, Any],
    long_form_evidence: Mapping[str, Any],
    expression_evidence: Mapping[str, Any] | None,
    security_evidence: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    content_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose checksummed automated and explicit evidence into schema v1."""

    parsed_subject = _subject(subject)
    automated_gates = automated_evaluation_gates(automated_evaluation)
    long_form = parse_blind_ab_evidence(long_form_evidence)
    expression_applicable = expression_evidence is not None
    if expression_applicable:
        expression = parse_blind_ab_evidence(expression_evidence)
    else:
        declared = automated_evaluation.get("runtime", {}).get("health", {}).get("expression")
        if (not isinstance(declared, Mapping) or declared.get("enabled") is not False
                or declared.get("profiles") != []):
            raise QualificationReportError("Expression listening evidence is required for this model")
        expression = {
            "gate": "expression-quality", "subject": parsed_subject, "decision": "passed",
            "summary": "Not applicable: the evaluated model declares no expression profiles",
            "protocol": {"completed_trials": 0, "listener_count": 0},
        }
    security = parse_security_evidence(security_evidence)
    if long_form["gate"] != "long-form-continuity":
        raise QualificationReportError("Long-form evidence used the wrong blind A/B gate")
    if expression["gate"] != "expression-quality":
        raise QualificationReportError("Expression evidence used the wrong blind A/B gate")
    for label, evidence_subject in (
        ("long-form", long_form["subject"]),
        ("expression", expression["subject"]),
        ("security", security["subject"]),
    ):
        if evidence_subject != parsed_subject:
            raise QualificationReportError(f"{label} evidence belongs to another subject")
    source_schemas = {
        "automated": AUTOMATED_EVALUATION_SCHEMA,
        "long_form": BLIND_AB_EVIDENCE_SCHEMA,
        "expression": BLIND_AB_EVIDENCE_SCHEMA,
        "security": SECURITY_EVIDENCE_SCHEMA,
    }
    if not expression_applicable:
        source_schemas.pop("expression")
    content = parse_content_review(content_evidence) if content_evidence is not None else None
    if content is not None:
        source_schemas["content"] = CONTENT_REVIEW_SCHEMA
        if content["subject"] != parsed_subject:
            raise QualificationReportError("Content review belongs to another subject")
    if set(sources) != set(source_schemas):
        raise QualificationReportError("Qualification source references are incomplete")
    refs = {
        key: _source_reference(sources[key], field=f"sources.{key}", schema=schema)
        for key, schema in source_schemas.items()
    }
    if len({ref["artifact_id"] for ref in refs.values()}) != len(refs):
        raise QualificationReportError("Qualification sources must be distinct artifacts")
    for gate in automated_gates:
        gate["evidence"] = refs["automated"]
    if content is not None:
        if content["automated_evaluation_sha256"] != refs["automated"]["sha256"]:
            raise QualificationReportError("Content review belongs to another evaluation")
        decisions = {row["language"]: row["decision"] == "passed" for row in content["reviews"]}
        passed = automated_evaluation["gates"].get("five_language_enqueue") is True and all(
            decisions.get(language, automatic)
            for language, automatic in _content_decisions(automated_evaluation).items()
        )
        multilingual = next(gate for gate in automated_gates if gate["id"] == "multilingual")
        multilingual.update(
            status="passed" if passed else "failed",
            summary="Content gate combines preserved automated results with exact-audio human review",
            evidence=refs["content"],
        )
        multilingual["metrics"]["human_reviewed_languages"] = sorted(decisions)
    manual_gates = [
        {
            "id": "long-form-continuity",
            "status": long_form["decision"],
            "summary": long_form["summary"],
            "metrics": {
                "completed_trials": long_form["protocol"]["completed_trials"],
                "listener_count": long_form["protocol"]["listener_count"],
                "decision_source": "human-blind-ab",
            },
            "evidence": refs["long_form"],
        },
        {
            "id": "expression-quality",
            "status": expression["decision"],
            "summary": expression["summary"],
            "metrics": {
                "completed_trials": expression["protocol"]["completed_trials"],
                "listener_count": expression["protocol"]["listener_count"],
                "decision_source": "human-blind-ab" if expression_applicable else "not-applicable",
                **({"applicable": False} if not expression_applicable else {}),
            },
            "evidence": refs["expression"] if expression_applicable else refs["automated"],
        },
        {
            "id": "security",
            "status": security["status"],
            "summary": (
                f"{len(security['checks'])} required release security checks verified"
            ),
            "metrics": {"required_checks": len(security["checks"])},
            "evidence": refs["security"],
        },
    ]
    gate_map = {gate["id"]: gate for gate in (*automated_gates, *manual_gates)}
    return {
        "schema": QUALIFICATION_REPORT_SCHEMA,
        "subject": parsed_subject,
        "gates": [gate_map[gate_id] for gate_id in REQUIRED_QUALIFICATION_GATES],
        "run_metadata": {
            "composition": {
                "schema": (QUALIFICATION_COMPOSITION_V3
                           if automated_evaluation["baseline"].get("available") is False
                           else QUALIFICATION_COMPOSITION_V2
                           if content is not None or not expression_applicable
                           else QUALIFICATION_COMPOSITION_SCHEMA),
                "sources": refs,
                "manual_gates": ["long-form-continuity"] + (["expression-quality"] if expression_applicable else []) + (["multilingual"] if content is not None else []),
            }
        },
    }


def parse_composed_qualification_report(payload: Any) -> dict[str, Any]:
    """Validate schema v1 and require the v1.4 fail-closed source composition."""

    report = parse_qualification_report(payload)
    composition = report["run_metadata"].get("composition")
    if not isinstance(composition, Mapping) or set(composition) != {
        "schema", "sources", "manual_gates"
    }:
        raise QualificationReportError("Qualification report has no verified composition")
    version = composition.get("schema")
    if version not in {QUALIFICATION_COMPOSITION_SCHEMA, QUALIFICATION_COMPOSITION_V2, QUALIFICATION_COMPOSITION_V3}:
        raise QualificationReportError("Qualification composition schema is unsupported")
    declared_sources = composition.get("sources")
    if not isinstance(declared_sources, Mapping):
        raise QualificationReportError("Qualification sources are malformed")
    content_reviewed = "content" in declared_sources
    expression_applicable = "expression" in declared_sources
    if version == QUALIFICATION_COMPOSITION_SCHEMA and (content_reviewed or not expression_applicable):
        raise QualificationReportError("Qualification extensions require composition v2")
    if composition.get("manual_gates") != ["long-form-continuity"] + (["expression-quality"] if expression_applicable else []) + (["multilingual"] if content_reviewed else []):
        raise QualificationReportError("Qualification manual gate contract is malformed")
    schemas = {
        "automated": AUTOMATED_EVALUATION_SCHEMA,
        "long_form": BLIND_AB_EVIDENCE_SCHEMA,
        "expression": BLIND_AB_EVIDENCE_SCHEMA,
        "security": SECURITY_EVIDENCE_SCHEMA,
    }
    if not expression_applicable:
        schemas.pop("expression")
    if content_reviewed:
        schemas["content"] = CONTENT_REVIEW_SCHEMA
    sources = composition.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(schemas):
        raise QualificationReportError("Qualification composition sources are incomplete")
    refs = {
        key: _source_reference(sources[key], field=f"composition.sources.{key}", schema=schema)
        for key, schema in schemas.items()
    }
    expected_source = {
        "multilingual": "automated",
        "speaker-identity": "automated",
        "streaming-parity": "automated",
        "long-form-continuity": "long_form",
        "expression-quality": "expression",
        "tensorrt-runtime": "automated",
        "latency-performance": "automated",
        "security": "security",
    }
    if not expression_applicable:
        expected_source["expression-quality"] = "automated"
    if content_reviewed:
        expected_source["multilingual"] = "content"
    if version == QUALIFICATION_COMPOSITION_V3:
        performance = next(g for g in report["gates"] if g["id"] == "latency-performance")
        if (performance["metrics"].get("measurement_only") is not True
                or performance["metrics"].get("comparison_status") != "unavailable"):
            raise QualificationReportError("First-use performance must remain explicitly measurement-only")
    for gate in report["gates"]:
        if gate["evidence"] != refs[expected_source[gate["id"]]]:
            raise QualificationReportError(
                f"Qualification gate {gate['id']} references the wrong evidence artifact"
            )
    report["run_metadata"]["composition"]["sources"] = refs
    return report


def parse_qualification_report(payload: Any) -> dict[str, Any]:
    """Validate a versioned, machine-produced qualification report.

    The report is evidence, not a UI checkbox. Promotion code still verifies the
    registered evaluation artifact checksum and lineage before trusting this data.
    """

    if not isinstance(payload, Mapping):
        raise QualificationReportError("Qualification report must be a JSON object")
    unknown = set(payload) - {"schema", "subject", "gates", "run_metadata"}
    if unknown:
        raise QualificationReportError(
            "Qualification report contains unsupported fields: "
            + ", ".join(sorted(str(value) for value in unknown))
        )
    if payload.get("schema") != QUALIFICATION_REPORT_SCHEMA:
        raise QualificationReportError("Qualification report schema is unsupported")

    subject = payload.get("subject")
    if not isinstance(subject, Mapping) or set(subject) != {"kind", "id"}:
        raise QualificationReportError("Qualification report subject is malformed")
    subject_kind = subject.get("kind")
    if subject_kind not in QUALIFICATION_SUBJECT_KINDS:
        raise QualificationReportError("Qualification report subject kind is unsupported")
    subject_id = _text(subject.get("id"), field="subject.id", maximum=80)

    gate_values = payload.get("gates")
    if not isinstance(gate_values, list) or not 1 <= len(gate_values) <= 32:
        raise QualificationReportError("Qualification report gates must contain 1 to 32 entries")
    gates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(gate_values):
        if not isinstance(raw, Mapping):
            raise QualificationReportError(f"gates[{index}] must be a JSON object")
        unknown_gate = set(raw) - {"id", "status", "summary", "metrics", "evidence"}
        if unknown_gate:
            raise QualificationReportError(
                f"gates[{index}] contains unsupported fields: "
                + ", ".join(sorted(str(value) for value in unknown_gate))
            )
        gate_id = _text(raw.get("id"), field=f"gates[{index}].id", maximum=80).lower()
        if gate_id in seen:
            raise QualificationReportError(f"Qualification gate is duplicated: {gate_id}")
        seen.add(gate_id)
        status = raw.get("status")
        if status not in {"passed", "failed"}:
            raise QualificationReportError(
                f"gates[{index}].status must be passed or failed"
            )
        summary = raw.get("summary", "")
        if summary:
            summary = _text(summary, field=f"gates[{index}].summary", maximum=1000)
        elif not isinstance(summary, str):
            raise QualificationReportError(f"gates[{index}].summary must be a string")
        metrics = _json_object(raw.get("metrics", {}), field=f"gates[{index}].metrics")
        for key, value in metrics.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise QualificationReportError(
                    f"gates[{index}].metrics.{key} must be finite"
                )
        evidence = _json_object(raw.get("evidence", {}), field=f"gates[{index}].evidence")
        gates.append(
            {
                "id": gate_id,
                "status": status,
                "summary": summary,
                "metrics": metrics,
                "evidence": evidence,
            }
        )

    missing = sorted(set(REQUIRED_QUALIFICATION_GATES) - seen)
    if missing:
        raise QualificationReportError(
            "Qualification report is missing required gates: " + ", ".join(missing)
        )
    run_metadata = _json_object(payload.get("run_metadata", {}), field="run_metadata")
    overall_status = (
        "passed"
        if all(gate["status"] == "passed" for gate in gates)
        else "failed"
    )
    return {
        "schema": QUALIFICATION_REPORT_SCHEMA,
        "subject": {"kind": subject_kind, "id": subject_id},
        "gates": gates,
        "run_metadata": run_metadata,
        "overall_status": overall_status,
    }
