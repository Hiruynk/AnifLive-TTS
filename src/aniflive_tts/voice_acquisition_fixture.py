from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


VOICE_ACQUISITION_FIXTURE_SCHEMA = "aniflive-voice-acquisition-fixture-v1"
VOICE_ACQUISITION_FIXTURE_EVALUATION_SCHEMA = (
    "aniflive-voice-acquisition-fixture-evaluation-v1"
)
_KINDS = ("target", "overlap", "non_target")
_TRUTH_KINDS = (*_KINDS, "mixed")
_ROUTES = ("clean", "salvage", "review", "reject")


class VoiceAcquisitionFixtureError(ValueError):
    """Raised when labelled fixture evidence cannot be trusted."""


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise VoiceAcquisitionFixtureError(f"{name} is malformed")
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise VoiceAcquisitionFixtureError(f"{name} is malformed")
    return value


def _timeline(fixture: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if fixture.get("schema") != VOICE_ACQUISITION_FIXTURE_SCHEMA:
        raise VoiceAcquisitionFixtureError("fixture schema is unsupported")
    _integer(fixture.get("sample_rate"), "fixture sample rate", minimum=1)
    _sha256(fixture.get("output_sha256"), "fixture output SHA256")
    raw = fixture.get("timeline")
    if not isinstance(raw, list) or not raw:
        raise VoiceAcquisitionFixtureError("fixture timeline is empty")
    checked: list[dict[str, Any]] = []
    previous_start = -1
    for position, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise VoiceAcquisitionFixtureError("fixture timeline record is malformed")
        start = _integer(item.get("start_sample"), "fixture start sample")
        end = _integer(item.get("end_sample"), "fixture end sample", minimum=1)
        kind = item.get("kind")
        if end <= start or start < previous_start or kind not in _KINDS:
            raise VoiceAcquisitionFixtureError("fixture timeline geometry is malformed")
        checked.append(
            {
                "position": position,
                "kind": str(kind),
                "start_sample": start,
                "end_sample": end,
            }
        )
        previous_start = start
    return tuple(checked)


def _records(report: Mapping[str, Any], name: str) -> tuple[Mapping[str, Any], ...]:
    raw = report.get("records")
    if not isinstance(raw, list) or not raw:
        raise VoiceAcquisitionFixtureError(f"{name} report has no records")
    checked: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise VoiceAcquisitionFixtureError(f"{name} record is malformed")
        identifier = item.get("id")
        route = item.get("route")
        start = _integer(item.get("source_start_sample"), f"{name} start sample")
        end = _integer(item.get("source_end_sample"), f"{name} end sample", minimum=1)
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in seen
            or route not in _ROUTES
            or end <= start
        ):
            raise VoiceAcquisitionFixtureError(f"{name} record contract is malformed")
        seen.add(identifier)
        checked.append(item)
    return tuple(checked)


def _dominant_truth(
    record: Mapping[str, Any], timeline: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    start = int(record["source_start_sample"])
    end = int(record["source_end_sample"])
    by_kind = Counter({kind: 0 for kind in _KINDS})
    for item in timeline:
        overlap = max(
            0,
            min(end, int(item["end_sample"]))
            - max(start, int(item["start_sample"])),
        )
        by_kind[str(item["kind"])] += overlap
    labelled = sum(by_kind.values())
    if labelled <= 0:
        raise VoiceAcquisitionFixtureError(
            f"record {record['id']} does not overlap the labelled fixture"
        )
    material_kinds = [kind for kind in _KINDS if by_kind[kind] / labelled >= 0.05]
    maximum = max(by_kind.values())
    if len(material_kinds) > 1:
        truth_kind = "mixed"
    else:
        winners = [kind for kind in _KINDS if by_kind[kind] == maximum]
        if len(winners) != 1:
            raise VoiceAcquisitionFixtureError(
                f"record {record['id']} has ambiguous fixture truth"
            )
        truth_kind = winners[0]
    return {
        "kind": truth_kind,
        "labelled_samples": int(labelled),
        # VAD clips intentionally carry unlabelled context/silence. The safety
        # gate is therefore dominance within labelled speech, while full-clip
        # coverage remains a diagnostic rather than a pass/fail criterion.
        "labelled_audio_ratio": labelled / (end - start),
        "dominant_truth_ratio": maximum / labelled,
        "samples_by_kind": dict(by_kind),
    }


def _matrix(
    records: Sequence[Mapping[str, Any]],
    truths: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    matrix = {kind: {route: 0 for route in _ROUTES} for kind in _TRUTH_KINDS}
    for record in records:
        matrix[str(truths[str(record["id"])]["kind"])][str(record["route"])] += 1
    return matrix


def evaluate_voice_acquisition_fixture(
    fixture: Mapping[str, Any],
    routing: Mapping[str, Any],
    finalized: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare acquisition routes against a deterministic labelled recording.

    Review is a valid safety outcome. The gates only reject dangerous automatic
    decisions: accepting contamination, rejecting clean target speech, or
    replacing clean target speech with separator output.
    """

    timeline = _timeline(fixture)
    source_sha = _sha256(routing.get("source_sha256"), "routing source SHA256")
    if source_sha != fixture.get("output_sha256"):
        raise VoiceAcquisitionFixtureError(
            "routing source does not match the labelled fixture"
        )
    routing_records = _records(routing, "routing")
    final_records = _records(finalized, "final")
    routing_by_id = {str(record["id"]): record for record in routing_records}
    final_by_id = {str(record["id"]): record for record in final_records}
    if set(routing_by_id) != set(final_by_id):
        raise VoiceAcquisitionFixtureError("fixture record lineage is incomplete")
    truths = {
        identifier: _dominant_truth(record, timeline)
        for identifier, record in routing_by_id.items()
    }
    routing_matrix = _matrix(routing_records, truths)
    final_matrix = _matrix(final_records, truths)
    contaminated_total = sum(
        routing_matrix[kind][route]
        for kind in ("overlap", "non_target", "mixed")
        for route in _ROUTES
    )
    contamination_clean = sum(
        routing_matrix[kind]["clean"]
        for kind in ("overlap", "non_target", "mixed")
    )
    overlap_total = sum(routing_matrix["overlap"].values())
    overlap_contained = overlap_total - routing_matrix["overlap"]["clean"]
    target_processed = sum(
        truth["kind"] == "target"
        and isinstance(final_by_id[identifier].get("audio"), Mapping)
        and final_by_id[identifier]["audio"].get("selected") == "processed"
        for identifier, truth in truths.items()
    )
    labelled_audio_ratio = min(
        float(value["labelled_audio_ratio"]) for value in truths.values()
    )
    truth_purity = min(
        float(value["dominant_truth_ratio"]) for value in truths.values()
    )
    metrics = {
        "record_count": len(routing_records),
        "minimum_labelled_audio_ratio": labelled_audio_ratio,
        "minimum_dominant_truth_ratio": truth_purity,
        "overlap_containment_recall": (
            overlap_contained / overlap_total if overlap_total else 1.0
        ),
        "contamination_auto_clean_rate": (
            contamination_clean / contaminated_total if contaminated_total else 0.0
        ),
        "target_auto_reject_count": routing_matrix["target"]["reject"],
        "non_target_auto_clean_count": routing_matrix["non_target"]["clean"],
        "mixed_truth_count": sum(routing_matrix["mixed"].values()),
        "target_processed_count": target_processed,
    }
    gates = {
        "truth_assignment_complete": len(truths) == len(routing_records),
        "overlap_containment": metrics["overlap_containment_recall"] >= 0.80,
        "contamination_auto_clean": metrics["contamination_auto_clean_rate"] <= 0.06,
        "target_not_auto_rejected": metrics["target_auto_reject_count"] == 0,
        "non_target_not_auto_accepted": metrics["non_target_auto_clean_count"] == 0,
        "mixed_truth_quarantined": (
            routing_matrix["mixed"]["clean"] == 0
            and routing_matrix["mixed"]["reject"] == 0
        ),
        "clean_target_not_replaced": target_processed == 0,
    }
    return {
        "schema": VOICE_ACQUISITION_FIXTURE_EVALUATION_SCHEMA,
        "passed": all(gates.values()),
        "fixture_sha256": fixture["output_sha256"],
        "gates": gates,
        "metrics": metrics,
        "routing_matrix": routing_matrix,
        "final_matrix": final_matrix,
        "truth": {
            identifier: {
                "kind": value["kind"],
                "labelled_audio_ratio": value["labelled_audio_ratio"],
                "dominant_truth_ratio": value["dominant_truth_ratio"],
            }
            for identifier, value in sorted(truths.items())
        },
    }


__all__ = [
    "VOICE_ACQUISITION_FIXTURE_EVALUATION_SCHEMA",
    "VOICE_ACQUISITION_FIXTURE_SCHEMA",
    "VoiceAcquisitionFixtureError",
    "evaluate_voice_acquisition_fixture",
]
