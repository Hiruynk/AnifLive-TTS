from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from .dataset_factory import DatasetFactory
from .workstation import WorkstationStore


VOICE_ACQUISITION_REPORT_SCHEMA = "aniflive-voice-acquisition-qualification-v1"


def _job_payload(job: Mapping[str, Any]) -> Mapping[str, Any]:
    value: Any = job.get("result", {})
    for key in ("backend", "payload"):
        if isinstance(value, Mapping) and isinstance(value.get(key), Mapping):
            value = value[key]
    if isinstance(value, Mapping) and isinstance(value.get("payload"), Mapping):
        value = value["payload"]
    return value if isinstance(value, Mapping) else {}


def _worker_path(artifact: Mapping[str, Any]) -> str:
    metadata = artifact.get("metadata")
    value = metadata.get("worker_relative_path") if isinstance(metadata, Mapping) else None
    return value if isinstance(value, str) else ""


def _vad_backend(payload: Mapping[str, Any]) -> str:
    value = payload.get("vad")
    backend = value.get("backend") if isinstance(value, Mapping) else None
    return backend if isinstance(backend, str) else ""


def build_voice_acquisition_report(
    store: WorkstationStore,
    factory: DatasetFactory,
    dataset_id: str,
) -> dict[str, Any]:
    """Compose an evidence-only Dataset-to-production report.

    The report never infers success from a project name or a queued job. Every
    production gate is backed by persisted Dataset Factory state, a succeeded
    worker job, a ready artifact, or a passed qualification record.
    """

    project = store.get_project(dataset_id)
    if project.get("kind") != "dataset":
        raise ValueError("voice acquisition reports require a Dataset project")
    state = factory.project_state(dataset_id)
    items = factory.list_items(dataset_id, kind="segment")
    accepted = [item for item in items if item.get("review_status") == "accepted"]
    reviewed = [item for item in items if item.get("review_complete") is True]

    routes = {route: 0 for route in ("clean", "salvage", "review", "reject")}
    backends: set[str] = set()
    for item in items:
        metadata = item.get("metadata")
        acquisition = metadata.get("acquisition") if isinstance(metadata, Mapping) else None
        if not isinstance(acquisition, Mapping):
            continue
        route = acquisition.get("route")
        if route == "salvaged":
            route = "salvage"
        if route in routes:
            routes[str(route)] += 1
        suggestion = acquisition.get("asr_suggestion")
        backend = suggestion.get("asr_backend") if isinstance(suggestion, Mapping) else None
        if isinstance(backend, str) and backend:
            backends.add(backend)

    project_jobs = [job for job in store.list_jobs() if job.get("project_id") == dataset_id]
    final_jobs = [
        job
        for job in project_jobs
        if job.get("type") == "dataset.finalize" and job.get("status") == "succeeded"
    ]
    process_jobs = [
        job
        for job in project_jobs
        if job.get("type") == "dataset.process" and job.get("status") == "succeeded"
    ]
    source_seconds = 0.0
    maximum_source_seconds = 0.0
    reference_seconds: list[float] = []
    final_payloads: list[Mapping[str, Any]] = []
    for job in final_jobs:
        payload = _job_payload(job)
        if payload.get("schema") != "aniflive-target-speaker-finalize-v1":
            continue
        final_payloads.append(payload)
        value = payload.get("source_seconds")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            duration = max(0.0, float(value))
            source_seconds += duration
            maximum_source_seconds = max(maximum_source_seconds, duration)
        value = payload.get("reference_seconds")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            reference_seconds.append(max(0.0, float(value)))
        backend = payload.get("transcription_backend")
        if isinstance(backend, str) and backend:
            backends.add(backend)
    for job in process_jobs:
        payload = _job_payload(job)
        duration = payload.get("duration_seconds")
        input_record = payload.get("input")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            duration = (
                input_record.get("duration_seconds")
                if isinstance(input_record, Mapping)
                else None
            )
        if (
            (not isinstance(duration, (int, float)) or isinstance(duration, bool))
            and isinstance(input_record, Mapping)
        ):
            probe = input_record.get("probe")
            duration = (
                probe.get("duration_seconds") if isinstance(probe, Mapping) else None
            )
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            source_duration = max(0.0, float(duration))
            source_seconds += source_duration
            maximum_source_seconds = max(maximum_source_seconds, source_duration)

    candidates = factory.expression_reference_candidates(dataset_id)
    expression_groups = candidates.get("expressions", {})
    reference_profiles = len(expression_groups) if isinstance(expression_groups, Mapping) else 0
    classified = sum(
        bool((item.get("annotations") or {}).get("expression")) for item in accepted
    )

    training_projects = [
        record
        for record in store.list_projects(kind="training")
        if (record.get("config") or {}).get("source_dataset_id") == dataset_id
    ]
    training_ids = {str(record["id"]) for record in training_projects}
    training_jobs = [
        job
        for job in store.list_jobs()
        if job.get("project_id") in training_ids and job.get("type") == "training.prepare"
    ]
    training_passed = any(
        job.get("status") == "succeeded"
        and _job_payload(job).get("schema")
        == "aniflive-tts-v2proplus-training-report-v1"
        for job in training_jobs
    )

    artifacts = [
        artifact
        for project_id in training_ids
        for artifact in store.list_artifacts(project_id=project_id, status="ready")
    ]
    engine_paths = {
        PurePosixPath(path).as_posix()
        for artifact in artifacts
        if artifact.get("type") == "engine"
        and (path := _worker_path(artifact)).casefold().endswith(".engine")
    }
    package_ids = {
        str(artifact["id"]) for artifact in artifacts if artifact.get("type") == "package"
    }
    qualifications = [
        qualification
        for qualification in store.list_qualifications(subject_kind="artifact")
        if qualification.get("subject_id") in package_ids
    ]
    qualification_passed = any(
        record.get("overall_status") == "passed" for record in qualifications
    )

    missing: list[str] = []
    if not state.get("frozen"):
        missing.append("dataset-freeze")
    if not accepted:
        missing.append("accepted-clips")
    if len(reviewed) != len(items):
        missing.append("human-review")
    require_expressions = bool(state["config"].get("require_expressions", True))
    if require_expressions and classified != len(accepted):
        missing.append("expression-classification")
    acquisition_mode = state["acquisition_mode"]
    if acquisition_mode == "target-speaker":
        if not final_payloads:
            missing.append("target-speaker-finalize-evidence")
        if maximum_source_seconds < 1_800:
            missing.append("thirty-minute-source-qualification")
        acquisition_backends_passed = bool(final_payloads) and all(
            payload.get("speaker_backend") == "workstation-eres2netv2-tensorrt11"
            and _vad_backend(payload).startswith("fsmn-vad-")
            and payload.get("separator") == "MossFormer2_SS_16K"
            and str(payload.get("transcription_backend", "")).startswith("sensevoice-")
            for payload in final_payloads
        )
        if not acquisition_backends_passed:
            missing.append("offline-acquisition-backends")
    if not training_passed:
        missing.append("training")
    if len(engine_paths) != 9:
        missing.append("nine-tensorrt-engines")
    if not qualification_passed:
        missing.append("production-qualification")

    return {
        "schema": VOICE_ACQUISITION_REPORT_SCHEMA,
        "dataset_id": dataset_id,
        "acquisition_mode": acquisition_mode,
        "source_audio_seconds": round(source_seconds, 6),
        "maximum_single_source_seconds": round(maximum_source_seconds, 6),
        "target_reference_seconds": (
            round(max(reference_seconds), 6) if reference_seconds else None
        ),
        "tse": routes,
        "acquisition_evidence": {
            "finalized_sources": (
                len(final_payloads)
                if acquisition_mode == "target-speaker"
                else len(process_jobs)
            ),
            "speaker_backend": sorted(
                {
                    str(payload.get("speaker_backend"))
                    for payload in final_payloads
                    if payload.get("speaker_backend")
                }
            ),
            "vad_backend": sorted(
                {
                    _vad_backend(payload)
                    for payload in final_payloads
                    if _vad_backend(payload)
                }
            ),
            "separator": sorted(
                {
                    str(payload.get("separator"))
                    for payload in final_payloads
                    if payload.get("separator")
                }
            ),
        },
        "dataset": {
            "accepted": len(accepted),
            "duration_seconds": round(
                sum(float(item.get("duration_seconds") or 0.0) for item in accepted),
                6,
            ),
            "frozen": bool(state.get("frozen")),
            "manifest_sha256": state.get("frozen_manifest_sha256"),
        },
        "transcription": {
            "backend": next(iter(backends)) if len(backends) == 1 else "mixed" if backends else None,
            "human_reviewed": len(reviewed),
            "pending_review": len(items) - len(reviewed),
        },
        "expressions": {
            "classified": classified,
            "reference_profiles": reference_profiles,
        },
        "training": {"passed": training_passed},
        "engines": {
            "count": len(engine_paths),
            "backend": "TensorRT-11" if engine_paths else None,
        },
        "qualification": {"passed": qualification_passed},
        "ready_for_release": not missing,
        "missing_gates": missing,
    }


__all__ = ["VOICE_ACQUISITION_REPORT_SCHEMA", "build_voice_acquisition_report"]
