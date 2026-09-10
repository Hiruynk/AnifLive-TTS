from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from aniflive_tts.webui import create_webui_app
from aniflive_tts.workstation import WorkstationStore


ROOT = Path(__file__).resolve().parents[1]


def _upstream() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        payloads = {
            "/openapi.json": {"info": {"version": "1.4.0-dev"}},
            "/health": {
                "ready": True,
                "backend": "TensorRT-11",
                "engine_count": 9,
                "model": "voice-v2proplus",
                "gpu": {},
            },
            "/model/config": {
                "model": "voice-v2proplus",
                "version": "v2ProPlus",
                "backend": "TensorRT-11",
                "engine_count": 9,
                "sample_rate": 32000,
                "pytorch_fallback": False,
            },
            "/v1/models": {
                "object": "list",
                "data": [{"id": "voice-v2proplus", "active": True}],
            },
            "/v1/expressions": {
                "object": "list",
                "enabled": False,
                "profiles": [],
                "policies": [],
            },
        }
        return httpx.Response(200, json=payloads[request.url.path])

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://upstream.test"
    )


@pytest.fixture(autouse=True)
def _trusted_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANIFLIVE_TTS_WEBUI_TRUSTED_HOSTS", "testserver")


def test_training_dialog_and_monitor_use_real_worker_contract() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "webui" / "studio.js").read_text(encoding="utf-8")

    for element_id in (
        "projectPretrainedGpt",
        "projectPretrainedSovitsG",
        "projectPretrainedSovitsD",
        "projectResumeCheckpoint",
        "projectTrainingStage",
        "trainingRunButton",
        "productionBuildButton",
        "trainingProgress",
        "trainingMetrics",
        "trainingEventList",
        "trainingArtifactList",
        "referenceReviewPanel",
        "referenceCandidateGrid",
        "referenceNoPreferenceButton",
        "referenceAllPoorButton",
    ):
        assert f'id="{element_id}"' in index
    for preset in ("quick", "balanced", "high-quality", "advanced"):
        assert f'value="{preset}"' in index
    for key in (
        "pretrained_gpt",
        "pretrained_sovits_g",
        "pretrained_sovits_d",
        "resume_checkpoint",
        'type: "training.prepare"',
        "aniflive-tts-v2proplus-training-report-v1",
        "lockTrainingReference",
        'decision !== "no-preference"',
        'decision: "all-poor"',
        "blind-reference-manifest.json",
    ):
        assert key in script
    assert "worker does not emit structured loss" in script
    assert "fake loss" not in script.lower()


def test_rejecting_all_blind_references_keeps_holdout_sealed(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(
        kind="training",
        name="Voice training",
        config={
            "source_dataset_id": "dataset_00000000-0000-4000-8000-000000000001",
            "reference_status": "pending-human-evidence",
        },
    )
    checkpoint = store.create_job(
        job_type="checkpoint.select", project_id=project["id"]
    )
    checkpoint_claim = store.claim_job(checkpoint["id"])
    store.update_job(
        checkpoint["id"],
        status="succeeded",
        result={"registered_artifact_ids": []},
        claim_token=checkpoint_claim.token,
    )

    output = tmp_path / "reference-output"
    output.mkdir()
    report = output / "reference-selection-report.json"
    report.write_text(
        json.dumps(
            {
                "automatic_recommendation": "item_00000000-0000-4000-8000-000000000001",
                "policy": "reviewed-quality-speaker-centroid-v2",
                "top_candidates": [
                    {"item_id": "item_00000000-0000-4000-8000-000000000001"}
                ],
            }
        ),
        encoding="utf-8",
    )
    reference = store.create_job(
        job_type="reference.select",
        project_id=project["id"],
        depends_on=(checkpoint["id"],),
    )
    reference_claim = store.claim_job(reference["id"])
    staged = store.register_worker_artifacts(
        job_id=reference["id"],
        claim_token=reference_claim.token,
        job_type="reference.select",
        project_id=project["id"],
        source_root=output,
        artifacts=[
            {
                "kind": "reference",
                "relative_path": report.name,
                "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "size_bytes": report.stat().st_size,
            }
        ],
        image_digest="sha256:" + "a" * 64,
    )
    store.update_job(
        reference["id"],
        status="succeeded",
        result={"registered_artifact_ids": [staged[0]["id"]]},
        claim_token=reference_claim.token,
    )

    app = create_webui_app(
        static_dir=ROOT / "webui", client=_upstream(), workstation=store
    )
    with TestClient(app) as client:
        response = client.post(
            f"/api/workstation/training/{project['id']}/reference-selection/lock",
            json={"decision": "all-poor"},
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["reference"]["status"] == "all-poor"
    assert payload["artifact"]["status"] == "rejected"
    assert payload["holdout_job"] is None
    assert store.get_project(project["id"])["config"]["reference_status"] == "all-poor"
    assert not [job for job in store.list_jobs() if job["type"] == "holdout.evaluate"]
    assert store.get_artifact(payload["artifact"]["id"])["parent_artifact_ids"] == [
        staged[0]["id"]
    ]


def test_all_poor_reference_can_be_superseded_before_holdout_access(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    dataset_id = "dataset_00000000-0000-4000-8000-000000000001"
    item_id = "item_00000000-0000-4000-8000-000000000001"
    project = store.create_project(
        kind="training",
        name="Voice training",
        config={
            "source_dataset_id": dataset_id,
            "reference_status": "pending-human-evidence",
        },
    )
    checkpoint = store.create_job(
        job_type="checkpoint.select", project_id=project["id"]
    )
    checkpoint_claim = store.claim_job(checkpoint["id"])
    store.update_job(
        checkpoint["id"],
        status="succeeded",
        result={"registered_artifact_ids": []},
        claim_token=checkpoint_claim.token,
    )
    output = tmp_path / "reference-output"
    output.mkdir()
    report = output / "reference-selection-report.json"
    report.write_text(
        json.dumps(
            {
                "automatic_recommendation": item_id,
                "policy": "reviewed-quality-speaker-centroid-v2",
                "top_candidates": [{"item_id": item_id}],
            }
        ),
        encoding="utf-8",
    )
    reference = store.create_job(
        job_type="reference.select",
        project_id=project["id"],
        depends_on=(checkpoint["id"],),
    )
    reference_claim = store.claim_job(reference["id"])
    staged = store.register_worker_artifacts(
        job_id=reference["id"],
        claim_token=reference_claim.token,
        job_type="reference.select",
        project_id=project["id"],
        source_root=output,
        artifacts=[
            {
                "kind": "reference",
                "relative_path": report.name,
                "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "size_bytes": report.stat().st_size,
            }
        ],
        image_digest="sha256:" + "a" * 64,
    )
    store.update_job(
        reference["id"],
        status="succeeded",
        result={"registered_artifact_ids": [staged[0]["id"]]},
        claim_token=reference_claim.token,
    )
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"RIFF-reference")
    audio_sha256 = hashlib.sha256(audio.read_bytes()).hexdigest()

    class ReferenceDataset:
        @staticmethod
        def get_item(requested_item_id: str) -> dict[str, object]:
            assert requested_item_id == item_id
            return {
                "id": item_id,
                "dataset_id": dataset_id,
                "sha256": audio_sha256,
                "annotations": {
                    "transcript": "今日は良い天気です。",
                    "language": "ja",
                },
            }

        @staticmethod
        def verified_audio_path(requested_item_id: str) -> Path:
            assert requested_item_id == item_id
            return audio

    app = create_webui_app(
        static_dir=ROOT / "webui",
        client=_upstream(),
        workstation=store,
        dataset_factory=ReferenceDataset(),
    )
    with TestClient(app) as client:
        rejected_response = client.post(
            f"/api/workstation/training/{project['id']}/reference-selection/lock",
            json={"decision": "all-poor"},
        )
        rejected_artifact_id = rejected_response.json()["artifact"]["id"]
        selected_response = client.post(
            f"/api/workstation/training/{project['id']}/reference-selection/lock",
            json={"decision": "preferred", "item_id": item_id},
        )

    assert rejected_response.status_code == 201
    assert selected_response.status_code == 201, selected_response.text
    payload = selected_response.json()
    assert payload["reference"]["item_id"] == item_id
    assert payload["reference"]["supersedes_reference_evidence"] == [
        rejected_artifact_id
    ]
    assert rejected_artifact_id in payload["artifact"]["parent_artifact_ids"]
    assert store.get_artifact(rejected_artifact_id)["status"] == "rejected"
    updated = store.get_project(project["id"])
    assert updated["config"]["reference_status"] == "human-locked"
    assert updated["config"]["reference_decision_supersedes_artifact_ids"] == [
        rejected_artifact_id
    ]
    assert "reference_rejection_artifact_id" not in updated["config"]
    assert payload["holdout_job"]["type"] == "holdout.evaluate"


def test_all_poor_reference_cannot_be_superseded_after_test_access(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(
        kind="training",
        name="Voice training",
        config={
            "source_dataset_id": "dataset_00000000-0000-4000-8000-000000000001",
            "reference_status": "all-poor",
            "test_split_accessed": True,
        },
    )
    app = create_webui_app(
        static_dir=ROOT / "webui", client=_upstream(), workstation=store
    )
    with TestClient(app) as client:
        response = client.post(
            f"/api/workstation/training/{project['id']}/reference-selection/lock",
            json={
                "decision": "preferred",
                "item_id": "item_00000000-0000-4000-8000-000000000001",
            },
        )

    assert response.status_code == 409
    assert response.json()["error"] == (
        "Reference rejection cannot be revised after test access"
    )


def test_no_reference_preference_locks_automatic_candidate_and_opens_holdout(
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    dataset_id = "dataset_00000000-0000-4000-8000-000000000001"
    item_id = "item_00000000-0000-4000-8000-000000000001"
    project = store.create_project(
        kind="training",
        name="Voice training",
        config={
            "source_dataset_id": dataset_id,
            "reference_status": "pending-human-evidence",
        },
    )
    checkpoint = store.create_job(
        job_type="checkpoint.select", project_id=project["id"]
    )
    checkpoint_claim = store.claim_job(checkpoint["id"])
    store.update_job(
        checkpoint["id"],
        status="succeeded",
        result={"registered_artifact_ids": []},
        claim_token=checkpoint_claim.token,
    )
    output = tmp_path / "reference-output"
    output.mkdir()
    report = output / "reference-selection-report.json"
    report.write_text(
        json.dumps(
            {
                "automatic_recommendation": item_id,
                "policy": "reviewed-quality-speaker-centroid-v2",
                "top_candidates": [{"item_id": item_id}],
            }
        ),
        encoding="utf-8",
    )
    reference = store.create_job(
        job_type="reference.select",
        project_id=project["id"],
        depends_on=(checkpoint["id"],),
    )
    reference_claim = store.claim_job(reference["id"])
    staged = store.register_worker_artifacts(
        job_id=reference["id"],
        claim_token=reference_claim.token,
        job_type="reference.select",
        project_id=project["id"],
        source_root=output,
        artifacts=[
            {
                "kind": "reference",
                "relative_path": report.name,
                "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "size_bytes": report.stat().st_size,
            }
        ],
        image_digest="sha256:" + "a" * 64,
    )
    store.update_job(
        reference["id"],
        status="succeeded",
        result={"registered_artifact_ids": [staged[0]["id"]]},
        claim_token=reference_claim.token,
    )
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"RIFF-reference")
    audio_sha256 = hashlib.sha256(audio.read_bytes()).hexdigest()

    class ReferenceDataset:
        @staticmethod
        def get_item(requested_item_id: str) -> dict[str, object]:
            assert requested_item_id == item_id
            return {
                "id": item_id,
                "dataset_id": dataset_id,
                "sha256": audio_sha256,
                "annotations": {
                    "transcript": "今日は良い天気です。",
                    "language": "ja",
                },
            }

        @staticmethod
        def verified_audio_path(requested_item_id: str) -> Path:
            assert requested_item_id == item_id
            return audio

    app = create_webui_app(
        static_dir=ROOT / "webui",
        client=_upstream(),
        workstation=store,
        dataset_factory=ReferenceDataset(),
    )
    with TestClient(app) as client:
        response = client.post(
            f"/api/workstation/training/{project['id']}/reference-selection/lock",
            json={"decision": "no-preference"},
        )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["reference"]["item_id"] == item_id
    assert payload["reference"]["human_decision"] == "no-preference"
    assert payload["holdout_job"]["type"] == "holdout.evaluate"
    updated = store.get_project(project["id"])
    assert updated["config"]["reference_status"] == "human-locked"
    assert updated["config"]["reference_item_id"] == item_id
    assert updated["config"]["reference_human_decision"] == "no-preference"


def test_evaluation_ui_only_enables_audio_from_ready_worker_artifacts() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "webui" / "studio.js").read_text(encoding="utf-8")

    for element_id in (
        "projectSharedDir",
        "projectAsrModel",
        "projectBaselineReport",
        "evaluationCandidateAudio",
        "evaluationBaselineAudio",
        "evaluationLanguageRows",
        "evaluationRunGateList",
        "evaluationTrainingDependency",
        "evaluationChainButton",
    ):
        assert f'id="{element_id}"' in index
    for value in ("zh", "yue", "ja", "en", "ko"):
        assert f'value="{value}"' in index
    assert 'artifact.status === "ready"' in script
    assert 'artifact.metadata?.source === "linux-docker"' in script
    assert 'artifact.name === `${language}-complete.wav`' in script
    assert "aniflive-tts-workstation-evaluation-v1" in script
    assert "stream_keepalive_audible_ttfa_p50_ms" in script
    assert "wall_rtf_p50" in script
    assert "Not measured by the current evaluation worker" in script
    assert 'createWorkstationJob("evaluation.prepare"' in script
    assert 'createWorkstationJob("engine.prepare"' in script
    assert 'createWorkstationJob("model.package"' in script
    assert "queueProductionBuild" in script
    assert "dependsOn: dependencies" in script
    assert "dependsOn: [engine.id]" in script
    assert "dependsOn: [modelPackage.id]" in script


def test_control_plane_preserves_training_to_release_dependencies(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    app = create_webui_app(
        static_dir=ROOT / "webui", client=_upstream(), workstation=store
    )
    with TestClient(app) as client:
        training = client.post(
            "/api/workstation/projects",
            json={
                "kind": "training",
                "name": "Voice training",
                "config": {
                    "dataset": "C:/datasets/voice",
                    "pretrained_gpt": "C:/weights/s1.ckpt",
                    "pretrained_sovits_g": "C:/weights/s2G.pth",
                    "pretrained_sovits_d": "C:/weights/s2D.pth",
                    "preset": "balanced",
                },
            },
        ).json()
        evaluation = client.post(
            "/api/workstation/projects",
            json={
                "kind": "evaluation",
                "name": "Voice qualification",
                "config": {
                    "model_package": "C:/models/voice-v2proplus",
                    "shared_dir": "C:/shared/aniflive",
                    "asr_model": "C:/models/whisper",
                    "baseline_report": "C:/reports/production.json",
                },
            },
        ).json()
        training_job = client.post(
            "/api/workstation/jobs",
            json={
                "type": "training.prepare",
                "project_id": training["id"],
                "parameters": {},
            },
        ).json()
        engine_job = client.post(
            "/api/workstation/jobs",
            json={
                "type": "engine.prepare",
                "project_id": evaluation["id"],
                "parameters": {},
                "depends_on": [training_job["id"]],
            },
        ).json()
        package_job = client.post(
            "/api/workstation/jobs",
            json={
                "type": "model.package",
                "project_id": evaluation["id"],
                "parameters": {},
                "depends_on": [engine_job["id"]],
            },
        ).json()
        evaluation_job = client.post(
            "/api/workstation/jobs",
            json={
                "type": "evaluation.prepare",
                "project_id": evaluation["id"],
                "parameters": {},
                "depends_on": [package_job["id"]],
            },
        ).json()
        records = client.get("/api/workstation/jobs").json()["data"]

    by_id = {record["id"]: record for record in records}
    assert by_id[engine_job["id"]]["depends_on"] == [training_job["id"]]
    assert by_id[package_job["id"]]["depends_on"] == [engine_job["id"]]
    assert by_id[evaluation_job["id"]]["depends_on"] == [package_job["id"]]
    assert json.loads(json.dumps(evaluation["config"])) == {
        "model_package": "C:/models/voice-v2proplus",
        "shared_dir": "C:/shared/aniflive",
        "asr_model": "C:/models/whisper",
        "baseline_report": "C:/reports/production.json",
    }
