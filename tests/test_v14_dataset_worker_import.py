from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

import httpx
from fastapi.testclient import TestClient

from aniflive_tts.dataset_factory import DatasetFactory
from aniflive_tts.webui import create_webui_app
from aniflive_tts.workstation import WorkstationStore


def _upstream() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "ready": True,
                    "backend": "TensorRT-11",
                    "engine_count": 9,
                    "model": "test-v2proplus",
                    "gpu": {},
                },
            )
        if request.url.path == "/v1/config":
            return httpx.Response(200, json={"sample_rate": 32000})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": []})
        if request.url.path == "/v1/expressions":
            return httpx.Response(200, json={"profiles": [], "policies": []})
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = (b"\x00\x00\x10\x00\xf0\xff\x00\x00") * 4000
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(32000)
        stream.writeframes(samples)


def test_verified_dataset_worker_segments_import_into_review_catalog(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ANIFLIVE_TTS_WEBUI_TRUSTED_HOSTS", "testserver")
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(
        kind="dataset",
        name="Processed dataset",
        config={"source": str(tmp_path / "source.mp4")},
    )
    job = store.create_job(job_type="dataset.process", project_id=project["id"])
    claim = store.claim_job(job["id"])
    output_root = tmp_path / "docker-output"
    artifact_path = output_root / "dataset-pipeline" / "segments" / "segment-0001.wav"
    _wav(artifact_path)
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    artifacts = store.register_worker_artifacts(
        job_id=job["id"],
        claim_token=claim.token,
        job_type="dataset.process",
        project_id=project["id"],
        source_root=output_root,
        artifacts=(
            {
                "kind": "dataset",
                "relative_path": "dataset-pipeline/segments/segment-0001.wav",
                "sha256": digest,
                "size_bytes": artifact_path.stat().st_size,
            },
        ),
        image_digest="sha256:" + "a" * 64,
    )
    artifact = artifacts[0]
    store.update_job(
        job["id"],
        status="succeeded",
        progress=1.0,
        result={"registered_artifact_ids": [artifact["id"]]},
        claim_token=claim.token,
    )
    factory = DatasetFactory(
        tmp_path / "dataset-factory",
        allowed_source_roots=(store.artifact_root,),
    )
    app = create_webui_app(
        client=_upstream(), workstation=store, dataset_factory=factory,
        static_dir=Path(__file__).resolve().parents[1] / "webui",
    )

    with TestClient(app) as client:
        first = client.post(
            f"/api/workstation/datasets/{project['id']}/import-worker-job/{job['id']}",
            json={},
        )
        second = client.post(
            f"/api/workstation/datasets/{project['id']}/import-worker-job/{job['id']}",
            json={},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["artifact_ids"] == [artifact["id"]]
    assert first.json()["count"] == 1
    assert second.json()["count"] == 1
    items = factory.list_items(project["id"])
    assert len(items) == 1
    assert items[0]["kind"] == "source"
    assert items[0]["pipeline_state"] == "ingested"


def test_reported_standard_segments_import_with_stable_lineage_and_asr_suggestion(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ANIFLIVE_TTS_WEBUI_TRUSTED_HOSTS", "testserver")
    store = WorkstationStore(tmp_path / "workstation")
    source = tmp_path / "source.wav"
    _wav(source)
    project = store.create_project(
        kind="dataset",
        name="Managed standard dataset",
        config={"acquisition_mode": "standard", "sources": [str(source)]},
    )
    job = store.create_job(
        job_type="dataset.process",
        project_id=project["id"],
        parameters={"source": str(source)},
    )
    claim = store.claim_job(job["id"])
    output_root = tmp_path / "docker-output"
    artifact_path = output_root / "dataset-pipeline" / "segments" / "segment-00000.wav"
    _wav(artifact_path)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    segment_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    report_path = output_root / "dataset-pipeline" / "dataset-report.json"
    report = {
        "schema": "aniflive-dataset-pipeline-v1",
        "input": {"name": source.name, "sha256": source_sha},
        "output": {
            "segments": [
                {
                    "position": 0,
                    "path": "segments/segment-00000.wav",
                    "sha256": segment_sha,
                    "region": {"start_frame": 3200, "end_frame": 9600},
                    "transcript": {
                        "text": "今日真係好開心。",
                        "language": "yue",
                        "emotion_suggestion": "happy",
                    },
                }
            ]
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    artifacts = store.register_worker_artifacts(
        job_id=job["id"],
        claim_token=claim.token,
        job_type="dataset.process",
        project_id=project["id"],
        source_root=output_root,
        artifacts=(
            {
                "kind": "dataset",
                "relative_path": "dataset-pipeline/segments/segment-00000.wav",
                "sha256": segment_sha,
                "size_bytes": artifact_path.stat().st_size,
            },
            {
                "kind": "dataset",
                "relative_path": "dataset-pipeline/dataset-report.json",
                "sha256": report_sha,
                "size_bytes": report_path.stat().st_size,
            },
        ),
        image_digest="sha256:" + "a" * 64,
    )
    store.update_job(
        job["id"],
        status="succeeded",
        progress=1.0,
        result={"registered_artifact_ids": [value["id"] for value in artifacts]},
        claim_token=claim.token,
    )
    factory = DatasetFactory(
        tmp_path / "dataset-factory",
        allowed_source_roots=(store.artifact_root,),
    )
    app = create_webui_app(
        client=_upstream(), workstation=store, dataset_factory=factory,
        static_dir=Path(__file__).resolve().parents[1] / "webui",
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/workstation/datasets/{project['id']}/import-worker-job/{job['id']}",
            json={},
        )

    assert response.status_code == 200, response.text
    assert response.json()["import"]["created_count"] == 1
    item = factory.list_items(project["id"])[0]
    assert item["kind"] == "segment"
    assert item["original_name"].startswith("seg_")
    assert item["review_status"] == "pending"
    assert item["annotations"]["transcript"] is None
    suggestion = item["metadata"]["acquisition"]["asr_suggestion"]
    assert suggestion["transcript"] == "今日真係好開心。"
    assert suggestion["expression_suggestion"] == "happy"
    assert item["metadata"]["lineage"]["source_start_sample"] == 3200
    assert item["metadata"]["lineage"]["source_end_sample"] == 9600
    assert factory.project_state(project["id"])["lifecycle_stage"] == "review"


def test_dataset_worker_import_controls_are_wired() -> None:
    root = Path(__file__).resolve().parents[1]
    index = (root / "webui" / "index.html").read_text(encoding="utf-8")
    script = (root / "webui" / "studio.js").read_text(encoding="utf-8")
    assert 'id="datasetProcessImportButton"' in index
    assert "/import-worker-job/" in script
    assert "importDatasetProcessArtifacts" in script
