from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient

from aniflive_tts.dataset_factory import DatasetFactory
from aniflive_tts.webui import create_webui_app
from aniflive_tts.workstation import WorkstationStore


def _write_wav(
    path: Path,
    *,
    sample_rate: int = 16_000,
    frequency: float = 220.0,
    duration_seconds: float = 0.3,
) -> None:
    time = np.arange(round(duration_seconds * sample_rate), dtype=np.float32) / sample_rate
    samples = np.rint(
        0.3 * np.sin(2.0 * np.pi * frequency * time) * 32767.0
    ).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples.tobytes())


def _offline_upstream() -> httpx.AsyncClient:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "offline"})

    return httpx.AsyncClient(
        base_url="http://upstream.invalid",
        transport=httpx.MockTransport(handler),
    )


class DatasetFactoryWebUIIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.static = self.root / "static"
        self.static.mkdir()
        (self.static / "index.html").write_text("<!doctype html>", encoding="utf-8")
        self.allowed = self.root / "allowed"
        self.outside = self.root / "outside"
        self.allowed.mkdir()
        self.outside.mkdir()
        self.audio = self.allowed / "voice.wav"
        self.outside_audio = self.outside / "private.wav"
        _write_wav(self.audio)
        _write_wav(self.outside_audio)
        self.previous_hosts = os.environ.get("ANIFLIVE_TTS_WEBUI_TRUSTED_HOSTS")
        self.previous_roots = os.environ.get("ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS")
        os.environ["ANIFLIVE_TTS_WEBUI_TRUSTED_HOSTS"] = "testserver"
        os.environ["ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS"] = str(self.allowed)

    def tearDown(self) -> None:
        if self.previous_hosts is None:
            os.environ.pop("ANIFLIVE_TTS_WEBUI_TRUSTED_HOSTS", None)
        else:
            os.environ["ANIFLIVE_TTS_WEBUI_TRUSTED_HOSTS"] = self.previous_hosts
        if self.previous_roots is None:
            os.environ.pop("ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS", None)
        else:
            os.environ["ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS"] = self.previous_roots
        self.temporary.cleanup()

    def test_app_mounts_dataset_routes_and_enforces_project_and_import_roots(self) -> None:
        store = WorkstationStore(self.root / "workstation")
        project = store.create_project(kind="dataset", name="Voice Dataset")
        upstream = _offline_upstream()
        app = create_webui_app(
            static_dir=self.static,
            client=upstream,
            workstation=store,
        )
        try:
            with TestClient(app) as client:
                response = client.post(
                    f"/api/workstation/datasets/{project['id']}/ingest",
                    json={"sources": [str(self.audio)]},
                )
                outside = client.post(
                    f"/api/workstation/datasets/{project['id']}/ingest",
                    json={"sources": [str(self.outside_audio)]},
                )
                missing_project = client.get(
                    "/api/workstation/datasets/dataset_missing/items"
                )
        finally:
            asyncio.run(upstream.aclose())

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["count"], 1)
        stored_path = response.json()["items"][0]["stored_path"]
        self.assertTrue((store.root / "dataset-factory" / stored_path).is_file())
        self.assertEqual(outside.status_code, 422)
        self.assertIn("outside", outside.json()["detail"])
        self.assertEqual(missing_project.status_code, 404)

    def test_standard_dataset_preparation_expands_sources_and_uses_managed_asr(self) -> None:
        second = self.allowed / "second.wav"
        _write_wav(second)
        component = self.allowed / "sensevoice-small"
        component.mkdir()
        vad_component = self.allowed / "fsmn-vad"
        vad_component.mkdir()
        store = WorkstationStore(self.root / "workstation")
        project = store.create_project(
            kind="dataset",
            name="Managed collection",
            config={
                "acquisition_mode": "standard",
                "sources": [str(self.allowed)],
                "asr_component": "sensevoice-small",
                "declared_language": "ja",
            },
        )
        upstream = _offline_upstream()
        app = create_webui_app(
            static_dir=self.static,
            client=upstream,
            workstation=store,
        )
        DatasetFactory(store.root / "dataset-factory").advance_project_stage(
            project["id"], "review"
        )

        class ReadyAssets:
            def status(self, component_id: str):
                self.assert_component(component_id)
                root = component if component_id == "sensevoice-small" else vad_component
                return {
                    "components": [
                        {"ready": True, "root": str(root), "state": "ready"}
                    ]
                }

            @staticmethod
            def assert_component(component_id: str) -> None:
                if component_id not in {"sensevoice-small", "fsmn-vad"}:
                    raise AssertionError(component_id)

        try:
            with patch(
                "aniflive_tts.workstation_assets.WorkstationAssetManager.status",
                ReadyAssets().status,
            ):
                with TestClient(app) as client:
                    response = client.post(
                        f"/api/workstation/datasets/{project['id']}/prepare-standard",
                        json={},
                    )
        finally:
            asyncio.run(upstream.aclose())

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["count"], 2)
        self.assertEqual(
            DatasetFactory(store.root / "dataset-factory")
            .project_state(project["id"])["lifecycle_stage"],
            "review",
        )
        jobs = store.list_jobs()
        self.assertEqual(len(jobs), 2)
        self.assertTrue(all(job["type"] == "dataset.process" for job in jobs))
        self.assertEqual(
            {job["parameters"]["asr_backend"] for job in jobs},
            {"sensevoice-small"},
        )
        self.assertEqual(
            {job["parameters"]["declared_language"] for job in jobs},
            {"ja"},
        )
        self.assertEqual(
            {Path(job["parameters"]["vad_model"]).name for job in jobs},
            {"fsmn-vad"},
        )
        self.assertEqual(
            {Path(job["parameters"]["source"]).name for job in jobs},
            {"voice.wav", "second.wav"},
        )

    def test_dataset_mutations_require_json_content_type(self) -> None:
        store = WorkstationStore(self.root / "workstation")
        project = store.create_project(kind="dataset", name="Voice Dataset")
        upstream = _offline_upstream()
        app = create_webui_app(
            static_dir=self.static,
            client=upstream,
            workstation=store,
        )
        try:
            with TestClient(app) as client:
                response = client.post(
                    f"/api/workstation/datasets/{project['id']}/ingest",
                    content='{"sources":[]}',
                    headers={"content-type": "text/plain"},
                )
        finally:
            asyncio.run(upstream.aclose())

        self.assertEqual(response.status_code, 415)
        self.assertEqual(
            response.json()["error"],
            "This WebUI mutation requires application/json",
        )

    def test_dataset_process_job_and_capability_are_exposed_by_the_api(self) -> None:
        store = WorkstationStore(self.root / "workstation")
        project = store.create_project(
            kind="dataset",
            name="Voice Dataset",
            config={"source": str(self.audio)},
        )
        upstream = _offline_upstream()
        app = create_webui_app(
            static_dir=self.static,
            client=upstream,
            workstation=store,
        )
        try:
            with TestClient(app) as client:
                capabilities = client.get("/api/workstation/datasets/capabilities")
                queued = client.post(
                    "/api/workstation/jobs",
                    json={
                        "type": "dataset.process",
                        "project_id": project["id"],
                        "parameters": {},
                    },
                )
        finally:
            asyncio.run(upstream.aclose())

        self.assertEqual(capabilities.status_code, 200, capabilities.text)
        process = capabilities.json()["jobs"]["dataset.process"]
        self.assertTrue(process["available"])
        self.assertEqual(process["mode"], "linux-docker")
        self.assertEqual(queued.status_code, 201, queued.text)
        self.assertEqual(queued.json()["type"], "dataset.process")
        self.assertEqual(queued.json()["resource_class"], "gpu-exclusive")

    def test_dataset_policy_asset_is_served_from_fixed_route(self) -> None:
        source = "globalThis.AnifLiveTTSDatasetModel = {};"
        (self.static / "dataset_factory_model.js").write_text(source, encoding="utf-8")
        store = WorkstationStore(self.root / "workstation")
        upstream = _offline_upstream()
        app = create_webui_app(
            static_dir=self.static,
            client=upstream,
            workstation=store,
        )
        try:
            with TestClient(app) as client:
                response = client.get("/assets/dataset_factory_model.js")
        finally:
            asyncio.run(upstream.aclose())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/javascript; charset=utf-8")
        self.assertEqual(response.text, source)

    def test_dataset_qualification_report_is_evidence_only(self) -> None:
        store = WorkstationStore(self.root / "workstation")
        project = store.create_project(
            kind="dataset",
            name="Voice acquisition",
            config={"acquisition_mode": "standard", "sources": [str(self.audio)]},
        )
        upstream = _offline_upstream()
        app = create_webui_app(
            static_dir=self.static,
            client=upstream,
            workstation=store,
        )
        try:
            with TestClient(app) as client:
                response = client.get(
                    f"/api/workstation/datasets/{project['id']}/qualification-report"
                )
        finally:
            asyncio.run(upstream.aclose())

        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertEqual(
            report["schema"], "aniflive-voice-acquisition-qualification-v1"
        )
        self.assertFalse(report["ready_for_release"])
        self.assertIn("dataset-freeze", report["missing_gates"])
        self.assertIn("nine-tensorrt-engines", report["missing_gates"])

    def test_frozen_dataset_handoff_registers_and_injects_training_lineage(self) -> None:
        store = WorkstationStore(self.root / "workstation")
        project = store.create_project(kind="dataset", name="Lineaged Voice")
        factory = DatasetFactory(
            store.root / "dataset-factory",
            allowed_source_roots=(self.allowed, store.artifact_root),
        )
        factory.ensure_project(project["id"], {})
        for index in range(3):
            audio = self.allowed / f"lineage-{index}.wav"
            _write_wav(
                audio,
                frequency=220.0 + index * 20.0,
                duration_seconds=4.0 + index * 0.25,
            )
            source = factory.ingest(project["id"], [audio], recursive=False)[0]
            segment = factory.segment(source["id"])[0]
            factory.update_annotations(
                segment["id"],
                {
                    "transcript": f"今日はテスト{index}です。",
                    "language": "ja",
                    "speaker": "voice",
                },
                source="manual",
                propagate=False,
            )
            factory.review(segment["id"], decision="accepted")
            factory.verify_annotation(segment["id"], kind="transcript")
            factory.verify_annotation(segment["id"], kind="speaker")
        factory.assign_splits(
            project["id"], train=2 / 3, validation=1 / 3, test=0.0, seed="fixed"
        )
        factory.freeze_dataset(project["id"], require_expressions=False)
        training_component = self.allowed / "gpt-sovits-v2proplus-training"
        training_component.mkdir()
        upstream = _offline_upstream()
        app = create_webui_app(
            static_dir=self.static,
            client=upstream,
            workstation=store,
            dataset_factory=factory,
        )
        def component_status(component_id: str) -> dict[str, object]:
            if component_id == "faster-whisper-small":
                return {
                    "components": [
                        {
                            "ready": False,
                            "root": None,
                            "state": "optional",
                            "reason": "component-directory-missing",
                        }
                    ]
                }
            return {
                "components": [
                    {
                        "ready": True,
                        "root": str(training_component),
                        "state": "ready",
                    }
                ]
            }
        try:
            with patch(
                "aniflive_tts.workstation_assets.WorkstationAssetManager.status",
                side_effect=component_status,
            ), TestClient(app) as client:
                handoff = client.post(
                    f"/api/workstation/datasets/{project['id']}/continue-to-training",
                    json={"model_id": "lineaged-voice-v2"},
                )
                self.assertEqual(handoff.status_code, 201, handoff.text)
                training = handoff.json()
                artifact_id = training["config"]["source_dataset_artifact_id"]
                self.assertEqual(
                    training["config"]["reference_status"],
                    "pending-checkpoint-selection",
                )
                self.assertNotIn("reference", training["config"])
                self.assertEqual(
                    training["config"]["reference_selection_policy"],
                    "reviewed-quality-speaker-centroid-v2",
                )
                self.assertEqual(training["config"]["voice_profile"], "default")
                self.assertEqual(training["config"]["model_id"], "lineaged-voice-v2")
                self.assertNotIn("asr_model", training["config"])
                self.assertTrue(training["config"]["auto_build_production"])
                projects = client.get("/api/workstation/projects")
                queued = client.post(
                    "/api/workstation/jobs",
                    json={
                        "type": "training.prepare",
                        "project_id": training["id"],
                        "parameters": {},
                    },
                )
        finally:
            asyncio.run(upstream.aclose())

        artifact = store.get_artifact(artifact_id)
        self.assertEqual(artifact["type"], "dataset")
        self.assertEqual(artifact["status"], "ready")
        self.assertEqual(artifact["project_id"], project["id"])
        self.assertEqual(projects.status_code, 200, projects.text)
        presented_dataset = next(
            item for item in projects.json()["data"] if item["id"] == project["id"]
        )
        self.assertEqual(presented_dataset["status"], "ready")
        self.assertEqual(presented_dataset["progress"], 1.0)
        self.assertEqual(
            presented_dataset["metrics"]["dataset_lifecycle"],
            {"stage": "ready", "frozen": True},
        )
        self.assertEqual(
            artifact["metadata"]["training_list_sha256"],
            training["config"]["training_bundle_sha256"],
        )
        self.assertEqual(queued.status_code, 201, queued.text)
        self.assertEqual(queued.json()["parameters"]["parent_artifact_ids"], [artifact_id])

    def test_reviewed_dataset_candidate_creates_lineaged_expression_draft(self) -> None:
        store = WorkstationStore(self.root / "workstation")
        project = store.create_project(kind="dataset", name="Voice Dataset")
        factory = DatasetFactory(
            store.root / "dataset-factory",
            allowed_source_roots=(self.allowed, store.artifact_root),
        )
        factory.ensure_project(project["id"], {})
        source = factory.ingest(project["id"], [self.audio], recursive=False)[0]
        normalized = factory.resample(source["id"])
        factory.analyze_vad(normalized["id"])
        segment = factory.segment(normalized["id"])[0]
        factory.update_annotations(
            segment["id"],
            {
                "transcript": "Hello there.",
                "language": "en",
                "speaker": "test-speaker",
                "expression": "happy",
            },
            source="manual",
        )
        factory.review(segment["id"], decision="accepted")
        upstream = _offline_upstream()
        app = create_webui_app(
            static_dir=self.static,
            client=upstream,
            workstation=store,
            dataset_factory=factory,
        )
        try:
            with TestClient(app) as client:
                created = client.post(
                    f"/api/workstation/datasets/{project['id']}/expression-candidates/"
                    f"{segment['id']}/draft",
                    json={"intensity": 0.8},
                )
                repeated = client.post(
                    f"/api/workstation/datasets/{project['id']}/expression-candidates/"
                    f"{segment['id']}/draft",
                    json={"intensity": 0.8},
                )
        finally:
            asyncio.run(upstream.aclose())

        self.assertEqual(created.status_code, 201, created.text)
        payload = created.json()
        self.assertEqual(payload["draft"]["profile_id"], "happy")
        self.assertEqual(payload["draft"]["model_id"], project["id"])
        self.assertEqual(payload["draft"]["prosody"]["reference_transcript"], "Hello there.")
        self.assertEqual(payload["artifact"]["metadata"]["dataset_item_id"], segment["id"])
        self.assertEqual(repeated.status_code, 200, repeated.text)
        self.assertTrue(repeated.json()["reused"])


if __name__ == "__main__":
    unittest.main()
