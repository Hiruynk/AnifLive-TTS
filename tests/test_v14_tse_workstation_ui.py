from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

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


def test_tse_page_exposes_real_docker_controls_and_review_output() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "webui" / "studio.js").read_text(encoding="utf-8")

    for element_id in (
        "tseRunForm",
        "tseSource",
        "tseReference",
        "tseModelPackage",
        "tseSeparationModel",
        "tseTargetThreshold",
        "tseReviewMargin",
        "tseOverlapReview",
        "tseJobProgress",
        "tseJobEvents",
        "tseTargetAudio",
        "tseAudioPlayer",
        "tseAudioToggle",
        "tseAudioSeek",
        "tseAudioMute",
        "tseSegmentRows",
        "tseRerunButton",
    ):
        assert f'id="{element_id}"' in index
    assert "/assets/tse_workstation_model.js?v=1.4.0-editorial20" in index
    assert 'type: "tse.prepare"' in script
    assert "queueTseRun({ reviewed: true })" in script
    assert "review_decisions" in (ROOT / "webui" / "tse_workstation_model.js").read_text(
        encoding="utf-8"
    )
    assert "separation_segments" in (ROOT / "webui" / "tse_workstation_model.js").read_text(
        encoding="utf-8"
    )
    assert ">Separate</th>" in index
    assert "linux docker" in index.lower()
    assert "placeholder success" not in script.lower()
    assert "roxy" not in (ROOT / "webui" / "tse_workstation_model.js").read_text(
        encoding="utf-8"
    ).lower()
    assert "miku" not in (ROOT / "webui" / "tse_workstation_model.js").read_text(
        encoding="utf-8"
    ).lower()
    audio_markup = index[index.index('<audio id="tseTargetAudio"') :]
    audio_markup = audio_markup[: audio_markup.index("</audio>")]
    assert "controls" not in audio_markup
    assert "toggleTseAudio" in script
    assert "renderTseAudioPlayer" in script


def test_tse_model_builds_exact_worker_parameters_and_validates_report() -> None:
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    module_path = json.dumps(str(ROOT / "webui" / "tse_workstation_model.js"))
    program = f"""
const assert = require("node:assert/strict");
const model = require({module_path});
const parameters = model.buildRunParameters({{
  source: " C:/voice/source.wav ",
  reference: "C:/voice/reference.wav",
  model_package: "C:/voice/package",
  separation_model: "C:/models/MossFormer2_SS_16K",
  target_threshold: "0.76",
  review_margin: "0.06",
  frame_ms: "20",
  minimum_speech_ms: "160",
  maximum_gap_ms: "120",
  context_ms: "40",
  extraction_gap_ms: "90",
  separation_ambiguity_margin: "0.03"
}}, new Map([["0", "target"], ["2", "rejected"]]), [
  {{start_seconds: 1.25, end_seconds: 2.5}}
]);
assert.deepEqual(parameters, {{
  source: "C:/voice/source.wav",
  reference: "C:/voice/reference.wav",
  model_package: "C:/voice/package",
  separation_model: "C:/models/MossFormer2_SS_16K",
  target_threshold: 0.76,
  review_margin: 0.06,
  frame_ms: 20,
  minimum_speech_ms: 160,
  maximum_gap_ms: 120,
  context_ms: 40,
  extraction_gap_ms: 90,
  separation_ambiguity_margin: 0.03,
  review_decisions: {{"0": "target", "2": "rejected"}},
  separation_segments: [{{start_seconds: 1.25, end_seconds: 2.5}}]
}});
assert.throws(() => model.buildRunParameters({{...parameters, target_threshold: 1.1}}), /Target threshold/);
assert.throws(() => model.normalizeSeparationSegments([
  {{start_seconds: 1, end_seconds: 2}}, {{start_seconds: 1.5, end_seconds: 3}}
]), /must not overlap/);
assert.throws(() => model.normalizeSeparationSegments([
  {{start_seconds: "1", end_seconds: 2}}
]), /JSON numbers/);
const report = model.validateReport({{
  schema: "aniflive-tts-tse-report-v1",
  sample_rate: 16000,
  source_seconds: 4,
  target_seconds: 1,
  review_seconds: 0.5,
  target_segments: 1,
  review_segments: 1,
  rejected_segments: 0,
  speaker_backend: "TensorRT-11 sv_embedding.engine",
  overlap_policy: "identity-change-review-v1",
  segments: [
    {{index: 0, start_sample: 0, end_sample: 16000, rms_dbfs: -18, similarity: 0.91, overlap: false, decision: "target"}},
    {{index: 1, start_sample: 18000, end_sample: 26000, rms_dbfs: -21, similarity: 0.73, overlap: true, decision: "review"}}
  ]
}});
assert.equal(report.segments[1].overlap, true);
assert.deepEqual([...model.decisionsFromReport(report)], [["0", "target"], ["1", "review"]]);
const separated = model.validateReport({{
  schema: "aniflive-tts-tse-report-v2",
  sample_rate: 16000,
  source_seconds: 4,
  target_seconds: 4,
  review_seconds: 0,
  target_segments: 1,
  review_segments: 0,
  rejected_segments: 0,
  speaker_backend: "TensorRT-11 sv_embedding.engine",
  overlap_policy: "mossformer2-ss-16k-plus-trt-target-selection-v1",
  separation_backend: {{name: "MossFormer2_SS_16K", target_selection: "TensorRT-11 sv_embedding.engine"}},
  separation_attempted_segments: 1,
  separation_accepted_segments: 1,
  separation_requested_ranges: [{{start_sample: 0, end_sample: 64000, start_seconds: 0, end_seconds: 4}}],
  segments: [{{index: 0, start_sample: 0, end_sample: 64000, rms_dbfs: -18, similarity: 0.91,
    overlap: true, decision: "target", separation_triggers: ["operator"], separation: {{accepted: true}}}}]
}});
assert.deepEqual(model.separationSegmentsFromReport(separated), [{{start_seconds: 0, end_seconds: 4}}]);
const artifacts = model.artifactsForJob([
  {{name: "target-speaker.wav", status: "ready", metadata: {{source: "linux-docker", job_id: "job_a", job_type: "tse.prepare"}}}},
  {{name: "wrong.wav", status: "ready", metadata: {{source: "manual", job_id: "job_a", job_type: "tse.prepare"}}}}
], "job_a");
assert.equal(artifacts.length, 1);
"""
    subprocess.run([node, "-e", program], check=True, cwd=ROOT)


def test_tse_control_plane_queues_without_claiming_success(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    app = create_webui_app(
        static_dir=ROOT / "webui", client=_upstream(), workstation=store
    )
    with TestClient(app) as client:
        project = client.post(
            "/api/workstation/projects",
            json={
                "kind": "tse",
                "name": "Target speaker study",
                "config": {
                    "source": "C:/audio/source.wav",
                    "reference": "C:/audio/reference.wav",
                    "model_package": "C:/models/voice-v2proplus",
                    "separation_model": "C:/models/MossFormer2_SS_16K",
                },
            },
        ).json()
        parameters = {
            "source": "C:/audio/source.wav",
            "reference": "C:/audio/reference.wav",
            "model_package": "C:/models/voice-v2proplus",
            "separation_model": "C:/models/MossFormer2_SS_16K",
            "target_threshold": 0.72,
            "review_margin": 0.08,
            "frame_ms": 20,
            "minimum_speech_ms": 160,
            "maximum_gap_ms": 120,
            "context_ms": 40,
            "extraction_gap_ms": 120,
            "separation_ambiguity_margin": 0.03,
            "review_decisions": {"0": "target"},
            "separation_segments": [{"start_seconds": 1.0, "end_seconds": 2.0}],
        }
        response = client.post(
            "/api/workstation/jobs",
            json={
                "type": "tse.prepare",
                "project_id": project["id"],
                "parameters": parameters,
            },
        )
        detail = client.get(f"/api/workstation/jobs/{response.json()['id']}")

    assert response.status_code == 201
    assert response.json()["status"] == "queued"
    assert response.json()["result"] == {}
    assert detail.json()["job"]["parameters"] == parameters
    assert detail.json()["logs"][-1]["message"] == "Job queued"


def test_ready_tse_artifacts_are_served_from_owned_store_only(tmp_path: Path) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(kind="tse", name="Speaker extraction", config={})
    output = store.artifact_root / "tests" / "target-speaker.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    artifact = store.register_artifact(
        artifact_type="dataset",
        name="target-speaker.wav",
        status="ready",
        project_id=project["id"],
        local_path=output,
        metadata={
            "source": "linux-docker",
            "job_id": "job_11111111-1111-4111-8111-111111111111",
            "job_type": "tse.prepare",
        },
    )
    planned = store.register_artifact(
        artifact_type="dataset",
        name="pending.wav",
        project_id=project["id"],
    )
    app = create_webui_app(
        static_dir=ROOT / "webui", client=_upstream(), workstation=store
    )
    with TestClient(app) as client:
        model_asset = client.get("/assets/tse_workstation_model.js")
        response = client.get(
            f"/api/workstation/artifacts/{artifact['id']}/content"
        )
        download = client.get(
            f"/api/workstation/artifacts/{artifact['id']}/content?download=1"
        )
        blocked = client.get(
            f"/api/workstation/artifacts/{planned['id']}/content"
        )

    assert model_asset.status_code == 200
    assert "buildRunParameters" in model_asset.text
    assert response.status_code == 200
    assert response.content == output.read_bytes()
    assert response.headers["content-type"].startswith("audio/wav")
    assert "content-disposition" not in response.headers
    assert "target-speaker.wav" in download.headers["content-disposition"]
    assert blocked.status_code == 409
    assert "Only ready artifacts" in blocked.json()["error"]
