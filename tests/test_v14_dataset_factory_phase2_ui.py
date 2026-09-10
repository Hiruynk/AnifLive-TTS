from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_phase_two_ui_uses_real_routes_and_inspector_controls() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "webui" / "studio.js").read_text(encoding="utf-8")
    style = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")

    for element_id in (
        "datasetImportListButton",
        "datasetItemInspector",
        "datasetWaveform",
        "datasetAudioPreview",
        "datasetAnnotationForm",
        "datasetTranscript",
        "datasetLanguage",
        "datasetSpeaker",
        "datasetExpression",
        "datasetQualityMetrics",
        "datasetItemStages",
    ):
        assert f'id="{element_id}"' in index

    for route in (
        "/import-list",
        "/annotations",
        "/quality",
        "/waveform?bins=640",
        "/audio",
        "/api/workstation/datasets/capabilities",
    ):
        assert route in script

    assert "payload?.peaks" in script
    assert "context.lineTo(x, middle - low * middle * .88)" in script
    assert "trusted Linux Docker decoder" in script
    assert ".dataset-item-inspector-grid" in style
    assert "innerHTML" not in script


def test_dataset_model_reports_annotations_quality_and_backend_capabilities() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    module_path = json.dumps(str(ROOT / "webui" / "dataset_factory_model.js"))
    program = f"""
const assert = require("node:assert/strict");
const model = require({module_path});
const item = {{
  id: "source",
  kind: "source",
  pipeline_state: "ingested",
  review_status: "pending",
  annotations: {{transcript: "Hello", language: "en", speaker: "voice"}},
  quality: {{quality_score: 92.5}}
}};
const capabilities = {{stages: {{
  decode: {{available: false, mode: "blocked", reason: "decoder unavailable"}},
  tse: {{available: true, mode: "delegated", reason: "Use TSE module"}},
  denoise: {{available: false, mode: "blocked", reason: "denoise unavailable"}},
  dereverb: {{available: false, mode: "blocked", reason: "dereverb unavailable"}},
  asr: {{available: false, mode: "blocked", reason: "asr unavailable"}}
}}}};
const summary = model.summarize([item]);
assert.equal(summary.annotated, 1);
assert.equal(summary.qualityAnalyzed, 1);
assert.equal(summary.qualityMean, 92.5);
const stages = model.pipeline([item], capabilities);
assert.equal(stages.find(stage => stage.id === "annotation").state, "complete");
assert.equal(stages.find(stage => stage.id === "quality").state, "complete");
assert.equal(stages.find(stage => stage.id === "decode").state, "complete");
assert.equal(stages.find(stage => stage.id === "tse").state, "delegated");
assert.equal(stages.find(stage => stage.id === "asr").state, "complete");
const blocked = model.pipeline([{{...item, pipeline_state: "decoder-required", annotations: {{}}, quality: null}}], capabilities);
assert.equal(blocked.find(stage => stage.id === "decode").state, "blocked");
assert.match(blocked.find(stage => stage.id === "asr").detail, /unavailable/);
const dockerCapabilities = {{stages: {{...capabilities.stages,
  decode: {{available: true, mode: "delegated-linux-docker", reason: "Process in Docker"}},
  denoise: {{available: true, mode: "delegated-linux-docker-optional", reason: "Optional afftdn"}}
}}}};
const delegated = model.pipeline([{{...item, pipeline_state: "decoder-required", annotations: {{}}, quality: null}}], dockerCapabilities);
assert.equal(delegated.find(stage => stage.id === "decode").state, "delegated");
assert.equal(delegated.find(stage => stage.id === "denoise").state, "delegated");
"""
    subprocess.run([node, "-e", program], check=True, cwd=ROOT)


def test_dataset_frontend_javascript_parses() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    subprocess.run(
        [node, "--check", str(ROOT / "webui" / "dataset_factory_model.js")],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        [node, "--check", str(ROOT / "webui" / "studio.js")],
        check=True,
        cwd=ROOT,
    )
