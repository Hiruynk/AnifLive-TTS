from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DatasetFactoryUIContractTests(unittest.TestCase):
    def test_dataset_page_is_wired_to_real_phase_one_routes(self) -> None:
        index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "webui" / "studio.js").read_text(encoding="utf-8")
        style = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")

        for element_id in (
            "datasetActiveName",
            "datasetProcessButton",
            "datasetProcessResult",
            "datasetProcessProgressBar",
            "datasetProcessSummary",
            "datasetProcessArtifacts",
            "datasetProcessOpenJob",
            "datasetIngestButton",
            "datasetRefreshButton",
            "datasetSplitButton",
            "datasetManifestButton",
            "datasetQualificationReportButton",
            "datasetAdvancedSpeakerReport",
            "datasetPipeline",
            "datasetStatus",
            "datasetItemRows",
        ):
            self.assertIn(f'id="{element_id}"', index)
        self.assertIn("/assets/dataset_factory_model.js?v=1.4.0-editorial20", index)
        for route_suffix in ("/items", "/ingest", "/resample", "/vad", "/segments", "/review", "/split", "/manifest"):
            self.assertIn(route_suffix, script)
        self.assertNotIn("innerHTML", script)
        self.assertNotIn("TSE / Speaker Filter", index)
        self.assertNotIn("Segment &amp; ASR", index)
        self.assertIn("/prepare-standard", script)
        self.assertIn("datasetProcessJobs", script)
        self.assertIn("latestDatasetProcessJob", script)
        self.assertIn("aniflive-dataset-pipeline-v1", script)
        self.assertIn("/qualification-report", script)
        self.assertIn("voice-acquisition-qualification.json", script)
        self.assertIn("renderDatasetSpeakerEvidence", script)
        self.assertIn("sensitive_diarization", script)
        self.assertIn("dataset-evidence-grid", style)
        self.assertIn("[hidden] { display: none !important; }", style)
        self.assertIn(".project-target-speaker-fields[hidden] { display: none; }", style)
        self.assertIn('i18n.setLocalizedAttribute(\n      byId("projectDatasetSources")', script)

    def test_dataset_stage_and_action_policy(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is unavailable")
        module_path = json.dumps(str(ROOT / "webui" / "dataset_factory_model.js"))
        program = f"""
const assert = require("node:assert/strict");
const model = require({module_path});

const source = {{id: "src", kind: "source", pipeline_state: "ingested", review_status: "pending"}};
let stages = model.pipeline([source]);
assert.equal(stages[0].state, "complete");
assert.equal(stages[1].state, "ready");
assert.deepEqual(model.actions(source), [{{id: "resample", label: "Resample"}}]);

const compressed = {{id: "mp3", kind: "source", pipeline_state: "decoder-required", review_status: "pending"}};
stages = model.pipeline([compressed]);
assert.equal(stages[1].state, "blocked");
assert.deepEqual(model.actions(compressed), []);

const normalized = {{id: "wav", kind: "resampled", pipeline_state: "resampled", review_status: "pending"}};
assert.deepEqual(model.actions(normalized), [{{id: "vad", label: "Run VAD"}}]);
const analyzed = {{...normalized, pipeline_state: "vad-analyzed"}};
assert.deepEqual(model.actions(analyzed), [{{id: "segment", label: "Segment"}}]);

const accepted = {{id: "seg", kind: "segment", pipeline_state: "segmented", review_status: "accepted", split_name: null}};
assert.equal(model.manifestReady([source, normalized, accepted]), false);
assert.equal(model.pipeline([source, normalized, accepted])[5].state, "ready");
accepted.split_name = "train";
assert.equal(model.manifestReady([source, normalized, accepted]), true);
assert.equal(model.pipeline([source, normalized, accepted])[5].state, "complete");

const workerSegment = {{
  id: "worker-segment",
  kind: "segment",
  pipeline_state: "segmented",
  review_status: "pending",
  metadata: {{acquisition: {{asr_suggestion: {{transcript: "今日はいい天気ですね。"}}}}}},
  quality: {{quality_score: 95}}
}};
stages = model.pipeline([workerSegment]);
assert.equal(stages[0].state, "complete");
assert.equal(stages[1].state, "complete");
assert.equal(stages[2].state, "complete");
assert.equal(stages[3].state, "complete");
assert.equal(stages[12].state, "complete");
assert.match(stages[12].detail, /human confirmation required/);
"""
        subprocess.run([node, "-e", program], check=True, cwd=ROOT)


if __name__ == "__main__":
    unittest.main()
