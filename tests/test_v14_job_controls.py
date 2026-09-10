from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_jobs_page_exposes_dependency_aware_controls() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "webui" / "studio.js").read_text(encoding="utf-8")
    controls = (ROOT / "webui" / "job_controls.js").read_text(encoding="utf-8")

    for element_id in (
        "queueJobButton",
        "jobType",
        "jobProject",
        "jobPriority",
        "jobDependencyList",
        "jobInspectorMeta",
    ):
        assert f'id="{element_id}"' in index
    assert "/assets/job_controls.js?v=1.4.0-editorial20" in index
    assert 'controlJob(job.id, "pause")' in script
    assert 'controlJob(job.id, "resume")' in script
    assert 'controlJob(job.id, "retry")' in script
    assert "depends_on: dependencies" in script
    assert "priority," in script
    assert "Pause requested; waiting for the worker to stop safely" in script
    assert "Cancellation requested; waiting for the worker to stop safely" in script
    assert "innerHTML" not in controls


def test_job_control_policy_is_state_accurate() -> None:
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    module_path = json.dumps(str(ROOT / "webui" / "job_controls.js"))
    program = f"""
const assert = require("node:assert/strict");
const controls = require({module_path});
const dependency = {{id: "job_dependency", status: "queued"}};
const base = {{
  id: "job_current",
  type: "dataset.inventory",
  status: "running",
  depends_on: [],
  pause_requested: false,
  cancel_requested: false
}};

let policy = controls.actionPolicy(base, [base], false);
assert.equal(policy.pause.visible, true);
assert.equal(policy.pause.disabled, false);
assert.equal(policy.resume.visible, false);

const pausing = {{...base, pause_requested: true}};
policy = controls.actionPolicy(pausing, [pausing], false);
assert.deepEqual(controls.statusPresentation(pausing), {{
  value: "pause-requested",
  label: "Pause requested"
}});
assert.equal(policy.pause.disabled, true);
assert.equal(policy.resume.visible, false);
assert.equal(policy.cancel.disabled, false);

const paused = {{...base, status: "paused", pause_requested: false}};
policy = controls.actionPolicy(paused, [paused], false);
assert.equal(policy.pause.visible, false);
assert.equal(policy.resume.visible, true);
assert.equal(policy.cancel.visible, true);

const waiting = {{...base, status: "queued", depends_on: [dependency.id]}};
policy = controls.actionPolicy(waiting, [waiting, dependency], false);
assert.equal(policy.run.visible, true);
assert.equal(policy.run.disabled, true);
const gpuWaiting = {{...waiting, wait_reason: "Waiting for gpu:0"}};
assert.deepEqual(controls.statusPresentation(gpuWaiting), {{
  value: "waiting-resource",
  label: "Waiting for GPU"
}});
dependency.status = "succeeded";
policy = controls.actionPolicy(waiting, [waiting, dependency], false);
assert.equal(policy.run.disabled, false);

const failed = {{...base, status: "failed"}};
policy = controls.actionPolicy(failed, [failed], false);
assert.equal(policy.retry.visible, true);
assert.equal(policy.retry.disabled, false);
const retry = {{...base, id: "job_retry", status: "queued", retry_of: failed.id}};
policy = controls.actionPolicy(failed, [failed, retry], false);
assert.equal(policy.retry.disabled, true);

const cancelling = {{...base, cancel_requested: true}};
assert.deepEqual(controls.statusPresentation(cancelling), {{
  value: "cancellation-requested",
  label: "Cancellation requested"
}});
policy = controls.actionPolicy(cancelling, [cancelling], false);
assert.equal(policy.pause.disabled, true);
assert.equal(policy.cancel.disabled, true);
"""
    subprocess.run([node, "-e", program], check=True, cwd=ROOT)
