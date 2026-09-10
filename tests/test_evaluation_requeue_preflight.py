import json

import pytest
from fastapi.testclient import TestClient

from aniflive_tts.webui import create_webui_app
from aniflive_tts.workstation import WorkstationStore
from aniflive_tts.workstation_evaluation import evaluation_workload, resolve_evaluation_plan


@pytest.mark.parametrize("action,status", [("retry", "failed"), ("resume", "paused")])
def test_requeue_rejects_stale_evaluation_workload_without_mutation(tmp_path, monkeypatch, action, status):
    static = tmp_path / "webui"
    static.mkdir()
    (static / "index.html").write_text("<html></html>")
    monkeypatch.setenv("ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS", str(tmp_path))
    store = WorkstationStore(tmp_path / "state")
    config = {"benchmark_sessions": 10, "benchmark_runs": 100, "benchmark_warmups": 10}
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({
        "schema": "aniflive-tts-workstation-evaluation-v1",
        "benchmark": {
            "workload": evaluation_workload(resolve_evaluation_plan(config)),
            "methodology": {
                "sessions": 10, "warmup_requests_per_session": 10,
                "full_wav_requests_per_session": 100, "stream_requests_per_session": 100,
                "keepalive_stream_requests_per_session": 100,
            },
        },
    }))
    project = store.create_project(kind="evaluation", name="Historical queued evaluation",
                                   config={**config, "baseline_report": str(path)})
    old_job = store.create_job(job_type="evaluation.prepare", project_id=project["id"],
                              parameters={"benchmark_text": "今日はいい天気なので、音声の品質を確認します。"})
    store.update_job(old_job["id"], status=status)
    before = store.get_job(old_job["id"])
    logs = store.list_job_logs(old_job["id"])
    with TestClient(create_webui_app(static_dir=static, workstation=store),
                    base_url="http://127.0.0.1") as client:
        response = client.post(f'/api/workstation/jobs/{old_job["id"]}/{action}')
    assert response.status_code == 409
    assert "workload" in response.json()["error"]
    assert store.list_jobs() == [before]
    assert store.list_job_logs(old_job["id"]) == logs


def test_scheduler_rejects_invalid_evaluation_before_gpu_handoff(tmp_path, monkeypatch):
    from aniflive_tts.workstation_worker import WorkstationWorker

    monkeypatch.setenv("ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS", str(tmp_path))
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="evaluation", name="Automatic evaluation",
                                   config={"benchmark_runs": 0})
    job = store.create_job(job_type="evaluation.prepare", project_id=project["id"])

    class NoGpuHandoff:
        def recover(self):
            pass

        def prepare(self, job):
            raise AssertionError("Invalid evaluation must not interrupt speech")

    worker = WorkstationWorker(store, allowed_path_roots=[tmp_path])
    worker.runtime_handoff = NoGpuHandoff()
    worker.run_once()
    rejected = store.get_job(job["id"])
    assert rejected["status"] == "failed"
    assert rejected["started_at"] is None
    assert "Evaluation preflight failed" in rejected["error"]
    assert not list((store.root / "worker-manifests").rglob("*.json"))

def test_worker_preflight_uses_cli_import_roots_without_environment(tmp_path, monkeypatch):
    from aniflive_tts.evaluation_preflight import preflight_workstation_evaluation
    from aniflive_tts.workstation import WorkstationError
    monkeypatch.delenv("ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS", raising=False)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    path = allowed / "baseline.json"
    plan = resolve_evaluation_plan({})
    path.write_text(json.dumps({
        "schema": "aniflive-tts-workstation-evaluation-v1",
        "benchmark": {
            "workload": evaluation_workload(plan),
            "methodology": {
                "sessions": 1, "warmup_requests_per_session": 2,
                "full_wav_requests_per_session": 10, "stream_requests_per_session": 10,
                "keepalive_stream_requests_per_session": 10,
            },
        },
    }))
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="evaluation", name="CLI roots",
                                   config={"baseline_report": str(path)})
    result = preflight_workstation_evaluation(store, project["id"], {}, allowed_roots=[allowed])
    assert result["status"] == "passed"
    outside = tmp_path / "outside.json"
    outside.write_text(path.read_text())
    with pytest.raises(WorkstationError, match="outside"):
        preflight_workstation_evaluation(
            store, project["id"], {"baseline_report": str(outside)}, allowed_roots=[allowed],
        )
    with pytest.raises(WorkstationError, match="disabled"):
        preflight_workstation_evaluation(store, project["id"], {})
