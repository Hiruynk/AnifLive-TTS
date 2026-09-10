from fastapi.testclient import TestClient

from aniflive_tts.webui import create_webui_app
from aniflive_tts.workstation import WorkstationStore


def test_missing_optional_animation_has_a_real_image_fallback(tmp_path):
    static = tmp_path / "webui"
    static.mkdir()
    (static / "index.html").write_text("<html></html>")
    app = create_webui_app(static_dir=static, workstation=WorkstationStore(tmp_path / "state"))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/assets/everynight_dance.gif")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/gif"
    assert response.content.startswith(b"GIF89a")
    assert response.content.endswith(b";")


def test_intentional_gpu_handoff_is_distinct_from_connection_failure(tmp_path):
    static = tmp_path / "webui"
    static.mkdir()
    (static / "index.html").write_text("<html></html>")
    store = WorkstationStore(tmp_path / "state")
    token = store.acquire_resource_lease("runtime:handoff", purpose="test")
    store.set_runtime_handoff(token, {"phase": "job-running", "job_id": None})
    app = create_webui_app(static_dir=static, workstation=store)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        state = client.get("/api/workstation/runtime-handoff")
        assert state.json()["phase"] == "job-running"
        response = client.get("/api/status")
        assert response.status_code == 503
        assert "resume automatically" in response.json()["error"]
        assert response.json()["runtime_handoff"]["phase"] == "job-running"
        assert "token" not in response.text and "owner_id" not in response.text


def test_evaluation_queue_preflights_text_before_creating_job(tmp_path, monkeypatch):
    import json

    from aniflive_tts.workstation_evaluation import (
        evaluation_workload,
        resolve_evaluation_plan,
    )
    static = tmp_path / "webui"
    static.mkdir()
    (static / "index.html").write_text("<html></html>")
    monkeypatch.setenv("ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS", str(tmp_path))
    store = WorkstationStore(tmp_path / "state")
    config = {"benchmark_sessions": 10, "benchmark_runs": 100, "benchmark_warmups": 10}
    workload = evaluation_workload(resolve_evaluation_plan(config))
    baseline = {
        "schema": "aniflive-tts-workstation-evaluation-v1",
        "benchmark": {
            "workload": workload,
            "methodology": {
                "sessions": 10, "warmup_requests_per_session": 10,
                "full_wav_requests_per_session": 100, "stream_requests_per_session": 100,
                "keepalive_stream_requests_per_session": 100,
            },
        },
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline))
    project = store.create_project(
        kind="evaluation", name="QA baseline", config={**config, "baseline_report": str(path)},
    )
    app = create_webui_app(static_dir=static, workstation=store)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        preflight = client.post("/api/workstation/evaluation/preflight",
                                json={"project_id": project["id"]})
        assert preflight.status_code == 200
        assert preflight.json()["workload"]["text"] == "今日はいい天気ですね。"
        rejected = client.post("/api/workstation/jobs", json={
            "type": "evaluation.prepare", "project_id": project["id"],
            "parameters": {"benchmark_text": "今日はいい天気なので、音声の品質を確認します。"},
        })
        assert rejected.status_code == 400
        assert "workload" in rejected.json()["error"]
        assert store.list_jobs() == []
        accepted = client.post("/api/workstation/jobs", json={
            "type": "evaluation.prepare", "project_id": project["id"],
        })
        assert accepted.status_code == 201
        assert len(store.list_jobs()) == 1
