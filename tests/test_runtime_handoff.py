from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aniflive_tts.runtime_handoff_control import RuntimeAdmission, RuntimeAdmissionMiddleware, install_runtime_control
from aniflive_tts.workstation import WorkstationStore, WorkstationError


def _store(tmp_path):
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="training", name="QA handoff")
    job = store.create_job(job_type="engine.prepare", project_id=project["id"])
    token = store.acquire_resource_lease("runtime:handoff", purpose="coordinator")
    return store, job, token


def test_handoff_fences_claims_and_transitions_atomically(tmp_path):
    store, job, token = _store(tmp_path)
    store.set_runtime_handoff(token, {"phase": "draining", "job_id": job["id"]})
    with pytest.raises(WorkstationError, match="busy"):
        store.claim_job(job["id"])
    with pytest.raises(WorkstationError, match="busy"):
        store.acquire_resource_lease("gpu:0", purpose="inference:startup")
    store.set_runtime_handoff(token, {"phase": "ready-for-job", "job_id": job["id"]})
    other = WorkstationStore(store.root)
    with pytest.raises(WorkstationError, match="busy"):
        other.claim_job(job["id"])
    claim = store.claim_job(job["id"])
    assert claim.job["status"] == "running"
    assert store.get_runtime_handoff()["phase"] == "job-running"


def test_orphan_handoff_remains_fenced_and_stale_coordinator_cannot_write(tmp_path):
    store, job, token = _store(tmp_path)
    store.set_runtime_handoff(token, {"phase": "stopping", "job_id": job["id"]})
    store.release_resource_lease(token)
    with pytest.raises(WorkstationError, match="lease"):
        store.set_runtime_handoff(token, {"phase": "idle"})
    reopened = WorkstationStore(store.root)
    with pytest.raises(WorkstationError, match="busy"):
        reopened.claim_job(job["id"])
    fresh = reopened.acquire_resource_lease("runtime:handoff", purpose="recovery")
    reopened.set_runtime_handoff(fresh, {"phase": "restoring", "job_id": job["id"]})
    runtime = reopened.acquire_resource_lease("gpu:0", purpose="inference:startup")
    assert runtime
    with pytest.raises(WorkstationError, match="busy"):
        reopened.claim_job(job["id"])


class Service:
    def health(self):
        return {"ready": True, "active_requests": 0}

    def activate(self, model):
        self.model = model


class Sessions:
    active = True
    async def has_active(self):
        return self.active


def test_private_drain_respects_open_sessions_and_restoration_identity():
    @asynccontextmanager
    async def lifespan(app):
        app.state.session_model_lock = asyncio.Lock()
        yield
    app = FastAPI(lifespan=lifespan)
    sessions = Sessions()
    snapshot = {"model": "voice", "voice": "default", "engine_fingerprint": "abc"}
    admission = RuntimeAdmission("x" * 32)
    install_runtime_control(app, Service(), sessions, lambda: snapshot, admission=admission)
    @app.post("/v1/sessions/s1/flush")
    async def flush():
        sessions.active = False
        return {}
    @app.post("/v1/audio/speech")
    async def speak():
        return {"ok": True}
    auth = {"Authorization": "Bearer " + "x" * 32}
    with TestClient(app) as client:
        assert client.post("/internal/workstation/runtime", json={"action": "drain"}).status_code == 403
        response = client.post("/internal/workstation/runtime", json={"action": "drain"}, headers=auth)
        assert not response.json()["ready_to_stop"]
        assert client.post("/v1/audio/speech").status_code == 503
        assert client.post("/v1/sessions/s1/flush").status_code == 200
        response = client.post("/internal/workstation/runtime", json={"action": "status"}, headers=auth)
        assert response.json()["ready_to_stop"]
        response = client.post("/internal/workstation/runtime", json={
            "action": "resume", "snapshot": {**snapshot, "engine_fingerprint": "wrong"},
        }, headers=auth)
        assert response.status_code == 409
        assert admission.draining
        assert client.post("/internal/workstation/runtime", json={
            "action": "resume", "snapshot": snapshot,
        }, headers=auth).status_code == 200
        assert client.post("/v1/audio/speech").status_code == 200


def test_admission_counts_stream_until_final_pcm_chunk():
    async def scenario():
        started, finish = asyncio.Event(), asyncio.Event()
        admission = RuntimeAdmission("x" * 32)
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            started.set()
            await finish.wait()
            await send({"type": "http.response.body", "body": b"pcm", "more_body": False})
        middleware = RuntimeAdmissionMiddleware(app, admission=admission)
        messages = []
        async def send(value):
            messages.append(value)
        async def receive():
            return {"type": "http.request", "body": b""}
        scope = {"type": "http", "path": "/v1/audio/speech"}
        task = asyncio.create_task(middleware(scope, receive, send))
        await started.wait()
        assert admission.inflight == 1
        admission.draining = True
        await middleware(scope, receive, send)
        assert any(message.get("status") == 503 for message in messages)
        finish.set()
        await task
        assert admission.inflight == 0
    asyncio.run(scenario())


def test_restarted_coordinator_waits_for_existing_owner_lease(tmp_path):
    import json
    from aniflive_tts.workstation_handoff import RuntimeHandoff

    store, job, token = _store(tmp_path)
    store.set_runtime_handoff(token, {"phase": "stopping", "job_id": job["id"]})
    secret = tmp_path / "control.token"
    secret.write_text("x" * 64)
    config = tmp_path / "handoff.json"
    config.write_text(json.dumps({
        "schema": "aniflive-runtime-handoff-config-v1",
        "container": "qa-runtime", "image": "local/runtime@sha256:" + "a" * 64,
        "control_url": "http://127.0.0.1:18880/internal/workstation/runtime",
        "token_file": str(secret),
    }))
    restarted = RuntimeHandoff(WorkstationStore(store.root), config)
    assert restarted.recover() is False
    assert restarted.token is None
    assert store.get_runtime_handoff()["phase"] == "stopping"
