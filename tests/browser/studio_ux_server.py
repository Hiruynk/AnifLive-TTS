from pathlib import Path
import math
import os
import struct
import httpx
import uvicorn
from aniflive_tts.webui import create_webui_app
from aniflive_tts.workstation import WorkstationStore

root = Path("/tmp/studio-ux-workstation")
store = WorkstationStore(root)
for number in range(6):
    project = store.create_project(
        kind="training" if number % 2 == 0 else "evaluation",
        name=f"UI QA {number + 1} — Long multilingual project name 長名稱測試",
        config={},
    )
    job = store.create_job(
        job_type="training.prepare" if number % 2 == 0 else "evaluation.prepare",
        project_id=project["id"],
    )
    if number < 3:
        claim = store.claim_job(job["id"])
        store.update_job(
            job["id"],
            claim_token=claim.token,
            status="failed",
            error="Test fixture: input is unavailable. Select a configured project.",
        )


def upstream(request):
    path = request.url.path
    payloads = {
        "/openapi.json": {"info": {"version": "1.4.0-dev"}},
        "/health": {
            "ready": True,
            "backend": "TensorRT-11",
            "engine_count": 9,
            "model": "ui-qa-voice",
            "gpu": {"name": "UI QA · synthetic runtime"},
        },
        "/model/config": {
            "model": "ui-qa-voice",
            "version": "v2ProPlus",
            "backend": "TensorRT-11",
            "engine_count": 9,
            "sample_rate": 32000,
            "pytorch_fallback": False,
        },
        "/v1/models": {
            "object": "list",
            "data": [
                {"id": "ui-qa-voice", "active": True},
                {"id": "ui-qa-alternate", "active": False},
            ],
        },
        "/v1/expressions": ({
            "object": "list", "enabled": True, "default": "neutral",
            "profiles": [
                {"id": "shy", "intensity_levels": [0.5, 0.8], "languages": ["ja", "zh", "en"]},
                {"id": "battle", "intensity_levels": [0.5, 0.8], "languages": ["ja", "zh", "en"]},
            ],
            "policies": ["full-switch", "identity-lock", "semantic-style"],
        } if os.environ.get("STUDIO_QA_EXPRESSIONS") == "1" else {
            "object": "list", "enabled": False, "profiles": [], "policies": [],
        }),
    }
    if path in payloads:
        return httpx.Response(200, json=payloads[path])
    if path.endswith("/audio"):
        pcm = b"".join(
            struct.pack("<h", int(2500 * math.sin(2 * math.pi * 330 * i / 32000)))
            for i in range(64000)
        )
        return httpx.Response(
            200,
            content=pcm,
            headers={
                "Content-Type": "application/octet-stream",
                "x-tensorrt-backend": "TensorRT-11",
                "x-tensorrt-engine-count": "9",
                "x-pytorch-fallback": "false",
                "x-tts-model": "ui-qa-voice",
                "x-tts-stream": "pcm_s16le",
                "x-tts-sample-rate": "32000",
                "x-tts-channels": "1",
                "x-tts-session-id": "ui-qa-session",
                "x-tts-session-context": "committed-neural-v1",
                "x-tts-acoustic-latent-continuity": "false",
                "x-tts-continuity-qualification": "experimental-unqualified",
                "x-tts-session-context-policy": "A",
                "x-tts-neural-state-continuity": "false",
            },
        )
    if path.endswith("/sessions"):
        return httpx.Response(
            200, json={"id": "ui-qa-session", "session_id": "ui-qa-session", "sample_rate": 32000}
        )
    if "/sessions/" in path:
        return httpx.Response(200, json={"id": "ui-qa-session", "status": "accepted"})
    return httpx.Response(200, json={})


client = httpx.AsyncClient(transport=httpx.MockTransport(upstream), base_url="http://upstream.test")
app = create_webui_app(
    upstream="http://upstream.test",
    static_dir=Path("/repo/webui"),
    client=client,
    workstation=store,
)


@app.middleware("http")
async def fixture_marker(request, call_next):
    response = await call_next(request)
    response.headers["X-Aniflive-UI-Fixture"] = "synthetic"
    return response


uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
