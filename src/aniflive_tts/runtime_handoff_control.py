"""Private request drain and recovery control for the GPU coordinator."""
from __future__ import annotations

import hmac
import json
import os
from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool


class RuntimeAdmission:
    def __init__(self, token: str | None, *, restoring: bool = False):
        self.token, self.draining, self.inflight = token, restoring, 0

    @classmethod
    def from_env(cls):
        configured = os.environ.get("ANIFLIVE_TTS_RUNTIME_CONTROL_TOKEN_FILE", "")
        if not configured:
            return cls(None)
        path = Path(configured)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 512:
            raise ValueError("Runtime control token file is invalid")
        token = path.read_text(encoding="ascii").strip()
        if len(token) < 32:
            raise ValueError("Runtime control token must contain at least 32 characters")
        restoring = False
        root = os.environ.get("ANIFLIVE_TTS_WORKSTATION_DIR")
        if root:
            from .workstation import WorkstationStore
            record = WorkstationStore(Path(root)).get_runtime_handoff()
            restoring = bool(record and record.get("phase") != "idle")
        return cls(token, restoring=restoring)


class RuntimeAdmissionMiddleware:
    def __init__(self, app, *, admission):
        self.app, self.admission = app, admission

    async def __call__(self, scope, receive, send):
        controlled = scope["type"] == "http" and scope.get("path") in {
            "/", "/v1/audio/speech", "/v1/sessions", "/v1/models/activate",
            "/set_model", "/change_refer",
        }
        if not controlled or self.admission.token is None:
            await self.app(scope, receive, send)
            return
        if self.admission.draining:
            await JSONResponse(
                {"code": 503, "message": "GPU handoff in progress; retry after runtime recovery"},
                status_code=503, headers={"Retry-After": "5"},
            )(scope, receive, send)
            return
        self.admission.inflight += 1
        try:
            # Include the complete PCM body, not only response headers.
            await self.app(scope, receive, send)
        finally:
            self.admission.inflight -= 1


def install_runtime_control(app, service, sessions, snapshot, *, admission=None):
    admission = admission or RuntimeAdmission.from_env()
    app.state.runtime_admission = admission
    app.add_middleware(RuntimeAdmissionMiddleware, admission=admission)

    @app.post("/internal/workstation/runtime")
    async def control(request: Request):
        if admission.token is None:
            return JSONResponse({"message": "Not found"}, status_code=404)
        if not hmac.compare_digest(
            request.headers.get("authorization", "").encode("utf-8"), ("Bearer " + admission.token).encode("utf-8")
        ):
            return JSONResponse({"message": "Forbidden"}, status_code=403)
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > 4096:
                return JSONResponse({"message": "Request exceeds 4 KiB"}, status_code=413)
            chunks.append(chunk)
        try:
            values = json.loads(b"".join(chunks))
        except (ValueError, UnicodeError):
            return JSONResponse({"message": "Invalid JSON"}, status_code=400)
        if not isinstance(values, dict):
            return JSONResponse({"message": "Expected object"}, status_code=400)
        action = values.get("action")
        if not isinstance(action, str) or action not in {"drain", "cancel-drain", "resume", "status"}:
            return JSONResponse({"message": "Invalid action"}, status_code=400)
        async with app.state.session_model_lock:
            if action == "drain":
                admission.draining = True
            elif action == "cancel-drain":
                admission.draining = False
            elif action == "resume":
                expected = values.get("snapshot")
                if not isinstance(expected, dict) or not isinstance(expected.get("model"), str):
                    return JSONResponse({"message": "Recovery requires model identity"}, status_code=400)
                if admission.inflight or await sessions.has_active():
                    return JSONResponse({"message": "Runtime is busy"}, status_code=409)
                if snapshot().get("model") != expected["model"]:
                    await run_in_threadpool(service.activate, expected["model"])
                if any(snapshot().get(k) != expected.get(k) for k in ("model", "voice", "engine_fingerprint")):
                    return JSONResponse({"message": "Restored identity mismatch"}, status_code=409)
                if not service.health().get("ready"):
                    return JSONResponse({"message": "Runtime is not ready"}, status_code=409)
                admission.draining = False
            active_sessions = await sessions.has_active()
            health = service.health()
            return {
                "draining": admission.draining, "inflight_requests": admission.inflight,
                "active_sessions": active_sessions,
                "ready_to_stop": bool(
                    admission.draining and not admission.inflight and not active_sessions
                    and not health.get("active_requests") and not health.get("switching")
                    and health.get("ready")
                ),
                "snapshot": snapshot(),
            }
    return admission
