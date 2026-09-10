"""Administrator-owned, fenced Docker inference/worker GPU handoffs."""
from __future__ import annotations

import http.client
import json
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from .workstation import WorkstationError
from .workstation_docker import SubprocessDockerRunner


class RuntimeHandoffError(WorkstationError):
    pass


class RuntimeHandoff:
    def __init__(self, store, config_path: Path, *, runner=None, stopping=lambda: False):
        if config_path.is_symlink() or config_path.stat().st_size > 16384:
            raise RuntimeHandoffError("Runtime handoff config is invalid")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("schema") != "aniflive-runtime-handoff-config-v1":
            raise RuntimeHandoffError("Runtime handoff schema is unsupported")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}", str(config.get("container", ""))):
            raise RuntimeHandoffError("Managed runtime container name is invalid")
        if not re.fullmatch(r"[a-z0-9][a-z0-9._:/-]+@sha256:[a-f0-9]{64}", str(config.get("image", ""))):
            raise RuntimeHandoffError("Managed runtime requires a digest-pinned image")
        url = urlsplit(str(config.get("control_url", "")))
        if (
            url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.path != "/internal/workstation/runtime" or url.username or url.password
            or url.query or url.fragment
        ):
            raise RuntimeHandoffError("Runtime control must use the fixed local endpoint")
        secret = Path(config["token_file"])
        if secret.is_symlink() or not secret.is_file() or secret.stat().st_size > 512:
            raise RuntimeHandoffError("Runtime control token file is invalid")
        self.secret = secret.read_text(encoding="ascii").strip()
        if len(self.secret) < 32:
            raise RuntimeHandoffError("Runtime control token is too short")
        self.store, self.config, self.url = store, config, url
        self.runner = runner or SubprocessDockerRunner()
        self.stopping = stopping
        self.token = None
        self.heartbeat_error = None
        self.stop_heartbeat = threading.Event()
        self.thread = None
        self.cooldown = False

    def _acquire(self):
        if self.token is not None:
            self._ensure()
            return
        self.token = self.store.acquire_resource_lease(
            "runtime:handoff", purpose="managed-inference-handoff", lease_seconds=120,
        )
        self.stop_heartbeat.clear()
        def heartbeat():
            while not self.stop_heartbeat.wait(5):
                try:
                    self.store.heartbeat_resource_lease(self.token, lease_seconds=120)
                except Exception as error:
                    self.heartbeat_error = error
                    return
        self.thread = threading.Thread(target=heartbeat, daemon=True, name="runtime-handoff-lease")
        self.thread.start()

    def _ensure(self):
        if self.heartbeat_error is not None:
            raise RuntimeHandoffError("Runtime handoff coordinator lease is unavailable") from self.heartbeat_error
        if self.token is None:
            raise RuntimeHandoffError("Runtime handoff coordinator lease is unavailable")
        self.store.heartbeat_resource_lease(self.token, lease_seconds=120)

    def close(self):
        self.stop_heartbeat.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.token is not None:
            self.store.release_resource_lease(self.token)
            self.token = None

    def _state(self, **changes):
        self._ensure()
        record = self.store.get_runtime_handoff() or {}
        record.update(changes)
        record.setdefault("schema", "aniflive-runtime-handoff-state-v1")
        return self.store.set_runtime_handoff(self.token, record)

    def _docker(self, *args, timeout=20):
        self._ensure()
        result = self.runner.run(("docker", *args), timeout_seconds=timeout)
        if result.returncode:
            # Do not retain inspect output, environment values or credentials.
            raise RuntimeHandoffError("Managed runtime Docker action failed: " + args[0])
        return result.stdout

    def _inspect(self):
        data = json.loads(self._docker("inspect", self.config["container"]))[0]
        if (
            data["Config"].get("Labels", {}).get("io.aniflive-tts.workstation.runtime") != "v1"
            or data["Config"]["Image"] != self.config["image"]
        ):
            raise RuntimeHandoffError("Runtime container ownership or image mismatch")
        destination = self.config.get("workstation_mount", "/data/workstation")
        mounts = [m for m in data.get("Mounts", []) if m.get("Destination") == destination]
        if (
            len(mounts) != 1 or not mounts[0].get("RW")
            or Path(mounts[0]["Source"]).resolve() != self.store.root.resolve()
            or "ANIFLIVE_TTS_WORKSTATION_DIR=" + destination not in data["Config"].get("Env", [])
        ):
            raise RuntimeHandoffError("Runtime and worker do not share the same workstation")
        return data

    def _control(self, action, snapshot=None):
        self._ensure()
        connection = http.client.HTTPConnection(self.url.hostname, self.url.port or 80, timeout=5)
        try:
            payload = {"action": action}
            if snapshot is not None:
                payload["snapshot"] = snapshot
            connection.request(
                "POST", self.url.path, json.dumps(payload),
                {"Authorization": "Bearer " + self.secret, "Content-Type": "application/json"},
            )
            response = connection.getresponse()
            body = response.read(65537)
            if response.status != 200 or len(body) > 65536:
                raise RuntimeHandoffError("Runtime control rejected " + action)
            value = json.loads(body)
            if not isinstance(value, dict):
                raise RuntimeHandoffError("Runtime control response is malformed")
            return value
        finally:
            connection.close()

    def _log(self, job_id, message, level="info"):
        if job_id:
            self.store.append_job_log(job_id, level, message)

    def prepare(self, job):
        started_handoff = False
        try:
            self._acquire()
            old = self.store.get_runtime_handoff()
            if old and old.get("phase") not in {"idle", "ready-for-job"}:
                if not self.recover():
                    return False
            current = self.store.get_runtime_handoff()
            if current and current.get("phase") == "ready-for-job":
                if current.get("job_id") == job["id"]:
                    return True
                return False
            container = self._inspect()
            running = bool(container["State"]["Running"])
            snapshot = None
            if running:
                status = self._control("status")
                if self.cooldown and (status.get("active_sessions") or status.get("inflight_requests")):
                    self.store.set_job_wait_reason(job["id"], "Waiting for existing speech sessions to finish")
                    return False
                self.cooldown = False
                snapshot = status["snapshot"]
            self._state(
                phase="draining" if running else "ready-for-job",
                job_id=job["id"], container_id=container["Id"],
                was_running=running, snapshot=snapshot, error=None,
            )
            started_handoff = True
            if not running:
                return True
            self._log(job["id"], "GPU handoff: waiting for existing speech to finish")
            self.store.set_job_wait_reason(job["id"], "GPU handoff: draining existing speech sessions")
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                latest = self.store.get_job(job["id"])
                if self.stopping() or latest["status"] != "queued":
                    self._control("cancel-drain")
                    self._state(phase="idle")
                    self.close()
                    return False
                status = self._control("drain")
                if status["ready_to_stop"]:
                    break
                time.sleep(0.25)
            else:
                self._control("cancel-drain")
                self._state(phase="idle")
                self.cooldown = True
                self.store.set_job_wait_reason(job["id"], "Waiting for existing speech sessions to finish")
                self.close()
                return False
            self._state(phase="stopping")
            self._log(job["id"], "GPU handoff: stopping managed inference and releasing CUDA")
            self._docker("stop", "--timeout", "60", container["Id"], timeout=70)
            stopped = self._inspect()
            if stopped["Id"] != container["Id"] or stopped["State"]["Running"]:
                raise RuntimeHandoffError("Managed inference did not stop safely")
            self._state(phase="ready-for-job")
            return True
        except Exception as error:
            if not started_handoff and isinstance(error, (OSError, http.client.HTTPException)):
                self.store.set_job_wait_reason(job["id"], "Waiting for managed inference to become ready")
                self.close()
                return False
            if self.token:
                try:
                    self._state(phase="failed", error=str(error))
                except Exception:
                    pass
            self.store.set_job_wait_reason(job["id"], "GPU handoff blocked: " + str(error))
            return False

    def recover(self):
        record = self.store.get_runtime_handoff()
        if not record or record.get("phase") == "idle":
            return True
        try:
            self._acquire()
        except WorkstationError as error:
            if "busy" in str(error).lower():
                return False
            raise
        job_id = record.get("job_id")
        job = self.store.get_job(job_id) if job_id else None
        if job and job["status"] == "running":
            return False
        if record.get("phase") == "failed":
            return False
        if record.get("phase") == "ready-for-job" and job and job["status"] == "queued":
            # Fence ownership when recovering after the previous coordinator exited.
            self._state(phase="ready-for-job")
            return True
        container = self._inspect()
        if container["Id"] != record.get("container_id"):
            self._state(phase="failed", error="Managed runtime container identity changed")
            return False
        if not record.get("was_running"):
            self._state(phase="idle")
            self.close()
            return True
        self._state(phase="restoring")
        self._log(job_id, "GPU handoff: restoring the original inference model")
        if not container["State"]["Running"]:
            self._docker("start", container["Id"])
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                result = self._control("resume", record["snapshot"])
                if not result.get("draining"):
                    self._state(phase="idle", error=None)
                    self._log(job_id, "GPU handoff: original model restored and ready")
                    self.close()
                    return True
            except (OSError, RuntimeHandoffError, http.client.HTTPException):
                pass
            time.sleep(1)
        self._state(phase="failed", error="Original model recovery timed out")
        self._log(job_id, "GPU handoff failed: original model recovery timed out", "error")
        return False
