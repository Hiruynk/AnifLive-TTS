"""Start a local Docker Studio and its owned inference container."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import subprocess
import sys
import time

OWNER = "io.aniflive-tts.studio.scope"


def docker(*args: str) -> str:
    return subprocess.check_output(["docker", *args], text=True).strip()


def host_path(value: str) -> str:
    value = value.replace("\\", "/").rstrip("/")
    if re.match(r"^[A-Za-z]:/", value):
        return "/run/desktop/mnt/host/" + value[0].lower() + value[2:]
    if not value.startswith("/"):
        raise ValueError("Studio requires an absolute host project path")
    return value


def digest_image(image: str) -> str:
    record = json.loads(docker("image", "inspect", image))[0]
    if record.get("Os") != "linux" or record.get("Architecture") != "amd64":
        raise ValueError("Studio images must be linux/amd64")
    refs = record.get("RepoDigests") or []
    if not refs:
        raise ValueError("Build the image with BuildKit before launching Studio")
    return refs[0]


def owned_container(name: str, scope: str):
    result = subprocess.run(["docker", "inspect", name], capture_output=True, text=True)
    if result.returncode:
        return None
    record = json.loads(result.stdout)[0]
    if record["Config"].get("Labels", {}).get(OWNER) != scope:
        raise ValueError("Container name is already owned by another application: " + name)
    return record


def write_config(path: Path, value: dict) -> None:
    if path.exists():
        previous = json.loads(path.read_text())
        if previous == value:
            return
        if {k: v for k, v in previous.items() if k != "image"} != {
                k: v for k, v in value.items() if k != "image"}:
            raise ValueError("Existing Studio configuration differs; preserve it and choose a separate data directory")
        backup = path.with_name(path.name + ".previous-" + hashlib.sha256(path.read_bytes()).hexdigest()[:12])
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        temporary = path.with_suffix(".new")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def recovery_state(local_data: Path) -> tuple[bool, bool]:
    database = local_data / "workstation/workstation.sqlite3"
    if not database.exists():
        return False, False
    with sqlite3.connect("file:" + str(database) + "?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'runtime_handoff'"
        ).fetchone()
        active = connection.execute("SELECT 1 FROM jobs WHERE status = 'running' LIMIT 1").fetchone()
    record = json.loads(row[0]) if row else {}
    return bool(record and record.get("phase") != "idle"), active is not None


def prepare_startup_marker(local_data: Path) -> Path:
    marker = local_data / "private/startup-ready"
    if marker.is_symlink():
        raise ValueError("Studio startup marker cannot be a symbolic link")
    marker.unlink(missing_ok=True)
    return marker


def model_available(local_data: Path) -> bool:
    from .model_registry import select_startup_package
    try:
        select_startup_package(local_data / "models/active")
        return True
    except (OSError, ValueError):
        return False


def launch(args) -> dict:
    local = args.project_root.resolve()
    host = host_path(args.host_project_root)
    scope = hashlib.sha256((host + "/" + args.data_directory).encode()).hexdigest()[:12]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.data_directory):
        raise ValueError("data-directory must be a simple directory name")
    local_data = local / args.data_directory
    data = host + "/" + args.data_directory
    state = data + "/workstation"
    for relative in ("workstation", "models", "shared", "cache", "imports", "private"):
        (local_data / relative).mkdir(parents=True, exist_ok=True)
    studio_image = digest_image(args.studio_image)
    worker_image = digest_image(args.worker_image)
    runtime_image = digest_image(args.runtime_image)
    studio_name = "aniflive-studio-" + scope
    runtime_name = "aniflive-runtime-" + scope
    previous = owned_container(studio_name, scope)
    runtime = owned_container(runtime_name, scope)
    pending_recovery, active_job = recovery_state(local_data)
    if previous and previous["State"]["Running"]:
        if previous["Config"]["Image"] != studio_image:
            raise ValueError("A different Studio image is running; stop that Studio before updating")
        if (runtime and not runtime["State"]["Running"] and not pending_recovery
                and model_available(local_data)):
            docker("start", runtime_name)
        return {"studio": studio_name, "url": f"http://127.0.0.1:{args.port}/", "reused": True}
    if (previous and runtime and previous["Config"]["Image"] == studio_image
            and runtime["Config"]["Image"] == runtime_image):
        ready_file = prepare_startup_marker(local_data)
        if runtime["State"]["Running"]:
            docker("stop", "--timeout", "60", runtime_name)
        docker("start", studio_name)
        if not pending_recovery and model_available(local_data):
            docker("start", runtime_name)
        ready_file.write_text("ready\n")
        return {"studio": studio_name, "url": f"http://127.0.0.1:{args.port}/",
                "reused": True, "recovery_pending": pending_recovery}
    if pending_recovery or active_job:
        raise ValueError("Resume the existing Studio image to recover unfinished work before updating")
    # Image updates recreate only this workstation's stopped containers.
    if runtime:
        if runtime["State"]["Running"]:
            raise ValueError("Owned inference is still running; stop it before restarting Studio")
        docker("rm", runtime_name)
    if previous:
        docker("rm", studio_name)
    token = local_data / "private/runtime-control.token"
    if not token.exists():
        with token.open("x", encoding="ascii") as stream:
            stream.write(secrets.token_hex(32))
        token.chmod(0o600)
    if token.is_symlink() or len(token.read_text().strip()) < 32:
        raise ValueError("Local runtime control token is invalid")
    write_config(local_data / "workstation/docker-broker.json", {
        "schema": "aniflive-tts-docker-broker-config-v1", "image": worker_image,
        "network": "none", "timeout_seconds": 86400, "poll_seconds": 0.25,
    })
    write_config(local_data / "workstation/runtime-handoff.json", {
        "schema": "aniflive-runtime-handoff-config-v1", "container": runtime_name,
        "image": runtime_image, "control_url": "http://127.0.0.1:9880/internal/workstation/runtime",
        "token_file": data + "/private/runtime-control.token",
    })
    ready_file = prepare_startup_marker(local_data)
    common = ["--label", OWNER + "=" + scope]
    command = [
        "create", "--name", studio_name, *common, "--init",
        "-p", f"127.0.0.1:{args.port}:9891",
        "-p", f"127.0.0.1:{args.api_port}:9880",
        "--mount", f"type=bind,source={data},target={data}",
        "--mount", "type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock",
        "-e", "ANIFLIVE_TTS_WORKSTATION_DIR=" + state,
        "-e", "ANIFLIVE_TTS_MANAGED_RUNTIME_START=1",
        "-e", "ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS=" + ":".join(
            (data + "/imports", data + "/models", data + "/shared", state)),
        "-e", "ANIFLIVE_TTS_BOOTSTRAP_READY_FILE=" + data + "/private/startup-ready",
        "--entrypoint", "python", studio_image, "-c",
        "from aniflive_tts.studio_docker import wait_and_exec; wait_and_exec()",
        "-m", "aniflive_tts", "workstation",
        "--host", "0.0.0.0", "--allow-non-loopback", "--port", "9891",
        "--upstream", "http://127.0.0.1:9880", "--workstation-dir", state,
        "--import-root", data + "/imports",
        "--import-root", data + "/models", "--import-root", data + "/shared",
    ]
    docker(*command)
    # Create inference before starting the worker, including on first use without a model.
    docker(
        "create", "--name", runtime_name, *common,
        "--label", "io.aniflive-tts.workstation.runtime=v1",
        "--network", "container:" + studio_name, "--gpus", "all", "--shm-size", "2g",
        "--mount", f"type=bind,source={state},target=/data/workstation",
        "--mount", f"type=bind,source={data}/models,target=/data/models,readonly",
        "--mount", f"type=bind,source={data}/shared,target=/data/shared,readonly",
        "--mount", f"type=bind,source={data}/cache,target=/data/cache",
        "--mount", f"type=bind,source={data}/private/runtime-control.token,target=/runtime-control.token,readonly",
        "-e", "ANIFLIVE_TTS_WORKSTATION_DIR=/data/workstation",
        "-e", "ANIFLIVE_TTS_MODEL_PACKAGE=/data/models/active",
        "-e", "ANIFLIVE_TTS_RUNTIME_CONTROL_TOKEN_FILE=/runtime-control.token",
        runtime_image,
    )
    docker("start", studio_name)
    ready = model_available(local_data)
    if ready:
        docker("start", runtime_name)
    ready_file.write_text("ready\n")
    return {
        "studio": studio_name, "runtime": runtime_name,
        "url": f"http://127.0.0.1:{args.port}/", "inference_started": ready,
        "first_use": None if ready else "Add or train a model in Studio. Rerun this launcher if the speech service remains offline.",
    }


def wait_and_exec():
    path = Path(os.environ["ANIFLIVE_TTS_BOOTSTRAP_READY_FILE"])
    deadline = time.monotonic() + 180
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError("Studio bootstrap did not finish; rerun the launcher")
        time.sleep(0.1)
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--host-project-root", required=True)
    parser.add_argument("--data-directory", default="data")
    parser.add_argument("--studio-image", default="aniflive-tts-studio:1.4.0-cu128")
    parser.add_argument("--worker-image", default="aniflive-tts-workstation-worker:1.4.0-cu128")
    parser.add_argument("--runtime-image", default="aniflive-tts:1.4.0-cu128")
    parser.add_argument("--port", type=int, default=9891)
    parser.add_argument("--api-port", type=int, default=9882)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.port <= 65535 and 1 <= args.api_port <= 65535) or args.port == args.api_port:
        parser.error("Choose two distinct valid ports")
    result = launch(args)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("[AnifLive-TTS Studio] " + result["url"])
        if result.get("first_use"):
            print("[AnifLive-TTS Studio] " + result["first_use"])


if __name__ == "__main__":
    main()
