from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Mapping


_WORKER_GRACEFUL_SHUTDOWN_SECONDS = 30.0
_WEBUI_GRACEFUL_SHUTDOWN_SECONDS = 8.0
_FORCED_SHUTDOWN_SECONDS = 5.0


class WorkstationSupervisorError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkstationSupervisorConfig:
    host: str
    port: int
    upstream: str
    workstation_dir: Path
    import_roots: tuple[Path, ...]
    docker_broker_config: Path | None = None
    allow_non_loopback: bool = False
    runtime_handoff_config: Path | None = None


def _existing_directory(path: Path, field: str) -> Path:
    candidate = path.expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise WorkstationSupervisorError(f"{field} was not found: {candidate}") from error
    if not resolved.is_dir():
        raise WorkstationSupervisorError(f"{field} must be a directory: {resolved}")
    return resolved


def _existing_file(path: Path, field: str) -> Path:
    candidate = path.expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise WorkstationSupervisorError(f"{field} was not found: {candidate}") from error
    if not resolved.is_file():
        raise WorkstationSupervisorError(f"{field} must be a file: {resolved}")
    return resolved


def default_broker_config(workstation_dir: Path) -> Path | None:
    configured = os.environ.get("ANIFLIVE_TTS_DOCKER_BROKER_CONFIG", "").strip()
    if configured:
        return _existing_file(Path(configured), "Docker broker config")
    candidate = workstation_dir / "docker-broker.json"
    return candidate.resolve() if candidate.is_file() else None


def webui_command(config: WorkstationSupervisorConfig) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        "aniflive_tts",
        "webui",
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--upstream",
        config.upstream,
        "--surface",
        "studio",
    ]
    if config.allow_non_loopback:
        command.append("--allow-non-loopback")
    return tuple(command)


def worker_command(config: WorkstationSupervisorConfig) -> tuple[str, ...] | None:
    if config.docker_broker_config is None:
        return None
    command = [
        sys.executable,
        "-m",
        "aniflive_tts",
        "worker",
        "--workstation-dir",
        str(config.workstation_dir),
        "--docker-broker-config",
        str(config.docker_broker_config),
    ]
    if config.runtime_handoff_config is not None:
        command.extend(("--runtime-handoff-config", str(config.runtime_handoff_config)))
    # Dataset Factory materializes frozen training bundles and other immutable
    # worker inputs inside the workstation root. Treat that managed storage as
    # an allowed read-only input root without exposing it as a user import root.
    worker_roots = tuple(dict.fromkeys((*config.import_roots, config.workstation_dir)))
    for root in worker_roots:
        command.extend(("--import-root", str(root)))
    return tuple(command)


def _spawn(
    argv: tuple[str, ...], *, environment: Mapping[str, str]
) -> subprocess.Popen[bytes]:
    options: dict[str, object] = {
        "env": dict(environment),
        "shell": False,
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(argv, **options)


def _request_cooperative_stop(child: subprocess.Popen[bytes]) -> None:
    if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
        child.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        child.terminate()


def _terminate(
    child: subprocess.Popen[bytes] | None,
    *,
    graceful_timeout: float,
    cooperative: bool,
) -> None:
    if child is None or child.poll() is not None:
        return
    if cooperative:
        try:
            _request_cooperative_stop(child)
        except (OSError, ValueError):
            child.terminate()
    else:
        child.terminate()
    try:
        child.wait(timeout=graceful_timeout)
    except subprocess.TimeoutExpired:
        # On Windows terminate() is already forceful. On POSIX it gives a
        # stubborn child one final SIGTERM window before SIGKILL.
        child.terminate()
        try:
            child.wait(timeout=_FORCED_SHUTDOWN_SECONDS)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=_FORCED_SHUTDOWN_SECONDS)


def run_workstation(
    config: WorkstationSupervisorConfig,
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    workstation_dir = config.workstation_dir.expanduser().resolve()
    workstation_dir.mkdir(parents=True, exist_ok=True)
    roots = tuple(
        _existing_directory(root, "Workstation import root")
        for root in config.import_roots
    )
    broker = (
        _existing_file(config.docker_broker_config, "Docker broker config")
        if config.docker_broker_config is not None
        else None
    )
    resolved = WorkstationSupervisorConfig(
        host=config.host,
        port=config.port,
        upstream=config.upstream,
        workstation_dir=workstation_dir,
        import_roots=roots,
        docker_broker_config=broker,
        runtime_handoff_config=(
            _existing_file(config.runtime_handoff_config, "Runtime handoff config")
            if config.runtime_handoff_config is not None else None
        ),
        allow_non_loopback=config.allow_non_loopback,
    )
    child_environment = dict(os.environ if environment is None else environment)
    child_environment["ANIFLIVE_TTS_WORKSTATION_DIR"] = str(workstation_dir)
    child_environment["ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS"] = os.pathsep.join(
        str(root) for root in roots
    )
    worker: subprocess.Popen[bytes] | None = None
    webui: subprocess.Popen[bytes] | None = None
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers: dict[int, object] = {}
    for candidate in (
        signal.SIGINT,
        getattr(signal, "SIGTERM", signal.SIGINT),
        getattr(signal, "SIGBREAK", signal.SIGINT),
    ):
        if candidate in previous_handlers:
            continue
        previous_handlers[candidate] = signal.getsignal(candidate)
        signal.signal(candidate, request_stop)
    try:
        worker_argv = worker_command(resolved)
        if worker_argv is None:
            print(
                "[AnifLive-TTS Studio] Linux Docker worker is not configured; "
                "GPU workstation jobs will remain blocked.",
                flush=True,
            )
        else:
            print(
                f"[AnifLive-TTS Studio] Linux Docker worker: {resolved.docker_broker_config}",
                flush=True,
            )
            worker = _spawn(worker_argv, environment=child_environment)
        print(
            f"[AnifLive-TTS Studio] Studio: http://{resolved.host}:{resolved.port}/",
            flush=True,
        )
        webui = _spawn(webui_command(resolved), environment=child_environment)
        while not stop_requested:
            webui_exit = webui.poll()
            if webui_exit is not None:
                return int(webui_exit)
            if worker is not None:
                worker_exit = worker.poll()
                if worker_exit is not None:
                    print(
                        f"[AnifLive-TTS Studio] Linux Docker worker stopped with exit code {worker_exit}.",
                        file=sys.stderr,
                        flush=True,
                    )
                    return int(worker_exit or 1)
            time.sleep(0.2)
        return 0
    finally:
        # Stop the worker first and allow its Docker broker enough time to stop,
        # remove and independently verify any active GPU container. The WebUI
        # remains available during that bounded cleanup window.
        _terminate(
            worker,
            graceful_timeout=_WORKER_GRACEFUL_SHUTDOWN_SECONDS,
            cooperative=True,
        )
        _terminate(
            webui,
            graceful_timeout=_WEBUI_GRACEFUL_SHUTDOWN_SECONDS,
            cooperative=False,
        )
        for candidate, handler in previous_handlers.items():
            signal.signal(candidate, handler)


__all__ = [
    "WorkstationSupervisorConfig",
    "WorkstationSupervisorError",
    "default_broker_config",
    "run_workstation",
    "webui_command",
    "worker_command",
]
