from __future__ import annotations

import subprocess
from pathlib import Path

from aniflive_tts import workstation_supervisor as supervisor_module
from aniflive_tts.workstation_supervisor import (
    WorkstationSupervisorConfig,
    default_broker_config,
    webui_command,
    worker_command,
)


def _config(
    tmp_path: Path, *, broker: Path | None = None
) -> WorkstationSupervisorConfig:
    root = tmp_path / "state"
    imports = tmp_path / "imports"
    root.mkdir()
    imports.mkdir()
    return WorkstationSupervisorConfig(
        host="127.0.0.1",
        port=9890,
        upstream="http://127.0.0.1:9882",
        workstation_dir=root,
        import_roots=(imports,),
        docker_broker_config=broker,
    )


def test_supervisor_commands_are_shell_free_and_share_state(tmp_path: Path) -> None:
    broker = tmp_path / "broker.json"
    broker.write_text("{}", encoding="utf-8")
    config = _config(tmp_path, broker=broker)

    webui = webui_command(config)
    worker = worker_command(config)

    assert webui[1:4] == ("-m", "aniflive_tts", "webui")
    assert webui[-2:] == ("--surface", "studio")
    assert worker is not None
    assert worker[1:4] == ("-m", "aniflive_tts", "worker")
    assert str(config.workstation_dir) in worker
    assert str(config.import_roots[0]) in worker
    import_values = [worker[index + 1] for index, token in enumerate(worker) if token == "--import-root"]
    assert import_values == [str(config.import_roots[0]), str(config.workstation_dir)]
    assert str(broker) in worker


def test_worker_is_optional(tmp_path: Path) -> None:
    assert worker_command(_config(tmp_path)) is None


def test_default_broker_config_uses_environment(monkeypatch, tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    broker = tmp_path / "broker.json"
    broker.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("ANIFLIVE_TTS_DOCKER_BROKER_CONFIG", str(broker))

    assert default_broker_config(state) == broker.resolve()


def test_terminate_allows_cooperative_cleanup_before_forcing(
    monkeypatch,
) -> None:
    events: list[object] = []

    class Child:
        def __init__(self) -> None:
            self.waits = 0

        def poll(self):
            return None

        def wait(self, *, timeout):
            self.waits += 1
            events.append(("wait", timeout))
            if self.waits == 1:
                raise subprocess.TimeoutExpired("child", timeout)
            return 0

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")

    child = Child()
    monkeypatch.setattr(
        supervisor_module,
        "_request_cooperative_stop",
        lambda current: events.append(("cooperative", current is child)),
    )

    supervisor_module._terminate(
        child,
        graceful_timeout=30.0,
        cooperative=True,
    )

    assert events == [
        ("cooperative", True),
        ("wait", 30.0),
        "terminate",
        ("wait", 5.0),
    ]


def test_supervisor_stops_worker_before_webui(
    monkeypatch, tmp_path: Path,
) -> None:
    broker = tmp_path / "broker.json"
    broker.write_text("{}", encoding="utf-8")
    config = _config(tmp_path, broker=broker)
    children = {}
    cleanup: list[tuple[str, float, bool]] = []

    class Child:
        def __init__(self, name: str) -> None:
            self.name = name

        def poll(self):
            return 0 if self.name == "webui" else None

    def fake_spawn(argv, *, environment):
        del environment
        name = "worker" if "worker" in argv else "webui"
        child = Child(name)
        children[name] = child
        return child

    def fake_terminate(child, *, graceful_timeout, cooperative):
        cleanup.append((child.name, graceful_timeout, cooperative))

    monkeypatch.setattr(supervisor_module, "_spawn", fake_spawn)
    monkeypatch.setattr(supervisor_module, "_terminate", fake_terminate)

    assert supervisor_module.run_workstation(config, environment={}) == 0
    assert cleanup == [
        ("worker", 30.0, True),
        ("webui", 8.0, False),
    ]
