from aniflive_tts.studio_docker import host_path
import pytest


def test_docker_desktop_host_paths_match_child_bind_mounts():
    assert host_path("D:/Voice Project") == "/run/desktop/mnt/host/d/Voice Project"
    assert host_path("/srv/voice") == "/srv/voice"


def test_relative_host_paths_are_rejected():
    with pytest.raises(ValueError, match="absolute"):
        host_path("relative/project")


@pytest.mark.parametrize("pending", [False, True])
def test_restart_preserves_container_identity_and_pending_handoff(tmp_path, monkeypatch, pending):
    import json
    import sqlite3
    from types import SimpleNamespace
    from aniflive_tts import studio_docker
    from aniflive_tts.workstation import WorkstationStore
    data = tmp_path / "data"
    store = WorkstationStore(data / "workstation")
    if pending:
        with sqlite3.connect(store.database_path) as connection:
            connection.execute("INSERT INTO metadata(key,value) VALUES (?,?)",
                               ("runtime_handoff", json.dumps({"phase": "ready-for-job"})))
    (data / "models/active").mkdir(parents=True)
    (data / "models/active/manifest.json").write_text("{}")
    monkeypatch.setattr(studio_docker, "digest_image", lambda value: value)
    def owned(name, scope):
        return {"Config": {"Image": "studio" if "studio" in name else "runtime"},
                "State": {"Running": False}, "Id": name + "-preserved"}
    monkeypatch.setattr(studio_docker, "owned_container", owned)
    calls = []
    monkeypatch.setattr(studio_docker, "docker", lambda *args: calls.append(args))
    result = studio_docker.launch(SimpleNamespace(
        project_root=tmp_path, host_project_root="/host/project",
        data_directory="data", studio_image="studio", worker_image="worker",
        runtime_image="runtime", port=9891, api_port=9882,
    ))
    assert result["recovery_pending"] is pending
    assert all(call[0] != "rm" for call in calls)
    assert sum(call[0] == "start" for call in calls) == (1 if pending else 2)
