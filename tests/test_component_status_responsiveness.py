from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient

from aniflive_tts.webui import create_webui_app
from aniflive_tts.workstation import WorkstationStore
from aniflive_tts.workstation_assets import WorkstationAssetManager


def test_component_integrity_check_does_not_block_job_status(tmp_path):
    static = tmp_path / "webui"
    static.mkdir()
    (static / "index.html").write_text("<html></html>")
    app = create_webui_app(static_dir=static, workstation=WorkstationStore(tmp_path / "state"))
    started, release = threading.Event(), threading.Event()
    def slow_status():
        started.set()
        assert release.wait(10)
        return {"components": []}
    with TestClient(app, base_url="http://127.0.0.1") as client:
        app.state.workstation_assets = SimpleNamespace(status_snapshot=slow_status)
        with ThreadPoolExecutor(2) as pool:
            component = pool.submit(client.get, "/api/workstation/components")
            try:
                assert started.wait(2)
                jobs = pool.submit(client.get, "/api/workstation/jobs")
                assert jobs.result(timeout=2).status_code == 200
            finally:
                release.set()
            assert component.result(timeout=2).status_code == 200


def test_display_snapshot_is_bounded_and_does_not_replace_fresh_validation(tmp_path):
    manager = WorkstationAssetManager.__new__(WorkstationAssetManager)
    manager.root = tmp_path
    manager._status_snapshot_lock = threading.Lock()
    manager._status_snapshot = None
    manager._status_snapshot_at = 0.0
    manager._status_snapshot_stamp = None
    checks = []
    def fresh():
        checks.append(True)
        return {"components": [{"ready": len(checks) == 1}]}
    manager.status = fresh
    first = manager.status_snapshot()
    first["components"][0]["ready"] = False
    cached = manager.status_snapshot()
    assert cached["components"][0]["ready"] is True
    assert len(checks) == 1
    assert manager.status()["components"][0]["ready"] is False
    manager._status_snapshot_at -= 31
    assert manager.status_snapshot()["components"][0]["ready"] is False
    assert len(checks) == 3
