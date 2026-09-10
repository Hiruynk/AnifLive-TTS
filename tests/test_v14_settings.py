from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from aniflive_tts.webui import create_webui_app
from aniflive_tts.workstation import WorkstationError, WorkstationStore


def _offline_upstream() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="http://aniflive.invalid",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, json={"error": "offline"}, request=request)
        ),
    )


def test_workstation_settings_persist_and_merge_defaults(tmp_path: Path) -> None:
    root = tmp_path / "workstation"
    store = WorkstationStore(root)

    defaults = store.get_settings()
    assert defaults["default_language"] == "ja"
    assert defaults["default_continuity_policy"] == "A"
    assert defaults["overview_motion"] is True

    saved = store.update_settings(
        {
            "default_language": "yue",
            "default_continuity_policy": "F",
            "auto_refresh_seconds": 9,
            "overview_motion": False,
        }
    )
    assert saved["default_language"] == "yue"
    assert saved["default_training_preset"] == "balanced"

    reloaded = WorkstationStore(root).get_settings()
    assert reloaded == saved


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"unknown": True}, "Unsupported"),
        ({"default_language": "fr"}, "default_language"),
        ({"default_continuity_policy": "G"}, "default_continuity_policy"),
        ({"default_tse_target_threshold": 1.1}, "target_threshold"),
        ({"auto_refresh_seconds": 1}, "auto_refresh_seconds"),
        ({"overview_motion": "yes"}, "overview_motion"),
    ],
)
def test_workstation_settings_fail_closed(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    with pytest.raises(WorkstationError, match=message):
        store.update_settings(changes)


def test_settings_api_is_persistent_and_json_guarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>", encoding="utf-8")
    monkeypatch.setenv("ANIFLIVE_TTS_WEBUI_TRUSTED_HOSTS", "testserver")
    store = WorkstationStore(tmp_path / "workstation")
    app = create_webui_app(
        static_dir=static,
        client=_offline_upstream(),
        workstation=store,
    )

    with TestClient(app) as client:
        initial = client.get("/api/workstation/settings")
        rejected = client.patch(
            "/api/workstation/settings",
            content='{"default_language":"en"}',
            headers={"Content-Type": "text/plain"},
        )
        saved = client.patch(
            "/api/workstation/settings",
            json={
                "default_language": "en",
                "default_training_preset": "high-quality",
            },
        )
        loaded = client.get("/api/workstation/settings")

    assert initial.status_code == 200
    assert rejected.status_code == 415
    assert saved.status_code == 200
    assert loaded.json()["data"]["default_language"] == "en"
    assert loaded.json()["data"]["default_training_preset"] == "high-quality"


def test_settings_ui_controls_real_defaults_and_speech_session_policy() -> None:
    root = Path(__file__).resolve().parents[1]
    index = (root / "webui" / "index.html").read_text(encoding="utf-8")
    studio = (root / "webui" / "studio.js").read_text(encoding="utf-8")
    synthesis = (root / "webui" / "synthesis.html").read_text(encoding="utf-8")

    assert 'id="workstationSettingsForm"' in index
    assert 'id="settingsContinuityPolicy"' in index
    assert 'api("/api/workstation/settings"' in studio
    assert 'type: "aniflive-tts:settings"' in studio
    assert "state.settings.default_training_preset" in studio
    assert "state.settings.default_benchmark_language" in studio
    assert "continuity_policy: state.continuityPolicy" in synthesis
    assert 'event.data?.type !== "aniflive-tts:settings"' in synthesis


def test_clear_visible_history_hides_records_without_deleting_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>", encoding="utf-8")
    monkeypatch.setenv("ANIFLIVE_TTS_WEBUI_TRUSTED_HOSTS", "testserver")
    store = WorkstationStore(tmp_path / "workstation")
    hidden = store.create_project(kind="dataset", name="Previous test record")
    app = create_webui_app(
        static_dir=static,
        client=_offline_upstream(),
        workstation=store,
    )

    with TestClient(app) as client:
        before = client.get("/api/workstation/projects")
        cleared = client.post("/api/workstation/ui-history/clear", json={})
        after = client.get("/api/workstation/projects")
        settings = client.get("/api/workstation/settings")
        created = client.post(
            "/api/workstation/projects",
            json={"kind": "dataset", "name": "Fresh project", "config": {}},
        )
        visible = client.get("/api/workstation/projects")

    assert [record["id"] for record in before.json()["data"]] == [hidden["id"]]
    assert cleared.status_code == 200
    assert cleared.json()["records_deleted"] == 0
    assert cleared.json()["assets_deleted"] == 0
    assert after.json()["data"] == []
    assert settings.json()["visible_history_cleared_at"] == cleared.json()["cleared_at"]
    assert created.status_code == 201
    assert [record["name"] for record in visible.json()["data"]] == ["Fresh project"]
    assert [record["id"] for record in store.list_projects()] == [
        created.json()["id"],
        hidden["id"],
    ]


def test_studio_uses_one_styled_select_system_on_both_surfaces() -> None:
    root = Path(__file__).resolve().parents[1]
    index = (root / "webui" / "index.html").read_text(encoding="utf-8")
    synthesis = (root / "webui" / "synthesis.html").read_text(encoding="utf-8")
    script = (root / "webui" / "styled_select.js").read_text(encoding="utf-8")
    style = (root / "webui" / "styled_select.css").read_text(encoding="utf-8")

    assert "/assets/styled_select.css" in index
    assert "/assets/styled_select.js" in index
    assert "/assets/styled_select.css" in synthesis
    assert "/assets/styled_select.js" in synthesis
    assert 'setAttribute("role", "listbox")' in script
    assert '(select.closest("dialog") || document.body).append(menu)' in script
    assert 'setAttribute("role", "option")' in script
    assert "MutationObserver" in script
    assert "innerHTML" not in script
    assert "window.setTimeout" in script
    assert "event.stopPropagation()" in script
    assert "window.visualViewport" in script
    assert "dialogRect ? dialogRect.right - viewportGap" in script
    assert "bounds.right - width" in script
    assert 'menu.dataset.placement = openAbove ? "top" : "bottom"' in script
    assert ".styled-select-listbox" in style
    assert ".styled-select-listbox:popover-open" in style
    assert ".styled-select-trigger" in style
