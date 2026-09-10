from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import aniflive_tts.workstation_path_picker as picker_module
from aniflive_tts.webui import create_webui_app
from aniflive_tts.workstation import WorkstationStore

ROOT = Path(__file__).resolve().parents[1]


def test_windows_path_picker_uses_sta_and_returns_native_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    selected = [str(tmp_path / "one.wav"), str(tmp_path / "two.wav")]
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(stdout=json.dumps(selected))

    monkeypatch.setattr(picker_module.sys, "platform", "win32")
    monkeypatch.setattr(picker_module.subprocess, "run", fake_run)

    result = picker_module.pick_workstation_paths(
        kind="files",
        title="Choose source media",
        initial_directory=tmp_path,
        file_filter="Audio (*.wav)|*.wav",
    )

    assert result == selected
    assert "-Sta" in observed["command"]
    assert "-NonInteractive" not in observed["command"]
    assert "ShowDialog($owner)" in picker_module._windows_picker_script()
    assert "PSObject.Properties.Name -contains 'UseDescriptionForTitle'" in picker_module._windows_picker_script()
    environment = observed["kwargs"]["env"]
    assert environment["ANIFLIVE_TTS_PICKER_KIND"] == "files"
    assert environment["ANIFLIVE_TTS_PICKER_INITIAL"] == str(tmp_path)


def test_workstation_path_picker_api_is_json_only_and_cancel_safe(
    monkeypatch,
    tmp_path: Path,
) -> None:
    store = WorkstationStore(tmp_path / "workstation")
    monkeypatch.setattr(
        "aniflive_tts.webui.pick_workstation_paths",
        lambda **_kwargs: [],
    )
    app = create_webui_app(
        static_dir=ROOT / "webui",
        workstation=store,
    )

    with TestClient(app, base_url="http://127.0.0.1") as client:
        rejected = client.post(
            "/api/workstation/path-picker",
            content="kind=file",
            headers={"Content-Type": "text/plain"},
        )
        response = client.post(
            "/api/workstation/path-picker",
            json={"kind": "file", "title": "Choose audio", "accept": "audio"},
        )

    assert rejected.status_code == 415
    assert response.status_code == 200
    assert response.json() == {
        "object": "workstation.path-selection",
        "cancelled": True,
        "paths": [],
    }


def test_studio_marks_every_user_path_surface_for_the_shared_picker() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "webui" / "studio.js").read_text(encoding="utf-8")
    style = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")

    required_fields = {
        "expressionReferencePath",
        "tseSource",
        "tseReference",
        "tseModelPackage",
        "tseSeparationModel",
        "componentBundlePath",
        "projectDatasetSources",
        "projectTargetReference",
        "projectSource",
        "projectReference",
        "projectModelPackage",
        "projectPretrainedGpt",
        "projectPretrainedSovitsG",
        "projectPretrainedSovitsD",
        "projectResumeCheckpoint",
        "projectSharedDir",
        "projectAsrModel",
        "projectBaselineReport",
    }
    for field_id in required_fields:
        marker = f'id="{field_id}"'
        start = index.index(marker)
        tag_end = index.index(">", start)
        assert "data-path-picker=" in index[start:tag_end]

    assert 'api("/api/workstation/path-picker"' in script
    assert '(input.closest("dialog") || document.body).append(menu)' in script
    assert 'option.addEventListener("pointerdown"' in script
    assert 'shell.addEventListener("dragenter"' in script
    assert 'shell.addEventListener("drop"' in script
    assert ".studio-path-picker.is-dragging" in style
    assert ".path-picker-menu[hidden]" in style


def test_studio_dense_panels_and_creation_surfaces_have_motion_contracts() -> None:
    script = (ROOT / "webui" / "studio.js").read_text(encoding="utf-8")
    style = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")

    assert "openAnimatedDialog" in script
    assert "closeAnimatedDialog" in script
    assert "animateSurface(form)" in script
    assert ".project-dialog.is-entering" in style
    assert ".project-dialog.is-closing" in style
    assert ".surface-enter" in style
    assert "@media (prefers-reduced-motion: reduce)" in style
    assert ".job-table:has(.empty-cell)" in style
    assert '"contract contract"' in style
