from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_worker_config_script_only_pins_an_existing_local_image() -> None:
    script = (ROOT / "scripts/configure_workstation_worker.ps1").read_text(
        encoding="utf-8"
    )
    lowered = script.casefold()
    assert "docker.exe image inspect" in lowered
    assert "docker.exe build" not in lowered
    assert "docker.exe pull" not in lowered
    assert 'network = "none"' in lowered
    assert "aniflive-tts-docker-broker-config-v1" in lowered
    assert "utf8encoding" in lowered
    assert "isnullorwhitespace($workstationdir)" in lowered
    assert 'join-path $psscriptroot "..\\data\\workstation"' in lowered


def test_webui_launcher_does_not_build_or_pull_worker_images() -> None:
    launcher = (ROOT / "run_studio.bat").read_text(encoding="utf-8").casefold()
    assert "docker build" not in launcher
    assert "docker pull" not in launcher
