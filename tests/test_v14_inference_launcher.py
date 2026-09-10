from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v14_inference_and_host_worker_share_the_workstation_mount() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts" / "run_v14_inference.ps1").read_text(
        encoding="utf-8"
    )

    assert "ANIFLIVE_TTS_WORKSTATION_HOST_DIR=./data/workstation" in example
    assert "ANIFLIVE_TTS_WORKSTATION_DIR: /data/workstation" in compose
    assert (
        "${ANIFLIVE_TTS_WORKSTATION_HOST_DIR:-./data/workstation}:/data/workstation"
        in compose
    )
    assert (
        "ANIFLIVE_TTS_WORKSTATION_HOST_DIR and ANIFLIVE_TTS_WORKSTATION_DIR "
        "must identify the same directory"
    ) in launcher
    assert "$env:ANIFLIVE_TTS_WORKSTATION_HOST_DIR = $composeWorkstation" in launcher
    assert "$env:ANIFLIVE_TTS_WORKSTATION_DIR = $hostWorkstation" in launcher


def test_run_tts_owns_foreground_docker_lifecycle() -> None:
    batch = (ROOT / "run_tts.bat").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts" / "run_v14_inference.ps1").read_text(
        encoding="utf-8"
    )

    assert "local_tts_cf.py" not in batch
    assert "scripts\\run_v14_inference.ps1" in batch
    assert '"compose", "up", "--no-build", "--pull", "never", "aniflive-tts"' in launcher
    assert "[Console]::add_CancelKeyPress" in launcher
    assert "compose stop --timeout 60 aniflive-tts" in launcher
    assert "WaitForExit(65000)" in launcher
