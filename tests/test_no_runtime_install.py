from pathlib import Path


def test_runtime_files_do_not_install_or_download() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime_files = [
        root / "src/aniflive_tts/api.py",
        root / "src/aniflive_tts/service.py",
        root / "src/aniflive_tts/streaming.py",
        root / "scripts/entrypoint.sh",
    ]
    forbidden = ("pip install", "snapshot_download", "hf_hub_download", "build_engines(")
    for path in runtime_files:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path

