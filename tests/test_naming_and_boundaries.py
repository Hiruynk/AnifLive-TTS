from __future__ import annotations

import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _project_text_files() -> list[Path]:
    excluded = {".venv", "dist", "__pycache__", ".pytest-tmp"}
    suffixes = {".py", ".md", ".toml", ".yml", ".yaml", ".sh", ".bat", ".ps1"}
    files: list[Path] = []
    for directory, names, filenames in os.walk(ROOT):
        parent = Path(directory)
        names[:] = [
            name for name in names
            if name not in excluded
            and name != ".git"
            and not (parent == ROOT and name == "reports")
        ]
        for filename in filenames:
            path = parent / filename
            if path.suffix in suffixes or path.name in {"Dockerfile", ".env.example"}:
                files.append(path)
    return files


def test_public_identifiers_use_full_aniflive_tts_name() -> None:
    violations: list[str] = []
    patterns = (
        re.compile(r"\b(?:from|import)\s+aniflive\b"),
        re.compile(r"python\s+-m\s+aniflive(?:\s|$)"),
        re.compile(r"\bANIFLIVE_(?!TTS_)"),
        re.compile(r"src[/\\]aniflive(?:[/\\]|$)"),
        re.compile(r"AnifLive(?!-TTS|TTS)"),
    )
    for path in _project_text_files():
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if pattern.search(text):
                violations.append(f"{path.relative_to(ROOT)}: {pattern.pattern}")
    assert not violations, "Forbidden legacy identifiers:\n" + "\n".join(violations)


def test_generic_compose_has_no_miku_or_cloudflare_configuration() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "cloudflared" not in compose.lower()
    assert "tunnel_token" not in compose.lower()
    assert "miku" not in compose.lower()
    assert "tunnel_token" not in env_example.lower()
    assert "miku" not in env_example.lower()

def test_public_source_scan_skips_report_snapshots_only_at_repository_root(
    tmp_path, monkeypatch
) -> None:
    paths = [
        "src/current.py",
        "src/reports/public_module.py",
        "docs/release.md",
        "reports/old-source/src/stale.py",
        ".venv/lib/cached.py",
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    assert {path.relative_to(tmp_path).as_posix() for path in _project_text_files()} == {
        "src/current.py", "src/reports/public_module.py", "docs/release.md"
    }
