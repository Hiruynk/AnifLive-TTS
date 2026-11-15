from __future__ import annotations

import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_IDENTIFIER_PATTERNS = (
    re.compile(r"\b(?:from|import)\s+aniflive\b"),
    re.compile(r"python\s+-m\s+aniflive(?:\s|$)"),
    re.compile(r"\bANIFLIVE_(?!TTS_)"),
    re.compile(r"src[/\\]aniflive(?:[/\\]|$)"),
    re.compile(r"AnifLive(?!-TTS|TTS)"),
)
# Permit only these exact removals of obsolete runtime flags; writes and reads stay scanned.
DOCUMENTED_LEGACY_CLEANUP_LINES = {
    "src/aniflive_tts/backend/macos_worker.py": frozenset(
        {'environment.pop("ANIFLIVE_COMPONENT_CPU_REFERENCE", None)'}
    ),
    "scripts/macos-release/start.sh": frozenset(
        {
            "unset PYTHONPATH ANIFLIVE_COMPONENT_CPU_REFERENCE ANIFLIVE_TTS_RESEARCH_AR_TOP_K",
            "unset ANIFLIVE_HTTP_AUDIT_DIR ANIFLIVE_HTTP_AUDIT_TRACE ANIFLIVE_TTS_AUDIT_TRACE",
        }
    ),
}


def _is_documented_legacy_cleanup(relative_path: str, line: str) -> bool:
    return line.strip() in DOCUMENTED_LEGACY_CLEANUP_LINES.get(relative_path, ())


def _project_text_files() -> list[Path]:
    excluded = {".venv", "dist", "__pycache__", ".pytest-tmp", ".local-sdk"}
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
            if filename.lower() == "agents.md" or filename.lower().endswith((
                "_codex_plan.md", "_implementation_plan.md", "_master_start_prompt.md",
            )):
                continue
            path = parent / filename
            if path.suffix in suffixes or path.name in {"Dockerfile", ".env.example"}:
                files.append(path)
    return files


def test_public_identifiers_use_full_aniflive_tts_name() -> None:
    violations: list[str] = []
    for path in _project_text_files():
        if path == Path(__file__).resolve():
            continue
        relative_path = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if _is_documented_legacy_cleanup(relative_path, line):
                continue
            for pattern in LEGACY_IDENTIFIER_PATTERNS:
                if pattern.search(line):
                    violations.append(f"{relative_path}: {pattern.pattern}")
    assert not violations, "Forbidden legacy identifiers:\n" + "\n".join(violations)


def test_legacy_cleanup_exception_does_not_allow_setting_removed_flag() -> None:
    relative_path = "src/aniflive_tts/backend/macos_worker.py"
    cleanup = 'environment.pop("ANIFLIVE_COMPONENT_CPU_REFERENCE", None)'
    setter = 'environment["ANIFLIVE_COMPONENT_CPU_REFERENCE"] = "1"'

    assert _is_documented_legacy_cleanup(relative_path, cleanup)
    assert not _is_documented_legacy_cleanup(relative_path, setter)
    assert any(pattern.search(setter) for pattern in LEGACY_IDENTIFIER_PATTERNS)


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
        "AGENTS.md",
        "EXAMPLE_CODEX_PLAN.md",
        "docs/nested/Example_Implementation_Plan.MD",
        "docs/nested/Example_Master_Start_Prompt.MD",
        ".local-sdk/private.md",
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    assert {path.relative_to(tmp_path).as_posix() for path in _project_text_files()} == {
        "src/current.py", "src/reports/public_module.py", "docs/release.md"
    }
