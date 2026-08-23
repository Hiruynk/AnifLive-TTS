#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    ".gitattributes",
    ".dockerignore",
    ".env.example",
    "LICENSE",
    "LICENSING.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/RELEASE_VERIFICATION.md",
    "docs/REPOSITORY_SECURITY_SETTINGS.md",
    "scripts/normalize_spdx_sbom.py",
    "scripts/check_trivy_report.py",
    "scripts/shared_assets_lock.json",
)
FORBIDDEN_PREFIXES = ("Miku/", "data/", "dist/", "reports/")
FORBIDDEN_SUFFIXES = (
    ".ckpt",
    ".engine",
    ".flac",
    ".mp3",
    ".onnx.data",
    ".p12",
    ".pem",
    ".pfx",
    ".pth",
    ".safetensors",
    ".wav",
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, encoding="utf-8"
    ).strip()


def _tracked_files() -> list[str]:
    output = _git("ls-files", "-z")
    return [name for name in output.split("\0") if name]


def _check_tracked_paths(files: list[str]) -> None:
    violations: list[str] = []
    for name in files:
        normalized = name.replace("\\", "/")
        lowered = normalized.lower()
        if lowered == ".env" or (
            lowered.startswith(".env.") and lowered != ".env.example"
        ):
            violations.append(normalized)
        if any(normalized.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            violations.append(normalized)
        if lowered.endswith(FORBIDDEN_SUFFIXES):
            violations.append(normalized)
        if any(word in lowered for word in ("cloudflare-token", "tunnel-token")):
            violations.append(normalized)
    if violations:
        raise SystemExit(
            "Release contains forbidden tracked paths:\n  " + "\n  ".join(sorted(set(violations)))
        )


def _check_credentials(files: list[str]) -> None:
    token_pattern = re.compile(
        r"(?i)(?:gh[pousr]_[A-Za-z0-9_]{30,}|"
        r"github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{30,}|"
        r"sk-[A-Za-z0-9_-]{30,})"
    )
    private_key_marker = "-----BEGIN " + "PRIVATE KEY-----"
    jwt_pattern = re.compile(
        r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{40,}\.[A-Za-z0-9_-]{40,}\.[A-Za-z0-9_-]{20,}"
    )
    violations: list[str] = []
    for name in files:
        path = ROOT / name
        if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if token_pattern.search(text) or jwt_pattern.search(text) or private_key_marker in text:
            violations.append(name)
    if violations:
        raise SystemExit(
            "Potential credential material found in tracked files:\n  "
            + "\n  ".join(sorted(violations))
        )


def _check_action_pins() -> None:
    pattern = re.compile(r"^\s*-?\s*uses:\s*([^@\s]+)@([^\s#]+)", re.MULTILINE)
    violations: list[str] = []
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        for action, revision in pattern.findall(path.read_text(encoding="utf-8")):
            if action.startswith("./"):
                continue
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                violations.append(f"{path.relative_to(ROOT)}: {action}@{revision}")
    if violations:
        raise SystemExit(
            "GitHub Actions must use full commit SHAs:\n  " + "\n  ".join(violations)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AnifLive-TTS release security gate")
    parser.add_argument("--expected-version")
    args = parser.parse_args()

    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("Missing release security files: " + ", ".join(missing))

    files = _tracked_files()
    untracked_required = [name for name in REQUIRED_FILES if name not in files]
    if untracked_required:
        raise SystemExit(
            "Release security files must be tracked by Git: "
            + ", ".join(untracked_required)
        )
    _check_tracked_paths(files)
    _check_credentials(files)
    _check_action_pins()

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    if 'ARG VCS_REF=' not in dockerfile:
        raise SystemExit("Dockerfile must accept VCS_REF")
    if 'org.opencontainers.image.revision="${VCS_REF}"' not in dockerfile:
        raise SystemExit("Dockerfile must publish the OCI source revision")

    if args.expected_version:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        if f'version = "{args.expected_version}"' not in pyproject:
            raise SystemExit("pyproject version does not match --expected-version")

    print(f"Release security gate passed for {len(files)} tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
