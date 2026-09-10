#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
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
    "run_webui.bat",
    "scripts/normalize_spdx_sbom.py",
    "scripts/check_trivy_report.py",
    "scripts/shared_assets_lock.json",
    "webui/index.html",
    "webui/annotation_editor.js",
    "webui/lucide.min.js",
    "webui/media/voice-workstation-background.mp4",
    "webui/media/voice-workstation-poster.jpg",
    "webui/playback_model.js",
    "webui/studio.css",
    "webui/studio.js",
    "webui/synthesis.html",
)
FORBIDDEN_PREFIXES = ("Miku/", "data/", "dist/", "reports/")
FORBIDDEN_GENERATED_ROOT_PREFIXES = (".pytest-", "tmp")
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
    ".sqlite3",
    ".wav",
)
SECURITY_EVIDENCE_SCHEMA = "aniflive-tts-security-verification-v1"


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, encoding="utf-8"
    ).strip()


def _release_files(*, include_untracked: bool = False) -> list[str]:
    args = ["ls-files", "-z"]
    if include_untracked:
        args.extend(("--cached", "--others", "--exclude-standard"))
    output = _git(*args)
    return [name for name in output.split("\0") if name]


def _check_tracked_paths(files: list[str]) -> None:
    violations: list[str] = []
    for name in files:
        normalized = name.replace("\\", "/")
        lowered = normalized.lower()
        root_component = lowered.split("/", 1)[0]
        if lowered == ".env" or (
            lowered.startswith(".env.") and lowered != ".env.example"
        ):
            violations.append(normalized)
        if any(normalized.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            violations.append(normalized)
        if root_component.startswith(FORBIDDEN_GENERATED_ROOT_PREFIXES):
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


def _check_public_webui() -> None:
    backend_file = ROOT / "src" / "aniflive_tts" / "webui.py"
    frontend_files = [ROOT / "run_webui.bat"]
    frontend_files.extend(
        path
        for path in sorted((ROOT / "webui").rglob("*"))
        if path.is_file()
        and path.suffix.lower() in {".css", ".html", ".js", ".json", ".md", ".txt"}
    )
    missing = [
        str(path.relative_to(ROOT))
        for path in (backend_file, frontend_files[0])
        if not path.is_file()
    ]
    if missing or not (ROOT / "webui" / "index.html").is_file():
        raise SystemExit("Missing public WebUI files: " + ", ".join(missing or ["webui/index.html"]))
    frontend = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in frontend_files
        if path.is_file()
    ).lower()
    frontend_forbidden = (
        "login.html",
        "username",
        "password",
        "password_hash",
        "credential",
        "session_cookie",
        "authorization",
        "sessionstorage",
    )
    backend = backend_file.read_text(encoding="utf-8", errors="ignore").lower()
    backend_forbidden = (
        "login.html",
        "password_hash",
        "session_cookie",
        "sessionstorage",
        '@app.get("/login',
        '@app.post("/login',
        '@app.get("/signin',
        '@app.post("/signin',
    )
    violations = [
        f"frontend:{value}" for value in frontend_forbidden if value in frontend
    ]
    violations.extend(
        f"backend:{value}" for value in backend_forbidden if value in backend
    )
    if violations:
        raise SystemExit(
            "Public WebUI contains login or credential state: " + ", ".join(violations)
        )
    # Classic WebUI and Studio each read and write the same interface-locale key.
    if frontend.count("localstorage") != 4 or (
        'const locale_key = "aniflive.uilocale"' not in frontend
    ):
        raise SystemExit("Public WebUI may only persist its interface locale")


def _release_tree_sha256(files: list[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        path = ROOT / name
        if not path.is_file():
            continue
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(name.replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_evidence(
    output: Path,
    *,
    subject_kind: str,
    subject_id: str,
    files: list[str],
) -> None:
    checks = (
        ("release-inventory", "Required and release-intended files were inventoried"),
        ("private-asset-scan", "Forbidden model, audio, runtime, and private asset paths were absent"),
        ("credential-scan", "Tracked text files contained no recognized credential material"),
        ("action-pin-review", "External GitHub Actions use immutable full commit SHAs"),
        ("public-webui-state", "Public WebUI persistence and authentication state passed policy"),
        ("oci-source-metadata", "Container source revision metadata contract is present"),
    )
    payload = {
        "schema": SECURITY_EVIDENCE_SCHEMA,
        "subject": {"kind": subject_kind, "id": subject_id},
        "tool": {"name": "scripts/check_release_security.py", "version": "1"},
        "release_tree_sha256": _release_tree_sha256(files),
        "checks": [
            {"id": check_id, "status": "passed", "summary": summary}
            for check_id, summary in checks
        ],
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
    }
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AnifLive-TTS release security gate")
    parser.add_argument("--expected-version")
    parser.add_argument(
        "--include-untracked",
        action="store_true",
        help="also inspect untracked files intended for the next release",
    )
    parser.add_argument(
        "--evidence-output",
        type=Path,
        help="write a checksummed security evidence artifact after every check passes",
    )
    parser.add_argument("--subject-kind", choices=("artifact", "expression"))
    parser.add_argument("--subject-id")
    args = parser.parse_args()
    if args.evidence_output is not None and (
        args.subject_kind is None or not args.subject_id
    ):
        parser.error("--evidence-output requires --subject-kind and --subject-id")
    if args.evidence_output is None and (
        args.subject_kind is not None or args.subject_id is not None
    ):
        parser.error("--subject-kind and --subject-id require --evidence-output")

    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("Missing release security files: " + ", ".join(missing))

    files = _release_files(include_untracked=args.include_untracked)
    untracked_required = [name for name in REQUIRED_FILES if name not in files]
    if untracked_required:
        raise SystemExit(
            "Release security files must be tracked by Git: "
            + ", ".join(untracked_required)
        )
    _check_tracked_paths(files)
    _check_credentials(files)
    _check_action_pins()
    _check_public_webui()

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    if 'ARG VCS_REF=' not in dockerfile:
        raise SystemExit("Dockerfile must accept VCS_REF")
    if 'org.opencontainers.image.revision="${VCS_REF}"' not in dockerfile:
        raise SystemExit("Dockerfile must publish the OCI source revision")

    if args.expected_version:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        if f'version = "{args.expected_version}"' not in pyproject:
            raise SystemExit("pyproject version does not match --expected-version")
        release_notes = ROOT / f"RELEASE_NOTES_v{args.expected_version}.md"
        if not release_notes.is_file():
            raise SystemExit(f"Missing release notes: {release_notes.name}")
        if release_notes.name not in files:
            raise SystemExit(f"Release notes must be tracked by Git: {release_notes.name}")

    if args.evidence_output is not None:
        _write_evidence(
            args.evidence_output,
            subject_kind=args.subject_kind,
            subject_id=args.subject_id,
            files=files,
        )

    print(f"Release security gate passed for {len(files)} release files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
