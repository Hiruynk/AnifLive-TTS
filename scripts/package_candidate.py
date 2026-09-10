#!/usr/bin/env python3
"""Create a local candidate from reviewed working-tree files, without Git mutation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIRECTORIES = {
    ".github", "assets", "benchmarks", "docs", "licenses", "minimal_inference",
    "requirements", "research", "scripts", "src", "tests", "webui",
}
PUBLIC_ROOT_FILES = {
    ".dockerignore", ".env.example", ".gitattributes", ".gitignore",
    "Dockerfile", "Dockerfile.dev", "Dockerfile.workstation-worker", "Dockerfile.studio",
    "LICENSE", "LICENSING.md", "README.md", "README_ZH_CN.md", "README_ZH_HK.md",
    "SECURITY.md", "THIRD_PARTY_NOTICES.md", "api.py", "local_tts_cf.py",
    "docker-compose.cu126.yml", "docker-compose.yml", "pyproject.toml",
    "run_studio.bat", "run_studio_docker.bat", "run_tts.bat", "run_webui.bat", "uv.lock",
}
PRIVATE_NAMES = {
    "login.html", "signin.html", "login.py", "local_auth.py", "auth.local.json",
    "docker-compose.local.yml", ".cloudflared-token",
}


def validate_candidate_path(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Unsafe candidate path")
    parts = [part.lower() for part in path.parts]
    if len(path.parts) == 1 and name not in PUBLIC_ROOT_FILES and not re.fullmatch(
            r"RELEASE_NOTES_v[0-9][A-Za-z0-9._-]*\.md", name):
        raise ValueError(f"Root file has not been approved for a public candidate: {name}")
    if (not parts or parts[-1] in PRIVATE_NAMES
            or re.search(r"login|signin|credentials|password|secrets", path.name, re.I)
            or path.stem.lower() in {"local_auth", "accounts", "users"}
            or any(part.startswith(("credentials", "secrets")) for part in parts)
            or any(part in {"private", ".git", ".venv", "__pycache__"} for part in parts)
            or (len(parts) > 1 and parts[0] not in PUBLIC_DIRECTORIES)):
        raise ValueError(f"Local-only path cannot enter a candidate: {name}")


def source_files(root: Path) -> list[str]:
    result = subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), "ls-files",
         "--cached", "--others", "--exclude-standard", "-z"],
    )
    names = sorted(set(result.decode("utf-8").split("\0")) - {""})
    files = []
    for name in names:
        path = root / name
        if not path.exists():  # Respect intentional working-tree deletions.
            continue
        validate_candidate_path(name)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Candidate source is not a contained regular file: {name}")
        files.append(name)
    return files


def load_gate(root: Path):
    spec = importlib.util.spec_from_file_location("candidate_security_gate",
                                                root / "scripts/check_release_security.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_candidate(root: Path, output: Path, version: str) -> Path:
    root = root.resolve()
    files = source_files(root)
    gate = load_gate(root)
    gate._check_tracked_paths(files)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aniflive-candidate-") as temp:
        stage = Path(temp) / "AnifLive-TTS"
        stage.mkdir()
        inventory = {}
        for name in files:
            source = root / name
            content = source.read_bytes()
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            inventory[name] = hashlib.sha256(content).hexdigest()
        # Validate the exact public bytes, not the live/local installation.
        gate.ROOT = stage
        missing = [name for name in gate.REQUIRED_FILES if name not in inventory]
        if missing:
            raise ValueError("Candidate lacks required release files: " + ", ".join(missing))
        gate._check_credentials(files)
        gate._check_action_pins()
        gate._check_public_webui()
        tree_hash = hashlib.sha256(json.dumps(inventory, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
        safe_version = version.replace(".", "").replace("-", "")
        if not safe_version.isalnum():
            raise ValueError("Invalid candidate version")
        archive = output / f"AnifLive-TTS-v{version}-candidate-{tree_hash[:12]}.zip"
        manifest = {
            "schema": "aniflive-local-release-candidate-v1",
            "version": version, "status": "candidate-created",
            "qualification": "separate-acceptance-record-required",
            "base_commit": subprocess.check_output(
                ["git", "-c", f"safe.directory={root}", "-C", str(root),
                 "rev-parse", "HEAD"], text=True).strip(),
            "source_tree_sha256": tree_hash, "files": inventory,
            "local_auth_included": False, "upload_authorized": False,
        }
        with archive.open("xb") as sink:
            with zipfile.ZipFile(sink, "w", zipfile.ZIP_DEFLATED) as bundle:
                for name in files:
                    bundle.write(stage / name, "AnifLive-TTS/" + name)
                bundle.writestr("AnifLive-TTS/CANDIDATE_MANIFEST.json",
                                json.dumps(manifest, indent=2) + "\n")
        with archive.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        archive.with_suffix(".sha256").write_text(f"{digest}  {archive.name}\n")
        return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="1.4.0")
    args = parser.parse_args()
    print(create_candidate(args.source_root, args.output_dir, args.version))


if __name__ == "__main__":
    main()
