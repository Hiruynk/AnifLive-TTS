"""Select an installed package for first-start use without an active-folder alias."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from .model_backend import model_backend_for_manifest
from .model_package import validate_safe_identifier


def select_startup_package(requested: Path) -> Path:
    requested = requested.expanduser().resolve()
    if (requested / "manifest.json").is_file():
        return requested
    if requested.name != "active" or not requested.parent.is_dir():
        raise FileNotFoundError("Configured model package was not found")
    candidates = []
    for directory in requested.parent.iterdir():
        if directory.name.startswith(".") or directory.is_symlink() or not directory.is_dir():
            continue
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
            backend = model_backend_for_manifest(manifest) if isinstance(manifest, dict) else None
            if backend is None or not backend.supports_manifest(manifest) or not (directory / "checksums.json").is_file():
                continue
            validate_safe_identifier(manifest.get("model_id"), "model_id")
        except (OSError, ValueError, RuntimeError):
            continue
        candidates.append((0 if manifest.get("qualification") else 1, directory.name, directory))
    if not candidates:
        raise FileNotFoundError("No installed model package was found")
    return sorted(candidates)[0][2].resolve()


if __name__ == "__main__":
    print(select_startup_package(Path(sys.argv[1])))
