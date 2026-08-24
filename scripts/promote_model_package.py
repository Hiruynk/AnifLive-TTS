from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from aniflive_tts.model_package import (
    validate_checksums,
    validate_safe_identifier,
    write_checksums,
)


def _assert_regular_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Model package must not contain symbolic links: {path}")


def promote(source: Path, output: Path, model_id: str) -> Path:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    model_id = validate_safe_identifier(model_id, "model_id")
    validate_checksums(source)
    _assert_regular_tree(source)

    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = validate_safe_identifier(
        manifest.get("active_engine_fingerprint"), "active_engine_fingerprint"
    )
    active_engine = source / "engines" / fingerprint
    if not active_engine.is_dir():
        raise ValueError(f"Active engine directory does not exist: {active_engine}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    try:
        staging.mkdir(parents=True)
        for entry in source.iterdir():
            if entry.name in {"checksums.json", "engines"}:
                continue
            destination = staging / entry.name
            if entry.is_dir():
                shutil.copytree(entry, destination)
            elif entry.is_file():
                shutil.copy2(entry, destination)
        shutil.copytree(active_engine, staging / "engines" / fingerprint)

        manifest["model_id"] = model_id
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(staging)
        validate_checksums(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote a validated model package without copying inactive engines."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    print(promote(args.source, args.output, args.model_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
