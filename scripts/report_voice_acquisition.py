from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from aniflive_tts.dataset_factory import DatasetFactory
from aniflive_tts.voice_acquisition_report import build_voice_acquisition_report
from aniflive_tts.workstation import WorkstationStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compose an evidence-only AnifLive-TTS voice acquisition report."
    )
    parser.add_argument("--workstation-root", type=Path, default=Path("data/workstation"))
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def _atomic_write(path: Path, payload: bytes) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    arguments = _parser().parse_args()
    root = arguments.workstation_root.expanduser().resolve(strict=True)
    store = WorkstationStore(root)
    factory = DatasetFactory(
        root / "dataset-factory",
        allowed_source_roots=store.allowed_import_roots(),
    )
    report = build_voice_acquisition_report(store, factory, arguments.dataset_id)
    encoded = (
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    if arguments.output is not None:
        _atomic_write(arguments.output, encoded)
    sys.stdout.buffer.write(encoded)
    return 0 if report["ready_for_release"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
