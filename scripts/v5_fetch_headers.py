"""Fetch checksum-pinned TensorRT headers for the V5 native helper build."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(output: Path) -> None:
    lock = json.loads((Path(__file__).with_name("v5_graph") / "tensorrt-headers.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    include = output / "include"
    include.mkdir()
    for row in lock["files"]:
        url = f"https://raw.githubusercontent.com/NVIDIA/TensorRT/{lock['commit']}/include/{row['name']}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"TensorRT header checksum mismatch: {row['name']}")
        (include / row["name"]).write_bytes(data)
    (output / "manifest.json").write_text(json.dumps(lock, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    fetch(parser.parse_args().output)
