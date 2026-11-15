import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest


@pytest.mark.parametrize("payload_layout", ["native", "root"])
@pytest.mark.parametrize("precision", ["FP16", "FP32"])
def test_refresh_copies_payload_without_modifying_hardlinked_original(tmp_path, payload_layout, precision):
    golden = tmp_path / "original"
    files = {
        "manifest.json": json.dumps({"precision": precision}).encode(),
        "checksums.json": b"original-checksums",
        "native/engine-manifest.json": b"original-engine",
        "native/bin/aniflive-macos-v5turbo": b"original-worker",
        "native/lib/libmlx.dylib": b"original-mlx",
        "native/lib/libjaccl.dylib": b"original-jaccl",
        "native/lib/mlx.metallib": b"original-metal",
        "native/licenses/notice.txt": b"original-license",
        "native/models/model.bin": b"original-model",
    }
    for name, data in files.items():
        target = golden / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    package = tmp_path / "copy"
    shutil.copytree(golden, package, copy_function=os.link)
    payload = tmp_path / "payload"
    root = payload / "native" if payload_layout == "native" else payload
    updated = {
        "bin/aniflive-macos-v5turbo": b"new-worker",
        "lib/libmlx.dylib": b"new-mlx",
        "lib/libjaccl.dylib": b"new-jaccl",
        "lib/mlx.metallib": b"new-metal",
        "licenses/notice.txt": b"new-license",
    }
    for name, data in updated.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (payload / "build-manifest.json").write_text(json.dumps({"source_commit": "source-commit"}))
    script = Path(__file__).resolve().parents[1] / "scripts/macos-qualification/fp16_package_refresh.py"
    subprocess.run([sys.executable, str(script), str(package), str(payload)], check=True, capture_output=True)

    for name, data in files.items():
        assert (golden / name).read_bytes() == data
    for name, data in updated.items():
        assert (package / "native" / name).read_bytes() == data
    engine = json.loads((package / "native/engine-manifest.json").read_text())
    inventory = json.loads((package / "checksums.json").read_text())
    assert engine["precision"] == precision
    assert engine["source_commit"] == "source-commit"
    assert engine["worker_sha256"] == hashlib.sha256(b"new-worker").hexdigest()
    assert inventory["native/models/model.bin"] == hashlib.sha256(b"original-model").hexdigest()
    assert inventory["native/licenses/notice.txt"] == hashlib.sha256(b"new-license").hexdigest()
    assert (package / "native/bin/aniflive-macos-v5turbo").stat().st_mode & stat.S_IXUSR
