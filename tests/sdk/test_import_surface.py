from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_package_and_native_import_without_runtime_side_effects(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    source_root = (repository_root / "src").resolve()
    existing_pythonpath = environment.get("PYTHONPATH")
    if existing_pythonpath:
        installed_paths = [
            value
            for value in existing_pythonpath.split(os.pathsep)
            if value and Path(value).resolve() != source_root
        ]
        if installed_paths:
            environment["PYTHONPATH"] = os.pathsep.join(installed_paths)
        else:
            environment.pop("PYTHONPATH", None)
    script = """
import sys

import aniflive_tts
import aniflive_tts._native as native

assert native.__version__ == aniflive_tts.__version__
assert not any(
    module in sys.modules
    for module in ("torch", "tensorrt", "fastapi", "uvicorn")
)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
