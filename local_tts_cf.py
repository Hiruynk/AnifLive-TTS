from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PACKAGE = Path(
    os.environ.get(
        "ANIFLIVE_TTS_MODEL_PACKAGE",
        PROJECT_ROOT / "data" / "models" / "active",
    )
)
SHARED_DIR = Path(os.environ.get("ANIFLIVE_TTS_SHARED_DIR", PROJECT_ROOT / "data" / "shared"))


def main() -> int:
    expected = (PROJECT_ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected:
        raise RuntimeError("local_tts_cf.py must run with .venv\\Scripts\\python.exe")
    command = [
        sys.executable,
        "-B",
        str(PROJECT_ROOT / "api.py"),
        "--model-package",
        str(MODEL_PACKAGE),
        "--shared-dir",
        str(SHARED_DIR),
        "-a",
        "127.0.0.1",
        "-p",
        "9880",
    ]
    try:
        return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
    except KeyboardInterrupt:
        print("\nAnifLive-TTS API server stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
