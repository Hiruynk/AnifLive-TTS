from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AnifLive-TTS v1 API")
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, default=PROJECT_ROOT / "data" / "shared")
    parser.add_argument("-a", "--host", default="127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=9880)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    os.environ["ANIFLIVE_TTS_MODEL_PACKAGE"] = str(args.model_package.resolve())
    os.environ["ANIFLIVE_TTS_SHARED_DIR"] = str(args.shared_dir.resolve())
    os.environ.setdefault("ANIFLIVE_TTS_SOURCE_DIR", str(PROJECT_ROOT / "minimal_inference"))
    from aniflive_tts.api import create_app
    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
