from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeSettings:
    model_package: Path
    shared_dir: Path
    cache_dir: Path
    host: str = "0.0.0.0"
    port: int = 9880

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            model_package=Path(os.environ.get("ANIFLIVE_TTS_MODEL_PACKAGE", "/data/models/active")),
            shared_dir=Path(os.environ.get("ANIFLIVE_TTS_SHARED_DIR", "/data/shared")),
            cache_dir=Path(os.environ.get("ANIFLIVE_TTS_CACHE_DIR", "/data/cache")),
            host=os.environ.get("ANIFLIVE_TTS_HOST", "0.0.0.0"),
            port=int(os.environ.get("ANIFLIVE_TTS_PORT", "9880")),
        )

    def manifest(self) -> dict[str, Any]:
        path = self.model_package / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Model package manifest is missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

