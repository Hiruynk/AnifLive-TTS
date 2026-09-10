from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any

LEGACY = "legacy-topk-v1"
NATIVE = "native-v2proplus-v1"


def package_sampling_contract(manifest: Mapping[str, Any]) -> str:
    value = manifest.get("semantic_sampling", LEGACY)
    if not isinstance(value, str) or value not in {LEGACY, NATIVE}:
        raise ValueError("Unsupported model package semantic sampling contract")
    return value


def effective_sampling_contract(manifest: Mapping[str, Any] | None = None) -> str:
    default = (package_sampling_contract(manifest) if manifest is not None
               else os.environ.get("ANIFLIVE_TTS_PACKAGE_SEMANTIC_SAMPLING", LEGACY))
    value = os.environ.get("ANIFLIVE_TTS_SEMANTIC_SAMPLING", default)
    if value not in {LEGACY, NATIVE}:
        raise ValueError("Unsupported semantic sampling contract")
    return value
