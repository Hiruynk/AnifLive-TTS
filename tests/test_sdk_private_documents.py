"""Local execution instructions must never enter release inventories."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "sdk_release_gate", ROOT / "scripts" / "check_release_security.py"
)
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.mark.parametrize("name", [
    "AGENTS.md",
    "docs/AGENTS.md",
    "ANIFLIVE_TTS_V1_5_BETA_RUST_PYTHON_SDK_CODEX_PLAN.md",
    ".local-sdk/evidence.json",
])
def test_private_instructions_rejected_even_if_force_tracked(name: str) -> None:
    with pytest.raises(SystemExit, match="forbidden tracked paths"):
        gate._check_tracked_paths([name])


def test_public_sdk_documents_remain_allowed() -> None:
    gate._check_tracked_paths([
        "docs/V1_5_SDK_API_AUDIT.md",
        "docs/V1_5_SDK_ARCHITECTURE.md",
        "docs/adr/ADR-V1_5-RUST-SDK-PACKAGING.md",
    ])
