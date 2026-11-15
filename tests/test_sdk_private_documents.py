"""Local execution instructions must never enter release inventories."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from scripts import package_candidate

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
    "EXAMPLE_CODEX_PLAN.md",
    "EXAMPLE_IMPLEMENTATION_PLAN.md",
    "EXAMPLE_MASTER_START_PROMPT.md",
    "docs/nested/aGeNtS.mD",
    "docs/nested/Example_Codex_Plan.MD",
    "docs/nested/Example_Implementation_Plan.MD",
    "docs/nested/Example_Master_Start_Prompt.MD",
    ".local-sdk/evidence.json",
    ".local-stable/acceptance/evidence.json",
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


@pytest.mark.parametrize("name", [
    "AGENTS.md",
    "EXAMPLE_CODEX_PLAN.md",
    "EXAMPLE_IMPLEMENTATION_PLAN.md",
    "EXAMPLE_MASTER_START_PROMPT.md",
    "docs/nested/aGeNtS.mD",
    "docs/nested/Example_Codex_Plan.MD",
    "docs/nested/Example_Implementation_Plan.MD",
    "docs/nested/Example_Master_Start_Prompt.MD",
])
def test_candidate_rejects_private_instructions(name: str) -> None:
    with pytest.raises(ValueError):
        package_candidate.validate_candidate_path(name)
