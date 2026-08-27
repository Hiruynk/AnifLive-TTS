from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_security_policy_preserves_private_reporting_and_closed_contributions() -> None:
    policy = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "/security/advisories/new" in policy
    assert "does not currently accept external contributions or pull requests" in policy
    assert "weights_only=True" in policy
    assert "not a security sandbox" in policy


def test_external_actions_are_pinned_to_full_commit_shas() -> None:
    pattern = re.compile(r"^\s*-?\s*uses:\s*([^@\s]+)@([^\s#]+)", re.MULTILINE)
    for workflow in (ROOT / ".github" / "workflows").glob("*.yml"):
        for action, revision in pattern.findall(workflow.read_text(encoding="utf-8")):
            if not action.startswith("./"):
                assert re.fullmatch(r"[0-9a-f]{40}", revision), (
                    f"{workflow.name}: {action}@{revision} is mutable"
                )


def test_container_release_records_source_and_evidence() -> None:
    workflow = (ROOT / ".github" / "workflows" / "container.yml").read_text(
        encoding="utf-8"
    )
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "VCS_REF=${{ github.sha }}" in workflow
    assert "BUILD_DATE=${{ steps.build_date.outputs.value }}" in workflow
    assert "index:org.opencontainers.image.revision=${{ github.sha }}" in workflow
    assert "RELEASE-METADATA-AnifLive-TTS-v1.2.0-" in workflow
    assert "provenance: mode=max" in workflow
    assert "pyspdxtools -i" in workflow
    assert workflow.count('python scripts/normalize_spdx_sbom.py "${output}"') == 2
    assert 'org.opencontainers.image.revision="${VCS_REF}"' in dockerfile


def test_release_checksums_cover_all_image_evidence() -> None:
    script = (ROOT / "scripts" / "update_release_checksums.ps1").read_text(
        encoding="utf-8"
    )
    assert "RELEASE-METADATA-AnifLive-TTS-v$Version-cu128.json" in script
    assert "RELEASE-METADATA-AnifLive-TTS-v$Version-cu126.json" in script
    assert "TRIVY-AnifLive-TTS-v$Version-cu128.json" in script
    assert "TRIVY-AnifLive-TTS-v$Version-cu126.json" in script
