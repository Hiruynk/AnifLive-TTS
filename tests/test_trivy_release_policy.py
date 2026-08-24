from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY = runpy.run_path(str(ROOT / "scripts" / "check_trivy_report.py"))


def _report(vulnerability: dict[str, str]) -> dict[str, object]:
    return {
        "SchemaVersion": 2,
        "ArtifactName": "ghcr.io/hiruynk/aniflive-tts@sha256:test",
        "Results": [
            {
                "Target": "aniflive-tts",
                "Vulnerabilities": [vulnerability],
            }
        ]
    }


def test_high_severity_torch_findings_always_block_release() -> None:
    finding = {
        "PkgName": "torch",
        "InstalledVersion": "2.10.0+cu126",
        "VulnerabilityID": "CVE-2026-24747",
        "Severity": "HIGH",
    }
    assert POLICY["find_blockers"](_report(finding), "cu126")
    assert POLICY["find_blockers"](_report(finding), "cu128")


def test_other_high_or_critical_findings_block_release() -> None:
    finding = {
        "PkgName": "example",
        "InstalledVersion": "1.0",
        "VulnerabilityID": "CVE-2099-0001",
        "Severity": "CRITICAL",
    }
    assert POLICY["find_blockers"](_report(finding), "cu126")
    assert POLICY["find_blockers"](_report(finding), "cu128")


def test_low_severity_findings_do_not_block_release() -> None:
    finding = {
        "PkgName": "example",
        "InstalledVersion": "1.0",
        "VulnerabilityID": "CVE-2099-0002",
        "Severity": "LOW",
    }
    assert POLICY["find_blockers"](_report(finding), "cu128") == []


def test_empty_or_malformed_reports_block_release() -> None:
    assert POLICY["find_blockers"]({}, "cu128")
    assert POLICY["find_blockers"](
        {"SchemaVersion": 2, "ArtifactName": "image", "Results": []}, "cu128"
    )
