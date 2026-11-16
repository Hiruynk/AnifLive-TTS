from __future__ import annotations

import runpy
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
POLICY = runpy.run_path(str(ROOT / "scripts" / "check_trivy_report.py"))
IMAGE = "localhost:5000/aniflive-tts:1.5.0-cu128@sha256:" + "a" * 64


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


def test_review_is_explicit_and_does_not_hide_other_versions():
    policy = POLICY["load_applicability"](ROOT / "scripts/release_vulnerability_applicability.json")
    finding = {"PkgName": "transformers", "InstalledVersion": "5.5.0",
               "VulnerabilityID": "CVE-2026-9856", "Severity": "HIGH"}
    report = _report(finding)
    report["Metadata"] = {"OS": {"Family": "ubuntu", "Name": "24.04"}}
    assert POLICY["find_blockers"](report, "cu126")
    assert POLICY["find_blockers"](report, "cu126", applicability=policy) == []
    finding["InstalledVersion"] = "5.6.0"
    assert POLICY["find_blockers"](report, "cu126", applicability=policy)


def test_secret_findings_block_without_echoing_secret_material():
    report = _report({"Severity": "LOW"})
    report["Results"][0]["Secrets"] = [{"RuleID": "test-token", "Match": "private-test-value"}]
    blockers = POLICY["find_blockers"](report, "cu128")
    assert blockers and all("private-test-value" not in value for value in blockers)


def _backport_report():
    report = _report({})
    report["ArtifactName"] = IMAGE
    report["Results"][0]["Vulnerabilities"] = [
        {"PkgName": "transformers", "InstalledVersion": "4.57.6",
         "VulnerabilityID": cve, "Severity": "HIGH"}
        for cve in sorted(POLICY["BACKPORTED_CVES"])
    ]
    return report


def _proof():
    return {
        "schema": "aniflive-transformers-backport-proof-v1", "image": IMAGE,
        "verification": {"status": "verified", "version": "4.57.6",
                         "upstream_fixes": deepcopy(POLICY["UPSTREAM_FIXES"]),
                         "sha256": deepcopy(POLICY["POST_HASHES"])},
    }


def test_backported_findings_require_explicit_complete_image_proof():
    report = _backport_report()
    assert len(POLICY["find_blockers"](report, "cu128", expected_image=IMAGE)) == 3
    assert POLICY["find_blockers"](report, "cu128", transformers_proof=_proof(), expected_image=IMAGE) == []
    assert POLICY["find_blockers"](report, "cu128", transformers_proof=_proof())
    assert POLICY["find_blockers"](report, "cu126", transformers_proof=_proof(), expected_image=IMAGE)


@pytest.mark.parametrize("change", [
    "wrong_digest", "mutable_image", "wrong_schema", "extra_field", "wrong_status", "wrong_version",
    "missing_hash", "changed_hash", "extra_hash", "missing_fix", "changed_fix", "extra_fix", "not_object",
])
def test_invalid_or_incomplete_backport_proof_blocks(change):
    proof = _proof()
    verification = proof["verification"]
    if change == "wrong_digest":
        proof["image"] = IMAGE[:-1] + "b"
    elif change == "mutable_image":
        proof["image"] = IMAGE.split("@")[0]
    elif change == "wrong_schema":
        proof["schema"] = "other"
    elif change == "extra_field":
        verification["unexpected"] = "private-test-value"
    elif change == "wrong_status":
        verification["status"] = "patched"
    elif change == "wrong_version":
        verification["version"] = "4.57.7"
    elif change == "not_object":
        proof = []
    else:
        mapping = verification["sha256" if change.endswith("hash") else "upstream_fixes"]
        key = next(iter(mapping))
        if change.startswith("missing"):
            del mapping[key]
        elif change.startswith("changed"):
            mapping[key] = "0" * 64
        else:
            mapping["unexpected"] = "private-test-value"
    blockers = POLICY["find_blockers"](_backport_report(), "cu128", transformers_proof=proof, expected_image=IMAGE)
    assert blockers and all("private-test-value" not in item for item in blockers)


def test_proof_must_match_the_report_and_an_immutable_expected_image():
    report = _backport_report()
    report["ArtifactName"] = IMAGE[:-1] + "b"
    assert POLICY["find_blockers"](report, "cu128", transformers_proof=_proof(), expected_image=IMAGE)
    report["ArtifactName"] = "image:mutable"
    proof = _proof()
    proof["image"] = report["ArtifactName"]
    assert POLICY["find_blockers"](report, "cu128", transformers_proof=proof,
                                     expected_image=report["ArtifactName"])


@pytest.mark.parametrize("field,value", [
    ("PkgName", "other-package"), ("InstalledVersion", "4.57.5"),
    ("VulnerabilityID", "CVE-2099-0001"),
])
def test_verified_backport_does_not_hide_other_findings(field, value):
    report = _backport_report()
    report["Results"][0]["Vulnerabilities"][0][field] = value
    blockers = POLICY["find_blockers"](report, "cu128", transformers_proof=_proof(), expected_image=IMAGE)
    assert len(blockers) == 1 and value in blockers[0]


def test_verified_backport_does_not_hide_secrets():
    report = _backport_report()
    report["Results"][0]["Secrets"] = [{"RuleID": "token", "Match": "private-test-value"}]
    assert POLICY["find_blockers"](report, "cu128", transformers_proof=_proof(), expected_image=IMAGE) == [
        "secret:aniflive-tts:token"
    ]
