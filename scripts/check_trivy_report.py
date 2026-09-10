#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from datetime import date
import json
from pathlib import Path


BLOCKING_SEVERITIES = {"HIGH", "CRITICAL"}


def find_blockers(report: dict[str, object], profile: str, *, applicability: dict | None = None) -> list[str]:
    schema_version = report.get("SchemaVersion")
    results = report.get("Results")
    if not isinstance(schema_version, int) or schema_version < 2:
        return ["invalid-report:missing-or-unsupported-schema"]
    if not isinstance(report.get("ArtifactName"), str) or not report["ArtifactName"]:
        return ["invalid-report:missing-artifact-name"]
    if not isinstance(results, list) or not results:
        return ["invalid-report:missing-scan-results"]

    blockers: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        target = str(result.get("Target", "unknown"))
        for vulnerability in result.get("Vulnerabilities") or []:
            if not isinstance(vulnerability, dict):
                continue
            severity = str(vulnerability.get("Severity", "")).upper()
            if severity not in BLOCKING_SEVERITIES:
                continue
            if applicability is not None and reviewed_not_affected(report, result, vulnerability, applicability):
                continue
            blockers.append(
                ":".join(
                    (
                        target,
                        str(vulnerability.get("PkgName", "unknown")),
                        str(vulnerability.get("InstalledVersion", "unknown")),
                        str(vulnerability.get("VulnerabilityID", "unknown")),
                        severity,
                    )
                )
            )
        for finding in result.get("Secrets") or []:
            blockers.append(f"secret:{target}:{finding.get('RuleID', 'unknown')}")
    return blockers


def load_applicability(path: Path) -> dict:
    policy = json.loads(path.read_text())
    if policy.get("schema") != "aniflive-release-vulnerability-applicability-v1":
        raise ValueError("Unsupported vulnerability applicability schema")
    if date.fromisoformat(policy["expires"]) < date.today():
        raise ValueError("Vulnerability applicability review has expired")
    root = Path(__file__).resolve().parents[1]
    guards = policy.get("source_guards")
    if not isinstance(guards, dict) or not guards:
        raise ValueError("Applicability review has no source guards")
    for name, digest in guards.items():
        source = (root / name).resolve()
        if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("Applicability review source changed: " + name)
    return policy


def reviewed_not_affected(report, result, vulnerability, policy) -> bool:
    os_record = report.get("Metadata", {}).get("OS", {})
    if any(os_record.get(key) != value for key, value in policy.get("os", {}).items()):
        return False
    all_packages = [p for r in report.get("Results", []) for p in r.get("Packages", [])]
    for review in policy.get("reviews", []):
        if (vulnerability.get("PkgName") != review["package"]
                or vulnerability.get("InstalledVersion") not in review["versions"]
                or vulnerability.get("VulnerabilityID") not in review["ids"]):
            continue
        if not review.get("reason"):
            continue
        if any(not any(p.get("Name") == name and p.get("Version") == version
                       for p in all_packages)
               for name, version in review.get("requires_package", {}).items()):
            continue
        if review.get("sbom_only"):
            matching = [p for p in result.get("Packages", [])
                        if p.get("Name") == review["package"]
                        and p.get("Version") == vulnerability.get("InstalledVersion")]
            if not matching or any(p.get("AnalyzedBy") != "sbom" for p in matching):
                continue
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the AnifLive-TTS release policy to a Trivy image report"
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile", choices=("cu126", "cu128"), required=True)
    parser.add_argument("--applicability", type=Path, help="Explicit source-bound not-affected review")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    policy = load_applicability(args.applicability) if args.applicability else None
    blockers = find_blockers(report, args.profile, applicability=policy)
    if blockers:
        print("Unaccepted HIGH/CRITICAL image vulnerabilities:")
        for blocker in blockers:
            print(f"- {blocker}")
        return 1
    print(f"Trivy release policy passed for {args.profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
