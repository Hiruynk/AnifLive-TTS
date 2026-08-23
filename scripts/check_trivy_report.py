#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


CU121_TORCH_EXCEPTIONS = {
    "CVE-2025-32434",
    "CVE-2026-24747",
}
BLOCKING_SEVERITIES = {"HIGH", "CRITICAL"}


def _is_reviewed_exception(profile: str, vulnerability: dict[str, object]) -> bool:
    package = str(vulnerability.get("PkgName", "")).lower()
    installed = str(vulnerability.get("InstalledVersion", ""))
    identifier = str(vulnerability.get("VulnerabilityID", ""))
    return (
        profile == "cu121"
        and package in {"torch", "pytorch"}
        and installed.startswith("2.5.1")
        and identifier in CU121_TORCH_EXCEPTIONS
    )


def find_blockers(report: dict[str, object], profile: str) -> list[str]:
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
            if _is_reviewed_exception(profile, vulnerability):
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
    return blockers


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the AnifLive-TTS release policy to a Trivy image report"
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile", choices=("cu121", "cu128"), required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    blockers = find_blockers(report, args.profile)
    if blockers:
        print("Unaccepted HIGH/CRITICAL image vulnerabilities:")
        for blocker in blockers:
            print(f"- {blocker}")
        return 1
    print(f"Trivy release policy passed for {args.profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
