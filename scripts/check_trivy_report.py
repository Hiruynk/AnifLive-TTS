#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from datetime import date
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.v5_transformers_security import POST_HASHES, UPSTREAM_FIXES, VERSION


BLOCKING_SEVERITIES = {"HIGH", "CRITICAL"}
BACKPORTED_CVES = {"CVE-2026-4372", "CVE-2026-5241", "CVE-2026-9856"}


def verified_transformers_proof(proof: dict, image: str, profile: str) -> bool:
    """Validate evidence emitted by the isolated verifier for the scanned digest.

    The trusted workflow supplies image identity; the verifier supplies source hashes.
    A version string or a vulnerability allowlist alone is never sufficient evidence.
    """
    if (profile != "cu128" or not isinstance(proof, dict)
            or VERSION != "4.57.6" or set(UPSTREAM_FIXES) != BACKPORTED_CVES or not POST_HASHES):
        return False
    if not isinstance(image, str) or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image):
        return False
    if set(proof) != {"schema", "image", "verification"}:
        return False
    if proof["schema"] != "aniflive-transformers-backport-proof-v1" or proof["image"] != image:
        return False
    verification = proof["verification"]
    return (
        isinstance(verification, dict)
        and set(verification) == {"status", "version", "upstream_fixes", "sha256"}
        and verification["status"] == "verified"
        and verification["version"] == VERSION
        and verification["upstream_fixes"] == UPSTREAM_FIXES
        and verification["sha256"] == POST_HASHES
    )


def find_blockers(report: dict[str, object], profile: str, *, applicability: dict | None = None,
                  transformers_proof: dict | None = None, expected_image: str | None = None) -> list[str]:
    schema_version = report.get("SchemaVersion")
    results = report.get("Results")
    if not isinstance(schema_version, int) or schema_version < 2:
        return ["invalid-report:missing-or-unsupported-schema"]
    if not isinstance(report.get("ArtifactName"), str) or not report["ArtifactName"]:
        return ["invalid-report:missing-artifact-name"]
    if not isinstance(results, list) or not results:
        return ["invalid-report:missing-scan-results"]
    if expected_image is not None and report["ArtifactName"] != expected_image:
        return ["invalid-report:artifact-does-not-match-expected-image"]

    backport_verified = False
    if transformers_proof is not None:
        backport_verified = verified_transformers_proof(transformers_proof, expected_image, profile)
        if not backport_verified:
            return ["invalid-transformers-backport-proof"]

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
            if (backport_verified and vulnerability.get("PkgName") == "transformers"
                    and vulnerability.get("InstalledVersion") == VERSION
                    and vulnerability.get("VulnerabilityID") in BACKPORTED_CVES):
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
    parser.add_argument("--transformers-proof", type=Path, help="Source-hash proof from the immutable image verifier")
    parser.add_argument("--expected-image", help="Exact immutable image reference passed to Trivy and the verifier")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    policy = load_applicability(args.applicability) if args.applicability else None
    proof = json.loads(args.transformers_proof.read_text(encoding="utf-8")) if args.transformers_proof else None
    if args.transformers_proof and proof is None:
        raise ValueError("Transformers proof must be a JSON object")
    blockers = find_blockers(report, args.profile, applicability=policy,
                             transformers_proof=proof, expected_image=args.expected_image)
    if blockers:
        print("Unaccepted HIGH/CRITICAL image vulnerabilities:")
        for blocker in blockers:
            print(f"- {blocker}")
        return 1
    print(f"Trivy release policy passed for {args.profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
