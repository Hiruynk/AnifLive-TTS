"""Normalize an image SPDX document to a strict package-level inventory."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


LICENSE_EXPRESSION_KEYS = {
    "licenseConcluded",
    "licenseDeclared",
    "licenseInfoFromFiles",
    "licenseInfoInFiles",
    "licenseInfoInSnippets",
}

# Some NVIDIA Python packages use either label for the same proprietary
# software license. Scanners can emit the shorter alias without defining it.
LICENSE_ALIASES = {
    "LicenseRef-NVIDIA-Proprietary": "LicenseRef-NVIDIA-Proprietary-Software",
    "LicenseRef-NVIDIA-SOFTWARE-LICENSE": "LicenseRef-NVIDIA-Proprietary-Software",
}

LICENSE_REF_PATTERN = re.compile(r"LicenseRef-[A-Za-z0-9.-]+")


def _rewrite_expression(value: Any, aliases: dict[str, str]) -> Any:
    if isinstance(value, str):
        return LICENSE_REF_PATTERN.sub(lambda match: aliases.get(match.group(0), match.group(0)), value)
    if isinstance(value, list):
        return [_rewrite_expression(item, aliases) for item in value]
    return value


def _walk_and_rewrite(value: Any, aliases: dict[str, str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in LICENSE_EXPRESSION_KEYS:
                value[key] = _rewrite_expression(item, aliases)
            else:
                _walk_and_rewrite(item, aliases)
    elif isinstance(value, list):
        for item in value:
            _walk_and_rewrite(item, aliases)


def _used_license_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in LICENSE_EXPRESSION_KEYS:
                strings = item if isinstance(item, list) else [item]
                for expression in strings:
                    if isinstance(expression, str):
                        refs.update(LICENSE_REF_PATTERN.findall(expression))
            else:
                refs.update(_used_license_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.update(_used_license_refs(item))
    return refs


def _remove_file_inventory(document: dict[str, Any]) -> None:
    """Keep package evidence while removing non-portable file checksum records."""
    removed_ids = {
        entry.get("SPDXID")
        for section in ("files", "snippets")
        for entry in document.pop(section, [])
        if isinstance(entry, dict) and isinstance(entry.get("SPDXID"), str)
    }
    document["relationships"] = [
        relationship
        for relationship in document.get("relationships", [])
        if relationship.get("spdxElementId") not in removed_ids
        and relationship.get("relatedSpdxElement") not in removed_ids
    ]
    for package in document.get("packages", []):
        package["filesAnalyzed"] = False
        package.pop("packageVerificationCode", None)


def normalize_document(document: dict[str, Any]) -> dict[str, Any]:
    _remove_file_inventory(document)
    extracted = document.get("hasExtractedLicensingInfos", [])
    defined = {
        entry.get("licenseId")
        for entry in extracted
        if isinstance(entry, dict) and isinstance(entry.get("licenseId"), str)
    }

    usable_aliases = {
        source: target
        for source, target in LICENSE_ALIASES.items()
        if source not in defined and target in defined
    }
    _walk_and_rewrite(document, usable_aliases)

    missing = sorted(_used_license_refs(document) - defined)
    if missing:
        raise ValueError(f"SPDX document contains undefined LicenseRef values: {', '.join(missing)}")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="SPDX JSON document to normalize in place")
    args = parser.parse_args()

    document = json.loads(args.path.read_text(encoding="utf-8"))
    normalize_document(document)
    args.path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Normalized and checked SPDX LicenseRef values in {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
