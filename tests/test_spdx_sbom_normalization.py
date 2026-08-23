from __future__ import annotations

import pytest

from scripts.normalize_spdx_sbom import normalize_document


def _document(license_id: str) -> dict[str, object]:
    return {
        "spdxVersion": "SPDX-2.3",
        "hasExtractedLicensingInfos": [
            {
                "licenseId": "LicenseRef-NVIDIA-Proprietary-Software",
                "extractedText": "NOASSERTION",
            }
        ],
        "packages": [
            {
                "SPDXID": "SPDXRef-Package",
                "name": "sample",
                "licenseDeclared": license_id,
                "filesAnalyzed": True,
                "packageVerificationCode": {"packageVerificationCodeValue": "abc"},
            },
            {"SPDXID": "SPDXRef-Dependency", "name": "dependency"},
        ],
        "files": [{"SPDXID": "SPDXRef-File", "fileName": "/sample.py"}],
        "relationships": [
            {
                "spdxElementId": "SPDXRef-Package",
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": "SPDXRef-File",
            },
            {
                "spdxElementId": "SPDXRef-Package",
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": "SPDXRef-Dependency",
            },
        ],
    }


def test_normalizes_known_nvidia_alias() -> None:
    document = normalize_document(_document("LicenseRef-NVIDIA-Proprietary"))

    assert document["packages"][0]["licenseDeclared"] == "LicenseRef-NVIDIA-Proprietary-Software"


def test_normalizes_nvidia_software_license_alias() -> None:
    document = normalize_document(_document("LicenseRef-NVIDIA-SOFTWARE-LICENSE"))

    assert document["packages"][0]["licenseDeclared"] == "LicenseRef-NVIDIA-Proprietary-Software"


def test_preserves_defined_license_reference() -> None:
    document = normalize_document(_document("LicenseRef-NVIDIA-Proprietary-Software"))

    assert document["packages"][0]["licenseDeclared"] == "LicenseRef-NVIDIA-Proprietary-Software"


def test_emits_a_package_level_inventory() -> None:
    document = normalize_document(_document("LicenseRef-NVIDIA-Proprietary-Software"))

    assert "files" not in document
    assert document["packages"][0]["filesAnalyzed"] is False
    assert "packageVerificationCode" not in document["packages"][0]
    assert len(document["relationships"]) == 1
    assert document["relationships"][0]["relationshipType"] == "DEPENDS_ON"


def test_rejects_unknown_undefined_license_reference() -> None:
    with pytest.raises(ValueError, match="LicenseRef-Unknown"):
        normalize_document(_document("LicenseRef-Unknown"))
