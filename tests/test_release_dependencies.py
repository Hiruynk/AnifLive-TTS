from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_requirements_exclude_unused_copyleft_packages() -> None:
    requirements = (ROOT / "requirements" / "base.txt").read_text(encoding="utf-8").lower()
    assert "distance" not in requirements
    assert "eunjeon" not in requirements
    assert "chardet" not in requirements
    assert "einx" not in requirements
    assert "x-transformers" not in requirements


def test_security_patched_dependency_floors_are_pinned() -> None:
    requirements = (ROOT / "requirements" / "base.txt").read_text(encoding="utf-8")
    cu128 = (ROOT / "requirements" / "torch-cu128.txt").read_text(encoding="utf-8")
    assert "onnx==1.22.0" in requirements
    assert "fastapi==0.141.1" in requirements
    assert "starlette==1.3.1" in requirements
    assert "transformers==5.5.0" in requirements
    assert "torch==2.10.0+cu128" in cu128


def test_canonical_transformer_loads_are_local_only() -> None:
    sources = (
        ROOT / "minimal_inference" / "export_onnx.py",
        ROOT / "minimal_inference" / "run_trt_inference.py",
        ROOT
        / "minimal_inference"
        / "GPT_SoVITS"
        / "feature_extractor"
        / "cnhubert.py",
    )
    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert "local_files_only=True" in text
        assert "trust_remote_code=False" in text


def test_container_base_images_and_fasttext_asset_are_immutable() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    compose_cu121 = (ROOT / "docker-compose.cu121.yml").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "container.yml").read_text(
        encoding="utf-8"
    )
    cu128_digest = "sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9"
    cu121_digest = "sha256:810756cab1c28ce693499a5c2ebb66f6d10a61d026998c8606bad449643a4c49"
    asset_lock = json.loads(
        (ROOT / "scripts" / "shared_assets_lock.json").read_text(encoding="utf-8")
    )
    fasttext_sha256 = (
        "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"
    )
    assert cu128_digest in dockerfile
    assert cu128_digest in compose
    assert cu128_digest in workflow
    assert cu121_digest in compose_cu121
    assert cu121_digest in workflow
    assert asset_lock["fasttext"]["sha256"] == fasttext_sha256
    assert "shared_assets_lock.json" in dockerfile
    assert "python -m nltk.downloader" not in dockerfile


def test_known_checkpoint_loader_risk_is_documented() -> None:
    policy = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "GHSA-53q9-r3pm-6pq6" in policy
    assert "GHSA-63cw-57p8-fm3p" in policy
    assert "cu121" in policy
    assert "trusted-administrator operation" in policy


def test_build_and_test_tool_versions_are_security_pinned() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    dev = (ROOT / "requirements" / "dev.txt").read_text(encoding="utf-8")
    assert 'setuptools==84.0.0' in pyproject
    assert 'wheel==0.48.0' in pyproject
    assert "pip==26.2.1 setuptools==84.0.0 wheel==0.48.0" in dockerfile
    assert "pytest==9.1.1" in dev
    assert "httpx==0.28.1" in dev
    assert "httpx2" not in dev
    assert "spdx-tools==0.8.3" in dev


def test_container_workflow_scans_built_image_digest() -> None:
    workflow = (ROOT / ".github" / "workflows" / "container.yml").read_text(
        encoding="utf-8"
    )
    assert "aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25" in workflow
    assert "${{ env.IMAGE }}@${{ steps.build.outputs.digest }}" in workflow
    assert "scripts/check_trivy_report.py" in workflow
    assert "TRIVY-AnifLive-TTS-v1.0.0-${{ matrix.tag }}.json" in workflow
    assert "spdx-tools==0.8.3" in workflow
    assert 'pyspdxtools -i "${output}"' in workflow


def test_english_g2p_is_vendored_without_distance_dependency() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    english = (
        ROOT / "minimal_inference" / "GPT_SoVITS" / "text" / "english.py"
    ).read_text(encoding="utf-8")
    vendored_g2p = (
        ROOT / "minimal_inference" / "GPT_SoVITS" / "text" / "g2p_en" / "g2p.py"
    ).read_text(encoding="utf-8")
    assert "pip install --no-deps g2p-en" not in dockerfile
    assert 'find_spec("distance") is None' in dockerfile
    assert "from text.g2p_en import G2p" in english
    assert "nltk.download" not in vendored_g2p
    assert "import distance" not in vendored_g2p


def test_windows_korean_frontend_cannot_install_dependencies_at_runtime() -> None:
    requirements = (ROOT / "requirements" / "base.txt").read_text(encoding="utf-8")
    korean = (
        ROOT / "minimal_inference" / "GPT_SoVITS" / "text" / "korean.py"
    ).read_text(encoding="utf-8")
    assert "from eunjeon" not in korean
    assert 'find_spec("eunjeon")' not in korean
    assert "pip install" not in korean
    assert "mecab-python3==1.0.12" in requirements
    assert 'python-mecab-ko==1.3.7; platform_system != "Windows"' in requirements
    assert "mecab-python3 and python-mecab-ko-dic are required" in korean
    assert "class _WindowsMecab" in korean
