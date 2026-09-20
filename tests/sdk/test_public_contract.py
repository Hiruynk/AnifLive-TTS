"""Public Python SDK export and protocol-ownership contracts."""

from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PACKAGE = REPO_ROOT / "src" / "aniflive_tts"


def test_sdk_exports_are_reexported_by_the_package_root() -> None:
    import aniflive_tts
    import aniflive_tts.sdk as sdk

    sdk_names = set(sdk.__all__)
    package_names = set(aniflive_tts.__all__)

    assert len(sdk_names) == len(sdk.__all__)
    assert sdk_names <= package_names
    assert "__version__" in package_names
    assert not any(name.startswith("Native") for name in package_names)
    for name in sdk_names:
        assert getattr(aniflive_tts, name) is getattr(sdk, name)


def test_python_facade_leaves_versioned_routes_in_the_native_client() -> None:
    facade_modules = ("client.py", "streams.py", "sessions.py")
    for module in facade_modules:
        source = (SOURCE_PACKAGE / "sdk" / module).read_text(encoding="utf-8")
        assert "/v1/" not in source, f"route literal leaked into {module}"


def test_python_and_rust_package_versions_match() -> None:
    import aniflive_tts

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    cargo = (REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8")
    project_section = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", pyproject)
    assert project_section is not None
    project_version = re.search(r'(?m)^version\s*=\s*"([^"]+)"', project_section.group(1))
    rust_version = re.search(r'(?m)^version\s*=\s*"([^"]+)"', cargo)
    assert project_version is not None
    assert rust_version is not None

    cargo_python_version = re.sub(
        r"^(\d+\.\d+\.\d+)-beta\.(\d+)$",
        r"\1b\2",
        rust_version.group(1),
    )
    assert aniflive_tts.__version__ == project_version.group(1)
    assert cargo_python_version == project_version.group(1)


def test_server_advertised_version_matches_package_without_importing_runtime() -> None:
    import aniflive_tts

    tree = ast.parse((SOURCE_PACKAGE / "service.py").read_text())
    versions = [ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "SERVICE_VERSION"
                    for target in node.targets)]
    assert versions == [aniflive_tts.__version__]
    tree = ast.parse((SOURCE_PACKAGE / "webui.py").read_text())
    versions = [ast.literal_eval(keyword.value) for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "FastAPI" for keyword in node.keywords
                if keyword.arg == "version"]
    assert versions == [aniflive_tts.__version__]
