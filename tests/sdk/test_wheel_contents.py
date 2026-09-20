"""Archive-level checks for the native SDK wheel privacy boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

CHECKER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_sdk_wheel.py"
SPEC = importlib.util.spec_from_file_location("check_sdk_wheel", CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)

DIST_INFO = "aniflive_tts-1.5.0b1.dist-info"
NATIVE = "aniflive_tts/_native.abi3.so"
TAG = "cp310-abi3-manylinux_2_28_x86_64"

RUNTIME_REQUIREMENTS = (
    "fastapi>=0.115,<1",
    "httpx>=0.28,<1",
    "numpy>=1.26,<3",
    "pydantic>=2.10,<3",
    "uvicorn[standard]>=0.34,<1",
)


def _metadata(
    requirements: tuple[str, ...] = RUNTIME_REQUIREMENTS, *, runtime_extra: bool = True
) -> bytes:
    marker = '; extra == "runtime"' if runtime_extra else ""
    extras = "".join(
        f"Requires-Dist: {requirement}{marker}\n" for requirement in requirements
    )
    return (
        "Metadata-Version: 2.4\n"
        "Name: aniflive-tts\n"
        "Version: 1.5.0b1\n"
        "Requires-Python: >=3.10,<3.13\n"
        "Provides-Extra: runtime\n"
        f"{extras}\n"
    ).encode()


def _wheel(tag: str = TAG, *, pure: bool = False) -> bytes:
    return (
        "Wheel-Version: 1.0\n"
        "Generator: setuptools\n"
        f"Root-Is-Purelib: {'true' if pure else 'false'}\n"
        f"Tag: {tag}\n"
    ).encode("ascii")


def _files(
    *,
    tag: str = TAG,
    pure: bool = False,
    requirements: tuple[str, ...] = RUNTIME_REQUIREMENTS,
    runtime_extra: bool = True,
) -> dict[str, bytes]:
    prefix = f"{DIST_INFO}/"
    return {
        "aniflive_tts/__init__.py": b'__version__ = "1.5.0b1"\n',
        "aniflive_tts/cli.py": b"def main():\n    return 0\n",
        "aniflive_tts/_native.pyi": b"def version() -> str: ...\n",
        "aniflive_tts/py.typed": b"",
        "aniflive_tts/workstation_assets_lock.json": b"{}\n",
        "aniflive_tts/_native.abi3.so": b"native-extension",
        "aniflive_tts/sdk/__init__.py": b"from .types import SynthesisRequest\n",
        "aniflive_tts/sdk/errors.py": b"class AnifLiveTTSError(Exception): ...\n",
        "aniflive_tts/sdk/types.py": b"class SynthesisRequest: ...\n",
        "aniflive_tts/sdk/client.py": b"class Client: ...\n",
        "aniflive_tts/sdk/streams.py": b"class AudioStream: ...\n",
        "aniflive_tts/sdk/sessions.py": b"class SpeechSession: ...\n",
        f"{prefix}METADATA": _metadata(requirements, runtime_extra=runtime_extra),
        f"{prefix}WHEEL": _wheel(tag, pure=pure),
        f"{prefix}entry_points.txt": (
            b"[console_scripts]\naniflive-tts = aniflive_tts.cli:main\n"
        ),
        f"{prefix}licenses/LICENSE": b"project license\n",
        f"{prefix}licenses/SDK-NATIVE-THIRD-PARTY-NOTICES.txt": b"native notices\n",
        f"{prefix}licenses/SDK-NATIVE-DEPENDENCIES.json": b"{}\n",
    }


def _write_wheel(tmp_path: Path, files: dict[str, bytes]) -> Path:
    path = tmp_path / "fixture.whl"
    with ZipFile(path, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    return path


@pytest.mark.parametrize(
    ("tag", "native_name"),
    [
        (TAG, NATIVE),
        ("cp310-abi3-win_amd64", "aniflive_tts/_native.abi3.pyd"),
        ("cp310-abi3-macosx_11_0_arm64", NATIVE),
    ],
)
def test_accepts_supported_native_abi3_wheel_metadata(
    tmp_path: Path, tag: str, native_name: str
) -> None:
    files = _files(tag=tag)
    contents = files.pop(NATIVE)
    files[native_name] = contents

    result = CHECKER.check_wheel(_write_wheel(tmp_path, files))

    assert result["version"] == "1.5.0b1"
    assert result["tag"] == tag
    assert result["native_bytes"] == len(b"native-extension")


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("aniflive_tts/_native.abi3.so", "native extension"),
        ("aniflive_tts/_native.pyi", "Missing package files"),
        ("aniflive_tts/py.typed", "Missing package files"),
        ("aniflive_tts/workstation_assets_lock.json", "Missing package files"),
        ("aniflive_tts/cli.py", "Missing package files"),
        ("aniflive_tts/sdk/__init__.py", "Missing package files"),
        ("aniflive_tts/sdk/types.py", "Missing package files"),
        ("aniflive_tts/sdk/errors.py", "Missing package files"),
        ("aniflive_tts/sdk/client.py", "Missing package files"),
        ("aniflive_tts/sdk/streams.py", "Missing package files"),
        ("aniflive_tts/sdk/sessions.py", "Missing package files"),
        (
            f"{DIST_INFO}/licenses/SDK-NATIVE-THIRD-PARTY-NOTICES.txt",
            "native dependency notices",
        ),
        (
            f"{DIST_INFO}/licenses/SDK-NATIVE-DEPENDENCIES.json",
            "native dependency notices",
        ),
    ],
)
def test_rejects_missing_native_typing_or_package_files(
    tmp_path: Path, missing: str, message: str
) -> None:
    files = _files()
    del files[missing]

    with pytest.raises(ValueError, match=message):
        CHECKER.check_wheel(_write_wheel(tmp_path, files))


def test_rejects_non_abi3_wheel_tag(tmp_path: Path) -> None:
    files = _files(tag="cp312-cp312-manylinux_2_28_x86_64")

    with pytest.raises(ValueError, match="ABI3"):
        CHECKER.check_wheel(_write_wheel(tmp_path, files))


def test_rejects_pure_python_wheel(tmp_path: Path) -> None:
    files = _files(tag="py3-none-any", pure=True)

    with pytest.raises(ValueError, match="pure Python"):
        CHECKER.check_wheel(_write_wheel(tmp_path, files))


def test_rejects_runtime_dependency_without_runtime_extra_marker(tmp_path: Path) -> None:
    files = _files(requirements=("httpx>=0.28,<1",), runtime_extra=False)

    with pytest.raises(ValueError, match="runtime dependencies"):
        CHECKER.check_wheel(_write_wheel(tmp_path, files))


@pytest.mark.parametrize(
    "private_name",
    [
        "AGENTS.md",
        "ANIFLIVE_TTS_V1_5_BETA_RUST_PYTHON_SDK_CODEX_PLAN.md",
    ],
)
def test_rejects_private_instruction_documents(tmp_path: Path, private_name: str) -> None:
    files = _files()
    files[f"docs/private/{private_name}"] = b"private project instructions"

    with pytest.raises(ValueError, match="Private instruction"):
        CHECKER.check_wheel(_write_wheel(tmp_path, files))


def test_rejects_local_qualification_artifacts(tmp_path: Path) -> None:
    files = _files()
    files[".local-sdk/wheelhouse/qualification.json"] = b"{}"

    with pytest.raises(ValueError, match="qualification artifacts"):
        CHECKER.check_wheel(_write_wheel(tmp_path, files))
