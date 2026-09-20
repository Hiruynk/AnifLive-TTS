from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

PRIVATE_NAMES = {
    "agents.md",
    "aniflive_tts_v1_5_beta_rust_python_sdk_codex_plan.md",
}
REQUIRED_FILES = {
    "aniflive_tts/__init__.py",
    "aniflive_tts/cli.py",
    "aniflive_tts/_native.pyi",
    "aniflive_tts/py.typed",
    "aniflive_tts/sdk/__init__.py",
    "aniflive_tts/sdk/types.py",
    "aniflive_tts/sdk/client.py",
    "aniflive_tts/sdk/streams.py",
    "aniflive_tts/sdk/sessions.py",
    "aniflive_tts/sdk/errors.py",
    "aniflive_tts/workstation_assets_lock.json",
}


def check_wheel(path: Path) -> dict[str, str | int]:
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = REQUIRED_FILES - names
        if missing:
            raise ValueError("Missing package files: " + ", ".join(sorted(missing)))
        if any(Path(name).name.lower() in PRIVATE_NAMES for name in names):
            raise ValueError("Private instruction document in wheel")
        if any(".local-sdk" in Path(name).parts for name in names):
            raise ValueError("Local qualification artifacts in wheel")
        extensions = [name for name in names if name.startswith("aniflive_tts/_native.")
                      and name.endswith((".so", ".pyd"))]
        if len(extensions) != 1:
            raise ValueError("Expected exactly one native extension")
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError("Expected one distribution metadata record")
        metadata_path = metadata_names[0]
        prefix = metadata_path.removesuffix("METADATA")
        for license_name in ("SDK-NATIVE-THIRD-PARTY-NOTICES.txt", "SDK-NATIVE-DEPENDENCIES.json"):
            if not any(name.startswith(prefix + "licenses/") and name.endswith("/" + license_name)
                       for name in names):
                raise ValueError("Missing native dependency notices: " + license_name)
        metadata = Parser().parsestr(archive.read(metadata_path).decode("utf-8"))
        wheel = Parser().parsestr(archive.read(prefix + "WHEEL").decode("utf-8"))
        if metadata["Name"] != "aniflive-tts":
            raise ValueError("Unexpected distribution name")
        if wheel["Root-Is-Purelib"] != "false":
            raise ValueError("Native SDK cannot be a pure Python wheel")
        tags = wheel.get_all("Tag") or []
        if not tags or any(not tag.startswith("cp310-abi3-") or tag.endswith("-any")
                           for tag in tags):
            raise ValueError("Expected platform-specific CPython 3.10+ ABI3 tags")
        for requirement in metadata.get_all("Requires-Dist") or []:
            _, separator, marker = requirement.partition(";")
            if not separator or marker.strip() != 'extra == "runtime"':
                raise ValueError("Bare SDK must not install Python runtime dependencies")
        entry_points = archive.read(prefix + "entry_points.txt").decode("utf-8")
        if "aniflive-tts = aniflive_tts.cli:main" not in entry_points:
            raise ValueError("Missing runtime CLI entry point")
        return {"version": metadata["Version"], "tag": ",".join(tags),
                "files": len(names), "native_bytes": archive.getinfo(extensions[0]).file_size}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SDK wheel contents and install boundary")
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    result = check_wheel(args.wheel)
    print("SDK wheel verified:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
