from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

TARGETS = ("x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc", "aarch64-apple-darwin")
NOTICE_NAME = re.compile(r"^(LICENSE|LICENCE|COPYING|NOTICE)([.-].*)?$|^COPYRIGHT$", re.IGNORECASE)


def collect(root: Path) -> tuple[str, dict]:
    lock_bytes = (root / "Cargo.lock").read_bytes()
    lock = tomllib.loads(lock_bytes.decode("utf-8"))
    checksums = {(item["name"], item["version"], item.get("source")): item.get("checksum")
                 for item in lock["package"]}
    packages = {}
    for target in TARGETS:
        result = subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--locked", "--filter-platform", target],
            cwd=root, check=True, capture_output=True, text=True,
        )
        metadata = json.loads(result.stdout)
        by_id = {item["id"]: item for item in metadata["packages"]}
        nodes = {item["id"]: item for item in metadata["resolve"]["nodes"]}
        pending = [item["id"] for item in metadata["packages"]
                   if item["name"] == "aniflive-tts-python"]
        if len(pending) != 1:
            raise ValueError("Expected one Python binding crate")
        visited = set()
        while pending:
            key = pending.pop()
            if key in visited:
                continue
            visited.add(key)
            for dependency in nodes[key]["deps"]:
                if any(kind["kind"] != "dev" for kind in dependency["dep_kinds"]):
                    pending.append(dependency["pkg"])
        for key in visited:
            if by_id[key]["source"]:
                packages[key] = by_id[key]

    sections = [(
        "AnifLive-TTS native SDK dependency notices\n"
        "Generated from the locked desktop-target dependency graph.\n"
        "Includes build dependencies; this inventory does not claim every crate is linked "
        "on every platform.\n"
        "Upstream license expressions and notice texts are retained without relicensing.\n"
    )]
    inventory = []
    for package in sorted(packages.values(), key=lambda item: (item["name"], item["version"])):
        crate = Path(package["manifest_path"]).parent
        paths = sorted(path for path in crate.rglob("*")
                       if path.is_file() and NOTICE_NAME.fullmatch(path.name))
        if not package["license"] or not paths:
            raise ValueError("Missing license evidence for " + package["name"])
        notices = []
        sections.append("\n" + "=" * 72 + "\n" + package["name"] + " " + package["version"]
                        + "\nDeclared license: " + package["license"]
                        + "\nSource: " + (package["repository"] or package["source"]) + "\n")
        for path in paths:
            relative = path.relative_to(crate).as_posix()
            content = path.read_bytes()
            text = content.decode("utf-8")
            notices.append({"path": relative, "sha256": hashlib.sha256(content).hexdigest()})
            sections.append("\n--- " + relative + " ---\n" + text.rstrip() + "\n")
        inventory.append({
            "name": package["name"], "version": package["version"],
            "license": package["license"], "repository": package["repository"],
            "source": package["source"],
            "crate_checksum": checksums[(package["name"], package["version"], package["source"])],
            "notices": notices,
        })
    document = {
        "schema": 1, "targets": TARGETS,
        "cargo_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "packages": inventory,
    }
    return "".join(sections), document


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect locked native SDK license notices (Python 3.11+ build tool)")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text, inventory = collect(args.root)
    outputs = {
        args.root / "licenses/SDK-NATIVE-THIRD-PARTY-NOTICES.txt": text,
        args.root / "licenses/SDK-NATIVE-DEPENDENCIES.json":
            json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
    }
    for path, content in outputs.items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise SystemExit("Native dependency notices need regeneration: " + path.name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    print("Native dependency notices verified:", len(inventory["packages"]), "packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
