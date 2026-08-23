#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from pathlib import PurePosixPath


GPT_SOVITS_REPO = "lj1995/GPT-SoVITS"
GPT_SOVITS_REVISION = "336b2ec4e8d4ac74740798dd40af44e74659ecaf"
GPT_SOVITS_FILES = (
    "chinese-hubert-base/config.json",
    "chinese-hubert-base/preprocessor_config.json",
    "chinese-hubert-base/pytorch_model.bin",
    "chinese-roberta-wwm-ext-large/config.json",
    "chinese-roberta-wwm-ext-large/pytorch_model.bin",
    "chinese-roberta-wwm-ext-large/tokenizer.json",
    "sv/pretrained_eres2netv2w24s4ep4.ckpt",
)
GPT_SOVITS_SHA256 = {
    "chinese-hubert-base/config.json": (
        "c3e5060a1277e0f078cc6be9da4528a605dba6ece93018981fe2c820e5c7b103"
    ),
    "chinese-hubert-base/preprocessor_config.json": (
        "dcd684124d06722947939d41ea6ae58dbf10968c60a11a29f23ddc602c64a29b"
    ),
    "chinese-hubert-base/pytorch_model.bin": (
        "24164f129c66499d1346e2aa55f183250c223161ec2770c0da3d3b08cf432d3c"
    ),
    "chinese-roberta-wwm-ext-large/config.json": (
        "3d57de2fd7e80d0e5c8ff194f0bbb6baa10df7e43fc262a0cc71298a78b0a3e5"
    ),
    "chinese-roberta-wwm-ext-large/pytorch_model.bin": (
        "e53a693acc59ace251d143d068096ae0d7b79e4b1b503fa84c9dcf576448c1d8"
    ),
    "chinese-roberta-wwm-ext-large/tokenizer.json": (
        "173796956820ea27bd14f76bf28162607ff4254807e2948253eb5b46f5bb643b"
    ),
    "sv/pretrained_eres2netv2w24s4ep4.ckpt": (
        "4f5a0bf73c61eb41b174e1bb54e7ee3c83233892be8e0af1f187024e8e581a35"
    ),
}
ASSET_LOCK_PATH = Path(__file__).with_name("shared_assets_lock.json")
ASSET_LOCK = json.loads(ASSET_LOCK_PATH.read_text(encoding="utf-8"))
FASTTEXT_URL = str(ASSET_LOCK["fasttext"]["url"])
FASTTEXT_SHA256 = str(ASSET_LOCK["fasttext"]["sha256"])
NLTK_REPOSITORY = str(ASSET_LOCK["nltk"]["repository"])
NLTK_REVISION = str(ASSET_LOCK["nltk"]["revision"])
NLTK_PACKAGES = dict(ASSET_LOCK["nltk"]["packages"])


def _digest(path: Path, algorithm: str = "sha256") -> str:
    checksum = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _download_fasttext(output: Path) -> Path:
    target = output / "fast_langdetect" / "lid.176.bin"
    if target.is_file() and _digest(target) == FASTTEXT_SHA256:
        print(f"[shared-assets] reuse {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".bin.part")
    print(f"[shared-assets] download {FASTTEXT_URL}")
    with urllib.request.urlopen(FASTTEXT_URL, timeout=120) as response:
        with temporary.open("wb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
    if _digest(temporary) != FASTTEXT_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("fastText lid.176.bin checksum mismatch")
    os.replace(temporary, target)
    return target


def _download_gpt_sovits(output: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download

    downloaded: list[Path] = []
    for filename in GPT_SOVITS_FILES:
        target = output / filename
        expected = GPT_SOVITS_SHA256[filename]
        if target.is_file() and _digest(target) == expected:
            print(f"[shared-assets] reuse {target}")
        else:
            target.unlink(missing_ok=True)
            print(f"[shared-assets] download {GPT_SOVITS_REPO}/{filename}")
            hf_hub_download(
                repo_id=GPT_SOVITS_REPO,
                filename=filename,
                revision=GPT_SOVITS_REVISION,
                local_dir=output,
            )
            if not target.is_file() or _digest(target) != expected:
                target.unlink(missing_ok=True)
                raise RuntimeError(f"GPT-SoVITS shared asset checksum mismatch: {filename}")
        downloaded.append(target)
    return downloaded


def _tree_digest(path: Path) -> str:
    checksum = hashlib.sha256()
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        checksum.update(candidate.relative_to(path).as_posix().encode("utf-8"))
        checksum.update(b"\0")
        checksum.update(_digest(candidate).encode("ascii"))
        checksum.update(b"\n")
    return checksum.hexdigest()


def _download_nltk(output: Path) -> tuple[list[Path], dict[str, str]]:
    nltk_dir = output / "nltk_data"
    nltk_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    tree_hashes: dict[str, str] = {}
    for package, metadata in NLTK_PACKAGES.items():
        subdir = str(metadata["subdir"])
        zip_sha256 = str(metadata["zip_sha256"])
        tree_sha256 = str(metadata["tree_sha256"])
        target = nltk_dir / subdir / package
        if target.is_dir() and _tree_digest(target) == tree_sha256:
            print(f"[shared-assets] reuse {target}")
        else:
            if target.exists():
                shutil.rmtree(target)
            archive = nltk_dir / subdir / f"{package}.zip"
            archive.parent.mkdir(parents=True, exist_ok=True)
            temporary_archive = archive.with_suffix(".zip.part")
            temporary_archive.unlink(missing_ok=True)
            url = (
                f"https://raw.githubusercontent.com/{NLTK_REPOSITORY}/"
                f"{NLTK_REVISION}/packages/{subdir}/{package}.zip"
            )
            print(f"[shared-assets] download {url}")
            with urllib.request.urlopen(url, timeout=120) as response:
                with temporary_archive.open("wb") as handle:
                    while block := response.read(1024 * 1024):
                        handle.write(block)
            if _digest(temporary_archive) != zip_sha256:
                temporary_archive.unlink(missing_ok=True)
                raise RuntimeError(f"NLTK archive checksum mismatch: {package}")
            os.replace(temporary_archive, archive)

            temporary_target = target.with_name(f".{package}.extract")
            if temporary_target.exists():
                shutil.rmtree(temporary_target)
            temporary_target.mkdir(parents=True)
            with zipfile.ZipFile(archive) as bundle:
                prefix = PurePosixPath(package)
                for member in bundle.infolist():
                    member_path = PurePosixPath(member.filename)
                    if member.is_dir():
                        continue
                    try:
                        relative = member_path.relative_to(prefix)
                    except ValueError as exc:
                        raise RuntimeError(
                            f"Unexpected NLTK archive member: {member.filename}"
                        ) from exc
                    if not relative.parts or ".." in relative.parts:
                        raise RuntimeError(
                            f"Unsafe NLTK archive member: {member.filename}"
                        )
                    destination = temporary_target.joinpath(*relative.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as source, destination.open("wb") as sink:
                        shutil.copyfileobj(source, sink)
            if _tree_digest(temporary_target) != tree_sha256:
                shutil.rmtree(temporary_target)
                raise RuntimeError(f"NLTK content checksum mismatch: {package}")
            os.replace(temporary_target, target)
        if not target.is_dir() or _tree_digest(target) != tree_sha256:
            raise RuntimeError(f"NLTK package is incomplete or modified: {package}")
        tree_hashes[package] = tree_sha256
        paths.append(target)
    return paths, tree_hashes


def main() -> int:
    parser = argparse.ArgumentParser(description="Download immutable AnifLive-TTS shared assets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept-third-party-licenses", action="store_true")
    args = parser.parse_args()
    if not args.accept_third_party_licenses:
        parser.error(
            "--accept-third-party-licenses is required; review THIRD_PARTY_NOTICES.md first"
        )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = _download_gpt_sovits(output)
    files.append(_download_fasttext(output))
    _, nltk_hashes = _download_nltk(output)
    manifest = {
        "schema": 1,
        "gpt_sovits": {
            "repo": GPT_SOVITS_REPO,
            "revision": GPT_SOVITS_REVISION,
        },
        "asset_lock_sha256": _digest(ASSET_LOCK_PATH),
        "files": {
            str(path.relative_to(output)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": _digest(path),
            }
            for path in files
        },
        "nltk_trees": nltk_hashes,
        "licenses": "See THIRD_PARTY_NOTICES.md and licenses/",
    }
    manifest_path = output / "shared-assets.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[shared-assets] ready: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
