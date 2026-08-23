from __future__ import annotations

import hashlib
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETUP = runpy.run_path(str(ROOT / "scripts" / "setup_shared_assets.py"))


def test_every_pinned_gpt_sovits_asset_has_a_sha256() -> None:
    files = SETUP["GPT_SOVITS_FILES"]
    hashes = SETUP["GPT_SOVITS_SHA256"]
    assert set(files) == set(hashes)
    assert all(len(value) == 64 for value in hashes.values())
    assert all(int(value, 16) >= 0 for value in hashes.values())


def test_language_assets_use_immutable_sources_and_sha256() -> None:
    assert len(SETUP["FASTTEXT_SHA256"]) == 64
    assert len(SETUP["NLTK_REVISION"]) == 40
    for metadata in SETUP["NLTK_PACKAGES"].values():
        assert len(metadata["zip_sha256"]) == 64
        assert len(metadata["tree_sha256"]) == 64


def test_nltk_tree_digest_detects_content_changes(tmp_path: Path) -> None:
    package = tmp_path / "taggers" / "example"
    package.mkdir(parents=True)
    resource = package / "weights.json"
    resource.write_text("first\n", encoding="utf-8")
    first = SETUP["_tree_digest"](package)

    resource.write_text("second\n", encoding="utf-8")
    second = SETUP["_tree_digest"](package)

    assert first != second
    assert first != hashlib.sha256(b"").hexdigest()
