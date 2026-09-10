from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from aniflive_tts import workstation_assets


class _Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _lock(tmp_path: Path, content: bytes, *, source_url: str | None = None) -> Path:
    record = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    if source_url is not None:
        record["source_url"] = source_url
    path = tmp_path / "lock.json"
    path.write_text(
        json.dumps(
            {
                "schema": workstation_assets.WORKSTATION_ASSET_LOCK_SCHEMA,
                "components": [
                    {
                        "id": "test-component",
                        "name": "Test Component",
                        "revision": "abc123",
                        "license": "Apache-2.0",
                        "runtime": "linux-worker",
                        "required": True,
                        "qualified": True,
                        "capability": "test",
                        "files": {"model.bin": record},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_component_download_is_explicit_pinned_and_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"immutable component"
    lock = _lock(
        tmp_path,
        content,
        source_url="https://huggingface.co/example/model/resolve/abc123/model.bin",
    )
    manager = workstation_assets.WorkstationAssetManager(
        tmp_path / "components", lock_path=lock
    )
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return _Response(content)

    monkeypatch.setattr(workstation_assets, "urlopen", fake_urlopen)
    progress = []
    result = manager.download_component(
        "test-component", progress=lambda completed, total: progress.append((completed, total))
    )

    assert result["ready"] is True
    assert calls == [
        ("https://huggingface.co/example/model/resolve/abc123/model.bin", 300)
    ]
    assert progress[-1] == (len(content), len(content))
    assert (manager.component_root("test-component") / "model.bin").read_bytes() == content


def test_component_download_rejects_bad_checksum_without_installing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _lock(
        tmp_path,
        b"expected",
        source_url="https://huggingface.co/example/model/resolve/abc123/model.bin",
    )
    manager = workstation_assets.WorkstationAssetManager(
        tmp_path / "components", lock_path=lock
    )
    monkeypatch.setattr(workstation_assets, "urlopen", lambda *_args, **_kwargs: _Response(b"tampered"))

    with pytest.raises(workstation_assets.WorkstationAssetError, match="fingerprint"):
        manager.download_component("test-component")
    assert not manager.component_root("test-component").exists()


@pytest.mark.parametrize(
    "source_url",
    [
        "http://huggingface.co/example/model/resolve/abc123/model.bin",
        "https://example.com/model/resolve/abc123/model.bin",
        "https://user:secret@huggingface.co/example/model/resolve/abc123/model.bin",
        "https://huggingface.co/example/model/blob/main/model.bin",
    ],
)
def test_component_lock_rejects_unapproved_download_sources(
    tmp_path: Path, source_url: str
) -> None:
    lock = _lock(tmp_path, b"content", source_url=source_url)
    with pytest.raises(workstation_assets.WorkstationAssetError, match="approved pinned source"):
        workstation_assets.load_asset_lock(lock)


def test_component_without_pinned_url_is_import_only(tmp_path: Path) -> None:
    manager = workstation_assets.WorkstationAssetManager(
        tmp_path / "components", lock_path=_lock(tmp_path, b"content")
    )
    status = manager.status()["components"][0]
    assert status["online_installable"] is False
    with pytest.raises(workstation_assets.WorkstationAssetError, match="built locally"):
        manager.download_component("test-component")


def test_component_bundle_round_trip_uses_exact_locked_inventory(tmp_path: Path) -> None:
    content = b"offline component"
    lock = _lock(tmp_path, content)
    source = workstation_assets.WorkstationAssetManager(
        tmp_path / "source-components", lock_path=lock
    )
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "model.bin").write_bytes(content)
    source.install_from_directory("test-component", staged)
    bundle = source.export_bundle(tmp_path / "components.zip")

    destination = workstation_assets.WorkstationAssetManager(
        tmp_path / "destination-components", lock_path=lock
    )
    result = destination.import_bundle(bundle)

    assert result["network_required"] is False
    assert result["installed"][0]["ready"] is True
    assert (
        destination.component_root("test-component") / "model.bin"
    ).read_bytes() == content


@pytest.mark.parametrize("unsafe_name", ["../escape", "/absolute"])
def test_component_bundle_rejects_unsafe_entries(
    tmp_path: Path, unsafe_name: str
) -> None:
    content = b"offline component"
    lock = _lock(tmp_path, content)
    manager = workstation_assets.WorkstationAssetManager(
        tmp_path / "components", lock_path=lock
    )
    bundle = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("bundle.json", "{}")
        archive.writestr(unsafe_name, b"bad")

    with pytest.raises(workstation_assets.WorkstationAssetError, match="unsafe"):
        manager.import_bundle(bundle)


def test_reinstall_preserves_live_files_and_repair_respects_busy_guard(tmp_path):
    content = b"verified-model"
    manager = workstation_assets.WorkstationAssetManager(
        tmp_path / "components", lock_path=_lock(tmp_path, content)
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(content)
    manager.install_from_directory("test-component", source)
    installed = manager.component_root("test-component") / "model.bin"
    inode = installed.stat().st_ino
    manager.replacement_allowed = lambda: False
    assert manager.install_from_directory("test-component", source)["ready"]
    assert installed.stat().st_ino == inode
    installed.write_bytes(b"damaged")
    with pytest.raises(workstation_assets.WorkstationAssetError, match="repair must wait"):
        manager.install_from_directory("test-component", source)
    assert installed.read_bytes() == b"damaged"
    manager.replacement_allowed = lambda: True
    assert manager.install_from_directory("test-component", source)["ready"]
