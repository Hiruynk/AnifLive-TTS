from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zipfile


WORKSTATION_ASSET_LOCK_SCHEMA = "aniflive-workstation-assets-lock-v1"
WORKSTATION_ASSET_BUNDLE_SCHEMA = "aniflive-workstation-assets-bundle-v1"
COMPONENT_STATES = frozenset({"ready", "missing", "invalid", "optional"})


class WorkstationAssetError(ValueError):
    """Raised when a managed workstation component fails its pinned contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WorkstationAssetError(f"Could not read component metadata: {path.name}") from error
    if not isinstance(value, dict):
        raise WorkstationAssetError("Component metadata must be a JSON object")
    return value


def _component_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 80
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in value)
    ):
        raise WorkstationAssetError("Component ID is malformed")
    return value


def _relative_file(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise WorkstationAssetError("Component file path is malformed")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        raise WorkstationAssetError("Component file path is unsafe")
    return path


@dataclass(frozen=True)
class ComponentFile:
    path: PurePosixPath
    sha256: str
    bytes: int
    source_url: str | None = None


@dataclass(frozen=True)
class ComponentContract:
    component_id: str
    name: str
    revision: str
    license: str
    runtime: str
    required: bool
    qualified: bool
    capability: str
    files: tuple[ComponentFile, ...]

    def as_dict(self) -> dict[str, Any]:
        files: dict[str, dict[str, Any]] = {}
        for item in self.files:
            record: dict[str, Any] = {"sha256": item.sha256, "bytes": item.bytes}
            if item.source_url is not None:
                record["source_url"] = item.source_url
            files[item.path.as_posix()] = record
        return {
            "id": self.component_id,
            "name": self.name,
            "revision": self.revision,
            "license": self.license,
            "runtime": self.runtime,
            "required": self.required,
            "qualified": self.qualified,
            "capability": self.capability,
            "files": files,
        }


def _source_url(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 2048:
        raise WorkstationAssetError("Component source URL is malformed")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "huggingface.co"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or not parsed.path.startswith("/")
        or "/resolve/" not in parsed.path
    ):
        raise WorkstationAssetError("Component source URL is not an approved pinned source")
    return value


def load_asset_lock(path: Path) -> dict[str, ComponentContract]:
    document = _strict_json(Path(path))
    if document.get("schema") != WORKSTATION_ASSET_LOCK_SCHEMA:
        raise WorkstationAssetError("Workstation component lock schema is unsupported")
    values = document.get("components")
    if not isinstance(values, list) or not values:
        raise WorkstationAssetError("Workstation component lock has no components")
    result: dict[str, ComponentContract] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise WorkstationAssetError("Component contract must be an object")
        component_id = _component_id(value.get("id"))
        if component_id in result:
            raise WorkstationAssetError("Component lock contains duplicate IDs")
        files_value = value.get("files")
        if not isinstance(files_value, Mapping):
            raise WorkstationAssetError(f"{component_id} files must be an object")
        files: list[ComponentFile] = []
        for raw_path, record in files_value.items():
            if not isinstance(record, Mapping):
                raise WorkstationAssetError(f"{component_id} file contract is malformed")
            relative = _relative_file(raw_path)
            sha256 = record.get("sha256")
            byte_count = record.get("bytes")
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
                or not isinstance(byte_count, int)
                or isinstance(byte_count, bool)
                or byte_count < 1
            ):
                raise WorkstationAssetError(f"{component_id} file fingerprint is malformed")
            files.append(ComponentFile(
                relative,
                sha256,
                byte_count,
                _source_url(record.get("source_url")),
            ))
        required = value.get("required", True)
        qualified = value.get("qualified", True)
        if not isinstance(required, bool) or not isinstance(qualified, bool):
            raise WorkstationAssetError(f"{component_id} readiness flags are malformed")
        if qualified and not files:
            raise WorkstationAssetError(
                f"{component_id} cannot be qualified without pinned files"
            )
        text_fields: dict[str, str] = {}
        for field in ("name", "revision", "license", "runtime", "capability"):
            field_value = value.get(field)
            if not isinstance(field_value, str) or not field_value.strip() or len(field_value) > 256:
                raise WorkstationAssetError(f"{component_id} {field} is malformed")
            text_fields[field] = field_value.strip()
        result[component_id] = ComponentContract(
            component_id,
            text_fields["name"],
            text_fields["revision"],
            text_fields["license"],
            text_fields["runtime"],
            required,
            qualified,
            text_fields["capability"],
            tuple(sorted(files, key=lambda item: item.path.as_posix())),
        )
    return result


class WorkstationAssetManager:
    """Pinned component inventory with explicit, offline-safe installation.

    Normal workers only read validated component directories. Network access is exposed
    solely by the explicit Studio setup action; inference and queued neural jobs never
    invoke it. Offline installations use :meth:`import_bundle` with the same inventory
    and checksum contract.
    """

    def __init__(self, root: Path, *, lock_path: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_path = Path(lock_path).expanduser().resolve(strict=True)
        self.contracts = load_asset_lock(self.lock_path)
        self._status_snapshot_lock = threading.Lock()
        self._status_snapshot = None
        self._status_snapshot_at = 0.0
        self._status_snapshot_stamp = None
        self._install_lock = threading.RLock()
        self.replacement_allowed: Callable[[], bool] = lambda: True

    def component_root(self, component_id: str) -> Path:
        return self.root / _component_id(component_id)

    def _validate_component(self, contract: ComponentContract, root: Path) -> tuple[bool, str | None]:
        if not contract.qualified:
            return False, "component-lock-awaits-qualified-fingerprints"
        if not root.is_dir() or root.is_symlink():
            return False, "component-directory-missing"
        actual_files: set[str] = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                return False, "symbolic-links-are-not-allowed"
            if path.is_file():
                actual_files.add(path.relative_to(root).as_posix())
        expected_files = {item.path.as_posix() for item in contract.files}
        if actual_files != expected_files:
            return False, "component-file-inventory-mismatch"
        for item in contract.files:
            path = root.joinpath(*item.path.parts)
            try:
                if path.stat().st_size != item.bytes or _sha256_file(path) != item.sha256:
                    return False, f"checksum-mismatch:{item.path.as_posix()}"
            except OSError:
                return False, f"unreadable:{item.path.as_posix()}"
        return True, None

    def status_snapshot(self) -> dict[str, Any]:
        """Coalesce display-only integrity checks; execution still calls status()."""
        with self._status_snapshot_lock:
            stamp = self.root.stat().st_mtime_ns
            now = time.monotonic()
            cached = (
                self._status_snapshot is not None
                and now - self._status_snapshot_at < 30
                and stamp == self._status_snapshot_stamp
            )
            if not cached:
                self._status_snapshot = self.status()
                self._status_snapshot_at = time.monotonic()
                self._status_snapshot_stamp = stamp
            result = copy.deepcopy(self._status_snapshot)
            result["display_status"] = {
                "cached": cached,
                "verification_age_seconds": round(time.monotonic() - self._status_snapshot_at, 3),
                "maximum_cache_seconds": 30,
                "execution_requires_fresh_verification": True,
            }
            return result

    def status(self, component_id: str | None = None) -> dict[str, Any]:
        selected: Sequence[ComponentContract]
        if component_id is None:
            selected = tuple(self.contracts.values())
        else:
            try:
                selected = (self.contracts[_component_id(component_id)],)
            except KeyError as error:
                raise WorkstationAssetError("Component is not present in the lock") from error
        records = []
        for contract in selected:
            component_root = self.component_root(contract.component_id)
            ready, reason = self._validate_component(contract, component_root)
            state = (
                "ready"
                if ready
                else "invalid"
                if component_root.exists() and contract.qualified
                else "missing"
                if contract.required
                else "optional"
            )
            records.append(
                {
                    **contract.as_dict(),
                    "state": state,
                    "ready": ready,
                    "reason": reason,
                    "root": str(self.component_root(contract.component_id)) if ready else None,
                    "network_required": bool(
                        not ready
                        and contract.qualified
                        and contract.files
                        and all(item.source_url is not None for item in contract.files)
                    ),
                    "online_installable": bool(
                        contract.qualified
                        and contract.files
                        and all(item.source_url is not None for item in contract.files)
                    ),
                }
            )
        return {
            "schema": WORKSTATION_ASSET_LOCK_SCHEMA,
            "components": records,
            "ready": all(record["ready"] or not record["required"] for record in records),
        }

    def install_from_directory(self, component_id: str, source: Path) -> dict[str, Any]:
        with self._install_lock:
            result = self._install_from_directory(component_id, source)
        with self._status_snapshot_lock:
            self._status_snapshot = None
        return result

    def _install_from_directory(self, component_id: str, source: Path) -> dict[str, Any]:
        component_id = _component_id(component_id)
        try:
            contract = self.contracts[component_id]
        except KeyError as error:
            raise WorkstationAssetError("Component is not present in the lock") from error
        if not contract.qualified:
            raise WorkstationAssetError(
                "Component cannot be installed until its revision, inventory and checksums are qualified"
            )
        source = Path(source).expanduser().resolve(strict=True)
        ready, reason = self._validate_component(contract, source)
        if not ready:
            raise WorkstationAssetError(
                f"Staged component does not match the lock: {reason}"
            )
        destination = self.component_root(component_id)
        existing_ready, _ = self._validate_component(contract, destination)
        if existing_ready:
            # Never replace identical files underneath an active worker bind mount.
            return self.status(component_id)["components"][0]
        if destination.exists() and not self.replacement_allowed():
            raise WorkstationAssetError(
                "Component repair must wait until running GPU jobs have finished"
            )
        staging = Path(tempfile.mkdtemp(prefix=f".{component_id}.", dir=self.root))
        try:
            shutil.copytree(source, staging / component_id, symlinks=False)
            copied = staging / component_id
            copied_ready, copied_reason = self._validate_component(contract, copied)
            if not copied_ready:
                raise WorkstationAssetError(
                    f"Copied component does not match the lock: {copied_reason}"
                )
            backup = self.root / f".{component_id}.previous"
            if backup.exists():
                shutil.rmtree(backup)
            if destination.exists():
                os.replace(destination, backup)
            try:
                os.replace(copied, destination)
            except Exception:
                if backup.exists() and not destination.exists():
                    os.replace(backup, destination)
                raise
            shutil.rmtree(backup, ignore_errors=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return self.status(component_id)["components"][0]

    def download_component(
        self,
        component_id: str,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Install one component from its immutable, checksum-pinned URLs.

        This method is intentionally separate from worker execution. The Studio
        control plane calls it only from the explicit Install AI Components action;
        all queued neural workers continue to run with networking disabled.
        """

        component_id = _component_id(component_id)
        try:
            contract = self.contracts[component_id]
        except KeyError as error:
            raise WorkstationAssetError("Component is not present in the lock") from error
        if not contract.qualified or not contract.files:
            raise WorkstationAssetError("Component has no qualified download contract")
        if any(item.source_url is None for item in contract.files):
            raise WorkstationAssetError(
                "Component must be built locally or imported from a verified offline bundle"
            )
        total_bytes = sum(item.bytes for item in contract.files)
        completed = 0
        staging = Path(tempfile.mkdtemp(prefix=f".{component_id}.download.", dir=self.root))
        source_root = staging / component_id
        try:
            for item in contract.files:
                assert item.source_url is not None
                destination = source_root.joinpath(*item.path.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                request = Request(
                    item.source_url,
                    headers={"User-Agent": "AnifLive-TTS-Studio/1.4 component-installer"},
                )
                written = 0
                digest = hashlib.sha256()
                try:
                    with urlopen(request, timeout=300) as response, destination.open("xb") as stream:
                        while block := response.read(8 * 1024 * 1024):
                            written += len(block)
                            if written > item.bytes:
                                raise WorkstationAssetError(
                                    f"Downloaded component file exceeded its pinned size: {item.path}"
                                )
                            stream.write(block)
                            digest.update(block)
                            if progress is not None:
                                progress(completed + written, total_bytes)
                except WorkstationAssetError:
                    raise
                except OSError as error:
                    raise WorkstationAssetError(
                        f"Could not download pinned component file: {item.path}"
                    ) from error
                if written != item.bytes or digest.hexdigest() != item.sha256:
                    raise WorkstationAssetError(
                        f"Downloaded component file failed its pinned fingerprint: {item.path}"
                    )
                completed += written
            return self.install_from_directory(component_id, source_root)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def export_bundle(self, destination: Path, component_ids: Sequence[str] | None = None) -> Path:
        selected = tuple(component_ids or self.contracts.keys())
        if not selected:
            raise WorkstationAssetError("At least one component must be exported")
        records: list[dict[str, Any]] = []
        components: list[tuple[ComponentContract, Path]] = []
        for component_id in selected:
            component_id = _component_id(component_id)
            try:
                contract = self.contracts[component_id]
            except KeyError as error:
                raise WorkstationAssetError("Component is not present in the lock") from error
            root = self.component_root(component_id)
            ready, reason = self._validate_component(contract, root)
            if not ready:
                raise WorkstationAssetError(
                    f"Component {component_id} is not ready: {reason}"
                )
            records.append(contract.as_dict())
            components.append((contract, root))
        destination = Path(destination).expanduser().resolve()
        if destination.suffix.casefold() != ".zip":
            raise WorkstationAssetError("Component bundle must use a .zip filename")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr(
                    "bundle.json",
                    json.dumps(
                        {
                            "schema": WORKSTATION_ASSET_BUNDLE_SCHEMA,
                            "lock_sha256": _sha256_file(self.lock_path),
                            "components": records,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n",
                )
                for contract, root in components:
                    for item in contract.files:
                        archive.write(
                            root.joinpath(*item.path.parts),
                            f"components/{contract.component_id}/{item.path.as_posix()}",
                        )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def import_bundle(self, bundle: Path) -> dict[str, Any]:
        bundle = Path(bundle).expanduser().resolve(strict=True)
        if not bundle.is_file() or bundle.is_symlink() or bundle.suffix.casefold() != ".zip":
            raise WorkstationAssetError("Component bundle must be a regular .zip file")
        staging = Path(tempfile.mkdtemp(prefix=".bundle.", dir=self.root))
        try:
            with zipfile.ZipFile(bundle, "r") as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if len(names) != len(set(names)):
                    raise WorkstationAssetError("Component bundle contains duplicate paths")
                for info in infos:
                    relative = PurePosixPath(info.filename)
                    mode = (info.external_attr >> 16) & 0o170000
                    if (
                        not relative.parts
                        or relative.is_absolute()
                        or "." in relative.parts
                        or ".." in relative.parts
                        or "\\" in info.filename
                        or info.flag_bits & 0x1
                        or mode == stat.S_IFLNK
                    ):
                        raise WorkstationAssetError(
                            "Component bundle contains an unsafe path or entry"
                        )
                if "bundle.json" not in names:
                    raise WorkstationAssetError("Component bundle manifest is missing")
                manifest_info = archive.getinfo("bundle.json")
                if manifest_info.is_dir() or manifest_info.file_size > 8 * 1024**2:
                    raise WorkstationAssetError("Component bundle manifest is malformed")
                try:
                    manifest = json.loads(archive.read(manifest_info).decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError) as error:
                    raise WorkstationAssetError(
                        "Component bundle manifest is unreadable"
                    ) from error
                if not isinstance(manifest, dict):
                    raise WorkstationAssetError(
                        "Component bundle manifest must be a JSON object"
                    )
                if manifest.get("schema") != WORKSTATION_ASSET_BUNDLE_SCHEMA:
                    raise WorkstationAssetError("Component bundle schema is unsupported")
                if manifest.get("lock_sha256") != _sha256_file(self.lock_path):
                    raise WorkstationAssetError("Component bundle was built for another lock")
                values = manifest.get("components")
                if not isinstance(values, list) or not values:
                    raise WorkstationAssetError("Component bundle is empty")
                selected: list[ComponentContract] = []
                selected_ids: set[str] = set()
                for value in values:
                    if not isinstance(value, Mapping):
                        raise WorkstationAssetError(
                            "Component bundle contract is malformed"
                        )
                    component_id = _component_id(value.get("id"))
                    if component_id in selected_ids:
                        raise WorkstationAssetError(
                            "Component bundle contains a duplicate component"
                        )
                    contract = self.contracts.get(component_id)
                    if contract is None or value != contract.as_dict():
                        raise WorkstationAssetError(
                            f"Component {component_id} does not match the active lock"
                        )
                    selected_ids.add(component_id)
                    selected.append(contract)
                expected = {"bundle.json": manifest_info.file_size}
                for contract in selected:
                    for item in contract.files:
                        expected[
                            f"components/{contract.component_id}/{item.path.as_posix()}"
                        ] = item.bytes
                actual = {
                    info.filename: info.file_size
                    for info in infos
                    if not info.is_dir()
                }
                if actual != expected or any(info.is_dir() for info in infos):
                    raise WorkstationAssetError(
                        "Component bundle file inventory does not match its lock"
                    )
                for relative_name, expected_bytes in expected.items():
                    destination = staging.joinpath(*PurePosixPath(relative_name).parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    written = 0
                    with archive.open(relative_name, "r") as source, destination.open("xb") as target:
                        while block := source.read(8 * 1024 * 1024):
                            written += len(block)
                            if written > expected_bytes:
                                raise WorkstationAssetError(
                                    "Component bundle entry exceeded its pinned size"
                                )
                            target.write(block)
                    if written != expected_bytes:
                        raise WorkstationAssetError(
                            "Component bundle entry did not match its pinned size"
                        )
            installed = [
                self.install_from_directory(
                    contract.component_id,
                    staging / "components" / contract.component_id,
                )
                for contract in selected
            ]
            return {
                "schema": WORKSTATION_ASSET_BUNDLE_SCHEMA,
                "installed": installed,
                "network_required": False,
            }
        except zipfile.BadZipFile as error:
            raise WorkstationAssetError("Component bundle is not a valid zip archive") from error
        finally:
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "COMPONENT_STATES",
    "WORKSTATION_ASSET_BUNDLE_SCHEMA",
    "WORKSTATION_ASSET_LOCK_SCHEMA",
    "ComponentContract",
    "ComponentFile",
    "WorkstationAssetError",
    "WorkstationAssetManager",
    "load_asset_lock",
]
