from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
from pathlib import Path
from typing import Any, Callable

from .backend.contracts import STAGE_ORDER
from .errors import EngineRebuildRequired, PackageValidationError


SAFE_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?\Z")
WINDOWS_RESERVED_IDENTIFIERS = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
ENGINE_RUNTIME_KEYS = (
    "tensorrt",
    "cuda_runtime",
    "compute_capability",
    "gpu",
    "gpu_sm_count",
    "gpu_total_memory_bytes",
    "platform_system",
    "platform_machine",
    "platform",
)
ENGINE_COMPATIBILITY_KEYS = tuple(key for key in ENGINE_RUNTIME_KEYS if key != "platform")


def validate_safe_identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not SAFE_IDENTIFIER_PATTERN.fullmatch(value)
        or value.split(".", 1)[0].casefold() in WINDOWS_RESERVED_IDENTIFIERS
    ):
        raise PackageValidationError(
            f"{label} must be a 1-64 character identifier containing only ASCII letters, "
            "digits, '.', '_' or '-', and must start and end with a letter or digit"
        )
    return value


def resolve_contained_path(root: Path, relative: Any, label: str) -> Path:
    root = root.resolve()
    if not isinstance(relative, (str, os.PathLike)):
        raise PackageValidationError(f"{label} must be a relative path")
    text = os.fspath(relative)
    if not isinstance(text, str):
        raise PackageValidationError(f"{label} must be a text path")
    if (
        not text
        or "\x00" in text
        or "\\" in text
        or ":" in text
        or Path(text).is_absolute()
        or any(part in {".", ".."} for part in text.split("/"))
    ):
        raise PackageValidationError(f"Unsafe {label}: {text!r}")
    candidate = (root / text).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise PackageValidationError(f"Unsafe {label}: {text!r}") from error
    if candidate == root:
        raise PackageValidationError(f"Unsafe {label}: {text!r}")
    return candidate


def _engine_rebuild_required(reason: str) -> EngineRebuildRequired:
    return EngineRebuildRequired(
        "ENGINE_REBUILD_REQUIRED: "
        f"{reason}. TensorRT engines are target-specific; rebuild them inside the target "
        "Linux container with 'aniflive-tts model rebuild-engines --model-package "
        "/data/models/active --force' before serving"
    )


def _canonical_machine(value: str) -> str:
    machine = value.strip().lower()
    return {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(machine, machine)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def runtime_fingerprint() -> dict[str, str]:
    import tensorrt as trt
    import torch

    if not torch.cuda.is_available():
        raise _engine_rebuild_required("CUDA is unavailable")
    properties = torch.cuda.get_device_properties(0)
    major, minor = properties.major, properties.minor
    return {
        "tensorrt": trt.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "compute_capability": f"{major}.{minor}",
        "gpu": " ".join(str(properties.name).split()),
        "gpu_sm_count": str(properties.multi_processor_count),
        "gpu_total_memory_bytes": str(properties.total_memory),
        "platform_system": platform.system().strip().lower(),
        "platform_machine": _canonical_machine(platform.machine()),
        "platform": platform.platform(),
    }


def engine_fingerprint(
    *, onnx_hashes: dict[str, str], profiles: dict[str, Any], build_config: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    payload = {
        **runtime_fingerprint(),
        "onnx_hash": canonical_hash(onnx_hashes),
        "profiles_hash": canonical_hash(profiles),
        "build_config_hash": canonical_hash(build_config),
    }
    return canonical_hash(payload)[:24], payload


def validate_checksums(package_dir: Path) -> dict[str, str]:
    package_dir = package_dir.resolve()
    path = package_dir / "checksums.json"
    if not path.is_file():
        raise PackageValidationError(f"Missing checksums.json: {path}")
    checksums = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(checksums, dict):
        raise PackageValidationError("checksums.json must contain an object")
    actual_files = {
        candidate.relative_to(package_dir).as_posix()
        for candidate in package_dir.rglob("*")
        if candidate.is_file() and candidate.name != "checksums.json"
    }
    declared_files = set(checksums)
    if declared_files != actual_files:
        missing = sorted(actual_files - declared_files)
        unexpected = sorted(declared_files - actual_files)
        details = []
        if missing:
            details.append("unlisted files: " + ", ".join(missing))
        if unexpected:
            details.append("missing files: " + ", ".join(unexpected))
        raise PackageValidationError("Incomplete checksum inventory (" + "; ".join(details) + ")")
    for relative, expected in checksums.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise PackageValidationError("Checksum entries must map text paths to SHA-256 strings")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise PackageValidationError(f"Invalid SHA-256 value for {relative}")
        candidate = resolve_contained_path(package_dir, relative, "package path")
        if not candidate.is_file():
            raise PackageValidationError(f"Missing package file: {relative}")
        actual = sha256_file(candidate)
        if actual != expected:
            raise PackageValidationError(f"SHA-256 mismatch for {relative}")
    return checksums


def select_engine_dir(package_dir: Path, manifest: dict[str, Any]) -> Path:
    package_dir = package_dir.resolve()
    expected = manifest.get("active_engine_fingerprint")
    if not expected:
        raise _engine_rebuild_required("model package has no engine fingerprint")
    try:
        expected = validate_safe_identifier(expected, "active_engine_fingerprint")
        engine_dir = resolve_contained_path(
            package_dir / "engines", expected, "engine fingerprint path"
        )
    except PackageValidationError as error:
        raise _engine_rebuild_required(str(error)) from error
    try:
        engine_manifest_path = resolve_contained_path(
            engine_dir, "engine-manifest.json", "engine manifest path"
        )
    except PackageValidationError as error:
        raise _engine_rebuild_required(str(error)) from error
    if not engine_manifest_path.is_file():
        raise _engine_rebuild_required(f"engine bundle is missing: {engine_dir}")
    try:
        engine_manifest = json.loads(engine_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _engine_rebuild_required("engine-manifest.json is unreadable") from error
    if not isinstance(engine_manifest, dict):
        raise _engine_rebuild_required("engine-manifest.json must contain an object")
    if engine_manifest.get("fingerprint") != expected:
        raise _engine_rebuild_required(
            "engine manifest fingerprint does not match active_engine_fingerprint"
        )
    runtime = runtime_fingerprint()
    recorded = engine_manifest.get("runtime", {})
    if not isinstance(recorded, dict):
        raise _engine_rebuild_required("engine runtime fingerprint is missing")
    for key in ENGINE_COMPATIBILITY_KEYS:
        package_value = recorded.get(key)
        runtime_value = runtime[key]
        if key in {"gpu", "platform_system", "platform_machine"}:
            matches = str(package_value).strip().casefold() == str(runtime_value).strip().casefold()
        else:
            matches = str(package_value) == str(runtime_value)
        if not matches:
            raise _engine_rebuild_required(
                f"{key} mismatch: package={package_value!r}, runtime={runtime_value!r}"
            )
    missing = [stage for stage in STAGE_ORDER if not (engine_dir / f"{stage}.engine").is_file()]
    if missing:
        raise _engine_rebuild_required("missing TensorRT engines: " + ", ".join(missing))
    return engine_dir


def write_checksums(package_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or path.name in {"checksums.json", "checksums.json.tmp"}:
            continue
        relative = path.relative_to(package_dir).as_posix()
        files[relative] = sha256_file(path)
    temp = package_dir / "checksums.json.tmp"
    temp.write_text(json.dumps(files, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, package_dir / "checksums.json")
    return files


def _validate_serialized_engines(engine_dir: Path) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    for stage in STAGE_ORDER:
        engine_path = engine_dir / f"{stage}.engine"
        if not engine_path.is_file():
            raise PackageValidationError(f"Missing TensorRT engine: {engine_path.name}")
        if runtime.deserialize_cuda_engine(engine_path.read_bytes()) is None:
            raise PackageValidationError(
                f"TensorRT could not deserialize engine on this runtime: {engine_path.name}"
            )


def migrate_engine_metadata(
    package_dir: Path,
    *,
    validate_engines: Callable[[Path], None] = _validate_serialized_engines,
) -> Path:
    """Upgrade a same-machine legacy engine identity without rebuilding engine bytes.

    Serving remains strict: an incomplete runtime fingerprint is never accepted. This
    explicit migration first verifies the legacy identity and deserializes every engine
    on the current runtime, then atomically moves the bundle under its full fingerprint.
    """

    package_dir = package_dir.expanduser().resolve()
    validate_checksums(package_dir)
    manifest_path = package_dir / "manifest.json"
    checksums_path = package_dir / "checksums.json"
    if not manifest_path.is_file():
        raise PackageValidationError(f"Missing manifest.json: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "aniflive-tts-model-package":
        raise PackageValidationError("Unsupported model package format")

    old_fingerprint = validate_safe_identifier(
        manifest.get("active_engine_fingerprint"), "active_engine_fingerprint"
    )
    old_engine_dir = resolve_contained_path(
        package_dir / "engines", old_fingerprint, "engine fingerprint path"
    )
    engine_manifest_path = old_engine_dir / "engine-manifest.json"
    if not engine_manifest_path.is_file():
        raise PackageValidationError(f"Missing engine-manifest.json: {engine_manifest_path}")
    engine_manifest = json.loads(engine_manifest_path.read_text(encoding="utf-8"))
    if engine_manifest.get("fingerprint") != old_fingerprint:
        raise PackageValidationError("Engine manifest fingerprint does not match its directory")

    recorded = engine_manifest.get("runtime")
    if not isinstance(recorded, dict):
        raise PackageValidationError("Engine runtime fingerprint is missing")
    current = runtime_fingerprint()
    legacy_identity_keys = ("tensorrt", "cuda_runtime", "compute_capability", "gpu")
    for key in legacy_identity_keys:
        package_value = recorded.get(key)
        runtime_value = current[key]
        if key == "gpu":
            matches = str(package_value).strip().casefold() == str(runtime_value).strip().casefold()
        else:
            matches = str(package_value) == str(runtime_value)
        if not matches:
            raise _engine_rebuild_required(
                f"{key} mismatch: package={package_value!r}, runtime={runtime_value!r}"
            )

    validate_engines(old_engine_dir)
    old_payload = engine_manifest.get("fingerprint_payload")
    if not isinstance(old_payload, dict):
        raise PackageValidationError("Engine fingerprint payload is missing")
    hash_keys = ("onnx_hash", "profiles_hash", "build_config_hash")
    if any(not old_payload.get(key) for key in hash_keys):
        raise PackageValidationError("Engine fingerprint payload is incomplete")
    new_payload = {
        **current,
        **{key: old_payload[key] for key in hash_keys},
    }
    new_fingerprint = canonical_hash(new_payload)[:24]
    if new_fingerprint == old_fingerprint and all(
        key in recorded for key in ENGINE_COMPATIBILITY_KEYS
    ):
        return old_engine_dir

    new_engine_dir = package_dir / "engines" / new_fingerprint
    if new_engine_dir.exists():
        raise PackageValidationError(
            f"Target engine fingerprint already exists: {new_fingerprint}"
        )

    original_manifest_text = manifest_path.read_text(encoding="utf-8")
    original_engine_manifest_text = engine_manifest_path.read_text(encoding="utf-8")
    original_checksums_text = checksums_path.read_text(encoding="utf-8")
    moved = False
    try:
        engine_manifest.update(
            {
                "fingerprint": new_fingerprint,
                "runtime": {key: new_payload[key] for key in ENGINE_RUNTIME_KEYS},
                "fingerprint_payload": new_payload,
            }
        )
        engine_manifest_path.write_text(
            json.dumps(engine_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["active_engine_fingerprint"] = new_fingerprint
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(old_engine_dir, new_engine_dir)
        moved = True
        write_checksums(package_dir)
    except Exception:
        if moved and new_engine_dir.exists() and not old_engine_dir.exists():
            os.replace(new_engine_dir, old_engine_dir)
        manifest_path.write_text(original_manifest_text, encoding="utf-8")
        (old_engine_dir / "engine-manifest.json").write_text(
            original_engine_manifest_text, encoding="utf-8"
        )
        checksums_path.write_text(original_checksums_text, encoding="utf-8")
        shutil.rmtree(new_engine_dir, ignore_errors=True)
        raise
    return new_engine_dir
