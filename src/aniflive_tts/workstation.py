from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from .qualification import (
    AUTOMATED_EVALUATION_SCHEMA,
    BLIND_AB_EVIDENCE_SCHEMA,
    CONTENT_REVIEW_SCHEMA,
    parse_content_review,
    QUALIFICATION_REPORT_SCHEMA,
    REQUIRED_QUALIFICATION_GATES,
    SECURITY_EVIDENCE_SCHEMA,
    QualificationReportError,
    automated_evaluation_gates,
    compose_qualification_report,
    parse_blind_ab_evidence,
    parse_composed_qualification_report,
    parse_security_evidence,
)
from .dataset_acquisition import DatasetAcquisitionConfig, DatasetAcquisitionError

WORKSTATION_SCHEMA = 9
WORKSTATION_SETTINGS_KEY = "workstation_settings"
WORKSTATION_VISIBLE_HISTORY_KEY = "visible_history_cleared_at"
DEFAULT_WORKSTATION_SETTINGS: dict[str, Any] = {
    "default_language": "ja",
    "default_continuity_policy": "A",
    "default_training_preset": "balanced",
    "default_benchmark_language": "yue",
    "default_tse_target_threshold": 0.72,
    "auto_refresh_seconds": 5,
    "overview_motion": True,
}
PROJECT_KINDS = frozenset({"dataset", "tse", "training", "evaluation"})
ARTIFACT_TYPES = frozenset(
    {
        "dataset",
        "checkpoint",
        "reference",
        "expression-bank",
        "evaluation",
        "engine",
        "package",
    }
)
ARTIFACT_STATUSES = frozenset(
    {"planned", "building", "ready", "failed", "rejected", "archived"}
)
EXPRESSION_QUALIFICATION_STATUSES = frozenset(
    {"draft", "pending", "qualified", "rejected", "archived"}
)
PROMOTABLE_ARTIFACT_TYPES = frozenset({"expression-bank", "engine", "package"})
JOB_TYPES = frozenset(
    {
        "dataset.inventory",
        "dataset.process",
        "dataset.decode",
        "dataset.target-speaker",
        "dataset.separate",
        "dataset.transcribe",
        "dataset.finalize",
        "tse.prepare",
        "training.prepare",
        "checkpoint.select",
        "reference.select",
        "holdout.evaluate",
        "evaluation.prepare",
        "engine.prepare",
        "conversion.parity",
        "model.package",
    }
)
TERMINAL_JOB_STATES = frozenset({"succeeded", "failed", "cancelled"})
JOB_STATES = frozenset({"queued", "running", "paused", *TERMINAL_JOB_STATES})
JOB_TRANSITIONS = {
    "queued": frozenset({"running", "paused", "cancelled", "failed"}),
    "running": frozenset({"paused", "succeeded", "failed", "cancelled"}),
    "paused": frozenset({"queued", "cancelled"}),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}
RESOURCE_CLASSES = frozenset({"cpu-shared", "gpu-exclusive", "io-heavy"})
JOB_RESOURCE_CLASSES = {
    "dataset.inventory": "io-heavy",
    "dataset.process": "gpu-exclusive",
    "dataset.decode": "gpu-exclusive",
    "dataset.target-speaker": "gpu-exclusive",
    "dataset.separate": "gpu-exclusive",
    "dataset.transcribe": "gpu-exclusive",
    "dataset.finalize": "io-heavy",
    "tse.prepare": "gpu-exclusive",
    "training.prepare": "gpu-exclusive",
    "checkpoint.select": "gpu-exclusive",
    "reference.select": "gpu-exclusive",
    "holdout.evaluate": "gpu-exclusive",
    "evaluation.prepare": "gpu-exclusive",
    "engine.prepare": "gpu-exclusive",
    "conversion.parity": "gpu-exclusive",
    "model.package": "gpu-exclusive",
}
JOB_PROJECT_KINDS = {
    "dataset.inventory": frozenset({"dataset"}),
    "dataset.process": frozenset({"dataset"}),
    "dataset.decode": frozenset({"dataset"}),
    "dataset.target-speaker": frozenset({"dataset"}),
    "dataset.separate": frozenset({"dataset"}),
    "dataset.transcribe": frozenset({"dataset"}),
    "dataset.finalize": frozenset({"dataset"}),
    "tse.prepare": frozenset({"tse"}),
    "training.prepare": frozenset({"training"}),
    "checkpoint.select": frozenset({"training"}),
    "reference.select": frozenset({"training"}),
    "holdout.evaluate": frozenset({"training"}),
    "evaluation.prepare": frozenset({"evaluation"}),
    "engine.prepare": frozenset({"training", "evaluation"}),
    "conversion.parity": frozenset({"training", "evaluation"}),
    "model.package": frozenset({"training", "evaluation"}),
}
GPU_RESOURCE_KEY = "gpu:0"
MEDIA_SUFFIXES = frozenset(
    {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".mp4", ".mkv", ".mov"}
)
REFERENCE_AUDIO_SUFFIXES = frozenset(
    {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}
)
MAX_EXPRESSION_REFERENCE_BYTES = 512 * 1024 * 1024
EXPRESSION_REFERENCE_ANALYSIS_KEY = "reference_analysis"
_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_PROFILE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_LANGUAGE_PATTERN = re.compile(r"^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$")
_WINDOWS_RESERVED_COMPONENT = re.compile(
    r"^(?:con|prn|aux|nul|com[1-9]|lpt[1-9]|conin\$|conout\$)(?:\..*)?$",
    re.IGNORECASE,
)
_UNSET = object()
_ARTIFACTS_TABLE_SQL = """CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT NOT NULL,
    project_id TEXT REFERENCES projects(id), status TEXT NOT NULL,
    local_path TEXT, sha256 TEXT, metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    CHECK(length(id) = 45 AND substr(id, 1, 9) = 'artifact_'),
    CHECK(type IN (
        'dataset', 'checkpoint', 'reference', 'expression-bank',
        'evaluation', 'engine', 'package'
    )),
    CHECK(status IN ('planned', 'building', 'ready', 'failed', 'rejected', 'archived')),
    CHECK(local_path IS NULL OR length(local_path) > 0),
    CHECK(sha256 IS NULL OR (
        length(sha256) = 64 AND sha256 = lower(sha256)
        AND sha256 NOT GLOB '*[^0-9a-f]*'
    ))
)"""
_DATASET_ACQUISITION_CONFIG_KEYS = frozenset(
    {
        "acquisition_mode",
        "sources",
        "reference_audio",
        "speaker_threshold",
        "ambiguity_margin",
        "review_margin",
        "speaker_component",
        "vad_component",
        "separation_component",
        "asr_component",
        "require_expressions",
    }
)

_WORKER_ARTIFACT_TYPE = {
    "dataset.process": "dataset",
    "dataset.decode": "dataset",
    "dataset.target-speaker": "dataset",
    "dataset.separate": "dataset",
    "dataset.transcribe": "dataset",
    "dataset.finalize": "dataset",
    "tse.prepare": "dataset",
    "training.prepare": "checkpoint",
    "checkpoint.select": "checkpoint",
    "reference.select": "reference",
    "holdout.evaluate": "evaluation",
    "evaluation.prepare": "evaluation",
    "engine.prepare": "engine",
    "conversion.parity": "evaluation",
    "model.package": "package",
}
_FAILED_DIAGNOSTIC_ARTIFACT_JOB_TYPES = frozenset(
    {"checkpoint.select", "holdout.evaluate", "conversion.parity"}
)


MAX_JOB_RESULT_BYTES = 256 * 1024

def worker_artifact_limit(job_type: str) -> int:
    """Bound validation evidence without changing other worker artifact budgets."""
    return 4096 if job_type == "checkpoint.select" else 1024


class WorkstationError(ValueError):
    pass


def _workstation_settings(value: Mapping[str, Any] | None) -> dict[str, Any]:
    settings = dict(DEFAULT_WORKSTATION_SETTINGS)
    if value is None:
        return settings
    if not isinstance(value, Mapping):
        raise WorkstationError("Workstation settings must be a JSON object")
    unknown = set(value) - set(settings)
    if unknown:
        raise WorkstationError(
            "Unsupported workstation settings: " + ", ".join(sorted(unknown))
        )
    settings.update(value)

    languages = {"zh", "yue", "en", "ja", "ko"}
    if settings["default_language"] not in languages:
        raise WorkstationError("default_language must be one of zh, yue, en, ja or ko")
    if settings["default_benchmark_language"] not in languages:
        raise WorkstationError(
            "default_benchmark_language must be one of zh, yue, en, ja or ko"
        )
    if settings["default_continuity_policy"] not in {"A", "B", "C", "D", "E", "F"}:
        raise WorkstationError("default_continuity_policy must be one of A, B, C, D, E or F")
    if settings["default_training_preset"] not in {
        "quick",
        "balanced",
        "high-quality",
        "advanced",
    }:
        raise WorkstationError("default_training_preset is unsupported")

    threshold = settings["default_tse_target_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise WorkstationError("default_tse_target_threshold must be a number")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise WorkstationError("default_tse_target_threshold must be between 0 and 1")
    settings["default_tse_target_threshold"] = threshold

    refresh = settings["auto_refresh_seconds"]
    if isinstance(refresh, bool) or not isinstance(refresh, int) or not 2 <= refresh <= 30:
        raise WorkstationError("auto_refresh_seconds must be an integer from 2 to 30")
    if not isinstance(settings["overview_motion"], bool):
        raise WorkstationError("overview_motion must be true or false")
    return settings


def validated_artifact_relative_path(value: Any, *, field: str) -> PurePosixPath:
    """Validate a portable artifact path before either Linux or Windows opens it."""

    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
    ):
        raise WorkstationError(f"{field} is malformed")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise WorkstationError(f"{field} is malformed")
    for part in relative.parts:
        if (
            ":" in part
            or part.endswith((".", " "))
            or _WINDOWS_RESERVED_COMPONENT.fullmatch(part) is not None
            or any(ord(character) < 32 for character in part)
        ):
            raise WorkstationError(f"{field} is not portable to Windows")
    return relative


def _utc_datetime() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    return (value or _utc_datetime()).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_now() -> str:
    return _timestamp()


def default_workstation_root() -> Path:
    configured = os.environ.get("ANIFLIVE_TTS_WORKSTATION_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / ".local" / "share"
    return (base / "AnifLive-TTS" / "workstation").resolve()


def _json_value(value: Any, *, field: str) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise WorkstationError(f"{field} must contain JSON-compatible values") from error
    maximum = MAX_JOB_RESULT_BYTES if field == "result" else 64 * 1024
    if len(encoded.encode("utf-8")) > maximum:
        raise WorkstationError(f"{field} is too large")
    return json.loads(encoded)


def _mapping_value(value: Mapping[str, Any] | None, *, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise WorkstationError(f"{field} must be a JSON object")
    parsed = _json_value(dict(value), field=field)
    if not isinstance(parsed, dict):
        raise WorkstationError(f"{field} must be a JSON object")
    return parsed


def _editable_expression_prosody(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    parsed = _mapping_value(value, field="prosody")
    if EXPRESSION_REFERENCE_ANALYSIS_KEY in parsed:
        raise WorkstationError(
            "prosody.reference_analysis is measured by the workstation and is read-only"
        )
    return parsed


def _json_text(value: Any, *, field: str) -> str:
    parsed = _json_value(value, field=field)
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _required_text(value: Any, *, field: str, maximum: int = 120) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkstationError(f"{field} must not be empty")
    result = " ".join(value.strip().split())
    if len(result) > maximum:
        raise WorkstationError(f"{field} is limited to {maximum} characters")
    return result


def _decode_json(value: str | None, *, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise WorkstationError("The workstation database contains malformed JSON") from error


def _progress(value: Any, *, field: str = "progress") -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise WorkstationError(f"{field} must be a number between 0 and 1") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise WorkstationError(f"{field} must be a finite number between 0 and 1")
    return result


def _job_priority(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkstationError("priority must be an integer between -100 and 100")
    if value < -100 or value > 100:
        raise WorkstationError("priority must be an integer between -100 and 100")
    return value


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4()}"


def _validated_id(value: Any, *, field: str, prefixes: Sequence[str]) -> str:
    text = _required_text(value, field=field, maximum=80)
    prefix, separator, suffix = text.partition("_")
    if not separator or prefix not in prefixes:
        raise WorkstationError(f"{field} is malformed")
    try:
        parsed = UUID(suffix)
    except (ValueError, AttributeError) as error:
        raise WorkstationError(f"{field} is malformed") from error
    if str(parsed) != suffix.lower():
        raise WorkstationError(f"{field} is malformed")
    return text


def _positive_integer_env(name: str, default: int, *, maximum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise WorkstationError(f"{name} must be a positive integer") from error
    if value < 1 or value > maximum:
        raise WorkstationError(f"{name} must be between 1 and {maximum}")
    return value


def _error_message(error: BaseException) -> str:
    message = " ".join(str(error).strip().split()) or error.__class__.__name__
    if len(message) <= 1000:
        return message
    marker = " ... [middle omitted] ... "
    head_length = 240
    tail_length = 1000 - head_length - len(marker)
    return message[:head_length] + marker + message[-tail_length:]


def _sha256_value(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise WorkstationError("sha256 must be a hexadecimal string")
    result = value.strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise WorkstationError("sha256 must contain exactly 64 hexadecimal characters")
    return result


def _bounded_float(value: Any, *, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise WorkstationError(f"{field} must be a finite number between {minimum} and {maximum}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise WorkstationError(
            f"{field} must be a finite number between {minimum} and {maximum}"
        ) from error
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise WorkstationError(
            f"{field} must be a finite number between {minimum} and {maximum}"
        )
    return result


def _profile_id(value: Any) -> str:
    result = _required_text(value, field="profile_id", maximum=64).lower()
    if _PROFILE_ID_PATTERN.fullmatch(result) is None:
        raise WorkstationError(
            "profile_id must use lowercase letters, numbers, dots, underscores, or hyphens"
        )
    return result


def _language_code(value: Any) -> str:
    result = _required_text(value, field="language", maximum=35).lower()
    if _LANGUAGE_PATTERN.fullmatch(result) is None:
        raise WorkstationError("language must be a valid lowercase language tag")
    return result


def _descriptions_value(value: Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise WorkstationError("descriptions must be a list of strings")
    if len(value) > 32:
        raise WorkstationError("descriptions is limited to 32 entries")
    return [
        _required_text(description, field=f"descriptions[{index}]", maximum=500)
        for index, description in enumerate(value)
    ]


def _vad_value(value: Mapping[str, Any] | None) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise WorkstationError("vad must be a JSON object")
    unknown = set(value) - {"valence", "arousal", "dominance"}
    if unknown:
        raise WorkstationError(f"vad contains unsupported fields: {', '.join(sorted(unknown))}")
    return {
        key: _bounded_float(value[key], field=f"vad.{key}", minimum=-1.0, maximum=1.0)
        for key in ("valence", "arousal", "dominance")
        if key in value
    }


def _expression_qualification_status(value: Any) -> str:
    if not isinstance(value, str) or value not in EXPRESSION_QUALIFICATION_STATUSES:
        raise WorkstationError(f"Unsupported expression qualification status: {value}")
    return value


def _is_reparse_point(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE)


def _stable_file_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
    )


def _checked_regular_file(root: Path, candidate: Path, *, field: str) -> Path:
    """Resolve one real regular file without allowing a linked path component."""

    root = root.absolute()
    if _is_reparse_point(root):
        raise WorkstationError(f"{field} store must be a real directory")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise WorkstationError(f"{field} store does not exist") from error
    if not root.is_dir() or _is_reparse_point(root):
        raise WorkstationError(f"{field} store must be a real directory")
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise WorkstationError(f"{field} must be contained by the artifact store") from error
    if not relative.parts:
        raise WorkstationError(f"{field} must identify a file")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as error:
            raise WorkstationError(f"{field} file does not exist") from error
        except OSError as error:
            raise WorkstationError(f"{field} could not be inspected") from error
        if current.is_symlink() or bool(
            getattr(info, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
        ):
            raise WorkstationError(
                f"{field} cannot contain a symbolic link or reparse point"
            )
    try:
        resolved = current.resolve(strict=True)
        info = resolved.stat(follow_symlinks=False)
    except OSError as error:
        raise WorkstationError(f"{field} file does not exist") from error
    if not resolved.is_relative_to(root):
        raise WorkstationError(f"{field} must be contained by the artifact store")
    if not stat.S_ISREG(info.st_mode):
        raise WorkstationError(f"{field} must identify a regular file")
    return resolved


def _ensure_real_directory_chain(
    root: Path, directory: Path, *, field: str
) -> tuple[Path, tuple[Path, ...]]:
    """Create a contained directory one component at a time without following links."""

    root = root.absolute()
    if _is_reparse_point(root):
        raise WorkstationError(f"{field} store must be a real directory")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise WorkstationError(f"{field} store does not exist") from error
    if not root.is_dir():
        raise WorkstationError(f"{field} store must be a real directory")
    directory = directory.absolute()
    try:
        relative = directory.relative_to(root)
    except ValueError as error:
        raise WorkstationError(f"{field} must be contained by the artifact store") from error
    if not relative.parts:
        raise WorkstationError(f"{field} must identify a child directory")

    current = root
    created: list[Path] = []
    try:
        for part in relative.parts:
            current = current / part
            try:
                info = current.lstat()
            except FileNotFoundError:
                try:
                    current.mkdir(exist_ok=False)
                    created.append(current)
                except FileExistsError:
                    pass
                try:
                    info = current.lstat()
                except OSError as error:
                    raise WorkstationError(
                        f"{field} directory could not be inspected"
                    ) from error
            except OSError as error:
                raise WorkstationError(
                    f"{field} directory could not be inspected"
                ) from error
            if current.is_symlink() or bool(
                getattr(info, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
            ):
                raise WorkstationError(
                    f"{field} cannot contain a symbolic link or reparse point"
                )
            if not stat.S_ISDIR(info.st_mode):
                raise WorkstationError(f"{field} must identify a directory")
            resolved = current.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise WorkstationError(
                    f"{field} must be contained by the artifact store"
                )
        return current, tuple(created)
    except Exception:
        for created_directory in reversed(created):
            try:
                created_directory.rmdir()
            except OSError:
                pass
        raise


def _sha256_regular_file(path: Path, *, field: str) -> tuple[str, int]:
    """Hash a regular file while detecting common replacement/write races."""

    try:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or _is_reparse_point(path):
            raise WorkstationError(f"{field} must identify a regular file")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if _stable_file_identity(opened) != _stable_file_identity(before):
                raise WorkstationError(f"{field} changed while it was being opened")
            while block := handle.read(1024 * 1024):
                digest.update(block)
            after = os.fstat(handle.fileno())
        if _stable_file_identity(after) != _stable_file_identity(before):
            raise WorkstationError(f"{field} changed while it was being hashed")
    except WorkstationError:
        raise
    except OSError as error:
        raise WorkstationError(f"{field} could not be read") from error
    return digest.hexdigest(), int(before.st_size)


def _copy_verified_file(
    source: Path,
    destination: Path,
    *,
    artifact_root: Path,
    expected_sha256: str,
    expected_size: int,
) -> tuple[Path, ...]:
    """Copy one worker output without trusting its manifest digest or size."""

    created_directories: tuple[Path, ...] = ()
    try:
        before = source.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or _is_reparse_point(source):
            raise WorkstationError("Docker artifact source must be a regular file")
        verified_parent, created_directories = _ensure_real_directory_chain(
            artifact_root,
            destination.parent,
            field="Docker artifact destination",
        )
        destination = verified_parent / destination.name
        if any(verified_parent.iterdir()):
            raise WorkstationError(
                "Docker artifact destination is not an empty real directory"
            )
        digest = hashlib.sha256()
        copied = 0
        with source.open("rb") as reader, destination.open("xb") as writer:
            opened = os.fstat(reader.fileno())
            if _stable_file_identity(opened) != _stable_file_identity(before):
                raise WorkstationError("Docker artifact changed while it was being opened")
            while block := reader.read(1024 * 1024):
                writer.write(block)
                digest.update(block)
                copied += len(block)
            writer.flush()
            os.fsync(writer.fileno())
            after = os.fstat(reader.fileno())
        if _stable_file_identity(after) != _stable_file_identity(before):
            raise WorkstationError("Docker artifact changed while it was being copied")
        if copied != expected_size:
            raise WorkstationError("Docker artifact has the wrong size")
        if digest.hexdigest() != expected_sha256:
            raise WorkstationError("Docker artifact has the wrong checksum")
        return created_directories
    except WorkstationError:
        destination.unlink(missing_ok=True)
        for directory in reversed(created_directories):
            try:
                directory.rmdir()
            except OSError:
                pass
        raise
    except OSError as error:
        destination.unlink(missing_ok=True)
        for directory in reversed(created_directories):
            try:
                directory.rmdir()
            except OSError:
                pass
        raise WorkstationError("Docker artifact could not be imported") from error


@dataclass(frozen=True)
class WorkstationSnapshot:
    projects: tuple[dict[str, Any], ...]
    jobs: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class JobClaim:
    job: dict[str, Any]
    token: str
    worker_id: str
    lease_expires_at: str


class WorkstationStore:
    """Transactional local metadata, leases, and job queue for AnifLive-TTS."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_workstation_root()).resolve()
        self.database_path = self.root / "workstation.sqlite3"
        self.legacy_manifest_path = self.root / "workstation.json"
        self.artifact_root = self.root / "artifacts"
        self.worker_id = _new_id("worker")
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self._migrate_legacy_manifest()
        self.recover_expired_jobs()
        self.reconcile_project_summaries()

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = self._open_connection()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        last_error: Exception | None = None
        for attempt in range(8):
            try:
                with self._connect() as connection:
                    connection.execute("PRAGMA journal_mode = WAL")
                    connection.execute("PRAGMA synchronous = NORMAL")
                    connection.execute("PRAGMA foreign_keys = OFF")
                    connection.execute("BEGIN IMMEDIATE")
                    try:
                        self._create_schema(connection)
                        self._upgrade_schema(connection)
                        violations = connection.execute(
                            "PRAGMA foreign_key_check"
                        ).fetchall()
                        if violations:
                            raise WorkstationError(
                                "The workstation database contains broken lineage"
                            )
                        connection.commit()
                    except Exception:
                        connection.rollback()
                        raise
                    finally:
                        connection.execute("PRAGMA legacy_alter_table = OFF")
                        connection.execute("PRAGMA foreign_keys = ON")
                    mode = connection.execute("PRAGMA journal_mode").fetchone()
                    if mode is None or str(mode[0]).lower() != "wal":
                        raise WorkstationError("The workstation database could not enable WAL mode")
                return
            except sqlite3.OperationalError as error:
                last_error = error
                if "locked" not in str(error).lower() or attempt == 7:
                    break
                time.sleep(0.025 * (2**attempt))
        raise WorkstationError("The workstation database could not be initialized") from last_error

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        statements = (
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            """CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
                status TEXT NOT NULL, progress REAL NOT NULL, config_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, type TEXT NOT NULL,
                project_id TEXT REFERENCES projects(id), status TEXT NOT NULL,
                progress REAL NOT NULL, parameters_json TEXT NOT NULL,
                result_json TEXT NOT NULL, error TEXT, wait_reason TEXT,
                resource_class TEXT NOT NULL,
                depends_on_json TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
                worker_id TEXT, claim_token TEXT, lease_expires_at TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                pause_requested INTEGER NOT NULL DEFAULT 0,
                priority INTEGER NOT NULL DEFAULT 0,
                attempt INTEGER NOT NULL DEFAULT 1,
                retry_of TEXT REFERENCES jobs(id)
            )""",
            """CREATE TABLE IF NOT EXISTS job_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                level TEXT NOT NULL, message TEXT NOT NULL, created_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS resource_leases (
                resource_key TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                token TEXT NOT NULL UNIQUE, purpose TEXT NOT NULL,
                lease_expires_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""",
            _ARTIFACTS_TABLE_SQL,
            """CREATE TABLE IF NOT EXISTS artifact_parents (
                artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
                parent_artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE RESTRICT,
                position INTEGER NOT NULL CHECK(position >= 0),
                created_at TEXT NOT NULL,
                PRIMARY KEY(artifact_id, parent_artifact_id),
                UNIQUE(artifact_id, position),
                CHECK(artifact_id <> parent_artifact_id)
            )""",
            """CREATE TABLE IF NOT EXISTS expression_drafts (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, profile_id TEXT NOT NULL,
                model_id TEXT, reference_scope TEXT, reference_root_id TEXT,
                reference_path TEXT, reference_sha256 TEXT,
                language TEXT NOT NULL, emotion TEXT NOT NULL,
                intensity REAL NOT NULL, descriptions_json TEXT NOT NULL,
                vad_json TEXT NOT NULL, prosody_json TEXT NOT NULL,
                qualification_status TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                CHECK(length(id) = 47 AND substr(id, 1, 11) = 'expression_'),
                CHECK(intensity >= 0.0 AND intensity <= 1.0),
                CHECK(qualification_status IN ('draft', 'pending', 'qualified', 'rejected', 'archived')),
                CHECK(
                    (reference_scope IS NULL AND reference_root_id IS NULL
                        AND reference_path IS NULL AND reference_sha256 IS NULL)
                    OR (reference_scope = 'artifact' AND reference_root_id IS NULL
                        AND reference_path IS NOT NULL AND length(reference_path) > 0
                        AND length(reference_sha256) = 64
                        AND reference_sha256 = lower(reference_sha256)
                        AND reference_sha256 NOT GLOB '*[^0-9a-f]*')
                    OR (reference_scope = 'import-root' AND length(reference_root_id) = 16
                        AND reference_path IS NOT NULL AND length(reference_path) > 0
                        AND length(reference_sha256) = 64
                        AND reference_sha256 = lower(reference_sha256)
                        AND reference_sha256 NOT GLOB '*[^0-9a-f]*')
                )
            )""",
            """CREATE TABLE IF NOT EXISTS qualification_runs (
                id TEXT PRIMARY KEY, subject_kind TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                evaluation_artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE RESTRICT,
                report_sha256 TEXT NOT NULL, report_schema TEXT NOT NULL,
                overall_status TEXT NOT NULL, run_metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                CHECK(length(id) = 50 AND substr(id, 1, 14) = 'qualification_'),
                CHECK(subject_kind IN ('artifact', 'expression')),
                CHECK(overall_status IN ('passed', 'failed')),
                CHECK(length(report_sha256) = 64 AND report_sha256 = lower(report_sha256)
                    AND report_sha256 NOT GLOB '*[^0-9a-f]*'),
                UNIQUE(subject_kind, subject_id, evaluation_artifact_id, report_sha256)
            )""",
            """CREATE TABLE IF NOT EXISTS qualification_gate_results (
                qualification_id TEXT NOT NULL REFERENCES qualification_runs(id) ON DELETE CASCADE,
                gate_id TEXT NOT NULL, status TEXT NOT NULL, summary TEXT NOT NULL,
                metrics_json TEXT NOT NULL, evidence_json TEXT NOT NULL,
                PRIMARY KEY(qualification_id, gate_id),
                CHECK(status IN ('passed', 'failed'))
            )""",
            """CREATE TABLE IF NOT EXISTS artifact_promotions (
                artifact_id TEXT PRIMARY KEY REFERENCES artifacts(id) ON DELETE RESTRICT,
                qualification_id TEXT NOT NULL UNIQUE REFERENCES qualification_runs(id) ON DELETE RESTRICT,
                promoted_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS expression_qualifications (
                expression_id TEXT PRIMARY KEY REFERENCES expression_drafts(id) ON DELETE CASCADE,
                qualification_id TEXT NOT NULL UNIQUE REFERENCES qualification_runs(id) ON DELETE RESTRICT,
                promoted_at TEXT NOT NULL
            )""",
            "CREATE INDEX IF NOT EXISTS jobs_status_updated ON jobs(status, updated_at DESC)",
            "CREATE INDEX IF NOT EXISTS job_events_job_id ON job_events(job_id, id)",
            "CREATE INDEX IF NOT EXISTS artifacts_project_updated ON artifacts(project_id, updated_at DESC)",
            "CREATE INDEX IF NOT EXISTS artifacts_type_status ON artifacts(type, status, updated_at DESC)",
            "CREATE INDEX IF NOT EXISTS artifact_parents_parent ON artifact_parents(parent_artifact_id)",
            "CREATE INDEX IF NOT EXISTS expression_drafts_updated ON expression_drafts(updated_at DESC)",
            "CREATE INDEX IF NOT EXISTS expression_drafts_model_status ON expression_drafts(model_id, qualification_status, updated_at DESC)",
            "CREATE UNIQUE INDEX IF NOT EXISTS expression_drafts_profile_model ON expression_drafts(profile_id, COALESCE(model_id, ''))",
            "CREATE INDEX IF NOT EXISTS qualifications_subject ON qualification_runs(subject_kind, subject_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS qualifications_evaluation ON qualification_runs(evaluation_artifact_id, created_at DESC)",
            """CREATE TRIGGER IF NOT EXISTS artifacts_immutable_identity
                BEFORE UPDATE OF id, type, local_path, sha256, created_at ON artifacts
                WHEN OLD.id IS NOT NEW.id OR OLD.type IS NOT NEW.type OR OLD.local_path IS NOT NEW.local_path
                    OR OLD.sha256 IS NOT NEW.sha256 OR OLD.created_at IS NOT NEW.created_at
                BEGIN SELECT RAISE(ABORT, 'artifact identity is immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS artifact_parent_links_immutable
                BEFORE UPDATE OF artifact_id, parent_artifact_id ON artifact_parents
                WHEN OLD.artifact_id IS NOT NEW.artifact_id
                    OR OLD.parent_artifact_id IS NOT NEW.parent_artifact_id
                BEGIN SELECT RAISE(ABORT, 'artifact lineage links are immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS artifact_parents_no_cycle
                BEFORE INSERT ON artifact_parents
                WHEN NEW.artifact_id = NEW.parent_artifact_id OR EXISTS(
                    SELECT 1 FROM artifact_parents
                    WHERE parent_artifact_id = NEW.artifact_id
                )
                BEGIN
                    SELECT CASE WHEN NEW.artifact_id = NEW.parent_artifact_id
                        THEN RAISE(ABORT, 'artifact lineage cycle') END;
                    WITH RECURSIVE ancestors(id) AS (
                        SELECT NEW.parent_artifact_id
                        UNION
                        SELECT ap.parent_artifact_id
                        FROM artifact_parents AS ap JOIN ancestors AS a
                            ON ap.artifact_id = a.id
                    )
                    SELECT CASE WHEN EXISTS(
                        SELECT 1 FROM ancestors WHERE id = NEW.artifact_id
                    ) THEN RAISE(ABORT, 'artifact lineage cycle') END;
                END""",
            """CREATE TRIGGER IF NOT EXISTS expression_drafts_immutable_source
                BEFORE UPDATE OF id, profile_id, model_id, reference_scope,
                    reference_root_id, reference_path, reference_sha256,
                    created_at ON expression_drafts
                WHEN OLD.id IS NOT NEW.id OR OLD.profile_id IS NOT NEW.profile_id
                    OR OLD.model_id IS NOT NEW.model_id
                    OR OLD.reference_scope IS NOT NEW.reference_scope
                    OR OLD.reference_root_id IS NOT NEW.reference_root_id
                    OR OLD.reference_path IS NOT NEW.reference_path
                    OR OLD.reference_sha256 IS NOT NEW.reference_sha256
                    OR OLD.created_at IS NOT NEW.created_at
                BEGIN SELECT RAISE(ABORT, 'expression source identity is immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS expression_qualified_insert_guard
                BEFORE INSERT ON expression_drafts
                WHEN NEW.qualification_status = 'qualified'
                BEGIN SELECT RAISE(ABORT, 'qualified expression requires evaluation evidence'); END""",
            """CREATE TRIGGER IF NOT EXISTS expression_qualified_update_guard
                BEFORE UPDATE OF qualification_status ON expression_drafts
                WHEN NEW.qualification_status = 'qualified' AND NOT EXISTS(
                    SELECT 1 FROM expression_qualifications WHERE expression_id = NEW.id
                )
                BEGIN SELECT RAISE(ABORT, 'qualified expression requires evaluation evidence'); END""",
            """CREATE TRIGGER IF NOT EXISTS artifact_promotion_guard
                BEFORE INSERT ON artifact_promotions
                WHEN NOT EXISTS(
                    SELECT 1 FROM qualification_runs AS q
                    WHERE q.id = NEW.qualification_id
                        AND q.subject_kind = 'artifact'
                        AND q.subject_id = NEW.artifact_id
                        AND q.overall_status = 'passed'
                )
                BEGIN SELECT RAISE(ABORT, 'artifact promotion requires passed qualification'); END""",
            """CREATE TRIGGER IF NOT EXISTS expression_promotion_guard
                BEFORE INSERT ON expression_qualifications
                WHEN NOT EXISTS(
                    SELECT 1 FROM qualification_runs AS q
                    WHERE q.id = NEW.qualification_id
                        AND q.subject_kind = 'expression'
                        AND q.subject_id = NEW.expression_id
                        AND q.overall_status = 'passed'
                )
                BEGIN SELECT RAISE(ABORT, 'expression promotion requires passed qualification'); END""",
        )
        for statement in statements:
            if statement.startswith("CREATE TRIGGER IF NOT EXISTS artifact_parents_no_cycle"):
                existing_trigger = connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
                    "AND name = 'artifact_parents_no_cycle'"
                ).fetchone()
                if (
                    existing_trigger is not None
                    and "WHERE parent_artifact_id = NEW.artifact_id"
                    not in str(existing_trigger["sql"])
                ):
                    connection.execute("DROP TRIGGER artifact_parents_no_cycle")
            connection.execute(statement)

    @staticmethod
    def _upgrade_schema(connection: sqlite3.Connection) -> None:
        row = connection.execute("SELECT value FROM metadata WHERE key = 'schema'").fetchone()
        try:
            existing = int(row["value"]) if row is not None else WORKSTATION_SCHEMA
        except (TypeError, ValueError) as error:
            raise WorkstationError("The workstation database schema is malformed") from error
        if existing > WORKSTATION_SCHEMA or existing < 2:
            raise WorkstationError("The workstation database schema is unsupported")
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)")}
        additions = {
            "worker_id": "TEXT",
            "claim_token": "TEXT",
            "lease_expires_at": "TEXT",
            "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
            "pause_requested": "INTEGER NOT NULL DEFAULT 0",
            "priority": "INTEGER NOT NULL DEFAULT 0",
            "attempt": "INTEGER NOT NULL DEFAULT 1",
            "retry_of": "TEXT",
            "wait_reason": "TEXT",
        }
        for name, sql_type in additions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} {sql_type}")
        for job_type, resource_class in JOB_RESOURCE_CLASSES.items():
            connection.execute(
                "UPDATE jobs SET resource_class = ? WHERE type = ? AND resource_class <> ?",
                (resource_class, job_type, resource_class),
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS jobs_lease_expiry ON jobs(status, lease_expires_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS jobs_queue_priority "
            "ON jobs(status, priority DESC, created_at, id)"
        )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS jobs_single_retry_attempt "
            "ON jobs(retry_of) WHERE retry_of IS NOT NULL"
        )
        if existing < 7:
            # v1.4 schema <=6 allowed a UI/API caller to mark an expression as
            # qualified without evidence. Never inherit that trust silently.
            connection.execute(
                "UPDATE expression_drafts SET qualification_status = 'pending', "
                "updated_at = ? WHERE qualification_status = 'qualified'",
                (utc_now(),),
            )
        if existing < 9:
            connection.execute("PRAGMA legacy_alter_table = ON")
            connection.execute("DROP TRIGGER IF EXISTS artifacts_immutable_identity")
            connection.execute("ALTER TABLE artifacts RENAME TO artifacts_schema_v8")
            connection.execute(_ARTIFACTS_TABLE_SQL)
            connection.execute(
                "INSERT INTO artifacts("
                "id, type, name, project_id, status, local_path, sha256, "
                "metadata_json, created_at, updated_at"
                ") SELECT "
                "id, type, name, project_id, status, local_path, sha256, "
                "metadata_json, created_at, updated_at "
                "FROM artifacts_schema_v8"
            )
            connection.execute("DROP TABLE artifacts_schema_v8")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS artifacts_project_updated "
                "ON artifacts(project_id, updated_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS artifacts_type_status "
                "ON artifacts(type, status, updated_at DESC)"
            )
            connection.execute(
                """CREATE TRIGGER IF NOT EXISTS artifacts_immutable_identity
                BEFORE UPDATE OF id, type, local_path, sha256, created_at ON artifacts
                WHEN OLD.id IS NOT NEW.id OR OLD.type IS NOT NEW.type
                    OR OLD.local_path IS NOT NEW.local_path
                    OR OLD.sha256 IS NOT NEW.sha256
                    OR OLD.created_at IS NOT NEW.created_at
                BEGIN SELECT RAISE(ABORT, 'artifact identity is immutable'); END"""
            )
            connection.execute("PRAGMA legacy_alter_table = OFF")
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES('schema', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(WORKSTATION_SCHEMA),),
        )

    def _migrate_legacy_manifest(self) -> None:
        if not self.legacy_manifest_path.is_file():
            return
        try:
            payload = json.loads(self.legacy_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise WorkstationError("The legacy workstation manifest is unreadable") from error
        if not isinstance(payload, dict):
            raise WorkstationError("The legacy workstation manifest is malformed")
        projects, jobs = payload.get("projects", []), payload.get("jobs", [])
        if not isinstance(projects, list) or not isinstance(jobs, list):
            raise WorkstationError("The legacy workstation manifest is malformed")
        migrated = False
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                already = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'legacy_manifest_migrated'"
                ).fetchone()
                if already is None:
                    project_ids: dict[str, str] = {}
                    normalized_projects: list[dict[str, Any]] = []
                    for index, record in enumerate(projects):
                        if not isinstance(record, Mapping):
                            raise WorkstationError(f"Legacy project record {index} is not an object")
                        try:
                            normalized = dict(record)
                            old_id = str(normalized.get("id", ""))
                            kind = str(normalized.get("kind", ""))
                            if kind not in PROJECT_KINDS:
                                raise WorkstationError(f"Unsupported project kind: {kind}")
                            normalized["name"] = _required_text(
                                normalized.get("name"), field="name"
                            )
                            normalized["status"] = _required_text(
                                normalized.get("status", "draft"), field="status", maximum=40
                            )
                            normalized["progress"] = _progress(normalized.get("progress", 0.0))
                            normalized["config"] = _mapping_value(
                                normalized.get("config"), field="config"
                            )
                            normalized["metrics"] = _mapping_value(
                                normalized.get("metrics"), field="metrics"
                            )
                            try:
                                normalized["id"] = _validated_id(
                                    old_id, field="project_id", prefixes=(kind,)
                                )
                            except WorkstationError:
                                normalized["id"] = _new_id(kind)
                            project_ids[old_id] = normalized["id"]
                            normalized_projects.append(normalized)
                        except WorkstationError as error:
                            raise WorkstationError(
                                f"Legacy project record {index} is malformed: {error}"
                            ) from error
                    for normalized in normalized_projects:
                        self._insert_project(connection, normalized, ignore=True)
                    job_ids: dict[str, str] = {}
                    normalized_jobs: list[dict[str, Any]] = []
                    for index, record in enumerate(jobs):
                        if not isinstance(record, Mapping):
                            raise WorkstationError(f"Legacy job record {index} is not an object")
                        try:
                            normalized = dict(record)
                            job_type = str(normalized.get("type", ""))
                            if job_type not in JOB_TYPES:
                                raise WorkstationError(f"Unsupported job type: {job_type}")
                            status = str(normalized.get("status", "queued"))
                            if status not in JOB_STATES:
                                raise WorkstationError(f"Unsupported job status: {status}")
                            normalized["status"] = status
                            normalized["progress"] = _progress(normalized.get("progress", 0.0))
                            normalized["parameters"] = _mapping_value(
                                normalized.get("parameters"), field="parameters"
                            )
                            normalized["result"] = _mapping_value(
                                normalized.get("result"), field="result"
                            )
                            dependencies = normalized.get("depends_on", [])
                            if isinstance(dependencies, (str, bytes)) or not isinstance(dependencies, Sequence):
                                raise WorkstationError("depends_on must be a list of job IDs")
                            normalized["depends_on"] = list(dependencies)
                            old_id = str(normalized.get("id", ""))
                            try:
                                normalized["id"] = _validated_id(
                                    old_id, field="job_id", prefixes=("job",)
                                )
                            except WorkstationError:
                                normalized["id"] = _new_id("job")
                            job_ids[old_id] = normalized["id"]
                            normalized_jobs.append(normalized)
                        except WorkstationError as error:
                            raise WorkstationError(
                                f"Legacy job record {index} is malformed: {error}"
                            ) from error
                    for normalized in normalized_jobs:
                        project_id = normalized.get("project_id")
                        if project_id is not None:
                            normalized["project_id"] = project_ids.get(str(project_id), project_id)
                            if connection.execute(
                                "SELECT 1 FROM projects WHERE id = ?", (normalized["project_id"],)
                            ).fetchone() is None:
                                raise WorkstationError(
                                    f"Legacy job references missing project: {project_id}"
                                )
                        normalized["depends_on"] = [
                            job_ids.get(str(value), value)
                            for value in normalized.get("depends_on", [])
                        ]
                        for dependency_id in normalized["depends_on"]:
                            if dependency_id not in job_ids.values() and connection.execute(
                                "SELECT 1 FROM jobs WHERE id = ?", (dependency_id,)
                            ).fetchone() is None:
                                raise WorkstationError(
                                    f"Legacy job references missing dependency: {dependency_id}"
                                )
                        normalized["resource_class"] = JOB_RESOURCE_CLASSES[normalized["type"]]
                        self._insert_job(connection, normalized, ignore=True)
                    connection.execute(
                        "INSERT INTO metadata(key, value) VALUES('legacy_manifest_migrated', ?)",
                        (utc_now(),),
                    )
                    migrated = True
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        if migrated and self.legacy_manifest_path.exists():
            destination = self.legacy_manifest_path.with_suffix(".json.migrated")
            try:
                self.legacy_manifest_path.replace(destination)
            except FileNotFoundError:
                pass

    @staticmethod
    def _project_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "kind": row["kind"], "name": row["name"],
            "status": row["status"], "progress": float(row["progress"]),
            "config": _decode_json(row["config_json"], fallback={}),
            "metrics": _decode_json(row["metrics_json"], fallback={}),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    @staticmethod
    def _job_record(row: sqlite3.Row) -> dict[str, Any]:
        keys = set(row.keys())
        return {
            "id": row["id"], "type": row["type"], "project_id": row["project_id"],
            "status": row["status"], "progress": float(row["progress"]),
            "parameters": _decode_json(row["parameters_json"], fallback={}),
            "result": _decode_json(row["result_json"], fallback={}),
            "error": row["error"], "resource_class": row["resource_class"],
            "depends_on": _decode_json(row["depends_on_json"], fallback=[]),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "started_at": row["started_at"], "finished_at": row["finished_at"],
            "wait_reason": row["wait_reason"] if "wait_reason" in keys else None,
            "lease_expires_at": row["lease_expires_at"] if "lease_expires_at" in keys else None,
            "cancel_requested": bool(row["cancel_requested"]) if "cancel_requested" in keys else False,
            "pause_requested": bool(row["pause_requested"]) if "pause_requested" in keys else False,
            "priority": int(row["priority"]) if "priority" in keys else 0,
            "attempt": int(row["attempt"]) if "attempt" in keys else 1,
            "retry_of": row["retry_of"] if "retry_of" in keys else None,
        }

    @staticmethod
    def _insert_project(connection: sqlite3.Connection, record: Mapping[str, Any], *, ignore: bool = False) -> None:
        now = utc_now()
        verb = "INSERT OR IGNORE" if ignore else "INSERT"
        connection.execute(
            f"""{verb} INTO projects(
                id, kind, name, status, progress, config_json, metrics_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record.get("id"), record.get("kind"), record.get("name"), record.get("status", "draft"),
             _progress(record.get("progress", 0.0)), _json_text(record.get("config", {}), field="config"),
             _json_text(record.get("metrics", {}), field="metrics"), record.get("created_at", now),
             record.get("updated_at", now)),
        )

    @staticmethod
    def _insert_job(connection: sqlite3.Connection, record: Mapping[str, Any], *, ignore: bool = False) -> None:
        now = utc_now()
        verb = "INSERT OR IGNORE" if ignore else "INSERT"
        connection.execute(
            f"""{verb} INTO jobs(
                id, type, project_id, status, progress, parameters_json, result_json,
                error, wait_reason, resource_class, depends_on_json, created_at, updated_at,
                started_at, finished_at, worker_id, claim_token, lease_expires_at,
                cancel_requested, pause_requested, priority, attempt, retry_of
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record.get("id"), record.get("type"), record.get("project_id"), record.get("status", "queued"),
             _progress(record.get("progress", 0.0)), _json_text(record.get("parameters", {}), field="parameters"),
             _json_text(record.get("result", {}), field="result"), record.get("error"),
             record.get("wait_reason"),
             record.get("resource_class", JOB_RESOURCE_CLASSES.get(str(record.get("type")), "cpu-shared")),
             _json_text(record.get("depends_on", []), field="depends_on"), record.get("created_at", now),
             record.get("updated_at", now), record.get("started_at"), record.get("finished_at"),
             None, None, None, 0, 0, _job_priority(record.get("priority", 0)),
             int(record.get("attempt", 1)), record.get("retry_of")),
        )

    @staticmethod
    def _append_job_log(connection: sqlite3.Connection, job_id: str, level: str, message: str) -> None:
        connection.execute(
            "INSERT INTO job_events(job_id, level, message, created_at) VALUES (?, ?, ?, ?)",
            (job_id, level, message, utc_now()),
        )

    @staticmethod
    def _refresh_project_summary(
        connection: sqlite3.Connection, project_id: str | None
    ) -> None:
        if project_id is None:
            return
        project = connection.execute(
            "SELECT status, progress, metrics_json, updated_at FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        if project is None:
            raise WorkstationError("Project was not found")
        jobs = connection.execute(
            "SELECT id, type, status, progress, cancel_requested, pause_requested, "
            "retry_of, updated_at "
            "FROM jobs WHERE project_id = ? ORDER BY updated_at DESC, rowid DESC",
            (project_id,),
        ).fetchall()
        if not jobs:
            return

        counts = {state: 0 for state in sorted(JOB_STATES)}
        for job in jobs:
            counts[str(job["status"])] += 1
        superseded_ids = {
            str(job["retry_of"]) for job in jobs if job["retry_of"] is not None
        }
        effective_jobs = [job for job in jobs if str(job["id"]) not in superseded_ids]
        effective_counts = {state: 0 for state in sorted(JOB_STATES)}
        for job in effective_jobs:
            effective_counts[str(job["status"])] += 1
        if effective_counts["running"]:
            status = "running"
        elif effective_counts["failed"]:
            status = "failed"
        elif effective_counts["queued"]:
            status = "queued"
        elif effective_counts["paused"]:
            status = "paused"
        elif effective_counts["succeeded"]:
            status = "succeeded"
        elif effective_counts["cancelled"]:
            status = "cancelled"
        else:
            status = "succeeded"

        progress = (
            1.0
            if status == "succeeded"
            else sum(float(job["progress"]) for job in effective_jobs)
            / len(effective_jobs)
        )
        metrics = _decode_json(project["metrics_json"], fallback={})
        if not isinstance(metrics, dict):
            raise WorkstationError("The workstation project metrics are malformed")
        latest = jobs[0]
        metrics["jobs"] = {
            "total": len(jobs),
            **counts,
            "terminal": sum(counts[state] for state in TERMINAL_JOB_STATES),
            "cancel_requested": sum(bool(job["cancel_requested"]) for job in jobs),
            "pause_requested": sum(bool(job["pause_requested"]) for job in jobs),
            "latest": {
                "id": latest["id"],
                "type": latest["type"],
                "status": latest["status"],
            },
        }
        metrics_json = _json_text(metrics, field="metrics")
        updated_at = str(latest["updated_at"])
        if (
            str(project["status"]) == status
            and float(project["progress"]) == progress
            and str(project["metrics_json"]) == metrics_json
            and str(project["updated_at"]) == updated_at
        ):
            return
        connection.execute(
            "UPDATE projects SET status = ?, progress = ?, metrics_json = ?, "
            "updated_at = ? WHERE id = ?",
            (status, progress, metrics_json, updated_at, project_id),
        )

    def reconcile_project_summaries(self) -> int:
        """Repair derived project state without changing projects that have no jobs."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    "SELECT DISTINCT project_id FROM jobs WHERE project_id IS NOT NULL"
                ).fetchall()
                before = connection.total_changes
                for row in rows:
                    self._refresh_project_summary(connection, row["project_id"])
                changed = connection.total_changes - before
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return changed

    def get_settings(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = ?",
                (WORKSTATION_SETTINGS_KEY,),
            ).fetchone()
        if row is None:
            return _workstation_settings(None)
        value = _decode_json(row["value"], fallback=None)
        if not isinstance(value, dict):
            raise WorkstationError("The persisted workstation settings are malformed")
        return _workstation_settings(value)

    def update_settings(self, changes: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(changes, Mapping):
            raise WorkstationError("Workstation settings must be a JSON object")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT value FROM metadata WHERE key = ?",
                    (WORKSTATION_SETTINGS_KEY,),
                ).fetchone()
                current: Mapping[str, Any] | None = None
                if row is not None:
                    decoded = _decode_json(row["value"], fallback=None)
                    if not isinstance(decoded, dict):
                        raise WorkstationError(
                            "The persisted workstation settings are malformed"
                        )
                    current = decoded
                merged = _workstation_settings({**(current or {}), **dict(changes)})
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (
                        WORKSTATION_SETTINGS_KEY,
                        _json_text(merged, field="workstation settings"),
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return merged

    def get_visible_history_cutoff(self) -> str | None:
        """Return the UI presentation cutoff without mutating workstation records."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = ?",
                (WORKSTATION_VISIBLE_HISTORY_KEY,),
            ).fetchone()
        if row is None:
            return None
        value = _decode_json(row["value"], fallback=None)
        if not isinstance(value, str) or not value.endswith("Z"):
            raise WorkstationError("The visible history cutoff is malformed")
        try:
            datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
        except ValueError as error:
            raise WorkstationError("The visible history cutoff is malformed") from error
        return value

    def clear_visible_history(self) -> str:
        """Hide existing UI records while retaining every database row and asset."""

        cutoff = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (
                        WORKSTATION_VISIBLE_HISTORY_KEY,
                        _json_text(cutoff, field="visible history cutoff"),
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return cutoff

    def snapshot(self) -> WorkstationSnapshot:
        self.recover_expired_jobs()
        self.propagate_dependency_failures()
        with self._connect() as connection:
            connection.execute("BEGIN")
            try:
                projects = tuple(
                    self._project_record(row)
                    for row in connection.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
                )
                jobs = tuple(
                    self._job_record(row)
                    for row in connection.execute("SELECT * FROM jobs ORDER BY updated_at DESC").fetchall()
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return WorkstationSnapshot(projects, jobs)

    def list_projects(self, *, kind: str | None = None) -> list[dict[str, Any]]:
        if kind is not None and kind not in PROJECT_KINDS:
            raise WorkstationError(f"Unsupported project kind: {kind}")
        self.recover_expired_jobs()
        self.propagate_dependency_failures()
        query, parameters = "SELECT * FROM projects", ()
        if kind is not None:
            query, parameters = query + " WHERE kind = ?", (kind,)
        with self._connect() as connection:
            rows = connection.execute(query + " ORDER BY updated_at DESC", parameters).fetchall()
        return [self._project_record(row) for row in rows]

    def get_project(self, project_id: str) -> dict[str, Any]:
        project_id = _validated_id(project_id, field="project_id", prefixes=tuple(PROJECT_KINDS))
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            raise WorkstationError("Project was not found")
        return self._project_record(row)

    def create_project(self, *, kind: str, name: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if kind not in PROJECT_KINDS:
            raise WorkstationError(f"Unsupported project kind: {kind}")
        project_config = _mapping_value(config, field="config")
        if kind == "dataset":
            try:
                acquisition = DatasetAcquisitionConfig.from_mapping(project_config)
                if _DATASET_ACQUISITION_CONFIG_KEYS.intersection(project_config):
                    project_config = {**project_config, **acquisition.as_dict()}
            except DatasetAcquisitionError as error:
                raise WorkstationError(str(error)) from error
        record = {
            "id": _new_id(kind), "kind": kind, "name": _required_text(name, field="name"),
            "status": "draft", "progress": 0.0, "config": project_config,
            "metrics": {}, "created_at": utc_now(), "updated_at": utc_now(),
        }
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._insert_project(connection, record)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return record

    def update_project(
        self,
        project_id: str,
        *,
        config: Mapping[str, Any] | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        project = self.get_project(project_id)
        next_config = project["config"] if config is None else _mapping_value(config, field="config")
        if project["kind"] == "dataset":
            try:
                acquisition = DatasetAcquisitionConfig.from_mapping(next_config)
                if _DATASET_ACQUISITION_CONFIG_KEYS.intersection(next_config):
                    next_config = {**next_config, **acquisition.as_dict()}
            except DatasetAcquisitionError as error:
                raise WorkstationError(str(error)) from error
        next_name = project["name"] if name is None else _required_text(name, field="name")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "UPDATE projects SET name = ?, config_json = ?, updated_at = ? WHERE id = ?",
                    (
                        next_name,
                        _json_text(next_config, field="config"),
                        utc_now(),
                        project_id,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get_project(project_id)

    def _artifact_local_path(self, value: str | os.PathLike[str] | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, (str, os.PathLike)):
            raise WorkstationError("local_path must be a filesystem path")
        text = os.fspath(value)
        if not text or not text.strip():
            raise WorkstationError("local_path must not be empty")
        raw = Path(text).expanduser()
        if any(part == ".." for part in raw.parts):
            raise WorkstationError("local_path cannot traverse outside the artifact store")
        root = self.artifact_root.resolve(strict=True)
        candidate = raw.absolute() if raw.is_absolute() else (root / raw).absolute()
        try:
            lexical_relative = candidate.relative_to(root).as_posix()
        except ValueError as error:
            raise WorkstationError(
                "local_path must be contained by the artifact store"
            ) from error
        validated_artifact_relative_path(lexical_relative, field="local_path")
        resolved = _checked_regular_file(root, candidate, field="local_path")
        relative = resolved.relative_to(root).as_posix()
        validated_artifact_relative_path(relative, field="local_path")
        return relative

    @staticmethod
    def _artifact_record(connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        parent_rows = connection.execute(
            "SELECT parent_artifact_id FROM artifact_parents "
            "WHERE artifact_id = ? ORDER BY position",
            (row["id"],),
        ).fetchall()
        metadata = _decode_json(row["metadata_json"], fallback={})
        if not isinstance(metadata, dict):
            raise WorkstationError("The workstation artifact metadata is malformed")
        promotion = connection.execute(
            "SELECT qualification_id, promoted_at FROM artifact_promotions "
            "WHERE artifact_id = ?",
            (row["id"],),
        ).fetchone()
        return {
            "id": row["id"],
            "type": row["type"],
            "name": row["name"],
            "project_id": row["project_id"],
            "status": row["status"],
            "local_path": row["local_path"],
            "sha256": row["sha256"],
            "metadata": metadata,
            "parent_artifact_ids": [parent["parent_artifact_id"] for parent in parent_rows],
            "promoted": promotion is not None,
            "promotion": (
                {
                    "qualification_id": promotion["qualification_id"],
                    "promoted_at": promotion["promoted_at"],
                }
                if promotion is not None
                else None
            ),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _staged_worker_artifacts(
        connection: sqlite3.Connection, job_id: str
    ) -> list[sqlite3.Row]:
        rows = connection.execute(
            "SELECT * FROM artifacts WHERE status = 'building' ORDER BY id"
        ).fetchall()
        staged: list[sqlite3.Row] = []
        for row in rows:
            metadata = _decode_json(row["metadata_json"], fallback={})
            if (
                isinstance(metadata, dict)
                and metadata.get("source") == "linux-docker"
                and metadata.get("job_id") == job_id
            ):
                staged.append(row)
        return staged

    def _verify_staged_worker_artifacts(
        self, rows: Sequence[sqlite3.Row]
    ) -> dict[str, tuple[int, ...]]:
        root = self.artifact_root.resolve(strict=True)
        identities = {}
        for row in rows:
            relative = validated_artifact_relative_path(
                row["local_path"], field="Staged worker artifact path"
            )
            path = _checked_regular_file(
                root,
                root.joinpath(*relative.parts),
                field="Staged worker artifact",
            )
            before_stat = path.stat()
            digest, _size = _sha256_regular_file(
                path, field="Staged worker artifact"
            )
            if digest != row["sha256"]:
                raise WorkstationError(
                    "Staged worker artifact changed before job completion"
                )
            stat = path.stat()
            before_identity = (before_stat.st_dev, before_stat.st_ino, before_stat.st_size, before_stat.st_mtime_ns, before_stat.st_ctime_ns)
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if before_identity != identity:
                raise WorkstationError("Staged worker artifact changed during verification")
            identities[str(row["id"])] = identity
        return identities

    def _accept_staged_verification(self, rows, verified_rows, identities):
        if verified_rows is None or [dict(row) for row in rows] != verified_rows:
            raise WorkstationError("Staged artifacts changed during verification")
        root = self.artifact_root.resolve(strict=True)
        for row in rows:
            relative = validated_artifact_relative_path(row["local_path"], field="Staged worker artifact path")
            path = _checked_regular_file(root, root.joinpath(*relative.parts), field="Staged worker artifact")
            stat = path.stat()
            current = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if identities.get(str(row["id"])) != current:
                raise WorkstationError("Staged worker artifact changed before job completion")

    def _remove_staged_worker_files(self, paths: Sequence[str]) -> None:
        root = self.artifact_root.resolve(strict=True)
        for value in paths:
            try:
                relative = validated_artifact_relative_path(
                    value, field="Staged worker artifact path"
                )
                path = root.joinpath(*relative.parts)
                checked = _checked_regular_file(
                    root, path, field="Staged worker artifact"
                )
                checked.unlink(missing_ok=True)
                parent = checked.parent
                while parent != root:
                    try:
                        parent.rmdir()
                    except OSError:
                        break
                    parent = parent.parent
            except (OSError, WorkstationError):
                # The registry row is already gone, so an unremovable file is
                # quarantined as unreferenced data rather than a published artifact.
                continue

    @staticmethod
    def _validate_artifact_parents(
        connection: sqlite3.Connection, artifact_id: str, parent_ids: Sequence[str]
    ) -> None:
        # Without an incoming edge, no existing parent can reach this node.
        # Self-links and missing parents are still rejected unconditionally.
        needs_cycle_check = connection.execute(
            "SELECT 1 FROM artifact_parents WHERE parent_artifact_id = ? LIMIT 1",
            (artifact_id,),
        ).fetchone() is not None
        for parent_id in parent_ids:
            if parent_id == artifact_id:
                raise WorkstationError("An artifact cannot be its own parent")
            if connection.execute(
                "SELECT 1 FROM artifacts WHERE id = ?", (parent_id,)
            ).fetchone() is None:
                raise WorkstationError(f"Parent artifact was not found: {parent_id}")
            if not needs_cycle_check:
                continue
            cycle = connection.execute(
                """WITH RECURSIVE ancestors(id) AS (
                    SELECT ?
                    UNION
                    SELECT ap.parent_artifact_id
                    FROM artifact_parents AS ap JOIN ancestors AS a
                        ON ap.artifact_id = a.id
                )
                SELECT 1 FROM ancestors WHERE id = ? LIMIT 1""",
                (parent_id, artifact_id),
            ).fetchone()
            if cycle is not None:
                raise WorkstationError("Artifact lineage cannot contain a cycle")

    def register_artifact(
        self,
        *,
        artifact_type: str,
        name: str,
        status: str = "planned",
        project_id: str | None = None,
        local_path: str | os.PathLike[str] | None = None,
        sha256: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        parent_artifact_ids: Sequence[str] | None = None,
        artifact_id: str | None = None,
    ) -> dict[str, Any]:
        if artifact_type not in ARTIFACT_TYPES:
            raise WorkstationError(f"Unsupported artifact type: {artifact_type}")
        if status not in ARTIFACT_STATUSES:
            raise WorkstationError(f"Unsupported artifact status: {status}")
        artifact_id = (
            _validated_id(artifact_id, field="artifact_id", prefixes=("artifact",))
            if artifact_id is not None
            else _new_id("artifact")
        )
        if project_id is not None:
            project_id = _validated_id(
                project_id, field="project_id", prefixes=tuple(PROJECT_KINDS)
            )
        if parent_artifact_ids is not None and (
            isinstance(parent_artifact_ids, (str, bytes))
            or not isinstance(parent_artifact_ids, Sequence)
        ):
            raise WorkstationError("parent_artifact_ids must be a list of artifact IDs")
        parent_ids = [
            _validated_id(value, field="parent_artifact_ids", prefixes=("artifact",))
            for value in (parent_artifact_ids or ())
        ]
        if len(parent_ids) != len(set(parent_ids)):
            raise WorkstationError("parent_artifact_ids contains duplicates")
        local_record = self._artifact_local_path(local_path)
        declared_digest = _sha256_value(sha256)
        if local_record is None:
            if declared_digest is not None:
                raise WorkstationError("sha256 requires an existing local_path file")
            if status == "ready":
                raise WorkstationError("Ready artifacts require an existing local_path file")
            actual_digest = None
        else:
            actual_path = self.artifact_root.joinpath(*PurePosixPath(local_record).parts)
            actual_digest, _size_bytes = _sha256_regular_file(
                actual_path, field="local_path"
            )
            if declared_digest is not None and declared_digest != actual_digest:
                raise WorkstationError("sha256 does not match the artifact file")
        record = {
            "id": artifact_id,
            "type": artifact_type,
            "name": _required_text(name, field="name", maximum=160),
            "project_id": project_id,
            "status": status,
            "local_path": local_record,
            "sha256": actual_digest,
            "metadata": _mapping_value(metadata, field="metadata"),
            "parent_artifact_ids": parent_ids,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        if (artifact_type == "evaluation" and status == "ready"
                and local_record is not None and actual_path.suffix.lower() == ".json"
                and _size_bytes <= 4 * 1024 * 1024):
            try:
                evidence = json.loads(actual_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeError):
                evidence = None
            if isinstance(evidence, Mapping):
                schema = evidence.get("schema")
                if not isinstance(schema, str):
                    schema = ""
                kind = {
                    AUTOMATED_EVALUATION_SCHEMA: "automated-evaluation",
                    SECURITY_EVIDENCE_SCHEMA: "security-verification",
                    CONTENT_REVIEW_SCHEMA: "content-review",
                    QUALIFICATION_REPORT_SCHEMA: "composed-qualification",
                }.get(schema)
                if schema == BLIND_AB_EVIDENCE_SCHEMA:
                    gate = evidence.get("gate")
                    kind = {"long-form-continuity": "long-form-blind-ab",
                            "expression-quality": "expression-blind-ab"}.get(
                                gate if isinstance(gate, str) else "")

                if kind is not None:
                    record["metadata"]["qualification_evidence_kind"] = kind
                    subject = evidence.get("subject")
                    if (isinstance(subject, Mapping)
                            and subject.get("kind") in {"artifact", "expression"}
                            and isinstance(subject.get("id"), str) and len(subject["id"]) <= 80):
                        record["metadata"]["subject_kind"] = subject["kind"]
                        record["metadata"]["subject_id"] = subject["id"]
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if project_id is not None and connection.execute(
                    "SELECT 1 FROM projects WHERE id = ?", (project_id,)
                ).fetchone() is None:
                    raise WorkstationError("Project was not found")
                self._validate_artifact_parents(connection, artifact_id, parent_ids)
                connection.execute(
                    """INSERT INTO artifacts(
                        id, type, name, project_id, status, local_path, sha256,
                        metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        artifact_id,
                        artifact_type,
                        record["name"],
                        project_id,
                        status,
                        record["local_path"],
                        record["sha256"],
                        _json_text(record["metadata"], field="metadata"),
                        record["created_at"],
                        record["updated_at"],
                    ),
                )
                for position, parent_id in enumerate(parent_ids):
                    connection.execute(
                        """INSERT INTO artifact_parents(
                            artifact_id, parent_artifact_id, position, created_at
                        ) VALUES (?, ?, ?, ?)""",
                        (artifact_id, parent_id, position, record["created_at"]),
                    )
                row = connection.execute(
                    "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
                ).fetchone()
                result = self._artifact_record(connection, row)
                connection.commit()
            except sqlite3.IntegrityError as error:
                connection.rollback()
                message = str(error).lower()
                if "cycle" in message:
                    raise WorkstationError("Artifact lineage cannot contain a cycle") from error
                if "unique" in message or "primary key" in message:
                    raise WorkstationError("Artifact ID already exists") from error
                raise WorkstationError("Artifact registration violated the registry contract") from error
            except Exception:
                connection.rollback()
                raise
        return result

    def register_worker_artifacts(
        self,
        *,
        job_id: str,
        claim_token: str,
        job_type: str,
        project_id: str,
        source_root: Path,
        artifacts: Sequence[Mapping[str, Any]],
        image_digest: str,
        parent_artifact_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Import trusted Docker outputs and atomically add immutable lineage.

        The Docker manifest is only a claim. This host-side boundary reopens
        every source, copies it into the workstation-owned artifact store,
        computes the digest again, and only then commits registry rows.
        """

        job_id = _validated_id(job_id, field="job_id", prefixes=("job",))
        claim_token = _validated_id(
            claim_token, field="claim_token", prefixes=("claim",)
        )
        project_id = _validated_id(
            project_id, field="project_id", prefixes=tuple(PROJECT_KINDS)
        )
        expected_type = _WORKER_ARTIFACT_TYPE.get(job_type)
        if expected_type is None:
            raise WorkstationError("This job type cannot publish Docker artifacts")
        if not isinstance(image_digest, str) or not image_digest.startswith("sha256:"):
            raise WorkstationError("Docker image digest is malformed")
        image_sha256 = _sha256_value(image_digest.removeprefix("sha256:"))
        if artifacts is None or isinstance(artifacts, (str, bytes)) or not isinstance(
            artifacts, Sequence
        ):
            raise WorkstationError("Docker artifacts must be a list")
        limit = worker_artifact_limit(job_type)
        if not 1 <= len(artifacts) <= limit:
            raise WorkstationError(f"Docker artifacts must contain between 1 and {limit} files")
        if parent_artifact_ids is not None and (
            isinstance(parent_artifact_ids, (str, bytes))
            or not isinstance(parent_artifact_ids, Sequence)
        ):
            raise WorkstationError("parent_artifact_ids must be a list of artifact IDs")
        parent_ids = [
            _validated_id(value, field="parent_artifact_ids", prefixes=("artifact",))
            for value in (parent_artifact_ids or ())
        ]
        if len(parent_ids) != len(set(parent_ids)):
            raise WorkstationError("parent_artifact_ids contains duplicates")

        with self._connect() as connection:
            job_row = self._owned_running_row(connection, job_id, claim_token)
            if job_row["type"] != job_type or job_row["project_id"] != project_id:
                raise WorkstationError(
                    "Docker artifact job does not match its type and project"
                )
            if bool(job_row["cancel_requested"]):
                raise WorkstationError("Docker artifact job cancellation was requested")

        source_candidate = Path(source_root).expanduser()
        if not source_candidate.is_absolute() or _is_reparse_point(source_candidate):
            raise WorkstationError("Docker artifact source root must be a real absolute directory")
        try:
            source = source_candidate.resolve(strict=True)
        except OSError as error:
            raise WorkstationError("Docker artifact source root does not exist") from error
        if not source.is_dir():
            raise WorkstationError("Docker artifact source root must be a directory")

        prepared: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for index, item in enumerate(artifacts):
            if not isinstance(item, Mapping) or set(item) != {
                "kind", "relative_path", "sha256", "size_bytes"
            }:
                raise WorkstationError(f"Docker artifact {index} is malformed")
            if item["kind"] != expected_type:
                raise WorkstationError(
                    f"Docker artifact {index} has type {item['kind']!r}; "
                    f"{job_type} may publish only {expected_type!r}"
                )
            relative_value = item["relative_path"]
            relative = validated_artifact_relative_path(
                relative_value, field=f"Docker artifact {index} path"
            )
            if relative_value in seen_paths:
                raise WorkstationError("Docker artifact paths contain duplicates")
            seen_paths.add(relative_value)
            declared_digest = _sha256_value(item["sha256"])
            size_value = item["size_bytes"]
            if (
                not isinstance(size_value, int)
                or isinstance(size_value, bool)
                or not 0 <= size_value <= 1024 * 1024 * 1024 * 1024
            ):
                raise WorkstationError(f"Docker artifact {index} size is malformed")
            source_path = _checked_regular_file(
                source,
                source.joinpath(*relative.parts),
                field=f"Docker artifact {index}",
            )
            actual_digest, actual_size = _sha256_regular_file(
                source_path, field=f"Docker artifact {index}"
            )
            if actual_digest != declared_digest:
                raise WorkstationError(f"Docker artifact {index} has the wrong checksum")
            if actual_size != size_value:
                raise WorkstationError(f"Docker artifact {index} has the wrong size")
            artifact_uuid = uuid5(
                NAMESPACE_URL,
                f"aniflive-tts-worker:{job_id}:{relative_value}:{actual_digest}",
            )
            artifact_id = f"artifact_{artifact_uuid}"
            file_name = relative.name
            destination = (
                self.artifact_root / "worker" / job_id / artifact_id / file_name
            )
            prepared.append(
                {
                    "id": artifact_id,
                    "type": expected_type,
                    "name": _required_text(file_name, field="artifact name", maximum=160),
                    "project_id": project_id,
                    "status": "building",
                    "source_path": source_path,
                    "destination": destination,
                    "sha256": actual_digest,
                    "size_bytes": actual_size,
                    "metadata": {
                        "source": "linux-docker",
                        "job_id": job_id,
                        "job_type": job_type,
                        "image_digest": f"sha256:{image_sha256}",
                        "worker_kind": expected_type,
                        "worker_relative_path": relative_value,
                        "size_bytes": actual_size,
                        **(
                            {"qualification_evidence_kind": "automated-evaluation"}
                            if job_type == "evaluation.prepare"
                            and relative.name == "evaluation-report.json"
                            else {}
                        ),
                    },
                }
            )

        created_files: list[Path] = []
        created_directories: list[Path] = []
        try:
            for item in prepared:
                destination = item["destination"]
                if destination.exists():
                    existing = _checked_regular_file(
                        self.artifact_root, destination, field="Imported artifact"
                    )
                    digest, size = _sha256_regular_file(
                        existing, field="Imported artifact"
                    )
                    if digest != item["sha256"] or size != item["size_bytes"]:
                        raise WorkstationError(
                            "An existing imported artifact does not match the worker output"
                        )
                    continue
                imported_directories = _copy_verified_file(
                    item["source_path"],
                    destination,
                    artifact_root=self.artifact_root,
                    expected_sha256=item["sha256"],
                    expected_size=item["size_bytes"],
                )
                created_files.append(destination)
                created_directories.extend(imported_directories)
                stored = _checked_regular_file(
                    self.artifact_root, destination, field="Imported artifact"
                )
                stored_digest, stored_size = _sha256_regular_file(
                    stored, field="Imported artifact"
                )
                if stored_digest != item["sha256"] or stored_size != item["size_bytes"]:
                    raise WorkstationError("Imported artifact verification failed")

            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    if connection.execute(
                        "SELECT 1 FROM projects WHERE id = ?", (project_id,)
                    ).fetchone() is None:
                        raise WorkstationError("Project was not found")
                    job_row = self._owned_running_row(
                        connection, job_id, claim_token
                    )
                    if job_row["type"] != job_type or job_row["project_id"] != project_id:
                        raise WorkstationError(
                            "Docker artifact job does not match its type and project"
                        )
                    if bool(job_row["cancel_requested"]):
                        raise WorkstationError(
                            "Docker artifact job cancellation was requested"
                        )
                    self._validate_artifact_parents(
                        connection, prepared[0]["id"], parent_ids
                    )
                    results: list[dict[str, Any]] = []
                    now = utc_now()
                    for item in prepared:
                        existing = connection.execute(
                            "SELECT * FROM artifacts WHERE id = ?", (item["id"],)
                        ).fetchone()
                        relative_path = item["destination"].relative_to(
                            self.artifact_root.resolve(strict=True)
                        ).as_posix()
                        if existing is not None:
                            record = self._artifact_record(connection, existing)
                            if (
                                record["type"] != item["type"]
                                or record["name"] != item["name"]
                                or record["project_id"] != project_id
                                or record["status"] != "building"
                                or record["local_path"] != relative_path
                                or record["sha256"] != item["sha256"]
                                or record["metadata"] != item["metadata"]
                                or record["parent_artifact_ids"] != parent_ids
                            ):
                                raise WorkstationError(
                                    "Existing Docker artifact lineage does not match this run"
                                )
                            results.append(record)
                            continue
                        self._validate_artifact_parents(
                            connection, item["id"], parent_ids
                        )
                        connection.execute(
                            """INSERT INTO artifacts(
                                id, type, name, project_id, status, local_path, sha256,
                                metadata_json, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, 'building', ?, ?, ?, ?, ?)""",
                            (
                                item["id"], item["type"], item["name"], project_id,
                                relative_path, item["sha256"],
                                _json_text(item["metadata"], field="metadata"), now, now,
                            ),
                        )
                        for position, parent_id in enumerate(parent_ids):
                            connection.execute(
                                """INSERT INTO artifact_parents(
                                    artifact_id, parent_artifact_id, position, created_at
                                ) VALUES (?, ?, ?, ?)""",
                                (item["id"], parent_id, position, now),
                            )
                        row = connection.execute(
                            "SELECT * FROM artifacts WHERE id = ?", (item["id"],)
                        ).fetchone()
                        results.append(self._artifact_record(connection, row))
                    connection.commit()
                    return results
                except Exception:
                    connection.rollback()
                    raise
        except Exception:
            for path in reversed(created_files):
                path.unlink(missing_ok=True)
            for directory in reversed(created_directories):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            raise

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        artifact_id = _validated_id(
            artifact_id, field="artifact_id", prefixes=("artifact",)
        )
        with self._connect() as connection:
            connection.execute("BEGIN")
            try:
                row = connection.execute(
                    "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
                ).fetchone()
                if row is None:
                    raise WorkstationError("Artifact was not found")
                result = self._artifact_record(connection, row)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return result

    def list_artifacts(
        self,
        *,
        artifact_type: str | None = None,
        status: str | None = None,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if artifact_type is not None and artifact_type not in ARTIFACT_TYPES:
            raise WorkstationError(f"Unsupported artifact type: {artifact_type}")
        if status is not None and status not in ARTIFACT_STATUSES:
            raise WorkstationError(f"Unsupported artifact status: {status}")
        if project_id is not None:
            project_id = _validated_id(
                project_id, field="project_id", prefixes=tuple(PROJECT_KINDS)
            )
        clauses: list[str] = []
        parameters: list[str] = []
        if artifact_type is not None:
            clauses.append("type = ?")
            parameters.append(artifact_type)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        if project_id is not None:
            clauses.append("project_id = ?")
            parameters.append(project_id)
        query = "SELECT * FROM artifacts"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, id"
        with self._connect() as connection:
            connection.execute("BEGIN")
            try:
                rows = connection.execute(query, parameters).fetchall()
                results = [self._artifact_record(connection, row) for row in rows]
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return results

    @staticmethod
    def _reference_root_id(root: Path) -> str:
        normalized = os.path.normcase(str(root.resolve())).casefold().encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()[:16]

    def _expression_reference(
        self, value: str | os.PathLike[str] | None
    ) -> dict[str, str] | None:
        if value is None:
            return None
        if not isinstance(value, (str, os.PathLike)):
            raise WorkstationError("reference_path must be a filesystem path")
        text = os.fspath(value).strip()
        if not text:
            raise WorkstationError("reference_path must not be empty")
        raw = Path(text).expanduser()
        if any(part == ".." for part in raw.parts):
            raise WorkstationError("reference_path cannot traverse outside an allowed root")
        artifact_root = self.artifact_root.resolve(strict=True)
        candidate = raw.absolute() if raw.is_absolute() else (artifact_root / raw).absolute()
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, ValueError) as error:
            raise WorkstationError("Expression reference audio does not exist") from error
        if not resolved.is_file():
            raise WorkstationError("Expression reference must be an audio file")
        if resolved.suffix.lower() not in REFERENCE_AUDIO_SUFFIXES:
            raise WorkstationError("Expression reference must use a supported audio extension")
        roots = (artifact_root, *self.allowed_import_roots())
        eligible = [
            root for root in roots if resolved != root and resolved.is_relative_to(root)
        ]
        if not eligible:
            raise WorkstationError("Expression reference is outside the allowed local roots")
        root = artifact_root if artifact_root in eligible else max(eligible, key=lambda item: len(item.parts))
        if not candidate.is_relative_to(root):
            raise WorkstationError("Expression reference reached an allowed root through a link")
        relative = candidate.relative_to(root)
        current = root
        if _is_reparse_point(root):
            raise WorkstationError("Expression reference roots cannot be reparse points")
        for part in relative.parts:
            current = current / part
            if _is_reparse_point(current):
                raise WorkstationError(
                    "Expression reference cannot contain a symbolic link or reparse point"
                )
        try:
            before = resolved.stat()
            if before.st_size > MAX_EXPRESSION_REFERENCE_BYTES:
                raise WorkstationError("Expression reference exceeds the local metadata limit")
            digest = hashlib.sha256()
            with resolved.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            after = resolved.stat()
        except WorkstationError:
            raise
        except OSError as error:
            raise WorkstationError("Expression reference could not be read") from error
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_identity != after_identity:
            raise WorkstationError("Expression reference changed while it was being registered")
        sha256 = digest.hexdigest()
        if root == artifact_root:
            return {
                "scope": "artifact",
                "path": resolved.relative_to(root).as_posix(),
                "sha256": sha256,
            }
        return {
            "scope": "import-root",
            "root_id": self._reference_root_id(root),
            "path": resolved.relative_to(root).as_posix(),
            "sha256": sha256,
        }

    def _registered_expression_reference_path(
        self, reference: Mapping[str, Any]
    ) -> Path:
        scope = reference.get("scope")
        if scope == "artifact":
            root = self.artifact_root.resolve(strict=True)
        elif scope == "import-root":
            root_id = reference.get("root_id")
            matches = [
                item
                for item in self.allowed_import_roots()
                if self._reference_root_id(item) == root_id
            ]
            if len(matches) != 1:
                raise WorkstationError(
                    "The registered expression import root is unavailable"
                )
            root = matches[0]
        else:
            raise WorkstationError("Expression reference identity is malformed")
        relative = validated_artifact_relative_path(
            reference.get("path"), field="Expression reference path"
        )
        if _is_reparse_point(root):
            raise WorkstationError(
                "Expression reference roots cannot be reparse points"
            )
        current = root
        for part in relative.parts:
            current = current / part
            if _is_reparse_point(current):
                raise WorkstationError(
                    "Expression reference cannot contain a symbolic link or reparse point"
                )
        try:
            resolved = current.resolve(strict=True)
        except (OSError, ValueError) as error:
            raise WorkstationError("Expression reference audio does not exist") from error
        if not resolved.is_file() or not resolved.is_relative_to(root):
            raise WorkstationError("Expression reference escaped its registered root")
        if resolved.suffix.lower() not in REFERENCE_AUDIO_SUFFIXES:
            raise WorkstationError("Expression reference has an unsupported audio extension")
        return resolved

    @staticmethod
    def _expression_record(
        connection: sqlite3.Connection, row: sqlite3.Row
    ) -> dict[str, Any]:
        descriptions = _decode_json(row["descriptions_json"], fallback=[])
        vad = _decode_json(row["vad_json"], fallback={})
        prosody = _decode_json(row["prosody_json"], fallback={})
        if not isinstance(descriptions, list) or not all(
            isinstance(value, str) for value in descriptions
        ):
            raise WorkstationError("The workstation expression descriptions are malformed")
        if not isinstance(vad, dict) or not isinstance(prosody, dict):
            raise WorkstationError("The workstation expression metadata is malformed")
        reference = None
        if row["reference_scope"] is not None:
            reference = {
                "scope": row["reference_scope"],
                "path": row["reference_path"],
                "sha256": _sha256_value(row["reference_sha256"]),
            }
            if row["reference_root_id"] is not None:
                reference["root_id"] = row["reference_root_id"]
        qualification = connection.execute(
            "SELECT qualification_id, promoted_at FROM expression_qualifications "
            "WHERE expression_id = ?",
            (row["id"],),
        ).fetchone()
        return {
            "id": row["id"],
            "name": row["name"],
            "profile_id": _profile_id(row["profile_id"]),
            "model_id": row["model_id"],
            "reference": reference,
            "language": _language_code(row["language"]),
            "emotion": row["emotion"],
            "intensity": _bounded_float(
                row["intensity"], field="intensity", minimum=0.0, maximum=1.0
            ),
            "descriptions": descriptions,
            "vad": _vad_value(vad),
            "prosody": _mapping_value(prosody, field="prosody"),
            "qualification_status": _expression_qualification_status(
                row["qualification_status"]
            ),
            "qualification": (
                {
                    "qualification_id": qualification["qualification_id"],
                    "promoted_at": qualification["promoted_at"],
                }
                if qualification is not None
                else None
            ),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create_expression_draft(
        self,
        *,
        name: str,
        profile_id: str,
        language: str,
        emotion: str,
        intensity: float,
        model_id: str | None = None,
        reference_path: str | os.PathLike[str] | None = None,
        descriptions: Sequence[str] | None = None,
        vad: Mapping[str, Any] | None = None,
        prosody: Mapping[str, Any] | None = None,
        qualification_status: str = "draft",
    ) -> dict[str, Any]:
        qualification_status = _expression_qualification_status(qualification_status)
        if qualification_status == "qualified":
            raise WorkstationError(
                "Qualified expressions require passed evaluation evidence and promotion"
            )
        reference = self._expression_reference(reference_path)
        now = utc_now()
        record = {
            "id": _new_id("expression"),
            "name": _required_text(name, field="name", maximum=160),
            "profile_id": _profile_id(profile_id),
            "model_id": (
                _required_text(model_id, field="model_id", maximum=160)
                if model_id is not None
                else None
            ),
            "reference": reference,
            "language": _language_code(language),
            "emotion": _required_text(emotion, field="emotion", maximum=120),
            "intensity": _bounded_float(
                intensity, field="intensity", minimum=0.0, maximum=1.0
            ),
            "descriptions": _descriptions_value(descriptions),
            "vad": _vad_value(vad),
            "prosody": _editable_expression_prosody(prosody),
            "qualification_status": qualification_status,
            "created_at": now,
            "updated_at": now,
        }
        reference_scope = reference["scope"] if reference is not None else None
        reference_root_id = reference.get("root_id") if reference is not None else None
        reference_relative = reference["path"] if reference is not None else None
        reference_sha256 = reference["sha256"] if reference is not None else None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """INSERT INTO expression_drafts(
                        id, name, profile_id, model_id, reference_scope, reference_root_id,
                        reference_path, reference_sha256, language, emotion, intensity, descriptions_json,
                        vad_json, prosody_json, qualification_status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record["id"], record["name"], record["profile_id"], record["model_id"],
                        reference_scope, reference_root_id, reference_relative, reference_sha256,
                        record["language"], record["emotion"], record["intensity"],
                        _json_text(record["descriptions"], field="descriptions"),
                        _json_text(record["vad"], field="vad"),
                        _json_text(record["prosody"], field="prosody"),
                        qualification_status, now, now,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM expression_drafts WHERE id = ?", (record["id"],)
                ).fetchone()
                result = self._expression_record(connection, row)
                connection.commit()
            except sqlite3.IntegrityError as error:
                connection.rollback()
                if "unique" in str(error).lower():
                    raise WorkstationError(
                        "An expression profile already exists for this model"
                    ) from error
                raise WorkstationError(
                    "Expression draft violated the metadata contract"
                ) from error
            except Exception:
                connection.rollback()
                raise
        return result

    def get_expression_draft(self, expression_id: str) -> dict[str, Any]:
        expression_id = _validated_id(
            expression_id, field="expression_id", prefixes=("expression",)
        )
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM expression_drafts WHERE id = ?", (expression_id,)
            ).fetchone()
            if row is None:
                raise WorkstationError("Expression draft was not found")
            return self._expression_record(connection, row)

    def list_expression_drafts(
        self,
        *,
        model_id: str | None = None,
        language: str | None = None,
        emotion: str | None = None,
        qualification_status: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if model_id is not None:
            clauses.append("model_id = ?")
            parameters.append(_required_text(model_id, field="model_id", maximum=160))
        if language is not None:
            clauses.append("language = ?")
            parameters.append(_language_code(language))
        if emotion is not None:
            clauses.append("emotion = ?")
            parameters.append(_required_text(emotion, field="emotion", maximum=120))
        if qualification_status is not None:
            qualification_status = _expression_qualification_status(qualification_status)
            clauses.append("qualification_status = ?")
            parameters.append(qualification_status)
        query = "SELECT * FROM expression_drafts"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
            return [self._expression_record(connection, row) for row in rows]

    def update_expression_draft(
        self,
        expression_id: str,
        *,
        name: str | object = _UNSET,
        language: str | object = _UNSET,
        emotion: str | object = _UNSET,
        intensity: float | object = _UNSET,
        descriptions: Sequence[str] | None | object = _UNSET,
        vad: Mapping[str, Any] | None | object = _UNSET,
        prosody: Mapping[str, Any] | None | object = _UNSET,
        qualification_status: str | object = _UNSET,
    ) -> dict[str, Any]:
        expression_id = _validated_id(
            expression_id, field="expression_id", prefixes=("expression",)
        )
        changes: dict[str, Any] = {}
        editable_prosody: dict[str, Any] | None = None
        if name is not _UNSET:
            changes["name"] = _required_text(name, field="name", maximum=160)
        if language is not _UNSET:
            changes["language"] = _language_code(language)
        if emotion is not _UNSET:
            changes["emotion"] = _required_text(emotion, field="emotion", maximum=120)
        if intensity is not _UNSET:
            changes["intensity"] = _bounded_float(
                intensity, field="intensity", minimum=0.0, maximum=1.0
            )
        if descriptions is not _UNSET:
            changes["descriptions_json"] = _json_text(
                _descriptions_value(descriptions), field="descriptions"
            )
        if vad is not _UNSET:
            changes["vad_json"] = _json_text(_vad_value(vad), field="vad")
        if prosody is not _UNSET:
            editable_prosody = _editable_expression_prosody(prosody)
        if qualification_status is not _UNSET:
            if qualification_status == "qualified":
                raise WorkstationError(
                    "Qualified expressions require passed evaluation evidence and promotion"
                )
            changes["qualification_status"] = _expression_qualification_status(
                qualification_status
            )
        if not changes and editable_prosody is None:
            raise WorkstationError("At least one mutable expression field must be provided")
        changes["updated_at"] = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current_row = connection.execute(
                    "SELECT * FROM expression_drafts WHERE id = ?", (expression_id,)
                ).fetchone()
                if current_row is None:
                    raise WorkstationError("Expression draft was not found")
                if editable_prosody is not None:
                    existing_prosody = _mapping_value(
                        _decode_json(current_row["prosody_json"], fallback={}),
                        field="prosody",
                    )
                    existing_transcript = existing_prosody.get("reference_transcript")
                    new_transcript = editable_prosody.get("reference_transcript")
                    if (
                        existing_transcript == new_transcript
                        and EXPRESSION_REFERENCE_ANALYSIS_KEY in existing_prosody
                    ):
                        editable_prosody[EXPRESSION_REFERENCE_ANALYSIS_KEY] = (
                            existing_prosody[EXPRESSION_REFERENCE_ANALYSIS_KEY]
                        )
                    changes["prosody_json"] = _json_text(
                        editable_prosody, field="prosody"
                    )
                assignments = ", ".join(f"{field} = ?" for field in changes)
                connection.execute(
                    f"UPDATE expression_drafts SET {assignments} WHERE id = ?",
                    (*changes.values(), expression_id),
                )
                row = connection.execute(
                    "SELECT * FROM expression_drafts WHERE id = ?", (expression_id,)
                ).fetchone()
                result = self._expression_record(connection, row)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return result

    def analyze_expression_reference(self, expression_id: str) -> dict[str, Any]:
        """Measure a registered reference without inferring affective semantics."""

        from .expression_reference_analysis import (
            ExpressionReferenceAnalysisError,
            analyze_reference_file,
        )

        expression_id = _validated_id(
            expression_id, field="expression_id", prefixes=("expression",)
        )
        record = self.get_expression_draft(expression_id)
        reference = record.get("reference")
        if not isinstance(reference, Mapping):
            raise WorkstationError(
                "Expression draft has no registered reference audio"
            )
        reference_path = self._registered_expression_reference_path(reference)
        prosody = record.get("prosody")
        transcript = (
            prosody.get("reference_transcript")
            if isinstance(prosody, Mapping)
            else None
        )
        if transcript is not None and not isinstance(transcript, str):
            raise WorkstationError("prosody.reference_transcript must be a string")
        try:
            analysis = analyze_reference_file(
                reference_path,
                expected_sha256=_sha256_value(reference.get("sha256")),
                language=record["language"],
                transcript=transcript,
            )
        except ExpressionReferenceAnalysisError as error:
            raise WorkstationError(str(error)) from error
        analysis["provenance"]["analyzed_at"] = utc_now()

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM expression_drafts WHERE id = ?", (expression_id,)
                ).fetchone()
                if row is None:
                    raise WorkstationError("Expression draft was not found")
                current = self._expression_record(connection, row)
                current_reference = current.get("reference")
                if (
                    not isinstance(current_reference, Mapping)
                    or current_reference.get("sha256") != analysis["reference_sha256"]
                ):
                    raise WorkstationError(
                        "Expression reference changed before analysis could be saved"
                    )
                merged = dict(current["prosody"])
                merged[EXPRESSION_REFERENCE_ANALYSIS_KEY] = analysis
                now = utc_now()
                connection.execute(
                    "UPDATE expression_drafts SET prosody_json = ?, updated_at = ? "
                    "WHERE id = ?",
                    (_json_text(merged, field="prosody"), now, expression_id),
                )
                refreshed = connection.execute(
                    "SELECT * FROM expression_drafts WHERE id = ?", (expression_id,)
                ).fetchone()
                result = self._expression_record(connection, refreshed)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return result

    def delete_expression_draft(self, expression_id: str) -> dict[str, Any]:
        expression_id = _validated_id(
            expression_id, field="expression_id", prefixes=("expression",)
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM expression_drafts WHERE id = ?", (expression_id,)
                ).fetchone()
                if row is None:
                    raise WorkstationError("Expression draft was not found")
                record = self._expression_record(connection, row)
                connection.execute("DELETE FROM expression_drafts WHERE id = ?", (expression_id,))
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return record

    @staticmethod
    def _qualification_record(
        connection: sqlite3.Connection, row: sqlite3.Row
    ) -> dict[str, Any]:
        gate_rows = connection.execute(
            "SELECT gate_id, status, summary, metrics_json, evidence_json "
            "FROM qualification_gate_results WHERE qualification_id = ? "
            "ORDER BY gate_id",
            (row["id"],),
        ).fetchall()
        gates: list[dict[str, Any]] = []
        for gate in gate_rows:
            metrics = _decode_json(gate["metrics_json"], fallback={})
            evidence = _decode_json(gate["evidence_json"], fallback={})
            if not isinstance(metrics, dict) or not isinstance(evidence, dict):
                raise WorkstationError("Qualification gate evidence is malformed")
            gates.append(
                {
                    "id": gate["gate_id"],
                    "status": gate["status"],
                    "summary": gate["summary"],
                    "metrics": metrics,
                    "evidence": evidence,
                }
            )
        run_metadata = _decode_json(row["run_metadata_json"], fallback={})
        if not isinstance(run_metadata, dict):
            raise WorkstationError("Qualification run metadata is malformed")
        return {
            "id": row["id"],
            "subject_kind": row["subject_kind"],
            "subject_id": row["subject_id"],
            "evaluation_artifact_id": row["evaluation_artifact_id"],
            "report_sha256": row["report_sha256"],
            "report_schema": row["report_schema"],
            "overall_status": row["overall_status"],
            "run_metadata": run_metadata,
            "gates": gates,
            "created_at": row["created_at"],
        }

    def _verified_json_evaluation_artifact(
        self, evaluation: Mapping[str, Any]
    ) -> tuple[dict[str, Any], str]:
        if evaluation.get("type") != "evaluation":
            raise WorkstationError("Qualification evidence must be an evaluation artifact")
        if evaluation.get("status") != "ready":
            raise WorkstationError("Qualification evaluation artifact is not ready")
        local_path = evaluation.get("local_path")
        declared_sha256 = evaluation.get("sha256")
        if not isinstance(local_path, str) or not isinstance(declared_sha256, str):
            raise WorkstationError("Qualification evaluation artifact has no verified file")
        relative = validated_artifact_relative_path(
            local_path, field="Qualification evaluation path"
        )
        if relative.suffix.lower() != ".json":
            raise WorkstationError("Qualification evaluation artifact must be JSON")
        root = self.artifact_root.resolve(strict=True)
        path = _checked_regular_file(
            root,
            root.joinpath(*relative.parts),
            field="Qualification evaluation artifact",
        )
        actual_sha256, size_bytes = _sha256_regular_file(
            path, field="Qualification evaluation artifact"
        )
        if actual_sha256 != declared_sha256:
            raise WorkstationError("Qualification evaluation artifact changed after registration")
        if size_bytes > 4 * 1024 * 1024:
            raise WorkstationError("Qualification evaluation artifact is too large")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise WorkstationError(f"Qualification evidence is invalid: {error}") from error
        if not isinstance(payload, dict):
            raise WorkstationError("Qualification evidence must be a JSON object")
        return payload, actual_sha256

    def _verify_content_review(self, payload, automated_artifact, expected_subject):
        try:
            review = parse_content_review(payload)
        except QualificationReportError as error:
            raise WorkstationError(f"Content review is invalid: {error}") from error
        if review["subject"] != expected_subject:
            raise WorkstationError("Content review belongs to another subject")
        automated, digest = self._verified_json_evaluation_artifact(automated_artifact)
        if review["automated_evaluation_sha256"] != digest:
            raise WorkstationError("Content review belongs to another automated evaluation")
        job_id = automated_artifact.get("metadata", {}).get("job_id")
        if not isinstance(job_id, str) or not job_id.startswith("job_"):
            raise WorkstationError("Content review requires traceable worker audio")
        for decision in review["reviews"]:
            audio = self.get_artifact(decision["audio_artifact_id"])
            expected_path = automated.get("languages", {}).get(
                decision["language"], {}).get("complete_path")
            metadata = audio.get("metadata", {})
            if (audio.get("type") != "evaluation" or audio.get("status") != "ready"
                    or metadata.get("job_id") != job_id
                    or not expected_path or metadata.get("worker_relative_path") != expected_path
                    or audio.get("sha256") != decision["audio_sha256"]):
                raise WorkstationError("Reviewed audio does not belong to this evaluation case")
            relative = validated_artifact_relative_path(
                audio.get("local_path"), field="Reviewed audio path")
            root = self.artifact_root.resolve(strict=True)
            path = _checked_regular_file(root, root.joinpath(*relative.parts),
                                         field="Reviewed audio")
            actual, _size = _sha256_regular_file(path, field="Reviewed audio")
            if actual != decision["audio_sha256"]:
                raise WorkstationError("Reviewed audio changed after the human decision")
        return review

    def _verified_evaluation_report(
        self, evaluation: Mapping[str, Any]
    ) -> dict[str, Any]:
        payload, _digest = self._verified_json_evaluation_artifact(evaluation)
        try:
            report = parse_composed_qualification_report(payload)
            composition = report["run_metadata"]["composition"]
            sources = composition["sources"]
            source_payloads: dict[str, dict[str, Any]] = {}
            for source_key, source_ref in sources.items():
                source_artifact = self.get_artifact(source_ref["artifact_id"])
                source_payload, digest = self._verified_json_evaluation_artifact(
                    source_artifact
                )
                if digest != source_ref["sha256"]:
                    raise WorkstationError(
                        f"Qualification source {source_key} changed after composition"
                    )
                if source_payload.get("schema") != source_ref["schema"]:
                    raise WorkstationError(
                        f"Qualification source {source_key} schema changed after composition"
                    )
                source_payloads[source_key] = source_payload
            if "content" in source_payloads:
                self._verify_content_review(
                    source_payloads["content"],
                    self.get_artifact(sources["automated"]["artifact_id"]),
                    report["subject"],
                )
            recomposed = compose_qualification_report(
                subject=report["subject"],
                automated_evaluation=source_payloads["automated"],
                content_evidence=source_payloads.get("content"),
                long_form_evidence=source_payloads["long_form"],
                expression_evidence=source_payloads.get("expression"),
                security_evidence=source_payloads["security"],
                sources=sources,
            )
            if parse_composed_qualification_report(recomposed) != report:
                raise WorkstationError(
                    "Qualification report does not match its verified source evidence"
                )
        except (KeyError, QualificationReportError) as error:
            raise WorkstationError(f"Qualification report is invalid: {error}") from error
        return report

    @staticmethod
    def _qualification_source_reference(
        artifact: Mapping[str, Any], *, schema: str
    ) -> dict[str, str]:
        artifact_id = artifact.get("id")
        digest = artifact.get("sha256")
        if not isinstance(artifact_id, str) or not isinstance(digest, str):
            raise WorkstationError("Qualification source has no immutable identity")
        return {"artifact_id": artifact_id, "sha256": digest, "schema": schema}

    def compose_qualification(
        self,
        *,
        subject_kind: str,
        subject_id: str,
        automated_evaluation_artifact_id: str,
        long_form_evidence_artifact_id: str,
        expression_evidence_artifact_id: str | None = None,
        security_evidence_artifact_id: str,
        content_evidence_artifact_id: str | None = None,
    ) -> dict[str, Any]:
        """Create and import a fail-closed qualification from verified artifacts."""

        if subject_kind == "artifact":
            subject_id = _validated_id(
                subject_id, field="subject_id", prefixes=("artifact",)
            )
            subject = self.get_artifact(subject_id)
            if subject.get("status") != "ready":
                raise WorkstationError("Qualification subject artifact is not ready")
        elif subject_kind == "expression":
            subject_id = _validated_id(
                subject_id, field="subject_id", prefixes=("expression",)
            )
            subject = self.get_expression_draft(subject_id)
        else:
            raise WorkstationError("Qualification subject kind is unsupported")
        source_ids = {
            "automated": _validated_id(
                automated_evaluation_artifact_id,
                field="automated_evaluation_artifact_id",
                prefixes=("artifact",),
            ),
            "long_form": _validated_id(
                long_form_evidence_artifact_id,
                field="long_form_evidence_artifact_id",
                prefixes=("artifact",),
            ),
            "expression": (_validated_id(
                expression_evidence_artifact_id,
                field="expression_evidence_artifact_id",
                prefixes=("artifact",),
            ) if expression_evidence_artifact_id is not None else None),
            "security": _validated_id(
                security_evidence_artifact_id,
                field="security_evidence_artifact_id",
                prefixes=("artifact",),
            ),
        }
        if expression_evidence_artifact_id is None:
            source_ids.pop("expression")
        if content_evidence_artifact_id is not None:
            source_ids["content"] = _validated_id(
                content_evidence_artifact_id, field="content_evidence_artifact_id",
                prefixes=("artifact",),
            )
        if len(set(source_ids.values())) != len(source_ids):
            raise WorkstationError("Qualification sources must be distinct artifacts")
        source_artifacts = {
            key: self.get_artifact(artifact_id) for key, artifact_id in source_ids.items()
        }
        source_payloads: dict[str, dict[str, Any]] = {}
        for key, artifact in source_artifacts.items():
            payload, _digest = self._verified_json_evaluation_artifact(artifact)
            source_payloads[key] = payload
        automated = source_payloads["automated"]
        if automated.get("schema") != AUTOMATED_EVALUATION_SCHEMA:
            raise WorkstationError("Automated qualification source has the wrong schema")
        try:
            automated_evaluation_gates(automated)
            long_form = parse_blind_ab_evidence(source_payloads["long_form"])
            expression = (parse_blind_ab_evidence(source_payloads["expression"])
                          if "expression" in source_payloads else None)
            security = parse_security_evidence(source_payloads["security"])
        except QualificationReportError as error:
            raise WorkstationError(f"Qualification source is invalid: {error}") from error
        expected_subject = {"kind": subject_kind, "id": subject_id}
        for label, evidence in (
            ("long-form", long_form),
            ("expression", expression),
            ("security", security),
        ):
            if evidence is not None and evidence["subject"] != expected_subject:
                raise WorkstationError(f"{label} qualification source belongs to another subject")
        if subject_kind == "artifact":
            if subject_id not in source_artifacts["automated"]["parent_artifact_ids"]:
                raise WorkstationError(
                    "Automated evaluation must directly reference its qualified artifact"
                )
        elif source_artifacts["automated"]["metadata"].get("expression_id") != subject_id:
            raise WorkstationError(
                "Automated evaluation metadata does not reference this expression"
            )
        schemas = {
            "automated": AUTOMATED_EVALUATION_SCHEMA,
            "long_form": BLIND_AB_EVIDENCE_SCHEMA,
            "expression": BLIND_AB_EVIDENCE_SCHEMA,
            "security": SECURITY_EVIDENCE_SCHEMA,
        }
        if "expression" not in source_payloads:
            schemas.pop("expression")
        if "content" in source_payloads:
            self._verify_content_review(
                source_payloads["content"], source_artifacts["automated"], expected_subject,
            )
            schemas["content"] = CONTENT_REVIEW_SCHEMA
        sources = {
            key: self._qualification_source_reference(
                source_artifacts[key], schema=schemas[key]
            )
            for key in schemas
        }
        try:
            report = compose_qualification_report(
                subject=expected_subject,
                automated_evaluation=automated,
                content_evidence=source_payloads.get("content"),
                long_form_evidence=source_payloads["long_form"],
                expression_evidence=source_payloads.get("expression"),
                security_evidence=source_payloads["security"],
                sources=sources,
            )
        except QualificationReportError as error:
            raise WorkstationError(f"Qualification composition failed: {error}") from error
        artifact_id = _new_id("artifact")
        relative = PurePosixPath("evaluations") / f"{artifact_id}-qualification.json"
        destination = self.artifact_root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        try:
            with destination.open("xb") as output:
                output.write(encoded)
            parent_ids = list(source_ids.values())
            if subject_kind == "artifact":
                parent_ids.insert(0, subject_id)
            evaluation = self.register_artifact(
                artifact_type="evaluation",
                name=f"Qualification composition for {subject.get('name') or subject_id}",
                status="ready",
                project_id=source_artifacts["automated"].get("project_id"),
                local_path=destination,
                metadata={
                    "qualification_evidence_kind": "composed-qualification",
                    "subject_kind": subject_kind,
                    "subject_id": subject_id,
                    **({"expression_id": subject_id} if subject_kind == "expression" else {}),
                },
                parent_artifact_ids=parent_ids,
                artifact_id=artifact_id,
            )
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        qualification = self.record_qualification(
            evaluation_artifact_id=evaluation["id"],
            subject_kind=subject_kind,
            subject_id=subject_id,
        )
        return {"evaluation_artifact": evaluation, "qualification": qualification}

    def record_qualification(
        self,
        *,
        evaluation_artifact_id: str,
        subject_kind: str,
        subject_id: str,
    ) -> dict[str, Any]:
        evaluation_artifact_id = _validated_id(
            evaluation_artifact_id,
            field="evaluation_artifact_id",
            prefixes=("artifact",),
        )
        if subject_kind == "artifact":
            subject_id = _validated_id(
                subject_id, field="subject_id", prefixes=("artifact",)
            )
            subject = self.get_artifact(subject_id)
        elif subject_kind == "expression":
            subject_id = _validated_id(
                subject_id, field="subject_id", prefixes=("expression",)
            )
            subject = self.get_expression_draft(subject_id)
        else:
            raise WorkstationError("Qualification subject kind is unsupported")
        evaluation = self.get_artifact(evaluation_artifact_id)
        report = self._verified_evaluation_report(evaluation)
        if report["subject"] != {"kind": subject_kind, "id": subject_id}:
            raise WorkstationError("Qualification report subject does not match the request")
        if subject_kind == "artifact":
            if subject_id not in evaluation["parent_artifact_ids"]:
                raise WorkstationError(
                    "Evaluation artifact must directly reference its qualified artifact"
                )
            if subject.get("status") != "ready":
                raise WorkstationError("Qualification subject artifact is not ready")
        else:
            if evaluation["metadata"].get("expression_id") != subject_id:
                raise WorkstationError(
                    "Evaluation artifact metadata does not reference this expression"
                )

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT * FROM qualification_runs WHERE subject_kind = ? "
                    "AND subject_id = ? AND evaluation_artifact_id = ? "
                    "AND report_sha256 = ?",
                    (
                        subject_kind,
                        subject_id,
                        evaluation_artifact_id,
                        evaluation["sha256"],
                    ),
                ).fetchone()
                if existing is not None:
                    result = self._qualification_record(connection, existing)
                    connection.commit()
                    return result
                qualification_id = _new_id("qualification")
                now = utc_now()
                connection.execute(
                    """INSERT INTO qualification_runs(
                        id, subject_kind, subject_id, evaluation_artifact_id,
                        report_sha256, report_schema, overall_status,
                        run_metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        qualification_id,
                        subject_kind,
                        subject_id,
                        evaluation_artifact_id,
                        evaluation["sha256"],
                        QUALIFICATION_REPORT_SCHEMA,
                        report["overall_status"],
                        _json_text(report["run_metadata"], field="run_metadata"),
                        now,
                    ),
                )
                for gate in report["gates"]:
                    connection.execute(
                        """INSERT INTO qualification_gate_results(
                            qualification_id, gate_id, status, summary,
                            metrics_json, evidence_json
                        ) VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            qualification_id,
                            gate["id"],
                            gate["status"],
                            gate["summary"],
                            _json_text(gate["metrics"], field="gate metrics"),
                            _json_text(gate["evidence"], field="gate evidence"),
                        ),
                    )
                row = connection.execute(
                    "SELECT * FROM qualification_runs WHERE id = ?",
                    (qualification_id,),
                ).fetchone()
                result = self._qualification_record(connection, row)
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise

    def get_qualification(self, qualification_id: str) -> dict[str, Any]:
        qualification_id = _validated_id(
            qualification_id,
            field="qualification_id",
            prefixes=("qualification",),
        )
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM qualification_runs WHERE id = ?", (qualification_id,)
            ).fetchone()
            if row is None:
                raise WorkstationError("Qualification run was not found")
            return self._qualification_record(connection, row)

    def list_qualifications(
        self,
        *,
        subject_kind: str | None = None,
        subject_id: str | None = None,
        evaluation_artifact_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[str] = []
        if subject_kind is not None:
            if subject_kind not in {"artifact", "expression"}:
                raise WorkstationError("Qualification subject kind is unsupported")
            clauses.append("subject_kind = ?")
            parameters.append(subject_kind)
        if subject_id is not None:
            prefixes = ("artifact", "expression") if subject_kind is None else (subject_kind,)
            subject_id = _validated_id(subject_id, field="subject_id", prefixes=prefixes)
            clauses.append("subject_id = ?")
            parameters.append(subject_id)
        if evaluation_artifact_id is not None:
            evaluation_artifact_id = _validated_id(
                evaluation_artifact_id,
                field="evaluation_artifact_id",
                prefixes=("artifact",),
            )
            clauses.append("evaluation_artifact_id = ?")
            parameters.append(evaluation_artifact_id)
        query = "SELECT * FROM qualification_runs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
            return [self._qualification_record(connection, row) for row in rows]

    def _assert_qualification_passes(
        self,
        connection: sqlite3.Connection,
        *,
        qualification_id: str,
        subject_kind: str,
        subject_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM qualification_runs WHERE id = ?", (qualification_id,)
        ).fetchone()
        if row is None:
            raise WorkstationError("Qualification run was not found")
        if row["subject_kind"] != subject_kind or row["subject_id"] != subject_id:
            raise WorkstationError("Qualification run belongs to another subject")
        if row["overall_status"] != "passed":
            raise WorkstationError("Qualification run did not pass all production gates")
        gate_rows = connection.execute(
            "SELECT gate_id, status FROM qualification_gate_results "
            "WHERE qualification_id = ?",
            (qualification_id,),
        ).fetchall()
        gate_status = {gate["gate_id"]: gate["status"] for gate in gate_rows}
        if any(gate_status.get(gate_id) != "passed" for gate_id in REQUIRED_QUALIFICATION_GATES):
            raise WorkstationError("Qualification run is missing a passed production gate")
        evaluation_row = connection.execute(
            "SELECT * FROM artifacts WHERE id = ?", (row["evaluation_artifact_id"],)
        ).fetchone()
        if evaluation_row is None:
            raise WorkstationError("Qualification evaluation artifact was not found")
        evaluation = self._artifact_record(connection, evaluation_row)
        if evaluation["sha256"] != row["report_sha256"]:
            raise WorkstationError("Qualification report fingerprint does not match")
        report = self._verified_evaluation_report(evaluation)
        if report["subject"] != {"kind": subject_kind, "id": subject_id}:
            raise WorkstationError("Qualification report subject changed after import")
        return row

    def promote_artifact(
        self, artifact_id: str, *, qualification_id: str, _before_commit: Any = None
    ) -> dict[str, Any]:
        artifact_id = _validated_id(
            artifact_id, field="artifact_id", prefixes=("artifact",)
        )
        qualification_id = _validated_id(
            qualification_id, field="qualification_id", prefixes=("qualification",)
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
                ).fetchone()
                if row is None:
                    raise WorkstationError("Artifact was not found")
                artifact = self._artifact_record(connection, row)
                if artifact["type"] not in PROMOTABLE_ARTIFACT_TYPES:
                    raise WorkstationError("This artifact type cannot be promoted")
                if artifact["status"] != "ready":
                    raise WorkstationError("Only ready artifacts can be promoted")
                self._assert_qualification_passes(
                    connection,
                    qualification_id=qualification_id,
                    subject_kind="artifact",
                    subject_id=artifact_id,
                )
                existing = connection.execute(
                    "SELECT qualification_id FROM artifact_promotions WHERE artifact_id = ?",
                    (artifact_id,),
                ).fetchone()
                if existing is not None:
                    if existing["qualification_id"] != qualification_id:
                        raise WorkstationError("Artifact was already promoted with other evidence")
                else:
                    connection.execute(
                        "INSERT INTO artifact_promotions(artifact_id, qualification_id, promoted_at) "
                        "VALUES (?, ?, ?)",
                        (artifact_id, qualification_id, utc_now()),
                    )
                refreshed = connection.execute(
                    "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
                ).fetchone()
                result = self._artifact_record(connection, refreshed)
                if _before_commit is not None:
                    _before_commit()
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise

    def promote_expression(
        self, expression_id: str, *, qualification_id: str
    ) -> dict[str, Any]:
        expression_id = _validated_id(
            expression_id, field="expression_id", prefixes=("expression",)
        )
        qualification_id = _validated_id(
            qualification_id, field="qualification_id", prefixes=("qualification",)
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                expression_row = connection.execute(
                    "SELECT * FROM expression_drafts WHERE id = ?", (expression_id,)
                ).fetchone()
                if expression_row is None:
                    raise WorkstationError("Expression draft was not found")
                if expression_row["qualification_status"] not in {"draft", "pending", "qualified"}:
                    raise WorkstationError("Expression is not eligible for promotion")
                self._assert_qualification_passes(
                    connection,
                    qualification_id=qualification_id,
                    subject_kind="expression",
                    subject_id=expression_id,
                )
                existing = connection.execute(
                    "SELECT qualification_id FROM expression_qualifications "
                    "WHERE expression_id = ?",
                    (expression_id,),
                ).fetchone()
                if existing is not None:
                    if existing["qualification_id"] != qualification_id:
                        raise WorkstationError(
                            "Expression was already promoted with other evidence"
                        )
                else:
                    now = utc_now()
                    connection.execute(
                        "INSERT INTO expression_qualifications(expression_id, qualification_id, promoted_at) "
                        "VALUES (?, ?, ?)",
                        (expression_id, qualification_id, now),
                    )
                    connection.execute(
                        "UPDATE expression_drafts SET qualification_status = 'qualified', "
                        "updated_at = ? WHERE id = ?",
                        (now, expression_id),
                    )
                refreshed = connection.execute(
                    "SELECT * FROM expression_drafts WHERE id = ?", (expression_id,)
                ).fetchone()
                result = self._expression_record(connection, refreshed)
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise

    def artifact_details(self, artifact_id: str) -> dict[str, Any]:
        artifact_id = _validated_id(
            artifact_id, field="artifact_id", prefixes=("artifact",)
        )
        with self._connect() as connection:
            root_row = connection.execute(
                "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
            if root_row is None:
                raise WorkstationError("Artifact was not found")
            relations: dict[str, tuple[str, int]] = {artifact_id: ("self", 0)}
            frontier = [(artifact_id, 0)]
            while frontier:
                current, depth = frontier.pop(0)
                parents = connection.execute(
                    "SELECT parent_artifact_id FROM artifact_parents WHERE artifact_id = ?",
                    (current,),
                ).fetchall()
                for parent in parents:
                    parent_id = parent["parent_artifact_id"]
                    if parent_id not in relations:
                        relations[parent_id] = ("ancestor", depth + 1)
                        frontier.append((parent_id, depth + 1))
            frontier = [(artifact_id, 0)]
            while frontier:
                current, depth = frontier.pop(0)
                children = connection.execute(
                    "SELECT artifact_id FROM artifact_parents WHERE parent_artifact_id = ?",
                    (current,),
                ).fetchall()
                for child in children:
                    child_id = child["artifact_id"]
                    if child_id not in relations:
                        relations[child_id] = ("descendant", depth + 1)
                        frontier.append((child_id, depth + 1))
            nodes: list[dict[str, Any]] = []
            for node_id, (relation, depth) in relations.items():
                row = connection.execute(
                    "SELECT * FROM artifacts WHERE id = ?", (node_id,)
                ).fetchone()
                if row is None:
                    continue
                record = self._artifact_record(connection, row)
                record["relation"] = relation
                record["depth"] = depth
                nodes.append(record)
            placeholders = ",".join("?" for _ in relations)
            edge_rows = connection.execute(
                f"SELECT artifact_id, parent_artifact_id, position FROM artifact_parents "
                f"WHERE artifact_id IN ({placeholders}) AND parent_artifact_id IN ({placeholders})",
                (*relations, *relations),
            ).fetchall()
            qualification_rows = connection.execute(
                "SELECT * FROM qualification_runs WHERE subject_kind = 'artifact' "
                "AND subject_id = ? ORDER BY created_at DESC, id",
                (artifact_id,),
            ).fetchall()
            order = {name: position for position, name in enumerate(
                ("dataset", "checkpoint", "expression-bank", "engine", "package", "evaluation")
            )}
            nodes.sort(key=lambda item: (order.get(item["type"], 99), item["created_at"], item["id"]))
            return {
                "artifact": self._artifact_record(connection, root_row),
                "lineage": {
                    "nodes": nodes,
                    "edges": [dict(edge) for edge in edge_rows],
                },
                "qualifications": [
                    self._qualification_record(connection, row)
                    for row in qualification_rows
                ],
            }

    def create_job(
        self, *, job_type: str, project_id: str | None,
        parameters: Mapping[str, Any] | None = None, resource_class: str | None = None,
        depends_on: Sequence[str] | None = None, priority: int = 0,
        retry_of: str | None = None, attempt: int = 1,
        start_paused: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(start_paused, bool):
            raise WorkstationError("start_paused must be a boolean")
        if job_type not in JOB_TYPES:
            raise WorkstationError(f"Unsupported job type: {job_type}")
        trusted_resource = JOB_RESOURCE_CLASSES[job_type]
        if resource_class is not None and resource_class != trusted_resource:
            raise WorkstationError(f"{job_type} requires resource class {trusted_resource}")
        if depends_on is not None and (isinstance(depends_on, (str, bytes)) or not isinstance(depends_on, Sequence)):
            raise WorkstationError("depends_on must be a list of job IDs")
        dependency_ids = [
            _validated_id(value, field="depends_on", prefixes=("job",)) for value in (depends_on or ())
        ]
        if len(dependency_ids) != len(set(dependency_ids)):
            raise WorkstationError("depends_on contains duplicates")
        priority = _job_priority(priority)
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise WorkstationError("attempt must be a positive integer")
        if retry_of is not None:
            retry_of = _validated_id(retry_of, field="retry_of", prefixes=("job",))
        elif attempt != 1:
            raise WorkstationError("A first job attempt must be 1")
        if project_id is not None:
            project_id = _validated_id(project_id, field="project_id", prefixes=tuple(PROJECT_KINDS))
        record = {
            "id": _new_id("job"), "type": job_type, "project_id": project_id,
            "status": "paused" if start_paused else "queued",
            "progress": 0.0, "parameters": _mapping_value(parameters, field="parameters"),
             "result": {}, "error": None, "wait_reason": None,
             "resource_class": trusted_resource,
            "depends_on": dependency_ids, "created_at": utc_now(), "updated_at": utc_now(),
            "started_at": None, "finished_at": None, "priority": priority,
            "attempt": attempt, "retry_of": retry_of,
        }
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if project_id is not None and connection.execute(
                    "SELECT 1 FROM projects WHERE id = ?", (project_id,)
                ).fetchone() is None:
                    raise WorkstationError("Project was not found")
                if project_id is not None:
                    project_kind = connection.execute(
                        "SELECT kind FROM projects WHERE id = ?", (project_id,)
                    ).fetchone()["kind"]
                    if project_kind not in JOB_PROJECT_KINDS[job_type]:
                        allowed = ", ".join(sorted(JOB_PROJECT_KINDS[job_type]))
                        raise WorkstationError(
                            f"{job_type} requires a project of kind: {allowed}"
                        )
                for dependency_id in dependency_ids:
                    if connection.execute("SELECT 1 FROM jobs WHERE id = ?", (dependency_id,)).fetchone() is None:
                        raise WorkstationError("Job dependency was not found")
                if retry_of is not None:
                    retry_source = connection.execute(
                        "SELECT type, project_id, status, attempt FROM jobs WHERE id = ?",
                        (retry_of,),
                    ).fetchone()
                    if retry_source is None:
                        raise WorkstationError("Retry source job was not found")
                    if (
                        retry_source["type"] != job_type
                        or retry_source["project_id"] != project_id
                    ):
                        raise WorkstationError(
                            "Retry source must use the same job type and project"
                        )
                    if retry_source["status"] not in {"failed", "cancelled"}:
                        raise WorkstationError(
                            "Retry source job must be failed or cancelled"
                        )
                    if int(retry_source["attempt"]) + 1 != attempt:
                        raise WorkstationError(
                            "Retry attempt must follow its source attempt"
                        )
                    if connection.execute(
                        "SELECT 1 FROM jobs WHERE retry_of = ?", (retry_of,)
                    ).fetchone() is not None:
                        raise WorkstationError("Retry source job was already retried")
                self._insert_job(connection, record)
                self._append_job_log(
                    connection, record["id"], "info",
                    "Job created paused" if start_paused else "Job queued",
                )
                self._refresh_project_summary(connection, project_id)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get_job(str(record["id"]))

    def get_job(self, job_id: str) -> dict[str, Any]:
        job_id = _validated_id(job_id, field="job_id", prefixes=("job",))
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise WorkstationError("Job was not found")
        return self._job_record(row)

    def list_jobs(self) -> list[dict[str, Any]]:
        self.recover_expired_jobs()
        self.propagate_dependency_failures()
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM jobs ORDER BY updated_at DESC").fetchall()
        return [self._job_record(row) for row in rows]

    def set_job_wait_reason(self, job_id: str, reason: str | None) -> dict[str, Any]:
        """Publish a stable queue reason without changing job ownership or state."""
        job_id = _validated_id(job_id, field="job_id", prefixes=("job",))
        next_reason = (
            _required_text(reason, field="wait_reason", maximum=240)
            if reason is not None
            else None
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
                if row is None:
                    raise WorkstationError("Job was not found")
                if row["status"] != "queued" or row["wait_reason"] == next_reason:
                    connection.commit()
                    return self._job_record(row)
                now = utc_now()
                connection.execute(
                    "UPDATE jobs SET wait_reason = ?, updated_at = ? "
                    "WHERE id = ? AND status = 'queued'",
                    (next_reason, now, job_id),
                )
                if next_reason is not None:
                    self._append_job_log(connection, job_id, "info", next_reason)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get_job(job_id)

    def append_job_log(self, job_id: str, level: str, message: str, *, claim_token: str | None = None) -> None:
        job_id = _validated_id(job_id, field="job_id", prefixes=("job",))
        level = _required_text(level, field="level", maximum=16).lower()
        if level not in {"debug", "info", "warning", "error"}:
            raise WorkstationError("Unsupported log level")
        message = _required_text(message, field="message", maximum=2000)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute("SELECT status, claim_token FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if row is None:
                    raise WorkstationError("Job was not found")
                if row["status"] == "running":
                    if claim_token is None:
                        raise WorkstationError("A claim token is required to log a running job")
                    claim_token = _validated_id(
                        claim_token, field="claim_token", prefixes=("claim",)
                    )
                    self._owned_running_row(connection, job_id, claim_token)
                self._append_job_log(connection, job_id, level, message)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def list_job_logs(self, job_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        job_id = _validated_id(job_id, field="job_id", prefixes=("job",))
        try:
            limit_value = int(limit)
        except (TypeError, ValueError) as error:
            raise WorkstationError("limit must be an integer") from error
        limit_value = max(1, min(1000, limit_value))
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is None:
                raise WorkstationError("Job was not found")
            rows = connection.execute(
                "SELECT id, level, message, created_at FROM job_events WHERE job_id = ? ORDER BY id DESC LIMIT ?",
                (job_id, limit_value),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def _lease_seconds(self) -> int:
        return _positive_integer_env("ANIFLIVE_TTS_WORKSTATION_LEASE_SECONDS", 120, maximum=86400)

    @staticmethod
    def _delete_expired_resource_leases(connection: sqlite3.Connection, now: str) -> None:
        connection.execute(
            "DELETE FROM resource_leases WHERE lease_expires_at <= ? "
            "AND NOT (resource_key = ? AND purpose LIKE 'job:%')",
            (now, GPU_RESOURCE_KEY),
        )

    @staticmethod
    def _runtime_handoff_record(connection: sqlite3.Connection) -> dict[str, Any] | None:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'runtime_handoff'"
        ).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        if not isinstance(value, dict):
            raise WorkstationError("Runtime handoff metadata is malformed")
        return value

    def get_runtime_handoff(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            return self._runtime_handoff_record(connection)

    def set_runtime_handoff(self, token: str, value: Mapping[str, Any]) -> dict[str, Any]:
        phases = {"idle", "draining", "stopping", "ready-for-job", "job-running", "restoring", "failed"}
        if not isinstance(value, Mapping) or value.get("phase") not in phases:
            raise WorkstationError("Runtime handoff phase is invalid")
        record = dict(value)
        if record.get("job_id") is not None:
            _validated_id(record["job_id"], field="job_id", prefixes=("job",))
        record.update(owner_id=self.worker_id, updated_at=utc_now())
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if len(encoded.encode("utf-8")) > 65536:
            raise WorkstationError("Runtime handoff metadata exceeds 64 KiB")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            lease = connection.execute(
                "SELECT 1 FROM resource_leases WHERE resource_key = 'runtime:handoff' "
                "AND token = ? AND owner_id = ? AND lease_expires_at > ?",
                (token, self.worker_id, utc_now()),
            ).fetchone()
            if lease is None:
                raise WorkstationError("Runtime handoff coordinator lease is missing or expired")
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('runtime_handoff', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (encoded,),
            )
            connection.commit()
        return record

    def acquire_resource_lease(
        self, resource_key: str, *, purpose: str, owner_id: str | None = None,
        lease_seconds: int | None = None,
    ) -> str:
        resource_key = _required_text(resource_key, field="resource_key", maximum=80)
        purpose = _required_text(purpose, field="purpose", maximum=200)
        owner_id = _required_text(owner_id or self.worker_id, field="owner_id", maximum=80)
        seconds = lease_seconds if lease_seconds is not None else self._lease_seconds()
        if not isinstance(seconds, int) or seconds < 1 or seconds > 86400:
            raise WorkstationError("lease_seconds must be between 1 and 86400")
        token = _new_id("lease")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                now = utc_now()
                expires = _timestamp(_utc_datetime() + timedelta(seconds=seconds))
                self._delete_expired_resource_leases(connection, now)
                handoff = self._runtime_handoff_record(connection)
                if resource_key == GPU_RESOURCE_KEY and handoff and handoff.get("phase") != "idle":
                    if handoff.get("phase") != "restoring" or not purpose.startswith("inference:"):
                        raise WorkstationError("Resource gpu:0 is busy: runtime handoff")
                if resource_key == GPU_RESOURCE_KEY and connection.execute(
                    "SELECT 1 FROM jobs WHERE status = 'running' "
                    "AND resource_class = 'gpu-exclusive' LIMIT 1"
                ).fetchone() is not None:
                    raise WorkstationError(f"Resource {resource_key} is busy")
                if connection.execute(
                    "SELECT 1 FROM resource_leases WHERE resource_key = ?", (resource_key,)
                ).fetchone() is not None:
                    raise WorkstationError(f"Resource {resource_key} is busy")
                connection.execute(
                    "INSERT INTO resource_leases(resource_key, owner_id, token, purpose, lease_expires_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (resource_key, owner_id, token, purpose, expires, now),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return token

    def heartbeat_resource_lease(self, token: str, *, lease_seconds: int | None = None) -> str:
        token = _validated_id(token, field="lease_token", prefixes=("lease",))
        seconds = lease_seconds if lease_seconds is not None else self._lease_seconds()
        if not isinstance(seconds, int) or seconds < 1 or seconds > 86400:
            raise WorkstationError("lease_seconds must be between 1 and 86400")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                now = utc_now()
                expires = _timestamp(_utc_datetime() + timedelta(seconds=seconds))
                changed = connection.execute(
                    "UPDATE resource_leases SET lease_expires_at = ?, updated_at = ? WHERE token = ? AND lease_expires_at > ?",
                    (expires, now, token, now),
                ).rowcount
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        if changed != 1:
            raise WorkstationError("The resource lease is missing or expired")
        return expires

    def release_resource_lease(self, token: str) -> None:
        token = _validated_id(token, field="lease_token", prefixes=("lease",))
        with self._connect() as connection:
            connection.execute("DELETE FROM resource_leases WHERE token = ?", (token,))

    def claim_job(self, job_id: str, *, allow_paused: bool = False) -> JobClaim:
        if not isinstance(allow_paused, bool):
            raise WorkstationError("allow_paused must be a boolean")
        job_id = _validated_id(job_id, field="job_id", prefixes=("job",))
        seconds = self._lease_seconds()
        token = _new_id("claim")
        dependency_failure: str | None = None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                now = utc_now()
                expires = _timestamp(_utc_datetime() + timedelta(seconds=seconds))
                row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if row is None:
                    raise WorkstationError("Job was not found")
                record = self._job_record(row)
                claimable = {"queued", "paused"} if allow_paused else {"queued"}
                if record["status"] not in claimable:
                    raise WorkstationError("Only a queued job can be claimed")
                for dependency_id in record["depends_on"]:
                    dependency = connection.execute("SELECT status FROM jobs WHERE id = ?", (dependency_id,)).fetchone()
                    if dependency is None or dependency["status"] in {"failed", "cancelled"}:
                        dependency_failure = f"Dependency {dependency_id} did not succeed"
                        break
                    if dependency["status"] != "succeeded":
                        raise WorkstationError("Job dependencies are not complete")
                if dependency_failure is not None:
                    connection.execute(
                        "UPDATE jobs SET status = 'failed', error = ?, wait_reason = NULL, "
                        "updated_at = ?, finished_at = ? WHERE id = ? AND status = ?",
                        (dependency_failure, now, now, job_id, record["status"]),
                    )
                    self._append_job_log(connection, job_id, "error", dependency_failure)
                    self._refresh_project_summary(connection, record["project_id"])
                    connection.commit()
                else:
                    if record["resource_class"] == "gpu-exclusive":
                        handoff = self._runtime_handoff_record(connection)
                        if handoff and handoff.get("phase") != "idle":
                            if not (
                                handoff.get("phase") == "ready-for-job"
                                and handoff.get("job_id") == job_id
                                and handoff.get("owner_id") == self.worker_id
                            ):
                                raise WorkstationError("The exclusive GPU resource is busy: runtime handoff")
                        self._delete_expired_resource_leases(connection, now)
                        if connection.execute(
                            "SELECT 1 FROM jobs WHERE status = 'running' "
                            "AND resource_class = 'gpu-exclusive' LIMIT 1"
                        ).fetchone() is not None:
                            raise WorkstationError("The exclusive GPU resource is busy")
                        if connection.execute(
                            "SELECT 1 FROM resource_leases WHERE resource_key = ?", (GPU_RESOURCE_KEY,)
                        ).fetchone() is not None:
                            raise WorkstationError("The exclusive GPU resource is busy")
                        connection.execute(
                            "INSERT INTO resource_leases(resource_key, owner_id, token, purpose, lease_expires_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                            (GPU_RESOURCE_KEY, self.worker_id, token, f"job:{job_id}", expires, now),
                        )
                        if handoff and handoff.get("phase") == "ready-for-job":
                            handoff.update(phase="job-running", updated_at=now)
                            connection.execute(
                                "UPDATE metadata SET value = ? WHERE key = 'runtime_handoff'",
                                (json.dumps(handoff, ensure_ascii=False, sort_keys=True),),
                            )
                    changed = connection.execute(
                        """UPDATE jobs SET status = 'running', progress = 0.01,
                            started_at = ?, updated_at = ?, worker_id = ?, claim_token = ?,
                            lease_expires_at = ?, cancel_requested = 0, pause_requested = 0,
                            wait_reason = NULL
                            WHERE id = ? AND status = ?""",
                        (now, now, self.worker_id, token, expires, job_id, record["status"]),
                    ).rowcount
                    if changed != 1:
                        raise WorkstationError("Job was claimed by another worker")
                    self._append_job_log(connection, job_id, "info", "Job started")
                    self._refresh_project_summary(connection, record["project_id"])
                    connection.commit()
            except Exception:
                connection.rollback()
                raise
        if dependency_failure is not None:
            raise WorkstationError(dependency_failure)
        return JobClaim(self.get_job(job_id), token, self.worker_id, expires)

    def _owned_running_row(self, connection: sqlite3.Connection, job_id: str, claim_token: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise WorkstationError("Job was not found")
        if row["status"] != "running":
            raise WorkstationError("Only a running job can be updated by a worker")
        if row["claim_token"] != claim_token or row["worker_id"] != self.worker_id:
            raise WorkstationError("The job claim token is invalid")
        if not row["lease_expires_at"] or row["lease_expires_at"] <= utc_now():
            raise WorkstationError("The job lease has expired")
        return row

    def heartbeat_job(self, job_id: str, claim_token: str) -> str:
        job_id = _validated_id(job_id, field="job_id", prefixes=("job",))
        claim_token = _validated_id(claim_token, field="claim_token", prefixes=("claim",))
        seconds = self._lease_seconds()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                now = utc_now()
                expires = _timestamp(_utc_datetime() + timedelta(seconds=seconds))
                row = self._owned_running_row(connection, job_id, claim_token)
                connection.execute(
                    "UPDATE jobs SET lease_expires_at = ?, updated_at = ? WHERE id = ?",
                    (expires, now, job_id),
                )
                if row["resource_class"] == "gpu-exclusive":
                    changed = connection.execute(
                        "UPDATE resource_leases SET lease_expires_at = ?, updated_at = ? WHERE token = ?",
                        (expires, now, claim_token),
                    ).rowcount
                    if changed != 1:
                        raise WorkstationError("The GPU resource lease is missing")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return expires

    def job_cancel_requested(self, job_id: str, claim_token: str) -> bool:
        job_id = _validated_id(job_id, field="job_id", prefixes=("job",))
        claim_token = _validated_id(claim_token, field="claim_token", prefixes=("claim",))
        with self._connect() as connection:
            row = self._owned_running_row(connection, job_id, claim_token)
            return bool(row["cancel_requested"])

    def job_pause_requested(self, job_id: str, claim_token: str) -> bool:
        job_id = _validated_id(job_id, field="job_id", prefixes=("job",))
        claim_token = _validated_id(claim_token, field="claim_token", prefixes=("claim",))
        with self._connect() as connection:
            row = self._owned_running_row(connection, job_id, claim_token)
            return bool(row["pause_requested"])

    def expire_job_for_docker_cleanup(
        self, job_id: str, claim_token: str, *, error: str
    ) -> None:
        """Fence a job for reconciliation without releasing its GPU lease."""

        job_id = _validated_id(job_id, field="job_id", prefixes=("job",))
        claim_token = _validated_id(
            claim_token, field="claim_token", prefixes=("claim",)
        )
        message = _required_text(error, field="error", maximum=1000)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._owned_running_row(connection, job_id, claim_token)
                if row["resource_class"] != "gpu-exclusive":
                    raise WorkstationError(
                        "Only a GPU job can be fenced for Docker cleanup"
                    )
                expired = "1970-01-01T00:00:00.000Z"
                connection.execute(
                    "UPDATE jobs SET lease_expires_at = ?, error = ?, updated_at = ? "
                    "WHERE id = ?",
                    (expired, message, utc_now(), job_id),
                )
                self._append_job_log(
                    connection,
                    job_id,
                    "error",
                    "Docker cleanup is uncertain; GPU lease quarantined",
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def expired_gpu_job_ids(self) -> tuple[str, ...]:
        now = utc_now()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM jobs WHERE status = 'running' "
                "AND resource_class = 'gpu-exclusive' "
                "AND (lease_expires_at IS NULL OR lease_expires_at <= ?) ORDER BY id",
                (now,),
            ).fetchall()
        return tuple(str(row["id"]) for row in rows)

    def update_job(
        self, job_id: str, *, status: str | None = None, progress: float | None = None,
        result: Mapping[str, Any] | None = None, error: str | None = None,
        claim_token: str | None = None,
    ) -> dict[str, Any]:
        job_id = _validated_id(job_id, field="job_id", prefixes=("job",))
        if status == "running":
            raise WorkstationError("Only claim_job can transition a job to running")
        next_progress = _progress(progress) if progress is not None else None
        next_result = _mapping_value(result, field="result") if result is not None else None
        next_error = _required_text(error, field="error", maximum=1000) if error is not None else None
        staged_cleanup_paths: list[str] = []
        verified_rows = None
        verified_identities = {}
        if status in {"succeeded", "failed"} and claim_token is not None:
            claim_token = _validated_id(claim_token, field="claim_token", prefixes=("claim",))
            needs_verification = False
            with self._connect() as verification_connection:
                verification_job = verification_connection.execute(
                    "SELECT * FROM jobs WHERE id = ?", (job_id,),
                ).fetchone()
                if verification_job is not None and verification_job["status"] == "running":
                    self._owned_running_row(verification_connection, job_id, claim_token)
                    verification_result = (next_result if next_result is not None
                                           else _decode_json(verification_job["result_json"], fallback={}))
                    needs_verification = status == "succeeded" or "diagnostic_artifact_ids" in verification_result
                    verification_rows = self._staged_worker_artifacts(verification_connection, job_id) if needs_verification else []
            if needs_verification:
                # Check multi-gigabyte outputs without blocking job/coordinator heartbeats.
                verified_rows = [dict(row) for row in verification_rows]
                verified_identities = self._verify_staged_worker_artifacts(verification_rows)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if row is None:
                    raise WorkstationError("Job was not found")
                previous = str(row["status"])
                if previous in TERMINAL_JOB_STATES:
                    raise WorkstationError("Terminal jobs are immutable")
                next_status = status or previous
                if next_status not in JOB_STATES:
                    raise WorkstationError(f"Unsupported job status: {next_status}")
                if previous == "queued":
                    if next_status not in {"paused", "cancelled", "failed"}:
                        raise WorkstationError(
                            "Queued jobs can only be claimed, paused, failed, or cancelled"
                        )
                    if progress is not None or result is not None:
                        raise WorkstationError("A queued job cannot publish worker progress or results")
                elif previous == "paused":
                    raise WorkstationError("Paused jobs must be resumed or cancelled explicitly")
                else:
                    if claim_token is None:
                        raise WorkstationError("A claim token is required to update a running job")
                    claim_token = _validated_id(claim_token, field="claim_token", prefixes=("claim",))
                    self._owned_running_row(connection, job_id, claim_token)
                    if next_status != previous and next_status not in JOB_TRANSITIONS[previous]:
                        raise WorkstationError(f"Job cannot transition from {previous} to {next_status}")
                    if bool(row["cancel_requested"]) and next_status == "succeeded":
                        next_status = "cancelled"
                    elif (bool(row["pause_requested"]) and next_status == "succeeded"
                          and not (row["type"] == "training.prepare"
                                   and isinstance(next_result, Mapping)
                                   and isinstance(next_result.get("backend"), Mapping)
                                   and isinstance(next_result["backend"].get("payload"), Mapping)
                                   and next_result["backend"]["payload"].get("status") == "passed")):
                        next_status = "paused"
                current = self._job_record(row)
                progress_value = next_progress if next_progress is not None else float(current["progress"])
                if progress_value < float(current["progress"]):
                    raise WorkstationError("Job progress cannot move backwards")
                result_value = dict(
                    next_result if next_result is not None else current["result"]
                )
                now = utc_now()
                finished = now if next_status in TERMINAL_JOB_STATES else None
                staged = self._staged_worker_artifacts(connection, job_id)
                if next_status == "succeeded":
                    published_ids = (
                        result_value.get("registered_artifact_ids", [])
                    )
                    if isinstance(published_ids, (str, bytes)) or not isinstance(
                        published_ids, Sequence
                    ):
                        raise WorkstationError(
                            "registered_artifact_ids must be a list of artifact IDs"
                        )
                    published_ids = [
                        _validated_id(
                            value,
                            field="registered_artifact_ids",
                            prefixes=("artifact",),
                        )
                        for value in published_ids
                    ]
                    if len(published_ids) != len(set(published_ids)):
                        raise WorkstationError(
                            "registered_artifact_ids contains duplicates"
                        )
                    staged_ids = [str(artifact["id"]) for artifact in staged]
                    if set(published_ids) != set(staged_ids):
                        raise WorkstationError(
                            "Worker result does not match its staged artifacts"
                        )
                    self._accept_staged_verification(staged, verified_rows, verified_identities)
                    if staged_ids:
                        connection.executemany(
                            "UPDATE artifacts SET status = 'ready', updated_at = ? "
                            "WHERE id = ? AND status = 'building'",
                            ((now, artifact_id) for artifact_id in staged_ids),
                        )
                    progress_value = 1.0
                elif next_status in {"paused", "failed", "cancelled"}:
                    result_value.pop("registered_artifact_ids", None)
                    diagnostic_ids_value = result_value.get(
                        "diagnostic_artifact_ids", []
                    )
                    if "diagnostic_artifact_ids" in result_value:
                        if (
                            next_status != "failed"
                            or row["type"] not in _FAILED_DIAGNOSTIC_ARTIFACT_JOB_TYPES
                        ):
                            raise WorkstationError(
                                "Only failed quality-gate jobs may retain diagnostic artifacts"
                            )
                        if isinstance(diagnostic_ids_value, (str, bytes)) or not isinstance(
                            diagnostic_ids_value, Sequence
                        ):
                            raise WorkstationError(
                                "diagnostic_artifact_ids must be a list of artifact IDs"
                            )
                        diagnostic_ids = [
                            _validated_id(
                                value,
                                field="diagnostic_artifact_ids",
                                prefixes=("artifact",),
                            )
                            for value in diagnostic_ids_value
                        ]
                        if len(diagnostic_ids) != len(set(diagnostic_ids)):
                            raise WorkstationError(
                                "diagnostic_artifact_ids contains duplicates"
                            )
                        staged_ids = [str(artifact["id"]) for artifact in staged]
                        if not staged_ids or set(diagnostic_ids) != set(staged_ids):
                            raise WorkstationError(
                                "Diagnostic result does not match its staged artifacts"
                            )
                        self._accept_staged_verification(staged, verified_rows, verified_identities)
                        connection.executemany(
                            "UPDATE artifacts SET status = 'rejected', updated_at = ? "
                            "WHERE id = ? AND status = 'building'",
                            ((now, artifact_id) for artifact_id in staged_ids),
                        )
                    elif staged:
                        staged_cleanup_paths = [
                            str(artifact["local_path"]) for artifact in staged
                        ]
                        connection.executemany(
                            "DELETE FROM artifacts WHERE id = ? AND status = 'building'",
                            ((artifact["id"],) for artifact in staged),
                        )
                result_text = _json_text(result_value, field="result")
                release_claim = next_status in TERMINAL_JOB_STATES or next_status == "paused"
                connection.execute(
                    """UPDATE jobs SET status = ?, progress = ?, result_json = ?, error = ?,
                        updated_at = ?, finished_at = COALESCE(?, finished_at),
                        worker_id = CASE WHEN ? THEN NULL ELSE worker_id END,
                        claim_token = CASE WHEN ? THEN NULL ELSE claim_token END,
                        lease_expires_at = CASE WHEN ? THEN NULL ELSE lease_expires_at END,
                        cancel_requested = CASE WHEN ? THEN 0 ELSE cancel_requested END,
                        pause_requested = CASE WHEN ? THEN 0 ELSE pause_requested END,
                        wait_reason = CASE WHEN ? THEN NULL ELSE wait_reason END
                        WHERE id = ?""",
                    (next_status, progress_value, result_text, next_error if error is not None else current["error"],
                     now, finished, release_claim, release_claim, release_claim,
                     release_claim, release_claim, release_claim, job_id),
                )
                if next_status != previous:
                    self._append_job_log(connection, job_id, "error" if next_status == "failed" else "info", f"Job {next_status}")
                if release_claim and row["resource_class"] == "gpu-exclusive" and row["claim_token"]:
                    connection.execute("DELETE FROM resource_leases WHERE token = ?", (row["claim_token"],))
                self._refresh_project_summary(connection, row["project_id"])
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        if staged_cleanup_paths:
            self._remove_staged_worker_files(staged_cleanup_paths)
        return self.get_job(job_id)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        job_id = _validated_id(job_id, field="job_id", prefixes=("job",))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if row is None:
                    raise WorkstationError("Job was not found")
                if row["status"] in TERMINAL_JOB_STATES:
                    connection.commit()
                    return self._job_record(row)
                now = utc_now()
                if row["status"] in {"queued", "paused"}:
                    connection.execute(
                        "UPDATE jobs SET status = 'cancelled', updated_at = ?, finished_at = ?, "
                        "cancel_requested = 0, pause_requested = 0, wait_reason = NULL WHERE id = ?",
                        (now, now, job_id),
                    )
                    self._append_job_log(connection, job_id, "info", "Job cancelled")
                else:
                    connection.execute("UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?", (now, job_id))
                    self._append_job_log(connection, job_id, "info", "Cancellation requested")
                self._refresh_project_summary(connection, row["project_id"])
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get_job(job_id)

    def pause_job(self, job_id: str) -> dict[str, Any]:
        job_id = _validated_id(job_id, field="job_id", prefixes=("job",))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
                if row is None:
                    raise WorkstationError("Job was not found")
                status = str(row["status"])
                if status in TERMINAL_JOB_STATES:
                    raise WorkstationError("Terminal jobs cannot be paused")
                if status == "paused" or bool(row["pause_requested"]):
                    connection.commit()
                    return self._job_record(row)
                if bool(row["cancel_requested"]):
                    raise WorkstationError("Job cancellation is already requested")
                now = utc_now()
                if status == "queued":
                    connection.execute(
                        "UPDATE jobs SET status = 'paused', pause_requested = 0, "
                        "updated_at = ?, wait_reason = NULL WHERE id = ?",
                        (now, job_id),
                    )
                    self._append_job_log(connection, job_id, "info", "Job paused")
                else:
                    connection.execute(
                        "UPDATE jobs SET pause_requested = 1, updated_at = ? WHERE id = ?",
                        (now, job_id),
                    )
                    self._append_job_log(connection, job_id, "info", "Pause requested")
                self._refresh_project_summary(connection, row["project_id"])
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get_job(job_id)

    def _resume_training_checkpoint(self, source: Mapping[str, Any]) -> dict[str, Any]:
        """Queue a distinct continuation without overwriting the paused run's evidence."""
        job_id = str(source["id"])
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM jobs WHERE retry_of = ?", (job_id,),
            ).fetchone()
        if existing is not None:
            return self.get_job(existing["id"])
        from .training_snapshot import TrainingSnapshotError, read_training_snapshot
        output = self.root / "docker-worker-output" / job_id
        candidates = list(output.glob("*/recovery-snapshots/latest.json"))
        if len(candidates) != 1:
            raise WorkstationError(
                "This training run has no unambiguous checkpoint-boundary resume record; "
                "refusing to restart it as fresh training"
            )
        checkpoint_root = candidates[0].parent
        if not checkpoint_root.resolve().is_relative_to(output.resolve()):
            raise WorkstationError("Training checkpoint escaped its job output")
        try:
            directory, manifest = read_training_snapshot(checkpoint_root)
        except (OSError, ValueError, KeyError, TrainingSnapshotError) as error:
            raise WorkstationError(f"Training checkpoint cannot be resumed: {error}") from error
        metadata = manifest["metadata"]
        if metadata.get("job_id") != job_id:
            raise WorkstationError("Training checkpoint belongs to another job")
        backend = source.get("result", {}).get("backend", {})
        ack = backend.get("payload", {}).get("pause_ack", {})
        if ack and (ack.get("snapshot_id") != directory.name
                    or ack.get("run_id") != checkpoint_root.parent.name):
            raise WorkstationError("Training checkpoint does not match the pause acknowledgement")
        record = {
            "id": _new_id("job"), "type": source["type"], "project_id": source["project_id"],
            "status": "queued", "progress": 0.0,
            "parameters": {**source["parameters"], "resume_checkpoint": str(checkpoint_root)},
            "result": {}, "error": None, "wait_reason": None,
            "resource_class": source["resource_class"], "depends_on": source["depends_on"],
            "created_at": utc_now(), "updated_at": utc_now(), "started_at": None,
            "finished_at": None, "priority": source["priority"],
            "attempt": int(source["attempt"]) + 1, "retry_of": job_id,
        }
        # Hashing above deliberately occurs outside the SQLite write transaction.
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if row is None or row["status"] != "paused" or row["updated_at"] != source["updated_at"]:
                    raise WorkstationError("Paused training changed while its checkpoint was verified")
                existing = connection.execute("SELECT id FROM jobs WHERE retry_of = ?", (job_id,)).fetchone()
                if existing is not None:
                    continuation_id = existing["id"]
                else:
                    self._insert_job(connection, record)
                    continuation_id = record["id"]
                    self._append_job_log(connection, job_id, "info",
                                         f"Checkpoint continuation queued as {continuation_id}")
                    self._append_job_log(connection, continuation_id, "info",
                                         f"Resuming {metadata['stage']} epoch {metadata['epoch']} "
                                         f"from {job_id}; the saved training budget is preserved")
                    self._refresh_project_summary(connection, source["project_id"])
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get_job(continuation_id)

    def resume_job(self, job_id: str) -> dict[str, Any]:
        job_id = _validated_id(job_id, field="job_id", prefixes=("job",))
        source = self.get_job(job_id)
        if (source["type"] == "training.prepare" and source["status"] == "paused"
                and source["started_at"] is not None):
            return self._resume_training_checkpoint(source)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
                if row is None:
                    raise WorkstationError("Job was not found")
                if row["status"] != "paused":
                    raise WorkstationError("Only a paused job can be resumed")
                if row["type"] == "training.prepare" and row["started_at"] is not None:
                    raise WorkstationError(
                        "This training run has no checkpoint-boundary resume record; "
                        "refusing to restart it as fresh training"
                    )
                now = utc_now()
                connection.execute(
                    """UPDATE jobs SET status = 'queued', progress = 0.0,
                        result_json = '{}', error = NULL, updated_at = ?,
                        started_at = NULL, finished_at = NULL, worker_id = NULL,
                        claim_token = NULL, lease_expires_at = NULL,
                        cancel_requested = 0, pause_requested = 0, wait_reason = NULL
                        WHERE id = ?""",
                    (now, job_id),
                )
                self._append_job_log(connection, job_id, "info", "Job resumed and queued")
                self._refresh_project_summary(connection, row["project_id"])
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get_job(job_id)

    def retry_job(self, job_id: str) -> dict[str, Any]:
        source = self.get_job(job_id)
        if source["status"] not in {"failed", "cancelled"}:
            raise WorkstationError("Only a failed or cancelled job can be retried")
        retried = self.create_job(
            job_type=str(source["type"]),
            project_id=source["project_id"],
            parameters=source["parameters"],
            depends_on=source["depends_on"],
            priority=int(source["priority"]),
            retry_of=str(source["id"]),
            attempt=int(source["attempt"]) + 1,
        )
        self.append_job_log(
            str(source["id"]), "info", f"Retry queued as {retried['id']}"
        )
        return retried

    def propagate_dependency_failures(self) -> int:
        now, changed = utc_now(), 0
        affected_projects: set[str] = set()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                while True:
                    iteration_changes = 0
                    rows = connection.execute("SELECT * FROM jobs WHERE status = 'queued'").fetchall()
                    for row in rows:
                        for dependency_id in _decode_json(row["depends_on_json"], fallback=[]):
                            dependency = connection.execute("SELECT status FROM jobs WHERE id = ?", (dependency_id,)).fetchone()
                            if dependency is None or dependency["status"] in {"failed", "cancelled"}:
                                message = f"Dependency {dependency_id} did not succeed"
                                updated = connection.execute(
                                    "UPDATE jobs SET status = 'failed', error = ?, wait_reason = NULL, "
                                    "updated_at = ?, finished_at = ? WHERE id = ? AND status = 'queued'",
                                    (message, now, now, row["id"]),
                                ).rowcount
                                if updated:
                                    self._append_job_log(connection, row["id"], "error", message)
                                    if row["project_id"] is not None:
                                        affected_projects.add(str(row["project_id"]))
                                    changed += 1
                                    iteration_changes += 1
                                break
                    if not iteration_changes:
                        break
                for project_id in affected_projects:
                    self._refresh_project_summary(connection, project_id)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return changed

    def recover_expired_jobs(
        self, *, confirmed_gpu_job_ids: Sequence[str] = ()
    ) -> int:
        if isinstance(confirmed_gpu_job_ids, (str, bytes)) or not isinstance(
            confirmed_gpu_job_ids, Sequence
        ):
            raise WorkstationError("confirmed_gpu_job_ids must be a list of job IDs")
        confirmed_gpu = {
            _validated_id(value, field="confirmed_gpu_job_ids", prefixes=("job",))
            for value in confirmed_gpu_job_ids
        }
        recovered = 0
        affected_projects: set[str] = set()
        staged_cleanup_paths: list[str] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                now = utc_now()
                rows = connection.execute(
                    "SELECT * FROM jobs WHERE status = 'running' AND (lease_expires_at IS NULL OR lease_expires_at <= ?)",
                    (now,),
                ).fetchall()
                for row in rows:
                    if (
                        row["resource_class"] == "gpu-exclusive"
                        and str(row["id"]) not in confirmed_gpu
                    ):
                        continue
                    cancelled = bool(row["cancel_requested"])
                    paused = bool(row["pause_requested"]) and not cancelled
                    status = "cancelled" if cancelled else "paused" if paused else "failed"
                    error = (
                        None
                        if cancelled or paused
                        else "Worker lease expired before the job completed"
                    )
                    finished_at = now if status in TERMINAL_JOB_STATES else None
                    connection.execute(
                        """UPDATE jobs SET status = ?, error = ?, updated_at = ?, finished_at = ?,
                            worker_id = NULL, claim_token = NULL, lease_expires_at = NULL,
                            cancel_requested = 0, pause_requested = 0,
                            wait_reason = NULL WHERE id = ?""",
                        (status, error, now, finished_at, row["id"]),
                    )
                    if row["claim_token"]:
                        connection.execute("DELETE FROM resource_leases WHERE token = ?", (row["claim_token"],))
                    staged = self._staged_worker_artifacts(connection, str(row["id"]))
                    staged_cleanup_paths.extend(
                        str(artifact["local_path"]) for artifact in staged
                    )
                    connection.executemany(
                        "DELETE FROM artifacts WHERE id = ? AND status = 'building'",
                        ((artifact["id"],) for artifact in staged),
                    )
                    self._append_job_log(connection, row["id"], "error" if status == "failed" else "info", f"Expired worker lease recovered as {status}")
                    if row["project_id"] is not None:
                        affected_projects.add(str(row["project_id"]))
                    recovered += 1
                for project_id in affected_projects:
                    self._refresh_project_summary(connection, project_id)
                self._delete_expired_resource_leases(connection, now)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        if staged_cleanup_paths:
            self._remove_staged_worker_files(staged_cleanup_paths)
        if recovered:
            self.propagate_dependency_failures()
        return recovered

    def allowed_import_roots(self) -> tuple[Path, ...]:
        configured = os.environ.get("ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS", "")
        roots: list[Path] = []
        for value in configured.split(os.pathsep):
            if not value.strip():
                continue
            raw = Path(value).expanduser().absolute()
            if not raw.exists():
                raise WorkstationError("A configured dataset import root does not exist")
            if _is_reparse_point(raw):
                raise WorkstationError("Dataset import roots cannot be symbolic links or reparse points")
            roots.append(raw.resolve(strict=True))
        return tuple(roots)

    def validate_import_path(self, value: str, *, allowed_roots: Sequence[Path] | None = None) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise WorkstationError("Dataset source is not configured")
        raw = Path(value).expanduser().absolute()
        roots = (self.allowed_import_roots() if allowed_roots is None
                 else tuple(Path(root).resolve(strict=True) for root in allowed_roots))
        if not roots:
            raise WorkstationError("Local inventory is disabled until ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS is configured")
        if _is_reparse_point(raw):
            raise WorkstationError("Dataset source cannot be a symbolic link or reparse point")
        try:
            source = raw.resolve(strict=True)
        except FileNotFoundError as error:
            raise WorkstationError("Dataset source does not exist") from error
        if not any(source == root or source.is_relative_to(root) for root in roots):
            raise WorkstationError("Dataset source is outside the configured import roots")
        return source

    def _safe_inventory_entries(
        self, source: Path, roots: tuple[Path, ...]
    ) -> Iterator[tuple[Path, os.stat_result, bool]]:
        def identity(info: os.stat_result) -> tuple[int, int]:
            return int(info.st_dev), int(info.st_ino)

        def has_stable_identity(info: os.stat_result) -> bool:
            device, inode = identity(info)
            return bool(device or inode)

        def verify(path: Path, info: os.stat_result | None = None) -> tuple[Path, os.stat_result]:
            before = info or path.stat(follow_symlinks=False)
            if path.is_symlink() or bool(
                getattr(before, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
            ):
                raise WorkstationError("Dataset source contains a symbolic link or reparse point")
            resolved = path.resolve(strict=True)
            if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
                raise WorkstationError("Dataset entry escaped the configured import roots")
            after = resolved.stat(follow_symlinks=False)
            if has_stable_identity(before) and identity(before) != identity(after):
                raise WorkstationError("Dataset entry changed while it was being inspected")
            return resolved, after

        root, root_info = verify(source)
        root_is_directory = stat.S_ISDIR(root_info.st_mode)
        yield root, root_info, root_is_directory
        if not root_is_directory:
            return
        stack = [(root, identity(root_info))]
        while stack:
            directory, expected_identity = stack.pop()
            try:
                current = directory.stat(follow_symlinks=False)
                if identity(current) != expected_identity:
                    raise WorkstationError("Dataset directory changed while it was being inspected")
                with os.scandir(directory) as entries:
                    for entry in entries:
                        path = Path(entry.path)
                        info = entry.stat(follow_symlinks=False)
                        if entry.is_symlink() or bool(
                            getattr(info, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
                        ):
                            raise WorkstationError("Dataset source contains a symbolic link or reparse point")
                        resolved, verified = verify(path, info)
                        is_directory = stat.S_ISDIR(verified.st_mode)
                        yield resolved, verified, is_directory
                        if is_directory:
                            stack.append((resolved, identity(verified)))
            except OSError as error:
                raise WorkstationError(f"Dataset entry could not be inspected: {directory}") from error

    def run_dataset_inventory(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        if job["type"] != "dataset.inventory":
            raise WorkstationError("This job does not support local execution")
        if not job["project_id"]:
            raise WorkstationError("Dataset inventory requires a project")
        project = self.get_project(str(job["project_id"]))
        config = project.get("config")
        if not isinstance(config, Mapping):
            raise WorkstationError("Dataset project config is malformed")
        source = self.validate_import_path(config.get("source"))
        roots = self.allowed_import_roots()
        maximum = _positive_integer_env("ANIFLIVE_TTS_WORKSTATION_MAX_INVENTORY_FILES", 200000, maximum=10_000_000)
        claim = self.claim_job(job_id)
        media_files = total_bytes = examined = 0
        try:
            source_is_file = source.is_file()
            for path, info, is_directory in self._safe_inventory_entries(source, roots):
                examined += 1
                if examined > maximum:
                    raise WorkstationError("Dataset inventory exceeds the configured file limit")
                if examined == 1 or examined % 64 == 0:
                    if self.job_cancel_requested(job_id, claim.token):
                        return self.update_job(job_id, status="cancelled", claim_token=claim.token)
                    if self.job_pause_requested(job_id, claim.token):
                        return self.update_job(job_id, status="paused", claim_token=claim.token)
                    self.heartbeat_job(job_id, claim.token)
                if not is_directory and path.suffix.lower() in MEDIA_SUFFIXES:
                    media_files += 1
                    total_bytes += info.st_size
            return self.update_job(
                job_id, status="succeeded", progress=1.0, claim_token=claim.token,
                result={"media_files": media_files, "total_bytes": total_bytes,
                        "examined_files": examined, "source_kind": "file" if source_is_file else "directory"},
            )
        except Exception as error:
            try:
                if self.job_cancel_requested(job_id, claim.token):
                    return self.update_job(job_id, status="cancelled", claim_token=claim.token)
                if self.job_pause_requested(job_id, claim.token):
                    return self.update_job(job_id, status="paused", claim_token=claim.token)
                return self.update_job(
                    job_id, status="failed", error=_error_message(error),
                    claim_token=claim.token,
                )
            except WorkstationError:
                raise error

    def resolve_dataset_media_paths(
        self,
        values: Sequence[str],
        *,
        maximum_files: int = 2_048,
    ) -> tuple[Path, ...]:
        """Expand files and folders through the same contained inventory walker."""

        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise WorkstationError("Dataset sources must be a list")
        if (
            isinstance(maximum_files, bool)
            or not isinstance(maximum_files, int)
            or not 1 <= maximum_files <= 100_000
        ):
            raise WorkstationError("Dataset media expansion limit is malformed")
        roots = self.allowed_import_roots()
        resolved: dict[str, Path] = {}
        for value in values:
            source = self.validate_import_path(value)
            for path, _info, is_directory in self._safe_inventory_entries(source, roots):
                if is_directory or path.suffix.casefold() not in MEDIA_SUFFIXES:
                    continue
                resolved.setdefault(str(path).casefold(), path)
                if len(resolved) > maximum_files:
                    raise WorkstationError(
                        f"Dataset sources exceed the {maximum_files}-file preparation limit"
                    )
        if not resolved:
            raise WorkstationError("Dataset sources contain no supported audio or video files")
        return tuple(sorted(resolved.values(), key=lambda path: path.as_posix().casefold()))


__all__ = [
    "ARTIFACT_STATUSES", "ARTIFACT_TYPES", "EXPRESSION_QUALIFICATION_STATUSES",
    "GPU_RESOURCE_KEY", "JOB_PROJECT_KINDS", "JOB_RESOURCE_CLASSES", "JOB_STATES",
    "JOB_TRANSITIONS", "JOB_TYPES", "MAX_EXPRESSION_REFERENCE_BYTES", "PROJECT_KINDS",
    "REFERENCE_AUDIO_SUFFIXES", "RESOURCE_CLASSES", "JobClaim", "WorkstationError",
    "WorkstationSnapshot", "WorkstationStore", "default_workstation_root",
]
