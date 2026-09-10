from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4
import wave

import numpy as np

from .dataset_quality import (
    DatasetQualityError,
    analyze_pcm_quality,
    canonical_language,
    infer_text_language,
    normalize_transcript,
    parse_gpt_sovits_list,
)
from .dataset_acquisition import (
    DATASET_WORKFLOW_STAGES,
    PURITY_ROUTES,
    DatasetAcquisitionConfig,
    DatasetAcquisitionError,
    stable_clip_id,
)


DATASET_FACTORY_SCHEMA = 3
INGEST_SUFFIXES = frozenset(
    {
        ".wav",
        ".flac",
        ".mp3",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".mp4",
        ".mkv",
        ".mov",
    }
)
ITEM_KINDS = frozenset({"source", "resampled", "segment"})
PIPELINE_STATES = frozenset(
    {"ingested", "decoder-required", "resampled", "vad-analyzed", "segmented", "no-speech"}
)
REVIEW_STATES = frozenset({"pending", "accepted", "rejected"})
SPLIT_NAMES = frozenset({"train", "validation", "test"})
ANNOTATION_VERIFICATION_KINDS = frozenset({"transcript", "speaker", "expression"})
ANNOTATION_VERIFICATION_DECISIONS = frozenset({"verified", "rejected"})

_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_FILENAME_REPLACEMENTS = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")
_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class DatasetFactoryError(ValueError):
    """Raised when a Dataset Factory operation cannot produce a valid result."""


@dataclass(frozen=True)
class EnergyVadConfig:
    frame_ms: int = 20
    hop_ms: int = 10
    threshold_dbfs: float = -42.0
    min_speech_ms: int = 120
    min_silence_ms: int = 180
    pad_ms: int = 40
    max_segment_ms: int = 15_000

    def validated(self) -> "EnergyVadConfig":
        if not 5 <= self.frame_ms <= 100:
            raise DatasetFactoryError("frame_ms must be between 5 and 100")
        if not 1 <= self.hop_ms <= self.frame_ms:
            raise DatasetFactoryError("hop_ms must be between 1 and frame_ms")
        if not math.isfinite(self.threshold_dbfs) or not -100.0 <= self.threshold_dbfs <= 0.0:
            raise DatasetFactoryError("threshold_dbfs must be between -100 and 0")
        if not 0 <= self.min_speech_ms <= 60_000:
            raise DatasetFactoryError("min_speech_ms must be between 0 and 60000")
        if not 0 <= self.min_silence_ms <= 60_000:
            raise DatasetFactoryError("min_silence_ms must be between 0 and 60000")
        if not 0 <= self.pad_ms <= 5_000:
            raise DatasetFactoryError("pad_ms must be between 0 and 5000")
        if not 250 <= self.max_segment_ms <= 300_000:
            raise DatasetFactoryError("max_segment_ms must be between 250 and 300000")
        return self


@dataclass(frozen=True)
class SpeechRegion:
    start_frame: int
    end_frame: int
    peak_dbfs: float
    mean_dbfs: float

    def as_dict(self, sample_rate: int) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_seconds": self.start_frame / sample_rate,
            "end_seconds": self.end_frame / sample_rate,
            "duration_seconds": (self.end_frame - self.start_frame) / sample_rate,
            "peak_dbfs": self.peak_dbfs,
            "mean_dbfs": self.mean_dbfs,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _dataset_id(value: str) -> str:
    if not isinstance(value, str) or _DATASET_ID.fullmatch(value) is None:
        raise DatasetFactoryError("dataset_id is malformed")
    return value


def _json_text(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(dict(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise DatasetFactoryError("metadata must be JSON-compatible") from error


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _annotation_verification_value(
    kind: str, annotations: Mapping[str, Any]
) -> dict[str, Any] | None:
    if kind == "transcript":
        transcript = annotations.get("transcript")
        language = annotations.get("language")
        if not isinstance(transcript, str) or not transcript.strip() or not isinstance(language, str):
            return None
        try:
            return {
                "transcript": normalize_transcript(transcript),
                "language": canonical_language(language),
            }
        except DatasetQualityError:
            return None
    if kind == "speaker":
        speaker = annotations.get("speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            return None
        return {"speaker": speaker.strip()}
    if kind == "expression":
        expression = annotations.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            return None
        return {
            "expression": expression.strip(),
            "expression_intensity": annotations.get("expression_intensity"),
            "valence": annotations.get("valence"),
            "arousal": annotations.get("arousal"),
            "dominance": annotations.get("dominance"),
            "style_description": annotations.get("style_description"),
        }
    raise DatasetFactoryError("Unsupported annotation verification kind")


def _annotation_verification_sha256(
    kind: str, annotations: Mapping[str, Any]
) -> str | None:
    value = _annotation_verification_value(kind, annotations)
    return None if value is None else _canonical_json_sha256(value)


def _optional_text(value: Any, *, field: str, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DatasetFactoryError(f"{field} must be text or null")
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > limit:
        raise DatasetFactoryError(f"{field} is limited to {limit} characters")
    return normalized


def _optional_finite(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetFactoryError(f"{field} must be a number or null")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise DatasetFactoryError(
            f"{field} must be between {minimum:g} and {maximum:g}"
        )
    return result


def _safe_filename(value: str) -> str:
    name = _FILENAME_REPLACEMENTS.sub("_", value).strip(" .")
    if not name:
        name = "audio.wav"
    return name[:180]


def _is_within(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return path.is_symlink() or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _read_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getcomptype() != "NONE":
                raise DatasetFactoryError("Compressed WAV input is unsupported")
            channels = audio.getnchannels()
            sample_rate = audio.getframerate()
            sample_width = audio.getsampwidth()
            frame_count = audio.getnframes()
            raw = audio.readframes(frame_count)
    except (wave.Error, EOFError) as error:
        raise DatasetFactoryError(f"Invalid PCM WAV input: {path.name}") from error
    if channels < 1 or channels > 32 or sample_rate < 1 or frame_count < 0:
        raise DatasetFactoryError("WAV metadata is invalid")
    if sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8)
        if octets.size % 3:
            raise DatasetFactoryError("24-bit WAV payload is truncated")
        values = octets.reshape(-1, 3).astype(np.int32)
        integers = values[:, 0] | (values[:, 1] << 8) | (values[:, 2] << 16)
        integers = np.where(integers & 0x800000, integers - 0x1000000, integers)
        samples = integers.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise DatasetFactoryError(f"Unsupported PCM sample width: {sample_width}")
    if samples.size != frame_count * channels:
        raise DatasetFactoryError("WAV payload length does not match its header")
    return samples.reshape(frame_count, channels), sample_rate


def _write_pcm16_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or samples.shape[1] < 1:
        raise DatasetFactoryError("Audio samples must have shape [frames, channels]")
    if not np.isfinite(samples).all():
        raise DatasetFactoryError("Audio samples contain NaN or infinity")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
    try:
        with wave.open(str(temporary), "wb") as audio:
            audio.setnchannels(samples.shape[1])
            audio.setsampwidth(2)
            audio.setframerate(sample_rate)
            audio.writeframes(pcm.tobytes(order="C"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or samples.shape[0] == 0:
        return samples.astype(np.float32, copy=True)
    output_frames = max(1, int(round(samples.shape[0] * target_rate / source_rate)))
    source_positions = np.arange(samples.shape[0], dtype=np.float64)
    target_positions = np.arange(output_frames, dtype=np.float64) * source_rate / target_rate
    target_positions = np.minimum(target_positions, samples.shape[0] - 1)
    result = np.empty((output_frames, samples.shape[1]), dtype=np.float32)
    for channel in range(samples.shape[1]):
        result[:, channel] = np.interp(
            target_positions, source_positions, samples[:, channel]
        ).astype(np.float32)
    return result


def _wav_header(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getcomptype() != "NONE":
                raise DatasetFactoryError("Compressed WAV input is unsupported")
            channels = audio.getnchannels()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            sample_width = audio.getsampwidth()
    except (wave.Error, EOFError) as error:
        raise DatasetFactoryError(f"Invalid PCM WAV input: {path.name}") from error
    if channels < 1 or sample_rate < 1 or frame_count < 0 or sample_width not in {1, 2, 3, 4}:
        raise DatasetFactoryError("WAV metadata is invalid")
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "frame_count": frame_count,
        "duration_seconds": frame_count / sample_rate,
        "sample_width_bytes": sample_width,
    }


def _annotation_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    raw = metadata.get("annotations")
    values = raw if isinstance(raw, Mapping) else {}
    return {
        "transcript": values.get("transcript"),
        "language": values.get("language"),
        "speaker": values.get("speaker"),
        "expression": values.get("expression"),
        "expression_intensity": values.get("expression_intensity"),
        "valence": values.get("valence"),
        "arousal": values.get("arousal"),
        "dominance": values.get("dominance"),
        "style_description": values.get("style_description"),
        "language_diagnostic": values.get("language_diagnostic"),
        "source": values.get("source"),
        "updated_at": values.get("updated_at"),
    }


def _item_stage_state(item: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    state = str(item.get("pipeline_state") or "unknown")
    kind = str(item.get("kind") or "unknown")
    annotations = item.get("annotations") if isinstance(item.get("annotations"), Mapping) else {}
    has_annotation = bool(
        annotations.get("transcript")
        and annotations.get("language")
        and annotations.get("speaker")
    )
    is_pcm = str(item.get("stored_path") or "").lower().endswith(".wav")
    vad_done = state in {"vad-analyzed", "segmented", "no-speech"} or kind == "segment"
    segmented = state == "segmented" or kind == "segment"
    review_status = str(item.get("review_status") or "pending")
    return {
        "ingest": {"state": "complete", "detail": "SHA-256 verified local copy"},
        "decode": {
            "state": "complete" if is_pcm else "blocked",
            "detail": (
                "PCM WAV; decode not required"
                if is_pcm
                else "Queue dataset.process to decode this source in Linux Docker"
            ),
        },
        "normalize": {
            "state": "complete" if kind in {"resampled", "segment"} else ("ready" if is_pcm else "blocked"),
            "detail": "Canonical PCM available" if kind in {"resampled", "segment"} else "Ready for 32 kHz mono normalization",
        },
        "vad": {
            "state": "complete" if vad_done else ("ready" if is_pcm else "blocked"),
            "detail": "Speech regions analyzed" if vad_done else "Energy VAD is available for PCM WAV",
        },
        "segment": {
            "state": "complete" if segmented else ("blocked" if state == "no-speech" else "waiting"),
            "detail": "Reviewable segment" if segmented else ("No speech at the active threshold" if state == "no-speech" else "Awaiting VAD"),
        },
        "annotation": {
            "state": "complete" if has_annotation else "ready",
            "detail": "Transcript, language and speaker saved" if has_annotation else "Manual or GPT-SoVITS .list annotation required",
        },
        "quality": {
            "state": "complete" if item.get("quality") else ("ready" if is_pcm else "blocked"),
            "detail": "Deterministic PCM report saved" if item.get("quality") else "PCM analysis available" if is_pcm else "Decode to PCM before analysis",
        },
        "review": {
            "state": "complete" if kind == "segment" and review_status != "pending" else ("ready" if kind == "segment" else "waiting"),
            "detail": review_status if kind == "segment" else "Review applies to segments",
        },
        "split": {
            "state": "complete" if item.get("split_name") else ("ready" if review_status == "accepted" else "waiting"),
            "detail": str(item.get("split_name") or "Awaiting accepted review"),
        },
    }


class DatasetFactory:
    """Persistent Dataset Factory catalog and deterministic local PCM stages.

    Local transforms deliberately accept PCM WAV only. Other supported media remains
    ``decoder-required`` until the source-owned ``dataset.process`` Linux Docker job
    produces verified canonical artifacts. This prevents ingest from masquerading as
    a completed audio pipeline.
    """

    def __init__(
        self,
        root: Path,
        *,
        allowed_source_roots: Sequence[Path] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "dataset_factory.sqlite3"
        self.media_root = self.root / "dataset-media"
        self.media_root.mkdir(parents=True, exist_ok=True)
        self._source_roots_enforced = allowed_source_roots is not None
        if allowed_source_roots is None:
            self.allowed_source_roots: tuple[Path, ...] = ()
        else:
            roots: list[Path] = []
            for value in allowed_source_roots:
                path = Path(value).expanduser().resolve(strict=True)
                if not path.is_dir():
                    raise DatasetFactoryError("Allowed source roots must be directories")
                roots.append(path)
            self.allowed_source_roots = tuple(roots)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dataset_items(
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    parent_item_id TEXT REFERENCES dataset_items(id) ON DELETE RESTRICT,
                    kind TEXT NOT NULL CHECK(kind IN ('source', 'resampled', 'segment')),
                    source_path TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    byte_count INTEGER NOT NULL CHECK(byte_count >= 0),
                    sample_rate INTEGER,
                    channels INTEGER,
                    frame_count INTEGER,
                    duration_seconds REAL,
                    start_frame INTEGER,
                    end_frame INTEGER,
                    rms_dbfs REAL,
                    peak_dbfs REAL,
                    speaker_similarity REAL,
                    overlap_status TEXT,
                    pipeline_state TEXT NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'pending',
                    split_name TEXT,
                    review_note TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK(review_status IN ('pending', 'accepted', 'rejected')),
                    CHECK(speaker_similarity IS NULL OR
                        (speaker_similarity >= 0 AND speaker_similarity <= 1)),
                    CHECK(split_name IS NULL OR split_name IN ('train', 'validation', 'test'))
                );
                CREATE UNIQUE INDEX IF NOT EXISTS dataset_source_sha
                    ON dataset_items(dataset_id, sha256) WHERE kind = 'source';
                CREATE INDEX IF NOT EXISTS dataset_items_dataset
                    ON dataset_items(dataset_id, kind, review_status, split_name);
                CREATE INDEX IF NOT EXISTS dataset_items_parent
                    ON dataset_items(parent_item_id);
                CREATE TABLE IF NOT EXISTS dataset_vad_regions(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL REFERENCES dataset_items(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    start_frame INTEGER NOT NULL,
                    end_frame INTEGER NOT NULL,
                    peak_dbfs REAL NOT NULL,
                    mean_dbfs REAL NOT NULL,
                    UNIQUE(item_id, position)
                );
                CREATE TABLE IF NOT EXISTS dataset_reviews(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL REFERENCES dataset_items(id) ON DELETE CASCADE,
                    decision TEXT NOT NULL CHECK(decision IN ('pending', 'accepted', 'rejected')),
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dataset_annotation_verifications(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL REFERENCES dataset_items(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL CHECK(kind IN ('transcript', 'speaker', 'expression')),
                    decision TEXT NOT NULL CHECK(decision IN ('verified', 'rejected')),
                    value_sha256 TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS dataset_annotation_verifications_item
                    ON dataset_annotation_verifications(item_id, kind, id DESC);
                CREATE TABLE IF NOT EXISTS dataset_projects(
                    dataset_id TEXT PRIMARY KEY,
                    acquisition_mode TEXT NOT NULL,
                    lifecycle_stage TEXT NOT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    frozen_manifest_sha256 TEXT,
                    frozen_manifest_path TEXT,
                    frozen_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK(acquisition_mode IN ('standard', 'target-speaker', 'gpt-sovits-list')),
                    CHECK(lifecycle_stage IN ('source', 'speaker', 'clean', 'text', 'review', 'style', 'ready'))
                );
                """
            )
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES('schema', ?)",
                    (str(DATASET_FACTORY_SCHEMA),),
                )
            elif row["value"] in {"1", "2"}:
                connection.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'schema'",
                    (str(DATASET_FACTORY_SCHEMA),),
                )
            elif row["value"] != str(DATASET_FACTORY_SCHEMA):
                raise DatasetFactoryError("Unsupported Dataset Factory database schema")

    def _source_path(self, value: Path) -> Path:
        lexical = Path(value).expanduser()
        if not lexical.is_absolute():
            lexical = Path.cwd() / lexical
        lexical = Path(os.path.abspath(lexical))
        current = lexical
        while True:
            if _is_link_or_reparse(current):
                raise DatasetFactoryError("Links and reparse points are not accepted as dataset input")
            if current.parent == current:
                break
            current = current.parent
        candidate = lexical.resolve(strict=True)
        if self._source_roots_enforced:
            if not self.allowed_source_roots:
                raise DatasetFactoryError(
                    "Dataset ingest is disabled until import roots are configured"
                )
            if not _is_within(candidate, self.allowed_source_roots):
                raise DatasetFactoryError("Source path is outside the configured import roots")
        return candidate

    def _stored_file(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not _is_within(path, (self.root,)):
            raise DatasetFactoryError("Dataset item path escapes the workstation root")
        return path

    @staticmethod
    def _item(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        try:
            item["metadata"] = json.loads(item.pop("metadata_json"))
        except json.JSONDecodeError as error:
            raise DatasetFactoryError("Dataset item metadata is malformed") from error
        item["annotations"] = _annotation_payload(item["metadata"])
        quality = item["metadata"].get("quality")
        item["quality"] = dict(quality) if isinstance(quality, Mapping) else None
        item["stage_state"] = _item_stage_state(item)
        return item

    @staticmethod
    def _verification_state(
        connection: sqlite3.Connection,
        item: Mapping[str, Any],
        *,
        require_expressions: bool,
    ) -> dict[str, Any]:
        rows = connection.execute(
            "SELECT kind, decision, value_sha256, note, created_at FROM "
            "dataset_annotation_verifications WHERE item_id = ? ORDER BY id DESC",
            (item["id"],),
        ).fetchall()
        latest: dict[str, sqlite3.Row] = {}
        for row in rows:
            latest.setdefault(str(row["kind"]), row)
        annotations = item.get("annotations")
        values = annotations if isinstance(annotations, Mapping) else {}
        result: dict[str, dict[str, Any]] = {}
        for kind in sorted(ANNOTATION_VERIFICATION_KINDS):
            current_sha256 = _annotation_verification_sha256(kind, values)
            row = latest.get(kind)
            valid = bool(
                row is not None
                and row["decision"] == "verified"
                and row["value_sha256"] == current_sha256
            )
            result[kind] = {
                "status": "verified" if valid else "pending",
                "valid": valid,
                "current_sha256": current_sha256,
                "decision": None if row is None else row["decision"],
                "verified_value_sha256": None if row is None else row["value_sha256"],
                "note": "" if row is None else row["note"],
                "created_at": None if row is None else row["created_at"],
            }
        review_complete = bool(
            item.get("kind") == "segment"
            and (
                item.get("review_status") == "rejected"
                or (
                    item.get("review_status") == "accepted"
                    and result["transcript"]["valid"]
                    and result["speaker"]["valid"]
                    and (not require_expressions or result["expression"]["valid"])
                )
            )
        )
        return {
            "kinds": result,
            "require_expressions": require_expressions,
            "review_complete": review_complete,
        }

    def _enrich_item(
        self, connection: sqlite3.Connection, item: dict[str, Any]
    ) -> dict[str, Any]:
        project = connection.execute(
            "SELECT config_json FROM dataset_projects WHERE dataset_id = ?",
            (item["dataset_id"],),
        ).fetchone()
        require_expressions = True
        if project is not None:
            try:
                config = json.loads(project["config_json"])
            except json.JSONDecodeError as error:
                raise DatasetFactoryError("Dataset acquisition config is malformed") from error
            require_expressions = bool(config.get("require_expressions", True))
        verification = self._verification_state(
            connection, item, require_expressions=require_expressions
        )
        item["verification"] = verification["kinds"]
        item["review_complete"] = verification["review_complete"]
        item["stage_state"]["review"] = {
            "state": "complete" if item["review_complete"] else "ready",
            "detail": (
                "audio and required annotations verified"
                if item["review_complete"]
                else "audio, transcript and speaker review required"
            ),
        }
        return item

    def get_item(self, item_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM dataset_items WHERE id = ?", (item_id,)
            ).fetchone()
            if row is None:
                raise DatasetFactoryError("Dataset item was not found")
            return self._enrich_item(connection, self._item(row))

    def list_items(
        self,
        dataset_id: str,
        *,
        kind: str | None = None,
        review_status: str | None = None,
        split_name: str | None = None,
    ) -> list[dict[str, Any]]:
        dataset = _dataset_id(dataset_id)
        filters = ["dataset_id = ?"]
        parameters: list[Any] = [dataset]
        if kind is not None:
            if kind not in ITEM_KINDS:
                raise DatasetFactoryError("Unsupported dataset item kind")
            filters.append("kind = ?")
            parameters.append(kind)
        if review_status is not None:
            if review_status not in REVIEW_STATES:
                raise DatasetFactoryError("Unsupported review status")
            filters.append("review_status = ?")
            parameters.append(review_status)
        if split_name is not None:
            if split_name not in SPLIT_NAMES:
                raise DatasetFactoryError("Unsupported split name")
            filters.append("split_name = ?")
            parameters.append(split_name)
        query = "SELECT * FROM dataset_items WHERE " + " AND ".join(filters)
        query += " ORDER BY created_at, id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
            return [self._enrich_item(connection, self._item(row)) for row in rows]

    def ensure_project(
        self,
        dataset_id: str,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        dataset = _dataset_id(dataset_id)
        try:
            acquisition = DatasetAcquisitionConfig.from_mapping(config or {})
        except DatasetAcquisitionError as error:
            raise DatasetFactoryError(str(error)) from error
        now = _utc_now()
        payload = acquisition.as_dict()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM dataset_projects WHERE dataset_id = ?", (dataset,)
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO dataset_projects(dataset_id, acquisition_mode, "
                    "lifecycle_stage, config_json, created_at, updated_at) "
                    "VALUES(?, ?, 'source', ?, ?, ?)",
                    (dataset, acquisition.acquisition_mode, _json_text(payload), now, now),
                )
            else:
                if row["frozen_at"] is not None and (
                    row["acquisition_mode"] != acquisition.acquisition_mode
                    or json.loads(row["config_json"]) != payload
                ):
                    raise DatasetFactoryError("A frozen dataset project is immutable")
                if row["frozen_at"] is None:
                    connection.execute(
                        "UPDATE dataset_projects SET acquisition_mode = ?, "
                        "config_json = ?, updated_at = ? WHERE dataset_id = ?",
                        (acquisition.acquisition_mode, _json_text(payload), now, dataset),
                    )
        return self.project_state(dataset)

    def project_state(self, dataset_id: str) -> dict[str, Any]:
        dataset = _dataset_id(dataset_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM dataset_projects WHERE dataset_id = ?", (dataset,)
            ).fetchone()
        if row is None:
            return self.ensure_project(dataset, {})
        try:
            config = json.loads(row["config_json"])
        except json.JSONDecodeError as error:
            raise DatasetFactoryError("Dataset acquisition config is malformed") from error
        frozen = row["frozen_at"] is not None
        qualification_status = "not-frozen"
        if frozen:
            qualification_status = "legacy-frozen-unverified"
            manifest_relative = row["frozen_manifest_path"]
            if isinstance(manifest_relative, str):
                manifest_path = self._stored_file(manifest_relative)
                try:
                    frozen_document = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    frozen_document = None
                if (
                    isinstance(frozen_document, Mapping)
                    and frozen_document.get("schema") == "aniflive-frozen-dataset-v2"
                ):
                    qualification_status = "production-reviewed"
        return {
            "schema": "aniflive-dataset-project-state-v1",
            "dataset_id": dataset,
            "acquisition_mode": row["acquisition_mode"],
            "lifecycle_stage": row["lifecycle_stage"],
            "workflow_stages": list(DATASET_WORKFLOW_STAGES),
            "config": config,
            "frozen": frozen,
            "qualification_status": qualification_status,
            "frozen_manifest_sha256": row["frozen_manifest_sha256"],
            "frozen_manifest_path": row["frozen_manifest_path"],
            "frozen_at": row["frozen_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _assert_mutable_dataset(self, dataset_id: str) -> None:
        state = self.project_state(dataset_id)
        if state["frozen"]:
            raise DatasetFactoryError("A frozen dataset project is immutable")

    def review_summary(self, dataset_id: str) -> dict[str, Any]:
        dataset = _dataset_id(dataset_id)
        items = self.list_items(dataset, kind="segment")
        accepted = [item for item in items if item["review_status"] == "accepted"]
        rejected = [item for item in items if item["review_status"] == "rejected"]
        pending = [item for item in items if item["review_status"] == "pending"]
        transcript_pending = [
            item
            for item in accepted
            if not item["verification"]["transcript"]["valid"]
        ]
        speaker_pending = [
            item for item in accepted if not item["verification"]["speaker"]["valid"]
        ]
        expression_pending = [
            item
            for item in accepted
            if item["verification"]["expression"]["status"] != "verified"
        ]
        require_expressions = bool(
            self.project_state(dataset)["config"].get("require_expressions", True)
        )
        return {
            "dataset_id": dataset,
            "items": len(items),
            "audio": {
                "accepted": len(accepted),
                "rejected": len(rejected),
                "pending": len(pending),
            },
            "annotations": {
                "transcript_pending": len(transcript_pending),
                "speaker_pending": len(speaker_pending),
                "expression_pending": len(expression_pending),
                "expression_required": require_expressions,
            },
            "review_complete": bool(
                accepted
                and not pending
                and not transcript_pending
                and not speaker_pending
                and (not require_expressions or not expression_pending)
            ),
        }

    def advance_project_stage(self, dataset_id: str, stage: str) -> dict[str, Any]:
        dataset = _dataset_id(dataset_id)
        if stage not in DATASET_WORKFLOW_STAGES:
            raise DatasetFactoryError("Dataset lifecycle stage is unsupported")
        state = self.project_state(dataset)
        if state["frozen"]:
            if stage != "ready":
                raise DatasetFactoryError("A frozen dataset project is immutable")
            return state
        current = DATASET_WORKFLOW_STAGES.index(state["lifecycle_stage"])
        target = DATASET_WORKFLOW_STAGES.index(stage)
        if target < current:
            raise DatasetFactoryError("Dataset lifecycle stages cannot move backwards")
        with self._connect() as connection:
            connection.execute(
                "UPDATE dataset_projects SET lifecycle_stage = ?, updated_at = ? "
                "WHERE dataset_id = ?",
                (stage, _utc_now(), dataset),
            )
        return self.project_state(dataset)

    def import_standard_clips(
        self,
        dataset_id: str,
        records: Sequence[Mapping[str, Any]],
        *,
        parent_artifact_id: str | None = None,
    ) -> dict[str, Any]:
        """Import Docker-produced clips without losing source or ASR lineage."""

        dataset = _dataset_id(dataset_id)
        state = self.project_state(dataset)
        if state["acquisition_mode"] != "standard":
            raise DatasetFactoryError(
                "Processed clips require a standard Dataset project"
            )
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise DatasetFactoryError("Processed clip records must be a list")
        if not 1 <= len(records) <= 100_000:
            raise DatasetFactoryError(
                "Processed clip records must contain between 1 and 100000 entries"
            )
        if parent_artifact_id is not None and (
            not isinstance(parent_artifact_id, str) or len(parent_artifact_id) > 100
        ):
            raise DatasetFactoryError("parent_artifact_id is malformed")

        imported: list[dict[str, Any]] = []
        created_count = 0
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise DatasetFactoryError("Processed clip record must be an object")
            path_value = record.get("path")
            if not isinstance(path_value, str):
                raise DatasetFactoryError("Processed clip path is missing")
            source = self._source_path(Path(path_value))
            if not source.is_file() or source.suffix.casefold() != ".wav":
                raise DatasetFactoryError("Processed clips must be WAV files")
            source_sha = record.get("source_sha256")
            if (
                not isinstance(source_sha, str)
                or len(source_sha) != 64
                or any(character not in "0123456789abcdef" for character in source_sha)
            ):
                raise DatasetFactoryError("Processed source SHA-256 is malformed")
            start = record.get("source_start_sample")
            end = record.get("source_end_sample")
            position = record.get("position", index)
            try:
                clip_id = stable_clip_id(source_sha, start, end, position)
            except DatasetAcquisitionError as error:
                raise DatasetFactoryError(str(error)) from error
            sha256 = _sha256_file(source)
            expected_sha = record.get("sha256")
            if expected_sha is not None and expected_sha != sha256:
                raise DatasetFactoryError("Processed clip failed its checksum")
            destination = (
                self.media_root / dataset / "standard" / "clips" / f"{clip_id}.wav"
            )
            if destination.exists():
                if _sha256_file(destination) != sha256:
                    raise DatasetFactoryError("Stable processed clip name collided")
            else:
                _atomic_copy(source, destination)
            samples, sample_rate = _read_pcm_wav(destination)
            if samples.shape[1] != 1:
                raise DatasetFactoryError("Processed clips must be mono")
            try:
                quality = analyze_pcm_quality(samples, sample_rate).as_dict()
            except DatasetQualityError as error:
                raise DatasetFactoryError(str(error)) from error
            suggestion_value = record.get("annotations")
            suggestion = (
                dict(suggestion_value)
                if isinstance(suggestion_value, Mapping)
                else {}
            )
            metadata = {
                "stable_clip_id": clip_id,
                "quality": quality,
                "quality_source_sha256": sha256,
                "annotations": _annotation_payload({}),
                "lineage": {
                    "source_sha256": source_sha,
                    "source_start_sample": start,
                    "source_end_sample": end,
                    "parent_artifact_id": parent_artifact_id,
                },
                "acquisition": {
                    "mode": "standard",
                    "route": "clean",
                    "asr_suggestion": suggestion,
                },
            }
            relative = destination.relative_to(self.root).as_posix()
            with self._connect() as connection:
                existing = connection.execute(
                    "SELECT * FROM dataset_items WHERE dataset_id = ? AND stored_path = ?",
                    (dataset, relative),
                ).fetchone()
                if existing is not None:
                    item = self._item(existing)
                    existing_metadata = dict(item["metadata"])
                    acquisition = dict(existing_metadata.get("acquisition") or {})
                    acquisition.update(
                        {
                            "mode": "standard",
                            "route": "clean",
                            "asr_suggestion": suggestion,
                        }
                    )
                    existing_metadata.update(
                        {
                            "quality": quality,
                            "quality_source_sha256": sha256,
                            "lineage": metadata["lineage"],
                            "acquisition": acquisition,
                        }
                    )
                    connection.execute(
                        "UPDATE dataset_items SET metadata_json = ?, updated_at = ? "
                        "WHERE id = ?",
                        (_json_text(existing_metadata), _utc_now(), item["id"]),
                    )
                    refreshed = connection.execute(
                        "SELECT * FROM dataset_items WHERE id = ?", (item["id"],)
                    ).fetchone()
                    assert refreshed is not None
                    item = self._item(refreshed)
                else:
                    created_count += 1
                    item = self._insert_item(
                        connection,
                        dataset_id=dataset,
                        parent_item_id=None,
                        kind="segment",
                        source_path=str(record.get("source_path") or source),
                        stored_path=relative,
                        original_name=destination.name,
                        sha256=sha256,
                        byte_count=destination.stat().st_size,
                        sample_rate=sample_rate,
                        channels=1,
                        frame_count=samples.shape[0],
                        duration_seconds=samples.shape[0] / sample_rate,
                        start_frame=start,
                        end_frame=end,
                        pipeline_state="segmented",
                        metadata=metadata,
                    )
            imported.append(item)
        self.advance_project_stage(dataset, "review")
        return {
            "schema": "aniflive-standard-clip-import-v1",
            "dataset_id": dataset,
            "count": len(imported),
            "created_count": created_count,
            "deduplicated_count": len(imported) - created_count,
            "items": imported,
        }

    def import_target_speaker_clips(
        self,
        dataset_id: str,
        records: Sequence[Mapping[str, Any]],
        *,
        parent_artifact_id: str | None = None,
    ) -> dict[str, Any]:
        dataset = _dataset_id(dataset_id)
        state = self.project_state(dataset)
        if state["acquisition_mode"] != "target-speaker":
            raise DatasetFactoryError(
                "Target-speaker clips require a target-speaker dataset project"
            )
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise DatasetFactoryError("Target-speaker clip records must be a list")
        if not 1 <= len(records) <= 100_000:
            raise DatasetFactoryError(
                "Target-speaker clip records must contain between 1 and 100000 entries"
            )
        if parent_artifact_id is not None and (
            not isinstance(parent_artifact_id, str) or len(parent_artifact_id) > 100
        ):
            raise DatasetFactoryError("parent_artifact_id is malformed")
        imported: list[dict[str, Any]] = []
        created_count = 0
        counts = {route: 0 for route in sorted(PURITY_ROUTES)}
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise DatasetFactoryError("Target-speaker clip record must be an object")
            route = str(record.get("route", "review")).casefold()
            route = "salvage" if route == "salvaged" else route
            if route not in PURITY_ROUTES:
                raise DatasetFactoryError("Target-speaker clip route is unsupported")
            path_value = record.get("path")
            if not isinstance(path_value, str):
                raise DatasetFactoryError("Target-speaker clip path is missing")
            source = self._source_path(Path(path_value))
            if not source.is_file() or source.suffix.casefold() != ".wav":
                raise DatasetFactoryError("Target-speaker clips must be WAV files")
            source_sha = record.get("source_sha256")
            if (
                not isinstance(source_sha, str)
                or len(source_sha) != 64
                or any(character not in "0123456789abcdef" for character in source_sha)
            ):
                raise DatasetFactoryError("Target-speaker source SHA-256 is malformed")
            start = record.get("source_start_sample")
            end = record.get("source_end_sample")
            position = record.get("position", index)
            try:
                clip_id = stable_clip_id(source_sha, start, end, position)
            except DatasetAcquisitionError as error:
                raise DatasetFactoryError(str(error)) from error
            sha256 = _sha256_file(source)
            expected_sha = record.get("sha256")
            if expected_sha is not None and expected_sha != sha256:
                raise DatasetFactoryError("Target-speaker clip failed its checksum")
            folder = "rejected" if route == "reject" else "clips"
            destination = (
                self.media_root / dataset / "target-speaker" / folder / f"{clip_id}.wav"
            )
            if destination.exists():
                if _sha256_file(destination) != sha256:
                    raise DatasetFactoryError("Stable target-speaker clip name collided")
            else:
                _atomic_copy(source, destination)
            samples, sample_rate = _read_pcm_wav(destination)
            if samples.shape[1] != 1:
                raise DatasetFactoryError("Target-speaker clips must be mono")
            try:
                quality = analyze_pcm_quality(samples, sample_rate).as_dict()
            except DatasetQualityError as error:
                raise DatasetFactoryError(str(error)) from error
            speaker_value = record.get("speaker")
            speaker = dict(speaker_value) if isinstance(speaker_value, Mapping) else {}
            acquisition_value = record.get("acquisition")
            acquisition = (
                dict(acquisition_value) if isinstance(acquisition_value, Mapping) else {}
            )
            suggestion_value = record.get("annotations")
            suggestion = (
                dict(suggestion_value)
                if isinstance(suggestion_value, Mapping)
                else {}
            )
            metadata = {
                "stable_clip_id": clip_id,
                "quality": quality,
                "quality_source_sha256": sha256,
                "annotations": _annotation_payload({}),
                "lineage": {
                    "source_sha256": source_sha,
                    "source_start_sample": start,
                    "source_end_sample": end,
                    "parent_artifact_id": parent_artifact_id,
                },
                "speaker": speaker,
                "acquisition": {
                    "mode": "target-speaker",
                    "route": route,
                    "asr_suggestion": suggestion,
                    **acquisition,
                },
            }
            with self._connect() as connection:
                existing = connection.execute(
                    "SELECT * FROM dataset_items WHERE dataset_id = ? AND stored_path = ?",
                    (dataset, destination.relative_to(self.root).as_posix()),
                ).fetchone()
                if existing is not None:
                    item = self._item(existing)
                else:
                    created_count += 1
                    target_similarity = speaker.get("target_similarity")
                    item = self._insert_item(
                        connection,
                        dataset_id=dataset,
                        parent_item_id=None,
                        kind="segment",
                        source_path=str(
                            record.get("source_path")
                            or record.get("source_name")
                            or source
                        ),
                        stored_path=destination.relative_to(self.root).as_posix(),
                        original_name=destination.name,
                        sha256=sha256,
                        byte_count=destination.stat().st_size,
                        sample_rate=sample_rate,
                        channels=1,
                        frame_count=samples.shape[0],
                        duration_seconds=samples.shape[0] / sample_rate,
                        start_frame=start,
                        end_frame=end,
                        pipeline_state="segmented",
                        metadata=metadata,
                        rms_dbfs=quality.get("rms_dbfs"),
                        peak_dbfs=quality.get("peak_dbfs"),
                        speaker_similarity=(
                            float(target_similarity)
                            if isinstance(target_similarity, (int, float))
                            and not isinstance(target_similarity, bool)
                            else None
                        ),
                        overlap_status=(
                            "detected"
                            if bool(acquisition.get("overlap_evidence"))
                            else "clear"
                        ),
                    )
                    if route == "reject":
                        connection.execute(
                            "UPDATE dataset_items SET review_status = 'rejected', "
                            "review_note = 'Rejected by target-speaker purity routing', "
                            "updated_at = ? WHERE id = ?",
                            (_utc_now(), item["id"]),
                        )
                        row = connection.execute(
                            "SELECT * FROM dataset_items WHERE id = ?", (item["id"],)
                        ).fetchone()
                        assert row is not None
                        item = self._item(row)
            counts[route] += 1
            imported.append(item)
        self.advance_project_stage(dataset, "review")
        return {
            "schema": "aniflive-target-speaker-clip-import-v1",
            "dataset_id": dataset,
            "count": len(imported),
            "created_count": created_count,
            "deduplicated_count": len(imported) - created_count,
            "routes": counts,
            "items": imported,
        }

    def review_queue(self, dataset_id: str) -> dict[str, Any]:
        dataset = _dataset_id(dataset_id)
        items = [
            item
            for item in self.list_items(dataset, kind="segment")
            if item["review_status"] != "rejected" and not item["review_complete"]
        ]
        ordered = sorted(
            items,
            key=lambda item: (
                -float(item.get("speaker_similarity") or 0.0),
                -float((item.get("quality") or {}).get("quality_score") or 0.0),
                str(item.get("id")),
            ),
        )
        return {
            "schema": "aniflive-dataset-review-queue-v1",
            "dataset_id": dataset,
            "count": len(ordered),
            "items": ordered,
        }

    def expression_reference_candidates(self, dataset_id: str) -> dict[str, Any]:
        dataset = _dataset_id(dataset_id)
        accepted = self.list_items(dataset, kind="segment", review_status="accepted")
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in accepted:
            expression = item.get("annotations", {}).get("expression")
            if isinstance(expression, str) and expression.strip():
                quality = item.get("quality") or {}
                duration = float(item.get("duration_seconds") or 0.0)
                duration_score = max(0.0, 1.0 - abs(duration - 5.5) / 8.0)
                score = (
                    0.42 * float(quality.get("quality_score") or 0.0) / 100.0
                    + 0.33 * float(item.get("speaker_similarity") or 0.0)
                    + 0.25 * duration_score
                )
                groups.setdefault(expression.strip(), []).append(
                    {
                        "item_id": item["id"],
                        "stable_clip_id": item.get("metadata", {}).get("stable_clip_id"),
                        "duration_seconds": duration,
                        "speaker_similarity": item.get("speaker_similarity"),
                        "quality_score": quality.get("quality_score"),
                        "expression_intensity": item.get("annotations", {}).get(
                            "expression_intensity"
                        ),
                        "vad": {
                            field: item.get("annotations", {}).get(field)
                            for field in ("valence", "arousal", "dominance")
                            if item.get("annotations", {}).get(field) is not None
                        },
                        "style_description": item.get("annotations", {}).get(
                            "style_description"
                        ),
                        "score": round(score, 6),
                    }
                )
        return {
            "schema": "aniflive-expression-reference-candidates-v1",
            "dataset_id": dataset,
            "expressions": {
                expression: sorted(values, key=lambda value: (-value["score"], value["item_id"]))
                for expression, values in sorted(groups.items())
            },
        }

    def select_deployment_reference(self, dataset_id: str) -> dict[str, Any]:
        """Select a reviewed clip suitable for V2ProPlus package conditioning.

        Selection is model-agnostic and deterministic.  It prefers clean, high-quality
        speech near the middle of the converter's useful reference-duration range while
        rejecting clipped, mostly silent, or incompletely annotated clips.
        """

        dataset = _dataset_id(dataset_id)
        candidates: list[tuple[float, str, dict[str, Any]]] = []
        for item in self.list_items(dataset, kind="segment", review_status="accepted"):
            annotations = item.get("annotations") or {}
            quality = item.get("quality") or {}
            transcript = annotations.get("transcript")
            language = annotations.get("language")
            duration = float(item.get("duration_seconds") or 0.0)
            quality_score = float(quality.get("quality_score") or 0.0)
            silence_value = quality.get("silence_ratio")
            silence_ratio = (
                float(silence_value)
                if isinstance(silence_value, (int, float))
                and not isinstance(silence_value, bool)
                else 1.0
            )
            clipped_ratio = float(quality.get("clipped_ratio") or 0.0)
            acquisition = item.get("metadata", {}).get("acquisition", {})
            route = acquisition.get("route") if isinstance(acquisition, Mapping) else None
            if (
                not isinstance(transcript, str)
                or not transcript.strip()
                or language not in {"zh", "yue", "en", "ja", "ko"}
                or not 3.0 <= duration <= 10.0
                or quality_score < 70.0
                or silence_ratio > 0.4
                or clipped_ratio > 0.001
                or route not in {None, "clean", "salvaged"}
            ):
                continue
            duration_score = max(0.0, 1.0 - abs(duration - 5.5) / 4.5)
            score = (
                0.62 * quality_score / 100.0
                + 0.23 * duration_score
                + 0.15 * (1.0 - silence_ratio)
            )
            candidates.append((score, str(item["id"]), item))
        if not candidates:
            raise DatasetFactoryError(
                "Dataset has no reviewed 3-10 second reference clip that passes "
                "quality, clipping and silence gates"
            )
        score, _item_id, selected = max(candidates, key=lambda value: (value[0], value[1]))
        audio = self.verified_audio_path(str(selected["id"]))
        annotations = selected["annotations"]
        quality = selected.get("quality") or {}
        return {
            "schema": "aniflive-v2proplus-deployment-reference-v1",
            "dataset_id": dataset,
            "item_id": selected["id"],
            "path": str(audio),
            "sha256": selected["sha256"],
            "text": annotations["transcript"].strip(),
            "language": annotations["language"],
            "duration_seconds": selected["duration_seconds"],
            "quality_score": quality.get("quality_score"),
            "silence_ratio": quality.get("silence_ratio"),
            "selection_score": round(score, 6),
            "policy": "reviewed-quality-duration-v1",
        }

    def freeze_dataset(
        self,
        dataset_id: str,
        *,
        require_expressions: bool | None = None,
    ) -> dict[str, Any]:
        dataset = _dataset_id(dataset_id)
        state = self.project_state(dataset)
        if state["frozen"]:
            path = self._stored_file(state["frozen_manifest_path"])
            if not path.is_file() or _sha256_file(path) != state["frozen_manifest_sha256"]:
                raise DatasetFactoryError("Frozen dataset manifest failed integrity validation")
            return json.loads(path.read_text(encoding="utf-8"))
        configured_requirement = bool(state["config"].get("require_expressions", True))
        effective_requirement = (
            configured_requirement
            if require_expressions is None
            else bool(require_expressions)
        )
        if effective_requirement != configured_requirement:
            config = dict(state["config"])
            config["require_expressions"] = effective_requirement
            state = self.ensure_project(dataset, config)
        all_segments = self.list_items(dataset, kind="segment")
        accepted = [item for item in all_segments if item["review_status"] == "accepted"]
        if not accepted:
            raise DatasetFactoryError("Dataset freeze requires accepted clips")
        failures: list[str] = []
        for item in all_segments:
            if item["review_status"] == "pending":
                failures.append(f"{item['id']}:audio-pending")
        for item in accepted:
            annotations = item.get("annotations", {})
            if not all(annotations.get(field) for field in ("transcript", "language", "speaker")):
                failures.append(f"{item['id']}:annotations")
            if not item["verification"]["transcript"]["valid"]:
                failures.append(f"{item['id']}:transcript-unverified")
            if not item["verification"]["speaker"]["valid"]:
                failures.append(f"{item['id']}:speaker-unverified")
            if effective_requirement and not annotations.get("expression"):
                failures.append(f"{item['id']}:expression")
            if effective_requirement and not item["verification"]["expression"]["valid"]:
                failures.append(f"{item['id']}:expression-unverified")
            if item.get("split_name") is None:
                failures.append(f"{item['id']}:split")
        if len(accepted) >= 40:
            split_counts = {
                name: sum(item.get("split_name") == name for item in accepted)
                for name in SPLIT_NAMES
            }
            for name, minimum in {"train": 2, "validation": 5, "test": 5}.items():
                if split_counts[name] < minimum:
                    failures.append(f"production-split:{name}-requires-{minimum}")
        if failures:
            raise DatasetFactoryError(
                "Dataset freeze gate failed: " + ", ".join(failures[:12])
            )
        base_manifest = self.manifest(dataset)
        document = {
            "schema": "aniflive-frozen-dataset-v2",
            "dataset_id": dataset,
            "acquisition": state,
            "review_contract": {
                "audio_review": "accepted",
                "transcript_verification": "required",
                "speaker_verification": "required",
                "expression_verification": (
                    "required" if effective_requirement else "not-required"
                ),
            },
            "manifest": base_manifest,
            "expression_candidates": (
                self.expression_reference_candidates(dataset)
                if effective_requirement
                else {"dataset_id": dataset, "expressions": {}}
            ),
        }
        encoded = (
            json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        destination = self.media_root / dataset / "frozen" / f"{digest}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != encoded:
                raise DatasetFactoryError("Frozen dataset manifest path collided")
        else:
            temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
            try:
                with temporary.open("xb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        now = _utc_now()
        relative = destination.relative_to(self.root).as_posix()
        with self._connect() as connection:
            connection.execute(
                "UPDATE dataset_projects SET lifecycle_stage = 'ready', "
                "frozen_manifest_sha256 = ?, frozen_manifest_path = ?, frozen_at = ?, "
                "updated_at = ? WHERE dataset_id = ?",
                (digest, relative, now, now, dataset),
            )
        return document

    def verified_frozen_manifest_path(self, dataset_id: str) -> Path:
        state = self.project_state(dataset_id)
        if not state["frozen"]:
            raise DatasetFactoryError("Dataset has not been frozen")
        path = self._stored_file(state["frozen_manifest_path"])
        if not path.is_file() or _sha256_file(path) != state["frozen_manifest_sha256"]:
            raise DatasetFactoryError("Frozen dataset manifest failed integrity validation")
        return path

    def materialize_training_bundle(self, dataset_id: str) -> dict[str, Any]:
        """Build an immutable bundle with isolated train, validation and test splits."""

        dataset = _dataset_id(dataset_id)
        manifest_path = self.verified_frozen_manifest_path(dataset)
        frozen_sha256 = _sha256_file(manifest_path)
        try:
            frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DatasetFactoryError("Frozen dataset manifest could not be read") from error
        if not isinstance(frozen, Mapping) or frozen.get("schema") != "aniflive-frozen-dataset-v2":
            raise DatasetFactoryError(
                "Legacy frozen dataset has no annotation verification evidence and "
                "cannot start a new production training run"
            )
        base_manifest = frozen.get("manifest") if isinstance(frozen, Mapping) else None
        records = base_manifest.get("items") if isinstance(base_manifest, Mapping) else None
        if not isinstance(records, list):
            raise DatasetFactoryError("Frozen dataset manifest has no item inventory")
        split_records = {
            name: [record for record in records if record.get("split") == name]
            for name in ("train", "validation", "test")
        }
        if len(split_records["train"]) < 2:
            raise DatasetFactoryError("Training split requires at least two accepted clips")

        destination = self.media_root / dataset / "training" / frozen_sha256
        descriptor_path = destination / "training-input.json"

        def verify_existing() -> dict[str, Any]:
            try:
                descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise DatasetFactoryError("Training bundle descriptor is unreadable") from error
            if (
                not isinstance(descriptor, dict)
                or descriptor.get("schema") != "aniflive-v2proplus-training-input-v2"
                or descriptor.get("dataset_id") != dataset
                or descriptor.get("frozen_manifest_sha256") != frozen_sha256
            ):
                raise DatasetFactoryError("Training bundle descriptor does not match the frozen dataset")
            files = descriptor.get("audio_files")
            if not isinstance(files, list) or len(files) != len(records):
                raise DatasetFactoryError("Training bundle audio inventory is incomplete")
            for record in files:
                if not isinstance(record, Mapping):
                    raise DatasetFactoryError("Training bundle audio inventory is malformed")
                relative = record.get("path")
                sha256 = record.get("sha256")
                if (
                    not isinstance(relative, str)
                    or not any(relative.startswith(f"{name}/wav/") for name in SPLIT_NAMES)
                    or ".." in Path(relative).parts
                    or not isinstance(sha256, str)
                ):
                    raise DatasetFactoryError("Training bundle audio inventory is unsafe")
                candidate = destination / relative
                if not candidate.is_file() or candidate.is_symlink() or _sha256_file(candidate) != sha256:
                    raise DatasetFactoryError("Training bundle audio failed integrity validation")
            list_path = destination / "train" / "voice.list"
            if (
                not list_path.is_file()
                or _sha256_file(list_path) != descriptor.get("training_list_sha256")
            ):
                raise DatasetFactoryError("Training bundle list failed integrity validation")
            return {
                "schema": descriptor["schema"],
                "dataset_id": dataset,
                "path": str(destination),
                "training_path": str(destination / "train"),
                "examples": len(split_records["train"]),
                "split_counts": {
                    name: len(values) for name, values in split_records.items()
                },
                "frozen_manifest_sha256": frozen_sha256,
                "training_list_sha256": descriptor["training_list_sha256"],
                "descriptor_sha256": _sha256_file(descriptor_path),
            }

        if destination.exists():
            return verify_existing()

        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        list_rows: list[str] = []
        audio_files: list[dict[str, Any]] = []
        try:
            for split_name, split_values in split_records.items():
                split_root = temporary / split_name
                audio_root = split_root / "wav"
                audio_root.mkdir(parents=True)
                split_manifest_records: list[dict[str, Any]] = []
                for index, record in enumerate(split_values):
                    if not isinstance(record, Mapping):
                        raise DatasetFactoryError("Frozen dataset item is malformed")
                    annotations = record.get("annotations")
                    verifications = record.get("annotation_verifications")
                    if not isinstance(annotations, Mapping) or not isinstance(verifications, Mapping):
                        raise DatasetFactoryError("Frozen dataset review evidence is incomplete")
                    transcript = annotations.get("transcript")
                    language = annotations.get("language")
                    speaker = annotations.get("speaker")
                    fields = (transcript, language, speaker)
                    if any(not isinstance(value, str) or not value.strip() for value in fields):
                        raise DatasetFactoryError("Frozen training annotations are incomplete")
                    if any("|" in value or "\n" in value or "\r" in value for value in fields):
                        raise DatasetFactoryError("Training annotations cannot contain pipes or newlines")
                    if not all(
                        isinstance(verifications.get(kind), Mapping)
                        and verifications[kind].get("valid") is True
                        for kind in ("transcript", "speaker")
                    ):
                        raise DatasetFactoryError("Frozen item lacks valid review verification hashes")
                    source_relative = record.get("path")
                    expected_sha = record.get("sha256")
                    if not isinstance(source_relative, str) or not isinstance(expected_sha, str):
                        raise DatasetFactoryError("Frozen training audio metadata is malformed")
                    source = self._stored_file(source_relative)
                    if (
                        not source.is_file()
                        or source.is_symlink()
                        or _sha256_file(source) != expected_sha
                    ):
                        raise DatasetFactoryError("Frozen training audio failed integrity validation")
                    name = f"seg_{index:06d}_{expected_sha[:12]}.wav"
                    target = audio_root / name
                    _atomic_copy(source, target)
                    if _sha256_file(target) != expected_sha:
                        raise DatasetFactoryError("Copied training audio failed integrity validation")
                    if split_name == "train":
                        list_rows.append(
                            f"{name}|{speaker.strip()}|{language.strip().casefold()}|{transcript.strip()}"
                        )
                    inventory = {
                        "path": f"{split_name}/wav/{name}",
                        "sha256": expected_sha,
                        "source_item_id": record.get("id"),
                        "source_parent_item_id": record.get("source_item_id"),
                        "split": split_name,
                        "duration_seconds": record.get("duration_seconds"),
                        "transcript": transcript.strip(),
                        "language": language.strip().casefold(),
                        "speaker": speaker.strip(),
                        "quality": record.get("quality"),
                        "acquisition_route": (
                            (record.get("metadata") or {}).get("acquisition", {}).get(
                                "route", "clean"
                            )
                            if isinstance(record.get("metadata"), Mapping)
                            and isinstance(
                                (record.get("metadata") or {}).get("acquisition", {}),
                                Mapping,
                            )
                            else "clean"
                        ),
                        "verification_hashes": {
                            kind: verifications[kind].get("verified_value_sha256")
                            for kind in ANNOTATION_VERIFICATION_KINDS
                            if isinstance(verifications.get(kind), Mapping)
                            and verifications[kind].get("valid") is True
                        },
                    }
                    audio_files.append(inventory)
                    split_manifest_records.append(inventory)
                (split_root / "manifest.json").write_text(
                    json.dumps(
                        {
                            "schema": "aniflive-training-split-manifest-v1",
                            "dataset_id": dataset,
                            "split": split_name,
                            "count": len(split_manifest_records),
                            "items": split_manifest_records,
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            training_list = temporary / "train" / "voice.list"
            training_list.write_text("\n".join(list_rows) + "\n", encoding="utf-8")
            shutil.copy2(manifest_path, temporary / "frozen-manifest.json")
            descriptor = {
                "schema": "aniflive-v2proplus-training-input-v2",
                "dataset_id": dataset,
                "frozen_manifest_sha256": frozen_sha256,
                "training_list_sha256": _sha256_file(training_list),
                "examples": len(split_records["train"]),
                "split_counts": {
                    name: len(values) for name, values in split_records.items()
                },
                "training_entrypoint": "train/voice.list",
                "validation_entrypoint": "validation/manifest.json",
                "test_entrypoint": "test/manifest.json",
                "audio_files": audio_files,
            }
            descriptor_path_temporary = temporary / "training-input.json"
            descriptor_path_temporary.write_text(
                json.dumps(descriptor, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, destination)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return verify_existing()

    @staticmethod
    def capabilities() -> dict[str, Any]:
        return {
            "schema": "aniflive-dataset-capabilities-v1",
            "contract_revision": 2,
            "execution_environment": "linux-docker-for-media-and-neural-stages",
            "acquisition_modes": ["standard", "target-speaker", "gpt-sovits-list"],
            "workflow_stages": list(DATASET_WORKFLOW_STAGES),
            "jobs": {
                "dataset.process": {
                    "available": True,
                    "mode": "linux-docker",
                    "resource_class": "gpu-exclusive",
                    "canonical_audio": "mono-pcm16-32000hz",
                    "artifacts": ["canonical.wav", "dataset-report.json", "segments/*.wav"],
                    "network_required": False,
                }
            },
            "stages": {
                "ingest": {"available": True, "mode": "local-deterministic"},
                "decode": {
                    "available": True,
                    "mode": "delegated-linux-docker",
                    "job_type": "dataset.process",
                    "reason": "Deterministic first-audio-stream decode to mono PCM16 at 32 kHz.",
                },
                "resample": {"available": True, "mode": "local-deterministic-pcm"},
                "vad": {
                    "available": True,
                    "mode": "managed-component-linux-docker",
                    "standard": "managed-fsmn-vad-linux-docker",
                    "target_speaker": "managed-fsmn-vad-linux-docker",
                },
                "tse": {
                    "available": True,
                    "mode": "delegated",
                    "entrypoint": "dataset-stage-linux-docker",
                    "job_type": "dataset.target-speaker",
                    "reason": (
                        "Target-speaker acquisition emits checksum-verified per-clip "
                        "artifacts with source sample lineage."
                    ),
                },
                "denoise": {
                    "available": True,
                    "mode": "delegated-linux-docker-optional",
                    "job_type": "dataset.process",
                    "reason": (
                        "Optional conservative FFmpeg afftdn; disabled unless "
                        "explicitly requested."
                    ),
                },
                "dereverb": {
                    "available": False,
                    "mode": "qualified-backend-required",
                    "configured_backend": "none",
                    "qualification_record": (
                        "docs/research/v1.4-dataset-dereverb-qualification.md"
                    ),
                    "reason": (
                        "No redistribution-safe dereverb backend with pinned assets has "
                        "passed the Linux Docker qualification contract."
                    ),
                },
                "segment": {"available": True, "mode": "mode-specific-vad"},
                "asr": {
                    "available": True,
                    "mode": "managed-component-linux-docker",
                    "job_type": "dataset.process",
                    "job_types": ["dataset.process", "dataset.transcribe"],
                    "default_backend": "sensevoice-small",
                    "fallback_backend": "faster-whisper-small",
                    "required_input": "asr_model",
                    "supported_languages": ["yue", "zh", "ja", "en", "ko"],
                    "network_required": False,
                    "reason": (
                        "Managed pinned assets transcribe checksum-verified segments "
                        "inside the CUDA worker; every transcript still requires review."
                    ),
                },
                "annotation": {"available": True, "mode": "manual-or-list-import"},
                "quality": {"available": True, "mode": "local-deterministic-pcm"},
                "review": {"available": True, "mode": "manual"},
                "split": {"available": True, "mode": "deterministic"},
            },
        }

    def verified_audio_path(self, item_id: str) -> Path:
        item = self.get_item(item_id)
        path = self._stored_file(item["stored_path"])
        if path.suffix.casefold() != ".wav":
            raise DatasetFactoryError("A trusted decoder is required before audio preview")
        if not path.is_file() or _sha256_file(path) != item["sha256"]:
            raise DatasetFactoryError("The dataset audio failed integrity verification")
        return path

    def waveform(self, item_id: str, *, bins: int = 640) -> dict[str, Any]:
        if not 32 <= bins <= 2_048:
            raise DatasetFactoryError("waveform bins must be between 32 and 2048")
        path = self.verified_audio_path(item_id)
        samples, sample_rate = _read_pcm_wav(path)
        mono = samples.mean(axis=1, dtype=np.float32)
        boundaries = np.linspace(0, mono.size, min(bins, max(1, mono.size)) + 1, dtype=int)
        peaks: list[list[float]] = []
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            window = mono[start:end]
            peaks.append(
                [
                    round(float(np.min(window)), 6) if window.size else 0.0,
                    round(float(np.max(window)), 6) if window.size else 0.0,
                ]
            )
        return {
            "schema": "aniflive-dataset-waveform-v1",
            "item_id": item_id,
            "sample_rate": sample_rate,
            "frame_count": int(mono.size),
            "duration_seconds": mono.size / sample_rate,
            "peaks": peaks,
        }

    def analyze_quality(self, item_id: str) -> dict[str, Any]:
        item = self.get_item(item_id)
        path = self.verified_audio_path(item_id)
        samples, sample_rate = _read_pcm_wav(path)
        try:
            report = analyze_pcm_quality(samples, sample_rate).as_dict()
        except DatasetQualityError as error:
            raise DatasetFactoryError(str(error)) from error
        metadata = dict(item["metadata"])
        metadata["quality"] = report
        metadata["quality_source_sha256"] = item["sha256"]
        with self._connect() as connection:
            connection.execute(
                "UPDATE dataset_items SET metadata_json = ?, updated_at = ? WHERE id = ?",
                (_json_text(metadata), _utc_now(), item_id),
            )
        return self.get_item(item_id)

    def update_annotations(
        self,
        item_id: str,
        values: Mapping[str, Any],
        *,
        source: str = "manual",
        provenance: Mapping[str, Any] | None = None,
        propagate: bool = True,
    ) -> dict[str, Any]:
        allowed = {
            "transcript",
            "language",
            "speaker",
            "expression",
            "expression_intensity",
            "valence",
            "arousal",
            "dominance",
            "style_description",
        }
        unknown = set(values) - allowed
        if unknown:
            raise DatasetFactoryError(f"Unsupported annotation fields: {', '.join(sorted(unknown))}")
        item = self.get_item(item_id)
        self._assert_mutable_dataset(str(item["dataset_id"]))
        current = _annotation_payload(item["metadata"])
        try:
            if "transcript" in values:
                value = values["transcript"]
                current["transcript"] = None if value is None or value == "" else normalize_transcript(value)
            if "language" in values:
                value = values["language"]
                current["language"] = None if value is None or value == "" else canonical_language(value)
        except DatasetQualityError as error:
            raise DatasetFactoryError(str(error)) from error
        if "speaker" in values:
            current["speaker"] = _optional_text(values["speaker"], field="speaker", limit=160)
        if "expression" in values:
            current["expression"] = _optional_text(
                values["expression"], field="expression", limit=240
            )
        if "expression_intensity" in values:
            current["expression_intensity"] = _optional_finite(
                values["expression_intensity"],
                field="expression_intensity",
                minimum=0.0,
                maximum=1.0,
            )
        for field in ("valence", "arousal", "dominance"):
            if field in values:
                current[field] = _optional_finite(
                    values[field], field=field, minimum=-1.0, maximum=1.0
                )
        if "style_description" in values:
            current["style_description"] = _optional_text(
                values["style_description"],
                field="style_description",
                limit=2_000,
            )
        if current.get("transcript"):
            try:
                current["language_diagnostic"] = infer_text_language(current["transcript"])
            except DatasetQualityError:
                current["language_diagnostic"] = None
        else:
            current["language_diagnostic"] = None
        current["source"] = _optional_text(source, field="annotation source", limit=80)
        current["updated_at"] = _utc_now()

        updates: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
            (item_id, dict(item["metadata"]), current)
        ]
        if propagate:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT id, parent_item_id, kind, metadata_json FROM dataset_items "
                    "WHERE dataset_id = ? ORDER BY created_at, id",
                    (item["dataset_id"],),
                ).fetchall()
            children: dict[str, list[sqlite3.Row]] = {}
            for row in rows:
                if row["parent_item_id"]:
                    children.setdefault(row["parent_item_id"], []).append(row)
            queue = list(children.get(item_id, ()))
            while queue:
                row = queue.pop(0)
                try:
                    metadata = json.loads(row["metadata_json"])
                except json.JSONDecodeError as error:
                    raise DatasetFactoryError("Dataset item metadata is malformed") from error
                inherited = dict(current)
                segment_siblings = [
                    sibling
                    for sibling in children.get(row["parent_item_id"], ())
                    if sibling["kind"] == "segment"
                ]
                if row["kind"] == "segment" and len(segment_siblings) > 1:
                    inherited["transcript"] = None
                    inherited["language_diagnostic"] = None
                    inherited["source"] = "inherited-requires-transcript"
                updates.append((row["id"], metadata, inherited))
                queue.extend(children.get(row["id"], ()))

        with self._connect() as connection:
            for target_id, metadata, annotation in updates:
                metadata["annotations"] = annotation
                if provenance is not None:
                    metadata["annotation_provenance"] = dict(provenance)
                connection.execute(
                    "UPDATE dataset_items SET metadata_json = ?, updated_at = ? WHERE id = ?",
                    (_json_text(metadata), current["updated_at"], target_id),
                )
        return self.get_item(item_id)

    def verify_annotation(
        self,
        item_id: str,
        *,
        kind: str,
        decision: str = "verified",
        note: str = "",
    ) -> dict[str, Any]:
        if kind not in ANNOTATION_VERIFICATION_KINDS:
            raise DatasetFactoryError("Unsupported annotation verification kind")
        if decision not in ANNOTATION_VERIFICATION_DECISIONS:
            raise DatasetFactoryError("Unsupported annotation verification decision")
        if not isinstance(note, str) or len(note) > 2_000:
            raise DatasetFactoryError("Verification note is limited to 2000 characters")
        item = self.get_item(item_id)
        self._assert_mutable_dataset(str(item["dataset_id"]))
        if item["kind"] != "segment":
            raise DatasetFactoryError("Only segmented audio annotations can be verified")
        value_sha256 = _annotation_verification_sha256(kind, item["annotations"])
        if value_sha256 is None:
            raise DatasetFactoryError(f"{kind} annotation is incomplete")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO dataset_annotation_verifications"
                "(item_id, kind, decision, value_sha256, note, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (item_id, kind, decision, value_sha256, note.strip(), _utc_now()),
            )
        result = self.get_item(item_id)
        summary = self.review_summary(str(result["dataset_id"]))
        if summary["review_complete"]:
            self.advance_project_stage(str(result["dataset_id"]), "style")
        return result

    def annotation_verification_history(
        self, item_id: str, *, kind: str | None = None
    ) -> list[dict[str, Any]]:
        self.get_item(item_id)
        parameters: list[Any] = [item_id]
        where = "item_id = ?"
        if kind is not None:
            if kind not in ANNOTATION_VERIFICATION_KINDS:
                raise DatasetFactoryError("Unsupported annotation verification kind")
            where += " AND kind = ?"
            parameters.append(kind)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT kind, decision, value_sha256, note, created_at FROM "
                f"dataset_annotation_verifications WHERE {where} ORDER BY id",
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def import_gpt_sovits_list(self, dataset_id: str, path: Path) -> dict[str, Any]:
        dataset = _dataset_id(dataset_id)
        source = self._source_path(Path(path))
        try:
            entries = parse_gpt_sovits_list(source)
        except DatasetQualityError as error:
            raise DatasetFactoryError(str(error)) from error
        normalized: list[tuple[Any, Path]] = []
        seen: dict[Path, tuple[str, str, str]] = {}
        seen_content: dict[str, tuple[str, str, str]] = {}
        for entry in entries:
            audio_path = self._source_path(entry.audio_path)
            if audio_path.suffix.casefold() not in INGEST_SUFFIXES:
                raise DatasetFactoryError(
                    f"GPT-SoVITS .list line {entry.line_number} uses an unsupported media suffix"
                )
            signature = (entry.speaker, entry.language, entry.text)
            if audio_path in seen and seen[audio_path] != signature:
                raise DatasetFactoryError(
                    f"GPT-SoVITS .list line {entry.line_number} conflicts with an earlier "
                    "annotation for the same audio file"
                )
            content_hash = _sha256_file(audio_path)
            if content_hash in seen_content and seen_content[content_hash] != signature:
                raise DatasetFactoryError(
                    f"GPT-SoVITS .list line {entry.line_number} conflicts with an earlier "
                    "annotation for identical audio content"
                )
            seen[audio_path] = signature
            seen_content[content_hash] = signature
            normalized.append((entry, audio_path))
        imported: list[dict[str, Any]] = []
        completed: set[str] = set()
        for entry, audio_path in normalized:
            item = self.ingest(dataset, (audio_path,), recursive=False)[0]
            if item["id"] in completed:
                continue
            completed.add(item["id"])
            if item["quality"] is None and audio_path.suffix.casefold() == ".wav":
                item = self.analyze_quality(item["id"])
            item = self.update_annotations(
                item["id"],
                {
                    "transcript": entry.text, "language": entry.language,
                    "speaker": entry.speaker, "expression": None,
                },
                source="gpt-sovits-list",
                provenance={
                    "format": "gpt-sovits-list-v1", "list_path": str(source),
                    "line_number": entry.line_number,
                },
            )
            if Path(item["stored_path"]).suffix.casefold() == ".wav":
                item = self._full_list_clip(item)
            imported.append(item)
        return {
            "schema": "aniflive-gpt-sovits-list-import-v1",
            "dataset_id": dataset,
            "source_list": str(source),
            "entry_count": len(entries),
            "item_count": len(imported),
            "items": imported,
        }

    def _full_list_clip(self, source: Mapping[str, Any]) -> dict[str, Any]:
        """Preserve an already segmented .list WAV without running VAD again."""
        self._assert_mutable_dataset(str(source["dataset_id"]))
        original = self._stored_file(source["stored_path"])
        samples, rate = _read_pcm_wav(original)
        if samples.shape[0] <= 0:
            raise DatasetFactoryError("GPT-SoVITS .list clip is empty")
        contract = "gpt-sovits-list-full-span-v1"
        destination = (
            self.media_root / source["dataset_id"] / "segments" / source["id"]
            / f"full-{source['sha256'][:16]}.wav"
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM dataset_items WHERE parent_item_id = ? AND kind = 'segment'",
                (source["id"],),
            ).fetchall()
            for row in rows:
                item = self._item(row)
                if item["metadata"].get("segment_contract") == contract:
                    path = self._stored_file(item["stored_path"])
                    if path.is_file() and _sha256_file(path) == source["sha256"]:
                        result = item
                        break
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, destination)
                if _sha256_file(destination) != source["sha256"]:
                    raise DatasetFactoryError("Full-span list clip failed integrity validation")
                quality = analyze_pcm_quality(samples, rate).as_dict()
                result = self._insert_item(
                    connection, dataset_id=source["dataset_id"], parent_item_id=source["id"],
                    kind="segment", source_path=source["source_path"],
                    stored_path=destination.relative_to(self.root).as_posix(),
                    original_name=source["original_name"], sha256=source["sha256"],
                    byte_count=destination.stat().st_size, sample_rate=rate,
                    channels=samples.shape[1], frame_count=samples.shape[0],
                    duration_seconds=samples.shape[0] / rate, start_frame=0,
                    end_frame=samples.shape[0], pipeline_state="segmented",
                    metadata={
                        "segment_contract": contract, "source_sha256": source["sha256"],
                        "annotations": _annotation_payload(source["metadata"]),
                        "quality": quality, "quality_source_sha256": source["sha256"],
                    },
                    rms_dbfs=quality["rms_dbfs"], peak_dbfs=quality["peak_dbfs"],
                )
            connection.commit()
        annotations = _annotation_payload(source["metadata"])
        return self.update_annotations(
            result["id"],
            {key: annotations.get(key) for key in ("transcript", "language", "speaker", "expression")},
            source="gpt-sovits-list", provenance={"segment_contract": contract},
        )

    def _insert_item(
        self,
        connection: sqlite3.Connection,
        *,
        dataset_id: str,
        parent_item_id: str | None,
        kind: str,
        source_path: str,
        stored_path: str,
        original_name: str,
        sha256: str,
        byte_count: int,
        sample_rate: int | None,
        channels: int | None,
        frame_count: int | None,
        duration_seconds: float | None,
        start_frame: int | None,
        end_frame: int | None,
        pipeline_state: str,
        metadata: Mapping[str, Any],
        rms_dbfs: float | None = None,
        peak_dbfs: float | None = None,
        speaker_similarity: float | None = None,
        overlap_status: str | None = None,
    ) -> dict[str, Any]:
        if kind not in ITEM_KINDS or pipeline_state not in PIPELINE_STATES:
            raise DatasetFactoryError("Dataset item state is unsupported")
        item_id = f"item_{uuid4()}"
        now = _utc_now()
        connection.execute(
            """
            INSERT INTO dataset_items(
                id, dataset_id, parent_item_id, kind, source_path, stored_path,
                original_name, sha256, byte_count, sample_rate, channels,
                frame_count, duration_seconds, start_frame, end_frame,
                rms_dbfs, peak_dbfs, speaker_similarity, overlap_status,
                pipeline_state, metadata_json, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                dataset_id,
                parent_item_id,
                kind,
                source_path,
                stored_path,
                original_name,
                sha256,
                byte_count,
                sample_rate,
                channels,
                frame_count,
                duration_seconds,
                start_frame,
                end_frame,
                rms_dbfs,
                peak_dbfs,
                speaker_similarity,
                overlap_status,
                pipeline_state,
                _json_text(metadata),
                now,
                now,
            ),
        )
        row = connection.execute("SELECT * FROM dataset_items WHERE id = ?", (item_id,)).fetchone()
        assert row is not None
        return self._item(row)

    def _candidate_files(self, sources: Sequence[Path], *, recursive: bool) -> list[Path]:
        candidates: set[Path] = set()
        for raw in sources:
            source = self._source_path(Path(raw))
            if source.is_file():
                if source.suffix.lower() in INGEST_SUFFIXES:
                    candidates.add(source)
                continue
            if not source.is_dir():
                raise DatasetFactoryError("Dataset source must be a file or directory")
            iterator = source.rglob("*") if recursive else source.glob("*")
            for entry in iterator:
                if entry.is_symlink():
                    continue
                if entry.is_file() and entry.suffix.lower() in INGEST_SUFFIXES:
                    candidates.add(entry.resolve(strict=True))
        return sorted(candidates, key=lambda value: str(value).casefold())

    def ingest(
        self,
        dataset_id: str,
        sources: Sequence[Path],
        *,
        recursive: bool = True,
    ) -> list[dict[str, Any]]:
        dataset = _dataset_id(dataset_id)
        if isinstance(sources, (str, bytes, Path)) or not isinstance(sources, Sequence):
            raise DatasetFactoryError("sources must be a sequence of paths")
        candidates = self._candidate_files([Path(value) for value in sources], recursive=recursive)
        results: list[dict[str, Any]] = []
        for source in candidates:
            suffix = source.suffix.lower()
            sha256 = _sha256_file(source)
            with self._connect() as connection:
                existing = connection.execute(
                    "SELECT * FROM dataset_items WHERE dataset_id = ? AND kind = 'source' "
                    "AND sha256 = ?",
                    (dataset, sha256),
                ).fetchone()
            if existing is not None:
                results.append(self._item(existing))
                continue
            destination = (
                self.media_root
                / dataset
                / "source"
                / sha256[:2]
                / f"{sha256}_{_safe_filename(source.name)}"
            )
            if not destination.exists():
                _atomic_copy(source, destination)
            if _sha256_file(destination) != sha256:
                raise DatasetFactoryError("The ingested copy failed its SHA-256 verification")
            header: dict[str, Any] = {}
            quality: dict[str, Any] | None = None
            state = "decoder-required"
            if suffix == ".wav":
                header = _wav_header(destination)
                samples, sample_rate = _read_pcm_wav(destination)
                try:
                    quality = analyze_pcm_quality(samples, sample_rate).as_dict()
                except DatasetQualityError as error:
                    raise DatasetFactoryError(str(error)) from error
                state = "ingested"
            relative = destination.relative_to(self.root).as_posix()
            with self._connect() as connection:
                item = self._insert_item(
                    connection,
                    dataset_id=dataset,
                    parent_item_id=None,
                    kind="source",
                    source_path=str(source),
                    stored_path=relative,
                    original_name=source.name,
                    sha256=sha256,
                    byte_count=destination.stat().st_size,
                    sample_rate=header.get("sample_rate"),
                    channels=header.get("channels"),
                    frame_count=header.get("frame_count"),
                    duration_seconds=header.get("duration_seconds"),
                    start_frame=None,
                    end_frame=None,
                    pipeline_state=state,
                    metadata={
                        "processing_supported": suffix == ".wav",
                        "media_suffix": suffix,
                        **(
                            {"sample_width_bytes": header["sample_width_bytes"]}
                            if header
                            else {}
                        ),
                        **({"quality": quality, "quality_source_sha256": sha256} if quality else {}),
                    },
                )
            results.append(item)
        return results

    def resample(
        self,
        item_id: str,
        *,
        target_rate: int = 32_000,
        mono: bool = True,
    ) -> dict[str, Any]:
        if not 8_000 <= target_rate <= 192_000:
            raise DatasetFactoryError("target_rate must be between 8000 and 192000")
        item = self.get_item(item_id)
        if item["kind"] not in {"source", "resampled"}:
            raise DatasetFactoryError("Only source or resampled items can be resampled")
        if Path(item["stored_path"]).suffix.lower() != ".wav":
            raise DatasetFactoryError("A trusted decoder is required before this media can be resampled")
        expected = {"target_rate": target_rate, "mono": bool(mono), "method": "linear-v1"}
        with self._connect() as connection:
            children = connection.execute(
                "SELECT * FROM dataset_items WHERE parent_item_id = ? AND kind = 'resampled'",
                (item_id,),
            ).fetchall()
        for child in children:
            parsed = self._item(child)
            if parsed["metadata"].get("resample") == expected:
                candidate = self._stored_file(parsed["stored_path"])
                if candidate.is_file() and _sha256_file(candidate) == parsed["sha256"]:
                    return parsed
        samples, source_rate = _read_pcm_wav(self._stored_file(item["stored_path"]))
        if mono and samples.shape[1] > 1:
            samples = samples.mean(axis=1, keepdims=True, dtype=np.float32)
        output = _resample_linear(samples, source_rate, target_rate)
        destination = (
            self.media_root
            / item["dataset_id"]
            / "resampled"
            / f"{item_id}_{target_rate}_{'mono' if mono else 'source-channels'}.wav"
        )
        _write_pcm16_wav(destination, output, target_rate)
        sha256 = _sha256_file(destination)
        try:
            quality = analyze_pcm_quality(output, target_rate).as_dict()
        except DatasetQualityError as error:
            raise DatasetFactoryError(str(error)) from error
        inherited_annotations = _annotation_payload(item["metadata"])
        with self._connect() as connection:
            return self._insert_item(
                connection,
                dataset_id=item["dataset_id"],
                parent_item_id=item_id,
                kind="resampled",
                source_path=item["source_path"],
                stored_path=destination.relative_to(self.root).as_posix(),
                original_name=destination.name,
                sha256=sha256,
                byte_count=destination.stat().st_size,
                sample_rate=target_rate,
                channels=output.shape[1],
                frame_count=output.shape[0],
                duration_seconds=output.shape[0] / target_rate,
                start_frame=None,
                end_frame=None,
                pipeline_state="resampled",
                metadata={
                    "resample": expected,
                    "source_rate": source_rate,
                    "quality": quality,
                    "quality_source_sha256": sha256,
                    "annotations": inherited_annotations,
                },
            )

    @staticmethod
    def _speech_regions(
        samples: np.ndarray,
        sample_rate: int,
        config: EnergyVadConfig,
    ) -> list[SpeechRegion]:
        mono = samples.mean(axis=1, dtype=np.float32)
        frame_size = max(1, round(sample_rate * config.frame_ms / 1000))
        hop_size = max(1, round(sample_rate * config.hop_ms / 1000))
        if mono.size == 0:
            return []
        starts = np.arange(0, mono.size, hop_size, dtype=np.int64)
        dbfs = np.empty(starts.size, dtype=np.float64)
        for position, start in enumerate(starts):
            frame = mono[start : min(start + frame_size, mono.size)]
            rms = math.sqrt(float(np.mean(np.square(frame, dtype=np.float64)))) if frame.size else 0.0
            dbfs[position] = 20.0 * math.log10(max(rms, 1e-8))
        active = dbfs >= config.threshold_dbfs
        active_positions = np.flatnonzero(active)
        if active_positions.size == 0:
            return []
        max_gap_steps = max(0, math.ceil(config.min_silence_ms / config.hop_ms))
        groups: list[tuple[int, int]] = []
        begin = int(active_positions[0])
        previous = begin
        for current_value in active_positions[1:]:
            current = int(current_value)
            if current - previous - 1 > max_gap_steps:
                groups.append((begin, previous))
                begin = current
            previous = current
        groups.append((begin, previous))
        pad_frames = round(sample_rate * config.pad_ms / 1000)
        minimum_frames = round(sample_rate * config.min_speech_ms / 1000)
        maximum_frames = round(sample_rate * config.max_segment_ms / 1000)
        regions: list[SpeechRegion] = []
        for first_step, last_step in groups:
            raw_start = int(starts[first_step])
            raw_end = min(mono.size, int(starts[last_step]) + frame_size)
            if raw_end - raw_start < minimum_frames:
                continue
            padded_start = max(0, raw_start - pad_frames)
            padded_end = min(mono.size, raw_end + pad_frames)
            cursor = padded_start
            while cursor < padded_end:
                end = min(padded_end, cursor + maximum_frames)
                relevant_first = max(0, int(cursor / hop_size))
                relevant_last = min(dbfs.size, int(math.ceil(end / hop_size)))
                relevant = dbfs[relevant_first:relevant_last]
                regions.append(
                    SpeechRegion(
                        start_frame=cursor,
                        end_frame=end,
                        peak_dbfs=float(np.max(relevant)) if relevant.size else -160.0,
                        mean_dbfs=float(np.mean(relevant)) if relevant.size else -160.0,
                    )
                )
                cursor = end
        return regions

    def analyze_vad(
        self,
        item_id: str,
        *,
        config: EnergyVadConfig | None = None,
    ) -> dict[str, Any]:
        settings = (config or EnergyVadConfig()).validated()
        item = self.get_item(item_id)
        if item["kind"] not in {"source", "resampled"}:
            raise DatasetFactoryError("VAD requires a source or resampled WAV item")
        path = self._stored_file(item["stored_path"])
        if path.suffix.lower() != ".wav":
            raise DatasetFactoryError("A trusted decoder is required before VAD can run")
        samples, sample_rate = _read_pcm_wav(path)
        regions = self._speech_regions(samples, sample_rate, settings)
        state = "vad-analyzed" if regions else "no-speech"
        metadata = dict(item["metadata"])
        metadata["vad"] = {
            "algorithm": "frame-rms-v1",
            "config": asdict(settings),
            "region_count": len(regions),
        }
        with self._connect() as connection:
            connection.execute("DELETE FROM dataset_vad_regions WHERE item_id = ?", (item_id,))
            for position, region in enumerate(regions):
                connection.execute(
                    "INSERT INTO dataset_vad_regions(item_id, position, start_frame, end_frame, "
                    "peak_dbfs, mean_dbfs) VALUES(?, ?, ?, ?, ?, ?)",
                    (
                        item_id,
                        position,
                        region.start_frame,
                        region.end_frame,
                        region.peak_dbfs,
                        region.mean_dbfs,
                    ),
                )
            connection.execute(
                "UPDATE dataset_items SET pipeline_state = ?, metadata_json = ?, updated_at = ? "
                "WHERE id = ?",
                (state, _json_text(metadata), _utc_now(), item_id),
            )
        return {
            "item_id": item_id,
            "sample_rate": sample_rate,
            "algorithm": "frame-rms-v1",
            "config": asdict(settings),
            "speech_regions": [region.as_dict(sample_rate) for region in regions],
            "state": state,
        }

    def segment(
        self,
        item_id: str,
        *,
        config: EnergyVadConfig | None = None,
    ) -> list[dict[str, Any]]:
        settings = (config or EnergyVadConfig()).validated()
        item = self.get_item(item_id)
        analysis = self.analyze_vad(item_id, config=settings)
        if not analysis["speech_regions"]:
            return []
        config_json = json.dumps(asdict(settings), sort_keys=True, separators=(",", ":"))
        generation = hashlib.sha256(
            f"{item['sha256']}:{config_json}".encode("utf-8")
        ).hexdigest()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM dataset_items WHERE parent_item_id = ? AND kind = 'segment'",
                (item_id,),
            ).fetchall()
        existing = [self._item(row) for row in rows]
        matching = [value for value in existing if value["metadata"].get("generation") == generation]
        if matching and all(
            (path := self._stored_file(value["stored_path"])).is_file()
            and _sha256_file(path) == value["sha256"]
            for value in matching
        ):
            return matching
        samples, sample_rate = _read_pcm_wav(self._stored_file(item["stored_path"]))
        results: list[dict[str, Any]] = []
        for position, region_value in enumerate(analysis["speech_regions"]):
            start = int(region_value["start_frame"])
            end = int(region_value["end_frame"])
            segment_samples = samples[start:end]
            destination = (
                self.media_root
                / item["dataset_id"]
                / "segments"
                / item_id
                / f"{generation[:12]}_{position:05d}.wav"
            )
            _write_pcm16_wav(destination, segment_samples, sample_rate)
            sha256 = _sha256_file(destination)
            try:
                quality = analyze_pcm_quality(segment_samples, sample_rate).as_dict()
            except DatasetQualityError as error:
                raise DatasetFactoryError(str(error)) from error
            inherited_annotations = _annotation_payload(item["metadata"])
            if len(analysis["speech_regions"]) > 1:
                inherited_annotations["transcript"] = None
                inherited_annotations["language_diagnostic"] = None
                inherited_annotations["source"] = "inherited-requires-transcript"
            with self._connect() as connection:
                result = self._insert_item(
                    connection,
                    dataset_id=item["dataset_id"],
                    parent_item_id=item_id,
                    kind="segment",
                    source_path=item["source_path"],
                    stored_path=destination.relative_to(self.root).as_posix(),
                    original_name=destination.name,
                    sha256=sha256,
                    byte_count=destination.stat().st_size,
                    sample_rate=sample_rate,
                    channels=segment_samples.shape[1],
                    frame_count=segment_samples.shape[0],
                    duration_seconds=segment_samples.shape[0] / sample_rate,
                    start_frame=start,
                    end_frame=end,
                    pipeline_state="segmented",
                    metadata={
                        "generation": generation,
                        "position": position,
                        "vad_config": asdict(settings),
                        "peak_dbfs": region_value["peak_dbfs"],
                        "mean_dbfs": region_value["mean_dbfs"],
                        "quality": quality,
                        "quality_source_sha256": sha256,
                        "annotations": inherited_annotations,
                    },
                    rms_dbfs=region_value["mean_dbfs"],
                    peak_dbfs=region_value["peak_dbfs"],
                )
            results.append(result)
        metadata = dict(item["metadata"])
        metadata["last_segment_generation"] = generation
        metadata["last_segment_count"] = len(results)
        with self._connect() as connection:
            connection.execute(
                "UPDATE dataset_items SET pipeline_state = 'segmented', metadata_json = ?, "
                "updated_at = ? WHERE id = ?",
                (_json_text(metadata), _utc_now(), item_id),
            )
        return results

    def review(self, item_id: str, *, decision: str, note: str = "") -> dict[str, Any]:
        if decision not in REVIEW_STATES:
            raise DatasetFactoryError("Unsupported review decision")
        if not isinstance(note, str) or len(note) > 2_000:
            raise DatasetFactoryError("Review note is limited to 2000 characters")
        item = self.get_item(item_id)
        self._assert_mutable_dataset(str(item["dataset_id"]))
        if item["kind"] != "segment":
            raise DatasetFactoryError("Only segmented audio can enter dataset review")
        split_name = item["split_name"] if decision == "accepted" else None
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE dataset_items SET review_status = ?, review_note = ?, split_name = ?, "
                "updated_at = ? WHERE id = ?",
                (decision, note.strip(), split_name, now, item_id),
            )
            connection.execute(
                "INSERT INTO dataset_reviews(item_id, decision, note, created_at) "
                "VALUES(?, ?, ?, ?)",
                (item_id, decision, note.strip(), now),
            )
            row = connection.execute("SELECT * FROM dataset_items WHERE id = ?", (item_id,)).fetchone()
        assert row is not None
        result = self._item(row)
        dataset_id = str(result["dataset_id"])
        if self.review_summary(dataset_id)["review_complete"]:
            self.advance_project_stage(dataset_id, "style")
        return result

    def review_history(self, item_id: str) -> list[dict[str, Any]]:
        self.get_item(item_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT decision, note, created_at FROM dataset_reviews WHERE item_id = ? "
                "ORDER BY id",
                (item_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _production_group_assignment(ordered, original):
        """Meet minimum clip counts without splitting a source group.

        Capped-count dynamic programming has only 108 states. Prefer retaining
        the existing deterministic assignment, then moving fewer clips.
        """
        names = ("train", "validation", "test")
        minimums = (2, 5, 5)
        states = {(0, 0, 0): ((0, 0, 0), None)}
        for group_id, items in ordered:
            previous = original[group_id]
            following = {}
            for counts, (cost, node) in states.items():
                for name in (previous, *(n for n in names if n != previous)):
                    position = names.index(name)
                    next_counts = list(counts)
                    next_counts[position] = min(minimums[position], counts[position] + len(items))
                    next_counts = tuple(next_counts)
                    moved = name != previous
                    next_cost = (
                        cost[0] + int(moved),
                        cost[1] + len(items) * int(moved),
                        cost[2] + int(moved and previous != "train"),
                    )
                    best = following.get(next_counts)
                    if best is None or next_cost < best[0]:
                        following[next_counts] = (next_cost, (node, group_id, name))
            states = following
        winner = states.get(minimums)
        if winner is None:
            return None
        assignment = {}
        node = winner[1]
        while node is not None:
            node, group_id, name = node
            assignment[group_id] = name
        return assignment

    def assign_splits(
        self,
        dataset_id: str,
        *,
        train: float = 0.85,
        validation: float = 0.1,
        test: float = 0.05,
        seed: str = "aniflive-tts-v1.4",
    ) -> dict[str, Any]:
        dataset = _dataset_id(dataset_id)
        self._assert_mutable_dataset(dataset)
        ratios = {"train": float(train), "validation": float(validation), "test": float(test)}
        if any(not math.isfinite(value) or value < 0.0 for value in ratios.values()):
            raise DatasetFactoryError("Split ratios must be finite and non-negative")
        if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise DatasetFactoryError("Split ratios must sum to 1")
        accepted = self.list_items(dataset, kind="segment", review_status="accepted")
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in accepted:
            groups.setdefault(item["parent_item_id"] or item["id"], []).append(item)
        ordered = sorted(
            groups.items(),
            key=lambda entry: hashlib.sha256(f"{seed}:{entry[0]}".encode("utf-8")).digest(),
        )
        group_total = len(ordered)
        raw_counts = {name: group_total * ratio for name, ratio in ratios.items()}
        group_counts = {name: math.floor(value) for name, value in raw_counts.items()}
        remaining = group_total - sum(group_counts.values())
        priority = {"train": 0, "validation": 1, "test": 2}
        for name in sorted(
            ratios,
            key=lambda value: (-(raw_counts[value] - group_counts[value]), priority[value]),
        )[:remaining]:
            group_counts[name] += 1
        assignments: dict[str, str] = {}
        offset = 0
        for name in ("train", "validation", "test"):
            for _group_id, items in ordered[offset : offset + group_counts[name]]:
                for item in items:
                    assignments[item["id"]] = name
            offset += group_counts[name]
        minimum_adjustment = False
        if len(accepted) >= 40 and all(ratios[name] > 0 for name in ratios):
            counts = {name: sum(value == name for value in assignments.values()) for name in ratios}
            if any(counts[name] < minimum for name, minimum in (("train", 2), ("validation", 5), ("test", 5))):
                original_groups = {group_id: assignments[items[0]["id"]] for group_id, items in ordered}
                adjusted = self._production_group_assignment(ordered, original_groups)
                if adjusted is not None:
                    assignments = {
                        item["id"]: adjusted[group_id]
                        for group_id, items in ordered for item in items
                    }
                    group_counts = {name: sum(value == name for value in adjusted.values()) for name in ratios}
                    minimum_adjustment = True
        with self._connect() as connection:
            connection.execute(
                "UPDATE dataset_items SET split_name = NULL, updated_at = ? "
                "WHERE dataset_id = ? AND kind = 'segment'",
                (_utc_now(), dataset),
            )
            for item_id, split_name in assignments.items():
                connection.execute(
                    "UPDATE dataset_items SET split_name = ?, updated_at = ? WHERE id = ?",
                    (split_name, _utc_now(), item_id),
                )
        item_counts = {
            name: sum(value == name for value in assignments.values()) for name in SPLIT_NAMES
        }
        production_ready = True
        production_failures: list[str] = []
        if len(accepted) >= 40:
            minimums = {"train": 2, "validation": 5, "test": 5}
            for name, minimum in minimums.items():
                if item_counts[name] < minimum:
                    production_failures.append(f"{name}-requires-{minimum}")
            production_ready = not production_failures
        return {
            "dataset_id": dataset,
            "accepted_items": len(accepted),
            "production_minimum_adjustment": minimum_adjustment,
            "groups": group_total,
            "group_counts": group_counts,
            "item_counts": item_counts,
            "assignments": assignments,
            "seed": seed,
            "production_ready": production_ready,
            "production_failures": production_failures,
        }

    def manifest(self, dataset_id: str) -> dict[str, Any]:
        dataset = _dataset_id(dataset_id)
        accepted = self.list_items(dataset, kind="segment", review_status="accepted")
        records = []
        for item in accepted:
            if item["split_name"] is None:
                raise DatasetFactoryError("Every accepted item must be assigned before export")
            path = self._stored_file(item["stored_path"])
            if not path.is_file() or _sha256_file(path) != item["sha256"]:
                raise DatasetFactoryError("A dataset item failed integrity verification")
            records.append(
                {
                    "id": item["id"],
                    "path": item["stored_path"],
                    "sha256": item["sha256"],
                    "duration_seconds": item["duration_seconds"],
                    "sample_rate": item["sample_rate"],
                    "channels": item["channels"],
                    "split": item["split_name"],
                    "source_item_id": item["parent_item_id"],
                    "annotations": item["annotations"],
                    "annotation_verifications": item["verification"],
                    "review_complete": item["review_complete"],
                    "quality": item["quality"],
                    "lineage": item["metadata"].get("lineage"),
                    "acquisition": item["metadata"].get("acquisition"),
                }
            )
        state = self.project_state(dataset)
        return {
            "schema": "aniflive-dataset-manifest-v2",
            "dataset_id": dataset,
            "acquisition_mode": state["acquisition_mode"],
            "lifecycle_stage": state["lifecycle_stage"],
            "item_count": len(records),
            "duration_seconds": sum(float(record["duration_seconds"]) for record in records),
            "items": records,
        }
