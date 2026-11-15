"""Persistent runtime voices and reference assets."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator, Mapping
import uuid

from .expression import validate_safe_identifier
from .errors import PackageValidationError

LANGUAGES = frozenset({"zh", "yue", "en", "ja", "ko"})
TRANSCRIPT_SOURCES = frozenset({"manual", "asr", "imported"})


class VoiceError(ValueError):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _identifier(value: str | None, prefix: str) -> str:
    try:
        return validate_safe_identifier(value or prefix + uuid.uuid4().hex, prefix + "id")
    except PackageValidationError as error:
        raise VoiceError("invalid_identifier", "Use a safe stable resource identifier") from error


def _name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise VoiceError("invalid_metadata", f"{field} must contain 1 to 256 characters")
    return value.strip()


class VoiceStore:
    """SQLite metadata plus server-named audio files; no conditioning tensors on disk."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.assets = self.root / "assets"
        self.assets.mkdir(exist_ok=True)
        self.database = self.root / "voices.sqlite3"
        self._lock = threading.RLock()
        self._users: Counter[str] = Counter()
        self._connection: sqlite3.Connection | None = None
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS voices (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    default_reference_id TEXT, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS voice_references (
                    voice_id TEXT NOT NULL REFERENCES voices(id) ON DELETE CASCADE,
                    id TEXT NOT NULL, metadata TEXT NOT NULL,
                    asset_name TEXT NOT NULL, audio_sha256 TEXT NOT NULL,
                    revision INTEGER NOT NULL, conditioning_revision INTEGER NOT NULL,
                    prepare_state TEXT NOT NULL, error_code TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY (voice_id, id)
                );
            """)
            # A prior process's GPU preparation cannot establish readiness here.
            db.execute("UPDATE voice_references SET prepare_state='unprepared', error_code=NULL "
                       "WHERE prepare_state IN ('ready','preparing','queued')")

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.open()
            assert self._connection is not None
            with self._connection as db:
                yield db

    def open(self) -> None:
        """Open one serialized connection, or retain the existing one."""
        with self._lock:
            if self._connection is None:
                db = sqlite3.connect(self.database, check_same_thread=False)
                db.row_factory = sqlite3.Row
                db.execute("PRAGMA foreign_keys=ON")
                self._connection = db

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    @staticmethod
    def _voice(db: sqlite3.Connection, voice_id: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM voices WHERE id=?", (voice_id,)).fetchone()
        if row is None:
            raise VoiceError("voice_not_found", "Voice does not exist", 404)
        return row

    @staticmethod
    def _reference(db: sqlite3.Connection, voice_id: str, reference_id: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM voice_references WHERE voice_id=? AND id=?",
                         (voice_id, reference_id)).fetchone()
        if row is None:
            raise VoiceError("reference_not_found", "Reference does not exist", 404)
        return row

    def _idle(self, voice_id: str) -> None:
        if self._users[voice_id]:
            raise VoiceError("voice_in_use", "Voice has an active request or session", 409)

    def create_voice(self, name: str, *, voice_id: str | None = None) -> dict[str, Any]:
        voice_id = _identifier(voice_id, "voice_")
        if voice_id == "default":
            raise VoiceError("reserved_voice_id", "default is reserved for legacy packages", 409)
        name = _name(name, "name")
        with self._db() as db:
            if db.execute("SELECT 1 FROM voices WHERE id=?", (voice_id,)).fetchone():
                raise VoiceError("voice_exists", "Voice ID already exists", 409)
            now = _now()
            db.execute("INSERT INTO voices VALUES (?, ?, NULL, ?, ?)",
                       (voice_id, name, now, now))
        return self.get_voice(voice_id)

    def list_voices(self) -> list[dict[str, Any]]:
        with self._db() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM voices ORDER BY created_at,id")]
        return [self.get_voice(i) for i in ids]

    def get_voice(self, voice_id: str) -> dict[str, Any]:
        with self._db() as db:
            row = dict(self._voice(db, voice_id))
            refs = db.execute("SELECT id,prepare_state,metadata FROM voice_references WHERE voice_id=?",
                              (voice_id,)).fetchall()
        default = next((json.loads(r["metadata"]) for r in refs if r["id"] == row["default_reference_id"]), None)
        row.update(object="voice", source="library", reference_count=len(refs),
                   reference_language=default["reference_language"] if default else "",
                   reference_configured=bool(default and default["transcript"]),
                   ready=any(r[0] == row["default_reference_id"] and r[1] == "ready" for r in refs))
        return row

    def update_voice(self, voice_id: str, *, name: str) -> dict[str, Any]:
        with self._db() as db:
            self._voice(db, voice_id)
            db.execute("UPDATE voices SET name=?,updated_at=? WHERE id=?",
                       (_name(name, "name"), _now(), voice_id))
        return self.get_voice(voice_id)

    def delete_voice(self, voice_id: str) -> None:
        with self._db() as db:
            self._voice(db, voice_id)
            self._idle(voice_id)
            assets = [r[0] for r in db.execute(
                "SELECT asset_name FROM voice_references WHERE voice_id=?", (voice_id,))]
            db.execute("DELETE FROM voices WHERE id=?", (voice_id,))
        for name in assets:
            (self.assets / name).unlink(missing_ok=True)

    @staticmethod
    def reference_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
        language = values.get("reference_language", "zh")
        if language not in LANGUAGES:
            raise VoiceError("invalid_language", "Unsupported reference language")
        transcript = values.get("transcript", "")
        if not isinstance(transcript, str) or len(transcript) > 4000:
            raise VoiceError("invalid_transcript", "Transcript must be text up to 4000 characters")
        source = values.get("transcript_source", "manual")
        if source not in TRANSCRIPT_SOURCES:
            raise VoiceError("invalid_transcript_source", "Unknown transcript source")
        confirmed = values.get("transcript_confirmed", False)
        if not isinstance(confirmed, bool):
            raise VoiceError("invalid_metadata", "transcript_confirmed must be boolean")
        intensity = values.get("intensity", 0.5)
        if isinstance(intensity, bool) or not isinstance(intensity, (float, int)) or not 0 <= intensity <= 1:
            raise VoiceError("invalid_intensity", "Intensity must be between zero and one")
        emotion = _name(values.get("emotion_label", "Neutral"), "emotion_label")
        emotion_id = values.get("emotion_id") or "emotion_" + hashlib.sha256(emotion.encode()).hexdigest()[:16]
        emotion_id = _identifier(emotion_id, "emotion_")
        provider = values.get("transcript_provider")
        if provider is not None:
            provider = _name(provider, "transcript_provider")
        return {"reference_language": language, "transcript": transcript.strip(),
                "transcript_source": source, "transcript_confirmed": confirmed,
                "transcript_provider": provider, "emotion_id": emotion_id,
                "emotion_label": emotion, "intensity": float(intensity)}

    def create_reference(self, voice_id: str, audio_wav: bytes, metadata: Mapping[str, Any],
                         *, reference_id: str | None = None,
                         duration_seconds: float, is_default: bool = False) -> dict[str, Any]:
        reference_id = _identifier(reference_id, "ref_")
        values = self.reference_metadata(metadata)
        asset_name = uuid.uuid4().hex + ".wav"
        with self._db() as db:
            voice = self._voice(db, voice_id)
            self._idle(voice_id)
            if db.execute("SELECT 1 FROM voice_references WHERE voice_id=? AND id=?",
                          (voice_id, reference_id)).fetchone():
                raise VoiceError("reference_exists", "Reference ID already exists", 409)
            values["duration_seconds"] = duration_seconds
            path = self.assets / asset_name
            with path.open("xb") as out:
                out.write(audio_wav)
            now = _now()
            try:
                db.execute("INSERT INTO voice_references VALUES (?,?,?,?,?,1,1,?,NULL,?,?)",
                           (voice_id, reference_id, json.dumps(values, ensure_ascii=False),
                            asset_name, hashlib.sha256(audio_wav).hexdigest(),
                            "unprepared" if values["transcript"] else "needs_transcript", now, now))
            except sqlite3.Error:
                path.unlink(missing_ok=True)
                raise
            if is_default or voice["default_reference_id"] is None:
                db.execute("UPDATE voices SET default_reference_id=?,updated_at=? WHERE id=?",
                           (reference_id, now, voice_id))
        return self.get_reference(voice_id, reference_id)

    def _public_reference(self, row: sqlite3.Row, default_id: str | None) -> dict[str, Any]:
        result = json.loads(row["metadata"])
        result.update({k:row[k] for k in ["id", "voice_id", "audio_sha256", "revision",
                      "conditioning_revision", "prepare_state", "error_code", "created_at", "updated_at"]})
        result.update(object="voice.reference", is_default=row["id"] == default_id)
        return result

    def list_references(self, voice_id: str) -> list[dict[str, Any]]:
        with self._db() as db:
            voice = self._voice(db, voice_id)
            return [self._public_reference(r, voice["default_reference_id"]) for r in db.execute(
                "SELECT * FROM voice_references WHERE voice_id=? ORDER BY created_at,id", (voice_id,))]

    def get_reference(self, voice_id: str, reference_id: str) -> dict[str, Any]:
        with self._db() as db:
            voice = self._voice(db, voice_id)
            return self._public_reference(self._reference(db, voice_id, reference_id),
                                          voice["default_reference_id"])

    def update_reference(self, voice_id: str, reference_id: str,
                         patch: Mapping[str, Any]) -> dict[str, Any]:
        with self._db() as db:
            self._voice(db, voice_id)
            row = self._reference(db, voice_id, reference_id)
            old = json.loads(row["metadata"])
            merged = {**old, **patch}
            if "transcript" in patch and patch["transcript"] != old["transcript"] and "transcript_confirmed" not in patch:
                merged["transcript_confirmed"] = False
            changed = self.reference_metadata(merged)
            changed["duration_seconds"] = old["duration_seconds"]
            needs_prepare = any(changed[k] != old[k] for k in ["transcript", "reference_language"])
            if needs_prepare or "is_default" in patch:
                self._idle(voice_id)
            state = ("unprepared" if changed["transcript"] else "needs_transcript") if needs_prepare else row["prepare_state"]
            now = _now()
            db.execute("UPDATE voice_references SET metadata=?,revision=revision+1,"
                       "conditioning_revision=conditioning_revision+?,prepare_state=?,error_code=?,updated_at=? "
                       "WHERE voice_id=? AND id=?",
                       (json.dumps(changed, ensure_ascii=False), int(needs_prepare), state,
                        None if needs_prepare else row["error_code"], now, voice_id, reference_id))
            if "is_default" in patch:
                current_default = self._voice(db, voice_id)["default_reference_id"]
                if patch["is_default"] or current_default == reference_id:
                    db.execute("UPDATE voices SET default_reference_id=?,updated_at=? WHERE id=?",
                               (reference_id if patch["is_default"] else None, now, voice_id))
        return self.get_reference(voice_id, reference_id)

    def replace_audio(self, voice_id: str, reference_id: str, audio_wav: bytes,
                      *, duration_seconds: float,
                      metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        asset_name = uuid.uuid4().hex + ".wav"
        with self._db() as db:
            self._voice(db, voice_id)
            self._idle(voice_id)
            row = self._reference(db, voice_id, reference_id)
            patch = metadata or {}
            values = self.reference_metadata({**json.loads(row["metadata"]), **patch})
            values["duration_seconds"] = duration_seconds
            if "transcript_confirmed" not in patch:
                values["transcript_confirmed"] = False
            path = self.assets / asset_name
            with path.open("xb") as out:
                out.write(audio_wav)
            try:
                db.execute("UPDATE voice_references SET asset_name=?,audio_sha256=?,metadata=?,"
                           "revision=revision+1,conditioning_revision=conditioning_revision+1,"
                           "prepare_state=?,error_code=NULL,updated_at=? WHERE voice_id=? AND id=?",
                           (asset_name, hashlib.sha256(audio_wav).hexdigest(), json.dumps(values),
                            "unprepared" if values["transcript"] else "needs_transcript",
                            _now(), voice_id, reference_id))
            except sqlite3.Error:
                path.unlink(missing_ok=True)
                raise
            if "is_default" in patch:
                default = self._voice(db, voice_id)["default_reference_id"]
                if patch["is_default"] or default == reference_id:
                    db.execute("UPDATE voices SET default_reference_id=?,updated_at=? WHERE id=?",
                               (reference_id if patch["is_default"] else None, _now(), voice_id))
        (self.assets / row["asset_name"]).unlink(missing_ok=True)
        return self.get_reference(voice_id, reference_id)

    def delete_reference(self, voice_id: str, reference_id: str) -> None:
        with self._db() as db:
            voice = self._voice(db, voice_id)
            self._idle(voice_id)
            row = self._reference(db, voice_id, reference_id)
            if voice["default_reference_id"] == reference_id:
                db.execute("UPDATE voices SET default_reference_id=NULL,updated_at=? WHERE id=?",
                           (_now(), voice_id))
            db.execute("DELETE FROM voice_references WHERE voice_id=? AND id=?", (voice_id, reference_id))
        (self.assets / row["asset_name"]).unlink(missing_ok=True)

    def audio_path(self, voice_id: str, reference_id: str) -> Path:
        with self._db() as db:
            return self.assets / self._reference(db, voice_id, reference_id)["asset_name"]

    def mark_preparation(self, voice_id: str, reference_id: str, revision: int,
                         state: str, error_code: str | None = None) -> bool:
        if state not in {"unprepared", "queued", "preparing", "ready", "failed"}:
            raise ValueError("Invalid preparation state")
        with self._db() as db:
            row = self._reference(db, voice_id, reference_id)
            if row["conditioning_revision"] != revision:
                return False
            if not json.loads(row["metadata"])["transcript"]:
                raise VoiceError("missing_transcript", "Reference requires a transcript")
            db.execute("UPDATE voice_references SET prepare_state=?,error_code=?,updated_at=? "
                       "WHERE voice_id=? AND id=?", (state, error_code, _now(), voice_id, reference_id))
            return True

    def acquire(self, voice_id: str) -> "VoiceLease":
        with self._db() as db:
            self._voice(db, voice_id)
            self._users[voice_id] += 1
        return VoiceLease(self, voice_id)

    def acquire_binding_snapshot(self, voice_id: str) -> tuple["VoiceLease", str | None, list[tuple[dict[str, Any], Path]]]:
        """Read a consistent voice/reference snapshot while acquiring its lease."""
        with self._lock:
            with self._db() as db:
                db.execute("BEGIN")
                default = self._voice(db, voice_id)["default_reference_id"]
                rows = db.execute(
                    "SELECT * FROM voice_references WHERE voice_id=? ORDER BY created_at,id", (voice_id,)
                ).fetchall()
                references = [(self._public_reference(row, default), self.assets / row["asset_name"])
                              for row in rows]
            self._users[voice_id] += 1
            return VoiceLease(self, voice_id), default, references


class VoiceLease:
    def __init__(self, store: VoiceStore, voice_id: str):
        self.store, self.voice_id = store, voice_id
        self.closed = False

    def close(self) -> None:
        with self.store._lock:
            if not self.closed:
                self.store._users[self.voice_id] -= 1
                self.closed = True
