"""Bounded per-engine reference conditioning, isolated by persistent Voice ID."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import logging
import threading
from typing import Any, Callable

from .expression import (
    ConditioningBundle, ConditioningPolicy, ExpressionCatalog,
    PreparedReference, PreparedReferenceBank, ReferenceDescriptor,
)
from .voices import VoiceError, VoiceLease, VoiceStore

LOGGER = logging.getLogger(__name__)


@dataclass
class VoiceBinding:
    voice_id: str
    bank: PreparedReferenceBank
    catalog: ExpressionCatalog
    lease: VoiceLease

    def conditioning(self, *, enabled: bool, profile: str | None, intensity: float,
                     language: str, policy: ConditioningPolicy) -> ConditioningBundle:
        if not enabled:
            return replace(self.bank.conditioning(reference_id=self.bank.identity_id,
                           policy=ConditioningPolicy.FULL_SWITCH),
                           identity_reference_id=self.bank.identity_id)
        try:
            selected = self.catalog.select(profile=profile or self.catalog.default_profile,
                                           intensity=intensity, language=language.removeprefix("all_"))
        except KeyError as error:
            raise VoiceError("reference_not_ready", "Requested reference or emotion is not ready", 409) from error
        return replace(self.bank.conditioning(reference_id=selected.id, policy=policy),
                       identity_reference_id=self.bank.identity_id)

    def close(self) -> None:
        self.lease.close()


class VoiceLibrary:
    def __init__(self, store: VoiceStore, *, maximum_prepared: int = 64):
        self.store = store
        self.maximum_prepared = maximum_prepared
        self._cache: OrderedDict[tuple[Any, ...], PreparedReference] = OrderedDict()
        self._pending: set[tuple[str, str, int]] = set()
        self._engine_epoch = 0
        self._lock = threading.RLock()
        self._executor: ThreadPoolExecutor | None = None
        self.prepare_callback: Callable[..., PreparedReference] | None = None
        self.engine_fingerprint: Callable[[], str] = lambda: "unloaded"
        self.prepare_on_demand = False

    def _key(self, ref: dict[str, Any]) -> tuple[Any, ...]:
        return (self.engine_fingerprint(), ref["voice_id"], ref["id"],
                ref["conditioning_revision"], ref["audio_sha256"],
                ref["transcript"], ref["reference_language"])

    def prepare(self, voice_id: str, reference_id: str) -> dict[str, Any]:
        ref = self.store.get_reference(voice_id, reference_id)
        if not ref["transcript"]:
            raise VoiceError("missing_transcript", "Provide a transcript before preparation", 409)
        if self.prepare_callback is None:
            raise VoiceError("engine_not_ready", "The shared inference engine is not ready", 503)
        task = (voice_id, reference_id, ref["conditioning_revision"])
        with self._lock:
            if self._key(ref) in self._cache:
                self.store.mark_preparation(*task, "ready")
                return self.store.get_reference(voice_id, reference_id)
            if task not in self._pending:
                self._pending.add(task)
                self.store.mark_preparation(*task, "queued")
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-reference")
                self._executor.submit(self._prepare, ref, task)
        return self.store.get_reference(voice_id, reference_id)

    def _prepare(self, ref: dict[str, Any], task: tuple[str, str, int]) -> PreparedReference | None:
        lease = None
        try:
            lease = self.store.acquire(ref["voice_id"])
            if not self.store.mark_preparation(*task, "preparing"):
                return
            cache_key = self._key(ref)
            with self._lock:
                engine_epoch = self._engine_epoch
            prepared = self.prepare_callback(
                reference_id=ref["id"], reference_wav=self.store.audio_path(ref["voice_id"], ref["id"]),
                reference_text=ref["transcript"], reference_language=ref["reference_language"],
            )
            if self._key(ref) != cache_key:
                raise VoiceError("engine_changed", "Engine changed during reference preparation", 409)
            with self._lock:
                if engine_epoch != self._engine_epoch:
                    raise VoiceError("engine_changed", "Engine changed during reference preparation", 409)
                self._cache[cache_key] = prepared
                self._cache.move_to_end(cache_key)
                while len(self._cache) > self.maximum_prepared:
                    evicted, _ = self._cache.popitem(last=False)
                    try:
                        self.store.mark_preparation(evicted[1], evicted[2], evicted[3], "unprepared")
                    except VoiceError as error:
                        if error.status_code != 404:
                            raise
                self.store.mark_preparation(*task, "ready")
                return prepared
        except Exception as error:
            code = error.code if isinstance(error, VoiceError) else "preparation_failed"
            LOGGER.exception("Reference preparation failed for voice/reference identifiers")
            try:
                self.store.mark_preparation(*task, "failed", code)
            except VoiceError as gone:
                if gone.status_code != 404:
                    raise
        finally:
            if lease is not None:
                lease.close()
            with self._lock:
                self._pending.discard(task)

    def bind(self, voice_id: str) -> VoiceBinding:
        lease, default, references = self.store.acquire_binding_snapshot(voice_id)
        try:
            if not default:
                raise VoiceError("default_reference_required", "Set a default reference for this voice", 409)
            bank = PreparedReferenceBank(identity_id=default)
            descriptors = []
            # Release the store lock before taking the cache lock: preparation
            # can acquire these locks in the opposite order.
            for ref, reference_path in references:
                with self._lock:
                    prepared = self._cache.get(self._key(ref))
                    if prepared is not None:
                        self._cache.move_to_end(self._key(ref))
                if self.prepare_on_demand and prepared is None and ref["transcript"]:
                    prepared = self._prepare(ref, (voice_id, ref["id"], ref["conditioning_revision"]))
                if prepared is None or (not self.prepare_on_demand and ref["prepare_state"] != "ready"):
                    continue
                bank.add(prepared)
                descriptors.append(ReferenceDescriptor(
                    id=ref["id"], emotion=ref["emotion_id"], intensity=ref["intensity"],
                    reference_audio=reference_path,
                    reference_text=ref["transcript"], reference_language=ref["reference_language"],
                    manual_verified=ref["transcript_confirmed"],
                    preferred_policy=ConditioningPolicy.IDENTITY_LOCK,
                ))
            if default not in bank.references:
                raise VoiceError("reference_not_ready", "The default reference is not prepared", 409)
            return VoiceBinding(voice_id, bank, ExpressionCatalog(default, tuple(descriptors)), lease)
        except BaseException:
            lease.close()
            raise

    def restore_default_references(self) -> None:
        """Restore registered default conditioning before the native server serves."""
        voices = [voice for voice in self.store.list_voices() if voice["default_reference_id"]]
        if len(voices) > self.maximum_prepared:
            raise VoiceError("reference_capacity", "Registered default voices exceed the prepared reference capacity", 503)
        for voice in voices:
            reference_id = voice["default_reference_id"]
            ref = self.store.get_reference(voice["id"], reference_id)
            if not ref["transcript"]:
                continue
            task = (voice["id"], reference_id, ref["conditioning_revision"])
            self._prepare(ref, task)
            restored = self.store.get_reference(voice["id"], reference_id)
            if restored["prepare_state"] != "ready":
                raise VoiceError("reference_restore_failed", "A registered default reference could not be restored", 503)

    def restore_registered_references(self) -> None:
        """Warm registered references up to capacity; prepare the rest on demand."""
        ordered = []
        for voice in self.store.list_voices():
            for ref in self.store.list_references(voice["id"]):
                if ref["transcript"]:
                    ordered.append((ref["id"] != voice["default_reference_id"], ref))
        ordered.sort(key=lambda item: item[0])
        for index, (_, ref) in enumerate(ordered):
            if index >= self.maximum_prepared:
                LOGGER.warning("Prepared-reference capacity reached; remaining references prepare on demand")
                break
            task = (ref["voice_id"], ref["id"], ref["conditioning_revision"])
            try:
                self._prepare(ref, task)
            except Exception:
                LOGGER.exception("Startup restore of a registered reference failed; it will prepare on demand")

    def invalidate_engine(self) -> None:
        with self._lock:
            self._engine_epoch += 1
            keys = list(self._cache)
            self._cache.clear()
        for key in keys:
            try:
                self.store.mark_preparation(key[1], key[2], key[3], "unprepared")
            except VoiceError as error:
                if error.status_code != 404:
                    raise

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        try:
            self.invalidate_engine()
        finally:
            self.store.close()
