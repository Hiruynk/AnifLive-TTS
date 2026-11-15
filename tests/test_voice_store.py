from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from aniflive_tts.voices import VoiceError, VoiceStore


def test_portable_ids_assets_and_restart_readiness(tmp_path: Path):
    store = VoiceStore(tmp_path)
    store.create_voice("角色甲", voice_id="chr_example")
    ref = store.create_reference("chr_example", b"reference-audio", {},
                                 reference_id="ref_example", duration_seconds=4)
    assert ref["prepare_state"] == "needs_transcript"
    ref = store.update_reference("chr_example", "ref_example", {
        "transcript": "你好。", "transcript_source": "asr", "transcript_provider": "SenseVoiceSmall",
    })
    assert ref["transcript_source"] == "asr" and not ref["transcript_confirmed"]
    store.mark_preparation("chr_example", "ref_example", ref["conditioning_revision"], "ready")
    restored = VoiceStore(tmp_path)
    assert restored.get_voice("chr_example")["default_reference_id"] == "ref_example"
    assert restored.audio_path("chr_example", "ref_example").read_bytes() == b"reference-audio"
    assert restored.get_reference("chr_example", "ref_example")["prepare_state"] == "unprepared"


def test_metadata_cache_revision_and_active_voice_protection(tmp_path: Path):
    store = VoiceStore(tmp_path)
    store.create_voice("A", voice_id="chr_a")
    ref = store.create_reference("chr_a", b"audio", {"transcript": "Hello", "reference_language": "en"},
                                 reference_id="ref_a", duration_seconds=4)
    lease = store.acquire("chr_a")
    renamed = store.update_reference("chr_a", "ref_a", {"emotion_label": "開心 / 嬉しい"})
    assert renamed["conditioning_revision"] == ref["conditioning_revision"]
    with pytest.raises(VoiceError, match="active request"):
        store.update_reference("chr_a", "ref_a", {"transcript": "Changed"})
    with pytest.raises(VoiceError, match="active request"):
        store.delete_voice("chr_a")
    lease.close()
    lease.close()
    changed = store.update_reference("chr_a", "ref_a", {"transcript": "Changed"})
    assert changed["conditioning_revision"] == ref["conditioning_revision"] + 1
    assert not store.mark_preparation("chr_a", "ref_a", ref["conditioning_revision"], "ready")
    store.delete_voice("chr_a")
    assert store.list_voices() == []


def test_persistent_connection_is_serialized_across_threads_and_reopens(tmp_path: Path):
    store = VoiceStore(tmp_path)
    store.create_voice("A", voice_id="chr_a")
    connection = store._connection
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: store.get_voice("chr_a")["id"], range(16))) == ["chr_a"] * 16
    assert store._connection is connection
    store.close()
    assert store._connection is None
    assert store.get_voice("chr_a")["id"] == "chr_a"
    assert store._connection is not connection
    store.close()


def test_binding_snapshot_keeps_multiple_references_and_lease(tmp_path: Path):
    from aniflive_tts.expression import ConditioningPolicy, PreparedReference
    from aniflive_tts.voice_library import VoiceLibrary

    store = VoiceStore(tmp_path)
    store.create_voice("A", voice_id="chr_a")
    library = VoiceLibrary(store)
    for identifier, intensity in (("neutral", 0.0), ("happy", 0.8)):
        ref = store.create_reference("chr_a", identifier.encode(), {
            "transcript": "Hello", "reference_language": "en",
            "emotion_id": identifier, "emotion_label": identifier, "intensity": intensity,
        }, reference_id=identifier, duration_seconds=4)
        store.mark_preparation("chr_a", identifier, ref["conditioning_revision"], "ready")
        ref = store.get_reference("chr_a", identifier)
        library._cache[library._key(ref)] = PreparedReference(None, None, None, [], None, id=identifier)
    binding = library.bind("chr_a")
    try:
        assert set(binding.bank.references) == {"neutral", "happy"}
        assert binding.conditioning(enabled=True, profile="happy", intensity=0.8,
                                    language="en", policy=ConditioningPolicy.FULL_SWITCH).semantic.id == "happy"
        with pytest.raises(VoiceError, match="active request"):
            store.delete_voice("chr_a")
    finally:
        binding.close()
        library.close()
    assert store._connection is None
