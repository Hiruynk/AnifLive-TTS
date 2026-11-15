"""Registered defaults must regain conditioning, or fail startup explicitly."""
import pytest

from aniflive_tts.expression import PreparedReference
from aniflive_tts.voice_library import VoiceLibrary
from aniflive_tts.voices import VoiceError, VoiceStore


def registered(tmp_path):
    store = VoiceStore(tmp_path)
    voice = store.create_voice(name="Registered", voice_id="registered")
    store.create_reference(voice["id"], b"preserved-reference", {
        "reference_language": "ja", "transcript": "こんにちは。"},
        reference_id="reference", duration_seconds=4.0, is_default=True)
    return store


def test_restore_default_is_bound_to_new_engine_and_preserved_audio(tmp_path):
    store = registered(tmp_path)
    library = VoiceLibrary(store)
    observed = []
    marker = object()
    def prepare(**kwargs):
        observed.append(kwargs)
        return PreparedReference(marker, marker, None, [1], None, id=kwargs["reference_id"])
    library.prepare_callback = prepare
    library.engine_fingerprint = lambda: "new-native-engine"
    try:
        library.restore_default_references()
        assert store.get_voice("registered")["ready"]
        assert len(observed) == 1
        assert observed[0]["reference_wav"].read_bytes() == b"preserved-reference"
        with_binding = library.bind("registered")
        try:
            assert with_binding.bank.identity.prompt_semantic is marker
        finally:
            with_binding.close()
    finally:
        library.close()
    assert store.get_reference("registered", "reference")["prepare_state"] == "unprepared"


def test_restore_failed_native_preparation_does_not_claim_ready(tmp_path):
    store = registered(tmp_path)
    library = VoiceLibrary(store)
    def prepare(**kwargs):
        raise RuntimeError("actual native preparation failure")
    library.prepare_callback = prepare
    try:
        with pytest.raises(VoiceError) as error:
            library.restore_default_references()
        assert error.value.code == "reference_restore_failed"
        assert store.get_reference("registered", "reference")["prepare_state"] == "failed"
        assert not store.get_voice("registered")["ready"]
    finally:
        library.close()
