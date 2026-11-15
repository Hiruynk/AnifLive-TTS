import io
import json
import math
import struct
import wave

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aniflive_tts.voice_api import install_voice_routes
from aniflive_tts.voice_library import VoiceLibrary
from aniflive_tts.voices import VoiceStore


def audio(seconds=4):
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"".join(struct.pack("<h", round(3000 * math.sin(i / 8)))
                                 for i in range(16000 * seconds)))
    return out.getvalue()


def test_reference_upload_manual_transcript_and_portable_identity(tmp_path):
    app = FastAPI()
    library = VoiceLibrary(VoiceStore(tmp_path))
    async def read_json(request):
        return await request.json()
    install_voice_routes(app, library, read_json)
    with TestClient(app) as client:
        assert client.post("/v1/voices", json={"id": "chr_portable", "name": "角色"}).status_code == 201
        path = "/v1/voices/chr_portable/references"
        response = client.post(path, files={"audio": ("reference.wav", audio(), "audio/wav")},
                               data={"metadata": json.dumps({"id": "ref_portable", "emotion_label": "開心"})})
        assert response.status_code == 201, response.text
        ref = response.json()
        assert ref["id"] == "ref_portable" and ref["prepare_state"] == "needs_transcript"
        assert client.post(path + "/ref_portable/prepare").status_code == 409
        response = client.patch(path + "/ref_portable", json={
            "transcript": "Hello.", "reference_language": "en", "transcript_source": "manual",
            "transcript_confirmed": True,
        })
        assert response.status_code == 200
        assert response.json()["prepare_state"] == "unprepared"
        assert client.get(path + "/ref_portable/audio").content[:4] == b"RIFF"
        assert client.post(path + "/ref_portable/prepare").status_code == 503
    library.close()


def test_audio_duration_and_metadata_fail_explicitly(tmp_path):
    app = FastAPI()
    library = VoiceLibrary(VoiceStore(tmp_path))
    async def read_json(request):
        return await request.json()
    install_voice_routes(app, library, read_json)
    with TestClient(app) as client:
        client.post("/v1/voices", json={"id": "chr_a", "name": "A"})
        response = client.post("/v1/voices/chr_a/references",
            files={"audio": ("reference.wav", audio(11), "audio/wav")}, data={"metadata": "{}"})
        assert response.status_code == 400
        assert response.json()["code"] == "reference_duration"
        response = client.post("/v1/voices", json={"id": "../escape", "name": "A"})
        assert response.status_code == 400
    library.close()


def test_invalid_replacement_metadata_does_not_change_reference(tmp_path):
    app = FastAPI()
    library = VoiceLibrary(VoiceStore(tmp_path))
    async def read_json(request):
        return await request.json()
    install_voice_routes(app, library, read_json)
    with TestClient(app) as client:
        client.post("/v1/voices", json={"id": "chr_a", "name": "A"})
        collection = "/v1/voices/chr_a/references"
        original = client.post(collection,
            files={"audio": ("original.wav", audio(), "audio/wav")},
            data={"metadata": json.dumps({"id": "neutral", "transcript": "Hello.",
                                         "reference_language": "en"})}).json()
        path = collection + "/neutral"
        before_audio = client.get(path + "/audio").content
        rejected = client.patch(path,
            files={"audio": ("replacement.wav", audio(5), "audio/wav")},
            data={"metadata": json.dumps({"reference_language": "invalid"})})
        assert rejected.status_code == 400
        assert client.get(path).json() == original
        assert client.get(path + "/audio").content == before_audio
        assert len(list(library.store.assets.iterdir())) == 1
        replaced = client.patch(path,
            files={"audio": ("replacement.wav", audio(5), "audio/wav")},
            data={"metadata": json.dumps({"emotion_label": "平靜", "transcript": "Goodbye.",
                                         "transcript_confirmed": True})})
        assert replaced.status_code == 200
        assert replaced.json()["revision"] == original["revision"] + 1
        assert replaced.json()["emotion_label"] == "平靜"
        assert replaced.json()["transcript_confirmed"] is True
        assert replaced.json()["audio_sha256"] != original["audio_sha256"]
    library.close()
