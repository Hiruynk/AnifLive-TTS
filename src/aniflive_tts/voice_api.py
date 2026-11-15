"""Voice/Reference HTTP resources; audio is accepted as content, never server paths."""
from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable
import wave

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from .voice_library import VoiceLibrary
from .voices import VoiceError

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
REFERENCE_FIELDS = frozenset({"emotion_id", "emotion_label", "intensity", "reference_language",
                              "transcript", "transcript_source", "transcript_provider",
                              "transcript_confirmed", "is_default"})


def normalise_reference_audio(data: bytes) -> tuple[bytes, float]:
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise VoiceError("audio_size", "Audio must contain at most 25 MiB", 413)
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        demuxer = "wav"
    elif data[:4] == b"fLaC":
        demuxer = "flac"
    elif data[:4] == b"OggS":
        demuxer = "ogg"
    elif data[:4] == b"\x1a\x45\xdf\xa3":
        demuxer = "matroska"
    elif data[4:8] == b"ftyp":
        demuxer = "mov"
    elif data[:3] == b"ID3" or (len(data) > 1 and data[0] == 255 and data[1] & 224 == 224):
        demuxer = "mp3"
    else:
        raise VoiceError("unsupported_audio", "Supported audio: WAV, FLAC, Ogg, MP3, M4A and WebM")
    with tempfile.TemporaryDirectory(prefix="voice-audio-") as temp:
        source, destination = Path(temp) / "input", Path(temp) / "reference.wav"
        source.write_bytes(data)
        try:
            result = subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
                "-f", demuxer, "-i", str(source), "-map", "0:a:0", "-vn",
                "-t", "10.05", "-ac", "1", "-ar", "32000", "-c:a", "pcm_s16le",
                str(destination),
            ], capture_output=True, timeout=30)
        except subprocess.TimeoutExpired as error:
            raise VoiceError("audio_decode_timeout", "Audio decoding exceeded its time limit") from error
        if result.returncode != 0 or not destination.is_file():
            raise VoiceError("audio_decode_failed", "Audio could not be decoded")
        audio = destination.read_bytes()
    with wave.open(io.BytesIO(audio), "rb") as decoded:
        duration = decoded.getnframes() / decoded.getframerate()
    if not 3 <= duration <= 10:
        raise VoiceError("reference_duration", "Reference audio must contain 3 to 10 seconds of speech")
    return audio, duration


async def _upload(request: Request) -> tuple[bytes, dict[str, Any]]:
    size = 0
    async def bounded():
        nonlocal size
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES + 64 * 1024:
                raise MultiPartException("Upload exceeds its size limit")
            yield chunk
    parser = MultiPartParser(request.headers, bounded(), max_files=1, max_fields=1,
                             max_part_size=64 * 1024)
    try:
        form = await parser.parse()
    except MultiPartException as error:
        raise VoiceError("invalid_upload", "Upload must contain bounded audio and JSON metadata",
                         413 if size > MAX_UPLOAD_BYTES else 400) from error
    try:
        audio, metadata = form.get("audio"), form.get("metadata", "{}")
        if not isinstance(audio, UploadFile) or not isinstance(metadata, str):
            raise VoiceError("invalid_upload", "Use audio file and metadata JSON form fields")
        try:
            values = json.loads(metadata)
        except (ValueError, UnicodeError) as error:
            raise VoiceError("invalid_metadata", "metadata must contain valid JSON") from error
        if not isinstance(values, dict):
            raise VoiceError("invalid_metadata", "metadata must be a JSON object")
        return await audio.read(MAX_UPLOAD_BYTES + 1), values
    finally:
        await form.close()


def _fields(values: dict[str, Any], *, allow_id: bool = False) -> None:
    allowed = REFERENCE_FIELDS | ({"id"} if allow_id else set())
    if set(values) - allowed:
        raise VoiceError("invalid_metadata", "Unsupported Reference metadata field")
    if "is_default" in values and not isinstance(values["is_default"], bool):
        raise VoiceError("invalid_metadata", "is_default must be boolean")


def install_voice_routes(app: FastAPI, library: VoiceLibrary,
                         read_json: Callable[..., Any]) -> None:
    store = library.store

    @app.exception_handler(VoiceError)
    async def voice_error(_: Request, error: VoiceError):
        return JSONResponse({"code": error.code, "message": str(error)}, status_code=error.status_code)

    @app.post("/v1/voices", status_code=201)
    async def create_voice(request: Request):
        body = await read_json(request)
        if set(body) - {"id", "name"}:
            raise VoiceError("invalid_metadata", "Voice accepts id and name only")
        return store.create_voice(body.get("name"), voice_id=body.get("id"))

    @app.get("/v1/voices/{voice_id}")
    async def get_voice(voice_id: str):
        return store.get_voice(voice_id)

    @app.patch("/v1/voices/{voice_id}")
    async def update_voice(voice_id: str, request: Request):
        body = await read_json(request)
        if set(body) != {"name"}:
            raise VoiceError("invalid_metadata", "Voice update requires name")
        return store.update_voice(voice_id, name=body["name"])

    @app.delete("/v1/voices/{voice_id}", status_code=204)
    async def delete_voice(voice_id: str):
        store.delete_voice(voice_id)
        return Response(status_code=204)

    @app.get("/v1/voices/{voice_id}/references")
    async def list_references(voice_id: str):
        return {"object": "list", "data": store.list_references(voice_id)}

    @app.post("/v1/voices/{voice_id}/references", status_code=201)
    async def create_reference(voice_id: str, request: Request):
        data, values = await _upload(request)
        _fields(values, allow_id=True)
        audio, duration = await normalise(data)
        return store.create_reference(voice_id, audio, values, reference_id=values.get("id"),
                                      duration_seconds=duration, is_default=values.get("is_default", False))

    @app.get("/v1/voices/{voice_id}/references/{reference_id}")
    async def get_reference(voice_id: str, reference_id: str):
        return store.get_reference(voice_id, reference_id)

    @app.get("/v1/voices/{voice_id}/references/{reference_id}/audio")
    async def reference_audio(voice_id: str, reference_id: str):
        return FileResponse(store.audio_path(voice_id, reference_id), media_type="audio/wav")

    @app.patch("/v1/voices/{voice_id}/references/{reference_id}")
    async def update_reference(voice_id: str, reference_id: str, request: Request):
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            data, values = await _upload(request)
            _fields(values)
            audio, duration = await normalise(data)
            return store.replace_audio(voice_id, reference_id, audio, duration_seconds=duration,
                                       metadata=values)
        else:
            values = await read_json(request)
            _fields(values)
        return store.update_reference(voice_id, reference_id, values)

    @app.delete("/v1/voices/{voice_id}/references/{reference_id}", status_code=204)
    async def delete_reference(voice_id: str, reference_id: str):
        store.delete_reference(voice_id, reference_id)
        return Response(status_code=204)

    @app.post("/v1/voices/{voice_id}/references/{reference_id}/prepare")
    async def prepare_reference(voice_id: str, reference_id: str):
        ref = library.prepare(voice_id, reference_id)
        return JSONResponse(ref, status_code=200 if ref["prepare_state"] == "ready" else 202)

    async def normalise(data: bytes):
        from fastapi.concurrency import run_in_threadpool
        return await run_in_threadpool(normalise_reference_audio, data)
