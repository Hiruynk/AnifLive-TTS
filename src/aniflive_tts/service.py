"""TensorRT-only API implementation for one preloaded AnifLive-TTS model package.

The file deliberately contains no checkpoint/ONNX fallback.  The container
entrypoint is responsible for validating and building this computer's engines
before Uvicorn starts; this API then loads only those serialized TensorRT
engines from ``ANIFLIVE_TTS_ENGINE_DIR``.

Two request formats are provided:

* ``GET /`` and ``POST /`` are compatible with the common GPT-SoVITS API
  parameter names.
* ``POST /v1/audio/speech`` is a small OpenAI-style compatibility endpoint.

Complete responses are ordinary PCM16 mono WAV files.  The optional low-latency
mode returns standards-compliant raw PCM16 chunks instead of pretending an
unknown-length stream is a valid RIFF/WAV file.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import io
import json
import logging
import math
import os
import queue
import re
import secrets
import sys
import threading
import time
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

from aniflive_tts.runtime_control import WarmRetentionController


LOGGER = logging.getLogger("aniflive_tts.service")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

SERVICE_NAME = "AnifLive-TTS"
SERVICE_VERSION = "1.2.0"
MODEL_ID = os.environ.get("ANIFLIVE_TTS_MODEL_ID", "unconfigured")
VOICE_ID = os.environ.get("ANIFLIVE_TTS_VOICE_PROFILE", "default")
REFERENCE_TEXT = os.environ.get("ANIFLIVE_TTS_REFERENCE_TEXT", "")
REFERENCE_LANGUAGE = os.environ.get("ANIFLIVE_TTS_REFERENCE_LANGUAGE", "ja")
REQUIRED_ENGINES = (
    "ssl",
    "bert",
    "vq_encoder",
    "gpt_encoder",
    "gpt_step",
    "sovits",
    "spectrogram",
    "sv_embedding",
    "sovits_stream",
)
DEFAULTS = {
    "top_k": 15,
    "top_p": 1.0,
    "temperature": 1.0,
    "speed": 1.0,
    "pause_length": 0.440,
    "noise_scale": 0.5,
    "seed": -1,
}
MAX_TEXT_CHARS = 1000
MAX_JSON_BODY_BYTES = 64 * 1024
SAFE_INFERENCE_BOUNDARIES = frozenset(",.;?!、，。？！；")
ALLOWED_CUT_PUNCTUATION = SAFE_INFERENCE_BOUNDARIES
BOUNDARY_CLOSERS = frozenset("\"'”’」』）》】〕〉")
STREAM_CALIBRATION_TEXT = {
    "zh": "你好，今天天氣很好。",
    "yue": "你好，今日天氣好好。",
    "en": "Hello, the weather is nice today.",
    "ja": "今日はいい天気ですね。",
    "ko": "안녕하세요, 오늘 날씨가 좋네요.",
}
NON_TERMINAL_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e"}
)
# A TensorRT profile bounds the length of one model invocation, not the total
# HTTP request.  Keep every internal invocation comfortably inside the profile
# selected by the per-GPU builder.  Long requests are split at sentence
# boundaries (or, as a final safeguard, at a character boundary).
PROFILE_SEGMENT_CHAR_LIMITS = {
    "small": 32,
    # The fitted profile has a 100-phoneme text maximum and a 250-token
    # semantic maximum. Thirty-two characters avoids unnecessarily resetting
    # prosody inside ordinary punctuation-free clauses. The runtime still
    # validates the actual phoneme count before every TensorRT enqueue.
    "fitted": 32,
    "medium": 72,
    "large": 160,
}
DEFAULT_SAFE_CUT_PUNCTUATION = ",.;?!、，。？！；"
STREAM_RECOMMENDED_PREBUFFER_MS = 32
STREAM_LONG_RECOMMENDED_PREBUFFER_MS = 64
STREAM_LONG_SEGMENT_CHAR_THRESHOLD = 16
MIN_NATURAL_SEGMENT_CONTENT_CHARS = 8

LANGUAGE_MAP = {
    "中文": "all_zh",
    "粤语": "all_yue",
    "英文": "en",
    "日文": "all_ja",
    "韩文": "all_ko",
    "中英混合": "zh",
    "粤英混合": "yue",
    "日英混合": "ja",
    "韩英混合": "ko",
    "多语种混合": "auto",
    "多语种混合(粤语)": "auto_yue",
    "all_zh": "all_zh",
    "all_yue": "all_yue",
    "all_ja": "all_ja",
    "all_ko": "all_ko",
    "zh": "zh",
    "yue": "yue",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "auto": "auto",
    "auto_yue": "auto_yue",
}
CANONICAL_LANGUAGE_CODES = frozenset({"zh", "yue", "en", "ja", "ko"})


class RequestError(ValueError):
    """A request can be rejected without treating the TensorRT runtime as bad."""

    status_code = 400


class RequestBodyTooLarge(RequestError):
    """The JSON request exceeds the public API's fixed memory budget."""

    status_code = 413


class ServiceBusy(RequestError):
    """The single active voice is already serving another request."""

    status_code = 429


class ModelSwitchConflict(RequestError):
    """The active voice cannot be replaced while inference is running."""

    status_code = 409


class TensorRTRuntimeError(RuntimeError):
    """The only supported backend is unavailable or produced invalid output."""


@dataclass(frozen=True)
class RuntimeSettings:
    """Container paths.  All defaults match the Docker deployment layout."""

    source_dir: Path
    engine_dir: Path
    bert_path: Path
    reference_wav: Path

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        # TRT_DIR/BERT_PATH make the script convenient to run with the source
        # project's conventions as well as the deployment-specific variables.
        source_dir = os.environ.get("ANIFLIVE_TTS_SOURCE_DIR", "/app/minimal_inference")
        engine_dir = (
            os.environ.get("ANIFLIVE_TTS_ENGINE_DIR")
            or os.environ.get("TRT_DIR")
            or "/data/models/active/engines/current"
        )
        bert_path = (
            os.environ.get("ANIFLIVE_TTS_BERT_PATH")
            or os.environ.get("BERT_PATH")
            or "/data/shared/chinese-roberta-wwm-ext-large"
        )
        reference_wav = os.environ.get(
            "ANIFLIVE_TTS_REFERENCE_WAV", "/data/models/active/voices/default/reference.wav"
        )
        return cls(
            source_dir=Path(source_dir).expanduser().resolve(),
            engine_dir=Path(engine_dir).expanduser().resolve(),
            bert_path=Path(bert_path).expanduser().resolve(),
            reference_wav=Path(reference_wav).expanduser().resolve(),
        )


@dataclass(frozen=True)
class SynthesisOptions:
    text: str
    text_language: str
    top_k: int
    top_p: float
    temperature: float
    speed: float
    pause_length: float
    noise_scale: float
    cut_punc: str
    seed: int


@dataclass(frozen=True)
class SynthesisResult:
    wav: bytes
    sample_rate: int
    output_samples: int
    segments: int
    elapsed_seconds: float
    profile: dict[str, float | int]


def _json_response(message: str, status_code: int = 400) -> JSONResponse:
    headers = {"Retry-After": "1"} if status_code == 429 else None
    return JSONResponse(
        {"code": status_code, "message": message},
        status_code=status_code,
        headers=headers,
    )


def _normalise_language(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestError(f"Missing required parameter: {field}")
    key = value.strip()
    key = key.lower() if key.isascii() else key
    try:
        return LANGUAGE_MAP[key]
    except KeyError as error:
        supported = "zh, yue, en, ja, ko, auto, auto_yue"
        raise RequestError(
            f"Unsupported {field}={value!r}; supported language codes: {supported}"
        ) from error


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RequestError(f"{field} must be a string")
    value = value.strip()
    return value or None


def _canonical_language(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestError(f"Missing required parameter: {field}")
    language = value.strip().lower()
    if language not in CANONICAL_LANGUAGE_CODES:
        supported = ", ".join(sorted(CANONICAL_LANGUAGE_CODES))
        raise RequestError(
            f"Unsupported {field}={value!r}; canonical language codes: {supported}"
        )
    return language


def _number(
    value: Any,
    field: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise RequestError(f"{field} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RequestError(f"{field} must be a number") from error
    if not math.isfinite(result):
        raise RequestError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise RequestError(f"{field} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise RequestError(f"{field} must be at most {maximum}")
    return result


def _integer(value: Any, field: str, default: int, *, minimum: int = 1) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise RequestError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise RequestError(f"{field} must be an integer") from error
    # ``int(1.5)`` must not quietly alter a sampling parameter.
    if isinstance(value, float) and not value.is_integer():
        raise RequestError(f"{field} must be an integer")
    if isinstance(value, str) and value.strip() != str(result):
        raise RequestError(f"{field} must be an integer")
    if result < minimum:
        raise RequestError(f"{field} must be at least {minimum}")
    return result


def _boolean(value: Any, field: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise RequestError(f"{field} must be a boolean")


def _is_safe_inference_boundary(text: str, index: int, selected: set[str]) -> bool:
    character = text[index]
    if character not in selected:
        return False
    previous = text[index - 1] if index else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    if character in {",", "，"} and previous.isdigit() and following.isdigit():
        return False
    if character != ".":
        return True
    if previous == "." or following == ".":
        return False
    if previous.isdigit() and following.isdigit():
        return False
    token_match = re.search(r"([^\s]+)$", text[:index])
    token = token_match.group(1) if token_match else ""
    lowered = token.lower().rstrip(".")
    if "://" in token or "@" in token or token.lower().startswith("www"):
        return False
    if lowered in NON_TERMINAL_ABBREVIATIONS:
        return False
    if len(token) == 1 and token.isalpha() and following:
        return False
    return True


def _cut_segments(text: str, cut_punc: str) -> list[str]:
    """Split only at speech-safe punctuation, preserving punctuation and quotes."""

    selected = set(DEFAULT_SAFE_CUT_PUNCTUATION)
    selected.update(character for character in cut_punc if character in ALLOWED_CUT_PUNCTUATION)
    segments: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        paragraph_break = text[index] == "\n" and index + 1 < len(text) and text[index + 1] == "\n"
        if not paragraph_break and not _is_safe_inference_boundary(text, index, selected):
            index += 1
            continue
        end = index + (2 if paragraph_break else 1)
        if paragraph_break:
            while end < len(text) and text[end] == "\n":
                end += 1
        else:
            while end < len(text) and _is_safe_inference_boundary(text, end, selected):
                end += 1
            while end < len(text) and text[end] in BOUNDARY_CLOSERS:
                end += 1
        segment = text[start:end].lstrip(" \t")
        if segment.strip():
            segments.append(segment)
        start = end
        index = end
    tail = text[start:].strip()
    if tail:
        segments.append(tail)
    return segments or [text]


def _bounded_segments(text: str, cut_punc: str, maximum_characters: int) -> list[str]:
    """Split a request into profile-safe model calls without dropping text."""

    if maximum_characters < 1:
        raise TensorRTRuntimeError("TensorRT engine has an invalid text segment limit")
    initial = _cut_segments(text, cut_punc)
    boundary_characters = set(DEFAULT_SAFE_CUT_PUNCTUATION)
    result: list[str] = []
    for segment in initial:
        remaining = segment.lstrip(" \t")
        while len(remaining) > maximum_characters:
            window = remaining[: maximum_characters + 1]
            split_at = max((window.rfind(character) + 1 for character in boundary_characters), default=0)
            # Do not create a near-empty fragment merely because a leading
            # punctuation mark happened to be present.
            if split_at < max(1, maximum_characters // 3):
                whitespace = max(window.rfind(" ") + 1, window.rfind("\n") + 1)
                split_at = whitespace if whitespace >= max(1, maximum_characters // 3) else maximum_characters
            chunk = remaining[:split_at].strip()
            if chunk:
                result.append(chunk)
            remaining = remaining[split_at:].lstrip(" \t")
        if remaining.strip():
            result.append(remaining)
    return result or [text]


def _natural_segment_content_length(segment: str) -> int:
    return sum(
        1
        for character in segment
        if not character.isspace()
        and character not in SAFE_INFERENCE_BOUNDARIES
        and character not in BOUNDARY_CLOSERS
    )


def _has_natural_inference_boundary(segment: str) -> bool:
    if segment.endswith(("\n\n", "\r\n\r\n")):
        return True
    ending = segment.rstrip().rstrip("".join(BOUNDARY_CLOSERS))
    return bool(ending) and ending[-1] in SAFE_INFERENCE_BOUNDARIES


def _merge_short_natural_segments(
    segments: list[str],
    maximum_characters: int,
    minimum_content_characters: int = MIN_NATURAL_SEGMENT_CONTENT_CHARS,
) -> list[str]:
    """Keep short interjections from becoming empty early-EOS model calls."""

    if maximum_characters < 1:
        raise TensorRTRuntimeError("TensorRT engine has an invalid text segment limit")
    if minimum_content_characters < 1:
        return list(segments)

    pending = [segment for segment in segments if segment.strip()]
    merged: list[str] = []
    index = 0
    while index < len(pending):
        segment = pending[index]
        is_short_natural = (
            _has_natural_inference_boundary(segment)
            and _natural_segment_content_length(segment) < minimum_content_characters
        )
        if is_short_natural and index + 1 < len(pending):
            combined = segment.rstrip(" \t") + pending[index + 1].lstrip(" \t")
            if len(combined) <= maximum_characters:
                pending[index + 1] = combined
                index += 1
                continue
        if is_short_natural and merged:
            combined = merged[-1].rstrip(" \t") + segment.lstrip(" \t")
            if len(combined) <= maximum_characters:
                merged[-1] = combined
                index += 1
                continue
        merged.append(segment)
        index += 1
    return merged


def _recommended_stream_prebuffer_ms(segments: list[str]) -> int:
    long_stream = len(segments) > 1 or any(
        len(segment) > STREAM_LONG_SEGMENT_CHAR_THRESHOLD for segment in segments
    )
    return (
        STREAM_LONG_RECOMMENDED_PREBUFFER_MS
        if long_stream
        else STREAM_RECOMMENDED_PREBUFFER_MS
    )


def _pcm16_wav(sample_rate: int, audio: np.ndarray) -> bytes:
    """Make an ordinary, complete RIFF/WAV payload from floating point mono audio."""

    if sample_rate <= 0:
        raise TensorRTRuntimeError(f"Invalid TensorRT output sample rate: {sample_rate}")
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        raise TensorRTRuntimeError("TensorRT produced an empty audio response")
    if not np.isfinite(mono).all():
        raise TensorRTRuntimeError("TensorRT produced non-finite audio samples")
    pcm = np.rint(np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return output.getvalue()


def _pcm16_bytes(audio: np.ndarray) -> bytes:
    """Encode one finite mono chunk for the streaming PCM16 transport."""

    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return b""
    if not np.isfinite(mono).all():
        raise TensorRTRuntimeError("TensorRT produced non-finite streaming audio")
    return np.rint(np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2", copy=False).tobytes()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TensorRTRuntimeError(f"Unable to read {description}") from error
    if not isinstance(loaded, dict):
        raise TensorRTRuntimeError(f"{description} must contain a JSON object")
    return loaded


class TensorRTService:
    """Owns exactly one fixed-reference TensorRT pipeline and CUDA context."""

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self._engine: Any | None = None
        self._streamer: Any | None = None
        self._torch: Any | None = None
        self._trt: Any | None = None
        self._sample_rate: int | None = None
        self._engine_manifest: dict[str, Any] = {}
        self._warmup: dict[str, Any] | None = None
        self._warm_retention: WarmRetentionController | None = None
        self._segment_char_limit = PROFILE_SEGMENT_CHAR_LIMITS["small"]
        # TensorRT execution contexts remain serialized per active voice.  The
        # admission semaphore rejects hidden queueing before HTTP 200 is sent.
        self._inference_lock = threading.RLock()
        self._request_slot = threading.BoundedSemaphore(1)
        self._active_guard = threading.Lock()
        self._active_requests = 0

    @property
    def ready(self) -> bool:
        return self._engine is not None

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            raise TensorRTRuntimeError("TensorRT pipeline is not loaded")
        return self._sample_rate

    def load(self) -> None:
        """Validate and load TensorRT only; never switch to Torch/ONNX inference."""

        if self._engine is not None:
            return
        self._validate_layout()
        try:
            import tensorrt as trt_module
            import torch as torch_module
        except ImportError as error:
            raise TensorRTRuntimeError(
                "TensorRT runtime dependencies are unavailable; PyTorch/ONNX fallback is disabled"
            ) from error
        if not torch_module.cuda.is_available():
            raise TensorRTRuntimeError(
                "No CUDA GPU is available to TensorRT; CPU and PyTorch fallback are disabled"
            )
        try:
            properties = torch_module.cuda.get_device_properties(0)
            capability = torch_module.cuda.get_device_capability(0)
        except Exception as error:
            raise TensorRTRuntimeError("Unable to inspect the CUDA GPU for TensorRT") from error
        if tuple(capability) < (7, 5):
            raise TensorRTRuntimeError(
                "The visible NVIDIA GPU is older than SM 7.5; this TensorRT deployment is unsupported"
            )
        if int(properties.total_memory) < 8 * 1024 * 1024 * 1024:
            raise TensorRTRuntimeError(
                "The visible NVIDIA GPU has less than the required 8 GiB of VRAM"
            )

        self._prepend_source_dir()
        try:
            source_module = importlib.import_module("run_trt_inference")
            source_file = Path(getattr(source_module, "__file__", "")).resolve()
            expected_source = (self.settings.source_dir / "run_trt_inference.py").resolve()
            if source_file != expected_source:
                raise TensorRTRuntimeError(
                    "Loaded run_trt_inference.py from an unexpected location"
                )
            inference_class = getattr(source_module, "GPTSoVITS_TRT_Inference")
            streaming_class = getattr(
                importlib.import_module("aniflive_tts.streaming"),
                "TensorRTFixedReferenceStreamer",
            )
            self._engine_manifest = _read_json(
                self.settings.engine_dir / "engine-manifest.json", "engine manifest"
            )
            self._segment_char_limit = self._resolve_segment_char_limit()
            self._torch = torch_module
            self._trt = trt_module
            self._engine = inference_class(
                trt_dir=str(self.settings.engine_dir),
                bert_path=str(self.settings.bert_path),
                device="cuda",
            )
            self._sample_rate = int(self._engine.hps["data"]["sampling_rate"])
            if self._sample_rate <= 0:
                raise TensorRTRuntimeError("Engine config contains an invalid sample rate")
            # Load the mixed-English frontend at startup.  This makes a missing
            # NLTK CMU dictionary fail loudly during startup rather than after a
            # client has already received an HTTP 200 streaming response.
            importlib.import_module("GPT_SoVITS.text.english")
            self._streamer = streaming_class(
                self._engine,
                reference_wav=self.settings.reference_wav,
                reference_text=REFERENCE_TEXT,
                reference_language=REFERENCE_LANGUAGE,
                logger=LOGGER,
            )
            self._streamer.prepare_reference()
            self._streamer.warm_frontends()
            self._run_strict_warmup()
            self._warm_retention = WarmRetentionController(
                pulse=self._streamer.keepwarm_pulse,
                inference_lock=self._inference_lock,
                is_busy=self._is_busy,
                retention_seconds=float(
                    os.environ.get("ANIFLIVE_TTS_WARM_RETENTION_SECONDS", "25")
                ),
                pulse_interval_seconds=float(
                    os.environ.get("ANIFLIVE_TTS_WARM_PULSE_INTERVAL_SECONDS", "6")
                ),
                maximum_temperature_c=int(
                    os.environ.get("ANIFLIVE_TTS_WARM_MAX_TEMP_C", "70")
                ),
                resume_temperature_c=int(
                    os.environ.get("ANIFLIVE_TTS_WARM_RESUME_TEMP_C", "65")
                ),
                maximum_utilization_percent=int(
                    os.environ.get("ANIFLIVE_TTS_WARM_MAX_GPU_UTILIZATION", "20")
                ),
                maximum_pulse_seconds=float(
                    os.environ.get("ANIFLIVE_TTS_WARM_MAX_PULSE_SECONDS", "0.040")
                ),
            )
            self._warm_retention.start()
            self._warm_retention.notify_real_activity()
        except Exception as error:
            self.unload()
            if isinstance(error, TensorRTRuntimeError):
                raise
            raise TensorRTRuntimeError(
                "Unable to initialize the required TensorRT pipeline"
            ) from error

    def unload(self) -> None:
        if self._warm_retention is not None:
            self._warm_retention.stop()
            self._warm_retention = None
        if self._streamer is not None:
            try:
                self._streamer.close()
            except Exception:
                LOGGER.debug("Streaming runtime cleanup failed", exc_info=True)
        self._engine = None
        self._streamer = None
        self._sample_rate = None
        self._warmup = None
        self._segment_char_limit = PROFILE_SEGMENT_CHAR_LIMITS["small"]
        if self._torch is not None:
            try:
                if self._torch.cuda.is_available():
                    self._torch.cuda.synchronize()
                    gc.collect()
                    self._torch.cuda.empty_cache()
                    self._torch.cuda.synchronize()
            except Exception:  # pragma: no cover - best-effort CUDA cleanup
                LOGGER.debug("CUDA cache cleanup failed", exc_info=True)
        self._torch = None
        self._trt = None

    def synthesize(self, options: SynthesisOptions) -> SynthesisResult:
        if self._engine is None or self._streamer is None or self._sample_rate is None:
            raise TensorRTRuntimeError("TensorRT pipeline is not loaded")

        self._begin_request()
        started = time.perf_counter()
        request_seed = self._effective_seed(options.seed)
        try:
            with self._inference_lock:
                segments = self._segments(options)
                outputs = list(self._streamer.iter_audio(
                    segments=segments,
                    text_language=options.text_language,
                    top_k=options.top_k,
                    top_p=options.top_p,
                    temperature=options.temperature,
                    noise_scale=options.noise_scale,
                    speed=options.speed,
                    pause_length=options.pause_length,
                    request_seed=request_seed,
                    # Full WAV downloads have no first-audio advantage from the
                    # low-latency PCM preview. Decode as much of a
                    # profile-safe sentence as the per-GPU engine permits.
                    chunk_length=self._streamer.complete_wav_chunk_length(),
                ))
        finally:
            self._end_request()
        if not outputs:
            raise TensorRTRuntimeError("TensorRT produced no audio chunks")
        postprocess_started = time.perf_counter()
        audio = np.concatenate(outputs).astype(np.float32, copy=False)
        audio = audio - np.mean(audio, dtype=np.float64)
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-5:
            audio = audio / peak * 0.9
        payload = _pcm16_wav(self._sample_rate, audio)
        profile = dict(self._streamer.last_profile or {})
        profile["audio_postprocess_seconds"] = time.perf_counter() - postprocess_started
        return SynthesisResult(
            wav=payload,
            sample_rate=self._sample_rate,
            output_samples=int(audio.size),
            segments=len(segments),
            elapsed_seconds=time.perf_counter() - started,
            profile=profile,
        )

    def stream_pcm(self, options: SynthesisOptions) -> Iterator[bytes]:
        """Return a producer-safe iterator of little-endian signed PCM16 bytes."""

        if self._engine is None or self._streamer is None or self._sample_rate is None:
            raise TensorRTRuntimeError("TensorRT pipeline is not loaded")

        self._begin_request()
        chunks: queue.Queue[bytes | BaseException | None] = queue.Queue(maxsize=3)
        cancelled = threading.Event()
        request_seed = self._effective_seed(options.seed)

        def put(item: bytes | BaseException | None) -> bool:
            while not cancelled.is_set():
                try:
                    chunks.put(item, timeout=0.2)
                    return True
                except queue.Full:
                    continue
            return False

        def produce() -> None:
            try:
                with self._inference_lock:
                    segments = self._segments(options)
                    for audio in self._streamer.iter_audio(
                        segments=segments,
                        text_language=options.text_language,
                        top_k=options.top_k,
                        top_p=options.top_p,
                        temperature=options.temperature,
                        noise_scale=options.noise_scale,
                        speed=options.speed,
                        pause_length=options.pause_length,
                        request_seed=request_seed,
                        cancelled=cancelled,
                        chunk_length=self._streamer.streaming_chunk_length(),
                    ):
                        payload = _pcm16_bytes(audio)
                        if payload and not put(payload):
                            return
            except BaseException as error:  # Logged by the response iterator.
                put(error)
            finally:
                put(None)
                self._end_request()

        worker = threading.Thread(target=produce, name="aniflive-tts-trt-stream", daemon=True)
        try:
            worker.start()
        except BaseException:
            self._end_request()
            raise

        def consume() -> Iterator[bytes]:
            try:
                while True:
                    item = chunks.get()
                    if item is None:
                        return
                    if isinstance(item, BaseException):
                        LOGGER.error(
                            "TensorRT streaming inference failed",
                            exc_info=(type(item), item, item.__traceback__),
                        )
                        raise TensorRTRuntimeError("TensorRT streaming synthesis failed") from item
                    yield item
            finally:
                cancelled.set()

        return consume()

    @staticmethod
    def _effective_seed(seed: int) -> int:
        return int(seed) if seed >= 0 else secrets.randbelow((1 << 63) - 1)

    def _begin_request(self) -> None:
        if not self._request_slot.acquire(blocking=False):
            raise ServiceBusy("AnifLive-TTS is already processing a synthesis request")
        with self._active_guard:
            self._active_requests += 1

    def _end_request(self) -> None:
        with self._active_guard:
            self._active_requests = max(0, self._active_requests - 1)
        self._request_slot.release()
        if self._warm_retention is not None:
            self._warm_retention.notify_real_activity()

    def _is_busy(self) -> bool:
        with self._active_guard:
            return self._active_requests > 0

    def _segments(self, options: SynthesisOptions) -> list[str]:
        return _merge_short_natural_segments(
            _bounded_segments(
                options.text,
                options.cut_punc,
                self._segment_char_limit,
            ),
            self._segment_char_limit,
        )

    def health(self) -> dict[str, Any]:
        gpu: dict[str, Any] | None = None
        if self._torch is not None:
            try:
                properties = self._torch.cuda.get_device_properties(0)
                major, minor = self._torch.cuda.get_device_capability(0)
                gpu = {
                    "name": self._torch.cuda.get_device_name(0),
                    "compute_capability": f"{major}.{minor}",
                    "vram_bytes": int(properties.total_memory),
                }
            except Exception:
                LOGGER.debug("Could not obtain CUDA device metadata", exc_info=True)

        manifest_fingerprint = (
            self._engine_manifest.get("fingerprint")
            or self._engine_manifest.get("engine_fingerprint")
            or self._engine_manifest.get("id")
        )
        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "status": "ok" if self.ready else "not_ready",
            "ready": self.ready,
            "backend": "TensorRT-11",
            "model": MODEL_ID,
            "voice": VOICE_ID,
            "cuda": getattr(getattr(self._torch, "version", None), "cuda", None),
            "tensorrt": getattr(self._trt, "__version__", None),
            "gpu": gpu,
            "engine_fingerprint": manifest_fingerprint,
            "engine_count": len(REQUIRED_ENGINES) if self.ready else 0,
            "engines": {name: self.ready for name in REQUIRED_ENGINES},
            "reference": {
                "configured": bool(REFERENCE_TEXT),
                "language": REFERENCE_LANGUAGE,
            },
            "request_segment_char_limit": self._segment_char_limit,
            "streaming": {
                "enabled": self._streamer is not None,
                "transport": "pcm_s16le",
                "complete_wav": "request stream=false",
            },
            "startup_warmup": self._warmup,
            "active_requests": self._active_requests,
            "warm_retention": (
                self._warm_retention.status() if self._warm_retention is not None else None
            ),
        }

    def _resolve_segment_char_limit(self) -> int:
        """Read the build profile from the verified per-machine manifest."""

        if self._engine_manifest.get("kind") in {
            "aniflive-tts-gsv-v2proplus-tensorrt11-engines",
        }:
            return PROFILE_SEGMENT_CHAR_LIMITS["fitted"]
        payload = self._engine_manifest.get("payload")
        settings = payload.get("settings") if isinstance(payload, dict) else None
        profile = settings.get("profile") if isinstance(settings, dict) else None
        if isinstance(profile, str) and profile in PROFILE_SEGMENT_CHAR_LIMITS:
            return PROFILE_SEGMENT_CHAR_LIMITS[profile]
        LOGGER.warning(
            "Engine manifest has no recognised TensorRT profile; using the conservative small-request limit"
        )
        return PROFILE_SEGMENT_CHAR_LIMITS["small"]

    def _validate_layout(self) -> None:
        required_paths = [
            self.settings.source_dir / "run_trt_inference.py",
            self.settings.bert_path,
            self.settings.reference_wav,
            self.settings.engine_dir / "config.json",
            self.settings.engine_dir / "engine-manifest.json",
            *(self.settings.engine_dir / f"{name}.engine" for name in REQUIRED_ENGINES),
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise TensorRTRuntimeError(
                "Required TensorRT deployment files are missing: " + ", ".join(missing)
            )
        if not self.settings.bert_path.is_dir():
            raise TensorRTRuntimeError("The configured tokenizer directory is invalid")
        if not self.settings.reference_wav.is_file():
            raise TensorRTRuntimeError("The fixed voice reference audio is invalid")

    def _prepend_source_dir(self) -> None:
        source_dir = str(self.settings.source_dir)
        if source_dir in sys.path:
            sys.path.remove(source_dir)
        sys.path.insert(0, source_dir)

        # Avoid accidentally reusing a same-named module imported from another
        # project in a long-lived interpreter (for example a test runner).
        existing = sys.modules.get("run_trt_inference")
        if existing is not None:
            existing_file = Path(getattr(existing, "__file__", "")).resolve()
            expected = (self.settings.source_dir / "run_trt_inference.py").resolve()
            if existing_file != expected:
                del sys.modules["run_trt_inference"]

    def _run_strict_warmup(self) -> None:
        # This goes through the complete engine bundle using the actual reference
        # audio.  A model constructor's dummy warmup is intentionally not enough
        # evidence that a per-GPU TensorRT build can synthesize speech.
        options = SynthesisOptions(
            text=REFERENCE_TEXT,
            text_language=REFERENCE_LANGUAGE,
            top_k=int(DEFAULTS["top_k"]),
            top_p=float(DEFAULTS["top_p"]),
            temperature=float(DEFAULTS["temperature"]),
            speed=float(DEFAULTS["speed"]),
            pause_length=0.0,
            noise_scale=float(DEFAULTS["noise_scale"]),
            cut_punc="",
            seed=1234,
        )
        result = self.synthesize(options)
        calibration_options = SynthesisOptions(
            text=STREAM_CALIBRATION_TEXT.get(REFERENCE_LANGUAGE, REFERENCE_TEXT),
            text_language=REFERENCE_LANGUAGE,
            top_k=options.top_k,
            top_p=options.top_p,
            temperature=options.temperature,
            speed=options.speed,
            pause_length=0.0,
            noise_scale=options.noise_scale,
            cut_punc="",
            seed=options.seed,
        )
        calibration_started = time.perf_counter()
        calibration_bytes = sum(len(chunk) for chunk in self.stream_pcm(calibration_options))
        calibration_elapsed = time.perf_counter() - calibration_started
        if calibration_bytes <= 0:
            raise TensorRTRuntimeError("TensorRT streaming calibration produced no PCM")
        self._warmup = {
            "completed": True,
            "sample_rate": result.sample_rate,
            "output_samples": result.output_samples,
            "elapsed_seconds": round(result.elapsed_seconds, 6),
            "stream_calibration": {
                "completed": True,
                "pcm_bytes": calibration_bytes,
                "elapsed_seconds": round(calibration_elapsed, 6),
            },
        }
        LOGGER.info(
            "TensorRT warmup completed in %.3fs (%d samples); stream calibration %.3fs",
            result.elapsed_seconds,
            result.output_samples,
            calibration_elapsed,
        )

def _sync_runtime_identity_from_env() -> None:
    global MODEL_ID, VOICE_ID, REFERENCE_TEXT, REFERENCE_LANGUAGE

    MODEL_ID = os.environ.get("ANIFLIVE_TTS_MODEL_ID", "unconfigured")
    VOICE_ID = os.environ.get("ANIFLIVE_TTS_VOICE_PROFILE", "default")
    REFERENCE_TEXT = os.environ.get("ANIFLIVE_TTS_REFERENCE_TEXT", "")
    REFERENCE_LANGUAGE = os.environ.get("ANIFLIVE_TTS_REFERENCE_LANGUAGE", "ja")


class RuntimeServiceManager:
    """Own one TensorRT voice at a time and switch packages transactionally."""

    def __init__(self, service: TensorRTService) -> None:
        self._service = service
        self._switch_lock = threading.RLock()
        self._switching = False
        self._package_dir = Path(
            os.environ.get("ANIFLIVE_TTS_MODEL_PACKAGE", "/data/models/active")
        ).expanduser().resolve()

    @property
    def settings(self) -> RuntimeSettings:
        return self._service.settings

    @property
    def ready(self) -> bool:
        return self._service.ready and not self._switching

    @property
    def switching(self) -> bool:
        return self._switching

    @property
    def sample_rate(self) -> int:
        return self._service.sample_rate

    @property
    def _sample_rate(self) -> int | None:
        return self._service._sample_rate

    @_sample_rate.setter
    def _sample_rate(self, value: int | None) -> None:
        self._service._sample_rate = value

    def load(self) -> None:
        with self._switch_lock:
            self._service.load()

    def unload(self) -> None:
        with self._switch_lock:
            self._service.unload()

    def synthesize(self, options: SynthesisOptions) -> SynthesisResult:
        with self._switch_lock:
            if self._switching:
                raise ServiceBusy("AnifLive-TTS is switching the active model")
            return self._service.synthesize(options)

    def synthesize_request(
        self, options: SynthesisOptions, requested_model: str | None
    ) -> tuple[SynthesisResult, str]:
        with self._switch_lock:
            self._assert_active_model(requested_model)
            active_model = MODEL_ID
            return self._service.synthesize(options), active_model

    def stream_pcm(self, options: SynthesisOptions) -> Iterator[bytes]:
        with self._switch_lock:
            if self._switching:
                raise ServiceBusy("AnifLive-TTS is switching the active model")
            # stream_pcm reserves the one request slot before returning, so a
            # later activation observes the active producer and is rejected.
            return self._service.stream_pcm(options)

    def prepare_stream(
        self, options: SynthesisOptions, requested_model: str | None
    ) -> tuple[list[str], Iterator[bytes], int, str]:
        with self._switch_lock:
            self._assert_active_model(requested_model)
            active_model = MODEL_ID
            segments = self._service._segments(options)
            pcm = self.stream_pcm(options)
            return segments, pcm, self._service.sample_rate, active_model

    def _segments(self, options: SynthesisOptions) -> list[str]:
        with self._switch_lock:
            return self._service._segments(options)

    def health(self) -> dict[str, Any]:
        payload = self._service.health()
        payload["ready"] = self.ready
        payload["status"] = "switching" if self._switching else payload["status"]
        payload["switching"] = self._switching
        payload["available_model_count"] = len(self.list_models())
        return payload

    def list_models(self) -> list[dict[str, Any]]:
        packages = self._discover_packages()
        return [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "aniflive-tts-local",
                "description": "GPT-SoVITS V2 Pro Plus, TensorRT 11 only",
                "model_family": str(record["manifest"]["model_family"]),
                "voice_profiles": list(record["manifest"].get("voice_profiles", [])),
                "active": model_id == MODEL_ID,
            }
            for model_id, record in sorted(packages.items())
        ]

    def activate(self, model_id: str) -> dict[str, Any]:
        from .model_package import validate_safe_identifier

        try:
            requested = validate_safe_identifier(model_id, "model")
        except Exception as error:
            raise RequestError("model must be a safe local model identifier") from error
        with self._switch_lock:
            if requested == MODEL_ID and self._service.ready:
                return {"changed": False, "model": MODEL_ID, "voice": VOICE_ID}
            if self._service._is_busy():
                raise ModelSwitchConflict(
                    "Cannot switch models while a synthesis request is active"
                )
            packages = self._discover_packages()
            if requested not in packages:
                raise RequestError(f"Unknown local model: {requested!r}")

            previous_package = self._package_dir
            previous_voice = VOICE_ID
            target_package = Path(packages[requested]["path"])
            target_voice = validate_safe_identifier(
                packages[requested]["manifest"].get("default_voice_profile", "default"),
                "voice_profile",
            )
            self._switching = True
            self._service.unload()
            try:
                replacement = self._load_package(target_package, target_voice)
            except Exception as switch_error:
                LOGGER.exception("Unable to activate model %s; restoring %s", requested, MODEL_ID)
                try:
                    self._service = self._load_package(previous_package, previous_voice)
                    self._package_dir = previous_package
                except Exception as rollback_error:
                    LOGGER.critical("Unable to restore the previous TensorRT model", exc_info=True)
                    raise TensorRTRuntimeError(
                        "Model activation failed and the previous model could not be restored"
                    ) from rollback_error
                raise TensorRTRuntimeError(
                    "Model activation failed; the previous model was restored"
                ) from switch_error
            finally:
                self._switching = False

            self._service = replacement
            self._package_dir = target_package
            return {"changed": True, "model": MODEL_ID, "voice": VOICE_ID}

    def _load_package(
        self, package_dir: Path, voice_profile: str = "default"
    ) -> TensorRTService:
        from .api import configure_runtime
        from .settings import RuntimeSettings as PackageRuntimeSettings

        os.environ["ANIFLIVE_TTS_MODEL_PACKAGE"] = str(package_dir)
        os.environ["ANIFLIVE_TTS_VOICE_PROFILE"] = voice_profile
        configure_runtime(PackageRuntimeSettings.from_env())
        _sync_runtime_identity_from_env()
        service = TensorRTService(RuntimeSettings.from_env())
        service.load()
        return service

    @staticmethod
    def _assert_active_model(requested_model: str | None) -> None:
        if requested_model is not None and requested_model != MODEL_ID:
            raise RequestError(f"model must match the active model {MODEL_ID!r}")

    def _discover_packages(self) -> dict[str, dict[str, Any]]:
        from .model_package import validate_safe_identifier

        registry_root = Path(
            os.environ.get("ANIFLIVE_TTS_MODEL_REGISTRY", str(self._package_dir.parent))
        ).expanduser().resolve()
        candidates = [self._package_dir]
        if registry_root.is_dir():
            candidates.extend(
                child.resolve()
                for child in registry_root.iterdir()
                if child.is_dir() and not child.is_symlink()
            )

        packages: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            manifest_path = candidate / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = _read_json(manifest_path, "model package manifest")
                if (
                    manifest.get("format") != "aniflive-tts-model-package"
                    or manifest.get("model_family") != "gsv-v2proplus"
                    or manifest.get("precision") != "FP16"
                ):
                    continue
                model_id = validate_safe_identifier(manifest.get("model_id"), "model_id")
            except Exception:
                LOGGER.warning("Ignoring invalid model package metadata in %s", candidate)
                continue
            existing = packages.get(model_id)
            if existing is None or candidate == self._package_dir:
                packages[model_id] = {"path": candidate, "manifest": manifest}
        return packages


SERVICE = RuntimeServiceManager(TensorRTService(RuntimeSettings.from_env()))


def _assert_fixed_reference(values: Mapping[str, Any]) -> None:
    """Permit the fixed reference values, but never permit a voice override."""

    reference_path = SERVICE.settings.reference_wav.resolve()
    for field in ("refer_wav_path", "ref_audio"):
        supplied = _optional_string(values.get(field), field)
        if supplied is None:
            continue
        candidate = Path(supplied).expanduser().resolve()
        if candidate != reference_path:
            raise RequestError(f"{field} cannot override the fixed voice profile reference")
    for field in ("prompt_text", "ref_text"):
        supplied = _optional_string(values.get(field), field)
        if supplied is not None and supplied != REFERENCE_TEXT:
            raise RequestError(f"{field} cannot override the fixed voice profile reference text")
    for field in ("prompt_language", "ref_lang"):
        supplied = _optional_string(values.get(field), field)
        if supplied is None:
            continue
        language = _normalise_language(supplied, field)
        configured = _normalise_language(
            REFERENCE_LANGUAGE, "ANIFLIVE_TTS_REFERENCE_LANGUAGE"
        )
        if language.removeprefix("all_") != configured.removeprefix("all_"):
            raise RequestError(
                f"{field} cannot override the fixed voice profile reference language"
            )

    extra_references = values.get("inp_refs")
    if extra_references not in (None, [], (), ""):
        raise RequestError("inp_refs is not supported by a fixed voice profile")


def _options_from_values(
    values: Mapping[str, Any], *, text_field: str, language_field: str
) -> SynthesisOptions:
    _assert_fixed_reference(values)
    text = _optional_string(values.get(text_field), text_field)
    if text is None:
        raise RequestError(f"Missing required parameter: {text_field}")
    if len(text) > MAX_TEXT_CHARS:
        raise RequestError(f"{text_field} is limited to {MAX_TEXT_CHARS} characters")

    text_language = _normalise_language(values.get(language_field), language_field)
    cut_punc = _optional_string(values.get("cut_punc"), "cut_punc") or ""
    if len(cut_punc) > 64:
        raise RequestError("cut_punc is too long")

    top_k = _integer(values.get("top_k"), "top_k", int(DEFAULTS["top_k"]))
    if top_k > 50:
        raise RequestError("top_k must be at most 50 for the exported TensorRT sampling graph")
    top_p = _number(
        values.get("top_p"), "top_p", float(DEFAULTS["top_p"]), minimum=0.000001, maximum=1.0
    )
    temperature = _number(
        values.get("temperature"),
        "temperature",
        float(DEFAULTS["temperature"]),
        minimum=0.000001,
    )
    speed = _number(
        values.get("speed"), "speed", float(DEFAULTS["speed"]), minimum=0.000001
    )
    if not math.isclose(speed, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise RequestError(
            "speed must be 1.0 for this V2 Pro Plus TensorRT engine; the exported graph has no speed input"
        )
    sample_steps = _integer(values.get("sample_steps"), "sample_steps", 32)
    if sample_steps != 32:
        raise RequestError(
            "sample_steps must remain 32; V2 Pro Plus does not contain the V4 CFM sampler"
        )
    if _boolean(values.get("if_sr"), "if_sr", False):
        raise RequestError("if_sr=true is not supported by the 32 kHz V2 Pro Plus engine")
    pause_length = _number(
        values.get("pause_length"),
        "pause_length",
        float(DEFAULTS["pause_length"]),
        minimum=0.0,
        maximum=10.0,
    )
    noise_scale = _number(
        values.get("noise_scale"),
        "noise_scale",
        float(DEFAULTS["noise_scale"]),
        minimum=0.0,
        maximum=10.0,
    )
    seed = _integer(values.get("seed"), "seed", int(DEFAULTS["seed"]), minimum=-1)
    return SynthesisOptions(
        text=text,
        text_language=text_language,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        speed=speed,
        pause_length=pause_length,
        noise_scale=noise_scale,
        cut_punc=cut_punc,
        seed=seed,
    )


async def _tts_response(
    request: Request,
    values: Mapping[str, Any],
    *,
    text_field: str,
    language_field: str,
) -> Response:
    try:
        options = _options_from_values(
            values, text_field=text_field, language_field=language_field
        )
        streaming = _boolean(values.get("stream"), "stream", False)
        response_format = (
            _optional_string(values.get("response_format"), "response_format")
            or ("pcm" if streaming else "wav")
        ).lower()
        if streaming:
            if response_format not in {"pcm", "pcm_s16le", "s16le"}:
                raise RequestError(
                    "stream=true requires response_format='pcm'. A valid RIFF/WAV file needs its final data length; use stream=false for a downloadable WAV."
                )
            requested_model = _optional_string(values.get("model"), "model")
            request_segments, pcm, sample_rate, response_model = SERVICE.prepare_stream(
                options, requested_model
            )
            recommended_prebuffer_ms = _recommended_stream_prebuffer_ms(request_segments)
            return StreamingResponse(
                pcm,
                media_type="application/octet-stream",
                headers={
                    "X-TTS-Service": SERVICE_NAME,
                    "X-TTS-Version": SERVICE_VERSION,
                    "X-TensorRT-Backend": "TensorRT-11",
                    "X-TensorRT-Engine-Count": str(len(REQUIRED_ENGINES)),
                    "X-PyTorch-Fallback": "false",
                    "X-TTS-Model": response_model,
                    "X-TTS-Stream": "pcm_s16le",
                    "X-TTS-Sample-Format": "s16le",
                    "X-TTS-Sample-Rate": str(sample_rate),
                    "X-TTS-Channels": "1",
                    "X-TTS-Recommended-Prebuffer-Ms": str(
                        recommended_prebuffer_ms
                    ),
                    "X-TTS-Queue-Ms": "0.000",
                    "X-TTS-Pause-Mode": "adaptive",
                    "Cache-Control": "no-store",
                },
            )
        if response_format != "wav":
            raise RequestError("response_format must be 'wav' when stream=false")
        requested_model = _optional_string(values.get("model"), "model")
        result, response_model = await run_in_threadpool(
            SERVICE.synthesize_request, options, requested_model
        )
    except RequestError as error:
        return _json_response(str(error), error.status_code)
    except Exception:
        LOGGER.exception("TensorRT inference failed")
        # Do not return source/container paths or a potentially misleading
        # fallback claim.  Container logs retain the original exception.
        return _json_response(
            "TensorRT synthesis failed; inspect the AnifLive-TTS container logs", 500
        )

    return Response(
        result.wav,
        media_type="audio/wav",
        headers={
            "X-TTS-Service": SERVICE_NAME,
            "X-TTS-Version": SERVICE_VERSION,
            "X-TensorRT-Backend": "TensorRT-11",
            "X-TensorRT-Engine-Count": str(len(REQUIRED_ENGINES)),
            "X-PyTorch-Fallback": "false",
            "X-TTS-Model": response_model,
            "X-TTS-Sample-Rate": str(result.sample_rate),
            "X-TTS-Output-Samples": str(result.output_samples),
            "X-TTS-Segments": str(result.segments),
            "X-TTS-Inference-Seconds": f"{result.elapsed_seconds:.6f}",
            "X-TTS-Seed": str(options.seed),
            "X-TTS-Queue-Ms": "0.000",
            "X-TTS-Pause-Mode": "adaptive",
            "X-TTS-Stage-Text-Seconds": f"{float(result.profile.get('text_processing_seconds', 0.0)):.6f}",
            "X-TTS-Stage-GPT-Encoder-Seconds": f"{float(result.profile.get('gpt_encoder_seconds', 0.0)):.6f}",
            "X-TTS-Stage-GPT-Decode-Seconds": f"{float(result.profile.get('gpt_decode_seconds', 0.0)):.6f}",
            "X-TTS-Stage-SoVITS-Seconds": f"{float(result.profile.get('sovits_seconds', 0.0)):.6f}",
            "X-TTS-Stage-Postprocess-Seconds": f"{float(result.profile.get('audio_postprocess_seconds', 0.0)):.6f}",
            "X-TTS-Semantic-Tokens": str(int(result.profile.get('semantic_tokens', 0))),
            "X-TTS-GPT-Steps": str(int(result.profile.get('gpt_steps', 0))),
            "X-TTS-Sample-CUDA-Graph": (
                "true" if int(result.profile.get("sample_cuda_graph_enabled", 0)) else "false"
            ),
            "X-TTS-SoVITS-Invocations": str(int(result.profile.get('sovits_invocations', 0))),
        },
    )


async def _read_json_body(request: Request) -> dict[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as error:
            raise RequestError("Content-Length must be a non-negative integer") from error
        if declared_length < 0:
            raise RequestError("Content-Length must be a non-negative integer")
        if declared_length > MAX_JSON_BODY_BYTES:
            raise RequestBodyTooLarge(f"Request body is limited to {MAX_JSON_BODY_BYTES} bytes")

    raw_body = bytearray()
    async for chunk in request.stream():
        if len(raw_body) + len(chunk) > MAX_JSON_BODY_BYTES:
            raise RequestBodyTooLarge(
                f"Request body is limited to {MAX_JSON_BODY_BYTES} bytes"
            )
        raw_body.extend(chunk)
    try:
        body = json.loads(bytes(raw_body))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RequestError("Request body must be valid JSON") from error
    if not isinstance(body, dict):
        raise RequestError("Request body must be a JSON object")
    return body


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup intentionally fails if a valid TensorRT pipeline cannot be loaded.
    # Serving a deceptively successful CPU/PyTorch fallback is prohibited.
    SERVICE.load()
    try:
        yield
    finally:
        SERVICE.unload()


app = FastAPI(
    title="AnifLive-TTS API",
    version=SERVICE_VERSION,
    description="Low-latency multilingual AnifLive-TTS v1.2 service backed by TensorRT 11.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def request_validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
    # Keep bad query/body shapes compatible with the legacy API's error object.
    return _json_response(f"Invalid request: {error.errors()[0]['msg']}", 400)


@app.get("/", response_class=Response)
async def legacy_tts_get(
    request: Request,
    text: str | None = None,
    text_language: str | None = None,
    refer_wav_path: str | None = None,
    prompt_text: str | None = None,
    prompt_language: str | None = None,
    top_k: int = int(DEFAULTS["top_k"]),
    top_p: float = float(DEFAULTS["top_p"]),
    temperature: float = float(DEFAULTS["temperature"]),
    speed: float = float(DEFAULTS["speed"]),
    cut_punc: str | None = None,
    pause_length: float = float(DEFAULTS["pause_length"]),
    noise_scale: float = float(DEFAULTS["noise_scale"]),
    seed: int = int(DEFAULTS["seed"]),
    stream: bool = False,
    response_format: str = "wav",
    inp_refs: list[str] = Query(default=[]),
) -> Response:
    return await _tts_response(
        request,
        {
            "text": text,
            "text_language": text_language,
            "refer_wav_path": refer_wav_path,
            "prompt_text": prompt_text,
            "prompt_language": prompt_language,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "speed": speed,
            "cut_punc": cut_punc,
            "pause_length": pause_length,
            "noise_scale": noise_scale,
            "seed": seed,
            "stream": stream,
            "response_format": response_format,
            "inp_refs": inp_refs,
        },
        text_field="text",
        language_field="text_language",
    )


@app.post("/", response_class=Response)
async def legacy_tts_post(request: Request) -> Response:
    try:
        body = await _read_json_body(request)
    except RequestError as error:
        return _json_response(str(error), error.status_code)
    return await _tts_response(
        request, body, text_field="text", language_field="text_language"
    )


@app.post("/v1/audio/speech", response_class=Response)
async def openai_speech(request: Request) -> Response:
    try:
        body = await _read_json_body(request)
        if "text" in body or "voice_profile" in body or "generation" in body:
            model = _optional_string(body.get("model"), "model") or MODEL_ID
            if model != MODEL_ID:
                raise RequestError(f"model must match the active model {MODEL_ID!r}")
            voice_profile = _optional_string(body.get("voice_profile"), "voice_profile") or VOICE_ID
            if voice_profile != VOICE_ID:
                raise RequestError(f"voice_profile must match the active profile {VOICE_ID!r}")
            expression = body.get("expression") or {}
            if not isinstance(expression, dict):
                raise RequestError("expression must be an object")
            if _boolean(expression.get("enabled"), "expression.enabled", False):
                return JSONResponse(
                    {
                        "error": {
                            "code": "expression_not_implemented",
                            "message": "Controlled expression is reserved but not implemented in v1",
                        }
                    },
                    status_code=501,
                )
            generation = body.get("generation") or {}
            if not isinstance(generation, dict):
                raise RequestError("generation must be an object")
            for key in ("top_k", "top_p", "temperature", "seed", "noise_scale", "speed"):
                if key in generation:
                    body[key] = generation[key]
            body["language"] = _canonical_language(body.get("language"), "language")
            return await _tts_response(
                request, body, text_field="text", language_field="language"
            )
        voice = _optional_string(body.get("voice"), "voice") or "default"
        if voice not in {"default", VOICE_ID}:
            raise RequestError(f"voice must be 'default' or {VOICE_ID!r}")
        model = _optional_string(body.get("model"), "model") or MODEL_ID
        if model != MODEL_ID:
            raise RequestError(f"model must be {MODEL_ID!r}")
        # OpenAI-style clients use text_lang; accepting text_language makes the
        # endpoint convenient for code shared with the legacy GPT-SoVITS route.
        if "text_lang" not in body and "text_language" in body:
            body["text_lang"] = body["text_language"]
        if body.get("text_lang") is None:
            body["text_lang"] = "auto"
    except RequestError as error:
        return _json_response(str(error), error.status_code)
    return await _tts_response(
        request, body, text_field="input", language_field="text_lang"
    )


@app.get("/v1/capabilities")
async def capabilities() -> dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "model_family": "gsv-v2proplus",
        "backend": "TensorRT-11",
        "precision": "FP16",
        "languages": ["zh", "yue", "en", "ja", "ko"],
        "streaming": {"pcm16": True, "wav": False},
        "expression": {
            "native": True,
            "controlled_profiles": False,
            "continuous_vector": False,
        },
        "pytorch_fallback": False,
        "adaptive_punctuation_segments": True,
        "warm_retention_seconds": 25,
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return SERVICE.health()


@app.get("/model/config")
async def model_config() -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "voice": VOICE_ID,
        "version": "v2ProPlus",
        "backend": "TensorRT-11",
        "reference_language": REFERENCE_LANGUAGE,
        "reference_configured": True,
        "languages": ["zh", "yue", "en", "ja", "ko", "auto", "auto_yue"],
        "sample_rate": SERVICE.sample_rate,
        "engine_count": len(REQUIRED_ENGINES),
        "pytorch_fallback": False,
        "service_version": SERVICE_VERSION,
        "adaptive_punctuation_segments": True,
    }


def _change_reference(values: Mapping[str, Any]) -> JSONResponse:
    try:
        _assert_fixed_reference(values)
        required = ("refer_wav_path", "prompt_text", "prompt_language")
        if not all(values.get(name) for name in required):
            raise RequestError(
                'Missing one or more parameters: "refer_wav_path", "prompt_text", "prompt_language"'
            )
        return JSONResponse({"code": 0, "message": "Success"})
    except RequestError as error:
        return _json_response(str(error), error.status_code)


@app.post("/change_refer")
async def change_refer_post(request: Request) -> JSONResponse:
    try:
        return _change_reference(await _read_json_body(request))
    except RequestError as error:
        return _json_response(str(error), error.status_code)


@app.get("/change_refer")
async def change_refer_get(
    refer_wav_path: str | None = None,
    prompt_text: str | None = None,
    prompt_language: str | None = None,
) -> JSONResponse:
    return _change_reference(locals())


def _set_model(values: Mapping[str, Any]) -> JSONResponse:
    return _json_response(
        "Runtime model switching is disabled. Convert a package and restart one process with that active model.",
        400,
    )


@app.post("/set_model")
async def set_model_post(request: Request) -> JSONResponse:
    try:
        return _set_model(await _read_json_body(request))
    except RequestError as error:
        return _json_response(str(error), error.status_code)


@app.get("/set_model")
async def set_model_get(
    gpt_model_path: str | None = None,
    sovits_model_path: str | None = None,
) -> JSONResponse:
    return _set_model(locals())


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": SERVICE.list_models(),
    }


@app.post("/v1/models/activate")
async def activate_model(request: Request) -> JSONResponse:
    try:
        body = await _read_json_body(request)
        model_id = _optional_string(body.get("model"), "model")
        if model_id is None:
            raise RequestError("Missing required parameter: model")
        result = await run_in_threadpool(SERVICE.activate, model_id)
        return JSONResponse(
            {
                "code": 0,
                "message": "Success",
                **result,
                "backend": "TensorRT-11",
                "pytorch_fallback": False,
            }
        )
    except RequestError as error:
        return _json_response(str(error), error.status_code)
    except Exception:
        LOGGER.exception("TensorRT model activation failed")
        return _json_response(
            "TensorRT model activation failed; inspect the AnifLive-TTS container logs",
            500,
        )


@app.get("/v1/voices")
async def list_voices() -> dict[str, Any]:
    # No filesystem paths are exposed here.  The reference is intentionally
    # fixed inside the container rather than being a caller-selectable asset.
    return {
        "object": "list",
        "data": [
            {
                "id": VOICE_ID,
                "reference_language": REFERENCE_LANGUAGE,
                "reference_configured": True,
            }
        ],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AnifLive-TTS TensorRT 11 API")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "9880")))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
