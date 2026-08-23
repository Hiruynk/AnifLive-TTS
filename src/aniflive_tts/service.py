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
import importlib
import io
import json
import logging
import math
import os
import queue
import re
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


LOGGER = logging.getLogger("aniflive_tts.service")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

SERVICE_NAME = "AnifLive-TTS"
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
)
DEFAULTS = {
    "top_k": 15,
    "top_p": 1.0,
    "temperature": 1.0,
    "speed": 1.0,
    "pause_length": 0.3,
    "noise_scale": 0.5,
    "seed": -1,
}
MAX_TEXT_CHARS = 1000
MAX_JSON_BODY_BYTES = 64 * 1024
ALLOWED_CUT_PUNCTUATION = frozenset(",.;?!、，。？！；：…")
# A TensorRT profile bounds the length of one model invocation, not the total
# HTTP request.  Keep every internal invocation comfortably inside the profile
# selected by the per-GPU builder.  Long requests are split at sentence
# boundaries (or, as a final safeguard, at a character boundary).
PROFILE_SEGMENT_CHAR_LIMITS = {
    "small": 32,
    # The fitted profile has a 100-phoneme SoVITS text maximum.  A conservative
    # 24-character input bound keeps multilingual Japanese/Cantonese text in
    # range; the reference features are cached, so segmentation no longer
    # repeats expensive SSL/VQ/spec/SV work.
    "fitted": 24,
    "medium": 72,
    "large": 160,
}
DEFAULT_SAFE_CUT_PUNCTUATION = "。！？.!?、，；："

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
    return JSONResponse({"code": status_code, "message": message}, status_code=status_code)


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


def _cut_segments(text: str, cut_punc: str) -> list[str]:
    """Apply the GPT-SoVITS ``cut_punc`` convention before native segmentation."""

    if not cut_punc:
        return [text]
    selected = "".join(character for character in cut_punc if character in ALLOWED_CUT_PUNCTUATION)
    if not selected:
        return [text]
    pieces = re.split(f"([{re.escape(selected)}])", text)
    segments: list[str] = []
    for index in range(0, len(pieces) - 1, 2):
        segment = (pieces[index] + pieces[index + 1]).strip()
        if segment:
            segments.append(segment)
    if len(pieces) % 2:
        tail = pieces[-1].strip()
        if tail:
            segments.append(tail)
    return segments or [text]


def _bounded_segments(text: str, cut_punc: str, maximum_characters: int) -> list[str]:
    """Split a request into profile-safe model calls without dropping text."""

    if maximum_characters < 1:
        raise TensorRTRuntimeError("TensorRT engine has an invalid text segment limit")
    initial = _cut_segments(text, cut_punc or DEFAULT_SAFE_CUT_PUNCTUATION)
    boundary_characters = set(cut_punc or DEFAULT_SAFE_CUT_PUNCTUATION)
    result: list[str] = []
    for segment in initial:
        remaining = segment.strip()
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
            remaining = remaining[split_at:].strip()
        if remaining:
            result.append(remaining)
    return result or [text]


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
        self._segment_char_limit = PROFILE_SEGMENT_CHAR_LIMITS["small"]
        # The minimal inference runtime owns one CUDA stream/context.  Concurrent
        # execution against it is not safe, so HTTP requests queue here.
        self._inference_lock = threading.RLock()

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
            self._run_strict_warmup()
        except Exception as error:
            self.unload()
            if isinstance(error, TensorRTRuntimeError):
                raise
            raise TensorRTRuntimeError(
                "Unable to initialize the required TensorRT pipeline"
            ) from error

    def unload(self) -> None:
        self._engine = None
        self._streamer = None
        self._sample_rate = None
        self._warmup = None
        self._segment_char_limit = PROFILE_SEGMENT_CHAR_LIMITS["small"]
        if self._torch is not None:
            try:
                if self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - best-effort CUDA cleanup
                LOGGER.debug("CUDA cache cleanup failed", exc_info=True)
        self._torch = None
        self._trt = None

    def synthesize(self, options: SynthesisOptions) -> SynthesisResult:
        if self._engine is None or self._streamer is None or self._sample_rate is None:
            raise TensorRTRuntimeError("TensorRT pipeline is not loaded")

        started = time.perf_counter()
        with self._inference_lock:
            if options.seed >= 0:
                self._torch.manual_seed(options.seed)
                self._torch.cuda.manual_seed_all(options.seed)
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
                # Full WAV downloads have no first-audio advantage from the
                # low-latency PCM preview. Decode as much of a
                # profile-safe sentence as the per-GPU engine permits.
                chunk_length=self._streamer.complete_wav_chunk_length(),
            ))
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

        chunks: queue.Queue[bytes | BaseException | None] = queue.Queue(maxsize=3)
        cancelled = threading.Event()

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
                    if options.seed >= 0:
                        self._torch.manual_seed(options.seed)
                        self._torch.cuda.manual_seed_all(options.seed)
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
                        chunk_length=self._streamer.streaming_chunk_length(),
                    ):
                        payload = _pcm16_bytes(audio)
                        if payload and not put(payload):
                            return
            except BaseException as error:  # Logged by the response iterator.
                put(error)
            finally:
                put(None)

        worker = threading.Thread(target=produce, name="aniflive-tts-trt-stream", daemon=True)
        worker.start()

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

    def _segments(self, options: SynthesisOptions) -> list[str]:
        return _bounded_segments(
            options.text,
            options.cut_punc,
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
        # This goes through all eight engines using the actual bundled reference
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
        self._warmup = {
            "completed": True,
            "sample_rate": result.sample_rate,
            "output_samples": result.output_samples,
            "elapsed_seconds": round(result.elapsed_seconds, 6),
        }
        LOGGER.info(
            "TensorRT warmup completed in %.3fs (%d samples)",
            result.elapsed_seconds,
            result.output_samples,
        )

SERVICE = TensorRTService(RuntimeSettings.from_env())


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
    values: Mapping[str, Any], *, text_field: str, language_field: str
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
            pcm = SERVICE.stream_pcm(options)
            return StreamingResponse(
                pcm,
                media_type="application/octet-stream",
                headers={
                    "X-TTS-Service": SERVICE_NAME,
                    "X-TensorRT-Backend": "TensorRT-11",
                    "X-TensorRT-Engine-Count": str(len(REQUIRED_ENGINES)),
                    "X-PyTorch-Fallback": "false",
                    "X-TTS-Model": MODEL_ID,
                    "X-TTS-Stream": "pcm_s16le",
                    "X-TTS-Sample-Format": "s16le",
                    "X-TTS-Sample-Rate": str(SERVICE.sample_rate),
                    "X-TTS-Channels": "1",
                    "Cache-Control": "no-store",
                },
            )
        if response_format != "wav":
            raise RequestError("response_format must be 'wav' when stream=false")
        result = await run_in_threadpool(SERVICE.synthesize, options)
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
            "X-TensorRT-Backend": "TensorRT-11",
            "X-TensorRT-Engine-Count": str(len(REQUIRED_ENGINES)),
            "X-PyTorch-Fallback": "false",
            "X-TTS-Model": MODEL_ID,
            "X-TTS-Sample-Rate": str(result.sample_rate),
            "X-TTS-Output-Samples": str(result.output_samples),
            "X-TTS-Segments": str(result.segments),
            "X-TTS-Inference-Seconds": f"{result.elapsed_seconds:.6f}",
            "X-TTS-Seed": str(options.seed),
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
    version="1.0.0",
    description="Low-latency multilingual AnifLive-TTS v1 service backed by TensorRT 11.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def request_validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
    # Keep bad query/body shapes compatible with the legacy API's error object.
    return _json_response(f"Invalid request: {error.errors()[0]['msg']}", 400)


@app.get("/", response_class=Response)
async def legacy_tts_get(
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
    return await _tts_response(body, text_field="text", language_field="text_language")


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
            return await _tts_response(body, text_field="text", language_field="language")
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
    return await _tts_response(body, text_field="input", language_field="text_lang")


@app.get("/v1/capabilities")
async def capabilities() -> dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "version": "1.0.0",
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
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "aniflive-tts-local",
                "description": "GPT-SoVITS V2 Pro Plus, TensorRT 11 only",
            }
        ],
    }


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
