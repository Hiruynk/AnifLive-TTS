"""Native GPT-SoVITS v5 Turbo service on macOS.

The existing HTTP/session layer owns its public contracts. This adapter uses a
single persistent native worker and the explicit CPU text frontend; neural
inference remains in the native MLX worker. PCM is generated incrementally
by the vocoder after AR and CFM complete.
"""
from __future__ import annotations

import io
import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from . import service as api
from .voices import VoiceError
from .expression import ConditioningBundle, ConditioningPolicy, PreparedReference, PreparedReferenceBank

_LEGACY_CFM_NOISE_POLICY = "splitmix64-domain-counter-v1"
_COORDINATE_CFM_NOISE_POLICY = "splitmix64-domain-counter-v1-cfm-coordinate1000-metal-v2"
_BOOTSTRAP_STREAM_GEOMETRY = {
    "core_frames": 20, "future_frames": 12, "holdback_frames": 50, "fade_frames": 4,
}
_CANONICAL_VOCODER_STREAM_CONTRACT = {
    "quality_qualified": False,
    "semantic_eos_before_pcm": True,
    "cfm_complete_before_pcm": True,
    "early_ar_audio": False,
    "vocoder_core_frames": 16,
    "vocoder_halo_frames": 12,
    "vocoder_bucket_frames": 205,
}


class SynthesisCancelled(getattr(api, "SynthesisCancelled", api.RequestError)):
    """A complete-WAV request stopped by /v1/audio/cancel; no audio is returned.

    Subclasses the shared SynthesisCancelled where the common service defines
    it (main-agent integration), so both backends report the same error type.
    """

    status_code = 409


class _Cancelled(Exception):
    pass


class _PCMIterator:
    """Own a producer and release its resources even if never iterated."""

    def __init__(self, chunks: queue.Queue[bytes | BaseException | None],
                 thread: threading.Thread, cancel: Any, abort: Any) -> None:
        self._chunks = chunks
        self._thread = thread
        self._cancel = cancel
        self._abort = abort
        self._closed = False
        self._lock = threading.Lock()

    def __iter__(self) -> _PCMIterator:
        return self

    def __next__(self) -> bytes:
        with self._lock:
            if self._closed:
                raise StopIteration
        item = self._chunks.get()
        if item is None:
            self.close()
            raise StopIteration
        if isinstance(item, BaseException):
            self.close()
            raise item
        return item

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        cancel_error: BaseException | None = None
        try:
            self._cancel()
        except BaseException as error:
            cancel_error = error
        if threading.current_thread() is self._thread:
            if cancel_error is not None:
                raise cancel_error
            return
        self._thread.join(timeout=35.0)
        if self._thread.is_alive():
            # A stuck native process must not keep ownership indefinitely.
            self._abort()
            self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("Native stream did not stop after cancellation")
        if cancel_error is not None:
            raise cancel_error


@dataclass(frozen=True)
class NativeReferenceHandle:
    """Opaque native conditioning identity carried by the existing voice bank."""
    inputs: Path
    audio: Path
    transcript: Path
    language: str
    cache_key: str | None = None


class MacOSV5Service(api.TensorRTService):
    """Existing Voice/Session contracts backed by one MLX/Metal worker."""

    def __init__(self, settings: api.RuntimeSettings) -> None:
        super().__init__(settings)
        self._precision = os.environ.get("ANIFLIVE_TTS_NATIVE_PRECISION", "FP32")
        if self._precision not in {"FP32", "FP16"}:
            raise RuntimeError("ANIFLIVE_TTS_NATIVE_PRECISION must be FP32 or FP16")
        self._package_bank: PreparedReferenceBank | None = None
        self._last_conditioning: list[dict[str, Any]] = []
        self._mlx_memory_ready: dict[str, Any] | None = None
        self._mlx_memory_last: dict[str, Any] | None = None
        self._client: Any | None = None
        self._reference_inputs: Path | None = None
        self._frontend_script: Path | None = None
        self._cache_dir: Path | None = None
        self._last_frontend_seconds: float | None = None
        self._last_frontend_segments = 0
        self._persistent_frontend: Any | None = None
        self._mandarin_enabled = False
        self._worker_supported_top_k: tuple[int, ...] = (15,)
        self._supported_top_k: tuple[int, ...] = (15,)
        self._top_k_1_requires_research_opt_in = False
        self._request_sampling = False
        self._complete_cfm_noise_policy = _LEGACY_CFM_NOISE_POLICY
        self._complete_mode = os.environ.get("ANIFLIVE_TTS_NATIVE_COMPLETE_MODE", "captured")
        if self._complete_mode not in {"captured", "cached"}:
            raise RuntimeError("Native complete mode must be captured or cached")
        cfm_prepare = os.environ.get("ANIFLIVE_TTS_NATIVE_CFM_PREPARE", "0")
        if cfm_prepare not in {"0", "1"}:
            raise RuntimeError("ANIFLIVE_TTS_NATIVE_CFM_PREPARE must be 0 or 1")
        self._cfm_prepare = cfm_prepare == "1"
        ar_combined = os.environ.get("ANIFLIVE_TTS_NATIVE_AR_COMBINED", "0")
        if ar_combined not in {"0", "1"}:
            raise RuntimeError("ANIFLIVE_TTS_NATIVE_AR_COMBINED must be 0 or 1")
        self._ar_combined = ar_combined == "1"
        ar_compiled = os.environ.get("ANIFLIVE_TTS_NATIVE_AR_COMPILED", "0")
        if ar_compiled not in {"0", "1"}:
            raise RuntimeError("ANIFLIVE_TTS_NATIVE_AR_COMPILED must be 0 or 1")
        self._ar_compiled = ar_compiled == "1"
        ar_mixed = os.environ.get("ANIFLIVE_TTS_NATIVE_AR_MIXED_GEMV", "0")
        if ar_mixed not in {"0", "1"}:
            raise RuntimeError("ANIFLIVE_TTS_NATIVE_AR_MIXED_GEMV must be 0 or 1")
        self._ar_mixed = ar_mixed == "1"
        ar_optimized = os.environ.get("ANIFLIVE_TTS_NATIVE_AR_OPTIMIZED", "0")
        if ar_optimized not in {"0", "1"}:
            raise RuntimeError("ANIFLIVE_TTS_NATIVE_AR_OPTIMIZED must be 0 or 1")
        self._ar_optimized = ar_optimized == "1"
        if self._ar_optimized and (not self._ar_combined or not self._ar_mixed or self._ar_compiled):
            raise RuntimeError("Optimized AR requires combined decisions, mixed GEMV and compiled AR disabled")
        if self._ar_mixed and self._ar_compiled:
            raise RuntimeError("Mixed GEMV and compiled AR cannot be combined")
        ar_static = os.environ.get("ANIFLIVE_TTS_NATIVE_AR_STATIC", "0")
        if ar_static not in {"0", "1"}:
            raise RuntimeError("ANIFLIVE_TTS_NATIVE_AR_STATIC must be 0 or 1")
        self._ar_static = ar_static == "1"
        if self._ar_static and not self._ar_optimized:
            raise RuntimeError("Static AR requires optimized AR")
        ar_residual_norm = os.environ.get("ANIFLIVE_TTS_NATIVE_AR_RESIDUAL_NORM", "0")
        if ar_residual_norm not in {"0", "1"}:
            raise RuntimeError("ANIFLIVE_TTS_NATIVE_AR_RESIDUAL_NORM must be 0 or 1")
        self._ar_residual_norm = ar_residual_norm == "1"
        if self._ar_residual_norm and not self._ar_static:
            raise RuntimeError("Residual norm fusion requires static optimized AR")
        cfm_stage = os.environ.get("ANIFLIVE_TTS_NATIVE_CFM_STAGE", "0")
        if cfm_stage not in {"0", "1"}:
            raise RuntimeError("ANIFLIVE_TTS_NATIVE_CFM_STAGE must be 0 or 1")
        self._cfm_stage = cfm_stage == "1"
        cfm_compiled = os.environ.get("ANIFLIVE_TTS_NATIVE_CFM_COMPILED", "0")
        if cfm_compiled not in {"0", "1"}:
            raise RuntimeError("ANIFLIVE_TTS_NATIVE_CFM_COMPILED must be 0 or 1")
        self._cfm_compiled = cfm_compiled == "1"
        if self._cfm_compiled and not self._cfm_stage:
            raise RuntimeError("Compiled CFM requires an isolated CFM stage")
        cfm_batched_qkv = os.environ.get("ANIFLIVE_TTS_NATIVE_CFM_BATCHED_QKV", "0")
        if cfm_batched_qkv not in {"0", "1"}:
            raise RuntimeError("ANIFLIVE_TTS_NATIVE_CFM_BATCHED_QKV must be 0 or 1")
        self._cfm_batched_qkv = cfm_batched_qkv == "1"
        if self._cfm_batched_qkv and not self._cfm_compiled:
            raise RuntimeError("Batched CFM QKV requires the compiled isolated CFM stage")
        if self._cfm_stage and self._cfm_prepare:
            raise RuntimeError("Isolated CFM currently requires local CFM preparation disabled")
        self._frontend_mode = os.environ.get("ANIFLIVE_TTS_NATIVE_FRONTEND_MODE", "process")
        if self._frontend_mode not in {"process", "persistent-python"}:
            raise RuntimeError("Native frontend mode must be process or persistent-python")
        metadata_cache = os.environ.get("ANIFLIVE_TTS_NATIVE_FRONTEND_METADATA_CACHE", "0")
        if metadata_cache not in {"0", "1"}:
            raise RuntimeError("ANIFLIVE_TTS_NATIVE_FRONTEND_METADATA_CACHE must be 0 or 1")
        self._frontend_metadata_cache = metadata_cache == "1"
        onset_trim = os.environ.get("ANIFLIVE_TTS_NATIVE_ONSET_TRIM", "0")
        if onset_trim not in {"0", "1"}:
            raise RuntimeError("ANIFLIVE_TTS_NATIVE_ONSET_TRIM must be 0 or 1")
        self._onset_trim = onset_trim == "1"
        onset_cap = os.environ.get("ANIFLIVE_TTS_NATIVE_ONSET_TRIM_MAX_MS", "60")
        if not onset_cap.isdigit() or not 0 <= int(onset_cap) <= 60:
            raise RuntimeError("ANIFLIVE_TTS_NATIVE_ONSET_TRIM_MAX_MS must be an integer in 0..60")
        self._onset_trim_max_ms = int(onset_cap)
        self._last_onset_trim: dict[str, Any] | None = None
        if self._frontend_metadata_cache and self._frontend_mode != "persistent-python":
            raise RuntimeError("Frontend metadata caching requires persistent-python mode")
        self._allow_unqualified_stream = os.environ.get("ANIFLIVE_TTS_ALLOW_UNQUALIFIED_STREAM", "0") == "1"
        bootstrap_stream_v2 = os.environ.get("ANIFLIVE_TTS_BOOTSTRAP_STREAM_V2", "0")
        if bootstrap_stream_v2 not in {"0", "1"}:
            raise RuntimeError("ANIFLIVE_TTS_BOOTSTRAP_STREAM_V2 must be 0 or 1")
        self._bootstrap_stream_v2 = bootstrap_stream_v2 == "1"
        canonical_stream = os.environ.get("ANIFLIVE_TTS_NATIVE_CANONICAL_STREAM", "0")
        if canonical_stream not in {"0", "1"}:
            raise RuntimeError("ANIFLIVE_TTS_NATIVE_CANONICAL_STREAM must be 0 or 1")
        self._canonical_vocoder_stream = canonical_stream == "1"
        if self._onset_trim and not self._canonical_vocoder_stream:
            raise RuntimeError("PCM onset trim requires canonical vocoder streaming")
        if self._bootstrap_stream_v2 and self._canonical_vocoder_stream:
            raise RuntimeError("Canonical vocoder and bootstrap stream modes are mutually exclusive")
        # The canonical vocoder stream is the accepted product stream (true
        # incremental PCM after AR EOS and CFM). Only the older worker-default
        # and bootstrap stream modes still need the research opt-in.
        self._product_stream = self._canonical_vocoder_stream or self._allow_unqualified_stream
        self._canonical_stream_worker_capability: dict[str, Any] | None = None
        coordinate_noise_v2 = os.environ.get("ANIFLIVE_TTS_NATIVE_CFM_COORDINATE_V2", "0")
        if coordinate_noise_v2 not in {"0", "1"}:
            raise RuntimeError("ANIFLIVE_TTS_NATIVE_CFM_COORDINATE_V2 must be 0 or 1")
        self._coordinate_noise_v2 = coordinate_noise_v2 == "1"
        # FP16 runs the production AR accelerations (combined/mixed/optimized/
        # static/residual norm); compiled AR and FP32-only CFM paths stay off.
        if self._precision == "FP16" and any((self._ar_compiled, self._cfm_stage,
                self._cfm_compiled, self._cfm_prepare, self._cfm_batched_qkv,
                self._bootstrap_stream_v2, self._coordinate_noise_v2)):
            raise RuntimeError("FP16 requires local ordinary CFM with v1 noise and no compiled AR")
        if self._coordinate_noise_v2 and not self._allow_unqualified_stream:
            raise RuntimeError("Coordinate CFM noise v2 requires ANIFLIVE_TTS_ALLOW_UNQUALIFIED_STREAM=1")

    @property
    def ready(self) -> bool:
        return self._client is not None and bool(self._client.ready)

    @staticmethod
    def _worker_sampling_contract(capabilities: dict[str, Any], research_opt_in: bool) -> dict[str, Any]:
        fixed = capabilities.get("fixed_sampling")
        if not isinstance(fixed, dict) or type(fixed.get("top_k")) is not int or fixed["top_k"] != 15:
            raise RuntimeError("Native worker must retain the default fixed top_k=15")
        raw_supported = capabilities.get("supported_top_k")
        if raw_supported is None:
            # Older workers advertise only their fixed_sampling default.
            worker_supported = (fixed["top_k"],)
        else:
            if (not isinstance(raw_supported, list) or not raw_supported
                    or any(type(value) is not int or value not in {1, 15} for value in raw_supported)
                    or len(set(raw_supported)) != len(raw_supported)
                    or fixed["top_k"] not in raw_supported):
                raise RuntimeError("Native worker reported invalid supported_top_k capabilities")
            worker_supported = tuple(sorted(raw_supported))
        requires_opt_in = capabilities.get("top_k_1_requires_research_opt_in", False)
        if type(requires_opt_in) is not bool:
            raise RuntimeError("Native worker top_k_1_requires_research_opt_in must be boolean")
        parameters = capabilities.get("sampling_parameters")
        if parameters is not None:
            # Request-level sampling worker: the Windows-legal ranges.
            if (not isinstance(parameters, dict) or parameters.get("top_k") != {"minimum": 1, "maximum": 50}
                    or parameters.get("default") != {"top_k": 15, "top_p": 1.0, "temperature": 1.0}
                    or requires_opt_in):
                raise RuntimeError("Native worker reported invalid sampling_parameters")
            return {"default": 15, "worker_supported": tuple(range(1, 51)), "available": tuple(range(1, 51)),
                    "top_k_1_requires_research_opt_in": False, "request_sampling": True}
        if 1 in worker_supported and not requires_opt_in:
            raise RuntimeError("Native worker must mark top_k=1 as requiring research opt-in")
        available = tuple(
            value for value in worker_supported
            if value != 1 or not requires_opt_in or research_opt_in
        )
        return {
            "default": fixed["top_k"],
            "worker_supported": worker_supported,
            "available": available,
            "top_k_1_requires_research_opt_in": requires_opt_in,
            "request_sampling": False,
        }

    def _record_worker_contract(self, ready: dict[str, Any]) -> None:
        if ready.get("precision") != ("float16" if self._precision == "FP16" else "float32"):
            raise RuntimeError("Native worker precision does not match configured precision")
        capabilities = ready.get("capabilities")
        if not isinstance(capabilities, dict):
            raise RuntimeError("Native worker omitted capabilities")
        frontend_precision = capabilities.get("frontend_compute_precision")
        if self._precision == "FP16" and frontend_precision != {"g2pw":"float32", "bert":"float16", "bert_transport":"float32"}:
            raise RuntimeError("Native FP16 frontend precision contract is missing or invalid")
        self._mlx_memory_ready = ready.get("memory")
        sampling = self._worker_sampling_contract(capabilities, self._allow_unqualified_stream)
        self._worker_supported_top_k = sampling["worker_supported"]
        self._supported_top_k = sampling["available"]
        self._top_k_1_requires_research_opt_in = sampling["top_k_1_requires_research_opt_in"]
        self._request_sampling = sampling["request_sampling"]
        policy = capabilities.get("complete_cfm_noise_policy")
        if self._coordinate_noise_v2:
            if policy != _COORDINATE_CFM_NOISE_POLICY:
                raise RuntimeError("Native worker did not report the requested coordinate CFM noise policy")
            self._complete_cfm_noise_policy = policy
        else:
            if policy not in (None, _LEGACY_CFM_NOISE_POLICY):
                raise RuntimeError("Native worker CFM noise policy differs from the default profile")
            self._complete_cfm_noise_policy = policy or _LEGACY_CFM_NOISE_POLICY

    @staticmethod
    def _validate_canonical_stream_contract(capabilities: dict[str, Any], requested: bool) -> dict[str, Any]:
        contract = capabilities.get("canonical_vocoder_stream")
        if not isinstance(contract, dict) or type(contract.get("enabled")) is not bool:
            raise RuntimeError("Native worker omitted canonical_vocoder_stream mode metadata")
        if contract["enabled"] is not requested:
            raise RuntimeError("Native worker canonical-vocoder stream mode differs from the requested mode")
        if contract.get("quality_qualified") is not False:
            raise RuntimeError("Native worker must keep canonical-vocoder streaming unqualified")
        if requested:
            for name, expected in _CANONICAL_VOCODER_STREAM_CONTRACT.items():
                actual = contract.get(name)
                matches = actual is expected if type(expected) is bool else type(actual) is int and actual == expected
                if not matches:
                    raise RuntimeError(f"Native worker canonical-vocoder stream metadata mismatch: {name}")
        return dict(contract)

    def _canonical_stream_metadata(self) -> dict[str, Any]:
        worker = self._canonical_stream_worker_capability
        enabled = self._canonical_vocoder_stream
        return {
            "enabled": enabled,
            "quality_qualified": False,
            "semantic_eos_before_first_pcm": (
                worker.get("semantic_eos_before_pcm") if enabled and worker is not None else None
            ),
            "cfm_complete_before_first_pcm": (
                worker.get("cfm_complete_before_pcm") if enabled and worker is not None else None
            ),
            "incremental_vocoder": bool(enabled and worker is not None),
            "early_ar_audio": worker.get("early_ar_audio") if enabled and worker is not None else None,
            "vocoder_core_frames": worker.get("vocoder_core_frames") if enabled and worker is not None else None,
            "vocoder_halo_frames": worker.get("vocoder_halo_frames") if enabled and worker is not None else None,
            "vocoder_bucket_frames": worker.get("vocoder_bucket_frames") if enabled and worker is not None else None,
            "onset_trim": {"enabled": self._onset_trim,
                           "policy": "conservative-pcm-head-v3-energy", "activity_rms_dbfs": -65,
                           "activity_peak_dbfs": -55, "guard_ms": 40, "maximum_cut_ms": self._onset_trim_max_ms,
                           "removed_prefix_peak_limit_dbfs": -65,
                           "cut_resolution_ms": 1,
                           "maximum_native_utterance_fraction": .025,
                           "scope": "initial PCM of each stream request; first native utterance bounds duration; complete WAV and reference unchanged"},
        }

    def _bootstrap_geometry(self) -> dict[str, int] | None:
        return dict(_BOOTSTRAP_STREAM_GEOMETRY) if self._bootstrap_stream_v2 else None

    def _validate_worker_result(self, result: dict[str, Any], requested_top_k: int) -> None:
        self._mlx_memory_last = result.get("memory")
        reported_precision = result.get("precision")
        if reported_precision != ("float16" if self._precision == "FP16" else "float32") and not (self._precision == "FP32" and reported_precision is None):
            raise RuntimeError("Native worker result precision differs from requested precision")
        reported_top_k = result.get("top_k")
        if reported_top_k is None:
            if requested_top_k != 15:
                raise RuntimeError("Native worker omitted top_k from an explicit top_k=1 result")
        elif type(reported_top_k) is not int or reported_top_k != requested_top_k:
            raise RuntimeError("Native worker result top_k does not match the request")
        policy = result.get("cfm_noise_policy")
        if self._coordinate_noise_v2:
            if policy != _COORDINATE_CFM_NOISE_POLICY:
                raise RuntimeError("Native worker result omitted the requested coordinate CFM noise policy")
        elif policy not in (None, _LEGACY_CFM_NOISE_POLICY):
            raise RuntimeError("Native worker result CFM noise policy differs from the default profile")
        if result.get("quality_qualified") not in (None, False):
            raise RuntimeError("Native research worker must not claim quality qualification")

    def _sampling_capabilities(self) -> dict[str, Any]:
        return {
            "top_k": 15,
            "top_k_supported": ({"minimum": 1, "maximum": 50} if self._request_sampling
                                else list(self._supported_top_k)),
            "worker_top_k_supported": ({"minimum": 1, "maximum": 50} if self._request_sampling
                                       else list(self._worker_supported_top_k)),
            "top_k_1_requires_research_opt_in": self._top_k_1_requires_research_opt_in,
            "top_k_1_research_opt_in_enabled": (
                self._top_k_1_requires_research_opt_in and self._allow_unqualified_stream
            ),
            "top_p": 1.0,
            "temperature": 1.0,
            "request_sampling": self._request_sampling,
            "ranges": ({"top_k": [1, 50], "top_p": "(0, 1]", "temperature": "> 0"} if self._request_sampling
                       else {"top_k": list(self._supported_top_k), "top_p": [1.0], "temperature": [1.0]}),
            "repetition_penalty": 1.35,
            "speed": {"supported": [1.0], "reason": "no speed input in the native engine (same as Windows)"},
            "noise_scale": {"supported": [0.5], "applied": False,
                            "reason": "official V5 Turbo has no VITS noise_scale input (same as Windows V5)"},
            "noise_scale_compatibility_value": 0.5,
            "cfm_temperature": 0.875,
            "cfm_steps": 4,
            "cfg_rate": 0.0,
        }

    @staticmethod
    def _required_path(name: str, *, directory: bool = False) -> Path:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"Missing required native runtime setting: {name}")
        path = Path(value).expanduser().resolve()
        if directory and not path.is_dir():
            raise RuntimeError(f"Native runtime directory is unavailable: {name}")
        if not directory and not path.is_file():
            raise RuntimeError(f"Native runtime file is unavailable: {name}")
        return path

    def load(self) -> None:
        if self._client is not None:
            if not self._client.ready:
                raise RuntimeError("Native worker is not ready")
            return
        binary = self._required_path("ANIFLIVE_TTS_NATIVE_WORKER")
        models = self._required_path("ANIFLIVE_TTS_NATIVE_MODEL_DIR", directory=True)
        metal = self._required_path("ANIFLIVE_TTS_NATIVE_METAL_LIB", directory=True)
        frontend = self._required_path("ANIFLIVE_TTS_NATIVE_FRONTEND_SCRIPT")
        if not (self.settings.source_dir / "GPT_SoVITS").is_dir():
            raise RuntimeError("Official GPT-SoVITS source directory is unavailable")
        references: Path | None = None
        if not self.settings.dynamic_voices:
            # Package mode: a fixed voice is part of the package.
            references = self._required_path("ANIFLIVE_TTS_NATIVE_REFERENCE_INPUTS", directory=True)
            if not self.settings.reference_wav.is_file():
                raise RuntimeError("Fixed reference audio is unavailable")
            for name in ("waveform.npy", "refer.npy", "reference-mel.npy", "reference-phones.npy", "reference.txt"):
                if not (references / name).is_file():
                    raise RuntimeError(f"Prepared fixed reference is missing {name}")
            transcript = (references / "reference.txt").read_text(encoding="utf-8").strip("\n")
            if not transcript or transcript != api.REFERENCE_TEXT.strip("\n") or api.REFERENCE_LANGUAGE not in {"ja", "all_ja"}:
                raise RuntimeError("Prepared reference transcript/language does not match the fixed Japanese voice")
        manifest_path = self.settings.engine_dir / "engine-manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("Native engine manifest is unavailable")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fingerprint = manifest.get("fingerprint") if isinstance(manifest, dict) else None
        if (not isinstance(manifest, dict)
                or manifest.get("kind") != "aniflive-tts-gsv-v5turbo-mlx-metal"
                or not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in fingerprint)):
            raise RuntimeError("Native engine manifest identity is invalid")
        cache_value = os.environ.get("ANIFLIVE_TTS_NATIVE_CACHE_DIR")
        if not cache_value:
            raise RuntimeError("Missing required native runtime setting: ANIFLIVE_TTS_NATIVE_CACHE_DIR")
        cache = Path(cache_value).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        enable_mandarin = os.environ.get("ANIFLIVE_TTS_NATIVE_ENABLE_MANDARIN", "0") == "1"
        if enable_mandarin and self._frontend_mode != "persistent-python":
            raise RuntimeError("Native Mandarin currently requires persistent-python text processing")
        persistent_frontend = None
        if self._frontend_mode == "persistent-python":
            from .backend.macos_frontend import NativePythonFrontend
            persistent_frontend = NativePythonFrontend(
                frontend, self.settings.source_dir, cache_metadata=self._frontend_metadata_cache
            )
        from .backend.macos_worker import NativeWorkerClient
        client = NativeWorkerClient(binary, models, cache, metal, cache / "native-worker.stderr.log")
        try:
            ready = client.start()
            if not client.ready:
                raise RuntimeError("Native worker did not become ready")
            if ready.get("backend") != "mlx-metal" or not ready.get("operational"):
                raise RuntimeError("Native worker did not report an operational MLX/Metal backend")
            self._record_worker_contract(ready)
            self._canonical_stream_worker_capability = self._validate_canonical_stream_contract(
                ready.get("capabilities", {}), self._canonical_vocoder_stream
            )
            if self._ar_mixed and not ready.get("capabilities", {}).get("ar_mixed_gemv"):
                raise RuntimeError("Native worker does not support lossless-storage GEMV")
            if self._ar_optimized:
                caps = ready.get("capabilities", {})
                if (caps.get("ar_optimized") is not True or caps.get("ar_batch_size") != 4
                        or caps.get("ar_gemv_rows") != 1 or caps.get("ar_fused_epilogue") is not True):
                    raise RuntimeError("Native worker did not enable the requested optimized AR path")
            if self._ar_compiled and not ready.get("capabilities", {}).get("ar_compiled"):
                raise RuntimeError("Native worker does not support compiled AR")
            if self._ar_static:
                caps = ready.get("capabilities", {})
                cache_info = caps.get("ar_static_cache")
                if (caps.get("ar_static_decode") is not True or not isinstance(cache_info, dict)
                        or cache_info.get("capacity") != 1024):
                    raise RuntimeError("Native worker did not enable bounded static AR")
            if ready.get("capabilities", {}).get("ar_residual_norm_fused", False) is not self._ar_residual_norm:
                raise RuntimeError("Native worker residual norm fusion differs from the requested mode")
            if self._cfm_stage and not ready.get("capabilities", {}).get("cfm_isolated_tf32"):
                raise RuntimeError("Native worker does not support the isolated CFM stage")
            if self._cfm_compiled and not ready.get("capabilities", {}).get("cfm_compiled"):
                raise RuntimeError("Native worker does not support compiled CFM")
            cfm_caps = ready.get("capabilities", {})
            if (self._onset_trim and cfm_caps.get("canonical_vocoder_stream", {}).get(
                    "original_total_samples_in_pcm") is not True):
                raise RuntimeError("PCM onset trim requires a worker with exact canonical sample-count metadata")
            if (cfm_caps.get("cfm_batched_qkv", False) is not self._cfm_batched_qkv
                    or cfm_caps.get("cfm_packed_qkv_blocks", 0) != (22 if self._cfm_batched_qkv else 0)):
                raise RuntimeError("Native worker CFM QKV layout differs from the requested mode")
            if self._ar_combined and not ready.get("capabilities", {}).get("ar_combined"):
                raise RuntimeError("Native worker does not support combined AR decisions")
            if self._cfm_prepare and not ready.get("capabilities", {}).get("cfm_prepared"):
                raise RuntimeError("Native worker does not support prepared CFM")
            if self._complete_mode == "cached" and not ready.get("capabilities", {}).get("synthesize_cached"):
                raise RuntimeError("Native worker does not support cached complete synthesis")
            if enable_mandarin:
                if not all(ready.get("capabilities", {}).get(name) for name in ("frontend_g2pw", "frontend_bert")):
                    raise RuntimeError("Native Mandarin package lacks required neural components")
                from .backend.macos_mandarin import MandarinBridge
                bridge = MandarinBridge(self.settings.source_dir, models / "g2pw/assets",
                                        models / "chinese-roberta-wwm-ext-large", client)
                persistent_frontend.enable_mandarin(bridge)
        except BaseException:
            try:
                client.close()
            except Exception:
                api.LOGGER.exception("Native worker startup cleanup failed")
            raise
        self._client = client
        self._persistent_frontend = persistent_frontend
        self._mandarin_enabled = enable_mandarin
        self._engine = client
        self._sample_rate = 48000
        self._reference_inputs = references
        self._frontend_script = frontend
        self._cache_dir = cache
        self._engine_manifest = manifest
        self._load_native_expression_references()

    def _load_native_expression_references(self) -> None:
        """Prepare the package-owned expression references at activation (v1.3 contract)."""
        self._package_bank = None
        self._expression_catalog = None
        if self.settings.dynamic_voices or self._reference_inputs is None:
            return
        identity = NativeReferenceHandle(self._reference_inputs, self.settings.reference_wav,
                                         self._reference_inputs / "reference.txt", "ja")
        phones = np.load(self._reference_inputs / "reference-phones.npy", allow_pickle=False).reshape(-1).tolist()
        bank = PreparedReferenceBank(identity_id="default")
        bank.add(PreparedReference(prompt_semantic=identity, spectrogram=identity, speaker_embedding=identity,
                                   phones=phones, bert=None, id="default"))
        catalog = None
        if self.settings.profile_manifest.is_file():
            from .expression import load_expression_catalog
            profile = json.loads(self.settings.profile_manifest.read_text(encoding="utf-8"))
            catalog = load_expression_catalog(profile_dir=self.settings.profile_manifest.parent,
                                              profile_manifest=profile)
        if catalog is not None:
            identity_audio = self.settings.reference_wav.resolve()
            for descriptor in catalog.profiles:
                if (descriptor.reference_audio.resolve() == identity_audio
                        and descriptor.reference_text == api.REFERENCE_TEXT
                        and descriptor.reference_language == api.REFERENCE_LANGUAGE.removeprefix("all_")):
                    bank.alias(alias_id=descriptor.id, reference_id=bank.identity_id)
                    continue
                bank.add(self.prepare_dynamic_reference(reference_id=descriptor.id,
                    reference_wav=descriptor.reference_audio, reference_text=descriptor.reference_text,
                    reference_language=descriptor.reference_language))
            api.LOGGER.info("Prepared %d native expression references", len(catalog.profiles))
        self._package_bank = bank
        self._expression_catalog = catalog

    def unload(self) -> None:
        self.cancel_active_stream()
        with self._inference_lock:
            client = self._client
            self._client = None
            self._engine = None
            self._sample_rate = None
            self._persistent_frontend = None
            self._mandarin_enabled = False
            self._engine_manifest = {}
            if client is not None:
                client.close()

    def _validate_native_options(self, options: api.SynthesisOptions) -> str:
        if not self.ready:
            raise RuntimeError("Native MLX worker is not ready")
        # Resolves every segment's conditioning (voice/package expression,
        # intensity, policy) and fails before inference if any is unavailable.
        self._segment_conditionings(options)
        # Windows-legal AR sampling (1 <= top_k <= 50, 0 < top_p <= 1,
        # temperature > 0) runs in the native sampler. A worker without
        # request-level sampling keeps its narrower, explicit contract.
        if type(options.top_k) is not int or not 1 <= options.top_k <= 50:
            raise api.RequestError("top_k must be an integer between 1 and 50")
        if not (math.isfinite(options.top_p) and 0.0 < options.top_p <= 1.0):
            raise api.RequestError("top_p must be in (0, 1]")
        if not (math.isfinite(options.temperature) and options.temperature > 0.0):
            raise api.RequestError("temperature must be positive")
        if not self._request_sampling:
            if options.top_k not in self._supported_top_k:
                if options.top_k == 1 and 1 in self._worker_supported_top_k:
                    raise api.RequestError("top_k=1 requires ANIFLIVE_TTS_ALLOW_UNQUALIFIED_STREAM=1")
                raise api.RequestError("The active native worker supports only the default top_k=15")
            if not math.isclose(options.top_p, 1.0, abs_tol=1e-8) or not math.isclose(options.temperature, 1.0, abs_tol=1e-8):
                raise api.RequestError("The active native worker supports only top_p=1 and temperature=1")
        # Same as Windows: no engine has a speed input, so speed stays 1.0.
        if not math.isclose(options.speed, 1.0, abs_tol=1e-6):
            raise api.RequestError("speed must be 1.0 for the current native V5 backend")
        # Same as the Windows V5 backend: V5 Turbo has no VITS noise_scale
        # input (official TTS.py passes none), so only the neutral 0.5 is valid.
        if not math.isclose(options.noise_scale, 0.5, abs_tol=1e-6):
            raise api.RequestError("V5 Turbo supports only neutral noise_scale=0.5")
        if not math.isfinite(options.pause_length) or not 0.0 <= options.pause_length <= 10.0:
            raise api.RequestError("pause_length must be between 0 and 10 seconds")
        if options.seed < -1 or options.seed > (1 << 64) - 1:
            raise api.RequestError("seed must be -1 or an unsigned 64-bit integer")
        # The request language is the Windows-normalized official mode
        # (all_ja, ja, en, ..., auto, auto_yue) and reaches the official
        # TextPreprocessor unchanged.
        lang = options.text_language
        if lang not in self._languages():
            if lang in {"zh", "all_zh"}:
                raise api.RequestError("Mandarin requires the enabled native G2PW/BERT frontend")
            raise api.RequestError("Unsupported or unavailable native frontend language")
        if self._persistent_frontend is None and lang not in {"all_ja", "en", "all_ko", "all_yue"}:
            raise api.RequestError("Mixed and automatic language modes require the persistent native frontend")
        return lang

    def _languages(self) -> list[str]:
        modes = ["all_ja", "ja", "en", "all_ko", "ko", "all_yue", "yue", "auto", "auto_yue"]
        return modes + (["all_zh", "zh"] if self._mandarin_enabled else [])

    def _worker_supports_conditioning_components(self) -> bool:
        metadata = getattr(self._client, "metadata", None) if self._client is not None else None
        capabilities = metadata.get("capabilities") if isinstance(metadata, dict) else None
        components = capabilities.get("conditioning_components") if isinstance(capabilities, dict) else None
        return isinstance(components, list) and all(
            component in components for component in ("semantic", "style", "acoustic_prompt")
        )

    def _supported_expression_policies(self) -> list[str]:
        return ([policy.value for policy in ConditioningPolicy]
                if self._worker_supports_conditioning_components()
                else [ConditioningPolicy.FULL_SWITCH.value])

    def expression_metadata(self) -> dict[str, Any]:
        metadata = super().expression_metadata()
        metadata["policies"] = self._supported_expression_policies()
        if metadata["enabled"]:
            metadata["preferred_policy_supported"] = metadata.get("preferred_policy") in metadata["policies"]
        return metadata

    def validate_expression(self, options: api.SynthesisOptions) -> None:
        super().validate_expression(options)
        if not self._worker_supports_conditioning_components():
            selections = options.expression_segments or (options,)
            for selection in selections:
                enabled = selection.enabled if options.expression_segments else selection.expression_enabled
                policy = selection.policy if options.expression_segments else selection.expression_policy
                if enabled and policy is not ConditioningPolicy.FULL_SWITCH:
                    raise api.RequestError("The active native worker supports only full-switch conditioning")

    def _identity_handle(self) -> NativeReferenceHandle:
        if self._reference_inputs is None:
            raise api.RequestError("This shared base model has no fixed voice; select a registered voice_profile")
        return NativeReferenceHandle(self._reference_inputs, self.settings.reference_wav,
                                     self._reference_inputs / "reference.txt", "ja")

    def _bundle_for(self, selection: Any, options: api.SynthesisOptions) -> Any:
        """Official ConditioningBundle for one segment (Windows contract)."""
        if options.voice_binding is not None:
            return options.voice_binding.conditioning(enabled=selection.enabled, profile=selection.profile,
                intensity=selection.intensity, language=options.text_language, policy=selection.policy)
        if options.voice_profile and options.voice_profile != api.VOICE_ID:
            raise VoiceError("reference_not_ready", "Select a prepared native voice", 409)
        bank = self._package_bank
        if bank is None:
            if selection.enabled:
                raise api.RequestError("Controlled expression is unavailable for the active model package")
            identity = self._identity_handle()
            return ConditioningBundle.full_switch(PreparedReference(prompt_semantic=identity, spectrogram=identity,
                speaker_embedding=identity, phones=[0], bert=None, id="default"))
        if not selection.enabled:
            return bank.conditioning(reference_id=bank.identity_id, policy=ConditioningPolicy.FULL_SWITCH)
        if self._expression_catalog is None:
            raise api.RequestError("Controlled expression is unavailable for the active model package")
        selected = self._expression_catalog.select(
            profile=selection.profile or self._expression_catalog.default_profile,
            intensity=selection.intensity, language=options.text_language.removeprefix("all_"))
        return bank.conditioning(reference_id=selected.id, policy=selection.policy)

    @staticmethod
    def _components(bundle: Any) -> tuple[NativeReferenceHandle, NativeReferenceHandle, NativeReferenceHandle]:
        """V5 conditioning components of a bundle.

        semantic -> AR semantic prompt (prompt codes, reference phones/BERT);
        spectrogram -> global style vector ge (reference spectrum);
        speaker_embedding -> CFM in-context acoustic prompt (reference features
        and mel). V5 Turbo has no separate SV embedding; the CFM acoustic prompt
        is the component that carries the reference voice identity, so it takes
        the speaker-embedding role of the shared policy contract.
        """
        semantic = bundle.semantic.prompt_semantic
        style = bundle.spectrogram
        prompt = bundle.speaker_embedding
        for value in (semantic, style, prompt):
            if not isinstance(value, NativeReferenceHandle):
                raise api.RequestError("Voice conditioning does not belong to the native runtime")
        return semantic, style, prompt

    def _segment_conditionings(self, options: api.SynthesisOptions) -> list[tuple[str, Any, dict[str, Any]]]:
        plan = self._segment_plan(options)
        result = []
        for segment in plan:
            if (segment.enabled and segment.policy is not ConditioningPolicy.FULL_SWITCH
                    and not self._worker_supports_conditioning_components()):
                raise api.RequestError("The active native worker supports only full-switch conditioning")
            bundle = self._bundle_for(segment, options)
            semantic, style, prompt = self._components(bundle)
            meta = {"expression_id": bundle.expression_id, "policy": bundle.policy.value,
                    "enabled": segment.enabled, "profile": segment.profile, "intensity": segment.intensity}
            result.append((segment.text, (semantic, style, prompt), meta))
        if not result:
            raise api.RequestError("Text produced no synthesis segments")
        return result

    @staticmethod
    def _source(handle: NativeReferenceHandle, semantic: NativeReferenceHandle) -> tuple[Path, Path, Path] | None:
        if (handle.inputs, handle.audio, handle.transcript) == (semantic.inputs, semantic.audio, semantic.transcript):
            return None
        return (handle.inputs, handle.audio, handle.transcript)

    def _segments_for(self, options: api.SynthesisOptions) -> list[str]:
        segments = self._segments(options)
        if not segments:
            raise api.RequestError("Text produced no synthesis segments")
        return segments

    def _run_frontend(self, text: str, language: str, directory: Path,
                      cancelled: threading.Event | None = None, *,
                      reference: NativeReferenceHandle) -> float:
        if self._frontend_script is None:
            raise RuntimeError("Native frontend is not configured")
        start = time.perf_counter()
        if self._persistent_frontend is not None:
            if cancelled is not None and cancelled.is_set():
                raise _Cancelled()
            self._persistent_frontend.prepare(reference_transcript=reference.transcript,
                reference_language=reference.language, text=text, language=language, output=directory)
            if cancelled is not None and cancelled.is_set():
                raise _Cancelled()
        else:
            command = [sys.executable, str(self._frontend_script), "--official", str(self.settings.source_dir),
                       "--reference-transcript", str(reference.transcript),
                       "--reference-language", reference.language, "--text", text,
                       # The one-shot helper names monolingual modes by base code.
                       "--language", {"all_ja": "ja", "en": "en", "all_ko": "ko", "all_yue": "yue"}[language],
                       "--output", str(directory)]
            try:
                timeout = float(os.environ.get("ANIFLIVE_TTS_FRONTEND_TIMEOUT_SECONDS", "120"))
            except ValueError as error:
                raise RuntimeError("Invalid frontend timeout") from error
            if not math.isfinite(timeout) or timeout <= 0:
                raise RuntimeError("Frontend timeout must be positive")
            start = time.perf_counter()
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                while True:
                    if cancelled is not None and cancelled.is_set():
                        raise _Cancelled()
                    remaining = timeout - (time.perf_counter() - start)
                    if remaining <= 0:
                        raise RuntimeError("Text frontend timed out")
                    try:
                        _, stderr = process.communicate(timeout=min(0.2, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue
                if process.returncode != 0:
                    raise RuntimeError(f"Text frontend failed: {stderr[-2000:]}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.communicate(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
        elapsed = time.perf_counter() - start
        for name in ("reference-phones.npy", "target-phones.npy", "phones.npy", "bert.npy"):
            if not (directory / name).is_file():
                raise RuntimeError(f"Text frontend did not produce {name}")
        expected = np.load(reference.inputs / "reference-phones.npy", allow_pickle=False)
        actual = np.load(directory / "reference-phones.npy", allow_pickle=False)
        if not np.array_equal(actual, expected):
            raise RuntimeError("Text frontend reference phones differ from the prepared voice")
        for name in ("waveform.npy", "refer.npy", "reference-mel.npy"):
            shutil.copyfile(reference.inputs / name, directory / name)
        with self._active_guard:
            self._last_frontend_seconds = elapsed
            self._last_frontend_segments += 1
        return elapsed

    @staticmethod
    def _read_wav(path: Path) -> bytes:
        with wave.open(str(path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 48000 or wav.getcomptype() != "NONE":
                raise RuntimeError("Native worker returned an invalid 48 kHz mono PCM16 WAV")
            count = wav.getnframes()
            frames = wav.readframes(count)
        if not frames or len(frames) != count * 2:
            raise RuntimeError("Native worker returned empty or malformed audio")
        return frames

    @staticmethod
    def _wav(pcm: bytes) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            wav.writeframes(pcm)
        return output.getvalue()

    def synthesize(self, options: api.SynthesisOptions) -> api.SynthesisResult:
        language = self._validate_native_options(options)
        planned = self._segment_conditionings(options)
        if self._complete_mode != "cached" and any(
            self._source(style, semantic) is not None or self._source(prompt, semantic) is not None
            for _, (semantic, style, prompt), _ in planned
        ):
            raise api.RequestError("Separate conditioning requires cached complete synthesis")
        lease = self._begin_request()
        cancel_event = threading.Event()
        with self._active_guard:
            self._active_complete_cancel = cancel_event
        try:
            return self._synthesize_planned(options, language, planned, lease, cancel_event)
        except _Cancelled as error:
            raise SynthesisCancelled("Synthesis was cancelled; no audio was returned") from error
        finally:
            with self._active_guard:
                if getattr(self, "_active_complete_cancel", None) is cancel_event:
                    self._active_complete_cancel = None

    def _synthesize_planned(self, options: api.SynthesisOptions, language: str, planned: list[Any],
                            lease: Any, cancel_event: threading.Event) -> api.SynthesisResult:
        started = time.perf_counter()
        conditioning_report: list[dict[str, Any]] = []
        seed = self._effective_seed(options.seed)
        frontend_seconds = 0.0
        primary_error: BaseException | None = None
        try:
            with self._inference_lock:
                pcm_parts: list[bytes] = []
                for index, (text, (reference, style, prompt), meta) in enumerate(planned):
                    with tempfile.TemporaryDirectory(prefix="aniflive-v5-") as temp:
                        root = Path(temp)
                        frontend = root / "input"
                        frontend.mkdir()
                        if cancel_event.is_set():
                            raise _Cancelled()
                        frontend_seconds += self._run_frontend(text, language, frontend, cancel_event,
                                                               reference=reference)
                        if self._client is None:
                            raise RuntimeError("Native worker closed during synthesis")
                        style_source, prompt_source = self._source(style, reference), self._source(prompt, reference)
                        if cancel_event.is_set():
                            raise _Cancelled()
                        if self._complete_mode == "cached":
                            result = self._client_complete(self._client.synthesize_cached,frontend, root / "output", seed,
                                audio=reference.audio, transcript=reference.transcript,
                                top_k=options.top_k, top_p=options.top_p, temperature=options.temperature, style=style_source, prompt=prompt_source)
                        else:
                            if style_source is not None or prompt_source is not None:
                                raise RuntimeError("Mixed conditioning requires cached complete synthesis")
                            result = self._client_complete(self._client.synthesize, frontend, root / "output", seed,
                                top_k=options.top_k, top_p=options.top_p, temperature=options.temperature)
                        if cancel_event.is_set():
                            # Cancel acknowledged after the worker finished:
                            # still no audio for this request.
                            raise _Cancelled()
                        conditioning_report.append({**meta, "segment_index": index,
                                                    "worker_conditioning": result.get("conditioning"),
                                                    "frontend": self._frontend_summary(frontend),
                                                    "sampling": self._sampling_summary(result)})
                        self._validate_worker_result(result, options.top_k)
                        result_path = Path(result["result_path"]).resolve(strict=True)
                        if not result_path.is_relative_to((root / "output").resolve()):
                            raise RuntimeError("Native worker returned an unexpected output path")
                        if index:
                            pcm_parts.append(b"\x00\x00" * round(options.pause_length * 48000))
                        pcm_parts.append(self._read_wav(result_path))
                pcm = b"".join(pcm_parts)
                if not pcm:
                    raise RuntimeError("Native worker produced no audio")
                lease.ensure_valid()
                with self._active_guard:
                    self._last_conditioning = conditioning_report
                return api.SynthesisResult(wav=self._wav(pcm), sample_rate=48000,
                    output_samples=len(pcm) // 2, segments=len(planned),
                    elapsed_seconds=time.perf_counter() - started,
                    profile={"text_processing_seconds": frontend_seconds,
                             "frontend_wall_seconds": frontend_seconds, "actual_seed": seed})
        except BaseException as error:
            primary_error = error
            raise
        finally:
            cleanup_error = self._end_request(lease)
            if cleanup_error is not None:
                if primary_error is None:
                    raise RuntimeError("Inference resource cleanup failed") from cleanup_error
                api.LOGGER.error("Native inference resource cleanup also failed", exc_info=(type(cleanup_error), cleanup_error, cleanup_error.__traceback__))

    @staticmethod
    def _frontend_summary(directory: Path) -> dict[str, Any] | None:
        """Official language mode and phone count actually used (for /health)."""
        try:
            fixture = json.loads((directory / "fixture.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return {"official_language": fixture.get("official_language"),
                "target_phone_count": fixture.get("target_phone_count"),
                "target_normalized_text": fixture.get("target_normalized_text"),
                "native_mandarin_bert": fixture.get("native_mandarin_neural")}

    @staticmethod
    def _sampling_summary(result: dict[str, Any]) -> dict[str, Any]:
        return {key: result.get(key) for key in ("top_k", "top_p", "temperature")}

    @staticmethod
    def _client_complete(call: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        from .backend.macos_worker import NativeWorkerError
        try:
            return call(*args, **kwargs)
        except NativeWorkerError as error:
            if error.code == "cancelled":
                raise _Cancelled() from error
            raise

    def stream_pcm(self, options: api.SynthesisOptions, *, continuity_input: Any = None,
                   continuity_capture: Any = None, continuity_policy: str = "A") -> Iterator[bytes]:
        if not self._product_stream:
            raise api.RequestError("Native PCM streaming needs the canonical vocoder stream (ANIFLIVE_TTS_NATIVE_CANONICAL_STREAM=1)")
        if continuity_policy != "A" or continuity_input is not None or continuity_capture is not None:
            raise api.RequestError("Cross-segment continuity is not supported by the macOS prototype")
        language = self._validate_native_options(options)
        planned = self._segment_conditionings(options)
        lease = self._begin_request()
        cancelled = threading.Event()
        conditioning_report: list[dict[str, Any]] = []
        self._last_onset_trim = None
        with self._active_guard:
            self._active_stream_cancel = cancelled
        seed = self._effective_seed(options.seed)
        chunks: queue.Queue[bytes | BaseException | None] = queue.Queue(maxsize=3)

        def cancel() -> None:
            cancelled.set()
            # The client cancel only writes the request; keep ownership locked
            # until it is sent so it cannot target a replacement stream.
            with self._active_guard:
                if self._active_stream_cancel is cancelled and self._client is not None:
                    self._client.cancel_active_stream()

        def abort() -> None:
            with self._active_guard:
                if self._active_stream_cancel is cancelled and self._client is not None:
                    self._client.close()

        def put(item: bytes) -> bool:
            while not cancelled.is_set():
                try:
                    chunks.put(item, timeout=0.2)
                    return True
                except queue.Full:
                    continue
            return False

        def finish(item: BaseException | None) -> None:
            while True:
                try:
                    chunks.put(item, timeout=0.2)
                    return
                except queue.Full:
                    if cancelled.is_set():
                        try:
                            chunks.get_nowait()
                        except queue.Empty:
                            pass

        def produce() -> None:
            error: BaseException | None = None
            try:
                with self._inference_lock:
                    for index, (text, (reference, style, prompt), meta) in enumerate(planned):
                        if cancelled.is_set():
                            break
                        with tempfile.TemporaryDirectory(prefix="aniflive-v5-stream-") as temp:
                            root = Path(temp)
                            frontend = root / "input"
                            frontend.mkdir()
                            self._run_frontend(text, language, frontend, cancelled, reference=reference)
                            if cancelled.is_set():
                                break
                            if self._client is None:
                                raise RuntimeError("Native worker closed during streaming")
                            if index and options.pause_length and not put(b"\x00\x00" * round(options.pause_length * 48000)):
                                break
                            stream = self._client.stream(frontend, root / "output", seed,
                                audio=reference.audio, transcript=reference.transcript,
                                geometry=self._bootstrap_geometry(), top_k=options.top_k, top_p=options.top_p, temperature=options.temperature,
                                style=self._source(style, reference), prompt=self._source(prompt, reference))
                            try:
                                first_chunk = True
                                for chunk in stream:
                                    if chunk.sample_rate != 48000 or chunk.channels != 1 or chunk.sample_format != "s16le":
                                        raise RuntimeError("Native worker emitted incompatible PCM metadata")
                                    payload = chunk.pcm
                                    if not isinstance(payload, bytes) or len(payload) % 2:
                                        raise RuntimeError("Native worker emitted malformed PCM16")
                                    if self._onset_trim and index == 0 and first_chunk:
                                        from .backend.pcm_onset import trim_initial_pcm
                                        raw_first_hash = hashlib.sha256(payload).hexdigest()
                                        payload, decision = trim_initial_pcm(payload, chunk.original_total_samples,
                                                                             self._onset_trim_max_ms)
                                        self._last_onset_trim = {**decision, "request_id": stream.request_id,
                                            "raw_first_chunk_sha256": raw_first_hash,
                                            "trimmed_first_chunk_sha256": hashlib.sha256(payload).hexdigest(),
                                            "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "seed": seed}
                                    first_chunk = False
                                    if payload and not put(payload):
                                        break
                                if not cancelled.is_set() and not stream.cancelled:
                                    stream_result = stream.result
                                    if not isinstance(stream_result, dict):
                                        raise RuntimeError("Native worker omitted terminal stream metadata")
                                    self._validate_worker_result(stream_result, options.top_k)
                                    conditioning_report.append({**meta, "segment_index": index,
                                        "worker_conditioning": stream_result.get("conditioning"),
                                        "frontend": self._frontend_summary(frontend),
                                        "sampling": self._sampling_summary(stream_result)})
                            finally:
                                stream.close()
                    lease.ensure_valid()
                    with self._active_guard:
                        self._last_conditioning = conditioning_report
            except _Cancelled:
                pass
            except BaseException as exc:
                error = exc
                api.LOGGER.error("Native streaming failed", exc_info=(type(exc), exc, exc.__traceback__))
            finally:
                with self._active_guard:
                    if self._active_stream_cancel is cancelled:
                        self._active_stream_cancel = None
                cleanup_error = self._end_request(lease)
                if cleanup_error is not None:
                    if error is None:
                        error = RuntimeError("Inference resource cleanup failed")
                        error.__cause__ = cleanup_error
                    else:
                        api.LOGGER.error("Native inference resource cleanup also failed", exc_info=(type(cleanup_error), cleanup_error, cleanup_error.__traceback__))
                finish(error)

        thread = threading.Thread(target=produce, name="aniflive-tts-mlx-stream", daemon=True)
        try:
            thread.start()
        except BaseException:
            with self._active_guard:
                if self._active_stream_cancel is cancelled:
                    self._active_stream_cancel = None
            self._end_request(lease)
            raise
        return _PCMIterator(chunks, thread, cancel, abort)

    def cancel_active_stream(self) -> bool:
        """Cancel the active PCM stream or complete-WAV synthesis."""
        with self._active_guard:
            cancelled = self._active_stream_cancel or getattr(self, "_active_complete_cancel", None)
            if cancelled is None:
                return False
            cancelled.set()
            if self._client is not None:
                self._client.cancel_active_stream()
            return True

    def prepare_dynamic_reference(self, *, reference_id: str, reference_wav: Path,
                                  reference_text: str, reference_language: str) -> PreparedReference:
        if not self.ready or self._cache_dir is None or self._frontend_script is None:
            raise VoiceError("engine_not_ready", "The native inference engine is not ready", 503)
        language = reference_language.removeprefix("all_")
        if language not in {"ja", "en", "yue", "ko", "zh"}:
            raise VoiceError("unsupported_reference_language", "Native references currently support ja, en, yue, ko and zh", 400)
        if language == "zh" and (not self._mandarin_enabled or self._persistent_frontend is None):
            raise VoiceError("unsupported_reference_language", "Mandarin references require the enabled native G2PW/BERT frontend", 400)
        helper = self._frontend_script.with_name("prepare_reference_inputs.py")
        if not helper.is_file():
            raise RuntimeError("Native reference preprocessing helper is unavailable")
        audio = Path(reference_wav).resolve(strict=True)
        identity = {"audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                    "transcript": reference_text, "language": language,
                    "engine": self._engine_manifest["fingerprint"],
                    "dsp_helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
                    "frontend_sha256": hashlib.sha256(self._frontend_script.read_bytes()).hexdigest()}
        if language == "zh":
            identity["mandarin_frontend_sources"] = self._persistent_frontend.mandarin_reference_source_identity()
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        root = self._cache_dir / "registered-reference-inputs"
        root.mkdir(parents=True, exist_ok=True)
        inputs = root / key
        lease = self._begin_request()
        primary_error: BaseException | None = None
        try:
            with self._inference_lock:
                if not inputs.exists():
                    with tempfile.TemporaryDirectory(prefix="registration-", dir=root) as temp:
                        staging = Path(temp)
                        transcript = staging / "transcript.txt"
                        transcript.write_text(reference_text, encoding="utf-8")
                        generated = staging / "inputs"
                        command = [sys.executable, str(helper), "--official", str(self.settings.source_dir),
                            "--audio", str(audio), "--transcript", str(transcript),
                            "--reference-language", language, "--output", str(generated)]
                        if self._precision == "FP16":
                            command.extend(["--dsp-device", "mps"])
                        if language == "zh":
                            command.append("--dsp-only")
                        result = subprocess.run(command,
                            capture_output=True, text=True, timeout=120)
                        if result.returncode != 0:
                            raise RuntimeError(f"Reference preprocessing failed: {result.stderr[-2000:]}")
                        if language == "zh":
                            mandarin_frontend = self._persistent_frontend.prepare_mandarin_reference(
                                text=reference_text, output=generated
                            )
                            if mandarin_frontend.get("source_files_sha256") != identity["mandarin_frontend_sources"]:
                                raise RuntimeError("Mandarin frontend sources changed during reference preparation")
                            dsp_manifest_path = generated / "fixture.json"
                            dsp_manifest = json.loads(dsp_manifest_path.read_text(encoding="utf-8"))
                            dsp_manifest["reference_text_frontend"] = "prepared_by_attached_native_mandarin_bridge"
                            dsp_manifest["mandarin_frontend"] = mandarin_frontend
                            dsp_manifest["output_artifacts"].update(mandarin_frontend["outputs"])
                            dsp_manifest_path.write_text(
                                json.dumps(dsp_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8",
                            )
                        (generated / "registration-identity.json").write_text(json.dumps(identity, ensure_ascii=False, sort_keys=True) + "\n")
                        generated.rename(inputs)
                if json.loads((inputs / "registration-identity.json").read_text()) != identity:
                    raise RuntimeError("Persisted reference preprocessing identity mismatch")
                reference_bert = None
                if language == "zh":
                    mandarin_report = json.loads((inputs / "mandarin-reference-frontend.json").read_text(encoding="utf-8"))
                    if mandarin_report.get("source_files_sha256") != identity["mandarin_frontend_sources"]:
                        raise RuntimeError("Persisted Mandarin frontend source identity mismatch")
                    for name in ("reference-phones.npy", "reference-bert.npy"):
                        output_record = mandarin_report.get("outputs", {}).get(name)
                        output_path = inputs / name
                        if (not isinstance(output_record, dict) or not output_path.is_file()
                                or output_record.get("sha256") != hashlib.sha256(output_path.read_bytes()).hexdigest()):
                            raise RuntimeError(f"Persisted Mandarin frontend output hash mismatch: {name}")
                    reference_phones_array = np.load(inputs / "reference-phones.npy", allow_pickle=False)
                    reference_bert = np.load(inputs / "reference-bert.npy", allow_pickle=False)
                    if (reference_phones_array.ndim != 2 or reference_phones_array.shape[0] != 1
                            or reference_phones_array.shape[1] <= 0
                            or reference_phones_array.dtype != np.int64
                            or reference_bert.shape != (1, 1024, reference_phones_array.shape[1])
                            or reference_bert.dtype != np.float32
                            or not np.isfinite(reference_bert).all()
                            or not np.count_nonzero(reference_bert)):
                        raise RuntimeError("Persisted Mandarin reference phones/BERT are invalid")
                if self._client is None:
                    raise RuntimeError("Native worker closed during reference preparation")
                # The handle must not point at a Voice-store asset that can be
                # deleted or replaced; keep a content-addressed copy beside the
                # inputs (same SHA-256, so the worker cache identity is unchanged).
                stable_audio = inputs / "reference-audio.wav"
                if not stable_audio.is_file() or hashlib.sha256(stable_audio.read_bytes()).hexdigest() != identity["audio_sha256"]:
                    staged = inputs / f".reference-audio.{os.getpid()}.tmp"
                    shutil.copyfile(audio, staged)
                    staged.replace(stable_audio)
                audio = stable_audio
                with tempfile.TemporaryDirectory(prefix="registration-result-", dir=root) as temp:
                    result = self._client.prepare_reference(inputs, Path(temp), audio=audio,
                                                            transcript=inputs / "reference.txt")
                self._mlx_memory_last = result.get("memory")
                cache_key = result.get("cache_key")
                if not isinstance(cache_key, str) or len(cache_key) != 64:
                    raise RuntimeError("Native worker returned an invalid reference identity")
                result["report_path"] = str(inputs / "native-preparation.json")
                (inputs / "native-preparation.json").write_text(json.dumps(result, sort_keys=True) + "\n")
                handle = NativeReferenceHandle(inputs, audio, inputs / "reference.txt", language, cache_key)
                phones = np.load(inputs / "reference-phones.npy", allow_pickle=False).reshape(-1).tolist()
                if not phones:
                    raise RuntimeError("Prepared reference contains no phones")
                lease.ensure_valid()
                return PreparedReference(prompt_semantic=handle, spectrogram=handle, speaker_embedding=handle,
                    phones=phones, bert=(reference_bert[0] if language == "zh"
                        else np.zeros((1024, len(phones)), dtype=np.float32)), id=reference_id)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            cleanup_error = self._end_request(lease)
            if cleanup_error is not None:
                if primary_error is None:
                    raise RuntimeError("Reference preparation resource cleanup failed") from cleanup_error
                api.LOGGER.error("Native reference cleanup also failed", exc_info=(type(cleanup_error), cleanup_error, cleanup_error.__traceback__))

    def _qualification_metadata(self) -> dict[str, Any]:
        """Distinguish runtime availability from the recorded host-specific evidence."""
        ready = self.ready
        mode = ("canonical_vocoder" if self._canonical_vocoder_stream else
                "bootstrap_v2" if self._bootstrap_stream_v2 else "worker_default")
        contract = self._canonical_stream_worker_capability
        worker = getattr(self._client, "metadata", None) if self._client is not None else None
        worker = worker if isinstance(worker, dict) else {}
        return {
            "current_runtime": "not_assessed",
            "reference_validation": "accepted_with_documented_exceptions",
            "reference_validation_scope": "historical FP16 k1 reference host and recorded cases, not the current package or host",
            "worker_protocol": {"prototype": worker.get("prototype"), "qualified": worker.get("qualified"),
                                "canonical_stream_quality_qualified": contract.get("quality_qualified") if contract is not None else None},
            "complete_wav": {"supported": True, "available": ready,
                             "historical_validation": "fp16_k1_and_p1_reference_host"},
            "pcm_stream": {"supported": self._product_stream,
                           "available": ready and self._product_stream,
                           "mode": mode,
                           "historical_validation": "fp16_k1_and_p1_canonical_vocoder_on_reference_host",
                           "worker_quality_qualified": contract.get("quality_qualified") if contract is not None else None},
            "speech_sessions": {"policy_a": "supported", "policy_a_stream_available": ready and self._product_stream,
                                "policy_a_historical_validation": "fp16_k1_and_p1_reference_host",
                                "policies_b_to_f": "unsupported"},
            "objective_quality": {
                "scope": "historical FP16 k/k1 17-case matrix; not a current-runtime assessment",
                "original_gate_passed": 14, "case_count": 17,
                "accepted_tolerances": ["ja-original-20261112 F0", "long-ja-20261112 one semantic token"],
                "official_model_limitations": ["yue-20261112 absolute CER"],
                "all_original_gates_passed": False,
                "subjective_listening": "waived_for_k1_objective_handoff",
                "p1_evidence": "default neutral output matched k1; full matrix not rerun",
            },
            "identity_style_equivalence": "passed_by_explicit_owner_acceptance",
            "host": {"current_host": "not_assessed",
                     "reference_host": "Mac Studio M5 Max, 40-core GPU, 128 GB, macOS 27.0",
                     "other_apple_silicon": "untested", "intel_mac": "unsupported"},
            "evidence": "macOS FP16 k1 reference qualification and p1 feature-parity report",
        }

    def capabilities(self, limits: dict[str, Any]) -> dict[str, Any]:
        streaming = self._product_stream
        stream_mode = ("canonical_vocoder" if self._canonical_vocoder_stream else
                       "bootstrap_v2" if self._bootstrap_stream_v2 else "worker_default")
        return {
            "service": api.SERVICE_NAME, "version": api.SERVICE_VERSION,
            "model_family": "gsv-v5turbo", "backend": "MLX-Metal", "precision": self._precision,
            "compute_precision": {"core": ("float16" if self._precision == "FP16" else "float32"), "cfm": "NAX-relaxed-enabled" if self._cfm_stage else ("float16" if self._precision == "FP16" else "float32"), "bert": ("float16" if self._precision == "FP16" else "float32"), "g2pw": "float32", "half_convolution_opmath": "float32" if self._precision == "FP16" else None, "half_weights_and_boundary_tensors": self._precision == "FP16", "cfm_compiled": self._cfm_compiled, "weight_files_modified": False, "vocoder_checkpoint_dtype": "float32", "vocoder_runtime_cast": "official_is_half" if self._precision == "FP16" else "none"},
            "languages": self._languages(),
            "streaming": {"pcm16": streaming, "wav": False, "quality_qualified": False,
                          "mode": stream_mode, "canonical_vocoder_stream": self._canonical_stream_metadata()},
            "expression": {"native": True, "controlled_profiles": True,
                           "continuous_vector": False, "segmented": True,
                           "policies": self._supported_expression_policies(),
                           "complete_policies": (self._supported_expression_policies() if self._complete_mode == "cached"
                                                 else [ConditioningPolicy.FULL_SWITCH.value]),
                           "conditioning_components": {
                               "semantic": "AR semantic prompt (prompt codes, reference phones/BERT)",
                               "spectrogram": "global style vector ge from the reference spectrum",
                               "speaker_embedding": "CFM in-context acoustic prompt (reference features and mel); V5 Turbo has no separate SV embedding"}},
            "pytorch_fallback": False, "adaptive_punctuation_segments": True,
            "voice_management": {"crud": True, "reference_upload": True,
                                 "reference_prepare": True, "persistent": True,
                                 "reference_languages": ["ja", "en", "yue", "ko"] + (["zh"] if self._mandarin_enabled else []),
                                 "transcript_required": True},
            "warm_retention_seconds": 0,
            "speech_sessions": {"committed_segments": True, "pcm_stream": streaming,
                                "cancel": True, "flush": True, "context_mode": "committed-neural-v1",
                                "default_policy": "A", "policies": {"A": dict(api.CONTINUITY_POLICY_CONTRACTS["A"])},
                                "neural_state_continuity": False, "acoustic_latent_continuity": False,
                                "qualification": "qualified-policy-a",
                                "qualification_scope": "historical canonical-stream policy A; current host and package not assessed",
                                "limits": limits},
            "prototype": False, "quality_qualified": False, "fixed_voice_only": False,
            "quality_qualification_scope": "all original quality gates are not established for the active configuration",
            "qualification": self._qualification_metadata(),
            "complete_synthesis": {"mode": self._complete_mode,
                "reference_conditioning_reused": self._complete_mode == "cached",
                "neural_capture": self._complete_mode == "captured",
                "cfm_static_prepared": self._cfm_prepare, "ar_combined_decision": self._ar_combined, "cfm_isolated_tf32": self._cfm_stage, "cfm_compiled": self._cfm_compiled, "cfm_batched_qkv": self._cfm_batched_qkv, "cfm_packed_qkv_blocks": 22 if self._cfm_batched_qkv else 0, "cfm_coordinate_v2": self._coordinate_noise_v2, "complete_cfm_noise_policy": self._complete_cfm_noise_policy, "ar_shapeless_compiled": self._ar_compiled, "ar_lossless_storage_gemv": self._ar_mixed, "ar_optimized": self._ar_optimized, "ar_residual_norm_fused": self._ar_residual_norm, "ar_batch_size": 4 if self._ar_optimized else 1, "ar_gemv_rows": (1 if self._ar_optimized else 4) if self._ar_mixed else None, "ar_fused_epilogue": self._ar_optimized},
            "frontend": {"temporary_python": True, "mode": self._frontend_mode,
                         "metadata_cache": self._frontend_metadata_cache,
                         "neural_fallback": False},
            "sampling": self._sampling_capabilities(),
            "native_worker": {"separate_conditioning_components": self._worker_supports_conditioning_components(),
                              "request_sampling": self._request_sampling},
        }

    def health(self) -> dict[str, Any]:
        ready = self.ready
        with self._active_guard:
            active = self._active_requests
            frontend = self._last_frontend_seconds
            frontend_segments = self._last_frontend_segments
        # Source defaults: core=200, future=12, holdback=50; updates every 16 tokens.
        needed = math.ceil((200 + 12 + 50) / 4)
        planned_tokens = math.ceil(needed / 16) * 16
        stream_mode = ("canonical_vocoder" if self._canonical_vocoder_stream else
                       "bootstrap_v2" if self._bootstrap_stream_v2 else "worker_default")
        return {"service": api.SERVICE_NAME, "version": api.SERVICE_VERSION,
                "status": "ok" if ready else "not_ready", "ready": ready,
                "backend": "MLX-Metal", "precision": self._precision, "mlx_memory_ready": self._mlx_memory_ready,
                "mlx_memory_last_job": self._mlx_memory_last, "model": api.MODEL_ID, "voice": api.VOICE_ID,
                "sample_rate": 48000 if ready else None, "gpu": None,
                "engine_fingerprint": self._engine_manifest.get("fingerprint"),
                "engine_count": (1 + int(self._cfm_stage)) if ready else 0,
                "engines": {"mlx_worker": ready, "isolated_cfm": ready and self._cfm_stage},
                "compute_precision": {"core": ("float16" if self._precision == "FP16" else "float32"), "cfm": "NAX-relaxed-enabled" if self._cfm_stage else ("float16" if self._precision == "FP16" else "float32"), "bert": ("float16" if self._precision == "FP16" else "float32"), "g2pw": "float32", "half_convolution_opmath": "float32" if self._precision == "FP16" else None, "half_weights_and_boundary_tensors": self._precision == "FP16", "cfm_compiled": self._cfm_compiled, "vocoder_checkpoint_dtype": "float32", "vocoder_runtime_cast": "official_is_half" if self._precision == "FP16" else "none"},
                "last_onset_trim": self._last_onset_trim,
                "last_request_conditioning": self._last_conditioning,
                "reference": {"configured": self._reference_inputs is not None,
                              "language": api.REFERENCE_LANGUAGE, "scope": "packaged and registered references"},
                "expression": self.expression_metadata(), "request_segment_char_limit": self._segment_char_limit,
                "streaming": {"enabled": ready and self._product_stream,
                              "product_stream": self._canonical_vocoder_stream,
                              "transport": "pcm_s16le", "complete_wav": "request stream=false",
                              "quality_qualified": False, "research_opt_in": self._allow_unqualified_stream,
                              "mode": stream_mode,
                              "canonical_vocoder_stream": self._canonical_stream_metadata(),
                              "semantic_context_tokens": None if self._canonical_vocoder_stream else planned_tokens,
                              "semantic_context_note": (
                                  "canonical mode completes AR and CFM before incremental vocoder PCM"
                                  if self._canonical_vocoder_stream and ready else
                                  "canonical mode requested; worker contract not yet verified"
                                  if self._canonical_vocoder_stream else
                                  "planned first-window budget; EOS may arrive earlier"
                              )},
                "active_requests": active, "complete_synthesis_mode": self._complete_mode, "cfm_static_prepared": self._cfm_prepare, "ar_combined_decision": self._ar_combined, "cfm_isolated_tf32": self._cfm_stage, "cfm_compiled": self._cfm_compiled, "cfm_batched_qkv": self._cfm_batched_qkv, "cfm_packed_qkv_blocks": 22 if self._cfm_batched_qkv else 0, "ar_shapeless_compiled": self._ar_compiled, "ar_static_decode": self._ar_static, "ar_static_cache_capacity": 1024 if self._ar_static else None, "ar_lossless_storage_gemv": self._ar_mixed, "ar_optimized": self._ar_optimized, "ar_residual_norm_fused": self._ar_residual_norm, "ar_batch_size": 4 if self._ar_optimized else 1, "ar_gemv_rows": (1 if self._ar_optimized else 4) if self._ar_mixed else None, "ar_fused_epilogue": self._ar_optimized, "frontend": {"temporary_python": True, "mode": self._frontend_mode, "metadata_cache": self._frontend_metadata_cache,
                    "last_wall_seconds": frontend, "completed_segments": frontend_segments},
                "prototype": False, "quality_qualified": False, "neural_fallback": False,
                "quality_qualification_scope": "all original quality gates are not established for the active configuration",
                "qualification": self._qualification_metadata(),
                "supported_languages": self._languages(),
                "continuity_policies": ["A"], "dynamic_voices": True}
