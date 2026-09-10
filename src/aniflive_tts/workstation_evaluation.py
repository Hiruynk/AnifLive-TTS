from __future__ import annotations

import hashlib
import http.client
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import wave
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

_SOURCE_DIR = Path("/app/minimal_inference")
_EVALUATION_TOOLS = Path("/opt/aniflive-tts/evaluation")
_SERVER_PORT = 18882
V13_BENCHMARK_TEXT = "今日はいい天気ですね。"
PERFORMANCE_POLICY = "v14-near-v13-10pct-v1"
_LANGUAGE_CASES = {
    "zh": "今天天氣很好，我們來測試語音品質。",
    "yue": "今日天氣幾好，我哋一齊測試語音品質。",
    "en": "The weather is pleasant today, so we can test the voice clearly.",
    "ja": "今日はいい天気なので、音声の品質を確認します。",
    "ko": "오늘은 날씨가 좋아서 음성 품질을 확인합니다.",
}
_ALLOWED_SETTINGS = frozenset(
    {
        "asr_compute_type",
        "semantic_sampling",
        "benchmark_language",
        "benchmark_text",
        "benchmark_runs",
        "benchmark_sessions",
        "benchmark_warmups",
        "request_timeout_seconds",
    }
)
_WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


class EvaluationWorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvaluationPlan:
    benchmark_sessions: int
    benchmark_warmups: int
    benchmark_runs: int
    benchmark_language: str
    asr_compute_type: str
    request_timeout_seconds: int
    benchmark_text: str | None = None
    semantic_sampling: str = "legacy-topk-v1"


def _bounded_int(value: Any, field: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise EvaluationWorkerError(f"{field} must be an integer between {minimum} and {maximum}")
    return value


def resolve_evaluation_plan(settings: Mapping[str, Any]) -> EvaluationPlan:
    if not isinstance(settings, Mapping):
        raise EvaluationWorkerError("evaluation settings must be a JSON object")
    unknown = set(settings) - _ALLOWED_SETTINGS
    if unknown:
        raise EvaluationWorkerError("unsupported evaluation settings: " + ", ".join(sorted(unknown)))
    language = settings.get("benchmark_language", "ja")
    if language not in _LANGUAGE_CASES:
        raise EvaluationWorkerError("benchmark_language must be zh, yue, en, ja or ko")
    compute_type = settings.get("asr_compute_type", "float16")
    if compute_type not in {"float16", "int8_float16"}:
        raise EvaluationWorkerError("asr_compute_type must be float16 or int8_float16")
    benchmark_text = settings.get("benchmark_text")
    if benchmark_text is not None and (
        not isinstance(benchmark_text, str) or not benchmark_text.strip()
        or len(benchmark_text) > 4096 or "\x00" in benchmark_text
    ):
        raise EvaluationWorkerError("benchmark_text must be nonempty text up to 4096 characters")
    semantic_sampling = settings.get("semantic_sampling", "legacy-topk-v1")
    if not isinstance(semantic_sampling, str) or semantic_sampling not in {"legacy-topk-v1", "native-v2proplus-v1"}:
        raise EvaluationWorkerError("Unsupported semantic_sampling contract")
    return EvaluationPlan(
        semantic_sampling=semantic_sampling,
        benchmark_sessions=_bounded_int(
            settings.get("benchmark_sessions", 1), "benchmark_sessions", 1, 10
        ),
        benchmark_warmups=_bounded_int(
            settings.get("benchmark_warmups", 2), "benchmark_warmups", 0, 20
        ),
        benchmark_runs=_bounded_int(
            settings.get("benchmark_runs", 10), "benchmark_runs", 1, 100
        ),
        benchmark_language=str(language),
        benchmark_text=(benchmark_text if benchmark_text is not None else
                        V13_BENCHMARK_TEXT if language == "ja" else _LANGUAGE_CASES[language]),
        asr_compute_type=str(compute_type),
        request_timeout_seconds=_bounded_int(
            settings.get("request_timeout_seconds", 180),
            "request_timeout_seconds",
            30,
            600,
        ),
    )


def evaluation_runtime_policy(plan: EvaluationPlan) -> dict[str, Any]:
    return {
        "semantic_sampling": plan.semantic_sampling,
        "repetition_penalty": 1.35 if plan.semantic_sampling == "native-v2proplus-v1" else 1.0,
    }


def evaluation_sampling_environment(plan: EvaluationPlan) -> dict[str, str]:
    policy = evaluation_runtime_policy(plan)
    return {
        "ANIFLIVE_TTS_SEMANTIC_SAMPLING": policy["semantic_sampling"],
        "ANIFLIVE_TTS_REPETITION_PENALTY": str(policy["repetition_penalty"]),
    }


def evaluation_workload(plan: EvaluationPlan) -> dict[str, Any]:
    return {
        "text": plan.benchmark_text or (
            V13_BENCHMARK_TEXT if plan.benchmark_language == "ja"
            else _LANGUAGE_CASES[plan.benchmark_language]
        ),
        "language": plan.benchmark_language,
        "seed": 1234, "top_k": 15, "top_p": 1.0, "temperature": 1.0,
        "expression": {"enabled": False, "profile": None,
                       "intensity": 0.7, "policy": "semantic-style"},
    }


def preflight_evaluation_baseline(
    plan: EvaluationPlan, baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject mismatched comparison inputs before starting CUDA or measuring."""
    if baseline.get("schema") != "aniflive-tts-workstation-evaluation-v1":
        raise EvaluationWorkerError("baseline_report schema is unsupported")
    benchmark = baseline.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise EvaluationWorkerError("baseline_report has no embedded canonical benchmark")
    expected = evaluation_workload(plan)
    previous = benchmark.get("workload")
    if previous != expected:
        fields = sorted(
            key for key in set(expected) | set(previous or {})
            if not isinstance(previous, Mapping) or previous.get(key) != expected.get(key)
        ) if isinstance(previous, Mapping) else ["workload"]
        raise EvaluationWorkerError(
            "baseline benchmark workload does not match: " + ", ".join(fields)
        )
    methodology = {
        "sessions": plan.benchmark_sessions,
        "warmup_requests_per_session": plan.benchmark_warmups,
        "full_wav_requests_per_session": plan.benchmark_runs,
        "stream_requests_per_session": plan.benchmark_runs,
        "keepalive_stream_requests_per_session": plan.benchmark_runs,
    }
    previous_method = benchmark.get("methodology")
    if not isinstance(previous_method, Mapping) or any(
        previous_method.get(key) != value for key, value in methodology.items()
    ):
        raise EvaluationWorkerError("baseline benchmark methodology does not match")
    return {"workload": expected, "methodology": methodology,
            "performance_policy": PERFORMANCE_POLICY, "status": "passed"}


def _emit_progress(progress: float, stage: str, message: str) -> None:
    print(
        "ANIFLIVE_TTS_PROGRESS "
        + json.dumps(
            {"progress": round(progress, 6), "stage": stage, "message": message},
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _regular_file(path: Path, field: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise EvaluationWorkerError(f"{field} must be a regular file")
    return path.resolve(strict=True)


def _directory(path: Path, field: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise EvaluationWorkerError(f"{field} must be a directory")
    return path.resolve(strict=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _asr_fingerprint(root: Path) -> dict[str, Any]:
    candidates = [
        root / "model.bin",
        root / "config.json",
        root / "tokenizer.json",
        root / "vocabulary.json",
        root / "vocabulary.txt",
    ]
    files = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if not (root / "model.bin").is_file() or not files:
        raise EvaluationWorkerError("asr_model is not a local CTranslate2 Whisper model")
    return {
        "files": {
            path.name: {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
            for path in files
        }
    }


def _finite_metric(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise EvaluationWorkerError(f"{field} must be a finite number >= {minimum}")
    return float(value)


def _request(
    method: str,
    path: str,
    *,
    timeout: float,
    body: Mapping[str, Any] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", _SERVER_PORT, timeout=timeout)
    payload = None
    headers: dict[str, str] = {"Connection": "close"}
    if body is not None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    try:
        connection.request(method, path, payload, headers)
        response = connection.getresponse()
        response_body = response.read()
        response_headers = {key.casefold(): value for key, value in response.getheaders()}
        return response.status, response_headers, response_body
    finally:
        connection.close()


def _wait_for_service(process: subprocess.Popen[str], timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "service did not answer"
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise EvaluationWorkerError(f"TensorRT API exited during startup with code {returncode}")
        try:
            status, _, payload = _request("GET", "/health", timeout=2)
            health = json.loads(payload)
            if status == 200 and health.get("ready") and health.get("backend") == "TensorRT-11":
                return health
            last_error = f"health returned HTTP {status}: {health!r}"
        except (OSError, json.JSONDecodeError) as error:
            last_error = str(error)
        time.sleep(0.25)
    raise EvaluationWorkerError(f"TensorRT API startup timed out: {last_error}")


def _validate_headers(headers: Mapping[str, str], language: str) -> None:
    if headers.get("x-tensorrt-backend") != "TensorRT-11":
        raise EvaluationWorkerError(f"{language} did not execute on TensorRT 11")
    if headers.get("x-pytorch-fallback") != "false":
        raise EvaluationWorkerError(f"{language} reported a PyTorch fallback")
    if headers.get("x-tts-model") in {None, ""}:
        raise EvaluationWorkerError(f"{language} omitted the active model identity")


def _decode_wav(payload: bytes) -> tuple[int, np.ndarray]:
    with wave.open(io.BytesIO(payload), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise EvaluationWorkerError("evaluation requires mono PCM16 WAV output")
        rate = audio.getframerate()
        frames = audio.readframes(audio.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if not samples.size or not np.isfinite(samples).all():
        raise EvaluationWorkerError("TensorRT returned invalid audio")
    return rate, samples


def _pcm_wav(pcm: bytes, sample_rate: int) -> bytes:
    destination = io.BytesIO()
    with wave.open(destination, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm)
    return destination.getvalue()


def _spectral_quality(complete_path: Path, stream_path: Path) -> dict[str, float]:
    from .backend.audio_quality import (
        AudioQualityConfig,
        _align,
        _load_wav,
        _resample,
        _spectral_metrics,
    )

    complete = _load_wav(complete_path)
    stream = _load_wav(stream_path)
    sample_rate = min(complete.sample_rate, stream.sample_rate)
    left = _resample(complete.mono, complete.sample_rate, sample_rate)
    right = _resample(stream.mono, stream.sample_rate, sample_rate)
    aligned_left, aligned_right, _ = _align(left, right, sample_rate, 1.0)
    metrics, _, _, _ = _spectral_metrics(
        aligned_left,
        aligned_right,
        sample_rate,
        AudioQualityConfig(),
    )
    complete_duration = complete.samples.shape[0] / complete.sample_rate
    stream_duration = stream.samples.shape[0] / stream.sample_rate
    return {
        "log_mel_cosine": float(metrics["log_mel_cosine_similarity"]),
        "duration_difference_ratio": abs(stream_duration - complete_duration) / complete_duration,
    }


def _trim_and_normalize(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    audio = audio - float(np.mean(audio, dtype=np.float64))
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        active = np.flatnonzero(np.abs(audio) >= max(1e-4, peak * 0.01))
        if active.size >= 2 and int(active[-1] - active[0] + 1) >= 1600:
            audio = audio[int(active[0]) : int(active[-1]) + 1]
        peak = float(np.max(np.abs(audio)))
        audio = audio / peak * 0.9
    return audio


class _TensorRTSpeakerEmbedder:
    def __init__(self, package: Path) -> None:
        import torch

        from .model_package import select_engine_dir

        source_text = str(_SOURCE_DIR)
        gpt_text = str(_SOURCE_DIR / "GPT_SoVITS")
        for value in (source_text, gpt_text):
            if value not in sys.path:
                sys.path.insert(0, value)
        from run_trt_inference import TRTModule

        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        engine = select_engine_dir(package, manifest) / "sv_embedding.engine"
        self._torch = torch
        self._module = TRTModule(
            str(engine), device="cuda", stream=torch.cuda.current_stream()
        )

    def __call__(self, path: Path) -> np.ndarray:
        import soundfile as sf
        import soxr

        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1, dtype=np.float32)
        if sample_rate != 16000:
            mono = soxr.resample(mono, sample_rate, 16000, quality="HQ")
        mono = _trim_and_normalize(mono)
        waveform = self._torch.from_numpy(mono)[None, :].to(
            "cuda", dtype=self._module.tensor_dtype["audio"]
        )
        embedding = self._module({"audio": waveform})["sv_embedding"]
        result = embedding.detach().float().cpu().numpy().reshape(-1)
        if not result.size or not np.isfinite(result).all() or float(np.linalg.norm(result)) <= 0:
            raise EvaluationWorkerError("TensorRT speaker engine produced an invalid embedding")
        return result


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(np.float32).eps:
        raise EvaluationWorkerError("speaker embedding cosine denominator is zero")
    return float(np.dot(left, right) / denominator)


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _content_units(text: str, language: str) -> tuple[list[str], str]:
    normalized = text.casefold().replace("_", " ")
    if language == "en":
        return _WORD_PATTERN.findall(normalized), "wer"
    return [character for character in normalized if character.isalnum()], "cer"


def content_error(reference: str, hypothesis: str, language: str) -> dict[str, Any]:
    expected, metric = _content_units(reference, language)
    actual, _ = _content_units(hypothesis, language)
    if not expected:
        raise EvaluationWorkerError("canonical evaluation text normalized to an empty sequence")
    edits = _edit_distance(expected, actual)
    return {
        "metric": metric,
        "edits": edits,
        "reference_units": len(expected),
        "hypothesis_units": len(actual),
        "error_rate": edits / len(expected),
    }


def orthographic_content_error(
    reference: str, hypothesis: str, language: str
) -> dict[str, Any]:
    """Compare Chinese script variants without hiding word or dialect substitutions."""
    if language not in {"zh", "yue"}:
        return content_error(reference, hypothesis, language)
    from importlib.metadata import version

    try:
        from opencc import OpenCC
    except ImportError as error:
        raise EvaluationWorkerError("Chinese content evaluation requires OpenCC") from error
    normalizer = OpenCC("t2s")
    expected = normalizer.convert(reference)
    actual = normalizer.convert(hypothesis)
    return {
        **content_error(expected, actual, language),
        "normalizer": "opencc-t2s",
        "normalizer_version": version("OpenCC"),
        "normalized_reference": expected,
        "normalized_hypothesis": actual,
    }


def content_error_rate(reference: str, hypothesis: str, language: str) -> float:
    return float(content_error(reference, hypothesis, language)["error_rate"])


def spoken_content_error(
    reference: str, hypothesis: str, language: str
) -> dict[str, Any]:
    if language != "ja":
        return content_error(reference, hypothesis, language)
    try:
        import pyopenjtalk
    except ImportError as error:
        raise EvaluationWorkerError(
            "Japanese spoken-content evaluation requires pyopenjtalk"
        ) from error
    expected = str(pyopenjtalk.g2p(reference, kana=False)).split()
    actual = str(pyopenjtalk.g2p(hypothesis, kana=False)).split()
    if not expected:
        raise EvaluationWorkerError(
            "Japanese evaluation text normalized to an empty phoneme sequence"
        )
    edits = _edit_distance(expected, actual)
    return {
        "metric": "per",
        "edits": edits,
        "reference_units": len(expected),
        "hypothesis_units": len(actual),
        "error_rate": edits / len(expected),
        "normalizer": "pyopenjtalk-phonemes",
        "normalizer_version": str(getattr(pyopenjtalk, "__version__", "unknown")),
    }


def spoken_content_error_rate(reference: str, hypothesis: str, language: str) -> float:
    return float(spoken_content_error(reference, hypothesis, language)["error_rate"])


def _benchmark_metrics(report: Mapping[str, Any], field: str) -> dict[str, float]:
    rows = report.get("table")
    if not isinstance(rows, list):
        raise EvaluationWorkerError(f"{field}.table must be a JSON array")
    metrics: dict[str, float] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise EvaluationWorkerError(f"{field}.table[{index}] must be a JSON object")
        key = row.get("key")
        if not isinstance(key, str) or not key:
            raise EvaluationWorkerError(f"{field}.table[{index}].key is malformed")
        if key in metrics:
            raise EvaluationWorkerError(f"{field}.table contains duplicate metric {key}")
        metrics[key] = _finite_metric(row.get("median"), f"{field}.{key}")
    return metrics


def compare_evaluation_baseline(
    *,
    languages: Mapping[str, Mapping[str, Any]],
    benchmark: Mapping[str, Any],
    asr_identity: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    if baseline.get("schema") != "aniflive-tts-workstation-evaluation-v1":
        raise EvaluationWorkerError("baseline_report schema is unsupported")
    if baseline.get("asr", {}).get("model_fingerprint") != asr_identity:
        raise EvaluationWorkerError("baseline_report used a different offline ASR model")
    baseline_languages = baseline.get("languages")
    if not isinstance(baseline_languages, Mapping) or set(baseline_languages) != set(
        _LANGUAGE_CASES
    ):
        raise EvaluationWorkerError("baseline_report must contain the same five languages")

    language_results: dict[str, dict[str, Any]] = {}
    for language in _LANGUAGE_CASES:
        candidate = languages.get(language)
        previous = baseline_languages.get(language)
        if not isinstance(candidate, Mapping) or not isinstance(previous, Mapping):
            raise EvaluationWorkerError(f"baseline_report language {language} is malformed")
        if candidate.get("text") != previous.get("text"):
            raise EvaluationWorkerError(
                f"baseline_report language {language} used a different text"
            )
        candidate_asr = candidate.get("asr")
        previous_asr = previous.get("asr")
        candidate_quality = candidate.get("quality")
        previous_quality = previous.get("quality")
        if not all(
            isinstance(value, Mapping)
            for value in (candidate_asr, previous_asr, candidate_quality, previous_quality)
        ):
            raise EvaluationWorkerError(
                f"baseline_report language {language} has incomplete quality evidence"
            )
        if candidate_asr.get("metric") != previous_asr.get("metric"):
            raise EvaluationWorkerError(
                f"baseline_report language {language} used a different content metric"
            )
        content = _finite_metric(
            candidate_asr.get("error_rate"), f"candidate.{language}.error_rate"
        )
        baseline_content = _finite_metric(
            previous_asr.get("error_rate"), f"baseline.{language}.error_rate"
        )
        normalization = None
        raw_content, raw_baseline_content = content, baseline_content
        if language in {"zh", "yue"}:
            hypotheses = (candidate_asr.get("hypothesis"), previous_asr.get("hypothesis"))
            if any(not isinstance(value, str) for value in hypotheses):
                raise EvaluationWorkerError(
                    f"baseline_report language {language} requires original ASR hypotheses"
                )
            candidate_normalized = orthographic_content_error(
                candidate["text"], hypotheses[0], language
            )
            baseline_normalized = orthographic_content_error(
                previous["text"], hypotheses[1], language
            )
            content = float(candidate_normalized["error_rate"])
            baseline_content = float(baseline_normalized["error_rate"])
            normalization = {
                "candidate": candidate_normalized, "baseline": baseline_normalized,
                "scope": "both original hypotheses rescored using the same script normalizer",
            }
        speaker = _finite_metric(
            candidate_quality.get("speaker_cosine_stream_vs_identity"),
            f"candidate.{language}.speaker_cosine_stream_vs_identity",
        )
        baseline_speaker = _finite_metric(
            previous_quality.get("speaker_cosine_stream_vs_identity"),
            f"baseline.{language}.speaker_cosine_stream_vs_identity",
        )
        content_regression = content - baseline_content
        speaker_drop = baseline_speaker - speaker
        language_results[language] = {
            "content_error_rate": content,
            "baseline_content_error_rate": baseline_content,
            "raw_content_error_rate": raw_content,
            "raw_baseline_content_error_rate": raw_baseline_content,
            "orthographic_normalization": normalization,
            "content_absolute_regression": content_regression,
            "content_passed": content_regression <= 0.005,
            "speaker_cosine": speaker,
            "baseline_speaker_cosine": baseline_speaker,
            "speaker_drop": speaker_drop,
            "speaker_passed": speaker_drop <= 0.005,
        }
        language_results[language]["passed"] = bool(
            language_results[language]["content_passed"]
            and language_results[language]["speaker_passed"]
        )

    baseline_benchmark = baseline.get("benchmark")
    if not isinstance(baseline_benchmark, Mapping):
        raise EvaluationWorkerError("baseline_report has no embedded canonical benchmark")
    for section in ("workload", "methodology"):
        candidate_section = benchmark.get(section)
        previous_section = baseline_benchmark.get(section)
        if section == "methodology":
            keys = {
                "sessions",
                "warmup_requests_per_session",
                "full_wav_requests_per_session",
                "stream_requests_per_session",
                "keepalive_stream_requests_per_session",
            }
            if not isinstance(candidate_section, Mapping) or not isinstance(
                previous_section, Mapping
            ):
                raise EvaluationWorkerError("baseline benchmark methodology is malformed")
            if {key: candidate_section.get(key) for key in keys} != {
                key: previous_section.get(key) for key in keys
            }:
                raise EvaluationWorkerError("baseline benchmark methodology does not match")
        elif candidate_section != previous_section:
            raise EvaluationWorkerError("baseline benchmark workload does not match")

    current_metrics = _benchmark_metrics(benchmark, "benchmark")
    previous_metrics = _benchmark_metrics(baseline_benchmark, "baseline.benchmark")
    performance_limits = {
        "stream_keepalive_audible_ttfa_p50_ms": 1.10,
        "stream_keepalive_audible_ttfa_p95_ms": 1.10,
        "wall_rtf_p50": 1.10,
    }
    performance: dict[str, dict[str, Any]] = {}
    for key, multiplier in performance_limits.items():
        if key not in current_metrics or key not in previous_metrics:
            raise EvaluationWorkerError(f"baseline benchmark is missing required metric {key}")
        current = current_metrics[key]
        previous = previous_metrics[key]
        limit = previous * multiplier
        performance[key] = {
            "candidate": current,
            "baseline": previous,
            "maximum": limit,
            "passed": current <= limit,
        }

    passed = all(row["passed"] for row in language_results.values()) and all(
        row["passed"] for row in performance.values()
    )
    return {
        "available": True,
        "content_absolute_regression_limit": 0.005,
        "speaker_similarity_drop_limit": 0.005,
        "performance_regression_limit_ratio": 0.10,
        "performance_policy": PERFORMANCE_POLICY,
        "languages": language_results,
        "performance": performance,
        "passed": passed,
    }


def _transcribe(model: Any, path: Path, language: str) -> str:
    whisper_language = "zh" if language == "yue" else language
    segments, _ = model.transcribe(
        str(path),
        language=whisper_language,
        beam_size=5,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    return "".join(segment.text for segment in segments).strip()


def _terminate(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _run_logged_benchmark(
    command: Sequence[str], *, cwd: Path, log_path: Path,
    timeout: float, sessions: int,
) -> tuple[int, str]:
    """Keep real session progress and partial logs without changing requests."""
    import codecs
    import selectors

    deadline = time.monotonic() + timeout
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    tail = ""
    last_session = 0
    with log_path.open("w", encoding="utf-8") as log, subprocess.Popen(
        command, cwd=str(cwd), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False,
    ) as process:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    for key, _ in selector.select(min(0.25, remaining)):
                        block = os.read(key.fileobj.fileno(), 65536)
                        if not block:
                            selector.unregister(key.fileobj)
                            chunk = decoder.decode(b"", final=True)
                        else:
                            chunk = decoder.decode(block)
                        log.write(chunk)
                        log.flush()
                        tail = (tail + chunk)[-6000:]
                        pending += chunk
                        while "\n" in pending:
                            line, pending = pending.split("\n", 1)
                            match = re.search(r"\[benchmark\].* session (\d+)/(\d+):", line)
                            if match is None:
                                continue
                            current, total = map(int, match.groups())
                            if total == sessions and last_session < current <= total:
                                last_session = current
                                _emit_progress(
                                    0.38 + 0.40 * (current - 1) / total,
                                    "benchmark",
                                    f"Canonical benchmark session {current}/{total} started",
                                )
                        # Do not retain arbitrarily long non-progress output in memory.
                        pending = pending[-8192:]
            code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
            return code, tail
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def _run_benchmark(
    plan: EvaluationPlan, model_id: str, output: Path, *, runtime_release: str = "unknown",
) -> tuple[Path, Path]:
    script = _regular_file(_EVALUATION_TOOLS / "benchmark_readme.py", "benchmark tool")
    report = output / "benchmark.json"
    markdown = output / "benchmark.md"
    command = (
        sys.executable,
        "-s",
        str(script),
        "--host",
        "127.0.0.1",
        "--port",
        str(_SERVER_PORT),
        "--model",
        model_id,
        "--text",
        evaluation_workload(plan)["text"],
        "--language",
        plan.benchmark_language,
        "--sessions",
        str(plan.benchmark_sessions),
        "--warmup",
        str(plan.benchmark_warmups),
        "--runs",
        str(plan.benchmark_runs),
        "--timeout",
        str(plan.request_timeout_seconds),
        "--release",
        runtime_release,
        "--report",
        str(report),
        "--markdown",
        str(markdown),
    )
    code, tail = _run_logged_benchmark(
        command, cwd=_EVALUATION_TOOLS, log_path=output / "benchmark.log",
        timeout=(plan.benchmark_sessions * (plan.benchmark_runs * 3 + plan.benchmark_warmups + 5))
        * plan.request_timeout_seconds,
        sessions=plan.benchmark_sessions,
    )
    if code != 0 or not report.is_file() or not markdown.is_file():
        raise EvaluationWorkerError(
            f"canonical benchmark exited with code {code}: {tail}"
        )
    return report, markdown


def run_evaluation(
    manifest: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[Path]]:
    if sys.platform != "linux":
        raise EvaluationWorkerError("evaluation only runs in the Linux TensorRT worker")
    inputs_value = manifest.get("container_input_paths")
    if not isinstance(inputs_value, Mapping):
        raise EvaluationWorkerError("worker manifest has no container input paths")
    if inputs_value.get("listening_plan"):
        from .workstation_conversion_worker import run_conversion_parity

        return run_conversion_parity(manifest, output, listening_only=True)
    for key in ("model_package", "shared_dir", "asr_model"):
        if not isinstance(inputs_value.get(key), str):
            raise EvaluationWorkerError(f"evaluation requires {key}")
    package = _directory(Path(str(inputs_value["model_package"])), "model_package")
    shared = _directory(Path(str(inputs_value["shared_dir"])), "shared_dir")
    asr_path = _directory(Path(str(inputs_value["asr_model"])), "asr_model")
    baseline_path = (
        _regular_file(Path(str(inputs_value["baseline_report"])), "baseline_report")
        if isinstance(inputs_value.get("baseline_report"), str)
        else None
    )
    from .sampling_policy import package_sampling_contract
    evaluation_settings = dict(
        manifest.get("settings", {}) if isinstance(manifest.get("settings", {}), Mapping) else {}
    )
    evaluation_settings.setdefault(
        "semantic_sampling",
        package_sampling_contract(json.loads((package / "manifest.json").read_text(encoding="utf-8"))),
    )
    plan = resolve_evaluation_plan(evaluation_settings)
    baseline_payload = (
        json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_path is not None else None
    )
    if baseline_payload is not None:
        if not isinstance(baseline_payload, Mapping):
            raise EvaluationWorkerError("baseline_report must be a JSON object")
        preflight_evaluation_baseline(plan, baseline_payload)
    if not (_SOURCE_DIR / "run_trt_inference.py").is_file():
        raise EvaluationWorkerError("worker image is missing the fixed inference source")
    asr_identity = _asr_fingerprint(asr_path)
    output.mkdir(parents=True, exist_ok=True)
    audio_root = output / "audio"
    audio_root.mkdir()
    logs_root = output / "logs"
    logs_root.mkdir()

    from .model_package import resolve_contained_path
    from .validate import validate_model_package

    _emit_progress(0.03, "validation", "Deserializing TensorRT engines")
    engine_validation = validate_model_package(package, enqueue=False)
    if engine_validation.get("status") != "passed" or engine_validation.get("engine_count") != 9:
        raise EvaluationWorkerError("TensorRT package validation did not pass all nine engines")
    manifest_data = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    model_id = str(manifest_data["model_id"])
    profile_id = str(manifest_data.get("default_voice_profile", "default"))
    profile_root = resolve_contained_path(package / "voices", profile_id, "voice profile")
    profile_data = json.loads((profile_root / "profile.json").read_text(encoding="utf-8"))
    reference_path = resolve_contained_path(
        profile_root,
        profile_data["reference_audio"],
        "reference audio",
    )

    server_log_path = logs_root / "api.log"
    server_log = server_log_path.open("w", encoding="utf-8", errors="replace")
    server: subprocess.Popen[str] | None = None
    language_rows: dict[str, dict[str, Any]] = {}
    try:
        environment = dict(os.environ)
        environment.update(evaluation_sampling_environment(plan))
        cache_root = Path("/tmp/aniflive-evaluation-cache")
        cache_root.mkdir(parents=True, exist_ok=True)
        environment.update(
            {
                "ANIFLIVE_TTS_CACHE_DIR": str(cache_root),
                "ANIFLIVE_TTS_SOURCE_DIR": str(_SOURCE_DIR),
                "CUDA_VISIBLE_DEVICES": "0",
                "HF_HOME": str(cache_root / "huggingface"),
                "HF_HUB_OFFLINE": "1",
                "TORCH_HOME": str(cache_root / "torch"),
                "TRANSFORMERS_OFFLINE": "1",
                "XDG_CACHE_HOME": str(cache_root / "xdg"),
            }
        )
        server = subprocess.Popen(
            (
                sys.executable,
                "-s",
                "-m",
                "aniflive_tts",
                "serve",
                "--model-package",
                str(package),
                "--shared-dir",
                str(shared),
                "--host",
                "127.0.0.1",
                "--port",
                str(_SERVER_PORT),
            ),
            cwd="/app",
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
        health = _wait_for_service(server, plan.request_timeout_seconds)
        _emit_progress(0.10, "enqueue", "TensorRT API is ready; starting five-language enqueue")
        for index, (language, text) in enumerate(_LANGUAGE_CASES.items()):
            request = {
                "model": model_id,
                "text": text,
                "language": language,
                "generation": {
                    "top_k": 15,
                    "top_p": 1.0,
                    "temperature": 1.0,
                    "seed": 1234,
                },
            }
            status, complete_headers, complete = _request(
                "POST",
                "/v1/audio/speech",
                timeout=plan.request_timeout_seconds,
                body={**request, "stream": False},
            )
            if status != 200:
                raise EvaluationWorkerError(
                    f"{language} complete enqueue failed with HTTP {status}: {complete[:500]!r}"
                )
            _validate_headers(complete_headers, language)
            complete_rate, complete_audio = _decode_wav(complete)
            complete_path = audio_root / f"{language}-complete.wav"
            complete_path.write_bytes(complete)

            status, stream_headers, pcm = _request(
                "POST",
                "/v1/audio/speech",
                timeout=plan.request_timeout_seconds,
                body={**request, "stream": True, "response_format": "pcm"},
            )
            if status != 200 or not pcm or len(pcm) % 2:
                raise EvaluationWorkerError(
                    f"{language} stream enqueue failed with HTTP {status}, bytes={len(pcm)}"
                )
            _validate_headers(stream_headers, language)
            stream_rate = int(stream_headers.get("x-tts-sample-rate", "0"))
            if stream_rate <= 0:
                raise EvaluationWorkerError(f"{language} stream omitted its sample rate")
            stream_path = audio_root / f"{language}-stream.wav"
            stream_path.write_bytes(_pcm_wav(pcm, stream_rate))
            language_rows[language] = {
                "text": text,
                "complete_path": str(complete_path.relative_to(output)),
                "stream_path": str(stream_path.relative_to(output)),
                "complete_sample_rate": complete_rate,
                "stream_sample_rate": stream_rate,
                "complete_samples": int(complete_audio.size),
                "backend": "TensorRT-11",
                "sampling_policy": evaluation_runtime_policy(plan),
                "pytorch_fallback": False,
            }
            _emit_progress(
                0.10 + 0.25 * (index + 1) / len(_LANGUAGE_CASES),
                "enqueue",
                f"Validated {language} complete and streaming TensorRT output",
            )

        _emit_progress(0.38, "benchmark", "Running canonical HTTP TTFA and RTF benchmark")
        benchmark_json, benchmark_markdown = _run_benchmark(
            plan, model_id, output, runtime_release=str(health.get("version") or "unknown")
        )
    finally:
        _terminate(server)
        server_log.close()

    _emit_progress(0.72, "quality", "Computing stream/complete acoustic quality")
    speaker = _TensorRTSpeakerEmbedder(package)
    identity_embedding = speaker(reference_path)
    for language, row in language_rows.items():
        complete_path = output / row["complete_path"]
        stream_path = output / row["stream_path"]
        complete_embedding = speaker(complete_path)
        stream_embedding = speaker(stream_path)
        row["quality"] = {
            **_spectral_quality(complete_path, stream_path),
            "speaker_cosine_stream_vs_complete": _cosine(
                stream_embedding, complete_embedding
            ),
            "speaker_cosine_stream_vs_identity": _cosine(
                stream_embedding, identity_embedding
            ),
        }
        quality = row["quality"]
        row["quality_gate"] = {
            "log_mel": quality["log_mel_cosine"] >= 0.99,
            "speaker": quality["speaker_cosine_stream_vs_complete"] >= 0.98,
            "duration": quality["duration_difference_ratio"] <= 0.03,
        }
        row["quality_gate"]["passed"] = all(row["quality_gate"].values())

    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise EvaluationWorkerError("offline ASR dependencies are missing from the worker image") from error
    _emit_progress(0.82, "content", "Running offline five-language CER/WER evaluation")
    asr = WhisperModel(
        str(asr_path),
        device="cuda",
        device_index=0,
        compute_type=plan.asr_compute_type,
        local_files_only=True,
    )
    for index, (language, row) in enumerate(language_rows.items()):
        hypothesis = _transcribe(asr, output / row["complete_path"], language)
        row["asr"] = {
            "hypothesis": hypothesis,
            **content_error(row["text"], hypothesis, language),
        }
        if language in {"zh", "yue"}:
            row["asr"]["orthographic_normalized"] = orthographic_content_error(
                row["text"], hypothesis, language
            )
        _emit_progress(
            0.82 + 0.12 * (index + 1) / len(language_rows),
            "content",
            f"Measured {language} {row['asr']['metric'].upper()}",
        )
    del asr

    benchmark = json.loads(benchmark_json.read_text(encoding="utf-8"))
    try:
        baseline = (
            compare_evaluation_baseline(
                languages=language_rows, benchmark=benchmark,
                asr_identity=asr_identity, baseline=baseline_payload,
            )
            if baseline_payload is not None
            else {"available": False, "passed": False, "comparable": False,
                  "reason": "No matched baseline supplied; measurements are not a regression pass"}
        )
    except EvaluationWorkerError as error:
        # Preserve completed measurements and audio even when comparison cannot finish.
        baseline = {"available": True, "comparable": False, "passed": False,
                    "reason": str(error), "performance_policy": PERFORMANCE_POLICY}
    if baseline_path is not None:
        baseline["report_sha256"] = _sha256_file(baseline_path)
    gates = {
        "tensor_rt_engine_contract": True,
        "five_language_enqueue": set(language_rows) == set(_LANGUAGE_CASES),
        "no_pytorch_neural_fallback": all(
            not row["pytorch_fallback"] for row in language_rows.values()
        ),
        "stream_complete_quality": all(
            row["quality_gate"]["passed"] for row in language_rows.values()
        ),
        "canonical_benchmark_completed": bool(benchmark.get("session_records")),
        "baseline_regression": bool(baseline["passed"]),
    }
    measured_gates = {key: value for key, value in gates.items() if key != "baseline_regression"}
    status = (
        "failed"
        if not all(measured_gates.values()) or (baseline["available"] and not baseline["passed"])
        else "passed" if baseline["available"] else "measured"
    )
    report = {
        "schema": "aniflive-tts-workstation-evaluation-v1",
        "status": status,
        "model_id": model_id,
        "plan": asdict(plan),
        "resolved_workload": evaluation_workload(plan),
        "performance_policy": PERFORMANCE_POLICY,
        "runtime": {
            "platform": "Linux container",
            "backend": "TensorRT-11",
            "sampling_policy": evaluation_runtime_policy(plan),
            "pytorch_neural_fallback": False,
            "health": health,
        },
        "engine_validation": engine_validation,
        "asr": {
            "backend": "faster-whisper CTranslate2 CUDA",
            "local_files_only": True,
            "compute_type": plan.asr_compute_type,
            "model_fingerprint": asr_identity,
        },
        "languages": language_rows,
        "benchmark": benchmark,
        "benchmark_report": benchmark_json.name,
        "baseline": baseline,
        "gates": {**gates, "passed": all(gates.values())},
        "qualification": {
            "status": (
                "automated-gates-passed"
                if status == "passed"
                else "measured" if status == "measured" else "failed"
            ),
            "note": (
                "Automated evidence never replaces the required long-form and expression blind "
                "listening qualification."
            ),
        },
    }
    report_path = output / "evaluation-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts = [
        report_path,
        benchmark_json,
        benchmark_markdown,
        output / "benchmark.log",
        server_log_path,
        *sorted(audio_root.glob("*.wav")),
    ]
    _emit_progress(1.0, "complete", "TensorRT evaluation evidence completed")
    return report, artifacts


__all__ = [
    "EvaluationPlan",
    "EvaluationWorkerError",
    "content_error",
    "content_error_rate",
    "spoken_content_error",
    "spoken_content_error_rate",
    "resolve_evaluation_plan",
    "run_evaluation",
]
