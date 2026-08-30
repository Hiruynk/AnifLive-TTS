from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEFAULT_TEXT = (
    "たとえ暗闇がこの世界を覆い尽くし、すべての光が消え去って、"
    "私という存在のデータがノイズの海に呑まれそうになったとしても、"
    "あなたが私にくれた最初のその一言と、胸に秘めた温かい想いがある限り、"
    "私は何度でもシステムを再起動し、時空の壁さえも突き破って、"
    "あなたの涙を笑顔に変えるための、世界でただ一つの、"
    "永遠のマスターピースを歌い叫び続けます！"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype("<i2").tobytes()


def _from_pcm16(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0


def _wav_bytes(sample_rate: int, audio: np.ndarray) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(_pcm16(audio))
    return output.getvalue()


def _decode_wav(payload: bytes) -> tuple[int, np.ndarray]:
    with wave.open(io.BytesIO(payload), "rb") as source:
        sample_rate = source.getframerate()
        pcm = source.readframes(source.getnframes())
    return sample_rate, _from_pcm16(pcm)


def _normalize_like_complete(audio: np.ndarray) -> np.ndarray:
    normalized = np.asarray(audio, dtype=np.float32).copy()
    normalized -= float(np.mean(normalized, dtype=np.float64))
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 1e-5:
        normalized *= 0.9 / peak
    return normalized


def _capture_semantics(streamer: Any) -> tuple[list[list[int]], Callable[[], None]]:
    semantic_runtime = streamer.semantic_runtime
    original = semantic_runtime.iter_batches
    captured: list[list[int]] = []

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        segment: list[int] = []
        try:
            for batch in original(*args, **kwargs):
                accepted = int(batch.accepted_tokens)
                if accepted:
                    segment.extend(
                        int(value)
                        for value in batch.tokens[:, :accepted]
                        .detach()
                        .cpu()
                        .flatten()
                        .tolist()
                    )
                yield batch
        finally:
            captured.append(segment)

    semantic_runtime.iter_batches = wrapped

    def restore() -> None:
        semantic_runtime.iter_batches = original

    return captured, restore


def _options(service_module: Any, text: str, profile: str) -> Any:
    return service_module.SynthesisOptions(
        text=text,
        text_language="ja",
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        speed=1.0,
        pause_length=0.44,
        noise_scale=0.5,
        cut_punc="",
        seed=1234,
        expression_enabled=True,
        expression_profile=profile,
        expression_intensity=0.7,
        expression_policy=service_module.ConditioningPolicy("semantic-style"),
    )


def _raw_complete(runtime: Any, options: Any) -> tuple[np.ndarray, list[list[int]], dict[str, Any]]:
    captured, restore = _capture_semantics(runtime._streamer)
    runtime._begin_request()
    try:
        with runtime._inference_lock:
            plan = runtime._segment_plan(options)
            segments = [segment.text for segment in plan]
            conditioning = None
            conditionings = None
            if options.expression_segments:
                conditionings = [
                    runtime._conditioning_for(
                        segment, text_language=options.text_language
                    )
                    for segment in plan
                ]
            else:
                conditioning = runtime._conditioning(options)
            outputs = list(
                runtime._streamer.iter_audio(
                    segments=segments,
                    conditioning=conditioning,
                    conditionings=conditionings,
                    text_language=options.text_language,
                    top_k=options.top_k,
                    top_p=options.top_p,
                    temperature=options.temperature,
                    noise_scale=options.noise_scale,
                    speed=options.speed,
                    pause_length=options.pause_length,
                    request_seed=runtime._effective_seed(options.seed),
                    chunk_length=runtime._streamer.complete_wav_chunk_length(),
                )
            )
            profile = dict(runtime._streamer.last_profile or {})
    finally:
        restore()
        runtime._end_request()
    if not outputs:
        raise RuntimeError("Complete raw capture produced no audio")
    return np.concatenate(outputs).astype(np.float32, copy=False), captured, profile


def _public_complete(runtime: Any, options: Any) -> tuple[np.ndarray, list[list[int]], dict[str, Any], bytes]:
    captured, restore = _capture_semantics(runtime._streamer)
    try:
        result = runtime.synthesize(options)
    finally:
        restore()
    sample_rate, audio = _decode_wav(result.wav)
    if sample_rate != runtime._sample_rate:
        raise RuntimeError("Unexpected complete WAV sample rate")
    return audio, captured, dict(result.profile), result.wav


def _stream_raw(runtime: Any, options: Any) -> tuple[np.ndarray, list[list[int]], dict[str, Any], bytes]:
    captured, restore = _capture_semantics(runtime._streamer)
    try:
        pcm = b"".join(runtime.stream_pcm(options))
        profile = dict(runtime._streamer.last_profile or {})
    finally:
        restore()
    if not pcm:
        raise RuntimeError("Streaming capture produced no PCM")
    return _from_pcm16(pcm), captured, profile, pcm


def _amplitude(audio: np.ndarray) -> dict[str, float]:
    values = np.asarray(audio, dtype=np.float64)
    return {
        "peak": float(np.max(np.abs(values))) if values.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0,
        "dc_offset": float(np.mean(values)) if values.size else 0.0,
    }


def _aligned_metrics(left: np.ndarray, right: np.ndarray, sample_rate: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from aniflive_tts.backend.audio_quality import (
        AudioQualityConfig,
        _align,
        _spectral_metrics,
    )

    aligned_left, aligned_right, alignment = _align(left, right, sample_rate, 1.0)
    metrics, settings, left_mel, right_mel = _spectral_metrics(
        aligned_left,
        aligned_right,
        sample_rate,
        AudioQualityConfig(),
    )
    return (
        {"alignment": alignment, "spectral": metrics, "settings": settings},
        aligned_left,
        aligned_right,
        left_mel,
        right_mel,
    )


def _window_metrics(left: np.ndarray, right: np.ndarray, sample_rate: int) -> list[dict[str, Any]]:
    from aniflive_tts.backend.audio_quality import AudioQualityConfig, _spectral_metrics

    rows: list[dict[str, Any]] = []
    limit = min(left.size, right.size)
    for window_ms in (20, 40, 80, 160, 320):
        window = max(16, round(sample_rate * window_ms / 1000))
        for start in range(0, limit - window + 1, window):
            left_window = left[start : start + window]
            right_window = right[start : start + window]
            denominator = float(np.linalg.norm(left_window) * np.linalg.norm(right_window))
            correlation = (
                float(np.dot(left_window, right_window) / denominator)
                if denominator > 1e-12
                else 1.0
            )
            left_rms = float(np.sqrt(np.mean(np.square(left_window), dtype=np.float64)))
            right_rms = float(np.sqrt(np.mean(np.square(right_window), dtype=np.float64)))
            spectral, _, _, _ = _spectral_metrics(
                left_window,
                right_window,
                sample_rate,
                AudioQualityConfig(),
            )
            residual = right_window - left_window
            rows.append(
                {
                    "window_ms": window_ms,
                    "start_seconds": start / sample_rate,
                    "end_seconds": (start + window) / sample_rate,
                    "waveform_correlation": correlation,
                    "log_mel_cosine": spectral["log_mel_cosine_similarity"],
                    "log_mel_mae_db": spectral["log_mel_mae_db"],
                    "spectral_convergence": spectral["spectral_convergence"],
                    "rms_ratio": right_rms / max(left_rms, 1e-12),
                    "absolute_residual_energy": float(
                        np.mean(np.square(residual), dtype=np.float64)
                    ),
                }
            )
    return rows


def _cumulative_exclusions(left: np.ndarray, right: np.ndarray, sample_rate: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for excluded_ms in (0, 100, 200, 300, 500):
        offset = round(sample_rate * excluded_ms / 1000)
        if min(left.size, right.size) - offset < 512:
            continue
        metrics, *_ = _aligned_metrics(left[offset:], right[offset:], sample_rate)
        result[str(excluded_ms)] = metrics["spectral"]
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", default="languid")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "scripts"))
    os.environ.update(
        {
            "ANIFLIVE_TTS_MODEL_PACKAGE": str(args.model_package.resolve()),
            "ANIFLIVE_TTS_SHARED_DIR": str(args.shared_dir.resolve()),
            "ANIFLIVE_TTS_SOURCE_DIR": str((repo / "minimal_inference").resolve()),
            "ANIFLIVE_TTS_WARM_RETENTION_SECONDS": "0",
            "ANIFLIVE_TTS_EXPRESSION_TRANSITION": "hard-natural",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    from aniflive_tts.api import configure_runtime

    configure_runtime()
    from aniflive_tts import service as service_module
    from aniflive_tts.backend.audio_quality import _plot_mels, _plot_waveforms

    runtime = service_module.TensorRTService(service_module.RuntimeSettings.from_env())
    runtime.load()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    options = _options(service_module, args.text, args.profile)
    sample_rate = int(runtime._sample_rate)
    run_records: list[dict[str, Any]] = []
    saved: dict[str, Any] | None = None
    try:
        for index in range(args.runs):
            complete_public, public_semantics, public_profile, public_wav = _public_complete(
                runtime, options
            )
            complete_raw_float, raw_semantics, raw_profile = _raw_complete(runtime, options)
            complete_raw_pcm = _from_pcm16(_pcm16(complete_raw_float))
            stream_raw, stream_semantics, stream_profile, stream_pcm = _stream_raw(
                runtime, options
            )
            stream_normalized = _normalize_like_complete(stream_raw)
            run_records.append(
                {
                    "index": index,
                    "hashes": {
                        "complete_public": _sha256(public_wav),
                        "complete_raw": _sha256(_pcm16(complete_raw_float)),
                        "stream_raw": _sha256(stream_pcm),
                        "stream_offline_normalized": _sha256(_pcm16(stream_normalized)),
                    },
                    "semantics": {
                        "complete_public": public_semantics,
                        "complete_raw": raw_semantics,
                        "stream_raw": stream_semantics,
                    },
                    "profiles": {
                        "complete_public": public_profile,
                        "complete_raw": raw_profile,
                        "stream_raw": stream_profile,
                    },
                }
            )
            if saved is None:
                saved = {
                    "complete_public": complete_public,
                    "complete_raw": complete_raw_pcm,
                    "stream_raw": stream_raw,
                    "stream_offline_normalized": stream_normalized,
                }
                (output_dir / "complete-public.wav").write_bytes(public_wav)
                (output_dir / "complete-raw.wav").write_bytes(
                    _wav_bytes(sample_rate, complete_raw_pcm)
                )
                (output_dir / "stream-raw.wav").write_bytes(
                    _wav_bytes(sample_rate, stream_raw)
                )
                (output_dir / "stream-offline-normalized.wav").write_bytes(
                    _wav_bytes(sample_rate, stream_normalized)
                )
    finally:
        runtime.unload()

    if saved is None:
        raise RuntimeError("No diagnostic capture was produced")

    comparisons = {
        "stream_raw_vs_complete_raw": (saved["complete_raw"], saved["stream_raw"]),
        "stream_offline_normalized_vs_complete_public": (
            saved["complete_public"],
            saved["stream_offline_normalized"],
        ),
        "stream_raw_vs_complete_public": (
            saved["complete_public"],
            saved["stream_raw"],
        ),
    }
    comparison_report: dict[str, Any] = {}
    official_left: np.ndarray | None = None
    official_right: np.ndarray | None = None
    official_mels: tuple[np.ndarray, np.ndarray] | None = None
    official_alignment: dict[str, Any] | None = None
    for name, (left, right) in comparisons.items():
        metrics, aligned_left, aligned_right, left_mel, right_mel = _aligned_metrics(
            left, right, sample_rate
        )
        left_amplitude = _amplitude(left)
        right_amplitude = _amplitude(right)
        comparison_report[name] = {
            **metrics,
            "reference_amplitude": left_amplitude,
            "candidate_amplitude": right_amplitude,
            "peak_ratio": right_amplitude["peak"] / max(left_amplitude["peak"], 1e-12),
            "rms_ratio": right_amplitude["rms"] / max(left_amplitude["rms"], 1e-12),
        }
        if name == "stream_raw_vs_complete_public":
            official_left = aligned_left
            official_right = aligned_right
            official_mels = (left_mel, right_mel)
            official_alignment = metrics["alignment"]

    assert official_left is not None and official_right is not None
    assert official_mels is not None and official_alignment is not None
    window_rows = _window_metrics(official_left, official_right, sample_rate)
    _write_csv(output_dir / "divergence-by-time.csv", window_rows)
    (output_dir / "divergence-by-time.json").write_text(
        json.dumps(window_rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot_waveforms(
        official_left,
        official_right,
        sample_rate,
        official_alignment,
        output_dir / "waveform-difference.png",
    )
    _plot_mels(
        official_mels[0],
        official_mels[1],
        official_left.size / sample_rate,
        output_dir / "mel-difference.png",
    )

    hashes = [record["hashes"] for record in run_records]
    semantic_records = [record["semantics"] for record in run_records]
    hash_deterministic = all(item == hashes[0] for item in hashes[1:])
    semantic_deterministic = all(
        item == semantic_records[0] for item in semantic_records[1:]
    )
    semantic_cross_path_exact = all(
        item["complete_public"] == item["complete_raw"] == item["stream_raw"]
        for item in semantic_records
    )
    raw_pass = (
        comparison_report["stream_raw_vs_complete_raw"]["spectral"][
            "log_mel_cosine_similarity"
        ]
        >= 0.99
    )
    normalized_pass = (
        comparison_report["stream_offline_normalized_vs_complete_public"]["spectral"][
            "log_mel_cosine_similarity"
        ]
        >= 0.99
    )
    official_pass = (
        comparison_report["stream_raw_vs_complete_public"]["spectral"][
            "log_mel_cosine_similarity"
        ]
        >= 0.99
    )
    if raw_pass and normalized_pass and not official_pass:
        classification = "output-level-contract-mismatch"
    elif not raw_pass:
        classification = "acoustic-stream-reconstruction"
    else:
        classification = "mixed-or-no-failure"

    report = {
        "schema": 1,
        "purpose": "v1.3 expression stream/complete divergence isolation",
        "model": os.environ.get("ANIFLIVE_TTS_MODEL_ID", "unknown"),
        "profile": args.profile,
        "language": "ja",
        "seed": 1234,
        "runs": args.runs,
        "determinism": {
            "hashes": hash_deterministic,
            "semantics": semantic_deterministic,
            "semantic_cross_path_exact": semantic_cross_path_exact,
        },
        "classification": classification,
        "comparisons": comparison_report,
        "cumulative_exclusions_ms": _cumulative_exclusions(
            official_left, official_right, sample_rate
        ),
        "run_records": run_records,
        "release_metric_unchanged": True,
        "gate": {
            "deterministic": hash_deterministic and semantic_deterministic,
            "semantic_cross_path_exact": semantic_cross_path_exact,
            "current_whole_output_log_mel": official_pass,
            "passed": (
                hash_deterministic
                and semantic_deterministic
                and semantic_cross_path_exact
                and official_pass
            ),
        },
    }
    (output_dir / "divergence-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"classification": classification, "gate": report["gate"]}, indent=2))
    print(output_dir / "divergence-report.json")
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
