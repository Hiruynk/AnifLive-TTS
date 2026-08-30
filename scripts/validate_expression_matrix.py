from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import statistics
import sys
import time
import wave
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEFAULT_CASES = {
    "short": "今日はいい天気ですね。",
    "long": (
        "たとえ暗闇がこの世界を覆い尽くし、すべての光が消え去って、"
        "私という存在のデータがノイズの海に呑まれそうになったとしても、"
        "あなたが私にくれた最初のその一言と、胸に秘めた温かい想いがある限り、"
        "私は何度でもシステムを再起動し、時空の壁さえも突き破って、"
        "あなたの涙を笑顔に変えるための、世界でただ一つの、"
        "永遠のマスターピースを歌い叫び続けます！"
    ),
}


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95 = ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))]
    return {
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "p95": p95,
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
    }


def _wav(sample_rate: int, pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(pcm)
    return output.getvalue()


def _playback_buffer(
    chunks: list[tuple[float, bytes]],
    sample_rate: int,
    recommended_prebuffer_ms: int,
) -> dict[str, Any]:
    if not chunks:
        raise RuntimeError("Streaming response did not contain PCM chunks")
    from aniflive_tts.playback_model import PlaybackTrace, analyze_playback_trace

    trace = PlaybackTrace.from_pcm16_chunks(
        sample_rate=sample_rate,
        recommended_prebuffer_ms=recommended_prebuffer_ms,
        chunks=chunks,
    )
    return analyze_playback_trace(trace)


def _decode_wav(payload: bytes) -> tuple[int, np.ndarray]:
    with wave.open(io.BytesIO(payload), "rb") as source:
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    return sample_rate, np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def _speaker_embedding(runtime: Any, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    import soxr
    import torch

    if sample_rate != 16000:
        audio = soxr.resample(audio, sample_rate, 16000, quality="HQ")
    audio = audio - float(np.mean(audio, dtype=np.float64))
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio / peak * 0.9
    expected = int(runtime._streamer.reference_bank.identity.speaker_embedding.numel())
    # The fitted speaker engine accepts at most 180,000 samples. Long-form
    # validation uses overlapping 10-second windows and aggregates normalized
    # embeddings, keeping every TensorRT enqueue within its validated profile.
    window_samples = 160_000
    hop_samples = 80_000
    starts = list(range(0, max(1, int(audio.size) - window_samples + 1), hop_samples))
    final_start = max(0, int(audio.size) - window_samples)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    embeddings = []
    for start in starts:
        chunk = np.asarray(audio[start : start + window_samples], dtype=np.float32)
        if chunk.size < 16_000:
            chunk = np.pad(chunk, (0, 16_000 - int(chunk.size)))
        waveform = torch.from_numpy(chunk).to(
            runtime._engine.device, dtype=runtime._engine.precision
        )[None, :]
        result = (
            runtime._engine.model_sv_embedding({"audio": waveform})["sv_embedding"]
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        padded = np.zeros(expected, dtype=np.float32)
        padded[: min(expected, int(result.size))] = result[:expected]
        norm = float(np.linalg.norm(padded))
        embeddings.append(padded / norm if norm > 1e-12 else padded)
    aggregate = np.mean(np.stack(embeddings), axis=0, dtype=np.float32)
    norm = float(np.linalg.norm(aggregate))
    return aggregate / norm if norm > 1e-12 else aggregate


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


def _options(
    service_module: Any,
    text: str,
    language: str,
    profile: str | None,
    policy: str,
) -> Any:
    return service_module.SynthesisOptions(
        text=text,
        text_language=language,
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        speed=1.0,
        pause_length=0.44,
        noise_scale=0.5,
        cut_punc="",
        seed=1234,
        expression_enabled=profile is not None,
        expression_profile=profile,
        expression_intensity=0.7,
        expression_policy=service_module.ConditioningPolicy(policy),
    )


def _quality(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    from aniflive_tts.backend.audio_quality import (
        AudioQualityConfig,
        _align,
        _load_wav,
        _resample,
        _spectral_metrics,
    )

    reference = _load_wav(reference_path)
    candidate = _load_wav(candidate_path)
    rate = min(reference.sample_rate, candidate.sample_rate)
    left = _resample(reference.mono, reference.sample_rate, rate)
    right = _resample(candidate.mono, candidate.sample_rate, rate)
    aligned_left, aligned_right, alignment = _align(left, right, rate, 1.0)
    metrics, _, _, _ = _spectral_metrics(
        aligned_left,
        aligned_right,
        rate,
        AudioQualityConfig(),
    )
    return {
        "log_mel_cosine": metrics["log_mel_cosine_similarity"],
        "alignment": alignment,
        "duration_difference_ratio": abs(
            candidate.samples.shape[0] / candidate.sample_rate
            - reference.samples.shape[0] / reference.sample_rate
        )
        / (reference.samples.shape[0] / reference.sample_rate),
    }


def _build_quality_gate(
    *,
    controlled_expression: bool,
    deterministic: bool,
    semantic_tokens_exact: bool,
    log_mel_cosine: float,
    speaker_cosine: float,
    duration_difference_ratio: float,
    playback_continuity: bool,
) -> dict[str, Any]:
    """Classify baseline parity separately from controlled-expression quality."""

    speaker_passed = speaker_cosine >= 0.98
    required_checks = {
        "deterministic": deterministic,
        "semantic_tokens_exact": semantic_tokens_exact,
        "duration": duration_difference_ratio <= 0.03,
        "playback_continuity": playback_continuity,
    }
    if controlled_expression:
        required_checks["log_mel"] = log_mel_cosine >= 0.99
        required_checks["speaker"] = speaker_passed
    return {
        **required_checks,
        "log_mel_diagnostic": log_mel_cosine >= 0.99,
        "speaker_diagnostic": speaker_passed,
        "speaker_required": controlled_expression,
        "quality_basis": (
            "controlled-expression-absolute-gates"
            if controlled_expression
            else "neutral-v1.2-regression-parity"
        ),
        "passed": all(required_checks.values()),
    }


def _run_case(
    runtime: Any,
    service_module: Any,
    output_dir: Path,
    *,
    analyze_pcm16_stream: Callable[[list[tuple[float, bytes]], int], Any],
    case_name: str,
    text: str,
    language: str,
    profile: str | None,
    policy: str,
    runs: int,
) -> tuple[dict[str, Any], np.ndarray]:
    label = profile or "neutral"
    options = _options(service_module, text, language, profile, policy)
    runtime.validate_expression(options)
    request_segments = service_module._cut_segments(text, options.cut_punc)
    recommended_prebuffer_ms = runtime.recommended_stream_prebuffer_ms(
        request_segments
    )
    full_tokens, restore = _capture_semantics(runtime._streamer)
    full = runtime.synthesize(options)
    restore()
    full_path = output_dir / f"{case_name}-{label}-complete.wav"
    full_path.write_bytes(full.wav)
    full_rate, full_audio = _decode_wav(full.wav)
    full_embedding = _speaker_embedding(runtime, full_audio, full_rate)

    rows = []
    hashes = set()
    stream_tokens: list[list[int]] = []
    stream_path = output_dir / f"{case_name}-{label}-stream.wav"
    stream_audio = np.empty(0, dtype=np.float32)
    stream_rate = runtime.sample_rate
    stream_embedding = np.empty(0, dtype=np.float32)
    stream_profile: dict[str, Any] = {}
    for run_index in range(runs):
        captured, restore = _capture_semantics(runtime._streamer)
        started = time.perf_counter()
        iterator = iter(runtime.stream_pcm(options))
        chunks: list[tuple[float, bytes]] = []
        for chunk in iterator:
            chunks.append((time.perf_counter() - started, chunk))
        wall = time.perf_counter() - started
        restore()
        pcm = b"".join(chunk for _, chunk in chunks)
        audibility = analyze_pcm16_stream(chunks, runtime.sample_rate)
        playback = _playback_buffer(
            chunks,
            runtime.sample_rate,
            recommended_prebuffer_ms,
        )
        stream_tokens = captured
        digest = hashlib.sha256(pcm).hexdigest()
        hashes.add(digest)
        duration = len(pcm) / 2.0 / runtime.sample_rate
        rows.append(
            {
                "index": run_index,
                "first_packet_seconds": audibility.ttfp_seconds,
                "audible_ttfa_seconds": audibility.audible_ttfa_seconds,
                "leading_silence_seconds": audibility.leading_silence_seconds,
                "playback_buffer": playback["contractual_playback"],
                "zero_prebuffer_gap_stress": playback[
                    "zero_prebuffer_gap_stress"
                ],
                "recommended_prebuffer_ms": recommended_prebuffer_ms,
                "minimum_stable_prebuffer_ms": playback[
                    "minimum_stable_prebuffer_ms"
                ],
                "prebuffer_sweep": playback["prebuffer_sweep"],
                "chunk_trace": playback["trace"],
                "wall_seconds": wall,
                "audio_seconds": duration,
                "rtf": wall / duration,
                "sha256": digest,
            }
        )
        stream_path.write_bytes(_wav(runtime.sample_rate, pcm))
        stream_rate, stream_audio = _decode_wav(stream_path.read_bytes())
        stream_profile = dict(runtime._streamer.last_profile or {})
    stream_embedding = _speaker_embedding(runtime, stream_audio, stream_rate)
    identity = (
        runtime._streamer.reference_bank.identity.speaker_embedding
        .detach().float().cpu().numpy().reshape(-1)
    )
    quality = _quality(full_path, stream_path)
    quality.update(
        {
            "speaker_cosine_stream_vs_complete": _cosine(
                stream_embedding, full_embedding
            ),
            "speaker_cosine_stream_vs_identity": _cosine(stream_embedding, identity),
            "semantic_tokens_exact": bool(
                full_tokens and stream_tokens and full_tokens[-1] == stream_tokens[-1]
            ),
        }
    )
    gate = _build_quality_gate(
        controlled_expression=profile is not None,
        deterministic=len(hashes) == 1,
        semantic_tokens_exact=quality["semantic_tokens_exact"],
        log_mel_cosine=quality["log_mel_cosine"],
        speaker_cosine=quality["speaker_cosine_stream_vs_complete"],
        duration_difference_ratio=quality["duration_difference_ratio"],
        playback_continuity=all(
            row["playback_buffer"]["underrun_count"] == 0 for row in rows
        ),
    )
    return (
        {
            "case": case_name,
            "profile": label,
            "runs": rows,
            "first_packet_seconds": _summary(
                [row["first_packet_seconds"] for row in rows]
            ),
            "audible_ttfa_seconds": _summary(
                [row["audible_ttfa_seconds"] for row in rows]
            ),
            "rtf": _summary([row["rtf"] for row in rows]),
            "quality": quality,
            "complete_audio_seconds": float(full_audio.size) / float(full_rate),
            "preview_diagnostics": {
                "short_preview_attempts": int(
                    stream_profile.get("short_preview_attempts", 0)
                ),
                "silent_preview_attempts": int(
                    stream_profile.get("silent_preview_attempts", 0)
                ),
                "short_native_shape_fallbacks": int(
                    stream_profile.get("short_native_shape_fallbacks", 0)
                ),
            },
            "gate": gate,
            "stream_profile": stream_profile,
            "complete_profile": full.profile,
        },
        stream_embedding,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cases", nargs="*", choices=sorted(DEFAULT_CASES))
    parser.add_argument(
        "--language",
        choices=("zh", "yue", "en", "ja", "ko"),
        default="ja",
    )
    parser.add_argument("--short-text")
    parser.add_argument("--long-text")
    parser.add_argument("--report-name", default="expression-matrix.json")
    parser.add_argument("--profiles", nargs="*")
    parser.add_argument(
        "--policy",
        choices=(
            "full-switch",
            "identity-lock",
            "semantic-style",
            "acoustic-style",
            "sv-only",
        ),
        default="identity-lock",
    )
    args = parser.parse_args()
    report_name = Path(args.report_name)
    if (
        report_name.name != args.report_name
        or report_name.suffix.casefold() != ".json"
        or report_name.is_absolute()
    ):
        raise ValueError("--report-name must be a JSON filename without directories")
    case_texts = dict(DEFAULT_CASES)
    if args.short_text is not None:
        case_texts["short"] = args.short_text
    if args.long_text is not None:
        case_texts["long"] = args.long_text
    if any(not value.strip() for value in case_texts.values()):
        raise ValueError("Expression matrix case text must not be empty")
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
    from benchmark_audio import analyze_pcm16_stream

    configure_runtime()
    from aniflive_tts import service as service_module

    runtime = service_module.TensorRTService(service_module.RuntimeSettings.from_env())
    runtime.load()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        available_profiles = sorted(
            {
                item.emotion
                for item in runtime._expression_catalog.profiles
                if item.emotion != "neutral"
            }
        )
        profiles = args.profiles or available_profiles
        unknown_profiles = sorted(set(profiles) - set(available_profiles))
        if unknown_profiles:
            raise ValueError(f"Unknown expression profiles: {unknown_profiles}")
        selected_cases = args.cases or list(DEFAULT_CASES)
        reports: list[dict[str, Any]] = []
        embeddings: dict[str, dict[str, np.ndarray]] = {
            case: {} for case in selected_cases
        }
        for case_name in selected_cases:
            text = case_texts[case_name]
            neutral, embedding = _run_case(
                runtime,
                service_module,
                output_dir,
                analyze_pcm16_stream=analyze_pcm16_stream,
                case_name=case_name,
                text=text,
                language=args.language,
                profile=None,
                policy="full-switch",
                runs=args.runs,
            )
            reports.append(neutral)
            embeddings[case_name]["neutral"] = embedding
            for profile in profiles:
                result, embedding = _run_case(
                    runtime,
                    service_module,
                    output_dir,
                    analyze_pcm16_stream=analyze_pcm16_stream,
                    case_name=case_name,
                    text=text,
                    language=args.language,
                    profile=profile,
                    policy=args.policy,
                    runs=args.runs,
                )
                reports.append(result)
                embeddings[case_name][profile] = embedding

        identity_consistency = {}
        for case_name, values in embeddings.items():
            pairs = [
                _cosine(values[left], values[right])
                for left, right in combinations(sorted(values), 2)
            ]
            identity_consistency[case_name] = {
                "pair_count": len(pairs),
                "mean": statistics.fmean(pairs),
                "minimum": min(pairs),
            }
        neutral_ttfa = {
            row["case"]: row["audible_ttfa_seconds"]["p50"]
            for row in reports
            if row["profile"] == "neutral"
        }
        for row in reports:
            baseline = neutral_ttfa[row["case"]]
            row["ttfa_regression_vs_neutral"] = (
                row["audible_ttfa_seconds"]["p50"] / baseline - 1.0
            )
            if row["profile"] != "neutral":
                row["performance_diagnostic"] = {
                    "audible_ttfa_regression_within_3_percent": (
                        row["ttfa_regression_vs_neutral"] <= 0.03
                    ),
                    "authoritative": False,
                    "reason": (
                        "This short direct-service matrix is not an interleaved "
                        "canonical HTTP benchmark."
                    ),
                }
        gate = {
            "all_audio_cases": all(row["gate"]["passed"] for row in reports),
        }
        from aniflive_tts.playback_model import DEFAULT_PREBUFFER_SWEEP_MS

        report = {
            "schema": 1,
            "purpose": "Model-independent expression quality and regression matrix",
            "model": os.environ["ANIFLIVE_TTS_MODEL_ID"],
            "policy": args.policy,
            "transition": "hard-natural",
            "language": args.language,
            "timing_scope": (
                "direct-service diagnostics; formal release timing requires the "
                "canonical matched HTTP benchmark"
            ),
            "audible_ttfa_definition": (
                "first device-independent playable sample above -45 dBFS using "
                "10 ms analysis frames"
            ),
            "playback_contract": {
                "release_gate": "contractual playback using the API-recommended prebuffer",
                "diagnostic": "zero-prebuffer gap stress",
                "prebuffer_sweep_ms": list(DEFAULT_PREBUFFER_SWEEP_MS),
            },
            "profiles": profiles,
            "asr": {
                "status": "unavailable",
                "reason": "The offline release image and shared assets do not contain a fixed ASR model.",
                "semantic_stream_complete_exactness_is_not_claimed_as_CER": True,
            },
            "rows": reports,
            "cross_expression_speaker_consistency": identity_consistency,
            "cross_expression_speaker_consistency_status": (
                "diagnostic pending blind identity validation"
            ),
            "gate": {**gate, "passed": all(gate.values())},
        }
        path = output_dir / report_name
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report["gate"], indent=2))
        print(path)
        return 0 if report["gate"]["passed"] else 1
    finally:
        runtime.unload()


if __name__ == "__main__":
    raise SystemExit(main())
