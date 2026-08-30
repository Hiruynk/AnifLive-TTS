from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import soxr
import torch


POLICIES = ("full-switch", "identity-lock", "semantic-style", "acoustic-style", "sv-only")


def _decode_wav(payload: bytes) -> tuple[int, np.ndarray]:
    with wave.open(io.BytesIO(payload), "rb") as source:
        sample_rate = source.getframerate()
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        frames = source.readframes(source.getnframes())
    if channels != 1 or sample_width != 2:
        raise RuntimeError("Expected mono PCM16 WAV")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return sample_rate, audio


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(np.float32).eps:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _leading_silence_seconds(audio: np.ndarray, sample_rate: int) -> float:
    frame_samples = max(1, int(round(sample_rate * 0.01)))
    threshold = 10.0 ** (-45.0 / 20.0)
    for begin in range(0, int(audio.size), frame_samples):
        frame = audio[begin : begin + frame_samples].astype(np.float64, copy=False)
        if frame.size and float(np.sqrt(np.mean(np.square(frame)))) > threshold:
            return begin / sample_rate
    return audio.size / sample_rate


def _log_spectrum(audio: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(audio)
    spectrum = torch.stft(
        tensor,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=torch.hann_window(1024),
        return_complex=True,
    ).abs()
    return torch.log1p(spectrum).numpy()


def _aligned_log_spectrum_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_spec = _log_spectrum(left)
    right_spec = _log_spectrum(right)
    frames = min(left_spec.shape[1], right_spec.shape[1])
    return _cosine(left_spec[:, :frames].reshape(-1), right_spec[:, :frames].reshape(-1))


def _speaker_embedding(runtime: Any, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != 16000:
        audio = soxr.resample(audio, sample_rate, 16000, quality="HQ")
    audio = audio - float(np.mean(audio, dtype=np.float64))
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio / peak * 0.9
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).to(
        runtime._engine.device, dtype=runtime._engine.precision
    )[None, :]
    embedding = runtime._engine.model_sv_embedding({"audio": waveform})["sv_embedding"]
    result = embedding.detach().float().cpu().numpy().reshape(-1)
    expected = int(
        runtime._streamer.reference_bank.identity.speaker_embedding.numel()
    )
    if result.size != expected:
        padded = np.zeros(expected, dtype=np.float32)
        copied = min(expected, int(result.size))
        padded[:copied] = result[:copied]
        result = padded
    return result


def _options(service_module: Any, args: argparse.Namespace, *, profile: str | None, policy: str):
    return service_module.SynthesisOptions(
        text=args.text,
        text_language=args.language,
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        speed=1.0,
        pause_length=0.0,
        noise_scale=0.5,
        cut_punc="",
        seed=1234,
        expression_enabled=profile is not None,
        expression_profile=profile,
        expression_intensity=args.intensity,
        expression_policy=service_module.ConditioningPolicy(policy),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--intensity", type=float, default=0.7)
    parser.add_argument("--text", default="今日はいい天気ですね。")
    parser.add_argument("--language", default="ja")
    args = parser.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    os.environ.update(
        {
            "ANIFLIVE_TTS_MODEL_PACKAGE": str(args.model_package.resolve()),
            "ANIFLIVE_TTS_SHARED_DIR": str(args.shared_dir.resolve()),
            "ANIFLIVE_TTS_SOURCE_DIR": str((repo / "minimal_inference").resolve()),
            "ANIFLIVE_TTS_WARM_RETENTION_SECONDS": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    from aniflive_tts.api import configure_runtime

    configure_runtime()
    from aniflive_tts import service as service_module

    runtime = service_module.TensorRTService(service_module.RuntimeSettings.from_env())
    runtime.load()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        neutral = runtime.synthesize(_options(service_module, args, profile=None, policy="full-switch"))
        neutral_rate, neutral_audio = _decode_wav(neutral.wav)
        neutral_embedding = _speaker_embedding(runtime, neutral_audio, neutral_rate)
        identity_reference_embedding = (
            runtime._streamer.reference_bank.identity.speaker_embedding
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        (output_dir / "neutral.wav").write_bytes(neutral.wav)
        rows: list[dict[str, Any]] = []
        for profile in args.profiles:
            selected = runtime._expression_catalog.select(
                profile=profile,
                intensity=args.intensity,
                language=args.language,
            )
            expression_reference_embedding = (
                runtime._streamer.reference_bank.get(selected.id).speaker_embedding
                .detach()
                .float()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            for policy in POLICIES:
                options = _options(service_module, args, profile=profile, policy=policy)
                runtime.validate_expression(options)
                started = time.perf_counter()
                result = runtime.synthesize(options)
                wall_seconds = time.perf_counter() - started
                sample_rate, audio = _decode_wav(result.wav)
                embedding = _speaker_embedding(runtime, audio, sample_rate)
                name = f"{profile}-{policy}"
                (output_dir / f"{name}.wav").write_bytes(result.wav)
                rows.append(
                    {
                        "profile": profile,
                        "policy": policy,
                        "wall_seconds": wall_seconds,
                        "audio_seconds": audio.size / sample_rate,
                        "rtf": wall_seconds / (audio.size / sample_rate),
                        "leading_silence_seconds": _leading_silence_seconds(audio, sample_rate),
                        "duration_difference_ratio_vs_neutral": abs(
                            audio.size - neutral_audio.size
                        )
                        / neutral_audio.size,
                        "speaker_cosine_vs_neutral": _cosine(
                            embedding, neutral_embedding
                        ),
                        "speaker_cosine_vs_identity_reference": _cosine(
                            embedding, identity_reference_embedding
                        ),
                        "speaker_cosine_vs_expression_reference": _cosine(
                            embedding, expression_reference_embedding
                        ),
                        "log_spectrum_cosine_vs_neutral": _aligned_log_spectrum_cosine(
                            audio, neutral_audio
                        ),
                        "semantic_tokens": int(result.profile.get("semantic_tokens", 0)),
                        "gpt_steps": int(result.profile.get("gpt_steps", 0)),
                        "sovits_seconds": float(result.profile.get("sovits_seconds", 0.0)),
                    }
                )
        report = {
            "schema": 1,
            "purpose": "Roxy-first conditioning-policy exploration",
            "text": args.text,
            "language": args.language,
            "seed": 1234,
            "neutral": {
                "audio_seconds": neutral_audio.size / neutral_rate,
                "leading_silence_seconds": _leading_silence_seconds(
                    neutral_audio, neutral_rate
                ),
            },
            "rows": rows,
        }
        report_path = output_dir / "policy-ablation.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(report_path)
        return 0
    finally:
        runtime.unload()


if __name__ == "__main__":
    raise SystemExit(main())
