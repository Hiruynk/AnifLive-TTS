"""Prepare official reference-only DSP arrays and optional non-Mandarin phones.

This produces inputs for the native frontend/inference path. It does not run
HuBERT, BERT, GPT, SoVITS, a vocoder, or any other neural model in Python.
Mandarin callers request ``--dsp-only`` and add live-bridge phones separately.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from official_source_extract import extract


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_LANGUAGES = {"ja", "en", "yue", "ko", "zh"}
SOURCE_SAMPLE_RATE = 32000
TARGET_SAMPLE_RATE = 16000
PAD_SAMPLES = int(32000 * 0.3)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_reference_spectrogram_functions(gpt_sovits: Path):
    module_dir = gpt_sovits / "module"
    mel_path = module_dir / "mel_processing.py"
    spec = importlib.util.spec_from_file_location("official_reference_mel_processing", mel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the official mel_processing source")
    mel_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mel_module)
    return mel_module.spectrogram_torch, mel_module.mel_spectrogram_torch, mel_path


def _source_resampler(gpt_sovits: Path):
    namespace: dict[str, Any] = {
        "torch": torch,
        "np": np,
        "resampy_filter_cache": {},
        "resample_transform_dict": {},
    }
    source = gpt_sovits / "TTS_infer_pack" / "TTS.py"
    extract(source, ["_resampy_gpu_filter", "resample"], namespace)
    return namespace["resample"], source


def _compare_existing_reference(**_kwargs) -> dict[str, Any]:
    return {"status": "not_applicable", "reason": "No external comparison fixture supplied", "exact_arrays": {}}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare official reference DSP arrays and optional phones; no neural model."
    )
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--reference-language", choices=sorted(REFERENCE_LANGUAGES), required=True)
    parser.add_argument("--dsp-device", choices=("cpu", "mps"), default="cpu",
                        help="reference resampling device; MPS is DSP only, never neural inference")
    parser.add_argument("--dsp-only", action="store_true",
                        help="prepare only DSP/text bytes; the caller must supply phone/BERT preprocessing")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.reference_language == "zh" and not args.dsp_only:
        raise ValueError("Mandarin reference phones/BERT require the attached live native bridge; use --dsp-only")
    output_dir = args.output.expanduser().resolve()
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"--output must be a fresh empty directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    official = args.official.expanduser().resolve(strict=True)
    gpt_sovits = official / "GPT_SoVITS"
    if not gpt_sovits.is_dir():
        raise ValueError(f"official source root is missing GPT_SoVITS: {official}")
    audio_path = args.audio.expanduser().resolve(strict=True)
    transcript_path = args.transcript.expanduser().resolve(strict=True)
    transcript_bytes = transcript_path.read_bytes()
    transcript_text = transcript_bytes.decode("utf-8")

    torch.set_num_threads(4)
    samples, source_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if source_rate != SOURCE_SAMPLE_RATE:
        raise ValueError(
            f"Only {SOURCE_SAMPLE_RATE} Hz reference WAV is supported; source-rate resampling "
            "through torchaudio is unavailable in this Torch 2.15.0 environment"
        )
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] not in (1, 2):
        raise ValueError(f"Reference audio must be mono or stereo; got shape {samples.shape}")
    if not np.isfinite(samples).all():
        raise ValueError("Reference audio contains non-finite samples")

    audio32 = torch.from_numpy(samples.T.copy()).mean(dim=0, keepdim=True)
    peak = audio32.abs().max()
    if peak > 1:
        audio32 = audio32 / min(2.0, float(peak))

    resample, resample_source = _source_resampler(gpt_sovits)
    start = time.perf_counter()
    dsp_device = torch.device(args.dsp_device)
    if args.dsp_device == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0" or not torch.backends.mps.is_available():
            raise RuntimeError("MPS reference DSP requires an available MPS device and fallback disabled")
        if torch.__version__.split("+")[0] != "2.9.1":
            raise RuntimeError("Qualified FP16 reference resampling requires source-matched Torch 2.9.1")
    audio16 = resample(
        audio32.to(dsp_device),
        source_rate,
        TARGET_SAMPLE_RATE,
        dsp_device,
        match_librosa=True,
    ).cpu()
    if not torch.isfinite(audio16).all():
        raise ValueError("Official resampler produced non-finite samples")
    if not 48000 <= audio16.shape[-1] <= 160000:
        raise ValueError("Reference duration must be 3 to 10 seconds after 16 kHz resampling")
    waveform = torch.cat(
        (audio16.to(torch.float32), torch.zeros((1, PAD_SAMPLES), dtype=torch.float32)), dim=1
    )
    if not torch.isfinite(waveform).all():
        raise ValueError("Padded 16 kHz waveform contains non-finite samples")
    waveform_seconds = time.perf_counter() - start

    spectrogram_torch, mel_spectrogram_torch, mel_source = _load_reference_spectrogram_functions(
        gpt_sovits
    )
    start = time.perf_counter()
    with torch.inference_mode():
        refer = spectrogram_torch(audio32.to(dsp_device), 2048, 32000, 640, 2048, center=False)
    if refer.dtype != torch.float32 or not torch.isfinite(refer).all():
        raise ValueError("Official 32 kHz spectrogram must be finite float32")
    spectrogram_seconds = time.perf_counter() - start

    start = time.perf_counter()
    with torch.inference_mode():
        reference_mel = mel_spectrogram_torch(
            audio32.to(dsp_device), 1280, 100, 32000, 320, 1280, 0, None, center=False
        )
    if reference_mel.dtype != torch.float32 or not torch.isfinite(reference_mel).all():
        raise ValueError("Official raw vocoder log-mel must be finite float32")
    reference_mel_seconds = time.perf_counter() - start

    data_root = Path(os.environ.get("ANIFLIVE_TTS_MACOS_DATA_ROOT", ROOT / "data/macos-v5turbo")).expanduser().resolve()
    frontend = None
    frontend_source = None
    phone_ids = None
    segments = None
    normalized_text = None
    frontend_seconds = None
    if not args.dsp_only:
        if args.reference_language in {"en", "ko"}:
            import prepare_frontend

            prepare_frontend._ensure_nltk_data(data_root)
        frontend = importlib.import_module("prepare_frontend")
        start = time.perf_counter()
        frontend_source = frontend._load_official_preprocessor(official, data_root / "cache")
        phone_ids, phone_bert, segments, normalized_text = frontend._preprocess(
            frontend_source["processor"],
            transcript_text,
            frontend.TARGET_LANGUAGES[args.reference_language],
            frontend_source["symbols"],
        )
        frontend_seconds = time.perf_counter() - start
        if not phone_ids:
            raise ValueError("Reference transcript produced no phone IDs")
        if phone_bert.shape != (1024, len(phone_ids)) or torch.count_nonzero(phone_bert).item() != 0:
            raise ValueError("Expected finite zero-BERT reference features aligned to non-Mandarin phones")
        if not torch.isfinite(phone_bert).all():
            raise ValueError("Reference BERT placeholder contains non-finite values")

    arrays = {
        "waveform.npy": waveform.cpu().numpy().astype(np.float32, copy=False),
        "refer.npy": refer.cpu().numpy().astype(np.float32, copy=False),
        "reference-mel.npy": reference_mel.cpu().numpy().astype(np.float32, copy=False),
    }
    if phone_ids is not None:
        arrays["reference-phones.npy"] = np.asarray([phone_ids], dtype=np.int64)
    if arrays["waveform.npy"].shape[0] != 1:
        raise ValueError("Prepared 16 kHz waveform must be [1,T]")
    if arrays["refer.npy"].ndim != 3 or arrays["refer.npy"].shape[1] != 1025:
        raise ValueError(f"Unexpected official spectrogram shape {arrays['refer.npy'].shape}")
    if arrays["reference-mel.npy"].shape[1] != 100:
        raise ValueError(f"Unexpected raw log-mel shape {arrays['reference-mel.npy'].shape}")
    if any(array.dtype.kind == "f" and not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("A prepared reference array contains non-finite values")

    (output_dir / "reference.txt").write_bytes(transcript_bytes)
    output_records: dict[str, Any] = {}
    for name, array in arrays.items():
        path = output_dir / name
        np.save(path, array, allow_pickle=False)
        output_records[name] = {
            "sha256": _sha256_file(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    output_records["reference.txt"] = {
        "sha256": _sha256_file(output_dir / "reference.txt"),
        "size_bytes": len(transcript_bytes),
    }

    comparison = _compare_existing_reference(
        audio_sha256=_sha256_file(audio_path),
        transcript_sha256=_sha256_file(transcript_path),
        language=args.reference_language,
        arrays=arrays,
    )
    source_paths = [
        resample_source,
        mel_source,
    ]
    if frontend is not None:
        source_paths.extend(
            frontend._source_files(
                gpt_sovits,
                {args.reference_language},
                frontend_source["preprocessor_path"],
                frontend_source["segment_source"],
            )
        )
        source_paths.append(frontend_source["dictionary_source"])
    unique_source_paths = sorted(set(source_paths))
    frontend_metadata = (
        {"reference_text_frontend": "pending_attached_native_mandarin_bridge"}
        if args.dsp_only
        else {
            "reference_text_frontend": "official_non_mandarin_zero_bert",
            "reference_normalized_text": normalized_text,
            "reference_segments": segments,
            "reference_phone_count": len(phone_ids),
            "reference_phone_ids": phone_ids,
            "reference_phone_symbols": [frontend_source["symbols"][phone] for phone in phone_ids],
            "zero_bert_placeholder": True,
        }
    )
    report = {
        "schema": "aniflive-tts-macos-v5turbo-reference-inputs-v1",
        "kind": "official_reference_dsp_only" if args.dsp_only else "official_reference_dsp_and_phones_cpu_only",
        "device": "cpu",
        "neural_models_loaded": False,
        "reference_audio_path": str(audio_path),
        "reference_audio_sha256": _sha256_file(audio_path),
        "reference_transcript_path": str(transcript_path),
        "reference_transcript_sha256": _sha256_file(transcript_path),
        "reference_text_bytes_preserved": transcript_bytes == (output_dir / "reference.txt").read_bytes(),
        "reference_language": args.reference_language,
        "official_language": "all_zh" if args.dsp_only else frontend.TARGET_LANGUAGES[args.reference_language],
        **frontend_metadata,
        "source_audio": {
            "sample_rate_hz": int(source_rate),
            "channels": int(samples.shape[1]),
            "sample_count": int(samples.shape[0]),
            "duration_seconds": float(samples.shape[0] / source_rate),
            "peak_before_source_scaling": float(peak),
            "source_peak_scaling": "if peak>1, divide by min(2, peak)",
        },
        "resampling": {
            "source_rate_hz": int(source_rate),
            "target_rate_hz": TARGET_SAMPLE_RATE,
            "method": "official TTS.resample with match_librosa=True (resampy kaiser_best)",
            "device": args.dsp_device,
            "torch_version": torch.__version__,
            "neural_inference": False,
            "reference_samples_16khz": int(audio16.shape[-1]),
            "appended_zero_samples": PAD_SAMPLES,
            "padding_rule": "int(32000 * 0.3), appended to the 16 kHz waveform",
            "waveform_shape": list(waveform.shape),
        },
        "unsupported_source_rates": "only 32000 Hz is accepted until the source torchaudio resample path is available and qualified",
        "spectrogram": {
            "n_fft": 2048,
            "hop_length": 640,
            "win_length": 2048,
            "sample_rate_hz": 32000,
            "center": False,
            "shape": list(refer.shape),
        },
        "reference_log_mel": {
            "n_fft": 1280,
            "hop_length": 320,
            "win_length": 1280,
            "mel_bins": 100,
            "sample_rate_hz": 32000,
            "center": False,
            "representation": "raw official log-mel for vocoder reference conditioning",
            "shape": list(reference_mel.shape),
        },
        "cpu_frontend_timing_seconds": {
            "resample_and_padding": waveform_seconds,
            "spectrogram": spectrogram_seconds,
            "reference_log_mel": reference_mel_seconds,
            "reference_text_frontend": frontend_seconds,
            "total": waveform_seconds + spectrogram_seconds + reference_mel_seconds
            + (frontend_seconds or 0.0),
        },
        "output_artifacts": output_records,
        "comparison": comparison,
        "python_dependencies": {
            name: _package_version(name)
            for name in ("torch", "numpy", "soundfile", "resampy", "librosa", "pyopenjtalk", "split-lang", "fast-langdetect")
        },
        "official_source_sha256": {
            str(path.relative_to(official)): _sha256_file(path)
            for path in unique_source_paths
            if path.is_file()
        },
        "scope": "reference DSP and text inputs only; no neural inference or Rust worker execution",
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "fixture.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "language": args.reference_language,
                "phones": len(phone_ids) if phone_ids is not None else None,
                "waveform_shape": list(waveform.shape),
                "refer_shape": list(refer.shape),
                "reference_mel_shape": list(reference_mel.shape),
                "comparison_status": comparison["status"],
                "comparison_passed": comparison.get("passed"),
            },
            sort_keys=True,
        )
    )
    return 0 if comparison["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
