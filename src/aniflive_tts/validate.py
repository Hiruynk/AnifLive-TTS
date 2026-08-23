from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Any

from .backend.contracts import STAGE_IO_CONTRACTS, STAGE_ORDER
from .model_package import select_engine_dir, sha256_file, validate_checksums


def validate_model_package(
    package_dir: Path,
    *,
    enqueue: bool = False,
    shared_dir: Path | None = None,
    source_dir: Path | None = None,
    text: str = "今日はいい天気ですね。",
    language: str = "ja",
) -> dict[str, Any]:
    import tensorrt as trt

    package_dir = package_dir.resolve()
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model_family") != "gsv-v2proplus":
        raise RuntimeError("AnifLive-TTS v1 currently supports gsv-v2proplus packages")
    validate_checksums(package_dir)
    engine_dir = select_engine_dir(package_dir, manifest)
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engines: dict[str, Any] = {}
    for stage in STAGE_ORDER:
        path = engine_dir / f"{stage}.engine"
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError(f"TensorRT failed to deserialize {path}")
        inputs, outputs = [], []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            target = inputs if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs
            target.append(name)
        expected_inputs, expected_outputs = STAGE_IO_CONTRACTS[stage]
        if not set(expected_inputs).issubset(inputs) or set(outputs) != set(expected_outputs):
            raise RuntimeError(f"{stage} I/O mismatch: inputs={inputs}, outputs={outputs}")
        engines[stage] = {"sha256": sha256_file(path), "layers": engine.num_layers}
    report = {
        "status": "passed",
        "model_id": manifest["model_id"],
        "engine_count": len(engines),
        "enqueue_requested": enqueue,
        "enqueue_verified": False,
        "engines": engines,
    }
    if enqueue:
        if shared_dir is None or source_dir is None:
            raise ValueError("--enqueue requires shared_dir and source_dir")
        os.environ.update(
            {
                "ANIFLIVE_TTS_MODEL_PACKAGE": str(package_dir),
                "ANIFLIVE_TTS_SHARED_DIR": str(shared_dir.expanduser().resolve()),
                "ANIFLIVE_TTS_SOURCE_DIR": str(source_dir.expanduser().resolve()),
            }
        )
        from .api import configure_runtime

        configure_runtime()
        from .service import SERVICE, SynthesisOptions

        SERVICE.load()
        try:
            result = SERVICE.synthesize(
                SynthesisOptions(
                    text=text,
                    text_language=language,
                    top_k=15,
                    top_p=1.0,
                    temperature=1.0,
                    speed=1.0,
                    pause_length=0.3,
                    noise_scale=0.5,
                    cut_punc="。！？.!?、，；：",
                    seed=1234,
                )
            )
        finally:
            SERVICE.unload()
        if result.output_samples <= 0 or len(result.wav) <= 44:
            raise RuntimeError("TensorRT enqueue smoke test produced invalid audio")
        report["enqueue_verified"] = True
        report["smoke_inference"] = {
            "backend": "TensorRT-11",
            "pytorch_fallback": False,
            "language": language,
            "output_samples": result.output_samples,
            "sample_rate": result.sample_rate,
            "elapsed_seconds": result.elapsed_seconds,
            "wav_sha256": hashlib.sha256(result.wav).hexdigest(),
            "semantic_tokens": int(result.profile.get("semantic_tokens", 0)),
        }
    return report
