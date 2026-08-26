#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import onnx
import tensorrt as trt

from aniflive_tts.backend.trt_builder import (
    DetailedTensorRTLogger,
    ShapeRange,
    TensorRTEngineBuilder,
)
from aniflive_tts.model_package import runtime_fingerprint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    model = onnx.load(str(path), load_external_data=False)
    result: dict[str, tuple[int, ...]] = {}
    for value in model.graph.input:
        dimensions: list[int] = []
        for index, dimension in enumerate(value.type.tensor_type.shape.dim):
            if dimension.dim_value:
                dimensions.append(int(dimension.dim_value))
            elif (
                (value.name in {"k_cache", "v_cache"} and index == 1)
                or value.name in {"x_len", "y_len", "idx"}
            ):
                dimensions.append(1)
            else:
                raise ValueError(
                    f"Experimental block ONNX requires static inputs: {value.name}"
                )
        result[value.name] = tuple(dimensions)
    return result


def _static_profile(path: Path) -> dict[str, ShapeRange]:
    shapes = _input_shapes(path)
    return {
        name: ShapeRange(shape, shape, shape)
        for name, shape in shapes.items()
        if name in {"k_cache", "v_cache", "x_len", "y_len", "idx"}
    }


def _validate_engine(path: Path, block_size: int) -> dict[str, Any]:
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"TensorRT could not deserialize {path}")
    tensors = {
        engine.get_tensor_name(index): {
            "mode": str(
                engine.get_tensor_mode(engine.get_tensor_name(index))
            ).rsplit(".", 1)[-1],
            "shape": list(engine.get_tensor_shape(engine.get_tensor_name(index))),
            "dtype": str(engine.get_tensor_dtype(engine.get_tensor_name(index))),
        }
        for index in range(engine.num_io_tensors)
    }
    expected = {
        "samples",
        "k_cache",
        "v_cache",
        "x_len",
        "y_len",
        "idx",
        "topk_values",
        "topk_indices",
        "k_cache_new",
        "v_cache_new",
    }
    if set(tensors) != expected:
        raise RuntimeError(f"Unexpected block engine tensors: {sorted(tensors)}")
    if tensors["samples"]["shape"] != [1, block_size]:
        raise RuntimeError("Block engine samples shape does not match its variant")
    return tensors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--workspace-mib", type=int, default=4096)
    parser.add_argument("--optimization-level", type=int, choices=range(0, 6), default=5)
    args = parser.parse_args()
    block_sizes = sorted(set(args.block_sizes))
    if any(size not in (2, 4) for size in block_sizes):
        parser.error("--block-sizes currently accepts only 2 and 4")

    onnx_dir = args.onnx_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex[:12]}"
    staging.mkdir()
    timing_cache = staging / "timing.cache"
    manifest: dict[str, Any] = {
        "schema": 1,
        "kind": "aniflive-tts-v1.2-gpt-block-tensorrt11",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime": runtime_fingerprint(),
        "workspace_mib": args.workspace_mib,
        "builder_optimization_level": args.optimization_level,
        "tf32_allowed": False,
        "blocks": {},
    }
    try:
        for block_size in block_sizes:
            source = onnx_dir / f"gpt_block_h{block_size}.onnx"
            if not source.is_file():
                raise FileNotFoundError(source)
            onnx.checker.check_model(str(source), full_check=True)
            target = staging / f"gpt_block_h{block_size}.engine"
            logger = DetailedTensorRTLogger(echo=False)
            builder = TensorRTEngineBuilder(
                workspace_bytes=args.workspace_mib * 1024 * 1024,
                allow_tf32=False,
                builder_optimization_level=args.optimization_level,
                logger=logger,
            )
            result = builder.build(
                source,
                target,
                profiles=(_static_profile(source),),
                timing_cache_path=timing_cache,
                inspector_path=staging / "inspectors" / f"gpt_block_h{block_size}.json",
                logger_path=staging / "logs" / f"gpt_block_h{block_size}.log",
            )
            manifest["blocks"][str(block_size)] = {
                "onnx": source.name,
                "onnx_sha256": _sha256(source),
                "engine": target.name,
                "engine_sha256": _sha256(target),
                "build": result.to_dict(),
                "io": _validate_engine(target, block_size),
            }
        (staging / "engine-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            backup = output_dir.parent / (
                f"{output_dir.name}.previous-{dt.datetime.now():%Y%m%dT%H%M%S}"
            )
            output_dir.replace(backup)
        staging.replace(output_dir)
    except Exception:
        failed = output_dir.parent / f"failed-build-{dt.datetime.now():%Y%m%dT%H%M%S}"
        if staging.exists():
            staging.replace(failed)
        raise
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
