from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx

from aniflive_tts.backend.trt_builder import (
    DetailedTensorRTLogger,
    ShapeRange,
    TensorRTEngineBuilder,
)


CACHE_NAMES = {"k_cache", "v_cache", "k_cache_new", "v_cache_new"}


def patch_cache_capacity(source: Path, target: Path, capacity: int) -> None:
    model = onnx.load(str(source), load_external_data=True)
    changed: list[str] = []
    for value in (*model.graph.input, *model.graph.output):
        if value.name not in CACHE_NAMES:
            continue
        dimensions = value.type.tensor_type.shape.dim
        if len(dimensions) != 4:
            raise RuntimeError(f"Unexpected cache rank for {value.name}: {len(dimensions)}")
        dimensions[2].dim_value = capacity
        dimensions[2].ClearField("dim_param")
        changed.append(value.name)
    if set(changed) != CACHE_NAMES:
        raise RuntimeError(f"Missing cache tensors while patching ONNX: {changed}")
    onnx.checker.check_model(model)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(target))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a fitted TensorRT 11 GPT-step engine with a shorter KV cache"
    )
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    parser.add_argument("--max-aux-streams", type=int)
    args = parser.parse_args()
    if args.capacity < 192:
        raise SystemExit("--capacity must be at least 192 for the validated reference profile")

    output = args.output_dir.resolve()
    patched = output / f"gpt_step_cache{args.capacity}.onnx"
    engine = output / f"gpt_step_cache{args.capacity}.engine"
    patch_cache_capacity(args.onnx.resolve(), patched, args.capacity)

    cache_shape = (24, 1, args.capacity, 512)
    profile = {
        "samples": ShapeRange((1, 1), (1, 1), (1, 1)),
        "k_cache": ShapeRange(cache_shape, cache_shape, cache_shape),
        "v_cache": ShapeRange(cache_shape, cache_shape, cache_shape),
        "x_len": ShapeRange((1,), (1,), (1,)),
        "y_len": ShapeRange((1,), (1,), (1,)),
        "idx": ShapeRange((1,), (1,), (1,)),
    }
    logger = DetailedTensorRTLogger(echo=True)
    builder = TensorRTEngineBuilder(
        workspace_bytes=int(args.workspace_gib * (1 << 30)),
        allow_tf32=False,
        builder_optimization_level=5,
        max_aux_streams=args.max_aux_streams,
        logger=logger,
    )
    result = builder.build(
        patched,
        engine,
        profiles=(profile,),
        timing_cache_path=output / f"gpt_step_cache{args.capacity}.timing.cache",
        inspector_path=output / f"gpt_step_cache{args.capacity}.inspector.json",
        logger_path=output / f"gpt_step_cache{args.capacity}.build.log",
    )
    report = output / f"gpt_step_cache{args.capacity}.build.json"
    report.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps({"engine": str(engine), "report": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
