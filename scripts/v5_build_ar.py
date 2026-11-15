"""Reconstruct and build the pinned V5 Turbo AR TensorRT graph.

Copyright 2026 Hiruynk. Licensed under the repository LICENSE.
The topology contains only integer shape constants; all learned tensors are
recovered and verified from the pinned checkpoint. No voice reference is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

GRAPH = Path(__file__).resolve().parent / "v5_graph"
CHECKPOINT_SHA256 = "87133414860ea14ff6620c483a3db5ed07b44be42e2c3fcdad65523a729a745a"
GRAPH_SHA256 = "e8f97fd994ad84090a702dce1ba09f9ecda39943a07f1e0ea0ab87b788142340"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def export(checkpoint: Path, output: Path) -> list[dict]:
    import numpy as np
    import onnx
    from onnx import numpy_helper
    import torch
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint does not match the pinned V5 Turbo AR weights")
    manifest = json.loads((GRAPH / "ar-bindings.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("Invalid V5 AR graph binding manifest")
    model = onnx.load(GRAPH / "ar-topology.onnx")
    if model.graph.initializer or model.graph.sparse_initializer:
        raise ValueError("V5 AR topology must not contain learned initializers")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)["weight"]
    rows = manifest["initializers"]
    if len(rows) != 291 or len({row["name"] for row in rows}) != 291:
        raise ValueError("V5 AR initializer inventory is invalid")
    for row in rows:
        if "checkpoint_key" in row:
            value = weights[row["checkpoint_key"]].detach().cpu().numpy()
            if row["layout"] == "transpose":
                value = value.T
            elif row["layout"] != "native":
                raise ValueError("Unsupported V5 AR parameter layout")
            value = np.ascontiguousarray(value)
            digest = hashlib.sha256(value.tobytes()).hexdigest()
            if list(value.shape) != row["shape"] or str(value.dtype) != row["dtype"] or digest != row["sha256"]:
                raise ValueError(f"Checkpoint tensor differs from graph binding: {row['name']}")
        elif row.get("constant") == "zero_bias":
            value = np.zeros((1, 16, 1, 512), dtype=np.float16)
        elif row.get("constant") == "seq_q":
            value = np.array([0, 1], dtype=np.int32)
        elif row.get("constant") == "seq_k":
            value = np.array([0, 512], dtype=np.int32)
        else:
            raise ValueError("Unsupported V5 AR graph constant")
        model.graph.initializer.append(numpy_helper.from_array(value, name=row["name"]))
    onnx.checker.check_model(model)
    onnx.save(model, output)
    if sha256(output) != GRAPH_SHA256:
        raise ValueError("Reconstructed V5 AR graph does not match the pinned topology and weights")
    projections = []
    for layer in range(24):
        tensor = weights[f"model.h.layers.{layer}.linear2.weight"].contiguous().numpy()
        projections.append({"index": 4 * layer + 3, "input_features": 2048,
                            "output_features": 512,
                            "native_sha256": hashlib.sha256(tensor.tobytes()).hexdigest()})
    return projections


def build_plan(onnx_path: Path, native_dir: Path, output: Path, projections: list[dict]) -> None:
    import torch
    import tensorrt as trt
    from aniflive_tts.backend.v5 import (
    attention_native_k8_shared_aot,
    trt_gemv_nativeorder_warp as trt_gemv_nativeorder_warp,
    trt_gemv_nativeorder_residual as trt_gemv_nativeorder_residual,
    trt_cpp_native_layernorm_direct,
)
    if sys.platform != "linux" or torch.__version__ != "2.10.0+cu128" or trt.__version__ != "11.2.1.2":
        raise RuntimeError("V5 AR plan requires Linux, PyTorch 2.10.0+cu128 and TensorRT 11.2.1.2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required to build the TensorRT plan")
    native = json.loads((native_dir / "native-build.json").read_text(encoding="utf-8"))
    if native.get("checkpoint_sha256") != CHECKPOINT_SHA256 or native.get("status") != "built_cpu_only":
        raise ValueError("Native helper build does not match the checkpoint")
    paths = {}
    for name in ("attention_guard", "gemv_guard", "weight_l2", "joint_graph", "layernorm_plugin"):
        info = native["files"][name]
        if Path(info["file"]).name != info["file"]:
            raise ValueError("Invalid native helper path")
        path = native_dir / info["file"]
        if sha256(path) != info["sha256"]:
            raise ValueError(f"Native helper checksum mismatch: {name}")
        paths[name] = path
    trt_cpp_native_layernorm_direct.load(str(paths["layernorm_plugin"]))
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(6 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.PREFER_AOT_PYTHON_PLUGINS))
    network.name = "v5-ar-neural-bucket512"
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("TensorRT ONNX parser: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.max_aux_streams = 0
    config.builder_optimization_level = 3
    config.clear_flag(trt.BuilderFlag.TF32)
    compiled_attention = attention_native_k8_shared_aot.compile_native_k8()
    attention_ptx = compiled_attention.asm["ptx"].encode()
    if (attention_ptx.count(b"mma.sync.aligned.m16n8k8") != 16
            or attention_ptx.count(b"mma.sync.aligned.m16n8k16")
            or attention_ptx.count(b"bar.sync") != 4):
        raise RuntimeError("Native K8 attention PTX contract failed")
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT AR plan build failed")
    plan = output / "ar_neural.plan"
    plan_bytes = memoryview(blob).tobytes()
    if plan_bytes.count(attention_ptx) != 24:
        raise RuntimeError("TensorRT plan does not contain all 24 expected native attention kernels")
    plan.write_bytes(plan_bytes)
    report = {"build": {"sha256": sha256(plan), "onnx_sha256": sha256(onnx_path)},
        "requires_native_k8_pointer_alignment_bytes": 16, "native_k8_alignment_guard_version": 2,
        "requires_gemv_pointer_alignment_bytes": 16, "requires_fused_native_prefix_mask": True,
        "native_k8_alignment_guard_sha256": sha256(paths["attention_guard"]),
        "gemv_alignment_guard_sha256": sha256(paths["gemv_guard"]),
        "projection_aot_weights": projections}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--native-build", type=Path, help="Native helper build directory")
    parser.add_argument("--export-only", action="store_true", help="Reconstruct ONNX on CPU without building a plan")
    args = parser.parse_args()
    if not args.export_only and args.native_build is None:
        parser.error("--native-build is required when building a plan")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    graph = output / "ar_neural.onnx"
    projections = export(args.checkpoint.resolve(strict=True), graph)
    if not args.export_only:
        build_plan(graph, args.native_build.resolve(strict=True), output, projections)
    print(json.dumps({"status": "exported_cpu_only" if args.export_only else "built_unqualified",
                      "onnx_sha256": sha256(graph)}, indent=2))


if __name__ == "__main__":
    main()
