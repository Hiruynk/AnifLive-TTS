"""Reconstruct and build the pinned V5 Turbo acoustic TensorRT graphs.

Copyright 2026 Hiruynk. Licensed under the repository LICENSE.
The stored topologies contain no learned weights or voice reference data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

GRAPH = Path(__file__).resolve().parent / "v5_graph"
S2_SHA256 = "287b32a06b7d486f5303ec520468b0728fb01aac8b0f5d86fd5455c9a1d3e022"
VOCODER_SHA256 = "b7543c83222aa2105bb4efb56845086bc5f2eeba61243d723ffa062052b3221f"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def time_banks(weights: dict, source: Path) -> dict:
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required to generate native FP16 DiT timestep banks")
    source_lock = {
        "GPT_SoVITS/f5_tts/model/modules.py": "d54f49402172959ee8a73f804735561d9bcbaf4d18c18f2f4138acf854dbccf1",
        "GPT_SoVITS/Accel/PyTorch/__init__.py": "2870d5422c49f61f2fbf112791774870bd26ccd3873d64052bb893efd3ef4f2d",
    }
    for relative, expected in source_lock.items():
        if sha256(source / relative) != expected:
            raise ValueError(f"Upstream source does not match the pinned version: {relative}")
    sys.path[:0] = [str(source), str(source / "GPT_SoVITS")]
    import GPT_SoVITS.Accel.PyTorch  # noqa: F401
    from f5_tts.model.modules import TimestepEmbedding
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(6 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    prefix = "cfm.estimator."
    embed = TimestepEmbedding(1024).half().cuda().eval()
    embed.load_state_dict({key.removeprefix(prefix + "time_embed."): value
                           for key, value in weights.items() if key.startswith(prefix + "time_embed.")})
    layers = []
    for i in range(22):
        base = prefix + f"transformer_blocks.{i}.attn_norm.linear."
        layers.append((weights[base + "weight"].cuda(), weights[base + "bias"].cuda()))
    final = prefix + "norm_out.linear."
    final_weight, final_bias = weights[final + "weight"].cuda(), weights[final + "bias"].cuda()
    blocks = [[] for _ in layers]
    finals = []
    with torch.inference_mode():
        for value in (0.0, 0.25, 0.5, 0.75):
            t = torch.full((1,), value, device="cuda", dtype=torch.float16)
            embedded = embed(t)
            for rows, (weight, bias) in zip(blocks, layers):
                rows.append(F.linear(F.silu(embedded), weight, bias))
            finals.append(F.linear(F.silu(embedded), final_weight, final_bias))
        result = {f"block_bank_{i}": torch.cat(rows, dim=0).contiguous().cpu().numpy()
                  for i, rows in enumerate(blocks)}
        result["final_bank"] = torch.cat(finals, dim=0).contiguous().cpu().numpy()
    return result


def export(component: str, checkpoint: Path, output: Path, source: Path | None) -> None:
    import numpy as np
    import onnx
    from onnx import numpy_helper
    import torch
    expected = VOCODER_SHA256 if component == "vocoder" else S2_SHA256
    if sha256(checkpoint) != expected:
        raise ValueError(f"{component} checkpoint does not match the pinned weights")
    manifest = json.loads((GRAPH / f"{component}-bindings.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest.get("checkpoint_sha256") != expected:
        raise ValueError("Invalid acoustic graph binding manifest")
    model = onnx.load(GRAPH / f"{component}-topology.onnx")
    if model.graph.initializer or model.graph.sparse_initializer:
        raise ValueError("Acoustic topology must not contain learned initializers")
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = raw if component == "vocoder" else raw["weight"]
    if component == "dit":
        if source is None:
            raise ValueError("The pinned upstream source is required for DiT timestep banks")
        banks = time_banks(weights, source)
    else:
        banks = {}
    rows = manifest["initializers"]
    if len({row["name"] for row in rows}) != len(rows):
        raise ValueError("Duplicate acoustic graph initializer")
    for row in rows:
        if "computed_bank" in row:
            value = banks[row["computed_bank"]]
        else:
            value = weights[row["checkpoint_key"]].detach().cpu().numpy()
            if row["layout"] == "transpose":
                value = value.T
            elif row["layout"] == "half":
                value = value.astype(np.float16)
            elif row["layout"] != "native":
                raise ValueError("Unsupported acoustic parameter layout")
        value = np.ascontiguousarray(value)
        digest = hashlib.sha256(value.tobytes()).hexdigest()
        if list(value.shape) != row["shape"] or str(value.dtype) != row["dtype"] or digest != row["sha256"]:
            raise ValueError(f"Acoustic tensor differs from pinned graph: {row['name']}")
        model.graph.initializer.append(numpy_helper.from_array(value, name=row["name"]))
    onnx.checker.check_model(model)
    onnx.save(model, output)
    if sha256(output) != manifest["graph_sha256"]:
        raise ValueError("Reconstructed acoustic graph does not match the pinned graph")


def plan(component: str, onnx_path: Path, output: Path, *, static_579: bool = False) -> None:
    import torch
    import tensorrt as trt
    if sys.platform != "linux" or torch.__version__ != "2.10.0+cu128" or trt.__version__ != "11.2.1.2":
        raise RuntimeError("V5 plans require Linux, PyTorch 2.10.0+cu128 and TensorRT 11.2.1.2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required to build the TensorRT plan")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("TensorRT ONNX parser: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    inputs = {network.get_input(i).name: network.get_input(i) for i in range(network.num_inputs)}
    if component == "dit":
        from v5_dit_attention import native_attention_sites, install_native_attention
        install_native_attention(network, trt, native_attention_sites(onnx_path), unmasked=True)
        ranges = {"x": ((1,100,128),(1,100,640),(1,100,1000)),
                  "condition": ((1,128,1024),(1,640,1024),(1,1000,1024)),
                  "mask": ((1,128),(1,640),(1,1000)), "rope": ((1,128,64),(1,640,64),(1,1000,64))}
        expected_inputs = {"x", "time", "condition", "mask", "rope"}
    elif component == "encoder":
        ranges = {"codes": ((1,1,1),(1,1,80),(1,1,1536)), "phones": ((1,1),(1,96),(1,2048))}
        expected_inputs = {"codes", "phones", "ge"}
    else:
        ranges = {"mel": ((1,100,16),(1,100,256),(1,100,1024))}
        expected_inputs = {"mel"}
    if set(inputs) != expected_inputs or network.num_outputs != 1:
        raise RuntimeError(f"Unexpected {component} graph inputs/outputs")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if component in ("dit", "encoder"):
        config.builder_optimization_level = 3
        config.max_aux_streams = 0
    if component == "dit":
        config.clear_flag(trt.BuilderFlag.TF32)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        if static_579:
            config.tiling_optimization_level = trt.TilingOptimizationLevel.FAST
    if static_579:
        shapes = {"x": (1,100,579), "condition": (1,579,1024), "mask": (1,579), "rope": (1,579,64)}
        for name, shape in shapes.items():
            inputs[name].shape = shape
        if any(-1 in tuple(t.shape) for t in inputs.values()):
            raise RuntimeError("Static DiT plan retains symbolic input dimensions")
    else:
        profile = builder.create_optimization_profile()
        for name, shapes in ranges.items():
            profile.set_shape(name, *shapes)
            if tuple(map(tuple, profile.get_shape(name))) != shapes:
                raise RuntimeError(f"TensorRT profile mismatch for {name}")
        if not profile or config.add_optimization_profile(profile) < 0:
            raise RuntimeError("TensorRT rejected the optimization profile")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT {component} plan build failed")
    output.write_bytes(memoryview(serialized).tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", required=True, choices=("encoder", "dit", "vocoder"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source", type=Path, help="Pinned upstream source, required for DiT")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.component == "dit" and args.source is None:
        parser.error("--source is required for DiT timestep banks")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    graph = output / f"{args.component}.onnx"
    export(args.component, args.checkpoint.resolve(strict=True), graph,
           args.source.resolve(strict=True) if args.source else None)
    built = {}
    if not args.export_only:
        target = output / f"{args.component}.plan"
        if args.component == "encoder":
            import subprocess
            subprocess.run([sys.executable, str(Path(__file__).with_name("v5_build_encoder_plan.py")),
                            "--onnx", str(graph), "--output", str(target)], check=True)
        else:
            plan(args.component, graph, target)
        built[target.name] = sha256(target)
        if args.component == "dit":
            target = output / "dit_579.plan"
            plan("dit", graph, target, static_579=True)
            built[target.name] = sha256(target)
    print(json.dumps({"status": "exported" if args.export_only else "built_unqualified",
                      "onnx_sha256": sha256(graph), "plans": built}, indent=2))


if __name__ == "__main__":
    main()
