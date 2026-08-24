from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import tensorrt as trt
import torch
from onnx import helper, numpy_helper

from .compatibility import ensure_tensorrt11
from .contracts import ModelPaths, STAGE_IO_CONTRACTS, STAGE_ORDER
from .profiles import profiles_for
from .trt_builder import DetailedTensorRTLogger, TensorRTEngineBuilder


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_onnx_bundle(onnx_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for stage in STAGE_ORDER:
        path = onnx_dir / f"{stage}.onnx"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required ONNX model is missing: {path}")
        onnx.checker.check_model(str(path))
        hashes[path.name] = sha256_file(path)
    config = onnx_dir / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Required ONNX config is missing: {config}")
    hashes[config.name] = sha256_file(config)
    return hashes


def patch_sovits_myelin_broadcast(source: Path, target: Path) -> dict[str, Any]:
    """Work around NVIDIA/TensorRT#4743 without changing numeric values.

    The V2 Pro Plus SoVITS export contains eight rank-1 MatMul biases. Myelin
    incorrectly derives a three-element stride order for a four-dimensional
    fused tensor on Blackwell. Replacing the eight equivalent Linear
    MatMul+Add pairs with 1x1 Conv operations removes both implicit batch and
    bias broadcasting while preserving the same affine transformation.
    """

    model = onnx.load(str(source), load_external_data=True)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    patched: list[dict[str, Any]] = []
    broadcast_adds: dict[str, tuple[str, str]] = {}
    for node in model.graph.node:
        if node.op_type != "Add":
            continue
        for input_index, input_name in enumerate(node.input):
            initializer = initializers.get(input_name)
            if initializer is None or len(initializer.dims) != 1:
                continue
            values = numpy_helper.to_array(initializer)
            tensor_name = node.input[1 - input_index]
            broadcast_adds[node.name] = (input_name, tensor_name)
            patched.append(
                {
                    "node": node.name,
                    "initializer": input_name,
                    "original_shape": [int(values.shape[0])],
                    "operation": "MatMul + Add converted to equivalent 1x1 Conv",
                }
            )
    if len(patched) != 8:
        raise RuntimeError(
            f"Expected 8 known SoVITS Myelin broadcast biases, found {len(patched)}"
        )
    producers = {
        output_name: node
        for node in model.graph.node
        for output_name in node.output
    }
    patched_weights: list[dict[str, Any]] = []
    linear_replacements: dict[str, tuple[str, str, str, str]] = {}
    replaced_matmuls: set[str] = set()
    for node_name, (bias_name, tensor_name) in broadcast_adds.items():
        producer = producers.get(tensor_name)
        if producer is None or producer.op_type != "MatMul":
            raise RuntimeError(f"Expected MatMul before broadcast Add {node_name}")
        weight_name = next(
            (
                input_name
                for input_name in producer.input
                if input_name in initializers and len(initializers[input_name].dims) == 2
            ),
            None,
        )
        if weight_name is None:
            raise RuntimeError(f"Expected rank-2 MatMul weight before {node_name}")
        activation_name = next(
            input_name for input_name in producer.input if input_name != weight_name
        )
        values = numpy_helper.to_array(initializers[weight_name])
        conv_weight_name = f"{weight_name}.trt11_conv"
        conv_values = values.T.reshape(values.shape[1], values.shape[0], 1)
        model.graph.initializer.append(
            numpy_helper.from_array(conv_values, name=conv_weight_name)
        )
        linear_replacements[node_name] = (
            activation_name,
            conv_weight_name,
            bias_name,
            tensor_name,
        )
        replaced_matmuls.add(producer.name)
        patched_weights.append(
            {
                "matmul": producer.name,
                "initializer": weight_name,
                "original_shape": list(values.shape),
                "conv_weight": conv_weight_name,
                "conv_shape": [int(value) for value in conv_values.shape],
            }
        )
    decomposed: list[dict[str, Any]] = []
    replacement_nodes: list[onnx.NodeProto] = []
    for node_index, node in enumerate(model.graph.node):
        if node.name in replaced_matmuls:
            continue
        if node.name in linear_replacements:
            activation_name, conv_weight_name, bias_name, _ = linear_replacements[node.name]
            prefix = f"trt11_linear_{node_index}"
            channels_first = f"{prefix}_channels_first"
            convolved = f"{prefix}_convolved"
            replacement_nodes.extend(
                [
                    helper.make_node(
                        "Transpose",
                        [activation_name],
                        [channels_first],
                        name=f"{prefix}/ToChannelsFirst",
                        perm=[0, 2, 1],
                    ),
                    helper.make_node(
                        "Conv",
                        [channels_first, conv_weight_name, bias_name],
                        [convolved],
                        name=f"{prefix}/Conv",
                        kernel_shape=[1],
                        strides=[1],
                        pads=[0, 0],
                        dilations=[1],
                        group=1,
                    ),
                    helper.make_node(
                        "Transpose",
                        [convolved],
                        list(node.output),
                        name=node.name,
                        perm=[0, 2, 1],
                    ),
                ]
            )
            continue
        if node.op_type != "LayerNormalization":
            replacement_nodes.append(node)
            continue
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        axis = int(attributes.get("axis", -1))
        epsilon = float(attributes.get("epsilon", 1e-5))
        if axis != -1 or len(node.input) != 3 or len(node.output) != 1:
            raise RuntimeError(f"Unsupported SoVITS LayerNormalization: {node.name}")
        input_name, scale_name, bias_name = node.input
        scale = initializers.get(scale_name)
        bias = initializers.get(bias_name)
        if scale is None or bias is None:
            raise RuntimeError(f"LayerNormalization constants are missing: {node.name}")
        scale_values = numpy_helper.to_array(scale)
        bias_values = numpy_helper.to_array(bias)
        if scale_values.shape != bias_values.shape or scale_values.ndim != 1:
            raise RuntimeError(f"Unexpected LayerNormalization weights: {node.name}")

        prefix = f"trt11_ln_{node_index}"
        epsilon_name = f"{prefix}_epsilon"
        axes_name = f"{prefix}_axes"
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.asarray(epsilon, dtype=scale_values.dtype), name=epsilon_name
            )
        )
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray([-1], dtype=np.int64), name=axes_name)
        )
        names = {
            key: f"{prefix}_{key}"
            for key in (
                "mean",
                "centered",
                "squared",
                "variance",
                "variance_shape",
                "epsilon_expanded",
                "variance_epsilon",
                "stddev",
                "normalized",
                "shape",
                "scale_expanded",
                "bias_expanded",
                "scaled",
            )
        }
        replacement_nodes.extend(
            [
                helper.make_node(
                    "ReduceMean", [input_name, axes_name], [names["mean"]],
                    name=f"{prefix}/ReduceMean", keepdims=1,
                ),
                helper.make_node(
                    "Sub", [input_name, names["mean"]], [names["centered"]],
                    name=f"{prefix}/Center",
                ),
                helper.make_node(
                    "Mul", [names["centered"], names["centered"]], [names["squared"]],
                    name=f"{prefix}/Square",
                ),
                helper.make_node(
                    "ReduceMean", [names["squared"], axes_name], [names["variance"]],
                    name=f"{prefix}/Variance", keepdims=1,
                ),
                helper.make_node(
                    "Shape", [names["variance"]], [names["variance_shape"]],
                    name=f"{prefix}/VarianceShape",
                ),
                helper.make_node(
                    "Expand", [epsilon_name, names["variance_shape"]], [names["epsilon_expanded"]],
                    name=f"{prefix}/ExpandEpsilon",
                ),
                helper.make_node(
                    "Add", [names["variance"], names["epsilon_expanded"]], [names["variance_epsilon"]],
                    name=f"{prefix}/AddEpsilon",
                ),
                helper.make_node(
                    "Sqrt", [names["variance_epsilon"]], [names["stddev"]],
                    name=f"{prefix}/Sqrt",
                ),
                helper.make_node(
                    "Div", [names["centered"], names["stddev"]], [names["normalized"]],
                    name=f"{prefix}/Normalize",
                ),
                helper.make_node(
                    "Shape", [input_name], [names["shape"]], name=f"{prefix}/Shape"
                ),
                helper.make_node(
                    "Expand", [scale_name, names["shape"]], [names["scale_expanded"]],
                    name=f"{prefix}/ExpandScale",
                ),
                helper.make_node(
                    "Expand", [bias_name, names["shape"]], [names["bias_expanded"]],
                    name=f"{prefix}/ExpandBias",
                ),
                helper.make_node(
                    "Mul", [names["normalized"], names["scale_expanded"]], [names["scaled"]],
                    name=f"{prefix}/Scale",
                ),
                helper.make_node(
                    "Add", [names["scaled"], names["bias_expanded"]], [node.output[0]],
                    name=f"{prefix}/Bias",
                ),
            ]
        )
        decomposed.append(
            {
                "node": node.name,
                "axis": axis,
                "epsilon": epsilon,
                "feature_count": int(scale_values.shape[0]),
            }
        )
    if len(decomposed) != 24:
        raise RuntimeError(
            f"Expected 24 known SoVITS LayerNormalization nodes, found {len(decomposed)}"
        )
    del model.graph.node[:]
    model.graph.node.extend(replacement_nodes)
    onnx.checker.check_model(model)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(target))
    return {
        "workaround": "NVIDIA/TensorRT#4743",
        "mathematically_equivalent": True,
        "patched_initializers": patched,
        "patched_matmul_weights": patched_weights,
        "decomposed_layer_normalizations": decomposed,
        "source_sha256": sha256_file(source),
        "patched_sha256": sha256_file(target),
    }


def _validate_engine(stage: str, engine_path: Path) -> tuple[dict[str, object], ...]:
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"TensorRT cannot deserialize {engine_path}")
    inputs: list[str] = []
    outputs: list[str] = []
    description: list[dict[str, object]] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        (inputs if mode == trt.TensorIOMode.INPUT else outputs).append(name)
        description.append(
            {
                "name": name,
                "mode": str(mode),
                "dtype": str(engine.get_tensor_dtype(name)),
                "shape": [int(value) for value in engine.get_tensor_shape(name)],
                "location": str(engine.get_tensor_location(name)),
            }
        )
    expected_inputs, expected_outputs = STAGE_IO_CONTRACTS[stage]
    if set(inputs) != set(expected_inputs) or set(outputs) != set(expected_outputs):
        raise RuntimeError(
            f"{stage} engine I/O mismatch: inputs={inputs}, outputs={outputs}, "
            f"expected_inputs={expected_inputs}, expected_outputs={expected_outputs}"
        )
    return tuple(description)


def _portable_build_result(result: dict[str, Any], bundle_root: Path) -> dict[str, Any]:
    """Replace staging paths with portable paths relative to the engine bundle."""
    portable = dict(result)
    for key in ("engine_path", "inspector_path", "logger_path", "timing_cache_path"):
        value = portable.get(key)
        if value is None:
            continue
        path = Path(str(value)).resolve()
        try:
            portable[key] = path.relative_to(bundle_root.resolve()).as_posix()
        except ValueError as error:
            raise RuntimeError(
                f"Build result {key} escapes the engine bundle: {path}"
            ) from error
    return portable


def build_engines(
    paths: ModelPaths,
    *,
    workspace_mib: int = 4096,
    optimization_level: int = 5,
) -> Path:
    version = ensure_tensorrt11()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; real TensorRT engines cannot be built")
    onnx_hashes = validate_onnx_bundle(paths.onnx_dir)
    target = paths.engine_dir.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex[:12]}"
    staging.mkdir(parents=True)
    timing_cache = staging / "timing.cache"
    build_results: dict[str, Any] = {}
    try:
        patched_sovits: dict[str, Path] = {}
        graph_patch: dict[str, Any] = {}
        for stage in ("sovits", "sovits_stream"):
            patched = staging / "source" / f"{stage}.trt11.onnx"
            graph_patch[stage] = patch_sovits_myelin_broadcast(
                paths.onnx_dir / f"{stage}.onnx", patched
            )
            patched_sovits[stage] = patched
        for stage in STAGE_ORDER:
            print(f"[converter] Building TensorRT 11 engine: {stage}", flush=True)
            logger = DetailedTensorRTLogger(echo=False)
            builder = TensorRTEngineBuilder(
                workspace_bytes=workspace_mib * 1024 * 1024,
                allow_tf32=False,
                builder_optimization_level=optimization_level,
                logger=logger,
            )
            result = builder.build(
                patched_sovits.get(stage, paths.onnx_dir / f"{stage}.onnx"),
                staging / f"{stage}.engine",
                profiles=profiles_for(stage),
                timing_cache_path=timing_cache,
                inspector_path=staging / "inspectors" / f"{stage}.json",
                logger_path=staging / "logs" / f"{stage}.log",
            )
            engine_io = _validate_engine(stage, staging / f"{stage}.engine")
            build_results[stage] = _portable_build_result(
                {**result.to_dict(), "validated_io": engine_io}, staging
            )

        shutil.copy2(paths.onnx_dir / "config.json", staging / "config.json")
        manifest = {
            "schema": 1,
            "kind": "aniflive-tts-gsv-v2proplus-tensorrt11-engines",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "tensorrt": version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "precision": "strongly-typed ONNX FP16/FP32",
            "tf32_allowed": False,
            "workspace_mib": workspace_mib,
            "builder_optimization_level": optimization_level,
            "payload": {"settings": {"profile": "fitted"}},
            "onnx_sha256": onnx_hashes,
            "graph_patch": graph_patch,
            "engine_sha256": {
                stage: sha256_file(staging / f"{stage}.engine") for stage in STAGE_ORDER
            },
            "build_results": build_results,
        }
        (staging / "engine-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            backup = target.parent / f"{target.name}.previous-{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
            target.replace(backup)
        staging.replace(target)
        return target
    except Exception:
        failed = target.parent / f"failed-build-{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
        if staging.exists():
            staging.replace(failed)
        raise


def main(argv: list[str] | None = None) -> int:
    del argv
    raise SystemExit(
        "legacy_converter is an internal TensorRT builder; use "
        "'aniflive-tts model convert' or 'aniflive-tts model rebuild-engines'"
    )
