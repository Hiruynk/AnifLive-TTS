"""Build the V5 encoder plan in a fresh TensorRT process.

Copyright 2026 Hiruynk. Licensed under the repository LICENSE.
"""
import argparse
import hashlib
from pathlib import Path
import tensorrt as trt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256()
    with args.onnx.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != "764cef38f7fd0a5220cb5da6edb03d7346bcb86a4e65ed96cda374567fb10416":
        raise ValueError("Encoder graph does not match the pinned export")
    if trt.__version__ != "11.2.1.2":
        raise RuntimeError("TensorRT 11.2.1.2 is required")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser_trt = trt.OnnxParser(network, logger)
    if not parser_trt.parse_from_file(str(args.onnx)):
        raise RuntimeError("TensorRT ONNX parser: " + "; ".join(str(parser_trt.get_error(i)) for i in range(parser_trt.num_errors)))
    inputs = {network.get_input(i).name: network.get_input(i) for i in range(network.num_inputs)}
    if (set(inputs) != {"codes", "phones", "ge"} or network.num_outputs != 1
            or inputs["codes"].dtype != trt.int64 or inputs["phones"].dtype != trt.int64
            or inputs["ge"].dtype != trt.float16 or tuple(inputs["codes"].shape) != (1,1,-1)
            or tuple(inputs["phones"].shape) != (1,-1)
            or network.get_output(0).name != "features" or network.get_output(0).dtype != trt.float16):
        raise RuntimeError("Unexpected encoder graph contract")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.builder_optimization_level = 3
    config.max_aux_streams = 0
    profile = builder.create_optimization_profile()
    for name, shapes in {"codes": ((1,1,1),(1,1,80),(1,1,1536)),
                         "phones": ((1,1),(1,96),(1,2048))}.items():
        profile.set_shape(name, *shapes)
        if tuple(map(tuple, profile.get_shape(name))) != shapes:
            raise RuntimeError(f"TensorRT profile mismatch for {name}")
    if not profile or config.add_optimization_profile(profile) < 0:
        raise RuntimeError("TensorRT rejected the encoder profile")
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT encoder build failed")
    args.output.write_bytes(memoryview(blob).tobytes())


if __name__ == "__main__":
    main()
