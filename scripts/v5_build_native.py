"""Build the V5 Turbo native runtime libraries from source without a GPU.

Copyright 2026 Hiruynk. Licensed under the repository LICENSE.
This builds native helpers only; TensorRT model plans require a separate export/build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

EXPECTED_CHECKPOINT = "87133414860ea14ff6620c483a3db5ed07b44be42e2c3fcdad65523a729a745a"
TRT_VERSION = "11.2.1.2"
TRT_COMMIT = "505170043e0e2799bfcfe8fab777a78d2fb52854"
SOURCES = Path(__file__).resolve().parent / "v5_native_sources"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_headers(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("version") != TRT_VERSION or manifest.get("commit") != TRT_COMMIT:
        raise ValueError("TensorRT headers do not match the pinned version and source commit")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("TensorRT header inventory is missing")
    names = set()
    for row in files:
        name = row.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+\.h", name) or name in names:
            raise ValueError("Invalid TensorRT header inventory")
        names.add(name)
        path = root / "include" / name
        if not path.is_file() or path.stat().st_size != row.get("bytes") or sha256(path) != row.get("sha256"):
            raise ValueError(f"TensorRT header checksum mismatch: {name}")
    if not {"NvInferRuntime.h", "NvInferVersion.h"}.issubset(names):
        raise ValueError("Required TensorRT headers are not inventoried")


def checkpoint_hashes(checkpoint: Path) -> list[str]:
    import torch
    if sha256(checkpoint) != EXPECTED_CHECKPOINT:
        raise ValueError("Checkpoint does not match the pinned V5 Turbo AR weights")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = payload["weight"]
    hashes = []
    for index in range(24):
        tensor = weights[f"model.h.layers.{index}.linear2.weight"]
        if tuple(tensor.shape) != (512, 2048) or tensor.dtype != torch.float16:
            raise ValueError(f"Unexpected FFN2 weight layout in layer {index}")
        hashes.append(hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest())
    if len(set(hashes)) != 24:
        raise ValueError("FFN2 weight hashes are not unique")
    return hashes


def write_hash_header(path: Path, hashes: list[str]) -> None:
    if len(hashes) != 24 or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
        raise ValueError("Expected 24 SHA-256 hashes")
    path.write_text("static const char *expected_ffn2_hashes[24] = {\n" +
                    "".join(f'    "{value}",\n' for value in hashes) + "};\n", encoding="ascii")


def build(checkpoint: Path, headers: Path, output: Path) -> None:
    import torch
    import tensorrt as trt
    if sys.platform != "linux":
        raise RuntimeError("V5 native helpers must be built in the target Linux container")
    if torch.__version__ != "2.10.0+cu128" or trt.__version__ != TRT_VERSION:
        raise RuntimeError("V5 native helpers require PyTorch 2.10.0+cu128 and TensorRT 11.2.1.2")
    gcc, cxx = shutil.which("gcc"), shutil.which("g++")
    if not gcc or not cxx:
        raise RuntimeError("gcc and g++ are required")
    validate_headers(headers)
    hashes = checkpoint_hashes(checkpoint)
    site = Path(torch.__file__).resolve().parent.parent
    torch_root = Path(torch.__file__).resolve().parent
    cuda = site / "nvidia" / "cuda_runtime"
    triton_include = site / "triton" / "backends" / "nvidia" / "include"
    trt_lib = site / "tensorrt_libs"
    cuda_header = cuda / "include" / "cuda.h"
    if not re.search(r"^#define CUDA_VERSION 12080$", cuda_header.read_text(), re.MULTILINE):
        raise RuntimeError("CUDA 12.8 runtime headers are required")
    if not (triton_include / "crt" / "host_defines.h").is_file():
        raise RuntimeError("Matching CUDA CRT headers are required")
    output.mkdir(parents=True, exist_ok=False)
    write_hash_header(output / "ffn2_candidate_hashes.h", hashes)
    common = [gcc, "-std=gnu11", "-O2", "-shared", "-fPIC", f"-I{cuda / 'include'}", f"-I{output}"]
    entries = [
        ("attention_guard", "attention_native_k8_shared_alignment_guard.c", "attention_guard.so"),
        ("gemv_guard", "nativeorder_residual_full_alignment_guard.c", "gemv_guard.so"),
        ("weight_l2", "native_ffn2_l2_scan.c", "weight_l2.so"),
        ("joint_graph", "native_ffn2_joint_l2_scan.c", "joint_graph.so"),
    ]
    built = {}
    for key, source, name in entries:
        target = output / name
        subprocess.run([*common, str(SOURCES / source), "-ldl", "-o", str(target)], check=True)
        built[key] = {"file": name, "sha256": sha256(target)}
    target = output / "layernorm_plugin.so"
    torch_lib = torch_root / "lib"
    cuda_lib = cuda / "lib"
    command = [cxx, "-std=c++17", "-O0", "-g0", "-fPIC", "-shared",
        "-DHAVE_AVX2_CPU_DEFINITION", "-DHAVE_AVX512_CPU_DEFINITION",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        f"-I{headers / 'include'}", f"-I{torch_root / 'include'}",
        f"-I{torch_root / 'include/torch/csrc/api/include'}", f"-I{cuda / 'include'}",
        f"-I{triton_include}", str(SOURCES / "trt_cpp_native_layernorm_direct.cpp"),
        "-o", str(target), f"-L{torch_lib}", f"-L{trt_lib}", f"-L{cuda_lib}",
        "-Wl,--no-undefined", "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda",
        "-l:libnvinfer.so.11", "-l:libcudart.so.12", f"-Wl,-rpath,{torch_lib}",
        f"-Wl,-rpath,{trt_lib}", f"-Wl,-rpath,{cuda_lib}"]
    subprocess.run(command, check=True)
    built["layernorm_plugin"] = {"file": target.name, "sha256": sha256(target)}
    report = {"status": "built_cpu_only", "checkpoint_sha256": EXPECTED_CHECKPOINT,
              "tensorrt": TRT_VERSION, "torch": torch.__version__,
              "header_manifest_sha256": sha256(headers / "manifest.json"), "files": built,
              "ffn2_sha256": hashes}
    (output / "native-build.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tensorrt-headers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    args = parser.parse_args()
    build(args.checkpoint.resolve(strict=True), args.tensorrt_headers.resolve(strict=True), args.output.resolve())


if __name__ == "__main__":
    main()
