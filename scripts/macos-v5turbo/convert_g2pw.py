"""Extract unmodified FP32 ONNX initializers to name-preserving safetensors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file

EXPECTED_ONNX_SHA256 = "2eb3c71fd95117b2e1abef8d2d0cd78aae894bbe7f0fac105ddc9c32ce63cbd0"


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.onnx) != EXPECTED_ONNX_SHA256:
        raise ValueError('Unexpected G2PW ONNX SHA-256; refusing conversion')
    model = onnx.load(str(args.onnx), load_external_data=False)
    tensors = {}
    meta = {}
    for t in model.graph.initializer:
        if t.data_type != onnx.TensorProto.FLOAT:
            continue
        if t.name in tensors:
            raise ValueError(f'duplicate tensor {t.name}')
        a = np.array(numpy_helper.to_array(t), order="C", copy=True)
        if a.dtype != np.float32:
            raise ValueError(f'unexpected dtype {t.name}: {a.dtype}')
        tensors[t.name] = a
        meta[t.name] = {'shape': list(a.shape), 'dtype': 'float32',
                        'raw_sha256': hashlib.sha256(a.tobytes()).hexdigest(),
                        'layout': 'unchanged from ONNX; no transpose'}
    if len(tensors) != 231:
        raise ValueError(f'expected 231 FP32 initializers, got {len(tensors)}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    save_file(tensors, str(args.output))
    manifest = {'source_onnx_sha256': EXPECTED_ONNX_SHA256,
                'onnx_version': onnx.__version__, 'tensor_count': len(tensors),
                'safetensors_sha256': sha256_file(args.output), 'tensors': meta,
                'conversion': 'Lossless FP32 values and names; no transposition or quantization. Integer graph constants are not neural weights.'}
    manifest_path = args.output.with_suffix('.manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'tensors': len(tensors), 'sha256': manifest['safetensors_sha256']}))


if __name__ == '__main__':
    main()
