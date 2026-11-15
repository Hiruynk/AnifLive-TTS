"""One-time lossless container conversion, not an inference fallback."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args()
    actual = digest(args.source)
    if actual != args.expected_sha256:
        raise ValueError('Source checkpoint SHA256 does not match trusted manifest')
    checkpoint = torch.load(args.source, map_location='cpu', weights_only=True)
    state = checkpoint.get('weight', checkpoint)
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a nonempty tensor state dictionary')
    tensors = {k: v.detach().contiguous().clone() for k, v in state.items()}
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.destination), metadata={'source_sha256': actual, 'converter': 'aniflive-lossless-container-v1'})
    restored = load_file(str(args.destination))
    for key, source in tensors.items():
        target = restored[key]
        if source.dtype != target.dtype or source.shape != target.shape or not torch.equal(source.view(torch.uint8), target.view(torch.uint8)):
            raise ValueError(f'Non-lossless tensor conversion: {key}')
    report = {'schema': 1, 'source_sha256': actual, 'output_sha256': digest(args.destination), 'tensor_count': len(tensors),
              'weight_values_modified': False, 'quantized': False, 'all_tensor_bytes_verified': True,
              'tensors': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in tensors.items()},
              'config': checkpoint.get('config'), 'torch_version': torch.__version__}
    args.destination.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('source_sha256','output_sha256','tensor_count','all_tensor_bytes_verified')}))


if __name__ == '__main__':
    main()
