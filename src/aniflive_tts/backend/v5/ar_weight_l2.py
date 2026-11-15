"""Validate captured AR FFN weights before assigning graph-node L2 policies."""
from collections import Counter
import ctypes
import hashlib
import json
from pathlib import Path


WEIGHT_BYTES = 2048 * 512 * 2
EXPECTED_NODES = 24


class WeightL2:
    def __init__(self, shim_path, reference_report, *, role='ffn1'):
        if role not in ('ffn1', 'ffn2'):
            raise ValueError('L2 weight role must be ffn1 or ffn2')
        self.role = role
        self.shim_path = Path(shim_path).resolve(strict=True)
        self.library = ctypes.CDLL(str(self.shim_path))
        c, p = ctypes, ctypes.POINTER
        signatures = {
            'l2_init': [c.c_char_p, c.c_size_t],
            'l2_inspect': [c.c_size_t, p(c.c_size_t), p(c.c_size_t), c.c_size_t, p(c.c_size_t), c.c_char_p, c.c_size_t],
            'l2_copy_weight': [c.c_size_t, c.c_void_p, c.c_size_t, c.c_char_p, c.c_size_t],
            'l2_apply': [c.c_size_t, c.c_size_t, c.c_size_t, c.c_float, c.c_char_p, c.c_size_t],
            'l2_get_limit': [p(c.c_size_t), c.c_char_p, c.c_size_t],
            'l2_set_limit': [c.c_size_t, c.c_char_p, c.c_size_t],
            'l2_reset': [c.c_char_p, c.c_size_t],
        }
        if role == 'ffn2':
            signatures['l2_inspect_ffn2'] = list(signatures['l2_inspect'])
        for name, argtypes in signatures.items():
            method = getattr(self.library, name)
            method.argtypes = argtypes
            method.restype = c.c_int
        self._call('l2_init')
        report = json.loads(Path(reference_report).read_text(encoding='utf-8'))
        offset, shape = (2, (512, 2048)) if role == 'ffn1' else (3, (2048, 512))
        weights = sorted((row for row in report['projection_aot_weights'] if row['index'] % 4 == offset),
                         key=lambda row: row['index'])
        if ([row['index'] for row in weights] != list(range(offset, 96, 4))
                or any((row['input_features'], row['output_features']) != shape for row in weights)):
            raise ValueError(f'Reference report must contain the 24 native-layout {role} weights')
        self.expected_hashes = Counter(row['native_sha256'] for row in weights)
        self.selected_weights = [{'index': row['index'], 'native_sha256': row['native_sha256']} for row in weights[:15]]

    def _call(self, name, *args):
        error = ctypes.create_string_buffer(1024)
        status = getattr(self.library, name)(*args, error, len(error))
        if status:
            raise RuntimeError(f'{name}: {error.value.decode(errors="replace")}')

    def get_limit(self):
        result = ctypes.c_size_t()
        self._call('l2_get_limit', ctypes.byref(result))
        return result.value

    def set_limit(self, value):
        self._call('l2_set_limit', int(value))

    def reset(self):
        self._call('l2_reset')

    def inspect_and_apply(self, graph, *, hit_ratio):
        if not 0.0 < float(hit_ratio) <= 1.0:
            raise ValueError('L2 hit ratio must be in (0,1]')
        nodes = (ctypes.c_size_t * EXPECTED_NODES)()
        pointers = (ctypes.c_size_t * EXPECTED_NODES)()
        found = ctypes.c_size_t()
        self._call('l2_inspect_ffn2' if self.role == 'ffn2' else 'l2_inspect',
                   int(graph.raw_cuda_graph()), nodes, pointers,
                   EXPECTED_NODES, ctypes.byref(found))
        if found.value != EXPECTED_NODES:
            raise RuntimeError(f'Expected 24 validated {self.role} nodes; found {found.value}')
        buffer = ctypes.create_string_buffer(WEIGHT_BYTES)
        hashes = []
        for pointer in pointers:
            self._call('l2_copy_weight', pointer, buffer, WEIGHT_BYTES)
            hashes.append(hashlib.sha256(buffer.raw).hexdigest())
        if Counter(hashes) != self.expected_hashes:
            raise RuntimeError(f'Captured {self.role} weight bytes do not match the qualified source manifest')
        remaining = Counter(row['native_sha256'] for row in self.selected_weights)
        selected_nodes = 0
        for node, pointer, digest in zip(nodes, pointers, hashes):
            if remaining[digest]:
                self._call('l2_apply', node, pointer, WEIGHT_BYTES, float(hit_ratio))
                remaining[digest] -= 1
                selected_nodes += 1
        if selected_nodes != 15 or any(remaining.values()):
            raise RuntimeError(f'Could not match all 15 selected {self.role} weight nodes')
        return {'role': self.role, 'nodes_verified': found.value, 'nodes_with_policy': selected_nodes,
                'weight_bytes_each': WEIGHT_BYTES, 'hit_ratio': float(hit_ratio),
                'weight_sha256': hashes, 'selected_weights': self.selected_weights,
                'policy': f'first 15 qualified {self.role} weights by index, persisting hit; normal on miss'}
