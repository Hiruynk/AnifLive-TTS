"""Verified local C event poller; ctypes releases the GIL during the wait."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path


# CUDA 12.8 driver_types.h: cudaErrorNotReady = 600.
CUDA_ERROR_NOT_READY = 600
SCHEMA = 'aniflive-eos-event-poll-v1'


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(library):
    library = Path(library).resolve(strict=True)
    if library.name != 'eos_event_poll.so':
        raise ValueError('Event poll library must be named eos_event_poll.so')
    manifest = Path(str(library) + '.json')
    data = {'schema': SCHEMA, 'library': str(library), 'library_sha256': _sha(library),
            'source_sha256': _sha(Path(__file__).with_suffix('.c')),
            'build': 'gcc -O2 -fPIC -shared eos_event_poll.c -o eos_event_poll.so'}
    with manifest.open('x', encoding='utf-8') as output:
        json.dump(data, output, indent=2)
    return manifest


class NativeEventPoll:
    def __init__(self, library):
        library = Path(library).resolve(strict=True)
        if library.name != 'eos_event_poll.so':
            raise ValueError('Event poll library must be named eos_event_poll.so')
        manifest = Path(str(library) + '.json')
        data = json.loads(manifest.read_text(encoding='utf-8'))
        if (data.get('schema') != SCHEMA or data.get('library') != str(library)
                or data.get('library_sha256') != _sha(library)
                or data.get('source_sha256') != _sha(Path(__file__).with_suffix('.c'))):
            raise ValueError('Event poll library manifest/source/hash mismatch')
        self.metadata = dict(data)
        # The loaded runtime must be the same CUDA-major runtime as Torch's
        # event. CDLL preserves the handles and releases the GIL for the call.
        self.cudart = ctypes.CDLL('libcudart.so.12')
        query = self.cudart.cudaEventQuery
        query.argtypes = [ctypes.c_void_p]
        query.restype = ctypes.c_int
        self.query_address = ctypes.cast(query, ctypes.c_void_p)
        if not self.query_address.value:
            raise RuntimeError('CUDA event query function has no address')
        class DlInfo(ctypes.Structure):
            _fields_ = [('filename', ctypes.c_char_p), ('base', ctypes.c_void_p),
                        ('symbol', ctypes.c_char_p), ('address', ctypes.c_void_p)]
        dladdr = ctypes.CDLL(None).dladdr
        dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
        dladdr.restype = ctypes.c_int
        info = DlInfo()
        if not dladdr(self.query_address, ctypes.byref(info)) or not info.filename:
            raise RuntimeError('Cannot identify the CUDA runtime used for event polling')
        self.metadata['cuda_runtime_path'] = str(Path(info.filename.decode()).resolve())
        self.library = ctypes.CDLL(str(library))
        self.wait = self.library.eos_event_poll
        self.wait.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
        self.wait.restype = ctypes.c_int

    def poll(self, event):
        handle = event.cuda_event
        if not handle:
            raise RuntimeError('CUDA event has not been recorded')
        count = ctypes.c_uint64()
        status = self.wait(ctypes.c_void_p(handle), self.query_address, CUDA_ERROR_NOT_READY, ctypes.byref(count))
        if status != 0:
            raise RuntimeError(f'CUDA event query failed with status {status}')
        return int(count.value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-manifest', required=True, metavar='LIBRARY')
    args = parser.parse_args()
    print(write_manifest(args.write_manifest))
