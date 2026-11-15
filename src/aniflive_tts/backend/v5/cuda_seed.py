"""Vendor-equivalent seed setup for a verified CUDA-only research runtime."""
import os
import random

import numpy as np
import torch


_verified = False


def initialize_cuda_only_seed():
    """Verify accelerator/precision assumptions once after CUDA model loading."""
    global _verified
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        raise RuntimeError('CUDA-only seed requires an initialized CUDA runtime')
    for name in ('xpu', 'mtia', 'hpu'):
        backend = getattr(torch, name, None)
        if backend is not None:
            available = getattr(backend, 'is_available', None)
            if available is None:
                raise RuntimeError(f'Cannot verify availability of accelerator {name}')
            if available():
                raise RuntimeError(f'CUDA-only seed cannot omit available accelerator {name}')
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        raise RuntimeError('CUDA-only seed cannot omit an available MPS accelerator')
    private_name = torch._C._get_privateuse1_backend_name()
    private = getattr(torch, private_name, None)
    if private is not None:
        available = getattr(private, 'is_available', None)
        if available is None or available():
            raise RuntimeError(f'Cannot exclude custom accelerator {private_name}')
    elif private_name != 'privateuseone':
        raise RuntimeError(f'Cannot verify registered custom accelerator {private_name}')
    precision = (torch.backends.fp32_precision, torch.backends.cuda.matmul.fp32_precision,
                 torch.backends.cudnn.conv.fp32_precision, torch.backends.cudnn.rnn.fp32_precision)
    if tuple(map(str, precision)) != ('ieee',) * 4:
        raise RuntimeError(f'CUDA-only seed requires the configured IEEE precision policy: {precision}')
    _verified = True


def set_seed(seed):
    if not _verified:
        raise RuntimeError('Initialize the CUDA-only seed path before use')
    seed = int(seed)
    seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
    print(f'Set seed to {seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return seed
