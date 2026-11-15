"""Load/register the experimental C++ out-projection plugin before TRT parsing."""
import ctypes
import hashlib
from pathlib import Path


_loaded = {}


def load(path):
    """Register and retain the library; return its serializable absolute path."""
    import torch  # Load the matching LibTorch dependencies first.
    import tensorrt as trt
    if trt.__version__ != '11.2.1.2' or torch.__version__ != '2.10.0+cu128':
        raise RuntimeError('The C++ out-projection plugin requires the matching TensorRT/PyTorch ABI')
    path = str(Path(path).resolve(strict=True))
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if path in _loaded:
        if digest != _loaded[path][0]:
            raise RuntimeError('The loaded C++ plugin file changed')
        return path
    if _loaded:
        loaded_digest, _ = next(iter(_loaded.values()))
        if digest != loaded_digest:
            raise RuntimeError('A different C++ out-projection plugin library is already loaded')
        return path
    registry = trt.get_plugin_registry()
    if registry.get_creator('cpp_native_layernorm_direct', '1', 'aniflive_research') is not None:
        raise RuntimeError('C++ native out-projection creator is already registered externally')
    library = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    register = library.register_cpp_native_layernorm_direct
    register.argtypes = []
    register.restype = ctypes.c_int
    if register() != 0:
        raise RuntimeError('C++ native out-projection creator registration failed')
    if registry.get_creator('cpp_native_layernorm_direct', '1', 'aniflive_research') is None:
        raise RuntimeError('TensorRT did not expose the registered C++ out-projection creator')
    _loaded[path] = (digest, library)  # Retain creator code until process exit.
    return path
