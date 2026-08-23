from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def ensure_tensorrt11() -> str:
    """Fail before parsing or deserializing when TensorRT 11 is not active."""

    candidates = ("tensorrt", "tensorrt-cu13", "tensorrt-cu12")
    detected = None
    for package in candidates:
        try:
            detected = version(package)
            break
        except PackageNotFoundError:
            continue
    if detected is None:
        raise RuntimeError("TensorRT 11 is required but is not installed")
    try:
        major = int(detected.split(".", 1)[0])
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"Cannot parse TensorRT version: {detected!r}") from error
    if major != 11:
        raise RuntimeError(f"TensorRT 11 is required, detected {detected}")
    return detected

