from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class ModelBackend(Protocol):
    """Manifest-level contract for one supported model family.

    Runtime implementations remain responsible for neural execution. This
    contract keeps package identity and compatibility policy out of callers.
    """

    model_family: str
    package_format: str
    precision: str
    display_name: str
    engine_manifest_kind: str

    def supports_manifest(
        self,
        manifest: Mapping[str, Any],
        *,
        require_format: bool = True,
        require_precision: bool = True,
    ) -> bool: ...

    def manifest_fields(self) -> dict[str, str]: ...


@dataclass(frozen=True, slots=True)
class V2ProPlusBackend:
    model_family: str = "gsv-v2proplus"
    package_format: str = "aniflive-tts-model-package"
    precision: str = "FP16"
    display_name: str = "GPT-SoVITS V2 Pro Plus, TensorRT 11 only"
    engine_manifest_kind: str = "aniflive-tts-gsv-v2proplus-tensorrt11-engines"

    def supports_manifest(
        self,
        manifest: Mapping[str, Any],
        *,
        require_format: bool = True,
        require_precision: bool = True,
    ) -> bool:
        if manifest.get("model_family") != self.model_family:
            return False
        if require_format and manifest.get("format") != self.package_format:
            return False
        if require_precision and manifest.get("precision") != self.precision:
            return False
        return True

    def manifest_fields(self) -> dict[str, str]:
        return {
            "format": self.package_format,
            "model_family": self.model_family,
            "precision": self.precision,
        }


@dataclass(frozen=True, slots=True)
class V5TurboBackend(V2ProPlusBackend):
    model_family: str = "gsv-v5turbo"
    display_name: str = "GPT-SoVITS V5 Turbo, TensorRT 11 and CUDA"
    engine_manifest_kind: str = "aniflive-tts-gsv-v5turbo-tensorrt11-engines"


@dataclass(frozen=True, slots=True)
class V5TurboMacOSBackend(V5TurboBackend):
    precision: str = "FP16"
    display_name: str = "GPT-SoVITS V5 Turbo, macOS MLX/Metal"
    engine_manifest_kind: str = "aniflive-tts-gsv-v5turbo-mlx-metal"


V5TURBO_BACKEND = V5TurboBackend()
V2PROPLUS_BACKEND = V2ProPlusBackend()
MODEL_BACKENDS: Mapping[str, ModelBackend] = MappingProxyType(
    {V2PROPLUS_BACKEND.model_family: V2PROPLUS_BACKEND, V5TURBO_BACKEND.model_family: V5TURBO_BACKEND}
)


def get_model_backend(model_family: object) -> ModelBackend | None:
    if not isinstance(model_family, str):
        return None
    return MODEL_BACKENDS.get(model_family)


def model_backend_for_manifest(manifest: Mapping[str, Any]) -> ModelBackend | None:
    if manifest.get("model_family") == "gsv-v5turbo" and isinstance(manifest.get("native_runtime"), Mapping):
        precision = manifest.get("precision")
        return V5TurboMacOSBackend(precision=precision) if precision in {"FP16", "FP32"} else None
    return get_model_backend(manifest.get("model_family"))

