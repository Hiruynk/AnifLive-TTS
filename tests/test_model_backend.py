from __future__ import annotations

from types import MappingProxyType

from aniflive_tts.model_backend import (
    MODEL_BACKENDS,
    V2PROPLUS_BACKEND,
    ModelBackend,
    V2ProPlusBackend,
    get_model_backend,
    model_backend_for_manifest,
)


def test_v2proplus_backend_implements_generic_contract() -> None:
    backend: ModelBackend = V2ProPlusBackend()

    assert isinstance(backend, ModelBackend)
    assert backend.manifest_fields() == {
        "format": "aniflive-tts-model-package",
        "model_family": "gsv-v2proplus",
        "precision": "FP16",
    }
    assert backend.supports_manifest(backend.manifest_fields()) is True
    assert backend.supports_manifest(
        {**backend.manifest_fields(), "precision": "FP32"}
    ) is False
    assert backend.supports_manifest(
        {**backend.manifest_fields(), "precision": "FP32"},
        require_precision=False,
    ) is True


def test_backend_registry_is_read_only_and_unknown_families_fail_closed() -> None:
    assert isinstance(MODEL_BACKENDS, MappingProxyType)
    assert get_model_backend("gsv-v2proplus") is V2PROPLUS_BACKEND
    assert get_model_backend("gsv-v3") is None
    assert get_model_backend(None) is None
    assert model_backend_for_manifest({"model_family": "gsv-v3"}) is None


def test_v5_cuda_and_metal_packages_are_selected_independently() -> None:
    from aniflive_tts.model_backend import V5TURBO_BACKEND, V5TurboMacOSBackend

    cuda = {"model_family": "gsv-v5turbo", "precision": "FP16"}
    assert model_backend_for_manifest(cuda) is V5TURBO_BACKEND
    for precision in ("FP16", "FP32"):
        metal = model_backend_for_manifest({**cuda, "precision": precision, "native_runtime": {}})
        assert isinstance(metal, V5TurboMacOSBackend)
        assert metal.precision == precision
        assert metal.engine_manifest_kind == "aniflive-tts-gsv-v5turbo-mlx-metal"

