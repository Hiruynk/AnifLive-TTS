from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ModelInspectionError
from .model_package import sha256_file


@dataclass(frozen=True)
class CheckpointInspection:
    path: Path
    sha256: str
    kind: str
    model_version: str
    config: dict[str, Any]
    state_key_count: int
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.path.name,
            "sha256": self.sha256,
            "kind": self.kind,
            "model_version": self.model_version,
            "state_key_count": self.state_key_count,
            "evidence": list(self.evidence),
        }


def _load(path: Path, *, allow_unsafe_pickle: bool) -> Any:
    import torch

    def source() -> Path | BytesIO:
        with path.open("rb") as stream:
            header = stream.read(2)
            if header in {b"03", b"04", b"05", b"06"}:
                return BytesIO(b"PK" + stream.read())
        return path

    try:
        return torch.load(source(), map_location="cpu", weights_only=True)
    except Exception as safe_error:
        if not allow_unsafe_pickle:
            raise ModelInspectionError(
                f"Safe checkpoint load failed for {path}. Re-run converter with "
                "--allow-unsafe-pickle only for a trusted local checkpoint. "
                f"Original error: {safe_error}"
            ) from safe_error
        try:
            return torch.load(source(), map_location="cpu", weights_only=False)
        except Exception as unsafe_error:
            raise ModelInspectionError(f"Checkpoint load failed for {path}: {unsafe_error}") from unsafe_error


def _config(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    value = payload.get("config") or payload.get("hps") or {}
    return _plain_mapping(value)


def _plain_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        source = value
    else:
        source = getattr(value, "__dict__", None)
        if not isinstance(source, dict):
            return {}
    result: dict[str, Any] = {}
    for key, item in source.items():
        nested = _plain_mapping(item)
        result[str(key)] = nested if nested else item
    return result


def _state(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    for key in ("weight", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return {key: value for key, value in payload.items() if hasattr(value, "shape")}


def inspect_checkpoint(path: Path, *, kind: str, allow_unsafe_pickle: bool = False) -> CheckpointInspection:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _load(path, allow_unsafe_pickle=allow_unsafe_pickle)
    config = _config(payload)
    state = _state(payload)
    evidence: list[str] = []
    version = str(
        config.get("version")
        or (config.get("model", {}) if isinstance(config.get("model"), dict) else {}).get("version")
        or (payload.get("version") if isinstance(payload, dict) else "")
    )
    normalized = version.lower().replace("_", "").replace("-", "")
    if "v2proplus" in normalized or "v2plusplus" in normalized:
        evidence.append(f"config.version={version}")
    if kind == "gpt":
        model_config = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
        output_dir = str(config.get("output_dir", ""))
        pretrained = str(config.get("pretrained_s1", ""))
        if "v2proplus" in output_dir.lower() or "s1v3" in pretrained.lower():
            evidence.append("GPT config declares the V2ProPlus/s1v3 training family")
        expected = {
            "embedding_dim": 512,
            "n_layer": 24,
            "vocab_size": 1025,
            "phoneme_vocab_size": 732,
        }
        if all(int(model_config.get(key, -1)) == value for key, value in expected.items()):
            evidence.append("GPT config matches the V2ProPlus AR dimensions")
        bert = state.get("model.bert_proj.weight")
        audio_embedding = state.get("model.ar_audio_embedding.word_embeddings.weight")
        if getattr(bert, "shape", None) == (512, 1024) and getattr(
            audio_embedding, "shape", None
        ) == (1025, 512):
            evidence.append("GPT tensor shapes match the V2ProPlus export contract")
    elif kind == "sovits":
        train_config = config.get("train", {}) if isinstance(config.get("train"), dict) else {}
        model_config = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
        pretrained = str(train_config.get("pretrained_s2G", ""))
        if "v2proplus" in pretrained.lower():
            evidence.append("SoVITS training config declares the V2ProPlus generator family")
        ssl_projection = state.get("enc_p.ssl_proj.weight")
        text_embedding = state.get("enc_p.text_embedding.weight")
        speaker_bias = state.get("sv_emb.bias")
        if (
            int(model_config.get("gin_channels", -1)) == 1024
            and getattr(ssl_projection, "shape", None) == (192, 768, 1)
            and getattr(text_embedding, "shape", None) == (732, 192)
            and getattr(speaker_bias, "shape", None) == (1024,)
        ):
            evidence.append("SoVITS tensor shapes match the V2ProPlus export contract")
    else:
        raise ValueError("kind must be 'gpt' or 'sovits'")
    if not state:
        raise ModelInspectionError(f"No tensor state dictionary found in {path}")
    # Require independent configuration and tensor-shape evidence. V2ProPlus
    # SoVITS checkpoints use ``sv_emb`` conditioning; CFM tensors belong to a
    # different model generation and must not be required here.
    if len(evidence) < 2:
        raise ModelInspectionError(
            f"{path} is not a verified GPT-SoVITS V2 Pro Plus {kind} checkpoint; evidence={evidence}"
        )
    return CheckpointInspection(
        path=path,
        sha256=sha256_file(path),
        kind=kind,
        model_version="gsv-v2proplus",
        config=config,
        state_key_count=len(state),
        evidence=tuple(evidence),
    )


def inspect_pair(gpt: Path, sovits: Path, *, allow_unsafe_pickle: bool = False) -> dict[str, Any]:
    left = inspect_checkpoint(gpt, kind="gpt", allow_unsafe_pickle=allow_unsafe_pickle)
    right = inspect_checkpoint(sovits, kind="sovits", allow_unsafe_pickle=allow_unsafe_pickle)
    return {"model_version": "gsv-v2proplus", "gpt": left.to_dict(), "sovits": right.to_dict()}
