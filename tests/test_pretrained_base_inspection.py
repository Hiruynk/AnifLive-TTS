from types import SimpleNamespace

import pytest

from aniflive_tts import inspector
from aniflive_tts.errors import ModelInspectionError


def test_official_pretrained_family_metadata_with_required_shapes(tmp_path, monkeypatch):
    checkpoint = tmp_path / "base.pth"
    checkpoint.write_bytes(b"fixture")
    payload = {
        "info": "pretrained_s2G_v2ProPlus",
        "config": {"model": {"gin_channels": 1024}},
        "weight": {
            "enc_p.ssl_proj.weight": SimpleNamespace(shape=(192, 768, 1)),
            "enc_p.text_embedding.weight": SimpleNamespace(shape=(732, 192)),
            "sv_emb.bias": SimpleNamespace(shape=(1024,)),
        },
    }
    monkeypatch.setattr(inspector, "_load", lambda *a, **k: payload)
    assert inspector.inspect_checkpoint(checkpoint, kind="sovits").model_version == "gsv-v2proplus"
    # Family claims must not substitute for actual export-compatible tensors.
    payload["config"]["version"] = "v2ProPlus"
    payload["weight"]["sv_emb.bias"] = SimpleNamespace(shape=(512,))
    with pytest.raises(ModelInspectionError):
        inspector.inspect_checkpoint(checkpoint, kind="sovits")
