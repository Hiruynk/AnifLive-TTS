from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aniflive_tts.dataset_acquisition import DatasetAcquisitionConfig
from aniflive_tts import workstation_vad


def _model_root(tmp_path: Path) -> Path:
    root = tmp_path / "fsmn-vad"
    root.mkdir()
    for name in ("am.mvn", "config.yaml", "configuration.json", "model.pt"):
        (root / name).write_bytes(f"pinned-{name}".encode("ascii"))
    return root


class _FakeVad:
    def __init__(self, value):
        self.value = value

    def generate(self, *, input, fs):
        assert input.dtype == np.float32
        assert input.ndim == 1
        assert fs == 16_000
        return [{"value": self.value}]


def test_fsmn_vad_converts_ranges_and_records_model_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workstation_vad, "assert_linux_docker_runtime", lambda: None)
    root = _model_root(tmp_path)
    detector = workstation_vad.WorkstationFsmnVad(
        root,
        model_factory=lambda _path: (
            _FakeVad([[100, 500], [900, 1900]]),
            {"funasr": "test"},
        ),
    )

    segments, profile = detector.detect(
        np.full(32_000, 0.1, dtype=np.float32),
        16_000,
        context_ms=40,
        maximum_segment_ms=600,
    )

    assert [(item.start_sample, item.end_sample) for item in segments] == [
        (960, 8_640),
        (13_760, 23_360),
        (23_360, 31_040),
    ]
    assert profile["backend"] == workstation_vad.FSMN_VAD_BACKEND
    assert profile["network_required"] is False
    assert profile["model"]["file_count"] == 4
    assert len(profile["model"]["tree_sha256"]) == 64
    assert [item.forced_split for item in segments] == [False, True, False]
    assert profile["forced_split_count"] == 1


def test_fsmn_vad_prefers_a_silence_validated_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workstation_vad, "assert_linux_docker_runtime", lambda: None)
    audio = np.full(32_000, 0.1, dtype=np.float32)
    audio[8_800:10_400] = 0.0
    detector = workstation_vad.WorkstationFsmnVad(
        _model_root(tmp_path),
        model_factory=lambda _path: (_FakeVad([[0, 2000]]), {"funasr": "test"}),
    )

    segments, profile = detector.detect(
        audio,
        16_000,
        context_ms=0,
        maximum_segment_ms=1_000,
    )

    assert len(segments) == 3
    assert segments[0].silence_validated is True
    assert segments[0].forced_split is False
    assert 9_000 <= segments[0].end_sample <= 10_200
    assert profile["forced_split_count"] == 1


def test_fsmn_vad_rejects_unordered_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workstation_vad, "assert_linux_docker_runtime", lambda: None)
    detector = workstation_vad.WorkstationFsmnVad(
        _model_root(tmp_path),
        model_factory=lambda _path: (_FakeVad([[500, 800], [300, 700]]), {}),
    )
    with pytest.raises(workstation_vad.WorkstationVadError, match="invalid or unordered"):
        detector.detect(np.ones(16_000, dtype=np.float32), 16_000)


def test_fsmn_vad_requires_exact_offline_inventory(tmp_path: Path) -> None:
    root = _model_root(tmp_path)
    (root / "untracked.bin").write_bytes(b"unexpected")
    with pytest.raises(workstation_vad.WorkstationVadError, match="inventory"):
        workstation_vad._inspect_model(root)


def test_fsmn_vad_never_silently_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workstation_vad, "assert_linux_docker_runtime", lambda: None)
    root = _model_root(tmp_path)

    def fail(_path: Path):
        raise RuntimeError("neural backend failed")

    detector = workstation_vad.WorkstationFsmnVad(root, model_factory=fail)
    with pytest.raises(RuntimeError, match="neural backend failed"):
        detector.detect(np.ones(16_000, dtype=np.float32), 16_000)


def test_target_speaker_config_uses_managed_fsmn_vad() -> None:
    config = DatasetAcquisitionConfig.from_mapping(
        {
            "acquisition_mode": "target-speaker",
            "sources": ["C:/voice/source.wav"],
            "reference_audio": "C:/voice/reference.wav",
        }
    )
    assert config.vad_component == "fsmn-vad"
    assert config.as_dict()["vad_component"] == "fsmn-vad"
