from __future__ import annotations

import os
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aniflive_tts.dataset_asr import (
    DATASET_ASR_BACKEND,
    SENSEVOICE_ASR_BACKEND,
    DatasetAsrError,
    inspect_ct2_whisper_model,
    transcribe_sensevoice_segments,
    transcribe_dataset_segments,
)
from aniflive_tts.dataset_pipeline import DatasetPipelineConfig, DatasetPipelineError


def _model(root: Path) -> Path:
    root.mkdir()
    (root / "model.bin").write_bytes(b"fixed-ctranslate2-weights")
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (root / "config.json").write_text('{"lang_ids":[1,2,3,4,5]}\n', encoding="utf-8")
    return root


def _segment(root: Path) -> tuple[list[dict[str, object]], Path]:
    segment_root = root / "segments"
    segment_root.mkdir(parents=True)
    path = segment_root / "segment-00000.wav"
    samples = np.zeros(3_200, dtype="<i2")
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(32_000)
        audio.writeframes(samples.tobytes())
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return [{"position": 0, "path": "segments/segment-00000.wav", "sha256": digest}], path


class _FakeWhisper:
    def __init__(self, text: str, detected_language: str, callback=None) -> None:
        self.text = text
        self.detected_language = detected_language
        self.callback = callback
        self.calls: list[tuple[str, dict[str, object]]] = []

    def transcribe(self, path: str, **kwargs):
        self.calls.append((path, kwargs))
        if self.callback is not None:
            self.callback()
        return iter([SimpleNamespace(text=self.text)]), SimpleNamespace(
            language=self.detected_language,
            language_probability=0.97,
        )


def test_ct2_model_inventory_is_complete_and_stable(tmp_path: Path) -> None:
    root = _model(tmp_path / "asr")
    first = inspect_ct2_whisper_model(root)
    second = inspect_ct2_whisper_model(root)
    assert first == second
    assert first.file_count == 3
    assert first.total_bytes > 0
    assert len(first.tree_sha256) == 64
    assert set(first.critical_files) == {"config.json", "model.bin", "tokenizer.json"}


def test_ct2_model_rejects_non_multilingual_or_incomplete_assets(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "config.json").write_text('{"lang_ids":[1,2,3,4,5]}', encoding="utf-8")
    with pytest.raises(DatasetAsrError, match="CTranslate2 Whisper"):
        inspect_ct2_whisper_model(missing)

    monolingual = _model(tmp_path / "mono")
    (monolingual / "config.json").write_text('{"lang_ids":[1]}', encoding="utf-8")
    with pytest.raises(DatasetAsrError, match="multilingual"):
        inspect_ct2_whisper_model(monolingual)


@pytest.mark.parametrize(
    ("language", "detected", "text"),
    [
        ("yue", "zh", "我今日真係好開心。"),
        ("zh", "zh", "今天天气很好。"),
        ("ja", "ja", "今日は良い天気です。"),
        ("en", "en", "Today is a good day."),
        ("ko", "ko", "오늘은 좋은 날입니다."),
    ],
)
def test_offline_asr_uses_one_fixed_five_language_cuda_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    language: str,
    detected: str,
    text: str,
) -> None:
    model_root = _model(tmp_path / "asr")
    records, _path = _segment(tmp_path / "stage")
    fake = _FakeWhisper(text, detected)
    monkeypatch.setattr("aniflive_tts.dataset_pipeline.assert_linux_docker_runtime", lambda: None)
    report = transcribe_dataset_segments(
        records,
        stage_root=tmp_path / "stage",
        model_path=model_root,
        declared_language=language,
        model_factory=lambda path: (fake, {"faster_whisper": "test", "ctranslate2": "test"}),
    )
    assert report["backend"] == DATASET_ASR_BACKEND
    assert report["language"] == language
    assert report["text"] == text
    assert report["network_required"] is False
    assert report["review_required"] is True
    assert "offline-asr-requires-human-review" in report["review_reasons"]
    assert report["model_qualification"] == (
        "structurally-verified-content-review-required"
    )
    assert report["decoding"] == {
        "device": "cuda:0",
        "compute_type": "float16",
        "beam_size": 5,
        "temperature": 0.0,
        "vad_filter": False,
        "condition_on_previous_text": False,
        "local_files_only": True,
    }
    assert fake.calls[0][1]["language"] == ("zh" if language == "yue" else language)
    assert fake.calls[0][1]["condition_on_previous_text"] is False


def test_offline_asr_rejects_model_mutation_during_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = _model(tmp_path / "asr")
    records, _path = _segment(tmp_path / "stage")
    fake = _FakeWhisper(
        "Stable text.",
        "en",
        callback=lambda: (model_root / "model.bin").write_bytes(b"changed"),
    )
    monkeypatch.setattr("aniflive_tts.dataset_pipeline.assert_linux_docker_runtime", lambda: None)
    with pytest.raises(DatasetAsrError, match="model changed"):
        transcribe_dataset_segments(
            records,
            stage_root=tmp_path / "stage",
            model_path=model_root,
            declared_language="en",
            model_factory=lambda path: (fake, {}),
        )


def test_offline_asr_marks_repetitive_whisper_output_for_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = _model(tmp_path / "asr")
    records, _path = _segment(tmp_path / "stage")
    fake = _FakeWhisper("稍微往後後後後後後後後", "zh")
    monkeypatch.setattr("aniflive_tts.dataset_pipeline.assert_linux_docker_runtime", lambda: None)
    report = transcribe_dataset_segments(
        records,
        stage_root=tmp_path / "stage",
        model_path=model_root,
        declared_language="yue",
        model_factory=lambda path: (fake, {}),
    )
    assert report["review_required"] is True
    assert "whisper-repetition-detected" in report["review_reasons"]
    assert "whisper-repetition-detected" in report["segments"][0]["review_reasons"]


def test_sensevoice_auto_language_and_emotion_remain_review_suggestions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "sensevoice"
    model_root.mkdir()
    (model_root / "model.pt").write_bytes(b"fixed-sensevoice-weights")
    (model_root / "config.json").write_text("{}\n", encoding="utf-8")
    records, _path = _segment(tmp_path / "stage")

    class FakeSenseVoice:
        def generate(self, **_kwargs):
            return [{"text": "<|yue|><|HAPPY|><|Speech|>今日真係好開心。"}]

    monkeypatch.setattr(
        "aniflive_tts.dataset_pipeline.assert_linux_docker_runtime", lambda: None
    )
    report = transcribe_sensevoice_segments(
        records,
        stage_root=tmp_path / "stage",
        model_path=model_root,
        model_factory=lambda path: (FakeSenseVoice(), {"funasr": "test"}),
    )

    assert report["backend"] == SENSEVOICE_ASR_BACKEND
    assert report["language"] == "yue"
    assert report["segments"][0]["language"] == "yue"
    assert report["segments"][0]["emotion_suggestion"] == "happy"
    assert report["segments"][0]["review_required"] is True
    assert "offline-asr-requires-human-review" in report["review_reasons"]


def test_sensevoice_removes_decoder_token_spacing_from_japanese_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "sensevoice"
    model_root.mkdir()
    (model_root / "model.pt").write_bytes(b"fixed-sensevoice-weights")
    (model_root / "config.json").write_text("{}\n", encoding="utf-8")
    records, _path = _segment(tmp_path / "stage")

    class FakeSenseVoice:
        def generate(self, **_kwargs):
            return [{"text": "<|ja|><|Speech|>ありがとう ございます、AI モデルです。"}]

    monkeypatch.setattr(
        "aniflive_tts.dataset_pipeline.assert_linux_docker_runtime", lambda: None
    )
    report = transcribe_sensevoice_segments(
        records,
        stage_root=tmp_path / "stage",
        model_path=model_root,
        declared_language="ja",
        model_factory=lambda path: (FakeSenseVoice(), {"funasr": "test"}),
    )

    assert report["segments"][0]["text"] == "ありがとうございます、AI モデルです。"
    assert "sensevoice-cjk-token-spacing-normalized" in report["review_reasons"]


@pytest.mark.skipif(os.name != "nt", reason="Windows runtime rejection contract")
def test_offline_asr_is_not_callable_as_windows_neural_inference(tmp_path: Path) -> None:
    model_root = _model(tmp_path / "asr")
    records, _path = _segment(tmp_path / "stage")
    with pytest.raises((DatasetAsrError, DatasetPipelineError), match="Linux Docker"):
        transcribe_dataset_segments(
            records,
            stage_root=tmp_path / "stage",
            model_path=model_root,
            declared_language="en",
            model_factory=lambda path: (_FakeWhisper("text", "en"), {}),
        )


def test_dereverb_contract_rejects_unqualified_backends() -> None:
    assert DatasetPipelineConfig(dereverb_backend="none").validated().dereverb_backend == "none"
    with pytest.raises(DatasetPipelineError, match="redistribution-safe"):
        DatasetPipelineConfig(dereverb_backend="resemble-enhance").validated()
