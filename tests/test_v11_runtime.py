from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from aniflive_tts.runtime_control import GpuTelemetrySample, WarmRetentionController
from aniflive_tts.streaming import TensorRTFixedReferenceStreamer


def test_native_latent_overlap_crossfade_never_drops_aligned_samples() -> None:
    phase = np.linspace(0.0, 24.0 * np.pi, 4096, dtype=np.float32)
    previous_tail = np.sin(phase).astype(np.float32)
    current = np.concatenate(
        (previous_tail.copy(), np.sin(phase + 0.25).astype(np.float32))
    )

    merged = TensorRTFixedReferenceStreamer._sola_merge(
        previous_tail,
        current,
        overlap_samples=previous_tail.size,
    )

    assert merged.size == current.size
    assert np.isfinite(merged).all()


def test_short_standalone_native_chunk_routes_around_static_overlap_shape() -> None:
    safe = TensorRTFixedReferenceStreamer._native_stream_shape_is_safe

    assert safe(new_token_count=1, overlap_frames=12, has_overlap=False) is False
    assert safe(new_token_count=6, overlap_frames=12, has_overlap=False) is True
    assert safe(new_token_count=1, overlap_frames=12, has_overlap=True) is True


def test_only_first_segment_uses_and_updates_cross_request_preview_hint() -> None:
    target = TensorRTFixedReferenceStreamer._preview_target_for_segment

    assert target(base_tokens=17, cached_tokens=23, segment_index=0) == 23
    assert target(base_tokens=17, cached_tokens=96, segment_index=0) == 32
    assert target(base_tokens=17, cached_tokens=96, segment_index=1) == 17
    assert target(base_tokens=17, cached_tokens=None, segment_index=0) == 17

    remember = TensorRTFixedReferenceStreamer._remember_preview_target
    assert remember(cached_tokens=21, successful_tokens=96) == 21
    assert remember(cached_tokens=21, successful_tokens=19) == 19
    assert remember(cached_tokens=None, successful_tokens=96) == 32


def test_full_context_refill_finds_preview_tail_without_proportional_crop() -> None:
    rng = np.random.default_rng(20260824)
    full_audio = rng.normal(0.0, 0.2, 16000).astype(np.float32)
    previous_tail = full_audio[2850:6850].copy()

    merged, start, score = TensorRTFixedReferenceStreamer._refill_from_full_context(
        previous_tail,
        full_audio,
        expected_start=2800,
        sample_rate=32000,
    )

    assert start == 2850
    assert score > 0.99
    assert merged.size == full_audio.size - start


def test_full_context_refill_cannot_jump_to_a_distant_repeated_period() -> None:
    sample_rate = 32000
    expected = 6000
    phase = np.linspace(0.0, 8.0 * np.pi, 1600, dtype=np.float32)
    repeated = np.sin(phase).astype(np.float32)
    full_audio = np.zeros(16000, dtype=np.float32)
    full_audio[expected : expected + repeated.size] = repeated
    full_audio[expected - 1600 : expected] = repeated
    previous_tail = repeated.copy()

    _, start, _ = TensorRTFixedReferenceStreamer._refill_from_full_context(
        previous_tail,
        full_audio,
        expected_start=expected,
        sample_rate=sample_rate,
    )

    assert abs(start - expected) <= int(sample_rate * 0.010)


def test_cli_source_dir_prefers_runtime_environment(monkeypatch, tmp_path) -> None:
    from aniflive_tts.cli import build_parser

    runtime_source = tmp_path / "runtime-source"
    monkeypatch.setenv("ANIFLIVE_TTS_SOURCE_DIR", str(runtime_source))

    args = build_parser().parse_args(
        ["validate", "--model-package", str(tmp_path / "model")]
    )

    assert args.source_dir == runtime_source


def test_model_registry_discovers_generic_v2proplus_packages(monkeypatch, tmp_path) -> None:
    import aniflive_tts.service as service_module

    class StubService:
        ready = False

    for model_id in ("miku-v2proplus", "roxy-v2proplus"):
        package = tmp_path / model_id
        package.mkdir()
        (package / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "aniflive-tts-model-package",
                    "model_family": "gsv-v2proplus",
                    "model_id": model_id,
                    "precision": "FP16",
                    "voice_profiles": ["default"],
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setenv("ANIFLIVE_TTS_MODEL_PACKAGE", str(tmp_path / "miku-v2proplus"))
    monkeypatch.setenv("ANIFLIVE_TTS_MODEL_REGISTRY", str(tmp_path))
    monkeypatch.setattr(service_module, "MODEL_ID", "miku-v2proplus")
    manager = service_module.RuntimeServiceManager(StubService())

    models = manager.list_models()

    assert [model["id"] for model in models] == ["miku-v2proplus", "roxy-v2proplus"]
    assert [model["active"] for model in models] == [True, False]


def test_model_switch_unloads_before_loading_replacement(monkeypatch, tmp_path) -> None:
    import aniflive_tts.service as service_module

    events: list[str] = []

    class StubService:
        def __init__(self, name: str) -> None:
            self.name = name
            self.ready = True

        def _is_busy(self) -> bool:
            return False

        def unload(self) -> None:
            events.append(f"unload:{self.name}")
            self.ready = False

    old_package = tmp_path / "old-v2proplus"
    new_package = tmp_path / "new-v2proplus"
    old = StubService("old")
    replacement = StubService("new")
    monkeypatch.setenv("ANIFLIVE_TTS_MODEL_PACKAGE", str(old_package))
    monkeypatch.setattr(service_module, "MODEL_ID", "old-v2proplus")
    monkeypatch.setattr(service_module, "VOICE_ID", "default")
    manager = service_module.RuntimeServiceManager(old)
    monkeypatch.setattr(
        manager,
        "_discover_packages",
        lambda: {
            "old-v2proplus": {"path": old_package, "manifest": {}},
            "new-v2proplus": {"path": new_package, "manifest": {}},
        },
    )

    def load_package(path: Path, voice_profile: str):
        assert events == ["unload:old"]
        assert path == new_package
        assert voice_profile == "default"
        monkeypatch.setattr(service_module, "MODEL_ID", "new-v2proplus")
        events.append("load:new")
        return replacement

    monkeypatch.setattr(manager, "_load_package", load_package)

    result = manager.activate("new-v2proplus")

    assert events == ["unload:old", "load:new"]
    assert result == {"changed": True, "model": "new-v2proplus", "voice": "default"}
    assert manager._service is replacement


def test_inspector_accepts_v2proplus_sv_emb_contract(tmp_path) -> None:
    from aniflive_tts.inspector import inspect_checkpoint

    checkpoint = tmp_path / "voice.pth"
    torch.save(
        {
            "config": {
                "version": "v2ProPlus",
                "train": {
                    "pretrained_s2G": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
                },
                "model": {"gin_channels": 1024},
            },
            "weight": {
                "enc_p.ssl_proj.weight": torch.zeros(192, 768, 1),
                "enc_p.text_embedding.weight": torch.zeros(732, 192),
                "sv_emb.bias": torch.zeros(1024),
            },
        },
        checkpoint,
    )

    result = inspect_checkpoint(checkpoint, kind="sovits")

    assert result.model_version == "gsv-v2proplus"
    assert "SoVITS tensor shapes match the V2ProPlus export contract" in result.evidence


def test_inspector_reads_official_v2proplus_tagged_header(tmp_path) -> None:
    from aniflive_tts.inspector import inspect_checkpoint

    checkpoint = tmp_path / "tagged.pth"
    torch.save(
        {
            "config": {
                "version": "v2ProPlus",
                "train": {"pretrained_s2G": "s2Gv2ProPlus.pth"},
                "model": {"gin_channels": 1024},
            },
            "weight": {
                "enc_p.ssl_proj.weight": torch.zeros(192, 768, 1),
                "enc_p.text_embedding.weight": torch.zeros(732, 192),
                "sv_emb.bias": torch.zeros(1024),
            },
        },
        checkpoint,
    )
    with checkpoint.open("r+b") as stream:
        stream.write(b"06")

    assert inspect_checkpoint(checkpoint, kind="sovits").model_version == "gsv-v2proplus"


def test_safe_punctuation_segments_preserve_semantic_tokens() -> None:
    from aniflive_tts.service import _cut_segments

    text = '價錢是3.14美元,不要切網址https://a.b/c。真的嗎?!「可以。」下一句：不要切冒號'
    assert _cut_segments(text, "") == [
        "價錢是3.14美元,",
        "不要切網址https://a.b/c。",
        "真的嗎?!",
        "「可以。」",
        "下一句：不要切冒號",
    ]


def test_safe_punctuation_does_not_split_numeric_commas_or_ellipsis() -> None:
    from aniflive_tts.service import _cut_segments

    text = "總數1,000個……仍然繼續，下一句。"
    assert _cut_segments(text, "") == ["總數1,000個……仍然繼續，", "下一句。"]


def test_bounded_segments_split_every_safe_boundary_before_hard_limit() -> None:
    from aniflive_tts.service import _bounded_segments

    text = "第一小句，第二小句！第三小句？第四小句。"
    assert _bounded_segments(text, "", 24) == [
        "第一小句，",
        "第二小句！",
        "第三小句？",
        "第四小句。",
    ]


def test_fitted_profile_keeps_typical_long_clause_in_one_inference() -> None:
    from aniflive_tts.service import PROFILE_SEGMENT_CHAR_LIMITS, _bounded_segments

    text = "これは十六文字を超える長い文章を自然につなげて再生する確認です"
    assert PROFILE_SEGMENT_CHAR_LIMITS["fitted"] == 32
    assert _bounded_segments(text, "", 32) == [text]


def test_short_natural_segments_are_merged_before_inference() -> None:
    from aniflive_tts.service import _bounded_segments, _merge_short_natural_segments

    text = "えっと、ルーデオスさん、その、ありがとうございました。"
    bounded = _bounded_segments(text, "", 32)

    assert _merge_short_natural_segments(bounded, 32) == [
        "えっと、ルーデオスさん、",
        "その、ありがとうございました。",
    ]


def test_normal_punctuation_segments_remain_independent() -> None:
    from aniflive_tts.service import _bounded_segments, _merge_short_natural_segments

    text = "これは十分に長い最初の文です、こちらも十分に長い次の文です。"
    bounded = _bounded_segments(text, "", 32)

    assert _merge_short_natural_segments(bounded, 32) == bounded


def test_short_segment_merge_never_exceeds_engine_profile() -> None:
    from aniflive_tts.service import _merge_short_natural_segments

    segments = ["えっと、", "これは既に上限近くまで長く作られた後続の推理単位です。"]

    assert _merge_short_natural_segments(segments, 24) == segments


def test_long_single_segment_uses_stable_stream_prebuffer() -> None:
    from aniflive_tts.service import _recommended_stream_prebuffer_ms

    assert _recommended_stream_prebuffer_ms(["短い文章です。"]) == 32
    assert _recommended_stream_prebuffer_ms(["これは十六文字を超える長い文章を自然につなげます"]) == 64
    assert _recommended_stream_prebuffer_ms(["前半。", "後半。"]) == 64


def test_adaptive_pause_targets_match_natural_balanced_preset() -> None:
    target = TensorRTFixedReferenceStreamer._target_pause_seconds
    assert target("前半句，", 0.440) == pytest.approx(0.220)
    assert target("分號；", 0.440) == pytest.approx(0.300)
    assert target("完整句。", 0.440) == pytest.approx(0.440)
    assert target("疑問句？", 0.440) == pytest.approx(0.400)
    assert target("技術性分割", 0.440) == pytest.approx(0.030)
    assert target("段落。\n\n", 0.440) == pytest.approx(0.640)


def test_trailing_silence_detection_uses_audio_tail() -> None:
    audio = np.concatenate(
        [np.full(1600, 0.25, dtype=np.float32), np.zeros(3200, dtype=np.float32)]
    )
    measured = TensorRTFixedReferenceStreamer._trailing_silence_seconds(audio, 32000)
    assert 0.09 <= measured <= 0.11


def test_trailing_silence_split_can_buffer_across_pcm_chunks() -> None:
    mixed = np.concatenate(
        [np.full(1600, 0.25, dtype=np.float32), np.zeros(3200, dtype=np.float32)]
    )
    active, trailing = TensorRTFixedReferenceStreamer._split_trailing_silence(
        mixed, 32000
    )
    silent_active, silent_trailing = (
        TensorRTFixedReferenceStreamer._split_trailing_silence(
            np.zeros(16000, dtype=np.float32), 32000
        )
    )
    assert active.size == 1600
    assert trailing.size == 3200
    assert silent_active.size == 0
    assert silent_trailing.size == 16000


def test_technical_boundary_detection_excludes_natural_punctuation() -> None:
    boundary = TensorRTFixedReferenceStreamer._has_natural_boundary
    assert boundary("自然な読点、") is True
    assert boundary("自然な句点。』") is True
    assert boundary("段落。\n\n") is True
    assert boundary("TensorRT上限による分割") is False


def test_technical_bridge_trims_only_excess_leading_silence() -> None:
    audio = np.concatenate(
        [np.zeros(3200, dtype=np.float32), np.full(1600, 0.25, dtype=np.float32)]
    )
    leading = TensorRTFixedReferenceStreamer._leading_silence_seconds(audio, 32000)
    trimmed, removed = TensorRTFixedReferenceStreamer._trim_excess_leading_silence(
        audio,
        sample_rate=32000,
        leading_silence_seconds=leading,
        retained_silence_seconds=0.005,
    )
    assert 0.09 <= leading <= 0.11
    assert removed == pytest.approx(0.095)
    assert trimmed.size == audio.size - 3040
    assert np.array_equal(trimmed[160:], audio[3200:])


def test_natural_segment_head_retains_configured_initial_silence() -> None:
    from aniflive_tts.streaming import INITIAL_LEADING_SILENCE_RETAIN_SECONDS

    audio = np.concatenate(
        [np.zeros(6400, dtype=np.float32), np.full(1600, 0.25, dtype=np.float32)]
    )
    trimmed, removed = TensorRTFixedReferenceStreamer._trim_excess_leading_silence(
        audio,
        sample_rate=32000,
        leading_silence_seconds=0.2,
        retained_silence_seconds=INITIAL_LEADING_SILENCE_RETAIN_SECONDS,
    )
    assert INITIAL_LEADING_SILENCE_RETAIN_SECONDS == pytest.approx(0.007)
    assert removed == pytest.approx(0.193)
    assert trimmed.size == audio.size - 6176
    assert np.array_equal(trimmed[224:], audio[6400:])


def test_initial_preview_gate_retains_enough_audio_for_continuous_playback() -> None:
    from aniflive_tts.streaming import MIN_INITIAL_PREVIEW_PUBLISHED_SECONDS

    assert MIN_INITIAL_PREVIEW_PUBLISHED_SECONDS == pytest.approx(0.112)
    minimum_samples = int(round(32000 * MIN_INITIAL_PREVIEW_PUBLISHED_SECONDS))
    assert 3200 < minimum_samples
    assert 6400 >= minimum_samples


def test_active_audio_gate_rejects_an_entirely_silent_preview() -> None:
    silent = np.zeros(4096, dtype=np.float32)
    audible = silent.copy()
    audible[2048:] = 0.02

    assert TensorRTFixedReferenceStreamer._contains_active_audio(silent, 32000) is False
    assert TensorRTFixedReferenceStreamer._contains_active_audio(audible, 32000) is True


def test_segment_tail_trims_only_silence_beyond_natural_pause() -> None:
    audio = np.concatenate(
        [np.full(1600, 0.25, dtype=np.float32), np.zeros(6400, dtype=np.float32)]
    )
    trimmed, retained, removed = (
        TensorRTFixedReferenceStreamer._trim_excess_trailing_silence(
            audio,
            sample_rate=32000,
            trailing_silence_seconds=0.2,
            target_pause_seconds=0.08,
        )
    )
    assert trimmed.size == 1600 + 2560
    assert retained == pytest.approx(0.08)
    assert removed == pytest.approx(0.12)
    assert np.array_equal(trimmed[:1600], audio[:1600])


class _FakeTelemetry:
    def __init__(self, sample: GpuTelemetrySample | None) -> None:
        self.current = sample
        self.available = sample is not None

    def sample(self) -> GpuTelemetrySample | None:
        return self.current

    def close(self) -> None:
        return None


def test_warm_retention_runs_only_inside_safety_gate() -> None:
    calls: list[int] = []
    telemetry = _FakeTelemetry(GpuTelemetrySample(50, 0, 0, 20.0))
    controller = WarmRetentionController(
        pulse=lambda: calls.append(1) or 0.003,
        inference_lock=threading.RLock(),
        is_busy=lambda: False,
        telemetry=telemetry,  # type: ignore[arg-type]
    )
    controller._attempt_pulse()
    assert calls == [1]
    assert controller.status()["state"] == "hot"

    telemetry.current = GpuTelemetrySample(70, 0, 0, 20.0)
    controller._attempt_pulse()
    assert calls == [1]
    assert controller.status()["state"] == "thermal_suspended"


def test_warm_retention_never_competes_with_real_request() -> None:
    calls: list[int] = []
    controller = WarmRetentionController(
        pulse=lambda: calls.append(1) or 0.001,
        inference_lock=threading.RLock(),
        is_busy=lambda: True,
        telemetry=_FakeTelemetry(GpuTelemetrySample(40, 0, 0)),  # type: ignore[arg-type]
    )
    controller._attempt_pulse()
    assert calls == []
    assert controller.status()["state"] == "request_busy"
