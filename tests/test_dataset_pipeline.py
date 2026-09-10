from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import subprocess
import wave

import numpy as np
import pytest

from aniflive_tts.dataset_factory import EnergyVadConfig
from aniflive_tts.dataset_pipeline import (
    DATASET_PIPELINE_SCHEMA,
    FASTTEXT_LID176_SHA256,
    DatasetPipelineConfig,
    DatasetPipelineError,
    assert_linux_docker_runtime,
    build_transcript_record,
    detect_dataset_language,
    energy_speech_regions,
    is_linux_docker_runtime,
    load_gpt_sovits_normalizer,
    normalize_dataset_transcript,
    parse_ffprobe_document,
    resolve_fasttext_model,
    run_dataset_pipeline,
    sha256_file,
)


class FakeDetector:
    def __init__(self, language: str, score: float) -> None:
        self.language = language
        self.score = score
        self.calls: list[tuple[str, bool]] = []

    def detect(self, text: str, *, low_memory: bool) -> dict[str, object]:
        self.calls.append((text, low_memory))
        return {"lang": self.language, "score": self.score}


def _probe_document(
    *,
    duration: object = "2.000000",
    sample_rate: object = "44100",
    channels: object = 2,
    stream_index: object = 1,
) -> dict[str, object]:
    return {
        "streams": [
            {
                "index": stream_index,
                "codec_name": "aac",
                "codec_type": "audio",
                "sample_rate": sample_rate,
                "channels": channels,
                "duration": duration,
            }
        ],
        "format": {"duration": "2.100000", "format_name": "mov,mp4"},
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_rate", 24_000),
        ("max_input_bytes", 0),
        ("max_duration_seconds", math.inf),
        ("max_duration_seconds", 3600.01),
        ("max_samples", 0),
        ("max_samples", 32_000 * 3_600 + 1),
        ("max_output_bytes", 44),
        ("max_output_bytes", 512 * 1024**2 + 1),
        ("max_artifact_bytes", 1024),
        ("max_segments", 0),
        ("ffprobe_timeout_seconds", 0),
        ("ffmpeg_timeout_seconds", 0),
        ("afftdn_noise_reduction_db", 12.01),
        ("afftdn_noise_floor_db", -81.0),
        ("afftdn_gain_smooth", 51),
        ("language_confidence_threshold", 0.0),
    ],
)
def test_pipeline_config_rejects_unsafe_values(field: str, value: object) -> None:
    with pytest.raises(DatasetPipelineError):
        replace(DatasetPipelineConfig(), **{field: value}).validated()


def test_linux_docker_runtime_guard_is_explicit(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assert not is_linux_docker_runtime(
        platform_name="win32",
        environ={"ANIFLIVE_TTS_WORKER_CONTAINER": "1"},
        marker_paths=[],
        cgroup_text="docker",
    )
    assert not is_linux_docker_runtime(
        platform_name="linux",
        environ={"ANIFLIVE_TTS_WORKER_CONTAINER": "1"},
        marker_paths=[missing],
        cgroup_text="0::/",
    )
    with pytest.raises(DatasetPipelineError, match="Linux Docker"):
        assert_linux_docker_runtime(
            platform_name="linux",
            environ={},
            marker_paths=[missing],
            cgroup_text="0::/",
        )
    marker = tmp_path / ".dockerenv"
    marker.write_text("", encoding="ascii")
    assert is_linux_docker_runtime(
        platform_name="linux",
        environ={},
        marker_paths=[marker],
        cgroup_text="",
    )
    assert_linux_docker_runtime(
        platform_name="linux",
        environ={},
        marker_paths=[marker],
        cgroup_text="",
    )


def test_parse_ffprobe_document_accepts_only_bounded_first_audio() -> None:
    probe = parse_ffprobe_document(_probe_document())
    assert probe.audio_stream_index == 1
    assert probe.source_sample_rate == 44_100
    assert probe.source_channels == 2
    assert probe.duration_seconds == 2.0
    assert probe.estimated_output_samples == 64_000
    assert probe.estimated_output_bytes == 128_044

    invalid_documents = [
        {"streams": [], "format": {"duration": 1}},
        {"streams": [{**_probe_document()["streams"][0], "codec_type": "video"}]},
        {
            "streams": [
                _probe_document()["streams"][0],
                _probe_document()["streams"][0],
            ]
        },
        {
            **_probe_document(duration="N/A"),
            "format": {"duration": "N/A", "format_name": "mov,mp4"},
        },
        _probe_document(sample_rate="zero"),
        _probe_document(sample_rate=44_100.5),
        _probe_document(channels=0),
        _probe_document(stream_index=-1),
        _probe_document(duration=[]),
    ]
    for document in invalid_documents:
        with pytest.raises(DatasetPipelineError):
            parse_ffprobe_document(document)


def test_parse_ffprobe_document_enforces_all_output_caps() -> None:
    document = _probe_document(duration="2.0")
    with pytest.raises(DatasetPipelineError, match="duration"):
        parse_ffprobe_document(
            document,
            replace(DatasetPipelineConfig(), max_duration_seconds=1.0),
        )
    with pytest.raises(DatasetPipelineError, match="max_samples"):
        parse_ffprobe_document(
            document,
            replace(DatasetPipelineConfig(), max_samples=63_999),
        )
    with pytest.raises(DatasetPipelineError, match="max_output_bytes"):
        parse_ffprobe_document(
            document,
            replace(DatasetPipelineConfig(), max_output_bytes=128_043),
        )


@pytest.mark.parametrize(
    ("text", "raw_language", "expected", "review_required"),
    [
        ("A clean English sentence.", "en", "en", False),
        ("今日は良い天気です。", "ja", "ja", False),
        ("오늘은 날씨가 좋습니다.", "ko", "ko", False),
        ("今天的天气很好。", "zh", "zh", True),
    ],
)
def test_fasttext_language_evidence_is_truthful(
    text: str,
    raw_language: str,
    expected: str,
    review_required: bool,
) -> None:
    detector = FakeDetector(raw_language, 0.97)
    result = detect_dataset_language(text, detector=detector)
    assert result["model"] == "injected-language-detector"
    assert result["model_sha256"] is None
    assert result["provider_verified"] is False
    assert result["model_language"] == expected
    assert result["suggested_language"] == expected
    assert result["review_required"] is review_required
    assert detector.calls == [(text, True)]


def test_cantonese_is_only_a_review_required_diagnostic() -> None:
    text = "我唔係話佢冇嚟，只係想問下啫。"
    result = detect_dataset_language(text, detector=FakeDetector("zh", 0.99))
    assert result["model_language"] == "zh"
    assert result["suggested_language"] == "yue"
    assert result["review_required"] is True
    assert "zh-yue-requires-human-review" in result["review_reasons"]
    assert result["cantonese_diagnostic"]["classifier"] is False
    assert len(result["cantonese_diagnostic"]["marker_hits"]) >= 2


def test_low_confidence_and_unknown_labels_require_review() -> None:
    low = detect_dataset_language(
        "A sentence.",
        detector=FakeDetector("en", 0.5),
        confidence_threshold=0.8,
    )
    assert low["review_required"] is True
    assert "fasttext-confidence-below-threshold" in low["review_reasons"]
    unknown = detect_dataset_language(
        "Une phrase.",
        detector=FakeDetector("fr", 0.99),
    )
    assert unknown["model_language"] == "und"
    assert unknown["review_required"] is True


@pytest.mark.parametrize("language", ["yue", "zh", "ja", "en", "ko"])
def test_normalization_preserves_original_and_normalized_text(language: str) -> None:
    calls: list[tuple[str, str]] = []

    def normalizer(text: str, lang: str) -> str:
        calls.append((text, lang))
        return f"{text} [{lang}]"

    original = "  Hello,\nworld!  "
    record = normalize_dataset_transcript(
        original,
        language,
        normalizer=normalizer,
    )
    assert record["original_text"] == original
    assert record["canonical_input_text"] == "Hello, world!"
    assert record["normalized_text"] == f"Hello, world! [{language}]"
    assert record["normalizer"]["name"] == "injected-normalizer"
    assert record["normalizer"]["provider_verified"] is False
    assert calls == [("Hello, world!", language)]


def test_declared_language_mismatch_is_never_silently_accepted() -> None:
    record = build_transcript_record(
        "An English sentence.",
        declared_language="ja",
        detector=FakeDetector("en", 0.99),
        normalizer=lambda text, _language: text,
    )
    assert record["language"] == "ja"
    assert record["review_required"] is True
    assert "declared-language-disagrees-with-fasttext" in record["review_reasons"]


def test_energy_vad_reuses_factory_segmentation_contract() -> None:
    sample_rate = 32_000
    silence = np.zeros(round(sample_rate * 0.25), dtype=np.float32)
    time_axis = np.arange(round(sample_rate * 0.5), dtype=np.float32) / sample_rate
    tone = 0.2 * np.sin(2 * np.pi * 440 * time_axis)
    samples = np.concatenate((silence, tone, silence))
    regions = energy_speech_regions(
        samples,
        sample_rate,
        EnergyVadConfig(
            threshold_dbfs=-45,
            min_speech_ms=100,
            min_silence_ms=100,
            pad_ms=20,
        ),
    )
    assert len(regions) == 1
    assert regions[0].start_frame < silence.size
    assert regions[0].end_frame > silence.size + tone.size


_LINUX_MEDIA_RUNTIME = (
    is_linux_docker_runtime()
    and shutil.which("ffmpeg") is not None
    and shutil.which("ffprobe") is not None
)


def _run_ffmpeg(arguments: list[str]) -> None:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr.decode("utf-8", errors="replace"))


@pytest.mark.skipif(
    not _LINUX_MEDIA_RUNTIME,
    reason="real media execution is intentionally Linux-Docker-only",
)
def test_linux_pipeline_decodes_first_audio_stream_deterministically(
    tmp_path: Path,
) -> None:
    source = tmp_path / "two-audio-streams.mkv"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=1.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=48000:duration=1.2",
            "-map",
            "0:a:0",
            "-map",
            "1:a:0",
            "-c:a",
            "flac",
            "-map_metadata",
            "-1",
            "-y",
            str(source),
        ]
    )
    config = replace(
        DatasetPipelineConfig(),
        max_duration_seconds=3.0,
        max_samples=96_000,
        max_output_bytes=300_000,
        ffprobe_timeout_seconds=10,
        ffmpeg_timeout_seconds=30,
        vad=EnergyVadConfig(
            threshold_dbfs=-50,
            min_speech_ms=100,
            min_silence_ms=100,
            pad_ms=10,
        ),
    )
    detector = FakeDetector("en", 0.99)
    first_output = tmp_path / "prepared-a"
    second_output = tmp_path / "prepared-b"
    first = run_dataset_pipeline(
        source,
        first_output,
        transcript="  First audio stream.\n",
        declared_language="en",
        config=config,
        detector=detector,
        normalizer=lambda text, _language: text,
    )
    second = run_dataset_pipeline(
        source,
        second_output,
        transcript="  First audio stream.\n",
        declared_language="en",
        config=config,
        detector=detector,
        normalizer=lambda text, _language: text,
    )
    assert first["schema"] == DATASET_PIPELINE_SCHEMA
    assert first["pipeline_identity_sha256"] == second["pipeline_identity_sha256"]
    assert (
        first["output"]["canonical"]["sha256"]
        == second["output"]["canonical"]["sha256"]
    )
    assert first["transcript"]["original_text"] == "  First audio stream.\n"
    assert first["transcript"]["canonical_input_text"] == "First audio stream."
    assert first["transcript"]["normalized_text"] == "First audio stream."
    assert first["input"]["probe"]["source_channels"] == 1
    assert first["output"]["canonical"]["sample_rate"] == 32_000
    assert first["output"]["canonical"]["channels"] == 1
    assert first["output"]["canonical"]["sample_width_bytes"] == 2
    assert len(first["output"]["segments"]) == 1
    canonical = first_output / "canonical.wav"
    with wave.open(str(canonical), "rb") as audio:
        pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    spectrum = np.abs(np.fft.rfft(pcm.astype(np.float32)))
    frequencies = np.fft.rfftfreq(pcm.size, 1 / 32_000)
    peak_hz = float(frequencies[int(np.argmax(spectrum))])
    assert peak_hz == pytest.approx(440.0, abs=2.0)
    stored = json.loads(
        (first_output / "dataset-report.json").read_text(encoding="utf-8")
    )
    assert stored == first
    assert str(tmp_path) not in json.dumps(stored, ensure_ascii=False)
    assert sha256_file(canonical) == first["output"]["canonical"]["sha256"]
    constrained = replace(
        config,
        max_output_bytes=100_000,
        max_artifact_bytes=110_000,
    )
    with pytest.raises(DatasetPipelineError, match="max_artifact_bytes"):
        run_dataset_pipeline(
            source,
            tmp_path / "artifact-budget-rejected",
            config=constrained,
        )
    assert not (tmp_path / "artifact-budget-rejected").exists()


@pytest.mark.skipif(
    not _LINUX_MEDIA_RUNTIME,
    reason="real media execution is intentionally Linux-Docker-only",
)
def test_linux_pipeline_optional_afftdn_is_real_and_reported(tmp_path: Path) -> None:
    source = tmp_path / "noisy.wav"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=white:amplitude=0.01:sample_rate=32000:duration=0.8",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:sample_rate=32000:duration=0.8",
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:normalize=0",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(source),
        ]
    )
    config = replace(
        DatasetPipelineConfig(),
        enable_afftdn=True,
        max_duration_seconds=2.0,
        max_samples=64_000,
        max_output_bytes=200_000,
        ffmpeg_timeout_seconds=30,
    )
    report = run_dataset_pipeline(source, tmp_path / "denoised", config=config)
    denoise = report["output"]["canonical"]["denoise"]
    assert denoise["enabled"] is True
    assert denoise["backend"] == "ffmpeg-afftdn"
    assert denoise["filter"].startswith("afftdn=")
    assert report["output"]["canonical"]["frame_count"] <= config.max_samples


@pytest.mark.skipif(
    not _LINUX_MEDIA_RUNTIME,
    reason="bundled assets are qualified only in the Linux Docker worker",
)
def test_linux_worker_uses_pinned_fasttext_model_without_download() -> None:
    model = resolve_fasttext_model()
    assert sha256_file(model) == FASTTEXT_LID176_SHA256
    result = detect_dataset_language("This is an English sentence.")
    assert result["model"] == "fastText-lid.176.bin"
    assert result["model_sha256"] == FASTTEXT_LID176_SHA256
    assert result["provider_verified"] is True
    assert result["model_language"] == "en"
    assert result["model_score"] > 0.5


@pytest.mark.skipif(
    not _LINUX_MEDIA_RUNTIME,
    reason="GPT-SoVITS normalizers are qualified only in the Linux Docker worker",
)
@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("yue", "我今日真係好開心。"),
        ("zh", "今天天气很好。"),
        ("ja", "今日は良い天気です。"),
        ("en", "Today is a good day."),
        ("ko", "오늘은 좋은 날입니다."),
    ],
)
def test_linux_worker_has_all_five_gpt_sovits_normalizers(
    language: str,
    text: str,
) -> None:
    assert callable(load_gpt_sovits_normalizer())
    record = normalize_dataset_transcript(text, language)
    assert record["language"] == language
    assert record["original_text"] == text
    assert record["normalized_text"]
    assert record["normalizer"]["name"] == "gpt-sovits-v2-five-language"
    assert record["normalizer"]["provider_verified"] is True
    assert record["normalizer"]["source_sha256"]


@pytest.mark.parametrize("preloaded", [False, True])
def test_normalizer_keeps_preferred_root_when_already_on_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preloaded: bool
) -> None:
    import importlib
    import sys
    from aniflive_tts import dataset_pipeline

    preferred, alternate = tmp_path / "preferred", tmp_path / "alternate"
    for root, label in ((preferred, "preferred"), (alternate, "alternate")):
        package = root / "text"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "cleaner.py").write_text(
            "def clean_text(value, language, version):\n"
            f"    return [], [], {label!r}\n", encoding="utf-8",
        )
    saved_modules = {
        name: module for name, module in sys.modules.items()
        if name == "text" or name.startswith("text.")
    }
    monkeypatch.setattr(sys, "path", [str(preferred), *sys.path])
    monkeypatch.setattr(dataset_pipeline, "_normalizer_roots", lambda: [preferred, alternate])
    dataset_pipeline.load_gpt_sovits_normalizer.cache_clear()
    try:
        for name in saved_modules:
            sys.modules.pop(name, None)
        if preloaded:
            sys.path.insert(0, str(alternate))
            other = importlib.import_module("text.cleaner")
            assert other.clean_text("", "ja", "v2")[2] == "alternate"
        normalize = dataset_pipeline.load_gpt_sovits_normalizer()
        assert normalize("input", "ja") == "preferred"
        assert normalize.__aniflive_provenance__["source_sha256"] == sha256_file(
            preferred / "text" / "cleaner.py"
        )
    finally:
        dataset_pipeline.load_gpt_sovits_normalizer.cache_clear()
        for name in tuple(sys.modules):
            if name == "text" or name.startswith("text."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
