from types import SimpleNamespace

import pytest

from aniflive_tts import service


@pytest.mark.parametrize("language", ["zh", "yue", "en", "ja", "ko"])
def test_strict_warmup_uses_fixed_language_probe_with_real_reference(language, monkeypatch):
    monkeypatch.setattr(service, "REFERENCE_TEXT", "private reference transcript")
    monkeypatch.setattr(service, "REFERENCE_LANGUAGE", language)
    requests = []

    def synthesize(options):
        requests.append(options)
        return SimpleNamespace(sample_rate=32000, output_samples=3200, elapsed_seconds=0.01)

    def stream(options):
        requests.append(options)
        return iter([b"pcm"])

    runtime = SimpleNamespace(synthesize=synthesize, stream_pcm=stream)
    service.TensorRTService._run_strict_warmup(runtime)
    assert len(requests) == 2
    assert all(item.text == service.STREAM_CALIBRATION_TEXT[language] for item in requests)
    assert all(item.text_language == language and item.seed == 1234 for item in requests)
    assert service.REFERENCE_TEXT == "private reference transcript"
    assert runtime._warmup["completed"] is True


def test_strict_warmup_still_rejects_empty_stream(monkeypatch):
    monkeypatch.setattr(service, "REFERENCE_LANGUAGE", "ja")
    runtime = SimpleNamespace(
        synthesize=lambda options: SimpleNamespace(sample_rate=32000, output_samples=3200, elapsed_seconds=0.01),
        stream_pcm=lambda options: iter([]),
    )
    with pytest.raises(service.TensorRTRuntimeError, match="no PCM"):
        service.TensorRTService._run_strict_warmup(runtime)
