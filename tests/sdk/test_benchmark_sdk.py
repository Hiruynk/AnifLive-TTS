from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import pytest

from scripts import benchmark_sdk as benchmark


def test_pairs_are_matched_alternating_and_warmups_excluded():
    calls = []
    request = {"text": "private text", "language": "ja", "generation": benchmark.GENERATION}
    raw, sdk = object(), object()

    def measured(client, body, **kwargs):
        calls.append((client, body, kwargs["streaming"]))
        assert body is request
        return {"wall_seconds": 1.2 if client is sdk else 1.0}

    report = benchmark.run_pairs(raw, sdk, request, pairs=2, warmups=1, measure_fn=measured)
    assert len(calls) == 12
    assert len(report["samples"]) == 8
    assert [(c[0], c[2]) for c in calls[4:]] == [
        (raw, False), (sdk, False), (raw, True), (sdk, True),
        (sdk, True), (raw, True), (sdk, False), (raw, False),
    ]
    assert report["summary"]["complete"]["wall_seconds"]["sdk_minus_raw"]["p50"]["percent"] == pytest.approx(20)
    assert report["connection_reuse"]["verified"] is False
    assert "private text" not in str(report)


def test_percentiles_and_investigation_are_not_a_qualification_pass():
    assert benchmark.summary(list(range(1, 21)))["p95"] == 19
    rows = [{"mode": "stream", "client": name,
             "measurement": {"first_packet_seconds": value, "audible_ttfa_seconds": value}}
            for name, value in (("raw", 1.0), ("sdk", 1.11))]
    _, flags = benchmark.aggregate(rows)
    assert len(flags) == 4
    assert all(flag["action"] == "investigate" for flag in flags)
    with pytest.raises(ValueError):
        benchmark.summary([float("nan")])


class Stream:
    sample_rate = 1000
    channels = 1
    sample_format = "s16le"

    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        yield from self.chunks

    def close(self):
        self.closed = True


def test_stream_timing_reuses_audible_algorithm_across_odd_chunks():
    # 10 ms silence followed by 10 ms audible PCM, with a sample split across chunks.
    pcm = b"\x00\x00" * 10 + b"\xff\x3f" * 10
    stream = Stream([pcm[:21], b"", pcm[21:]])
    clock = iter([0.0, 0.1, 0.2, 0.4, 0.5])
    result = benchmark.measure(SimpleNamespace(stream=lambda **_: stream), {},
                               streaming=True, clock=lambda: next(clock))
    assert result["first_packet_seconds"] == 0.2
    assert result["audible_ttfa_seconds"] == pytest.approx(0.4)
    assert result["steady_bytes_per_second"] == pytest.approx(19 / 0.2)
    assert result["audio_duration_seconds"] == 0.02
    assert stream.closed


@pytest.mark.parametrize("chunks,limit", [([b"\x00\x00" * 10], 100), ([b"x" * 101], 100)])
def test_stream_closes_on_silence_or_memory_limit_failure(chunks, limit):
    stream = Stream(chunks)
    with pytest.raises((RuntimeError, ValueError)):
        benchmark.measure(SimpleNamespace(stream=lambda **_: stream), {},
                          streaming=True, max_audio_bytes=limit)
    assert stream.closed


def test_complete_wav_is_measured_without_persisting_audio():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(b"\x00\x01" * 10)
    result = benchmark.measure(
        SimpleNamespace(synthesize=lambda **_: SimpleNamespace(audio=buffer.getvalue())),
        {}, streaming=False,
    )
    assert result["audio_duration_seconds"] == 0.01
    assert "audio" not in result


@pytest.mark.parametrize("url", ["http://user:secret@localhost", "http://localhost/?token=x",
                                 "file:///tmp/test", "https://localhost/#secret"])
def test_url_validation_does_not_expose_secrets(url):
    with pytest.raises(ValueError) as error:
        benchmark.validate_url(url)
    assert "secret" not in str(error.value)
    assert url not in str(error.value)


def test_raw_and_sdk_adapters_send_matching_request_values(monkeypatch):
    import json
    import sys

    import httpx

    sent = []
    real_client = httpx.Client

    def respond(request):
        sent.append(json.loads(request.content))
        if sent[-1]["stream"]:
            return httpx.Response(200, headers={
                "content-type": "application/octet-stream", "x-tts-sample-rate": "1000",
                "x-tts-sample-format": "s16le", "x-tts-channels": "1",
            }, content=b"\xff\x3f" * 10)
        return httpx.Response(200, headers={"content-type": "audio/wav"}, content=b"WAV")

    monkeypatch.setattr(httpx, "Client",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    sdk_calls = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def synthesize(self, **kwargs):
            sdk_calls.append(kwargs)

        stream = synthesize

        def close(self):
            pass

    class Generation:
        def __init__(self, **kwargs):
            self.values = kwargs

    monkeypatch.setitem(sys.modules, "aniflive_tts",
                        SimpleNamespace(Client=Client, Generation=Generation))
    request = {"text": benchmark.DEFAULT_TEXT, "language": "ja",
               "generation": dict(benchmark.GENERATION), "model": "fixture",
               "voice_profile": "default"}
    raw = benchmark.RawHTTP("http://localhost/prefix/", timeout=10, max_audio_bytes=100)
    sdk = benchmark.SDKAdapter("http://localhost/prefix/", timeout=10)
    try:
        raw.synthesize(**request)
        stream = raw.stream(**request)
        assert b"".join(stream) == b"\xff\x3f" * 10
        stream.close()
        sdk.synthesize(**request)
        sdk.stream(**request)
    finally:
        raw.close()
        sdk.close()
    for raw_body, sdk_body in zip(sent, sdk_calls):
        assert raw_body.pop("response_format") in {"wav", "pcm"}
        raw_body.pop("stream")
        sdk_body["generation"] = sdk_body["generation"].values
        assert raw_body == sdk_body == request


def test_cli_failure_redacts_adapter_exception_and_writes_no_report(tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise RuntimeError("Authorization secret and private synthesis text")

    monkeypatch.setattr(benchmark, "RawHTTP", fail)
    output = tmp_path / "report.json"
    assert benchmark.main(["--output", str(output)]) == 1
    assert not output.exists()
    assert "secret" not in capsys.readouterr().out
