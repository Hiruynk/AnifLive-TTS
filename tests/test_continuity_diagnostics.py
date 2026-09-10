from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import wave

import pytest

from aniflive_tts.continuity_diagnostics import (
    PCM16_FULL_SCALE,
    ContinuityDiagnosticError,
    evaluate_ordered_segments,
    load_audio_segment,
)


def _write_wav(
    path: Path,
    samples: list[int],
    *,
    sample_rate: int = 1000,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(sample_width)
        stream.setframerate(sample_rate)
        if sample_width == 2:
            payload = struct.pack(f"<{len(samples)}h", *samples)
        else:
            payload = bytes(len(samples) * sample_width)
        stream.writeframes(payload)


def test_ordered_boundaries_emit_exact_metrics_and_aggregates(tmp_path: Path) -> None:
    first, second, third = (tmp_path / name for name in ("first.wav", "second.wav", "third.wav"))
    _write_wav(first, [1000, 1000])
    _write_wav(second, [-1000, -1000])
    _write_wav(third, [0, 0])

    report = evaluate_ordered_segments(
        [first, second, third],
        expected_sample_rate=1000,
        window_ms=2.0,
        silence_threshold_dbfs=-60.0,
    )

    assert report["diagnostic_only"] is True
    assert report["qualification"] is False
    assert report["configuration"]["channels"] == 1
    assert report["configuration"]["pcm_format"] == "signed-pcm16-little-endian"
    assert report["aggregate"]["segment_count"] == 3
    assert report["aggregate"]["boundary_count"] == 2
    assert report["aggregate"]["total_duration_seconds"] == pytest.approx(0.006)
    assert report["boundaries"][0]["sample_discontinuity_raw"] == 2000
    assert report["boundaries"][0]["sample_discontinuity_normalized"] == pytest.approx(
        2000 / PCM16_FULL_SCALE
    )
    assert report["boundaries"][0]["rms_jump"] == pytest.approx(0.0)
    assert report["boundaries"][0]["dc_jump"] == pytest.approx(2000 / PCM16_FULL_SCALE)
    assert report["boundaries"][1]["pause_duration_seconds"] == pytest.approx(0.002)
    discontinuity = report["aggregate"]["boundary_metrics"][
        "sample_discontinuity_normalized"
    ]
    assert discontinuity["p50"] == pytest.approx(1500 / PCM16_FULL_SCALE)
    assert discontinuity["p95"] == pytest.approx(1950 / PCM16_FULL_SCALE)
    assert discontinuity["max"] == pytest.approx(2000 / PCM16_FULL_SCALE)
    assert set(report["aggregate"]["segment_duration_seconds"]) == {
        "unit", "count", "p50", "p95", "max"
    }
    json.dumps(report, allow_nan=False)


def test_pause_only_counts_consecutive_samples_at_the_boundary(tmp_path: Path) -> None:
    left, right = tmp_path / "left.wav", tmp_path / "right.wav"
    _write_wav(left, [0, 2000, 0, 0, 0], sample_rate=10_000)
    _write_wav(right, [0, 0, -2000, 0], sample_rate=10_000)
    report = evaluate_ordered_segments(
        [left, right], window_ms=1.0, silence_threshold_dbfs=-60.0
    )
    boundary = report["boundaries"][0]
    assert boundary["trailing_pause_seconds"] == pytest.approx(0.0003)
    assert boundary["leading_pause_seconds"] == pytest.approx(0.0002)
    assert boundary["pause_duration_seconds"] == pytest.approx(0.0005)


def test_wav_contract_rejects_rate_channels_width_empty_and_malformed(tmp_path: Path) -> None:
    valid = tmp_path / "valid.wav"
    other_rate = tmp_path / "other-rate.wav"
    stereo = tmp_path / "stereo.wav"
    pcm24 = tmp_path / "pcm24.wav"
    empty = tmp_path / "empty.wav"
    malformed = tmp_path / "malformed.wav"
    _write_wav(valid, [1, 2], sample_rate=32_000)
    _write_wav(other_rate, [1, 2], sample_rate=24_000)
    _write_wav(stereo, [1, 2, 3, 4], sample_rate=32_000, channels=2)
    _write_wav(pcm24, [0, 0], sample_rate=32_000, sample_width=3)
    _write_wav(empty, [], sample_rate=32_000)
    malformed.write_bytes(b"RIFF\x00\x00truncated")

    with pytest.raises(ContinuityDiagnosticError, match="same sample rate"):
        evaluate_ordered_segments([valid, other_rate])
    with pytest.raises(ContinuityDiagnosticError, match="mono"):
        load_audio_segment(stereo)
    with pytest.raises(ContinuityDiagnosticError, match="16-bit"):
        load_audio_segment(pcm24)
    with pytest.raises(ContinuityDiagnosticError, match="empty"):
        load_audio_segment(empty)
    with pytest.raises(ContinuityDiagnosticError, match="malformed or unreadable"):
        load_audio_segment(malformed)


def test_raw_pcm16_requires_rate_and_complete_samples(tmp_path: Path) -> None:
    first, second, odd = tmp_path / "first.pcm", tmp_path / "second.raw", tmp_path / "odd.pcm"
    first.write_bytes(struct.pack("<3h", 0, 100, 200))
    second.write_bytes(struct.pack("<2h", -200, 0))
    odd.write_bytes(b"\x00")
    with pytest.raises(ContinuityDiagnosticError, match="raw_sample_rate"):
        load_audio_segment(first)
    with pytest.raises(ContinuityDiagnosticError, match="incomplete sample"):
        load_audio_segment(odd, raw_sample_rate=32_000)
    report = evaluate_ordered_segments(
        [first, second], raw_sample_rate=32_000, expected_sample_rate=32_000
    )
    assert report["segments"][0]["input_format"] == "pcm16le"
    assert report["configuration"]["sample_rate_hz"] == 32_000


def test_diagnostic_limits_and_format_inference_are_enforced(tmp_path: Path) -> None:
    first, second, unknown = tmp_path / "first.wav", tmp_path / "second.wav", tmp_path / "audio.bin"
    _write_wav(first, [1, 2])
    _write_wav(second, [3, 4])
    unknown.write_bytes(b"\x00\x00")
    with pytest.raises(ContinuityDiagnosticError, match="At least two"):
        evaluate_ordered_segments([first])
    with pytest.raises(ContinuityDiagnosticError, match="configured limit"):
        evaluate_ordered_segments([first, second, first], max_segments=2)
    with pytest.raises(ContinuityDiagnosticError, match="total limit"):
        evaluate_ordered_segments([first, second], max_total_bytes=2)
    with pytest.raises(ContinuityDiagnosticError, match="exceeds"):
        load_audio_segment(first, max_file_bytes=1)
    with pytest.raises(ContinuityDiagnosticError, match="Cannot infer"):
        load_audio_segment(unknown, raw_sample_rate=1000)
    with pytest.raises(ContinuityDiagnosticError, match="ordered sequence"):
        evaluate_ordered_segments(str(first))  # type: ignore[arg-type]


def test_cli_writes_machine_readable_report_and_machine_readable_error(tmp_path: Path) -> None:
    first, second = tmp_path / "first.wav", tmp_path / "second.wav"
    report_path = tmp_path / "report.json"
    _write_wav(first, [1, 2], sample_rate=32_000)
    _write_wav(second, [3, 4], sample_rate=32_000)
    repository = Path(__file__).resolve().parents[1]
    script = repository / "scripts" / "evaluate_long_form_continuity.py"
    environment = os.environ.copy()
    source_path = str(repository / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(first),
            str(second),
            "--expected-sample-rate",
            "32000",
            "--output",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    standard_output = json.loads(completed.stdout)
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved == standard_output
    assert saved["diagnostic_only"] is True

    failed = subprocess.run(
        [sys.executable, str(script), str(first)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert failed.returncode == 2
    error = json.loads(failed.stderr)
    assert error["diagnostic_only"] is True
    assert "At least two" in error["error"]
