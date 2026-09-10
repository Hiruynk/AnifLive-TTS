from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aniflive_tts import workstation_diarization as diar


def test_rttm_parser_preserves_true_overlap_and_merges_same_speaker() -> None:
    segments = diar.parse_rttm(
        "\n".join(
            (
                "SPEAKER source 1 0.000 1.000 <NA> <NA> speaker_0 <NA> <NA>",
                "SPEAKER source 1 0.800 0.800 <NA> <NA> speaker_1 <NA> <NA>",
                "SPEAKER source 1 1.000 0.500 <NA> <NA> speaker_0 <NA> <NA>",
            )
        ),
        sample_rate=16_000,
        maximum_samples=32_000,
    )

    assert [(item.speaker, item.start_sample, item.end_sample) for item in segments] == [
        ("speaker_0", 0, 24_000),
        ("speaker_1", 12_800, 25_600),
    ]
    evidence = diar.interval_diarization_evidence(
        segments,
        start_sample=0,
        end_sample=32_000,
        sample_rate=16_000,
    )
    assert evidence["speaker_count"] == 2
    assert evidence["speaker_change_evidence"] is True
    assert evidence["overlap_evidence"] is True
    assert evidence["overlap_samples"] == 11_200
    assert evidence["decision"] == "overlap"


def test_short_diarization_blip_does_not_trigger_contamination() -> None:
    segments = (
        diar.DiarizationSegment("speaker_0", 0, 32_000),
        diar.DiarizationSegment("speaker_1", 8_000, 9_000),
    )

    evidence = diar.interval_diarization_evidence(
        segments,
        start_sample=0,
        end_sample=32_000,
        sample_rate=16_000,
    )

    assert evidence["speakers"] == ["speaker_0"]
    assert evidence["speaker_change_evidence"] is False
    assert evidence["overlap_evidence"] is False
    assert evidence["decision"] == "single-speaker"


def test_pinned_postprocess_contract_is_machine_readable() -> None:
    profile = diar.DiarizationPostprocess()

    assert profile.as_dict() == {
        "schema": "aniflive-sortformer-postprocess-v1",
        "onset": 0.4,
        "offset": 0.7,
        "pad_onset_seconds": 0.05,
        "pad_offset_seconds": 0.0,
        "minimum_on_seconds": 0.2,
        "minimum_off_seconds": 0.2,
    }
    with pytest.raises(diar.WorkstationDiarizationError, match="between zero"):
        diar.DiarizationPostprocess(onset=1.1)


def test_native_diarizer_uses_fixed_local_model_and_rttm_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    component = tmp_path / "component"
    component.mkdir()
    model = component / diar.SORTFORMER_MODEL_FILENAME
    model.write_bytes(b"model")
    executable = tmp_path / "nemo-speech"
    executable.write_bytes(b"runtime")
    calls: list[tuple[list[str], Path, float]] = []

    monkeypatch.setattr(diar, "assert_linux_docker_runtime", lambda: None)
    monkeypatch.setattr(diar, "_model_file", lambda _root: model)

    def runner(argv, working_directory, timeout_seconds):
        calls.append((list(argv), working_directory, timeout_seconds))
        output = Path(argv[argv.index("--output") + 1])
        output.write_text(
            "SPEAKER source 1 0.000 0.500 <NA> <NA> speaker_0 <NA> <NA>\n",
            encoding="utf-8",
        )
        return diar.DiarizationCommandResult(0)

    backend = diar.SortformerDiarizer(
        component,
        executable=executable,
        runner=runner,
    )
    result = backend.diarize(np.zeros(16_000, dtype=np.float32), 16_000)

    assert len(result.segments) == 1
    argv = calls[0][0]
    assert argv[0] == str(executable)
    assert argv[1:3] == ["--quiet", "diarize"]
    assert argv[argv.index("--model") + 1] == str(model)
    assert argv[argv.index("--device") + 1] == "cuda:0"
    assert argv[argv.index("--format") + 1] == "rttm"
    assert argv[argv.index("--onset") + 1] == "0.4"
    assert argv[argv.index("--offset") + 1] == "0.7"


def test_native_diarizer_fails_closed_when_runtime_does_not_write_rttm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    component = tmp_path / "component"
    component.mkdir()
    model = component / diar.SORTFORMER_MODEL_FILENAME
    model.write_bytes(b"model")
    executable = tmp_path / "nemo-speech"
    executable.write_bytes(b"runtime")
    monkeypatch.setattr(diar, "assert_linux_docker_runtime", lambda: None)
    monkeypatch.setattr(diar, "_model_file", lambda _root: model)
    backend = diar.SortformerDiarizer(
        component,
        executable=executable,
        runner=lambda *_args: diar.DiarizationCommandResult(0),
    )

    with pytest.raises(diar.WorkstationDiarizationError, match="RTTM"):
        backend.diarize(np.zeros(16_000, dtype=np.float32), 16_000)


def test_native_diarizer_batches_vad_clips_with_offline_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    component = tmp_path / "component"
    component.mkdir()
    model = component / diar.SORTFORMER_MODEL_FILENAME
    model.write_bytes(b"model")
    executable = tmp_path / "nemo-speech"
    executable.write_bytes(b"runtime")
    calls: list[list[str]] = []
    monkeypatch.setattr(diar, "assert_linux_docker_runtime", lambda: None)
    monkeypatch.setattr(diar, "_model_file", lambda _root: model)

    def runner(argv, _working_directory, _timeout_seconds):
        calls.append(list(argv))
        output = Path(argv[argv.index("--output-dir") + 1])
        output.joinpath("clip_000000.rttm").write_text(
            "SPEAKER clip_000000 1 0.000 0.500 <NA> <NA> speaker_0 <NA> <NA>\n",
            encoding="utf-8",
        )
        output.joinpath("clip_000001.rttm").write_text(
            "SPEAKER clip_000001 1 0.000 0.500 <NA> <NA> speaker_1 <NA> <NA>\n",
            encoding="utf-8",
        )
        return diar.DiarizationCommandResult(0)

    result = diar.SortformerDiarizer(
        component,
        executable=executable,
        runner=runner,
    ).diarize_clips(
        (
            np.zeros(16_000, dtype=np.float32),
            np.zeros(8_000, dtype=np.float32),
        ),
        16_000,
    )

    assert len(result.clips) == 2
    assert result.as_dict()["mode"] == "vad-segment-offline-batch"
    assert result.as_dict()["postprocess"]["onset"] == 0.4
    assert result.as_dict()["clip_count"] == 2
    argv = calls[0]
    assert "--offline" in argv
    assert argv[argv.index("--output-dir") + 1].endswith("output")
    assert "--output" not in argv
    assert argv[argv.index("--pad-onset") + 1] == "0.05"
    assert argv[argv.index("--min-duration-on") + 1] == "0.2"
