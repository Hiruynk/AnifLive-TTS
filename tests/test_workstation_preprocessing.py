from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from aniflive_tts import workstation_preprocessing as preprocessing


def _examples(tmp_path: Path) -> tuple[preprocessing.TrainingExample, ...]:
    source = tmp_path / "voice.list"
    source.write_text(
        "clips/a.wav|voice|ja|hello\n"
        "nested\\b.wav|voice|YUE|world\n",
        encoding="utf-8",
    )
    return preprocessing.parse_training_list(source)


def test_training_list_is_generic_five_language_and_basename_safe(tmp_path: Path) -> None:
    examples = _examples(tmp_path)
    assert [(item.wav_name, item.language) for item in examples] == [
        ("a.wav", "ja"),
        ("b.wav", "yue"),
    ]
    duplicate = tmp_path / "duplicate.list"
    duplicate.write_text(
        "one/a.wav|voice|ja|one\ntwo/a.wav|voice|en|two\n", encoding="utf-8"
    )
    with pytest.raises(preprocessing.PreprocessingWorkerError, match="duplicate"):
        preprocessing.parse_training_list(duplicate)


def test_v2proplus_06_checkpoint_is_repaired_only_in_private_copy(tmp_path: Path) -> None:
    source = tmp_path / "voice.pth"
    original = b"06" + b"private-model-payload"
    source.write_bytes(original)
    source_digest = hashlib.sha256(original).hexdigest()

    staged = preprocessing.stage_v2proplus_checkpoint(source, tmp_path / "scratch")

    assert staged.original_header == "06"
    assert staged.repaired is True
    assert staged.source_sha256 == source_digest
    assert source.read_bytes() == original
    assert staged.path.read_bytes() == b"PK" + original[2:]
    assert staged.path != source


def test_checkpoint_staging_accepts_regular_torch_archive_and_rejects_unknown_header(
    tmp_path: Path,
) -> None:
    regular = tmp_path / "regular.pth"
    regular.write_bytes(b"PK" + b"archive")
    staged = preprocessing.stage_v2proplus_checkpoint(regular, tmp_path / "regular-scratch")
    assert staged.repaired is False
    assert staged.path.read_bytes() == regular.read_bytes()

    invalid = tmp_path / "invalid.pth"
    invalid.write_bytes(b"XXpayload")
    with pytest.raises(preprocessing.PreprocessingWorkerError, match="06 header"):
        preprocessing.stage_v2proplus_checkpoint(invalid, tmp_path / "invalid-scratch")


def test_speaker_vector_gate_rejects_upstream_zero_exit_with_missing_outputs(
    tmp_path: Path,
) -> None:
    examples = _examples(tmp_path)
    output = tmp_path / "output"
    (output / "7-sv_cn").mkdir(parents=True)
    with pytest.raises(preprocessing.PreprocessingWorkerError, match="expected 2, got 0"):
        preprocessing.validate_speaker_embedding_outputs(output, examples)

    (output / "7-sv_cn" / "a.wav.pt").write_bytes(b"a")
    with pytest.raises(preprocessing.PreprocessingWorkerError, match="missing=b.wav.pt"):
        preprocessing.validate_speaker_embedding_outputs(output, examples)

    (output / "7-sv_cn" / "b.wav.pt").write_bytes(b"b")
    assert preprocessing.validate_speaker_embedding_outputs(output, examples) == 2


def test_preprocessing_pythonpath_includes_upstream_and_eres2net_roots(tmp_path: Path) -> None:
    source = tmp_path / "gpt-sovits"
    values = preprocessing._pythonpath(source, str(tmp_path / "existing")).split(os.pathsep)
    assert values == [
        str(source),
        str(source / "GPT_SoVITS"),
        str(source / "GPT_SoVITS" / "eres2net"),
        str(tmp_path / "existing"),
    ]
