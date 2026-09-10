from pathlib import Path
import json
import subprocess

import pytest

from aniflive_tts import converter


def test_failed_export_retains_exact_graph_and_inputs(tmp_path: Path, monkeypatch) -> None:
    shared = tmp_path / "shared"
    source = tmp_path / "source"
    for path in (
        shared / "chinese-hubert-base/pytorch_model.bin",
        shared / "chinese-roberta-wwm-ext-large/pytorch_model.bin",
        shared / "sv/pretrained_eres2netv2w24s4ep4.ckpt",
        source / "export_onnx.py",
        source / "onnx_to_fp16.py",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    def fail_after_export(command, *, cwd, env):
        output = Path(command[command.index("--output_dir") + 1])
        output.mkdir()
        (output / "sovits.onnx").write_bytes(b"exact failing graph")
        (output / "validation_report.json").write_text(
            json.dumps([{"model_name": "SoVITS", "passed": False}])
        )
        (output / "validation-inputs").mkdir()
        (output / "validation-inputs/probe.npz").write_bytes(b"exact failing input")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(converter, "_run", fail_after_export)
    staging = tmp_path / "staging"
    with pytest.raises(subprocess.CalledProcessError):
        converter.export_onnx(
            gpt=tmp_path / "model.ckpt", sovits=tmp_path / "model.pth",
            shared_dir=shared, source_dir=source, output_dir=staging / "onnx",
            max_len=1000,
        )
    assert (staging / "failed-onnx-export/sovits.onnx").read_bytes() == b"exact failing graph"
    assert (staging / "validation-inputs/probe.npz").read_bytes() == b"exact failing input"
    assert (staging / "pytorch-onnx-validation.json").is_file()
    assert not list(staging.glob(".onnx.fp32-*"))
