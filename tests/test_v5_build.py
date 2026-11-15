from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import onnx
from onnx import numpy_helper
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v5_graph_templates_have_no_initializers():
    graph_dir = ROOT / "scripts" / "v5_graph"
    for component, count in (("ar", 291), ("encoder", 291), ("dit", 294), ("vocoder", 194)):
        model = onnx.load(graph_dir / f"{component}-topology.onnx")
        assert not model.graph.initializer and not model.graph.sparse_initializer
        for node in model.graph.node:
            for attribute in node.attribute:
                assert attribute.type not in (onnx.AttributeProto.TENSORS,
                    onnx.AttributeProto.SPARSE_TENSOR, onnx.AttributeProto.SPARSE_TENSORS)
                if attribute.type != onnx.AttributeProto.TENSOR:
                    continue
                value = numpy_helper.to_array(attribute.t)
                assert value.nbytes <= 2048
                if value.dtype.kind == "f":
                    assert np.all(np.isin(value, [0, 1, 2, 3, 4, -10000, 9.796875, 11.3125, -np.inf]) | np.isnan(value))
        bindings = json.loads((graph_dir / f"{component}-bindings.json").read_text())
        assert len(bindings["initializers"]) == count
        assert len({row["name"] for row in bindings["initializers"]}) == count


def test_v5_native_header_requires_exact_inventory(tmp_path):
    native = load_script("v5_build_native")
    path = tmp_path / "ffn2_candidate_hashes.h"
    with pytest.raises(ValueError):
        native.write_hash_header(path, ["a" * 64] * 23)
    hashes = [f"{i:064x}" for i in range(24)]
    native.write_hash_header(path, hashes)
    contents = path.read_text()
    assert all(value in contents for value in hashes)


def test_v5_acoustic_rejects_unrecognized_checkpoint(tmp_path):
    acoustic = load_script("v5_build_acoustic")
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="does not match"):
        acoustic.export("encoder", checkpoint, tmp_path / "encoder.onnx", None)
    assert not (tmp_path / "encoder.onnx").exists()


def test_v5_model_hardlinks_only_weights_and_reports_cross_device(tmp_path, monkeypatch):
    import errno
    import os
    package = load_script("v5_assemble_package")
    weight, metadata = tmp_path / "weight.pth", tmp_path / "config.json"
    weight.write_bytes(b"weights")
    metadata.write_text("{}")
    linked, copied = tmp_path / "linked.pth", tmp_path / "copied.json"
    package.copy_model_file(weight, linked, link_models=True)
    package.copy_model_file(metadata, copied, link_models=True)
    assert os.path.samefile(weight, linked)
    assert not os.path.samefile(metadata, copied)
    def cross_device(*_):
        raise OSError(errno.EXDEV, "cross-device link")
    monkeypatch.setattr(package.os, "link", cross_device)
    with pytest.raises(OSError, match="same filesystem"):
        package.copy_model_file(weight, tmp_path / "other.pth", link_models=True)


def test_v5_assembler_locks_vendored_english_g2p():
    package = load_script("v5_assemble_package")
    package.validate_g2p_source(package.G2P_SOURCE, package.G2P_PROVENANCE)
    assert package.G2P_FILES["checkpoint20.npz"] == "b8af35e4596d8dd5836dfd3fe9b2ba4f97b9c311efe8879544cbcfcbd566d8c6"


def test_v5_assembler_copies_english_g2p_with_notices(tmp_path):
    package = load_script("v5_assemble_package")
    destination = tmp_path / "source" / "g2p_en"
    destination.parent.mkdir()
    package.copy_g2p_source(package.G2P_SOURCE, package.G2P_PROVENANCE, destination)
    assert set(package.G2P_FILES).issubset({path.name for path in destination.iterdir()})
    assert (destination / "ANIFLIVE_TTS_PROVENANCE.json").is_file()
    assert "Apache-2.0" in (destination / "NOTICE.txt").read_text()
    with pytest.raises(ValueError, match="already contains"):
        package.copy_g2p_source(package.G2P_SOURCE, package.G2P_PROVENANCE, destination)
