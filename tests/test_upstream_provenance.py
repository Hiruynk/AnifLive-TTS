import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_PATH = ROOT / "minimal_inference" / "ANIFLIVE_TTS_PROVENANCE.json"
NOTICE = "Modified by AnifLive-TTS"


def _load_provenance():
    return json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))


def _all_modified_files(provenance):
    yield from provenance["modified_files"]
    for component in provenance["modified_embedded_components"]:
        yield from component["modified_files"]
    for component in provenance["vendored_apache_components"]:
        yield from component["modified_files"]


def test_apache_provenance_is_pinned_and_complete():
    provenance = _load_provenance()
    upstream = provenance["upstream"]

    assert upstream["license"] == "Apache-2.0"
    assert re.fullmatch(r"[0-9a-f]{40}", upstream["revision"])
    assert (ROOT / "minimal_inference" / upstream["license_file"]).is_file()

    expected = {
        "requirements.txt",
        "docker/requirements-cpu.txt",
        "export_onnx.py",
        "config/voices.json",
        "onnx_to_fp16.py",
        "run_onnx_inference.py",
        "run_onnx_streaming_inference.py",
        "run_onnx_long_inference.py",
        "run_inference.py",
        "run_streaming_inference.py",
        "run_trt_inference.py",
        "utils.py",
        "README.md",
        "README_zh.md",
        "GPT_SoVITS/AR/models/t2s_lightning_module.py",
        "GPT_SoVITS/AR/models/t2s_model.py",
        "GPT_SoVITS/feature_extractor/cnhubert.py",
        "GPT_SoVITS/utils.py",
        "GPT_SoVITS/text/g2p_en/g2p.py",
        "GPT_SoVITS/text/g2p_en/homographs.en",
    }
    records = list(_all_modified_files(provenance))
    assert {record["path"] for record in records} == expected

    for record in records:
        assert re.fullmatch(r"[0-9a-f]{64}", record["upstream_sha256"])
        assert record["changes"]
        source = ROOT / "minimal_inference" / record["path"]
        assert source.is_file()
        header = "\n".join(source.read_text(encoding="utf-8").splitlines()[:6])
        assert NOTICE in header


def test_third_party_notice_links_provenance():
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert "minimal_inference/ANIFLIVE_TTS_PROVENANCE.json" in notices
    assert "g2p-en 2.1.0" in notices
    assert NOTICE in notices
