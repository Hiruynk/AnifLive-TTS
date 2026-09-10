from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_worker_image_pins_training_source_and_dependencies() -> None:
    dockerfile = (ROOT / "Dockerfile.workstation-worker").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements/workstation-worker.txt").read_text(
        encoding="utf-8"
    )
    assert "48b1a0169a28582a8984402f82cf438d3bfa6aca" in dockerfile
    assert "1c967e31777b2b88468af3e7481bcb770ac4d09dd854bdb9ee065d8c8c75fcb6" in dockerfile
    assert "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61" in dockerfile
    assert "f8f8d2f2190b9909b51e91ce886d1c7efedb7349b3ff6d3ab521451165fbd8da" in dockerfile
    assert "ANIFLIVE_TTS_CLEARVOICE_REVISION" in dockerfile
    assert "nemo-speech-0.1.0-linux-x86_64-cuda.tar.gz" in dockerfile
    assert "e68628f396489c98fb353e070efaea5bc4977409ae7734fce56c251a79e29147" in dockerfile
    assert "test -x /opt/nemo-speech/bin/nemo-speech" in dockerfile
    assert 'target / "LICENSE"' in dockerfile
    assert 'extracted / "LICENSE"' in dockerfile
    assert "last_best_checkpoint.pt" not in dockerfile
    assert "s1_train.py" in dockerfile and "s2_train.py" in dockerfile
    assert "COPY minimal_inference /app/minimal_inference" in dockerfile
    assert 'py_compile.compile("/app/minimal_inference/export_onnx.py"' in dockerfile
    assert "pytorch-lightning==" in requirements
    assert "faster-whisper==" in requirements
    assert "ctranslate2==" in requirements
    assert "funasr==1.4.11" in requirements
    assert "rotary-embedding-torch==0.8.3" in requirements
    assert "torchinfo==1.8.0" in requirements
    assert "torchcodec==0.10.0" in requirements
    assert "torchaudio.load(probe)" in dockerfile
    assert "import torchcodec" in dockerfile
    assert "import funasr" in dockerfile
    assert "git clone" not in dockerfile
    assert "latest" not in requirements.casefold()
