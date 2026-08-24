ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9
FROM ${CUDA_BASE_IMAGE}

ARG TORCH_REQUIREMENTS=requirements/torch-cu128.txt
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="AnifLive-TTS"
LABEL org.opencontainers.image.version="1.1.0"
LABEL org.opencontainers.image.description="First-party TTS runtime for AnifEngine-Voice"
LABEL org.opencontainers.image.authors="Hiruynk"
LABEL org.opencontainers.image.vendor="AnifEngine"
LABEL org.opencontainers.image.licenses="PolyForm-Noncommercial-1.0.0"
LABEL org.opencontainers.image.source="https://github.com/Hiruynk/AnifLive-TTS"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    ANIFLIVE_TTS_SOURCE_DIR=/app/minimal_inference \
    ANIFLIVE_TTS_MODEL_PACKAGE=/data/models/active \
    ANIFLIVE_TTS_SHARED_DIR=/data/shared \
    ANIFLIVE_TTS_CACHE_DIR=/data/cache \
    HF_HOME=/data/cache/huggingface \
    TORCH_HOME=/data/cache/torch \
    XDG_CACHE_HOME=/data/cache/xdg \
    NLTK_DATA=/opt/aniflive-tts/nltk_data \
    ANIFLIVE_TTS_FAST_LANGDETECT_CACHE=/app/pretrained_models/fast_langdetect \
    ANIFLIVE_TTS_FAST_LANGDETECT_MODEL=/app/pretrained_models/fast_langdetect/lid.176.bin \
    ANIFLIVE_TTS_PORT=9880

RUN --mount=type=cache,id=aniflive-apt,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=aniflive-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ffmpeg libffi-dev libsndfile1 python3 python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && python -m pip install --upgrade \
      pip==26.2.1 setuptools==84.0.0 wheel==0.48.0

COPY requirements /tmp/requirements
COPY scripts/shared_assets_lock.json /tmp/shared_assets_lock.json
RUN --mount=type=cache,id=aniflive-pip,target=/root/.cache/pip,sharing=locked \
    python -m pip install -r "/tmp/${TORCH_REQUIREMENTS}" \
    && python -m pip install -r /tmp/requirements/base.txt \
    && python -m pip install -r /tmp/requirements/tensorrt11-cu12-linux.txt \
    && python -m pip install --no-deps tensorrt-cu12==11.2.1.2

# split-lang's CJK extras install overlapping MeCab modules. Reinstall the
# selected BSD/Apache-licensed Korean bindings last so imports are deterministic.
RUN --mount=type=cache,id=aniflive-pip,target=/root/.cache/pip,sharing=locked \
    python -m pip install --no-deps --force-reinstall \
    python-mecab-ko==1.3.7 \
    python-mecab-ko-dic==2.1.1.post2

# English G2P resources are immutable, SHA-256-pinned image inputs. Serving
# never invokes the NLTK downloader, so recreating a container remains offline.
RUN python - <<'PY'
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile

lock = json.loads(Path("/tmp/shared_assets_lock.json").read_text(encoding="utf-8"))
nltk = lock["nltk"]
root = Path("/opt/aniflive-tts/nltk_data")
for package, metadata in nltk["packages"].items():
    subdir = metadata["subdir"]
    url = (
        f"https://raw.githubusercontent.com/{nltk['repository']}/"
        f"{nltk['revision']}/packages/{subdir}/{package}.zip"
    )
    archive = root / subdir / f"{package}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != metadata["zip_sha256"]:
        raise RuntimeError(f"NLTK archive checksum mismatch: {package}")
    target = root / subdir / package
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        prefix = PurePosixPath(package)
        for member in bundle.infolist():
            if member.is_dir():
                continue
            member_path = PurePosixPath(member.filename)
            try:
                relative = member_path.relative_to(prefix)
            except ValueError as exc:
                raise RuntimeError(f"Unexpected NLTK archive member: {member.filename}") from exc
            if not relative.parts or ".." in relative.parts:
                raise RuntimeError(f"Unsafe NLTK archive member: {member.filename}")
            destination = target.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink)
    tree = hashlib.sha256()
    for candidate in sorted(path for path in target.rglob("*") if path.is_file()):
        tree.update(candidate.relative_to(target).as_posix().encode("utf-8"))
        tree.update(b"\0")
        tree.update(hashlib.sha256(candidate.read_bytes()).hexdigest().encode("ascii"))
        tree.update(b"\n")
    if tree.hexdigest() != metadata["tree_sha256"]:
        raise RuntimeError(f"NLTK content checksum mismatch: {package}")
PY

# Cache fastText language detection in a dependency-only layer. Source edits
# must not trigger another 125 MB download, and runtime remains offline.
RUN python - <<'PY'
import fast_langdetect
import hashlib
import json
from pathlib import Path
import shutil

fast_langdetect.infer._default_detector.detect(
    "AnifLive-TTS offline resources", low_memory=False
)
source = Path("/tmp/fasttext-langdetect/lid.176.bin")
target = Path("/app/pretrained_models/fast_langdetect/lid.176.bin")
if not source.is_file():
    raise RuntimeError(f"fastText build cache was not created: {source}")
lock = json.loads(Path("/tmp/shared_assets_lock.json").read_text(encoding="utf-8"))
if hashlib.sha256(source.read_bytes()).hexdigest() != lock["fasttext"]["sha256"]:
    raise RuntimeError("fastText lid.176.bin checksum mismatch")
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, target)
PY

COPY pyproject.toml README.md LICENSE LICENSING.md THIRD_PARTY_NOTICES.md /app/
COPY licenses /app/licenses
COPY src /app/src
COPY minimal_inference /app/minimal_inference
COPY scripts /app/scripts
RUN python -m pip install --no-deps /app \
    && mkdir -p /data/models /data/shared /data/cache /data/reports \
    && chmod 0755 /app/scripts/entrypoint.sh \
    && PYTHONPATH=/app/minimal_inference:/app/minimal_inference/GPT_SoVITS python - <<'PY'
import pyopenjtalk
import importlib.util
import text.japanese
import text.korean
import text.english
from text.LangSegmenter.langsegmenter import LangSegmenter

assert importlib.util.find_spec("distance") is None
assert importlib.util.find_spec("eunjeon") is None
assert importlib.util.find_spec("chardet") is None
assert importlib.util.find_spec("g2p_en") is None
pyopenjtalk.g2p("こんにちは")
assert text.korean.g2p("오늘은 날씨가 좋습니다.")
assert text.english.g2p("AnifLive-TTS speaks English.")
assert LangSegmenter.getTexts("Hello，初めまして。", default_lang="ja")
PY

EXPOSE 9880
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["serve"]
