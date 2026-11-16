# Native macOS Installation

**Backend:** Apple Silicon（macOS／MLX／Metal）. Neural inference runs in a Rust / MLX / Metal worker; Python provides the HTTP API and text frontend. FP16 is recommended. The Windows backend remains CUDA／TensorRT（Docker）.

## Hardware and macOS support

- Verified host: Mac Studio M5 Max, 40-core GPU, 128 GB unified memory, macOS 27.0.
- Other Apple Silicon models (M1–M5 variants) are untested. No minimum memory requirement has been verified.
- Intel Macs are unsupported.
- The bundled MLX libraries set a macOS 26.2+ deployment floor; only macOS 27.0 has been verified.

## Install prebuilt runtime

Download the `AnifLive-TTS-v1.5.0-macos-arm64-runtime.zip` runtime archive and the `aniflive_tts-1.5.0-cp310-abi3-macosx_11_0_arm64.whl` SDK wheel from the project release. Use a macOS checkout and a user-owned environment file for paths:

```bash
export ANIFLIVE_TTS_MACOS_ENV_FILE="$HOME/.config/aniflive-tts/macos.env"
mkdir -p "$(dirname "$ANIFLIVE_TTS_MACOS_ENV_FILE")"
cp scripts/macos-release/macos-v5turbo.env.example "$ANIFLIVE_TTS_MACOS_ENV_FILE"
```

Edit the file to point to a **writable copy** of an existing authorized, converted v5Turbo model package, its upstream frontend source, and the local Open JTalk dictionary. A dynamic shared-base package needs no fixed voice: register your own Voice and Reference after startup. Provide the model weights and converted assets separately; the runtime archive and SDK wheel do not contain them. Keep an untouched original package for rollback.

Before installation, review the [third-party notices](../THIRD_PARTY_NOTICES.md). The command below reuses `setup_shared_assets.py` to download and checksum only the NLTK and fastText text-frontend assets, without downloading GPT-SoVITS model weights. The first run needs network access; verified caches are reused. Place the pinned upstream frontend at `$ANIFLIVE_TTS_SOURCE_DIR/GPT_SoVITS/` and provide an Open JTalk dictionary containing `sys.dic`.

The following example assumes both downloaded files are in `~/Downloads`:

```bash
source "$ANIFLIVE_TTS_MACOS_ENV_FILE"
python3.12 - "$ANIFLIVE_TTS_MACOS_DATA_ROOT" <<'PY'
from pathlib import Path
import shutil
import sys
from scripts.setup_shared_assets import NLTK_PACKAGES, _digest, _download_fasttext, _download_nltk

root = Path(sys.argv[1]).expanduser().resolve()
_download_nltk(root)
cache = root / "cache" / "nltk"
cache.mkdir(parents=True, exist_ok=True)
for package, spec in NLTK_PACKAGES.items():
    source = root / "nltk_data" / spec["subdir"] / f"{package}.zip"
    destination = cache / f"{package}.zip"
    if not destination.is_file() or _digest(destination) != spec["zip_sha256"]:
        shutil.copyfile(source, destination)
    if _digest(destination) != spec["zip_sha256"]:
        raise RuntimeError(f"NLTK cache checksum mismatch: {package}")
_download_fasttext(root / "cache")
PY
mkdir -p "$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT/download"
unzip "$HOME/Downloads/AnifLive-TTS-v1.5.0-macos-arm64-runtime.zip" -d "$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT/download"
ANIFLIVE_TTS_BUILD_WORKER=0 \
  ANIFLIVE_TTS_SDK_WHEEL="$HOME/Downloads/aniflive_tts-1.5.0-cp310-abi3-macosx_11_0_arm64.whl" \
  bash scripts/macos-release/install.sh
"$ANIFLIVE_TTS_VENV/bin/python" scripts/macos-qualification/fp16_package_refresh.py \
  "$ANIFLIVE_TTS_MODEL_PACKAGE" \
  "$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT/download/AnifLive-TTS-v1.5.0-macos-arm64-runtime"
```

The installer installs the pinned Python dependencies and the prebuilt SDK wheel; no Rust toolchain is needed for these flags. The refresh helper replaces the worker, MLX libraries and licenses in the package copy and regenerates package metadata and checksums. It does not supply or convert model weights or voice data. For source conversion utilities see `scripts/macos-v5turbo/convert_weights.py` and `convert_g2pw.py`; they do not themselves create a complete package.

The example selects FP16. For the frozen FP32 rollback package, set `ANIFLIVE_TTS_PRECISION=FP32` and use a separate FP32 package; it retains its original feature set. Check the active worker's `expression.policies` and `expression.complete_policies` before selecting a policy.

The package layout is illustrated by `scripts/macos-release/model-package.template.json`.
Set `native_mandarin: true` when the package includes the converted G2PW and
BERT frontend assets; otherwise its capability response excludes Mandarin.

## Build from source

To build the SDK and worker instead, install the Rust toolchain, omit `ANIFLIVE_TTS_SDK_WHEEL`, and run:

```bash
ANIFLIVE_TTS_BUILD_WORKER=0 bash scripts/macos-release/install.sh
MACOSX_DEPLOYMENT_TARGET=26.2 bash scripts/macos-release/build.sh --output "$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT/build/native-worker"
"$ANIFLIVE_TTS_VENV/bin/python" scripts/macos-qualification/fp16_package_refresh.py "$ANIFLIVE_TTS_MODEL_PACKAGE" "$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT/build/native-worker"
```

The installer can also build the worker directly with its default settings.

## Start and manage the service

Run the commands from the repository root. `start.sh` runs in the foreground and binds to loopback at the configured `ANIFLIVE_TTS_PORT`; use another terminal for health, stop, and restart commands while it is running.

```bash
bash scripts/macos-release/start.sh
bash scripts/macos-release/health.sh
bash scripts/macos-release/stop.sh
bash scripts/macos-release/restart.sh
```

Use `health.sh` after startup or restart to check service readiness. `stop.sh` stops the worker and releases its local port.

## Runtime capabilities

The current FP16 worker returns 48 kHz mono PCM16, using native four-step CFM and CFG 0. It supports complete WAV, PCM streaming, the official synchronous and asynchronous SDK, and all five Reference-conditioning policies: `full-switch`, `identity-lock`, `semantic-style`, `acoustic-style`, and `sv-only`. These policies map to v5's semantic, style, and CFM voice-anchor components; v5 has no separate v2ProPlus SV encoder.

Registered Voices and expression profiles appear in `/v1/voices` and `/v1/expressions`; query a Voice-specific expression catalog with `GET /v1/expressions?voice_profile=<id>`. SDK requests select a registered Voice with `voice_profile=...`; References are prepared on demand and cached. Per-segment expression and mixed policies are supported. Generation accepts `top_k` 1–50, `top_p` in (0, 1], `temperature` greater than 0, and integer `seed` ≥ -1. v5 requires `speed=1.0`, `noise_scale=0.5`, four sample steps, and CFG 0. Compatibility adapters accept `auto`, `auto_yue`, and supported aliases; canonical `ja` uses the mixed Japanese/English frontend mode.

PCM streaming is incremental in the vocoder: autoregressive generation completes through EOS and the full CFM finishes before the first PCM; vocoder windows then emit PCM as they complete. There is no pre-EOS semantic or acoustic stream. Cancelling complete WAV returns HTTP 409 with no partial WAV; the service can recover for the next request. Session A supports append, Flush, Cancel, and recovery. Session strategies B–F remain research features, not formal API promises. v2ProPlus and Studio remain Windows features.

To open the shared WebUI, run `ANIFLIVE_TTS_WEBUI_DIR="$PWD/webui" "$ANIFLIVE_TTS_VENV/bin/aniflive-tts" webui --upstream "http://${ANIFLIVE_TTS_HOST}:${ANIFLIVE_TTS_PORT}"`, then visit `http://127.0.0.1:9890/`. Mac browser playback is not verified. Studio workstation features remain Windows-only.

FP16 uses PyTorch 2.9.1 MPS only for Reference DSP such as resampling, STFT, and Mel processing. Neural speech inference stays in Rust / MLX / Metal, with no CPU, PyTorch, or MPS neural fallback.

For benchmark methods and comparative performance details, see [performance engineering](PERFORMANCE_ENGINEERING.md), the [platform matrix](V1_5_PLATFORM_MATRIX.md), and the [API contract](API_EN.md#macos-apple-silicon-behavior).
