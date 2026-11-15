# Installation And Shared Assets

Prepare a model package and shared resources for **CUDA／TensorRT（Docker）**. The commands below use an isolated Windows `.venv`. For **Apple Silicon（macOS／MLX／Metal）**, use the [macOS installation guide](MACOS_INSTALLATION_EN.md).

## Prerequisites

- Python 3.11
- For v1.5 source/editable builds: Rust 1.98.1, CMake, and MSVC C++ Build Tools. Docker builds provide their own compiler toolchain.
- A compatible NVIDIA display driver
- A CUDA 12.x-compatible environment
- Git and FFmpeg
- Approximately 2 GB for shared models and NLTK/fastText data

## Create The Environment

```powershell
git clone https://github.com/Hiruynk/AnifLive-TTS.git
Set-Location AnifLive-TTS
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements\torch-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\base.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\tensorrt11-cu12.txt
.\.venv\Scripts\python.exe -m pip install -e ".[runtime]"
```

These commands are explicit setup operations. API startup and request handling
never download or install packages.

## Download Shared Resources

Review [Third-Party Notices](../THIRD_PARTY_NOTICES.md), then run:

```powershell
.\.venv\Scripts\python.exe scripts\setup_shared_assets.py `
  --output D:\models\shared `
  --accept-third-party-licenses
```

The command downloads GPT-SoVITS HuBERT, BERT, and speaker models from a pinned
revision, plus fastText `lid.176.bin` and three NLTK Data packages. Existing
files that pass validation are reused without another download. The resulting
layout includes:

```text
D:\models\shared\
├── chinese-hubert-base\pytorch_model.bin
├── chinese-roberta-wwm-ext-large\pytorch_model.bin
├── sv\pretrained_eres2netv2w24s4ep4.ckpt
├── fast_langdetect\lid.176.bin
├── nltk_data\
└── shared-assets.json
```

## v2ProPlus conversion

```powershell
.\.venv\Scripts\aniflive-tts.exe model convert `
  --gpt D:\models\voice.ckpt --sovits D:\models\voice.pth `
  --reference-audio D:\models\reference.wav `
  --reference-text-file D:\models\reference.txt --reference-language ja `
  --model-id my-v2proplus --voice-profile default `
  --shared-dir D:\models\shared --output D:\models\my-v2proplus
```

The converter exports ONNX, builds FP16 TensorRT engines and validates the package. Use `--allow-unsafe-pickle` only with a trusted checkpoint.

## v5Turbo package

The `gsv-v5turbo` family uses FP16, 48 kHz, four CFM steps and CFG 0. Follow the [v5Turbo build guide](V1_5_V5_BUILD.md); `model rebuild-engines` is the v2ProPlus workflow. Validate rebuilt engines on the target GPU before deployment.

## Python SDK

See the [SDK manual](SDK_EN.md) for wheel installation and sync/async examples.

## Next Steps

Run the v2ProPlus conversion command above with
`--shared-dir D:\models\shared`. Place the resulting model package under the
`ANIFLIVE_TTS_MODELS_DIR` configured in `.env`, then explicitly build the
Docker image. Later container recreation reuses the existing image, bind-mounted
model package, and cache without downloading again.
