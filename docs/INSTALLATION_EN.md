# Installation And Shared Assets

This guide starts from a fresh clone, creates an isolated Windows environment,
downloads the shared resources required by the converter, and prepares a model
package for Docker. Python packages are installed only in the project `.venv`;
the system Python and CUDA installation are not modified.

## Prerequisites

- Python 3.11
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
.\.venv\Scripts\python.exe -m pip install -e .
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

## Next Steps

Run the model conversion command in the README with
`--shared-dir D:\models\shared`. Place the resulting model package under the
`ANIFLIVE_TTS_MODELS_DIR` configured in `.env`, then explicitly build the
Docker image. Later container recreation reuses the existing image, bind-mounted
model package, and cache without downloading again.
