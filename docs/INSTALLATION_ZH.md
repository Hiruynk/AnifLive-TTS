# 安裝與共享資源

本指南為 **CUDA／TensorRT（Docker）** 準備模型套件及共享資源，以下命令使用獨立的 Windows `.venv`。**Apple Silicon（macOS／MLX／Metal）** 請參閱 [macOS 安裝指南](MACOS_INSTALLATION_ZH.md)。

## 前置條件

- Python 3.11
- v1.5 原始碼／editable 建置：Rust 1.98.1、CMake 與 MSVC C++ Build Tools；Docker 建置自備編譯工具。
- 相容的 NVIDIA 顯示卡驅動程式
- CUDA 12.x 相容環境
- Git、FFmpeg
- 約 2 GB 空間供共享模型與 NLTK／fastText 資料使用

## 建立環境

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

上述指令只用於明確的安裝階段。API 啟動與請求處理期間不會執行下載或安裝。

## 下載共享資源

先閱讀 [第三方授權聲明](../THIRD_PARTY_NOTICES.md)，再執行：

```powershell
.\.venv\Scripts\python.exe scripts\setup_shared_assets.py `
  --output D:\models\shared `
  --accept-third-party-licenses
```

此命令會下載固定 revision 的 GPT-SoVITS HuBERT、BERT 與 speaker model，
以及 fastText `lid.176.bin` 和三個 NLTK Data 套件。已存在且通過檢查的檔案會直接重用，
不會再次下載。結果至少包含：

```text
D:\models\shared\
├── chinese-hubert-base\pytorch_model.bin
├── chinese-roberta-wwm-ext-large\pytorch_model.bin
├── sv\pretrained_eres2netv2w24s4ep4.ckpt
├── fast_langdetect\lid.176.bin
├── nltk_data\
└── shared-assets.json
```

## v2ProPlus 轉換

```powershell
.\.venv\Scripts\aniflive-tts.exe model convert `
  --gpt D:\models\voice.ckpt --sovits D:\models\voice.pth `
  --reference-audio D:\models\reference.wav `
  --reference-text-file D:\models\reference.txt --reference-language ja `
  --model-id my-v2proplus --voice-profile default `
  --shared-dir D:\models\shared --output D:\models\my-v2proplus
```

轉換器會匯出 ONNX、建立 FP16 TensorRT 引擎及驗證模型套件。`--allow-unsafe-pickle` 只適用於可信任的 checkpoint。

## v5Turbo 套件

`gsv-v5turbo` 家族使用 FP16、48 kHz、4-step CFM 及 CFG 0。請依照 [v5Turbo 建置指南](V1_5_V5_BUILD.md)準備；`model rebuild-engines` 是 v2ProPlus 的流程。重新建置引擎後，先在目標 GPU 驗證再部署。

## Python SDK

wheel 安裝及同步／非同步範例見 [SDK 使用手冊](SDK_ZH.md)。

## 後續步驟

以 `--shared-dir D:\models\shared` 執行上方 v2ProPlus 轉換命令。轉換完成後，
把模型套件放到 `.env` 指定的 `ANIFLIVE_TTS_MODELS_DIR`，再明確建立 Docker image。
後續只重建 container 時會使用既有 image、模型 bind mount 與 cache，不會重新下載。
