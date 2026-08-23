# 安裝與共享資源

本文件說明如何從全新 clone 建立 Windows 開發環境、下載轉換器所需的共享資源，
再建立可供 Docker 使用的模型套件。所有 Python 套件只安裝在專案 `.venv`；
不修改系統 Python 或 CUDA。

## 前置條件

- Python 3.11
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
.\.venv\Scripts\python.exe -m pip install -e .
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

## 後續步驟

以 `--shared-dir D:\models\shared` 執行 README 的模型轉換命令。轉換完成後，
把模型套件放到 `.env` 指定的 `ANIFLIVE_TTS_MODELS_DIR`，再明確建立 Docker image。
後續只重建 container 時會使用既有 image、模型 bind mount 與 cache，不會重新下載。
