# 安装与共享资源

本文说明如何从全新 clone 创建 Windows 开发环境、下载转换器所需的共享资源，
再生成可供 Docker 使用的模型包。所有 Python 包只安装在项目 `.venv` 中；
不会修改系统 Python 或 CUDA。

## 前置条件

- Python 3.11
- 兼容的 NVIDIA 显卡驱动
- CUDA 12.x 兼容环境
- Git、FFmpeg
- 约 2 GB 空间用于共享模型与 NLTK／fastText 数据

## 创建环境

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

以上命令只用于明确的安装阶段。API 启动和请求处理期间不会执行下载或安装。

## 下载共享资源

先阅读[第三方许可声明](../THIRD_PARTY_NOTICES.md)，再运行：

```powershell
.\.venv\Scripts\python.exe scripts\setup_shared_assets.py `
  --output D:\models\shared `
  --accept-third-party-licenses
```

该命令会下载固定 revision 的 GPT-SoVITS HuBERT、BERT 与 speaker model，
以及 fastText `lid.176.bin` 和三个 NLTK Data 包。已存在且通过检查的文件会直接复用，
不会重复下载。结果至少包含：

```text
D:\models\shared\
├── chinese-hubert-base\pytorch_model.bin
├── chinese-roberta-wwm-ext-large\pytorch_model.bin
├── sv\pretrained_eres2netv2w24s4ep4.ckpt
├── fast_langdetect\lid.176.bin
├── nltk_data\
└── shared-assets.json
```

## 后续步骤

使用 `--shared-dir D:\models\shared` 运行 README 中的模型转换命令。转换完成后，
把模型包放到 `.env` 指定的 `ANIFLIVE_TTS_MODELS_DIR`，再明确构建 Docker image。
以后只重建 container 时会使用现有 image、模型 bind mount 与 cache，不会重新下载。
