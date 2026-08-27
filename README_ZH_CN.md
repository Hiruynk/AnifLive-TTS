<div align="center">

<img src="assets/everynight_dance.gif" alt="Evernight dance" width="260">

# AnifLive-TTS

**面向粤语／广东话的低延迟、高音质、多语言声音克隆 TTS 推理系统**

[![版本](https://img.shields.io/badge/%E7%89%88%E6%9C%AC-v1.2.0-2563eb?style=flat-square)](RELEASE_NOTES_v1.2.0.md)
[![TensorRT](https://img.shields.io/badge/TensorRT-11.2.1.2-76b900?style=flat-square&logo=nvidia)](https://docs.nvidia.com/deeplearning/tensorrt/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![模型](https://img.shields.io/badge/%E6%A8%A1%E5%9E%8B-GPT--SoVITS_V2ProPlus-0f766e?style=flat-square)](https://github.com/RVC-Boss/GPT-SoVITS)
[![许可](https://img.shields.io/badge/%E8%AE%B8%E5%8F%AF-PolyForm_Noncommercial_1.0.0-22c55e?style=flat-square)](LICENSING.md)

[繁體中文](README_ZH_HK.md) · [English](README.md) · **简体中文**

</div>

## 关于项目

AnifLive-TTS 是 AnifEngine-Voice 的第一方 TTS，诞生于一个很实际的需求：在开发 AnifEngine-Voice 时，我一直找不到一个能同时满足 **粤语／广东话**、**低延迟**、**高音质**和可自托管、自主管理的 TTS 方案。😮‍💨

因此我选择以 GPT-SoVITS 为基础，深入改造其推理底层，将 **低延迟** 和 **高音质** 定为 AnifLive-TTS 的核心目标。🤓👆

v1 首发完整支持 V2ProPlus；未来版本将沿用相同 API 与模型封装格式，逐步支持更多 GPT-SoVITS 模型世代。

## 核心能力

- 九个神经网络模型均由 TensorRT 11 通过 `execute_async_v3()` 执行。
- 支持 普通话(`zh`)、粤语／广东话(`yue`)、英语(`en`)、日语(`ja`)、韩语(`ko`)，并兼容 GPT-SoVITS 传统 API 格式与 OpenAI API 格式。
- 支持 GPT-SoVITS V2ProPlus 声音克隆模型，可将自定义 GPT／SoVITS 模型检查点与参考音频封装为独立音色。
- 完整单声道 PCM16 WAV 与低延迟 PCM16 流式输出共用同一套推理流程。
- v1 可用一条命令完成 V2ProPlus 模型检查、ONNX 导出、FP16 转换、TensorRT 引擎构建、实际推理验证及模型封装。
- 推理服务启动前完成模型与引擎准备，`serve` 热路径完全离线。

## 《无职转生》洛琪希·米格路迪亚：V2ProPlus 粤语／广东话演示

https://github.com/user-attachments/assets/a471c2d5-9382-407a-82ab-ab57b2ea35c1

WebUI 界面仍处于测试阶段，尚未在此版本开放，敬请期待后续版本。

## 实测性能

> [!NOTE]
> **环境：** RTX 5070 Ti 16 GB / NVIDIA 驱动程序 596.36 / CUDA 运行环境 12.8 / PyTorch
> 2.7.0+cu128 / TensorRT 11.2.1.2 / FP16

测试采用外部 Miku 和 Roxy V2ProPlus 音色包，并固定短句、随机种子和采样参数。每个音色各测试 10 轮，同一轮内交替测试两个音色；每组模型轮次先预热 10 次，再测试 100 次完整 WAV、100 次新连接流式输出和 100 次持久连接流式输出。主值取全部 20 组模型轮次统计值的中位数，范围反映各组之间的波动。

正式测试均为单并发。新连接数据会为每个请求建立本地 HTTP/1.1 连接；持久连接数据则在每轮复用一条已单独预热的连接。首包延迟从发送请求起计算，直到客户端读取服务器发出的第一个 PCM 音频块。可听 TTFA 取最早有效 10 ms 均方根分析帧内，第一个超过 -45 dBFS 的 PCM 采样点，并受该音频块的实际到达时间约束；不包含播放设备延迟。

| 指标 | 20 组模型轮次统计值的中位数 | 各组范围 |
|---|---:|---:|
| 完整 REST WAV 端到端 P50 | **192.880 ms** | 169.381–227.030 ms |
| 完整 REST WAV 端到端 P95 | **230.750 ms** | 195.755–273.346 ms |
| 服务器推理 P50 | **166.226 ms** | 139.831–201.973 ms |
| RTF P50 | **0.088774** | 0.071140–0.106101 |
| 流式首包延迟 P50 | **87.140 ms** | 79.405–92.391 ms |
| 流式首包延迟 P95 | **103.113 ms** | 87.924–122.153 ms |
| 持久连接流式首包延迟 P50 | **69.698 ms** | 63.808–72.610 ms |
| 持久连接流式首包延迟 P95 | **85.297 ms** | 76.910–99.756 ms |
| 流式有效音频 TTFA P50 | **96.034 ms** | 88.759–101.671 ms |
| 流式有效音频 TTFA P95 | **111.565 ms** | 98.174–128.778 ms |
| 持久连接流式有效音频 TTFA P50 | **77.717 ms** | 71.789–82.833 ms |
| 持久连接流式有效音频 TTFA P95 | **94.550 ms** | 86.578–110.006 ms |
| GPU 占用率 P50 | **53.0%** | 46–56% |
| GPU 占用率 P95 | **60.0%** | 58–62% |

全部 2,000 个完整 WAV、2,000 个新连接流式输出和 2,000 个持久连接流式输出请求均报告 `TensorRT-11`，且 `X-PyTorch-Fallback: false`。机器可读摘要见 [`benchmarks/README_BENCHMARK_SUMMARY.json`](benchmarks/README_BENCHMARK_SUMMARY.json)。

`nvidia-smi` 显示的是采样区间内的 GPU 占用率，并非 SM 占用率。在单并发下，串行执行的 GPT 自回归流程仍是 GPU 占用率无法接近 100% 的主要原因。

### 重现性能表格

[`scripts/benchmark_readme.py`](scripts/benchmark_readme.py) 是公开性能测试的标准脚本。它输出的 Markdown 表格与上表完全相同，只包含同样的 14 个指标，并采用相同的测试内容和统计方法。

对已经运行的本地 API 执行：

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_readme.py `
  --host 127.0.0.1 --port 9881 --locale zh-CN `
  --model miku-v2proplus `
  --model roxy-v2proplus `
  --report .\reports\benchmark.json `
  --markdown .\reports\benchmark.md
```

也可以直接在现有 Docker 容器内执行，不会重新创建或重建容器：

```powershell
docker exec aniflive-tts /app/scripts/entrypoint.sh benchmark `
  --host 127.0.0.1 --port 9880 --locale zh-CN `
  --model miku-v2proplus --model roxy-v2proplus `
  --report /data/reports/benchmark.json `
  --markdown /data/reports/benchmark.md
```

默认对每个音色执行 10 轮；每轮先预热 10 次，再测试 100 次完整 WAV、100 次新连接流式输出和 100 次持久连接流式输出。重复使用 `--model` 可合并多个音色包的结果。所有测试均为单并发。

### GPT-SoVITS 性能数据对比

#### 首次输出延迟（越低越快）

| 项目／系统 | 指标 | 延迟 | 测试条件 | 来源 |
|---|---|---:|---|---|
| **AnifLive-TTS v1.2** | **可听 TTFA P50** | **77.717 ms** 🚀 | **RTX 5070 Ti；HTTP/1.1 持久连接；20 组 Miku／Roxy 交错模型轮次** | **[本机实测](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| GPT-SoVITS C++ TRT 流式 | 首包 | 460 ms | RTX 2080 Ti 22 GB | [GPT-SoVITS C++](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference ONNX 流式 | 首个 token | 1,000 ms | RTX 2080 Ti 22 GB；FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference TRT 固定尺寸优化版 | 首个语义标记 | 2,022 ms | RTX 2080 Ti 22 GB；FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |

#### RTF（越低越快）

| 项目／系统 | RTF | 后端 | 测试条件 | 来源 |
|---|---:|---|---|---|
| GPT-SoVITS V2ProPlus | 0.014 | PyTorch 并行推理 | RTX 4090；约 4 分钟长文本 | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS#features) |
| GPT-SoVITS V2ProPlus | 0.028 | PyTorch 并行推理 | RTX 4060 Ti | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS#features) |
| **AnifLive-TTS v1.2** | **0.088774** | **TensorRT 11 FP16** | **RTX 5070 Ti；20 组 Miku／Roxy 交错模型轮次** | **[本机实测](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| GPT-SoVITS C++ TRT | 0.1020 | TensorRT | RTX 2080 Ti 22 GB | [GPT-SoVITS C++](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference TRT 固定尺寸优化版 | 0.2096 | TensorRT；针对固定尺寸优化 | RTX 2080 Ti 22 GB；FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |

### 其他开源 TTS 的公开性能数据

以下并非同等条件下的受控基准测试。除 AnifLive-TTS 外，数据均由各来源自行公布；GPU、模型能力、输入内容、首包大小、并发度及测量方法并不相同。表格只整理相同指标的公开数据，不代表同等条件下的排名。

#### 首音频延迟（越低越快）

| 系统 | 指标 | 延迟 | 统计口径 | 测试条件 | 来源 |
|---|---|---:|---|---|---|
| **AnifLive-TTS v1.2** | **可听 TTFA** | **77.717 ms** 🚀 | **P50** | **RTX 5070 Ti；HTTP/1.1 持久连接；20 组 Miku／Roxy 交错模型轮次** | **[本机实测](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| Qwen3-TTS-12Hz-0.6B | 首包延迟 | 97 ms | 并发数 1 | 单加速器；320 ms 语音包 | [Qwen3-TTS 技术报告](https://arxiv.org/abs/2601.15621) |
| Fish Audio S2 | TTFA | 约 100 ms | 项目发布值 | H200；单卡 | [Fish Audio S2](https://github.com/fishaudio/fish-speech#performance) |
| Chatterbox-Flash（D=32，α=0.75） | TTFP | 103 ms | 单并发；50 句 | H100 | [Chatterbox-Flash 论文](https://arxiv.org/abs/2605.30748) |
| Chatterbox-Flash（默认 D=16，α=0.5） | TTFP | 118 ms | 单并发；50 句 | H100 | [Chatterbox-Flash 论文](https://arxiv.org/abs/2605.30748) |
| CosyVoice2 | 首个音频块 | 196.13 ms | P50 | L20；单并发；客户端／服务器 | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice2.DiT.md#benchmark-with-client-server-mode) |

IndexTTS 2.0／2.5 和 VoxCPM2 未提供同口径的首音频延迟数值。

#### RTF（越低越快）

| 系统 | RTF | 推理后端／模型 | 测试条件 | 来源 |
|---|---:|---|---|---|
| Chatterbox-Flash（D=32，α=0.75） | 0.076 | 分块扩散 | H100；单并发；50 句 | [Chatterbox-Flash 论文](https://arxiv.org/abs/2605.30748) |
| **AnifLive-TTS v1.2** | **0.088774** | **TensorRT 11 FP16** | **RTX 5070 Ti；20 组 Miku／Roxy 交错模型轮次** | **[本机实测](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| Chatterbox-Flash（默认 D=16，α=0.5） | 0.107 | 分块扩散 | H100；单并发；50 句 | [Chatterbox-Flash 论文](https://arxiv.org/abs/2605.30748) |
| CosyVoice3 | 0.1091 | TRT-LLM；离线批次 1 | L20 | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice3.md#benchmark-with-offline-inference-mode) |
| CosyVoice2 | 0.1228 | TRT-LLM | L20；单并发；客户端／服务器 | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice2.DiT.md#benchmark-with-client-server-mode) |
| VoxCPM2 | 约 0.13 | Nano-vLLM / vLLM-Omni | RTX 4090 | [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM#-highlights) |
| Fish Audio S2 | 0.195 | 基于 SGLang 的推理引擎 | H200；单卡 | [Fish Audio S2](https://github.com/fishaudio/fish-speech#performance) |
| IndexTTS 2.5 | 0.2065 | 2.5 BF16；KV 缓存 | RTX 4090；整体 | [index-tts/index-tts](https://github.com/index-tts/index-tts#-inference-speed) |
| Qwen3-TTS-12Hz-0.6B | 0.288 | vLLM V0；并发数 1 | 单加速器；CUDA Graph | [Qwen3-TTS 技术报告](https://arxiv.org/abs/2601.15621) |
| IndexTTS 2.0 | 0.3257 | 2.0 FP16；KV 缓存 | RTX 4090；整体 | [index-tts/index-tts](https://github.com/index-tts/index-tts#-inference-speed) |

AnifLive-TTS 会预先封装音色配置并加载参考音频特征，适合长时间运行的本地声音克隆服务。用户可在目标 GPU 上重建 TensorRT 引擎，并通过可复现的测试验证音质。

## 音质一致性验证

> [!NOTE]
> **音质验收口径**　每个最终流式路径都会在相同随机种子和采样设置下，与完整 WAV 路径比较。这些客观回归检查不能替代主观 MOS 评测。

| 音色包 | 波形相关系数 | Log-mel 余弦相似度 | 说话者余弦相似度 | 时长差 | 结果 |
|---|---:|---:|---:|---:|---:|
| Miku V2ProPlus | 0.859774 | 0.992814 | 0.983066 | 0.000% | 通过 |
| Roxy V2ProPlus | 0.853972 | 0.990298 | 0.989324 | 0.032% | 通过 |

硬性阈值为 Log-mel 余弦相似度 `>=0.99`、说话者余弦相似度 `>=0.98`，以及时长差 `<=3%`。

在固定 Miku 回归用例中，v1.2 的完整输出在相同随机种子和采样设置下，与 v1.1 完全一致。上表则分别比较两个 v1.2 流式路径与各自的完整 WAV 输出。

## 优化内容与实测边界

- 九段神经网络阶段均通过 TensorRT 11 `execute_async_v3()` 执行。
- 每个音色使用针对其尺寸构建的 GPT 引擎，复用 TensorRT 执行环境和固定 KV 缓冲区，并停用辅助流。
- 采样 CUDA Graph 捕获 softmax、multinomial 和 gather，并保留随机数生成语义。
- 只有第一个文本分段使用既有的 9+8 语义标记预览；后续分段沿用原生完整上下文补充路径。
- 每 2 步批量检查 EOS，并将运行环境的热状态保留 25 秒。
- 使用 HTTP/1.1 持续连接并在启动时预热。

完整 GPT 步骤 CUDA Graph 目前受 TensorRT 图捕获兼容性限制，详见[性能工程记录](docs/PERFORMANCE_ENGINEERING.md)。

## v1.2 评估过的架构

| 候选方案 | 结果 | 决定 |
|---|---|---|
| Transformer + TensorRT 运行优化 | 通过端到端延迟及音质门槛 | 采用 |
| MTP-4 | 未来标记预测准确度未通过语义音质门槛 | 不采用 |
| Mamba-2 混合架构 | 端到端收益不足以抵消音质与复杂度代价 | 不采用 |
| Mamba-2 混合架构 + MTP | 未通过综合音质与性能门槛 | 不采用 |

AnifLive-TTS 不会只凭理论计算量采用新架构。实验语义后端必须在不牺牲
语音音质的前提下优于正式运行基线，才会进入正式运行路径。详见
[v1.2 语义架构实验记录](docs/research/v1.2-semantic-experiments.md)。

## 架构

AnifLive-TTS 是 AnifEngine-Voice 的第一方 FP16 TensorRT 11 语音推理平台；
v1 首个完成验证的声学后端是 `gsv-v2proplus`，后续将沿用相同 API 与模型封装格式，扩展至更多 GPT-SoVITS 模型世代。
Python 负责 API、五语文本处理、模型封装、转换工具和 GPT AR 调度；
CUDA/TensorRT 负责九个模型的执行、GPU 采样与缓冲区复用。每个进程只会预先加载一个模型。

- 语言：普通话(`zh`)、粤语／广东话(`yue`)、英语(`en`)、日语(`ja`)、韩语(`ko`)；旧版兼容接口另接受 `auto`、`auto_yue`。
- 标准 API：`POST /v1/audio/speech`。
- 状态查询：`GET /health`、`/v1/capabilities`、`/v1/models`、`/v1/voices`。
- 模型选择：`POST /v1/models/activate` 会先卸载当前模型包，再加载一个兼容的本地模型包。
- `stream=false` 返回单声道 PCM16 WAV；`stream=true` 返回 PCM16 音频块。

## 快速开始

已有 GPT-SoVITS 声音克隆模型的用户，可直接将 GPT／SoVITS 模型检查点与参考音频转换为 AnifLive-TTS 模型包。

### 1. 安装本地工具

安装任何软件包之前，先创建项目专用虚拟环境。API 启动时不会安装依赖或下载共享资源。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements\torch-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\base.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\tensorrt11-cu12.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\setup_shared_assets.py `
  --output D:\models\shared --accept-third-party-licenses
```

如使用 CUDA 12.6 配置，请将 `requirements\torch-cu128.txt` 替换为
`requirements\torch-cu126.txt`。接受共享资源许可之前，请先阅读
[第三方许可声明](THIRD_PARTY_NOTICES.md)。

### 2. 模型转换

```powershell
.\.venv\Scripts\aniflive-tts.exe model convert `
  --gpt D:\models\voice.ckpt --sovits D:\models\voice.pth `
  --reference-audio D:\models\reference.wav `
  --reference-text-file D:\models\reference.txt --reference-language ja `
  --model-id my-v2proplus --voice-profile default `
  --shared-dir D:\models\shared --output D:\models\my-v2proplus
```

转换流程包括原始模型检查、ONNX 导出、FP16 转换、TensorRT 11 计算图修补，
针对常用输入尺寸构建优化引擎，加载引擎并执行实际推理验证，最后以原子方式发布模型包。

> [!IMPORTANT]
> 安全模式使用 `torch.load(weights_only=True)`；只有可信的本地模型检查点才可使用 `--allow-unsafe-pickle`。

引擎指纹包含 TensorRT、CUDA 运行环境、
GPU 计算能力、ONNX、优化设置和构建参数。

### 3. 引擎验证

> [!TIP]
> 每个新模型包完成转换后，先使用 `--enqueue` 执行真实 TensorRT 推理验证，再启动 API。

```powershell
.\.venv\Scripts\aniflive-tts.exe validate `
  --model-package D:\models\my-v2proplus `
  --shared-dir D:\models\shared `
  --enqueue
```

### 4. Docker API

```powershell
Copy-Item .env.example .env
.\scripts\run_docker.ps1 -CudaProfile cu128 -Build
```

首次使用 `-Build` 构建镜像；后续直接启动即可。模型与缓存目录通过主机绑定挂载保存，详见[部署指南](docs/DEPLOYMENT_ZH.md)。

默认 Docker 配置只在本地回环接口公开 API。AnifLive-TTS 不内置公网身份验证；
需要从外部访问时，必须置于具备身份验证和请求限制的反向代理之后。

## API

```powershell
curl.exe -X POST "http://127.0.0.1:9882/v1/audio/speech" `
  -H "Content-Type: application/json" `
  --output output.wav `
  --data '{"model":"my-v2proplus","voice_profile":"default","text":"今日はいい天気ですね。","language":"ja","stream":false,"generation":{"top_k":15,"top_p":1.0,"temperature":1.0,"seed":1234}}'
```

| API 路径 | 用途 |
|---|---|
| `POST /v1/audio/speech` | 标准 TTS 与 OpenAI 兼容接口 |
| `GET/POST /` | GPT-SoVITS 旧版兼容接口 |
| `GET /health` | 服务状态、GPU、CUDA、TensorRT 与实际引擎执行信息 |
| `GET /v1/capabilities` | 语言、流式输出和情感控制能力 |
| `GET /v1/models` | 启用模型 |
| `POST /v1/models/activate` | 切换到兼容的本地模型包 |
| `GET /v1/voices` | 启动时缓存的音色配置 |

## 路线图

**v1**

- [x] V2ProPlus 模型转换与九段 TensorRT 11 推理
- [x] 五语 API、完整 WAV 与低延迟 PCM 流式输出
- [x] Docker 发行版本、音质一致性验证与完全离线的推理流程

**下一阶段：神经情感适配器**

- [ ] 实现可控情感、强度和风格调节，并通过音色保真及延迟验收

**后续阶段：更多 GPT-SoVITS 模型世代**

- [ ] 沿用同一 API 与模型封装格式，支持 V2 / V2Pro、V3 与 V4

## 兼容性限制

> [!WARNING]
> **部署前请注意**　`cu128` 已完成本机 GPU 端到端验收；`cu126` 兼容配置目前仅完成源代码及构建规则验证，必须等待镜像通过发布流程并在兼容主机完成验收后，才可宣称 GPU 端到端支持。

v1 当前支持 V2ProPlus；其他 GPT-SoVITS 版本仍在路线图中。RTX 50 系列／Blackwell 请使用 `cu128`；`cu126` 尚未在 RTX 5070 Ti 通过端到端 GPU 验证，详见[部署指南](docs/DEPLOYMENT_ZH.md)。

## 文档与许可证

- [API 规范](docs/API_ZH.md)
- [部署指南](docs/DEPLOYMENT_ZH.md)
- [性能工程记录](docs/PERFORMANCE_ENGINEERING.md)
- [验收报告](docs/ACCEPTANCE.md)
- [回滚方案](docs/ROLLBACK.md)
- [许可说明](LICENSING.md)
- [第三方许可声明](THIRD_PARTY_NOTICES.md)
- [第三方媒体声明](assets/THIRD_PARTY_MEDIA.md)

AnifLive-TTS 是 AnifEngine-Voice 的第一方 TTS。其当前声学实现建立在 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)、[GPT-SoVITS Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference) 和 [GPT-SoVITS C++](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp) 的研究与工程成果之上。特别感谢 GPT-SoVITS 原作者 **花儿不哭** 及其他 GPT-SoVITS 贡献者。

AnifLive-TTS 原创代码采用 [PolyForm Noncommercial 1.0.0](LICENSE) 许可证；商业使用须另行取得 Hiruynk 的书面商业许可证。GPT-SoVITS 衍生部分保留 MIT，Minimal Inference 衍生部分及适用的 GPT-SoVITS C++ 参考部分保留 Apache-2.0；第三方依赖适用各自条款。详见 [许可说明](LICENSING.md)、[第三方许可声明](THIRD_PARTY_NOTICES.md)，以及 v1.2.0 Release 附带的 cu126／cu128 镜像衍生 SPDX SBOM。
