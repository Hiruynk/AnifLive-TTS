<div align="center">

<img src="assets/everynight_dance.gif" alt="Evernight dance" width="260">

# AnifLive-TTS

**面向粤语／广东话的低延迟、高音质、多语言声音克隆 TTS 推理系统**

[![版本](https://img.shields.io/badge/%E7%89%88%E6%9C%AC-v1.3.0-2563eb?style=flat-square)](RELEASE_NOTES_v1.3.0.md)
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
- 模型包可加入经管理员验证的情感配置，支持整句和分段演绎，无需更改 V2ProPlus 神经网络架构。
- 本地五语言 WebUI 支持音色切换、自然语言情感选择、流式播放和实时延迟数据。
- v1 可用一条命令完成 V2ProPlus 模型检查、ONNX 导出、FP16 转换、TensorRT 引擎构建、实际推理验证及模型封装。
- 推理服务启动前完成模型与引擎准备，`serve` 热路径完全离线。

## 《无职转生》洛琪希·米格路迪亚：V2ProPlus 粤语／广东话演示

https://github.com/user-attachments/assets/0be5a03d-94a9-4d33-b05f-6bae6fb3cc40

演示中的本地 WebUI 已包含在 v1.3 中；API 就绪后运行 `run_webui.bat` 即可启动。

## 实测性能

> [!NOTE]
> **环境：** RTX 5070 Ti 16 GB / NVIDIA 驱动程序 596.36 / CUDA 运行环境 12.8 / PyTorch
> 2.7.0+cu128 / TensorRT 11.2.1.2 / FP16

v1.3 正式测试只采用外部 Roxy V2ProPlus 音色包，并固定测试文本、随机种子和采样参数。共执行 10 轮；每轮先预热 10 次，再测试 100 次完整 WAV、100 次新连接流式输出和 100 次持久连接流式输出。主值取 10 轮统计值的中位数，范围反映各轮之间的波动。Miku 存在仍在单独调查的模型特定流式现象，因此不计入性能主数据。

正式测试均为单并发。新连接数据会为每个请求建立本地 HTTP/1.1 连接；持久连接数据则在每轮复用一条已单独预热的连接。首包延迟从发送请求起计算，直到客户端读取服务器发出的第一个 PCM 音频块。可听 TTFA 取最早有效 10 ms 均方根分析帧内，第一个超过 -45 dBFS 的 PCM 采样点，并受该音频块的实际到达时间约束；不包含播放设备延迟。

| 指标 | 10 轮统计值的中位数 | 各轮范围 |
|---|---:|---:|
| 完整 REST WAV 端到端 P50 | **153.177 ms** | 146.508–166.100 ms |
| 完整 REST WAV 端到端 P95 | **189.548 ms** | 158.981–201.439 ms |
| 服务器推理 P50 | **148.591 ms** | 142.121–161.275 ms |
| RTF P50 | **0.087680** | 0.083863–0.095077 |
| 流式首包延迟 P50 | **67.077 ms** | 64.866–74.544 ms |
| 流式首包延迟 P95 | **90.853 ms** | 71.753–120.376 ms |
| 持久连接流式首包延迟 P50 | **67.953 ms** | 65.456–76.345 ms |
| 持久连接流式首包延迟 P95 | **92.019 ms** | 81.640–136.052 ms |
| 流式有效音频 TTFA P50 | **73.702 ms** | 71.491–81.169 ms |
| 流式有效音频 TTFA P95 | **97.478 ms** | 78.378–127.001 ms |
| 持久连接流式有效音频 TTFA P50 | **74.578 ms** | 72.081–82.970 ms |
| 持久连接流式有效音频 TTFA P95 | **98.644 ms** | 88.265–142.677 ms |
| GPU 占用率 P50 | **53.0%** | 50.5–54% |
| GPU 占用率 P95 | **56.0%** | 56–57% |

全部 1,000 个完整 WAV、1,000 个新连接流式输出和 1,000 个持久连接流式输出请求均报告 `TensorRT-11`，且 `X-PyTorch-Fallback: false`。机器可读摘要见 [`benchmarks/README_BENCHMARK_SUMMARY.json`](benchmarks/README_BENCHMARK_SUMMARY.json)。

`nvidia-smi` 显示的是采样区间内的 GPU 占用率，并非 SM 占用率。在单并发下，串行执行的 GPT 自回归流程仍是 GPU 占用率无法接近 100% 的主要原因。

### 重现性能表格

[`scripts/benchmark_readme.py`](scripts/benchmark_readme.py) 是公开性能测试的标准脚本。它输出的 Markdown 表格与上表完全相同，只包含同样的 14 个指标，并采用相同的测试内容和统计方法。

对已经运行的本地 API 执行：

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_readme.py `
  --host 127.0.0.1 --port 9881 --locale zh-CN `
  --model roxy-v2proplus `
  --report .\reports\benchmark.json `
  --markdown .\reports\benchmark.md
```

也可以直接在现有 Docker 容器内执行，不会重新创建或重建容器：

```powershell
docker exec aniflive-tts /app/scripts/entrypoint.sh benchmark `
  --host 127.0.0.1 --port 9880 --locale zh-CN `
  --model roxy-v2proplus `
  --report /data/reports/benchmark.json `
  --markdown /data/reports/benchmark.md
```

默认对每个音色执行 10 轮；每轮先预热 10 次，再测试 100 次完整 WAV、100 次新连接流式输出和 100 次持久连接流式输出。正式发行表格只使用 `roxy-v2proplus`，所有测试均为单并发。

### GPT-SoVITS 性能数据对比

#### 首次输出延迟（越低越快）

| 项目／系统 | 指标 | 延迟 | 测试条件 | 来源 |
|---|---|---:|---|---|
| **AnifLive-TTS v1.3** | **可听 TTFA P50** | **74.578 ms** 🚀 | **RTX 5070 Ti；HTTP/1.1 持久连接；10 轮 Roxy 测试** | **[本机实测](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| GPT-SoVITS C++ TRT 流式 | 首包 | 460 ms | RTX 2080 Ti 22 GB | [GPT-SoVITS C++](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference ONNX 流式 | 首个 token | 1,000 ms | RTX 2080 Ti 22 GB；FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference TRT 固定尺寸优化版 | 首个语义标记 | 2,022 ms | RTX 2080 Ti 22 GB；FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |

#### RTF（越低越快）

| 项目／系统 | RTF | 后端 | 测试条件 | 来源 |
|---|---:|---|---|---|
| GPT-SoVITS V2ProPlus | 0.014 | PyTorch 并行推理 | RTX 4090；约 4 分钟长文本 | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS#features) |
| GPT-SoVITS V2ProPlus | 0.028 | PyTorch 并行推理 | RTX 4060 Ti | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS#features) |
| **AnifLive-TTS v1.3** | **0.087680** | **TensorRT 11 FP16** | **RTX 5070 Ti；10 轮 Roxy 测试** | **[本机实测](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| GPT-SoVITS C++ TRT | 0.1020 | TensorRT | RTX 2080 Ti 22 GB | [GPT-SoVITS C++](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference TRT 固定尺寸优化版 | 0.2096 | TensorRT；针对固定尺寸优化 | RTX 2080 Ti 22 GB；FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |

### 其他开源 TTS 的公开性能数据

以下并非同等条件下的受控基准测试。除 AnifLive-TTS 外，数据均由各来源自行公布；GPU、模型能力、输入内容、首包大小、并发度及测量方法并不相同。表格只整理相同指标的公开数据，不代表同等条件下的排名。

#### 首音频延迟（越低越快）

| 系统 | 指标 | 延迟 | 统计口径 | 测试条件 | 来源 |
|---|---|---:|---|---|---|
| **AnifLive-TTS v1.3** | **可听 TTFA** | **74.578 ms** 🚀 | **P50** | **RTX 5070 Ti；HTTP/1.1 持久连接；10 轮 Roxy 测试** | **[本机实测](benchmarks/README_BENCHMARK_SUMMARY.json)** |
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
| **AnifLive-TTS v1.3** | **0.087680** | **TensorRT 11 FP16** | **RTX 5070 Ti；10 轮 Roxy 测试** | **[本机实测](benchmarks/README_BENCHMARK_SUMMARY.json)** |
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
> **音质验收口径**　每段受控情感流式输出都会在相同随机种子和设置下，与其完整 WAV 输出比较；中性完整 WAV、流式 PCM 和语义输出也会对照不可变的 v1.2 基准。这些客观检查不能替代主观试听。

| 音色包 | 日语情感测试 | 最低 Log-mel 相似度 | 最低说话者相似度 | 最大时长差 | 缓冲中断 |
|---|---:|---:|---:|---:|---:|
| Miku V2ProPlus | 18/18 | 0.999454 | 0.989519 | 0.302% | 0 |
| Roxy V2ProPlus | 18/18 | 0.994325 | 0.987721 | 0.300% | 0 |

以上项目只描述本地私有验证 overlay；相关情感参考音频、逐字稿和角色媒体不会
包含在公开源代码、镜像或发行包中。

五语言长短句验收中，每个音色均通过 20/20 项。受控情感的最低 Log-mel／说话者相似度，Miku 为 `0.999660`／`0.988638`，Roxy 为 `0.998827`／`0.993495`。两个音色在 `zh`、`yue`、`en`、`ja`、`ko` 的中性完整 WAV、流式 PCM、完整语义和流式语义均与 v1.2 完全一致。

六组短句、长句和混合情感盲听中，五组判定无明显差异，一组偏好 Roxy 完整输出，没有发现音频瑕疵。硬性阈值保持为 Log-mel 相似度 `>=0.99`、说话者相似度 `>=0.98` 和时长差 `<=3%`。

## 优化内容与实测边界

- 九段神经网络阶段均通过 TensorRT 11 `execute_async_v3()` 执行。
- 每个音色使用针对其尺寸构建的 GPT 引擎，复用 TensorRT 执行环境和固定 KV 缓冲区，并停用辅助流。
- 采样 CUDA Graph 捕获 softmax、multinomial 和 gather，并保留随机数生成语义。
- 只有第一个文本分段使用既有的 9+8 语义标记预览；后续分段沿用原生完整上下文补充路径。
- 每 2 步批量检查 EOS，并将运行环境的热状态保留 25 秒。
- 情感参考数据在模型启用时完成准备并常驻 GPU；中性请求与 v1.2 输出完全一致。
- 情感选择和转场设置由模型包决定，运行环境不会根据 Miku 或 Roxy 名称分支。
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

## v1.3 评估过的架构

| 候选方案 | 结果 | 决定 |
|---|---|---|
| 基于 MTP 未来上下文的硬式语义预读 | 未同时通过跨模型音质和端到端延迟门槛 | 不采用 |
| 循环式推测解码 | 目标模型执行次数下降，但草稿和验证成本限制了实际延迟收益 | 不采用 |
| SemanticPiece／EABPE | 序列压缩未集中在影响首音频延迟的前 17 个语义标记 | 不采用 |
| FastStart 蒸馏／剪枝 Transformer | 未通过续写和上下文补充的可靠性门槛 | 不采用 |
| 参考导向情感控制 | 通过客观、兼容性和盲听验收 | 采用 |

以上结论只适用于本次 AnifLive-TTS V2ProPlus 实现和工作负载，并非否定其他实现方式。详见 [v1.3 延迟实验记录](docs/research/v1.3-latency-experiments.md)和[情感控制设计记录](docs/research/v1.3-reference-expression-design.md)。

## 架构

AnifLive-TTS 是 AnifEngine-Voice 的第一方 FP16 TensorRT 11 语音推理平台；
v1 首个完成验证的声学后端是 `gsv-v2proplus`，后续将沿用相同 API 与模型封装格式，扩展至更多 GPT-SoVITS 模型世代。
Python 负责 API、五语文本处理、模型封装、转换工具和 GPT AR 调度；
CUDA/TensorRT 负责九个模型的执行、GPU 采样与缓冲区复用。每个进程只会预先加载一个模型。

- 语言：普通话(`zh`)、粤语／广东话(`yue`)、英语(`en`)、日语(`ja`)、韩语(`ko`)；旧版兼容接口另接受 `auto`、`auto_yue`。
- 标准 API：`POST /v1/audio/speech`。
- 状态查询：`GET /health`、`/v1/capabilities`、`/v1/models`、`/v1/voices`。
- 情感信息：`GET /v1/expressions`；整句和分段情感请求均使用标准语音 API。
- 模型选择：`POST /v1/models/activate` 会先卸载当前模型包，再加载一个兼容的本地模型包。
- 取消流式输出：`POST /v1/audio/cancel` 会在下一个请求前释放已放弃的流式请求。
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

情感控制是可选功能。公开源代码、镜像和发行包不预设任何情感 profile
或参考媒体。请只使用你有权使用的参考音频，并创建配置文件：

```json
{
  "schema": 1,
  "default_profile": "neutral",
  "preferred_policy": "semantic-style",
  "profiles": [
    {
      "id": "gentle-1",
      "emotion": "gentle",
      "intensity": 0.7,
      "reference_audio": "gentle.wav",
      "reference_text": "逐字稿必须与参考音频完全一致。",
      "reference_language": "zh",
      "manual_verified": true
    }
  ]
}
```

然后把已验证的参考音频加入新模型包，不会修改原有中性模型包：

```powershell
.\.venv\Scripts\aniflive-tts.exe model import-expressions `
  --model-package D:\models\my-v2proplus `
  --voice-profile default `
  --spec-file D:\models\expressions.json `
  --asset-root D:\models\expression-audio `
  --output D:\models\my-v2proplus-expression
```

导入工具只依赖模型包信息，不会判断模型名称，因此所有通过验证的 V2ProPlus 音色均使用同一套规范。
参考 WAV 应只有一位说话者且声音干净，逐字稿必须准确。导入工具会把
资产复制到新的输出模型包并更新 checksum，原模型包保持不变。

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

### 5. 启动本地 WebUI

```powershell
.\run_webui.bat
```

启用带有情感 profile 的模型包后，高亮一个完整子句并包含末尾的逗号、
句号、分号、问号或感叹号，再选择情感。未标记文字会沿用音色原生的中性
表达；没有导入 profile 的模型包不会启用情感选项。

## API

```powershell
curl.exe -X POST "http://127.0.0.1:9882/v1/audio/speech" `
  -H "Content-Type: application/json" `
  --output output.wav `
  --data '{"model":"my-v2proplus","voice_profile":"default","text":"今日はいい天気ですね。","language":"ja","stream":false,"generation":{"top_k":15,"top_p":1.0,"temperature":1.0,"seed":1234}}'
```

整句情感可以使用 `GET /v1/expressions` 返回的 symbolic profile：

```json
{
  "model": "my-v2proplus",
  "text": "今天天气很好。",
  "language": "zh",
  "stream": true,
  "expression": {
    "enabled": true,
    "profile": "gentle",
    "intensity": 0.7,
    "policy": "semantic-style"
  }
}
```

多段情感必须使用以安全标点结束的完整子句。服务器会拒绝词语中间的
情感切换，避免漏字或发音不清：

```json
{
  "model": "my-v2proplus",
  "language": "zh",
  "stream": true,
  "segments": [
    {
      "text": "虽然有点担心，",
      "expression": {"enabled": true, "profile": "shy", "intensity": 0.6}
    },
    {
      "text": "但我已经准备好了。",
      "expression": {"enabled": true, "profile": "confident", "intensity": 0.8}
    }
  ]
}
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
| `GET /v1/expressions` | 模型包提供的情感配置和策略 |
| `POST /v1/audio/cancel` | 取消当前流式请求 |

## 路线图

**v1.3**

- [x] V2ProPlus 模型转换与九段 TensorRT 11 推理
- [x] 五语言 API、完整 WAV、低延迟 PCM 流式输出和 Docker 发行
- [x] 模型包情感配置、分段演绎和本地 WebUI

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

AnifLive-TTS 原创代码采用 [PolyForm Noncommercial 1.0.0](LICENSE) 许可证；商业使用须另行取得 Hiruynk 的书面商业许可证。GPT-SoVITS 衍生部分保留 MIT，Minimal Inference 衍生部分及适用的 GPT-SoVITS C++ 参考部分保留 Apache-2.0；第三方依赖适用各自条款。详见 [许可说明](LICENSING.md)和[第三方许可声明](THIRD_PARTY_NOTICES.md)。
