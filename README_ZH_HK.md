<div align="center">

<img src="assets/everynight_dance.gif" alt="Evernight dance" width="260">

# AnifLive-TTS

**面向粵語／廣東話的低延遲、高音質、多語言聲音複製 TTS 推理系統**

[![版本](https://img.shields.io/badge/%E7%89%88%E6%9C%AC-v1.3.0-2563eb?style=flat-square)](RELEASE_NOTES_v1.3.0.md)
[![TensorRT](https://img.shields.io/badge/TensorRT-11.2.1.2-76b900?style=flat-square&logo=nvidia)](https://docs.nvidia.com/deeplearning/tensorrt/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![模型](https://img.shields.io/badge/%E6%A8%A1%E5%9E%8B-GPT--SoVITS_V2ProPlus-0f766e?style=flat-square)](https://github.com/RVC-Boss/GPT-SoVITS)
[![授權](https://img.shields.io/badge/%E6%8E%88%E6%AC%8A-PolyForm_Noncommercial_1.0.0-22c55e?style=flat-square)](LICENSING.md)

**繁體中文** · [English](README.md) · [简体中文](README_ZH_CN.md)

</div>

## 關於項目

AnifLive-TTS 是 AnifEngine-Voice 的第一方 TTS，誕生於一個很實際的需求：在開發 AnifEngine-Voice 時，我一直找不到一個能同時滿足 **粵語／廣東話**、**低延遲**、**高音質**與可自託管、自主管理的 TTS 方案。😮‍💨

因此我選擇以 GPT-SoVITS 為基礎，深入改造其推理底層，將 **低延遲** 與 **高音質** 定為 AnifLive-TTS 的核心目標。🤓👆

v1 首發完整支援 V2ProPlus；未來版本將沿用相同 API 與模型封裝格式，逐步支援更多 GPT-SoVITS 模型世代。

## 核心能力

- 九個神經網路模型皆由 TensorRT 11 透過 `execute_async_v3()` 執行。
- 支援 普通話(`zh`)、粵語／廣東話(`yue`)、英語(`en`)、日語(`ja`)、韓語(`ko`)，並兼容 GPT-SoVITS 傳統 API 格式與 OpenAI API 格式。
- 支援 GPT-SoVITS V2ProPlus 聲音複製模型，可將自訂 GPT／SoVITS 模型檢查點與參考音訊封裝為獨立音色。
- 完整單聲道 PCM16 WAV 與低延遲 PCM16 串流共用同一套推理流程。
- 模型套件可加入經管理員驗證的情感設定，支援全句及分段演繹，無須更改 V2ProPlus 神經網路架構。
- 本機五語 WebUI 支援音色切換、自然語言情感選擇、串流播放及即時延遲數據。
- v1 可用一條命令完成 V2ProPlus 模型檢查、ONNX 匯出、FP16 轉換、TensorRT 引擎建立、實際推理驗證及模型封裝。
- 推理服務啟動前完成模型與引擎準備，`serve` 熱路徑完全離線。

## 《無職轉生》洛琪希·米格路迪亞：V2ProPlus 粵語／廣東話演示

https://github.com/user-attachments/assets/0be5a03d-94a9-4d33-b05f-6bae6fb3cc40

演示中的本機 WebUI 已納入 v1.3；API 就緒後執行 `run_webui.bat` 即可啟動。

## 實測性能

> [!NOTE]
> **環境：** RTX 5070 Ti 16 GB / NVIDIA 驅動程式 596.36 / CUDA 執行環境 12.8 / PyTorch
> 2.7.0+cu128 / TensorRT 11.2.1.2 / FP16

v1.3 正式測試只採用外部 Roxy V2ProPlus 音色套件，並固定測試文字、隨機種子與採樣參數。共執行 10 輪；每輪先預熱 10 次，再測 100 次完整 WAV、100 次新連線串流及 100 次持續連線串流。主值取 10 輪統計值的中位數，範圍反映各輪之間的波動。Miku 存在仍在獨立調查的模型特定串流現象，因此不納入性能主數據。

正式測試均為單併發。新連線數據會為每個請求建立本機 HTTP/1.1 連線；持續連線數據則在每輪重用一條已獨立預熱的連線。首包延遲從送出請求起計，直至用戶端讀取伺服器送出的第一個 PCM 音訊區塊。可聽 TTFA 取最早有效 10 ms 均方根分析幀內，第一個超過 -45 dBFS 的 PCM 取樣點，並受該音訊區塊的實際抵達時間約束；不包含播放裝置延遲。

| 指標 | 10 輪統計值的中位數 | 各輪範圍 |
|---|---:|---:|
| 完整 REST WAV 端到端 P50 | **153.177 ms** | 146.508–166.100 ms |
| 完整 REST WAV 端到端 P95 | **189.548 ms** | 158.981–201.439 ms |
| 伺服器推理 P50 | **148.591 ms** | 142.121–161.275 ms |
| RTF P50 | **0.087680** | 0.083863–0.095077 |
| 串流首包延遲 P50 | **67.077 ms** | 64.866–74.544 ms |
| 串流首包延遲 P95 | **90.853 ms** | 71.753–120.376 ms |
| 持續連線串流首包延遲 P50 | **67.953 ms** | 65.456–76.345 ms |
| 持續連線串流首包延遲 P95 | **92.019 ms** | 81.640–136.052 ms |
| 串流有效音訊 TTFA P50 | **73.702 ms** | 71.491–81.169 ms |
| 串流有效音訊 TTFA P95 | **97.478 ms** | 78.378–127.001 ms |
| 持續連線串流有效音訊 TTFA P50 | **74.578 ms** | 72.081–82.970 ms |
| 持續連線串流有效音訊 TTFA P95 | **98.644 ms** | 88.265–142.677 ms |
| GPU 佔用率 P50 | **53.0%** | 50.5–54% |
| GPU 佔用率 P95 | **56.0%** | 56–57% |

全部 1,000 個完整 WAV、1,000 個新連線串流及 1,000 個持續連線串流請求均回報 `TensorRT-11`，且 `X-PyTorch-Fallback: false`。可機讀摘要見 [`benchmarks/README_BENCHMARK_SUMMARY.json`](benchmarks/README_BENCHMARK_SUMMARY.json)。

`nvidia-smi` 顯示的是取樣區間內的 GPU 佔用率，而非 SM 佔用率。在單併發下，序列化的 GPT 自回歸流程仍是 GPU 佔用率無法接近 100% 的主要原因。

### 重現性能表格

[`scripts/benchmark_readme.py`](scripts/benchmark_readme.py) 是公開性能測試的標準腳本。它輸出的 Markdown 表格與上表完全相同，只包含同樣的 14 個指標，並採用相同的測試內容與統計方法。

對已經運行的本機 API 執行：

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_readme.py `
  --host 127.0.0.1 --port 9881 --locale zh-HK `
  --model roxy-v2proplus `
  --report .\reports\benchmark.json `
  --markdown .\reports\benchmark.md
```

也可以直接在現有 Docker 容器內執行，不會重新建立或重建容器：

```powershell
docker exec aniflive-tts /app/scripts/entrypoint.sh benchmark `
  --host 127.0.0.1 --port 9880 --locale zh-HK `
  --model roxy-v2proplus `
  --report /data/reports/benchmark.json `
  --markdown /data/reports/benchmark.md
```

預設為每個音色執行 10 輪；每輪先預熱 10 次，再測 100 次完整 WAV、100 次新連線串流及 100 次持續連線串流。正式發行表格只使用 `roxy-v2proplus`，所有測試均為單併發。

### GPT-SoVITS 性能數據對比

#### 首輸出延遲（越低越快）

| 專案／系統 | 指標 | 延遲 | 測試條件 | 來源 |
|---|---|---:|---|---|
| **AnifLive-TTS v1.3** | **可聽 TTFA P50** | **74.578 ms** 🚀 | **RTX 5070 Ti；HTTP/1.1 持續連線；10 輪 Roxy 測試** | **[本機實測](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| GPT-SoVITS C++ TRT 串流 | 首包 | 460 ms | RTX 2080 Ti 22 GB | [GPT-SoVITS C++](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference ONNX 串流 | 首個 token | 1,000 ms | RTX 2080 Ti 22 GB；FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference TRT 固定尺寸最佳化版 | 首個語意標記 | 2,022 ms | RTX 2080 Ti 22 GB；FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |

#### RTF（越低越快）

| 專案／系統 | RTF | 後端 | 測試條件 | 來源 |
|---|---:|---|---|---|
| GPT-SoVITS V2ProPlus | 0.014 | PyTorch 平行推理 | RTX 4090；約 4 分鐘長文 | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS#features) |
| GPT-SoVITS V2ProPlus | 0.028 | PyTorch 平行推理 | RTX 4060 Ti | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS#features) |
| **AnifLive-TTS v1.3** | **0.087680** | **TensorRT 11 FP16** | **RTX 5070 Ti；10 輪 Roxy 測試** | **[本機實測](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| GPT-SoVITS C++ TRT | 0.1020 | TensorRT | RTX 2080 Ti 22 GB | [GPT-SoVITS C++](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference TRT 固定尺寸最佳化版 | 0.2096 | TensorRT；針對固定尺寸最佳化 | RTX 2080 Ti 22 GB；FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |

### 其他開源 TTS 的公開性能數據

以下並非同條件的受控基準測試。除 AnifLive-TTS 外，數據均由各來源自行公布；GPU、模型能力、輸入內容、首包大小、併發度及測量方法並不相同。表格只整理相同指標的公開數據，不代表同條件排名。

#### 首音訊延遲（越低越快）

| 系統 | 指標 | 延遲 | 統計口徑 | 測試條件 | 來源 |
|---|---|---:|---|---|---|
| **AnifLive-TTS v1.3** | **可聽 TTFA** | **74.578 ms** 🚀 | **P50** | **RTX 5070 Ti；HTTP/1.1 持續連線；10 輪 Roxy 測試** | **[本機實測](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| Qwen3-TTS-12Hz-0.6B | 首包延遲 | 97 ms | 單併發 | 單加速器；320 ms 語音包 | [Qwen3-TTS 技術報告](https://arxiv.org/abs/2601.15621) |
| Fish Audio S2 | TTFA | 約 100 ms | 專案發布值 | H200；單卡 | [Fish Audio S2](https://github.com/fishaudio/fish-speech#performance) |
| Chatterbox-Flash（D=32，α=0.75） | TTFP | 103 ms | 單併發；50 句 | H100 | [Chatterbox-Flash 論文](https://arxiv.org/abs/2605.30748) |
| Chatterbox-Flash（預設 D=16，α=0.5） | TTFP | 118 ms | 單併發；50 句 | H100 | [Chatterbox-Flash 論文](https://arxiv.org/abs/2605.30748) |
| CosyVoice2 | 首個區塊 | 196.13 ms | P50 | L20；單併發；用戶端／伺服器 | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice2.DiT.md#benchmark-with-client-server-mode) |

IndexTTS 2.0／2.5 與 VoxCPM2 未提供同口徑的首音訊延遲數值。

#### RTF（越低越快）

| 系統 | RTF | 推理後端／模型 | 測試條件 | 來源 |
|---|---:|---|---|---|
| Chatterbox-Flash（D=32，α=0.75） | 0.076 | 區塊擴散 | H100；單併發；50 句 | [Chatterbox-Flash 論文](https://arxiv.org/abs/2605.30748) |
| **AnifLive-TTS v1.3** | **0.087680** | **TensorRT 11 FP16** | **RTX 5070 Ti；10 輪 Roxy 測試** | **[本機實測](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| Chatterbox-Flash（預設 D=16，α=0.5） | 0.107 | 區塊擴散 | H100；單併發；50 句 | [Chatterbox-Flash 論文](https://arxiv.org/abs/2605.30748) |
| CosyVoice3 | 0.1091 | TRT-LLM；離線批次 1 | L20 | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice3.md#benchmark-with-offline-inference-mode) |
| CosyVoice2 | 0.1228 | TRT-LLM | L20；單併發；用戶端／伺服器 | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice2.DiT.md#benchmark-with-client-server-mode) |
| VoxCPM2 | 約 0.13 | Nano-vLLM / vLLM-Omni | RTX 4090 | [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM#-highlights) |
| Fish Audio S2 | 0.195 | SGLang 推理引擎 | H200；單卡 | [Fish Audio S2](https://github.com/fishaudio/fish-speech#performance) |
| IndexTTS 2.5 | 0.2065 | 2.5 BF16；KV 快取 | RTX 4090；整體 | [index-tts/index-tts](https://github.com/index-tts/index-tts#-inference-speed) |
| Qwen3-TTS-12Hz-0.6B | 0.288 | vLLM V0；單併發 | 單加速器；CUDA Graph | [Qwen3-TTS 技術報告](https://arxiv.org/abs/2601.15621) |
| IndexTTS 2.0 | 0.3257 | 2.0 FP16；KV 快取 | RTX 4090；整體 | [index-tts/index-tts](https://github.com/index-tts/index-tts#-inference-speed) |

AnifLive-TTS 會預先封裝音色設定並載入參考音訊特徵，適合長時間運行的本機聲音複製服務。使用者可在目標 GPU 上重建 TensorRT 引擎，並透過可重現的測試驗證音質。

## 音質一致性驗證

> [!NOTE]
> **音質驗收口徑**　每段受控情感串流都會在相同隨機種子與設定下，與其完整 WAV 輸出比較；中性完整 WAV、串流 PCM 及語意輸出亦會對照不可變的 v1.2 基準。這些客觀檢查不能取代主觀聆聽。

| 音色套件 | 日語情感測試 | 最低 Log-mel 相似度 | 最低說話者相似度 | 最大時長差 | 緩衝中斷 |
|---|---:|---:|---:|---:|---:|
| Miku V2ProPlus | 18/18 | 0.999454 | 0.989519 | 0.302% | 0 |
| Roxy V2ProPlus | 18/18 | 0.994325 | 0.987721 | 0.300% | 0 |

以上項目只描述本機私有驗證 overlay；相關情感參考音訊、逐字稿及角色媒體不會
包含於公開原始碼、映像或發行套件。

五語長短句驗收中，每個音色均通過 20/20 項。受控情感的最低 Log-mel／說話者相似度，Miku 為 `0.999660`／`0.988638`，Roxy 為 `0.998827`／`0.993495`。兩個音色在 `zh`、`yue`、`en`、`ja`、`ko` 的中性完整 WAV、串流 PCM、完整語意及串流語意均與 v1.2 完全一致。

六組短句、長句及混合情感盲聽中，五組判定無明顯差異，一組偏好 Roxy 完整輸出，沒有發現音訊瑕疵。硬性門檻維持 Log-mel 相似度 `>=0.99`、說話者相似度 `>=0.98` 及時長差 `<=3%`。

## 優化內容與已測邊界

- 九段神經網路階段均由 TensorRT 11 `execute_async_v3()` 執行。
- 每個音色使用針對其尺寸建立的 GPT 引擎，重用 TensorRT 執行環境與固定 KV 緩衝區，並停用輔助串流。
- 採樣的 `softmax`、`multinomial` 與 `gather` 使用 CUDA Graph，保留 PyTorch RNG 語義。
- 只有第一個文字分段採用既有的 9+8 語意標記預覽；後續分段沿用原生完整上下文補充路徑。
- 每 2 步批次檢查 EOS，並將執行環境的熱狀態保留 25 秒。
- 情感參考資料在模型啟用時完成準備並常駐 GPU；中性請求與 v1.2 輸出完全一致。
- 情感選擇及轉場設定由模型套件決定，執行環境不會根據 Miku 或 Roxy 名稱分支。
- 使用 HTTP/1.1 持續連線並在啟動時預熱。

完整 GPT-step CUDA Graph 目前受 TensorRT capture 相容性限制，詳見[性能工程紀錄](docs/PERFORMANCE_ENGINEERING.md)。

## v1.2 評估過的架構

| 候選方案 | 結果 | 決定 |
|---|---|---|
| Transformer + TensorRT 執行最佳化 | 通過端到端延遲及音質門檻 | 採用 |
| MTP-4 | 未來標記預測準確度未通過語意音質門檻 | 不採用 |
| Mamba-2 混合架構 | 端到端收益不足以抵銷音質與複雜度代價 | 不採用 |
| Mamba-2 混合架構 + MTP | 未通過綜合音質與性能門檻 | 不採用 |

AnifLive-TTS 不會只憑理論運算量採用新架構。實驗語意後端必須在不犧牲
語音音質的前提下勝過正式版基準，才會進入正式執行路徑。詳見
[v1.2 語意架構實驗紀錄](docs/research/v1.2-semantic-experiments.md)。

## v1.3 評估過的架構

| 候選方案 | 結果 | 決定 |
|---|---|---|
| 參考導向情感控制 | 通過客觀、相容性及盲聽驗收 | 採用 |
| MTP 未來上下文的硬式語意預讀 | 未同時通過跨模型音質及端到端延遲門檻 | 不採用 |
| 循環式推測解碼 | 目標模型執行次數下降，但草稿與驗證成本限制了實際延遲收益 | 不採用 |
| SemanticPiece／EABPE | 序列壓縮未集中於影響首音訊延遲的前 17 個語意標記 | 不採用 |
| FastStart 蒸餾／剪枝 Transformer | 未通過續寫及上下文補充的可靠性門檻 | 不採用 |
| D16 區塊擴散 | 已評估的低秩適配方案未通過語意品質門檻 | 不採用 |

以上結論只適用於本次 AnifLive-TTS V2ProPlus 實作與工作負載，並非否定其他實作方式。詳見 [v1.3 延遲實驗紀錄](docs/research/v1.3-latency-experiments.md)及[情感控制設計紀錄](docs/research/v1.3-reference-expression-design.md)。

## 架構

AnifLive-TTS 是 AnifEngine-Voice 的第一方 FP16 TensorRT 11 語音推理平台；
v1 首個完成驗證的聲學後端是 `gsv-v2proplus`，後續將沿用相同 API 與模型封裝格式，擴展至更多 GPT-SoVITS 模型世代。
Python 負責 API、五語文字處理、模型封裝、轉換工具與 GPT AR 排程；CUDA／TensorRT
負責九個模型的執行、GPU 採樣與緩衝區重用。每個處理程序只會預先載入一個模型。

- 語言：普通話(`zh`)、粵語／廣東話(`yue`)、英語(`en`)、日語(`ja`)、韓語(`ko`)；舊版相容介面另接受 `auto`、`auto_yue`。
- 標準 API：`POST /v1/audio/speech`。
- 狀態查詢：`GET /health`、`/v1/capabilities`、`/v1/models`、`/v1/voices`。
- 情感資料：`GET /v1/expressions`；全句及分段情感請求均使用標準語音 API。
- 模型選擇：`POST /v1/models/activate` 會先卸載目前套件，再載入一個相容的本機模型套件。
- 取消串流：`POST /v1/audio/cancel` 會在下一個請求前釋放已放棄的串流。
- `stream=false` 回傳單聲道 PCM16 WAV；`stream=true` 回傳 PCM16 音訊區塊。

## 快速開始

已有 GPT-SoVITS 聲音複製模型的使用者，可直接將 GPT／SoVITS 模型檢查點與參考音訊轉換為 AnifLive-TTS 模型套件。

### 1. 安裝本機工具

安裝任何套件前，先建立專案專用虛擬環境。API 啟動時不會安裝依賴或下載共享資源。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements\torch-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\base.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\tensorrt11-cu12.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\setup_shared_assets.py `
  --output D:\models\shared --accept-third-party-licenses
```

如使用 CUDA 12.6 配置，將 `requirements\torch-cu128.txt` 改為
`requirements\torch-cu126.txt`。接受共享資源授權前，請先閱讀
[第三方授權聲明](THIRD_PARTY_NOTICES.md)。

### 2. 模型轉換

```powershell
.\.venv\Scripts\aniflive-tts.exe model convert `
  --gpt D:\models\voice.ckpt `
  --sovits D:\models\voice.pth `
  --reference-audio D:\models\reference.wav `
  --reference-text-file D:\models\reference.txt `
  --reference-language ja `
  --model-id my-v2proplus `
  --voice-profile default `
  --shared-dir D:\models\shared `
  --output D:\models\my-v2proplus
```

流程包括原始模型檢查、ONNX 匯出、FP16 轉換、TensorRT 11 運算圖修補、
針對常用輸入尺寸建立最佳化引擎、載入引擎並執行實際推理驗證，最後以原子方式發佈模型套件。

> [!IMPORTANT]
> 預設使用 `torch.load(weights_only=True)`；不可信的 Pickle 檔案不應使用 `--allow-unsafe-pickle`。

引擎指紋包含 TensorRT、CUDA 執行環境、GPU 計算能力、
ONNX、最佳化設定與建置參數；更換 GPU 或執行環境後需明確執行
`aniflive-tts model rebuild-engines`。

情感控制是選用功能。公開原始碼、映像及發行套件不會預設任何情感 profile
或參考媒體。請只使用你有權使用的參考音訊，並建立設定檔：

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
      "reference_text": "逐字稿必須與參考音訊完全一致。",
      "reference_language": "yue",
      "manual_verified": true
    }
  ]
}
```

然後將已驗證的參考音訊加入新套件，不會修改原有中性套件：

```powershell
.\.venv\Scripts\aniflive-tts.exe model import-expressions `
  --model-package D:\models\my-v2proplus `
  --voice-profile default `
  --spec-file D:\models\expressions.json `
  --asset-root D:\models\expression-audio `
  --output D:\models\my-v2proplus-expression
```

導入工具只依賴模型套件資料，不會判斷模型名稱，因此所有通過驗證的 V2ProPlus 音色均使用同一套規格。
參考 WAV 應只有一位說話者、聲音乾淨，逐字稿必須準確。導入工具會把
資產複製至新輸出套件並更新 checksum，原有套件維持不變。

### 3. 引擎驗證

> [!TIP]
> 每個新模型套件完成轉換後，先使用 `--enqueue` 執行真實 TensorRT 推理驗證，再啟動 API。

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

首次使用 `-Build` 建立映像；後續直接啟動即可。模型與快取目錄以主機綁定掛載保存，詳見[部署指南](docs/DEPLOYMENT_ZH.md)。

預設 Docker 配置只在本機回環介面公開 API。AnifLive-TTS 不內建對外網路身分驗證；
需要從外部存取時，必須置於具備身分驗證與請求限制的反向代理之後。

### 5. 啟動本機 WebUI

```powershell
.\run_webui.bat
```

啟用含情感 profile 的套件後，反白一個完整子句並包含末尾的逗號、句號、
分號、問號或感嘆號，再選擇情感。未標記文字會沿用音色原生的中性表達；
沒有匯入 profile 的套件不會啟用情感選項。

## API

```powershell
curl.exe -X POST "http://127.0.0.1:9882/v1/audio/speech" `
  -H "Content-Type: application/json" `
  --output output.wav `
  --data '{"model":"my-v2proplus","voice_profile":"default","text":"今日はいい天気ですね。","language":"ja","stream":false,"generation":{"top_k":15,"top_p":1.0,"temperature":1.0,"seed":1234}}'
```

全句情感可使用 `GET /v1/expressions` 回傳的 symbolic profile：

```json
{
  "model": "my-v2proplus",
  "text": "今日天氣好好。",
  "language": "yue",
  "stream": true,
  "expression": {
    "enabled": true,
    "profile": "gentle",
    "intensity": 0.7,
    "policy": "semantic-style"
  }
}
```

多段情感必須使用以安全標點結束的完整子句。伺服器會拒絕詞語中間的
情感切換，避免漏字或發音不清：

```json
{
  "model": "my-v2proplus",
  "language": "yue",
  "stream": true,
  "segments": [
    {
      "text": "雖然有啲擔心，",
      "expression": {"enabled": true, "profile": "shy", "intensity": 0.6}
    },
    {
      "text": "但我已經準備好喇。",
      "expression": {"enabled": true, "profile": "confident", "intensity": 0.8}
    }
  ]
}
```

| API 路徑 | 用途 |
|---|---|
| `POST /v1/audio/speech` | 標準 TTS 與 OpenAI 相容介面 |
| `GET/POST /` | GPT-SoVITS 舊版相容介面 |
| `GET /health` | 服務狀態、GPU、CUDA、TensorRT 與實際引擎執行資訊 |
| `GET /v1/capabilities` | 語言、串流與情感控制能力 |
| `GET /v1/models` | 目前啟用模型 |
| `POST /v1/models/activate` | 切換至相容的本機模型套件 |
| `GET /v1/voices` | 啟動時快取的語音設定檔 |
| `GET /v1/expressions` | 模型套件提供的情感設定及策略 |
| `POST /v1/audio/cancel` | 取消目前串流 |

## 路線圖

**v1.3**

- [x] V2ProPlus 模型轉換與九段 TensorRT 11 推理
- [x] 五語 API、完整 WAV、低延遲 PCM 串流及 Docker 發行
- [x] 模型套件情感設定、分段演繹及本機 WebUI

**下一階段：神經情感適配器**

- [ ] 實作可控情感、強度與風格調節，並通過音色保真及延遲驗收

**後續階段：更多 GPT-SoVITS 模型世代**

- [ ] 沿用同一 API 與模型封裝格式，支援 V2／V2Pro、V3 與 V4

## 相容性限制

> [!WARNING]
> **部署前請注意**　`cu128` 是本機完成 GPU 端到端驗收的配置；`cu126` 相容配置目前只完成原始碼及建置規則驗證，必須待映像通過發布流程及在相容主機完成驗收後，才可宣稱 GPU 端到端支援。

v1 目前只對 V2ProPlus 作出正式支援承諾；其他 GPT-SoVITS 版本仍在上述路線圖。RTX 50 系列／Blackwell 請使用 `cu128`；`cu126` 未在 RTX 5070 Ti 通過端到端 GPU 驗證，詳見[部署指南](docs/DEPLOYMENT_ZH.md)。

## 文件與授權

- [API 規格](docs/API_ZH.md)
- [部署指南](docs/DEPLOYMENT_ZH.md)
- [性能工程紀錄](docs/PERFORMANCE_ENGINEERING.md)
- [驗收報告](docs/ACCEPTANCE.md)
- [回復方案](docs/ROLLBACK.md)
- [授權說明](LICENSING.md)
- [第三方授權聲明](THIRD_PARTY_NOTICES.md)
- [第三方媒體聲明](assets/THIRD_PARTY_MEDIA.md)

AnifLive-TTS 是 AnifEngine-Voice 的第一方 TTS。其當前聲學實作建立於 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)、[GPT-SoVITS Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference) 與 [GPT-SoVITS C++](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp) 的研究與工程成果之上。特別感謝 GPT-SoVITS 原作者 **花儿不哭** 及其他 GPT-SoVITS 貢獻者。

AnifLive-TTS 原創程式碼採用 [PolyForm Noncommercial 1.0.0](LICENSE) 授權；商業使用須另行取得 Hiruynk 的書面商業授權。GPT-SoVITS 衍生部分保留 MIT，Minimal Inference 衍生部分及適用的 GPT-SoVITS C++ 參考部分保留 Apache-2.0；第三方依賴適用各自條款。詳見 [授權說明](LICENSING.md)及[第三方授權聲明](THIRD_PARTY_NOTICES.md)。
