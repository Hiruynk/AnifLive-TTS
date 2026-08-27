# AnifLive-TTS v1.2.0

## English

AnifLive-TTS v1.2 improves interactive latency and complete-generation speed
while preserving the established FP16 TensorRT 11 acoustic path and v1 API.

### Highlights

- Adds fitted GPT engines for each compatible voice package, with persistent
  TensorRT execution contexts, fixed KV buffers, and zero auxiliary streams.
- Keeps semantic sampling on the GPU through a reusable CUDA Graph while
  preserving the existing seed and sampling contract.
- Restricts the 9+8 semantic-token preview to the first text segment; later
  segments retain native full-context refill for stable long-form delivery.
- Retains batched two-step EOS checks and the guarded 25-second warm window.
- Validates the release with interleaved Miku and Roxy model sessions rather
  than reporting a result from one voice package.
- Restores the default repetition penalty to `1.0`; the fixed Miku complete-WAV
  regression case is byte-identical to v1.1 at the same seed and settings.
- Keeps the Transformer semantic runtime as the production backend. MTP,
  Viterbi, Mamba-2, full GPT-step CUDA Graph, and EOS4 experiments did not pass
  their combined performance and quality gates and are not enabled at runtime.

The machine-readable benchmark and gate reports are available in
[`benchmarks/README_BENCHMARK_SUMMARY.json`](benchmarks/README_BENCHMARK_SUMMARY.json)
and [`benchmarks/V1_2_RELEASE_GATE.json`](benchmarks/V1_2_RELEASE_GATE.json).

### Runtime

- API version: `1.2.0`
- Backend: TensorRT 11, FP16
- Languages: `zh`, `yue`, `en`, `ja`, `ko`
- Release container tag: `ghcr.io/hiruynk/aniflive-tts:1.2.0-cu128`
- Compatibility profile: `cu126`; publish its tag only after release-workflow
  and target-host validation

The v1 API and schema-1 model packages remain compatible. Model packages and
TensorRT engines remain external runtime assets and must match the target GPU,
CUDA runtime, and TensorRT fingerprint.

Original AnifLive-TTS code remains licensed under PolyForm Noncommercial
1.0.0. Commercial use requires a separate written Commercial License from
Hiruynk; upstream and third-party components retain their respective licenses.

## 繁體中文

AnifLive-TTS v1.2 在維持既有 FP16 TensorRT 11 聲學路徑及 v1 API 的同時，
進一步降低互動延遲及完整語音生成時間。

### 主要更新

- 為每個相容音色套件建立針對其尺寸的 GPT 引擎，重用 TensorRT 執行環境與
  固定 KV 緩衝區，並停用輔助串流。
- 以可重用 CUDA Graph 在 GPU 上完成語意採樣，同時保留既有隨機種子與採樣規格。
- 只有第一個文字分段使用 9+8 語意標記預覽；後續分段沿用原生完整上下文補充路徑，
  維持長句穩定性。
- 保留每兩步批次檢查 EOS，以及受保護的 25 秒熱狀態維持。
- 正式數據採用 Miku 與 Roxy 交錯模型輪次，不再只以單一音色套件作為代表。
- 將預設重複懲罰恢復為 `1.0`；固定 Miku 完整 WAV 回歸案例在相同隨機種子及
  設定下，與 v1.1 完全一致。
- production 繼續使用穩定的 Transformer 語意後端。MTP、Viterbi、Mamba-2、
  完整 GPT 步驟 CUDA Graph 及 EOS4 實驗未同時通過性能與音質門檻，因此不會在
  執行環境啟用。

可機讀性能與驗收結果見
[`benchmarks/README_BENCHMARK_SUMMARY.json`](benchmarks/README_BENCHMARK_SUMMARY.json)
及 [`benchmarks/V1_2_RELEASE_GATE.json`](benchmarks/V1_2_RELEASE_GATE.json)。

### 執行環境

- API 版本：`1.2.0`
- 後端：TensorRT 11、FP16
- 語言：`zh`、`yue`、`en`、`ja`、`ko`
- 發布容器標籤：`ghcr.io/hiruynk/aniflive-tts:1.2.0-cu128`
- 相容配置：`cu126`；必須通過發布流程與目標主機驗收後才發布其標籤

v1 API 與 schema-1 模型套件保持相容。模型套件及 TensorRT 引擎繼續作為外部
執行資產保存，並須符合目標 GPU、CUDA 執行環境與 TensorRT fingerprint。

AnifLive-TTS 原創程式碼繼續採用 PolyForm Noncommercial 1.0.0；商業使用須另行
取得 Hiruynk 的書面商業授權，上游與第三方組件則維持各自授權。
