# AnifLive-TTS v1.3.0

## English

AnifLive-TTS v1.3 adds package-curated expression control and the local WebUI
while preserving the validated V2ProPlus FP16 TensorRT 11 inference path.

### Highlights

- Adds global and per-segment symbolic expression requests with discrete
  intensity selection.
- Prepares every package-owned expression reference at model activation and
  keeps the neural conditioning resident on the GPU.
- Adds the generic `model import-expressions` administration command; it uses
  model-package metadata and does not branch on a character or model name.
- Includes a five-language local WebUI with model switching, natural-language
  expression selection, streaming playback, cancellation, and live metrics.
- Preserves neutral complete WAV, streaming PCM, and semantic output exactly
  against the immutable v1.2 baseline for both qualified packages in all five
  languages.
- Retains the existing v1 API, GPT-SoVITS flat adapter, OpenAI-compatible
  adapter, schema-1 model packages, and Transformer/TensorRT acoustic runtime.

### Quality and architecture decision

Both qualified V2ProPlus packages passed the Japanese full-expression matrix,
the five-language long/short matrix, real TensorRT API acceptance, and the
independent blind-listening gate. v1.3 deliberately keeps the v1.2 Transformer
semantic decoder after hard-token acoustic lookahead, recurrent speculative
decoding, SemanticPiece/EABPE, FastStart, and the evaluated D16 block-diffusion
recipe failed their respective release gates.

The implementation and expression policy are package-driven. The runtime has
no Miku- or Roxy-specific branch, and one process continues to load only one
active model package at a time.

### WebUI

Start the API, then run `run_webui.bat`. The local WebUI binds to
`127.0.0.1:9890` and provides model switching, segmented expression control,
streaming playback, and live latency metrics.

### Runtime

- API version: `1.3.0`
- Backend: TensorRT 11, FP16
- Languages: `zh`, `yue`, `en`, `ja`, `ko`
- Release container tag: `ghcr.io/hiruynk/aniflive-tts:1.3.0-cu128`
- Compatibility profile: `cu126`; publish its tag only after release-workflow
  and target-host validation

The canonical public performance report uses Roxy only. Miku remains outside
the performance headline while its model-specific long-form streaming behavior
is investigated separately.

Original AnifLive-TTS code remains licensed under PolyForm Noncommercial
1.0.0. Commercial use requires a separate written Commercial License from
Hiruynk; upstream and third-party components retain their respective licenses.

## 繁體中文

AnifLive-TTS v1.3 在保留既有 V2ProPlus FP16 TensorRT 11 推理路徑的同時，
加入由模型套件管理的情感控制及本機 WebUI。

### 主要更新

- 支援全句及分段的情感設定，並可選擇離散強度。
- 模型啟用時會準備套件內所有情感參考資料，神經網路條件資料持續保留於 GPU。
- 新增通用的 `model import-expressions` 管理命令；工具只依賴模型套件資料，
  不會根據角色或模型名稱分支。
- 加入五語本機 WebUI，提供音色切換、自然語言情感選擇、串流播放、取消請求及
  即時數據。
- 兩個完成驗收的模型套件，在五種語言下的中性完整 WAV、串流 PCM 及語意輸出
  均與不可變的 v1.2 基準完全一致。
- 保留既有 v1 API、GPT-SoVITS 舊版相容介面、OpenAI 相容介面、schema-1
  模型套件及 Transformer／TensorRT 聲學推理流程。

### 音質與架構決定

兩個完成驗收的 V2ProPlus 模型套件均通過日語完整情感矩陣、五語長短句矩陣、
真實 TensorRT API 驗收及獨立盲聽。硬式語意預讀、循環式推測解碼、
SemanticPiece／EABPE、FastStart 及本次 D16 區塊擴散方案未通過各自的發行門檻，
因此 v1.3 刻意保留 v1.2 Transformer 語意解碼器。

情感策略完全由模型套件決定，執行環境沒有 Miku 或 Roxy 專屬分支；每個處理程序
仍然只會載入一個啟用中的模型套件。

### WebUI

先啟動 API，再執行 `run_webui.bat`。本機 WebUI 綁定
`127.0.0.1:9890`，提供音色切換、分段情感控制、串流播放及即時延遲數據。

### 執行環境

- API 版本：`1.3.0`
- 後端：TensorRT 11、FP16
- 語言：`zh`、`yue`、`en`、`ja`、`ko`
- 發行容器標籤：`ghcr.io/hiruynk/aniflive-tts:1.3.0-cu128`
- 相容配置：`cu126`；必須通過發行流程及目標主機驗收後才發布其標籤

公開性能主數據只採用 Roxy。Miku 的模型特定長句串流現象仍在獨立調查，
因此不納入性能主數據。

AnifLive-TTS 原創程式碼繼續採用 PolyForm Noncommercial 1.0.0；商業使用須
另行取得 Hiruynk 的書面商業授權，上游與第三方組件維持各自授權。
