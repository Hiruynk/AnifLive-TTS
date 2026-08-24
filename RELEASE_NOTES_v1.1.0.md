# AnifLive-TTS v1.1.0

## English

AnifLive-TTS v1.1 focuses on interactive latency and natural long-form delivery while preserving the FP16 TensorRT 11 inference path.

### Highlights

- Punctuation-aware inference starts a new model invocation at safe comma, period, semicolon, exclamation-mark, and question-mark boundaries.
- Long-form streaming starts each punctuation-delimited segment after prior PCM is handed to the HTTP stream; it does not wait for client-side playback to finish.
- Adaptive boundary timing removes excessive model-generated trailing silence while retaining natural comma, sentence, and paragraph pauses.
- A guarded 25-second warm-retention window reduces latency after short idle periods without locking GPU clocks or changing power and fan settings.
- All five advertised language frontends are prepared during startup, removing first-request lazy-loading stalls.
- Client disconnects cancel streaming work promptly; concurrent requests receive HTTP 429 instead of waiting in a hidden queue.
- Long semantic sequences are decoded in TensorRT profile-safe chunks without a PyTorch fallback.
- Compatible local V2ProPlus packages can be discovered and activated without changing the synthesis API; the active package is fully unloaded before its replacement loads.
- Checkpoint inspection and package promotion use model metadata and tensor contracts rather than character-specific filenames.

### Runtime

- API version: `1.1.0`
- Backend: TensorRT 11, FP16
- Languages: `zh`, `yue`, `en`, `ja`, `ko`
- Release container tag: `ghcr.io/hiruynk/aniflive-tts:1.1.0-cu128`
- Provisional compatibility profile: `cu126`; publish its tag only after release-workflow and target-host validation

Model packages and TensorRT engines remain external runtime assets. TensorRT engines must match the target GPU, CUDA runtime, and TensorRT fingerprint.

Original AnifLive-TTS code remains licensed under PolyForm Noncommercial 1.0.0. Commercial use requires a separate written Commercial License from Hiruynk; upstream and third-party components retain their respective licenses.

## 繁體中文

AnifLive-TTS v1.1 集中改善互動延遲與長句自然度，同時維持 FP16 TensorRT 11 推理路徑。

### 主要更新

- 遇到適合分句的逗號、句號、分號、感嘆號與問號時，立即以新的模型調用推理下一小句。
- 上一小句的 PCM 交給 HTTP 串流後便會開始處理下一個標點分段，不會等待用戶端播放完畢。
- 自適應句界處理會移除模型產生的過長尾靜音，同時保留自然的逗號、句末與段落停頓。
- 加入受溫度、GPU 佔用率與請求狀態保護的 25 秒熱啟動維持，不鎖定 GPU 時脈，也不修改功耗或風扇設定。
- 五種語言的文字前端會在啟動時完成準備，避免首次請求出現延遲尖峰。
- 用戶端中斷連線後會及時取消串流工作；並行請求會收到 HTTP 429，不會進入隱藏佇列。
- 過長語意序列會按 TensorRT profile 安全範圍分塊解碼，不會回退至 PyTorch 模型推理。
- 可在不改變語音合成 API 的情況下尋找並切換相容的本機 V2ProPlus 套件；目前套件會先完整卸載，之後才載入替代套件。
- 模型檢查與套件發佈依據模型資料及張量規格，不依賴特定角色檔名。

### 執行環境

- API 版本：`1.1.0`
- 後端：TensorRT 11、FP16
- 語言：`zh`、`yue`、`en`、`ja`、`ko`
- 發布容器標籤：`ghcr.io/hiruynk/aniflive-tts:1.1.0-cu128`
- 暫定相容配置：`cu126`；必須通過發布流程與相容主機驗收後才發布其標籤

模型套件與 TensorRT 引擎繼續作為外部執行資產保存。TensorRT 引擎必須與目標 GPU、CUDA runtime 及 TensorRT fingerprint 相符。

AnifLive-TTS 原創程式碼繼續採用 PolyForm Noncommercial 1.0.0；商業使用須另行取得 Hiruynk 的書面商業授權，上游與第三方組件則維持各自授權。
