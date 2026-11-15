# API 規格

Python 應用程式可使用[官方 SDK](SDK_ZH.md)。伺服器 API 是 wire contract；Rust client core 負責傳輸與協定，Python facade 提供型別化同步／非同步介面。

## Canonical request

`POST /v1/audio/speech`

```json
{
  "model": "my-v2proplus",
  "voice_profile": "default",
  "text": "こんにちは。",
  "language": "ja",
  "stream": true,
  "expression": {"enabled": true, "profile": "calm", "intensity": 0.6},
  "generation": {"top_k": 15, "top_p": 1.0, "temperature": 1.0, "seed": 1234}
}
```

- `language`: `zh`、`yue`、`en`、`ja`、`ko`。
- `stream=false`: `audio/wav` PCM16 mono。
- `stream=true`: 回傳小端序 16 位元有號 PCM（S16LE）音訊區塊，Content-Type 為
  `application/octet-stream`，並包含 `X-TTS-Sample-Format: s16le` 回應標頭。
- `expression.enabled=true`：啟用模型套件內經管理員驗證的情感設定；可用設定及
  套件偏好策略由 `GET /v1/expressions` 回傳。未提供情感資料的模型套件會回傳
  HTTP 400，不會靜默忽略請求。
- 請求必須在 `text` 與 `segments` 之間二選一。每個分段可提供自己的
  `expression`；省略的強度及策略會沿用頂層設定。Generation 接受 `top_k` 1–50、`top_p` 大於 0 且不超過 1、`temperature` 大於 0，以及大於或等於 -1 的整數 `seed`。語音合成 endpoint 不接受本機路徑、網址、原始音訊上傳或單次請求參考資料覆寫。請透過既有 Voice／Reference 路徑管理儲存的資料。
- 情感切換必須位於逗號、句號、分號、問號、感嘆號或段落邊界之後。每段應為
  完整子句並包含結尾標點；詞語中間的切換會被拒絕，以免漏字或發音不清。
- `model` 必須指向 active model；`voice_profile` 必須是已登錄 Voice。

## Discovery

- `GET /health`
- `GET /v1/capabilities`
- `GET /v1/models`
- `GET /v1/voices`
- `GET /v1/expressions`
- `POST /v1/audio/cancel`

`POST /v1/models/activate` 接受 `{"model":"my-v2proplus"}`，用於切換至
已存在本機模型登錄目錄中的相容套件。進行語音推理時不允許切換；執行時會先卸載目前
套件，再載入替代套件。切換後原有語音合成 API 與網址保持不變。

## Compatibility adapters

舊 GPT-SoVITS `GET /`、`POST /` flat schema 保留。OpenAI-style
`input`、`voice`、`model` schema 亦由同一 `/v1/audio/speech` endpoint 接受。
`auto` 與 `auto_yue` 只在 adapter 使用。

## Windows v5Turbo 行為

**Backend：** CUDA／TensorRT（Docker）。v2ProPlus 輸出 32 kHz。v5Turbo 使用
`gsv-v5turbo`、FP16、原生 4-step CFM、CFG 0，輸出 48 kHz。
v5 固定 `speed=1.0`、`noise_scale=0.5`、4 個 sample steps，且不接受 `cut_punc`。

v5Turbo backend 接受全部五種 Reference-conditioning policy：
`full-switch`、`identity-lock`、`semantic-style`、`acoustic-style` 及
`sv-only`。`full-switch` 是預設值，可按 ID 選取已儲存 Reference；已儲存的
`emotion_id` 加上受支援的離散強度亦可選出相應 Reference。其他 policy 會把 Reference
對應至 v5 的語義、風格及 CFM voice-anchor 元件；v5 沒有獨立的 v2ProPlus SV encoder。

Voice、Reference 及 Speech Session metadata 沿用既有 API contract。操作方式請參閱
[Voice／Reference API](V1_5_VOICE_API.md)及[平台矩陣](V1_5_PLATFORM_MATRIX.md)。

## macOS Apple Silicon 行為

**Backend：** Apple Silicon（macOS／MLX／Metal）。目前 FP16 p1 worker 輸出 48 kHz 單聲道 PCM16，使用四步 CFM 及 CFG 0。`stream=false` 回傳完整 WAV；`stream=true` 回傳原始 PCM。串流在 AR 到達 EOS 並完成 CFM 後才開始，之後由 vocoder 增量輸出。

FP16 p1 支援全部五種 policy：`full-switch`、`identity-lock`、`semantic-style`、`acoustic-style`、`sv-only`，亦支援逐段 expression 及混合 policy。v5 會把 Reference 對應至可用的語義、風格及 CFM voice-anchor 元件，沒有獨立的 v2ProPlus SV encoder。凍結 FP32 rollback worker 保留原有 policy 集；選策略前請查詢 capabilities 的 `expression.policies`、`expression.complete_policies` 及 `native_worker`。

已登錄 Voice 及 expression profiles 可由 `/v1/voices` 及 `/v1/expressions` 查詢；可用 `GET /v1/expressions?voice_profile=<id>` 查詢指定聲線。用 `voice_profile` 選擇 Voice，Reference 按需準備並快取。Generation 接受 `top_k` 1–50、`top_p` 大於 0 且不超過 1、`temperature` 大於 0，以及大於或等於 -1 的整數 `seed`。v5 限定 `speed=1.0`、`noise_scale=0.5`、4 個 sample steps 並要求 CFG 0。相容介面接受 `auto`、`auto_yue` 及受支援別名；canonical `ja` 使用日英混合前處理模式。

Session A 支援 append、flush、取消及恢復；策略 B–F 仍屬研究功能。取消完整 WAV 請求會以 HTTP 409 結束且不回傳部分 WAV，服務之後可繼續合成。共用 WebUI 可連接 Mac 服務；v2ProPlus 及 Studio 工作站流程仍屬 Windows 功能。硬件要求及 Reference DSP 依賴見 [macOS 安裝指南](MACOS_INSTALLATION_ZH.md)。另見 [Voice／Reference API](V1_5_VOICE_API.md)及[平台矩陣](V1_5_PLATFORM_MATRIX.md)。
