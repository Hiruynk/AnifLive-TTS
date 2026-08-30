# API 規格

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
  `expression`；省略的強度及策略會沿用頂層設定。公開 API 不接受本機路徑、
  網址、上傳音訊或參考資料覆寫。
- 情感切換必須位於逗號、句號、分號、問號、感嘆號或段落邊界之後。每段應為
  完整子句並包含結尾標點；詞語中間的切換會被拒絕，以免漏字或發音不清。
- process 只接受 active `model` 和 startup cache 的 `voice_profile`。

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
