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
  "expression": {"enabled": false, "profile": "calm", "intensity": 0.6},
  "generation": {"top_k": 15, "top_p": 1.0, "temperature": 1.0, "seed": 1234}
}
```

- `language`: `zh`、`yue`、`en`、`ja`、`ko`。
- `stream=false`: `audio/wav` PCM16 mono。
- `stream=true`: 回傳小端序 16 位元有號 PCM（S16LE）音訊區塊，Content-Type 為
  `application/octet-stream`，並包含 `X-TTS-Sample-Format: s16le` 回應標頭。
- `expression.enabled=true`: HTTP 501，error code
  `expression_not_implemented`。
- process 只接受 active `model` 和 startup cache 的 `voice_profile`。

## Discovery

- `GET /health`
- `GET /v1/capabilities`
- `GET /v1/models`
- `GET /v1/voices`

`POST /v1/models/activate` 接受 `{"model":"my-v2proplus"}`，用於切換至
已存在本機模型登錄目錄中的相容套件。進行語音推理時不允許切換；執行時會先卸載目前
套件，再載入替代套件。切換後原有語音合成 API 與網址保持不變。

## Compatibility adapters

舊 GPT-SoVITS `GET /`、`POST /` flat schema 保留。OpenAI-style
`input`、`voice`、`model` schema 亦由同一 `/v1/audio/speech` endpoint 接受。
`auto` 與 `auto_yue` 只在 adapter 使用。
