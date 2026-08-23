# AnifLive-TTS v1 Acceptance

## Passed On CUDA 12.8 / RTX 5070 Ti

| Gate | Result |
|---|---|
| Repeated performance | 10 sessions; each has 10 warmups + 100 complete + 100 streaming requests |
| Complete WAV wall P50 | 252.862 ms session median; 249.456 ms best complete session |
| RTF P50 | 0.085426 session median |
| Streaming TTFA P50 / P95 | 78.964 / 104.440 ms session median, below 150 / 200 ms gates |
| TensorRT execution | 8/8 deserialize and real enqueue; no PyTorch model fallback |
| Five-language API | zh/yue/en/ja/ko passed |
| Legacy/OpenAI adapters | Passed |
| Expression contract | HTTP 501 `expression_not_implemented` passed |
| Complete output parity | Same 74 tokens, 2.96 s, exact WAV SHA-256 |
| Waveform/log-mel/speaker | 1.0 / 1.0 / 1.0 |
| Offline container recreate | Passed with `--pull never --no-build`; no download/build |
| Private character overlay | Local and tunneled health/TTS returned TensorRT-11, fallback false |

Source reports are stored outside the source tree.

## Attempted But Not Claimed

- Full GPT CUDA Graph: TensorRT internal train-station capture error, including aux0 engine and thread-local capture.
- C++ hot path: no nvcc/headers in the runtime toolchain; native sampler/plugin remains research.
- cu121 on RTX 5070 Ti: engines build/load, but PyTorch cu121 has no Blackwell sm_120 kernel.
- CER/WER: no fixed, licensed offline ASR reference set was available for this run.

Any future performance change must preserve semantic tokens and pass waveform,
log-mel, speaker, duration, five-language, stream, and no-fallback gates.
