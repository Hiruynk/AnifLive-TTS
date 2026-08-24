# AnifLive-TTS v1.1 Acceptance

## Passed On CUDA 12.8 / RTX 5070 Ti

| Gate | Result |
|---|---|
| Repeated performance | Miku and Roxy; 10 sessions per voice; 10 warmups + 100 complete + 100 new-connection stream + 100 keep-alive stream requests per session |
| Complete WAV wall P50 | 245.769 ms across 20 model-session statistics |
| RTF P50 | 0.110447 across 20 model-session statistics |
| New-connection first-packet P50 / P95 | 97.081 / 121.201 ms |
| Keep-alive first-packet P50 / P95 | 85.854 / 111.460 ms |
| New-connection audible TTFA P50 / P95 | 127.081 / 151.201 ms |
| Keep-alive audible TTFA P50 / P95 | 115.854 / 141.460 ms |
| TensorRT execution | 9/9 deserialize and real enqueue; no PyTorch model fallback |
| Five-language API | zh/yue/en/ja/ko passed on both voice packages |
| Legacy/OpenAI adapters | Passed |
| Expression contract | HTTP 501 `expression_not_implemented` passed |
| Long-form streaming | Punctuation and profile-safe technical segmentation passed |
| Model switching | One active voice; previous model unloaded before replacement loads |
| Miku quality | Log-mel 0.995100; speaker 0.987893; duration difference 0.000% |
| Roxy quality | Log-mel 0.993346; speaker 0.983652; duration difference 0.629% |
| Offline container recreate | Passed with existing local image and persistent host data |

The canonical performance report is
`benchmarks/README_BENCHMARK_SUMMARY.json`. Voice-specific model packages,
audio, and detailed acceptance reports remain outside the source tree.

## Attempted But Not Claimed

- Full GPT CUDA Graph: rejected by TensorRT Myelin capture with `cudaError 900`.
- C++ hot path: not adopted because the native ABI and deterministic CUDA sampler require a separate fully revalidated implementation.
- cu126 compatibility image: source and build-policy validation only; GPU E2E support is not claimed before target-host validation.
- CER/WER: not claimed without a fixed licensed multilingual ASR reference set.

Future performance changes must continue to pass log-mel, speaker, duration,
five-language, streaming, and no-fallback gates on more than one V2ProPlus
voice package.
