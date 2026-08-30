# AnifLive-TTS Acceptance Records

## v1.3 Passed On CUDA 12.8 / RTX 5070 Ti

| Gate | Result |
|---|---|
| Canonical performance | Roxy only; 10 sessions; 10 warmups + 100 complete + 100 new-connection stream + 100 keep-alive stream requests per session |
| Complete WAV wall P50 / P95 | 153.177 / 189.548 ms |
| RTF P50 | 0.087680 |
| New-connection first-packet P50 / P95 | 67.077 / 90.853 ms |
| Keep-alive first-packet P50 / P95 | 67.953 / 92.019 ms |
| New-connection audible TTFA P50 / P95 | 73.702 / 97.478 ms |
| Keep-alive audible TTFA P50 / P95 | 74.578 / 98.644 ms |
| TensorRT execution | 9/9 deserialize and real enqueue; no PyTorch neural fallback |
| Five-language API | `zh` / `yue` / `en` / `ja` / `ko` passed on both qualified packages |
| Legacy/OpenAI adapters | Passed |
| Expression contract | Global and segmented package-curated profiles passed; continuous vector remains unsupported |
| Japanese expression matrix | Miku 18/18 and Roxy 18/18 passed; no contractual underruns |
| Five-language expression matrix | Miku 20/20 and Roxy 20/20 passed; no contractual underruns |
| Neutral parity | Complete WAV, stream PCM, and semantics exactly match immutable v1.2 for both packages and all five languages |
| Blind listening | Six pairs: five no meaningful difference, one Roxy complete-output preference, zero artifacts |
| Model switching | One active package; previous model unloaded before replacement loads |
| WebUI | Model switching, expression resolution, streaming playback, cancellation, and three interface languages passed |
| Public WebUI privacy | No login page, account, password, session credential, or credential storage |
| Offline container | Existing local image and persistent host data; no runtime model or dependency download |

The canonical machine-readable performance report is
`benchmarks/README_BENCHMARK_SUMMARY.json`. Voice-specific packages, reference
audio, blind-listening media, and detailed private QA evidence remain outside
the source tree. Evaluated v1.3 alternatives are documented in
`docs/research/v1.3-latency-experiments.md`.

## v1.1 Historical Acceptance

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

The table above is retained as the historical v1.1 record. The current
canonical report is `benchmarks/README_BENCHMARK_SUMMARY.json`. Voice-specific
model packages, audio, and detailed acceptance reports remain outside the
source tree.

## Attempted But Not Claimed

- Full GPT CUDA Graph: rejected by TensorRT Myelin capture with `cudaError 900`.
- C++ hot path: not adopted because the native ABI and deterministic CUDA sampler require a separate fully revalidated implementation.
- cu126 compatibility image: source and build-policy validation only; GPU E2E support is not claimed before target-host validation.
- CER/WER: not claimed without a fixed licensed multilingual ASR reference set.

Future performance changes must continue to pass log-mel, speaker, duration,
five-language, streaming, and no-fallback gates on more than one V2ProPlus
voice package.
