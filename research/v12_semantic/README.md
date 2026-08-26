# AnifLive-TTS v1.2 Semantic Research

This directory contains training and export tools for optional v1.2 semantic
backends. Nothing here is imported by the production API hot path.

## Phase A: causal GPT block step

The H2/H4 block step consumes already-known semantic tokens and updates the
Transformer KV cache with one causal TensorRT enqueue. It does not predict
future tokens and is not a standalone replacement for the v1.1 AR loop.

Measured on the RTX 5070 Ti development system with TensorRT 11.2.1.2:

| Variant | Sequential CUDA P50 | Block CUDA P50 | Neural speedup |
|---|---:|---:|---:|
| H2 | 5.018 ms | 2.973 ms | 1.688x |
| H4 | 9.893 ms | 2.846 ms | 3.476x |

The PyTorch implementation matches sequential decoding within `7e-6`. The
TensorRT FP16 engines preserve top-1 candidates on the real Miku path and keep
logit/KV maximum absolute error within `0.03125`.

Parallel FP16 execution does not guarantee bit-exact stochastic sampling. In a
three-seed real-path audit, H2 differed at 2 of 60 inspected boundaries and H4
differed at 1 of 59. Therefore:

- the block engine must not be presented as an exact v1.1 replacement;
- an MTP backend must verify proposals against target block logits;
- architecture candidates use the v1.2 WER/CER, speaker, log-mel, duration,
  repetition, artifact, and blinded-listening gates;
- the original Transformer runtime remains available as the golden baseline.

Machine-readable reports are stored outside the repository under the v1.2
investigation report directory. Model checkpoints, ONNX files, engines, and
voice assets are never stored in this source tree.
