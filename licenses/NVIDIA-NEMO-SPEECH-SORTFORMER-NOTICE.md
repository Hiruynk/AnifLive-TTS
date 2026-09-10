# NVIDIA NeMo-Speech.cpp and Streaming Sortformer Notice

AnifLive-TTS Studio uses the following third-party components in its isolated
Linux workstation worker:

- **NeMo-Speech.cpp 0.1.0** by NVIDIA Corporation and affiliates, distributed
  under the Apache License 2.0. The worker downloads the unmodified official
  `linux-x86_64-cuda` release archive and verifies its published SHA-256 before
  installation. The archive retains NVIDIA's `LICENSE`, `NOTICE`, and third-party
  notices.
- **nvidia/diar_streaming_sortformer_4spk-v2**, distributed under CC BY 4.0.
  AnifLive-TTS Studio uses NVIDIA's published Q8_0 GGUF without modifying the
  model weights. It is pinned to revision
  `5240a64075176943f677d30fa2171c780229f341` and SHA-256
  `0679cfeb1ce356d0dea9470b31274f4bfc7eb927497d82005483770666da998a`.

Sources:

- https://github.com/NVIDIA/NeMo-Speech.cpp
- https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2
- https://creativecommons.org/licenses/by/4.0/

The integration code and dataset-routing policy are modifications made by
AnifLive-TTS; NVIDIA has not endorsed this integration.
