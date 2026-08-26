# AnifLive-TTS v1.2 Acceptance Contract

## Scope

v1.2 is a baseline-preserving semantic acceleration program. The production
Transformer decoder remains the golden implementation. Optional MTP and
Mamba-2 variants may be released only after passing the gates below. The
SoVITS streaming and acoustic path is frozen unless a separately reviewed
quality fix is required.

Development is isolated from v1.0 and v1.1:

- base tag: `v1.1.0` (`674902e`)
- branch: `v1.2-semantic-research`
- worktree: `D:\Win\Work\Projects\AnifLive-TTS-v1.2`
- Python: the worktree-local `.venv`

No v1.2 experiment may overwrite a v1.0/v1.1 model package, engine, image,
report, tag, or accepted audio artifact.

## Reference Measurements

These historical v1.1 values are orientation only. Every decision uses
interleaved A/B measurements built from the same baseline commit, on the same
GPU, with the same models and workload.

| Metric | Historical v1.1 value |
|---|---:|
| Keep-alive first packet P50 | 82.671 ms |
| Keep-alive audible TTFA P50 | 89.296 ms |
| New-connection first packet P50 | 95.096 ms |
| New-connection audible TTFA P50 | 101.721 ms |
| Server inference P50 | 204.186 ms |
| Complete-WAV RTF P50 | 0.128717 |
| GPU busy-time P50 | 44% |

Canonical release measurements use 10 independent sessions. Each session has
10 warmups, 100 complete requests, 100 new-connection streaming requests, and
100 keep-alive streaming requests at concurrency one.

## Release Gates

### Performance

- keep-alive audible TTFA P50: at least 10% faster than matched baseline
- keep-alive audible TTFA P95: no regression
- new-connection audible TTFA P50: at least 5% faster
- GPT decode P50: at least 25% faster
- complete-WAV RTF P50: no more than 3% regression
- peak process VRAM: no more than 10% regression unless a hybrid backend
  provides a separately reported material memory reduction

### Reliability

- TensorRT-only neural inference
- no PyTorch neural fallback
- all `zh`, `yue`, `en`, `ja`, and `ko` cases pass
- at least 1,000 serialized requests without NaN, empty, or malformed audio
- explicitly selected semantic variants fail startup rather than silently
  falling back to Transformer
- v1.0/v1.1 schema-1 packages and public API requests remain compatible

### Quality

Candidate streaming output versus candidate complete output:

- log-mel cosine >= 0.99
- speaker cosine >= 0.98
- duration difference <= 3%

Candidate versus matched v1.1 baseline:

- WER/CER absolute regression <= 0.5 percentage point
- speaker similarity drop <= 0.005
- no systematic repetition, omission, clipped onset, electronic artifact, or
  pronunciation regression
- blinded listening gate passes independently for Miku and Roxy

Miku is a release blocker even when automatic similarity metrics pass. v1.2
must restore or improve the accepted Miku listening quality that was traded for
latency during v1.1 tuning. A latency win cannot compensate for a Miku quality
failure.

## Phase Gates

1. Semantic runtime extraction: exact semantic sequence, complete WAV SHA256,
   stream PCM SHA256, GPT step count, and RNG parity; TTFA/RTF regression <1.5%.
2. Block-step H=2/H=4: numerical equivalence to sequential steps and at least
   1.5x neural-stage speedup for H=2 before MTP work continues.
3. MTP-2/MTP-4: adapter base-checkpoint SHA256 must match; no modified base
   checkpoint is published.
4. MTP-4 + GPU Viterbi: GPT decode <=60% of baseline and keep-alive audible
   TTFA <=90% of baseline while all quality gates pass.
5. Mamba-2 hybrid H=1: quality pass plus either >=10% ms/NFE improvement or
   >=20% state-memory reduction with no TTFA regression.
6. Hybrid + MTP-4: ship only when it materially improves the accepted MTP-only
   candidate. Otherwise MTP remains the v1.2 release backend.

Every phase produces implementation, tests, a machine-readable benchmark, a
quality report, and an explicit pass/fail decision before the next phase begins.

