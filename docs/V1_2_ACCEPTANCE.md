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
- GPT decode P50: at least 25% faster for promotion of a replacement semantic
  backend; a retained Transformer backend reports this as a research subgate
  rather than an end-to-end release blocker
- complete-WAV RTF P50: no more than 3% regression
- peak process VRAM: required before claiming a VRAM improvement; v1.2 makes
  no process-isolated VRAM claim

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

Miku remains part of the manual release evaluation even when automatic
similarity metrics pass. The final blind gate accepted v1.2 after short
Japanese, long Japanese, and long Cantonese listening. Slight overlap was
heard in both v1.1 and v1.2 Miku long-form streaming and is retained as a
non-blocking model-specific follow-up; Roxy passed without that observation.

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

## Release Candidate Results

The selected candidate keeps the Transformer semantic backend and the complete
v1.1 acoustic path. It uses per-model fitted GPT engines, persistent TensorRT
contexts, zero auxiliary streams, sampling CUDA Graph, the established 9+8
first-preview policy, two-step EOS checks, and a 25-second guarded warm window.

Formal measurement alternated Miku and Roxy across 10 sessions per model. Each
of the 20 model-sessions used 10 warmups followed by 100 complete-WAV, 100
new-connection streaming, and 100 keep-alive streaming requests.

| Gate | Baseline | Candidate | Result |
|---|---:|---:|---|
| Keep-alive audible TTFA P50 | 89.296 ms | 77.717 ms | Pass, 12.967% faster |
| Keep-alive audible TTFA P95 | 123.190 ms | 94.550 ms | Pass |
| New-connection audible TTFA P50 | 101.721 ms | 96.034 ms | Pass, 5.591% faster |
| Complete-WAV RTF P50 | 0.128717 | 0.088774 | Pass, 31.032% faster |
| GPT decode P50 | >=25% replacement-backend research target | 13.385–19.007% in preserved A/B reports | Research target not met; production Transformer retained |
| Peak process VRAM | Required for a VRAM improvement claim | Not isolated | No claim published |

All five language cases, API checks, long-form checks, deterministic checks,
and TensorRT execution checks passed. Every formal request reported
`TensorRT-11` with `X-PyTorch-Fallback: false`.

Automated streaming quality passed for both model packages:

| Model | Log-mel cosine | Speaker cosine | Duration difference | Result |
|---|---:|---:|---:|---|
| Miku V2ProPlus | 0.992814 | 0.983066 | 0.000% | Pass |
| Roxy V2ProPlus | 0.990298 | 0.989324 | 0.032% | Pass |

The fixed Miku complete-WAV regression case is byte-identical to v1.1 and has
speaker cosine 0.9999999. The default repetition penalty is therefore `1.0`;
the rejected `1.10` experiment is not part of the candidate.

MTP, GPU Viterbi, Mamba-2 hybrid, full GPT-step CUDA Graph, and EOS4 were not
selected because they failed parity, quality, compatibility, or measured-gain
requirements. None is enabled in the production runtime.

Current decision: **READY FOR RELEASE**. End-to-end performance, reliability,
automated quality, and manual blinded-listening gates pass. The GPT decode
research target was not met, so no replacement semantic backend is promoted;
the production Transformer path remains selected. Process-isolated VRAM is not
reported or claimed for v1.2.
The machine-readable decision is stored in
[`benchmarks/V1_2_RELEASE_GATE.json`](../benchmarks/V1_2_RELEASE_GATE.json).
