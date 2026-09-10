# AnifLive-TTS v1.4 TSE Overlap Separation

## Decision

The true two-speaker overlap path is **implemented and qualified** for the
isolated v1.4 workstation. It is not a Windows neural fallback and it does not
alter v1.3.

The worker uses a pinned `MossFormer2_SS_16K` blind separator in Linux Docker.
The separator is not allowed to decide which stream belongs to the requested
speaker. Every 2-second output window is scored against the supplied reference
by the workstation-owned TensorRT 11 ERes2NetV2 verifier; ambiguous windows
fail closed to review. Dataset acquisition therefore has no circular dependency
on a voice package that does not exist yet.

## Provenance

| Asset | Pinned identity |
|---|---|
| ClearerVoice-Studio source | `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61` |
| Source archive SHA-256 | `f8f8d2f2190b9909b51e91ce886d1c7efedb7349b3ff6d3ab521451165fbd8da` |
| MossFormer2 model revision | `407cb030cd66340918ebb6c8cc63b18f8592cdbe` |
| Checkpoint SHA-256 | `00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a` |
| Checkpoint size | `670,353,271` bytes |
| License | Apache-2.0 |

Only the small pinned model source required at runtime is copied into the
worker image. The checkpoint remains in an external operator-managed directory
and is mounted read-only. Startup rejects a missing revision record, wrong
checkpoint size, wrong checksum, missing tensor, or incompatible tensor shape.

## Runtime Contract

1. Decode source and reference to mono 16 kHz float PCM.
2. Use the generic workstation TensorRT speaker engine for the reference
   embedding and the ordinary segment verification pass.
3. Combine deterministic VAD regions with any operator-confirmed
   `separation_segments` ranges. The ranges are expressed as sorted,
   non-overlapping `{start_seconds, end_seconds}` pairs and are rejected unless
   they are finite and wholly inside the decoded source.
4. Send detector- or operator-marked regions to the pinned CUDA separator in 2-second windows
   with a 75% stride.
5. Apply the upstream ClearerVoice two-stage -25 dB input normalization and
   restore the original gain after model execution.
6. Score both blind candidates with the same TensorRT speaker engine.
7. Accept only when the winning target cosine passes the project threshold and
   exceeds the other source by the configured ambiguity margin.
8. Resolve source permutation per window before overlap-add. A rejected window
   leaves the segment in review; it is never labelled as a successful target
   extraction.

The worker report records source/model/checkpoint provenance, every candidate
similarity, selected source, reconstruction RMSE, attempted/accepted segment
counts, exact operator-requested sample ranges, per-segment `detector` versus
`operator` triggers, and the TensorRT target-selection contract. The WebUI
review table can mark a detected segment for separation and re-run the same
generic contract; no voice name changes its behavior.

The corresponding `tse.prepare` job setting is:

```json
{
  "separation_segments": [
    {"start_seconds": 12.4, "end_seconds": 18.75}
  ]
}
```

Unknown fields, strings in place of JSON numbers, non-finite values, reversed
bounds, unsorted or overlapping ranges, and ranges beyond the decoded source
duration are rejected before the separator is loaded.

## RTX 5070 Ti Qualification

Qualification ran with `--network none` in the CUDA 12.8 Linux worker on the
RTX 5070 Ti. Two external V2ProPlus references were mixed at equal active RMS,
with the interferer entering after one second. Private audio, model packages,
generated WAV files and raw reports remain outside the repository.

| Target / interferer | Mixture SI-SDR | Separated SI-SDR | Improvement | Target cosine after | Minimum winning cosine | Result |
|---|---:|---:|---:|---:|---:|---|
| Roxy / Miku | 0.251 dB | 17.466 dB | **+17.215 dB** | 0.9295 | 0.8162 | PASS |
| Miku / Roxy | 1.746 dB | 15.769 dB | **+14.024 dB** | 0.8509 | 0.7640 | PASS |

Both directions passed the production `0.72` target threshold and `0.03`
ambiguity margin in all three windows. Measured separation wall time was
1.213 seconds and 1.446 seconds for four seconds of audio (RTF 0.303 and 0.361).
The result demonstrates package-generic selection rather than a voice-name
branch.

The reusable runner is `scripts/qualify_tse_overlap.py`. Its pass gate requires
finite exact-length output, every window accepted by TensorRT identity,
threshold and ambiguity compliance, and at least +3 dB SI-SDR improvement.

The same Roxy-target mixture also completed through the strict production
worker manifest and artifact path, rather than only the research runner. Its
operator-requested `0.0–4.0 s` range produced one accepted 64,000-sample target
artifact, a minimum selected cosine of `0.8160`, and `+17.217 dB` SI-SDR over
the input mixture after the worker's PCM16 write. The report recorded both the
detector and operator triggers, exact sample bounds, image identity and artifact
checksums.

An additional repeated stationary-mixture probe produced no identity-change
trigger (`detector=0`, `operator=1`). The manual contract still invoked the
separator, but the unnatural repeated probe failed the `0.72` TensorRT identity
threshold and therefore returned review with no claimed target audio. This is
the intended fail-closed behavior and is not counted as a separation-quality
pass.

## Known Boundary

The Dataset Factory path no longer relies on identity-change evidence alone. It
first runs the pinned Sortformer postprocess (`onset=0.4`, `offset=0.7`) across
every VAD clip and fuses its speaker regions with the multi-reference TensorRT
speaker prototype. A second, more sensitive `onset=0.3` pass runs only for
primary-clean candidates close to the identity threshold. Clips below the
qualified clean-SNR floor also leave the automatic-clean path. These checks are
generic and are selected from measured evidence, never a voice name.

The complete offline 31-minute labelled fixture produced 363 records. All nine
pure-overlap records and all 26 mixed-truth records were contained in
review/salvage, no non-target record was routed clean, no target record was
auto-rejected, and no contaminated record was routed directly to clean. The
machine-readable public summary is
`benchmarks/V1_4_VOICE_ACQUISITION_QUALIFICATION.json`.

That fixture qualifies the recorded policy on the RTX 5070 Ti; it is not a
claim that one threshold is universal across all speakers, rooms and capture
devices. Uncertain evidence remains quarantined for review. The bidirectional
SI-SDR results above continue to qualify the separator and TensorRT target
selector directly.

Known overlaps remain reachable through the strictly validated
`separation_segments` manifest setting and the WebUI review checkbox. A
new acoustic domain should repeat the labelled overlap matrix before claiming
automatic coverage.
