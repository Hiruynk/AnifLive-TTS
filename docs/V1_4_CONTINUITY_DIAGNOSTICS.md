# v1.4 Long-Form Boundary Diagnostics

`continuity_diagnostics.py` measures directly observable waveform behavior where ordered
AnifLive-TTS segments meet. It accepts uncompressed mono PCM16 WAV files and raw little-endian
mono PCM16 files. All inputs in one report must use the same sample rate.

This tool is diagnostic only. It does not measure or establish pronunciation, intelligibility,
speaker identity, semantic correctness, expression quality, naturalness, or production readiness.
It must not be used as a replacement for multilingual CER/WER, speaker-similarity evaluation,
long-form listening, or blind human review.

## Usage

```powershell
python scripts/evaluate_long_form_continuity.py `
  segment-001.wav segment-002.wav segment-003.wav `
  --expected-sample-rate 32000 `
  --window-ms 10 `
  --silence-threshold-dbfs -50 `
  --output continuity-report.json
```

Raw PCM requires an explicit rate:

```powershell
python scripts/evaluate_long_form_continuity.py `
  segment-001.pcm segment-002.pcm `
  --raw-sample-rate 32000
```

Raw PCM has no header from which channel count or sample encoding can be discovered. The tool
therefore accepts it only under the explicit `pcm16le` contract: one channel, signed 16-bit
little-endian samples, and the caller-supplied sample rate. WAV inputs carry these fields and are
validated directly from their headers.

The command prints JSON to standard output and optionally writes the same report atomically to
`--output`. Validation failures are emitted as JSON on standard error and return exit code `2`.

## Speech Session A-F Runtime Matrix

`benchmark_long_form_context.py` exercises the real committed-segment API with
the same text, generation seed and model identity under policies A-F:

```powershell
python scripts/benchmark_long_form_context.py `
  --base-url http://127.0.0.1:9880 `
  --model roxy-v2proplus `
  --language yue `
  --segment "第一個已提交段落，" `
  --segment "這是相同段落內的延續。" `
  --repetitions 2 `
  --output context-af-report.json
```

For every policy the report records PCM SHA-256, fixed-seed determinism,
first-packet wall time, audible TTFA, complete stream time, the final bounded
context summary and the upstream continuity headers. The runner rejects streams
without audible PCM. Its report always says `diagnostic_only=true` and
`release_qualified=false`; it cannot substitute for the production TensorRT
parity, multilingual quality or blind-listening gates.

## Reported Signals

- `sample_discontinuity_normalized`: absolute difference between the last PCM sample on the left
  and the first PCM sample on the right, divided by the PCM16 full scale.
- `rms_jump`: absolute difference between normalized RMS levels in the short windows around the
  boundary. `rms_jump_db` reports the corresponding absolute dBFS difference.
- `energy_jump`: absolute difference between normalized mean-square energy in those windows.
- `dc_jump`: absolute difference between normalized window means.
- `pause_duration_seconds`: consecutive below-threshold samples at the left tail plus the right
  head. Its meaning depends on the configured silence threshold.
- `duration_seconds`: duration derived from the validated sample count and sample rate.

Each boundary is reported separately. Aggregate boundary metrics and segment durations include
deterministic P50, P95, and maximum values. The report records the window, silence threshold,
format, and resource limits required to interpret those values.

The default resource bounds are 256 segments, 512 MiB per input file, and 1 GiB of decoded PCM in
one report. They can be reduced by the caller; increasing them should be deliberate because this
initial implementation keeps each validated segment in memory while it evaluates all boundaries.
