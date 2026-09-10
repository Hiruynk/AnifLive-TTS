# Evaluation workload and comparison contract

This document describes the v1.4 evaluation repair. It does not certify a release
or replace the final acceptance matrix.

## Fixed Japanese performance workload

Japanese performance measurements default to:

    今日はいい天気ですね。

The five-language quality probes are independent. In particular, the longer
Japanese quality probe must not become the default performance benchmark text.

Explicit `benchmark_text` remains supported for noncanonical experiments. A
baseline comparison requires exact workload equality, including text, language,
sampling parameters and expression configuration. Merely matching the language
or request count is insufficient.

The release comparison uses ten sessions, ten warmups per session, and one
hundred requests per session for each of complete WAV, ordinary streaming and
persistent-connection streaming. The historical report determines the matched
measurement method and statistical definitions.

## Preflight

`POST /api/workstation/evaluation/preflight` accepts an existing `project_id`
and optional job `parameters`. It reads the effective project configuration and
returns the resolved workload and comparison status without creating a job.

`POST /api/workstation/jobs` runs the same check for `evaluation.prepare`.
A mismatched baseline returns HTTP 400 and does not enqueue a job. Retry and
resume use the same preflight and return HTTP 409 without altering the original
job when its effective workload is invalid. This also covers historical jobs
created before the preflight was added.

The scheduler checks ready evaluation jobs before draining speech or handing
off the GPU, including jobs created automatically or outside the HTTP route.
Invalid queued jobs fail with a preflight reason and never start a GPU worker.
The neural worker checks again before starting the CUDA evaluation process.
Conversion listening plans remain a separate diagnostic workflow.

A baseline path must be inside administrator-configured import roots and refer
to a JSON report of at most 16 MiB at the HTTP boundary. Job parameters do not
grant access to arbitrary host paths.

## Performance policy

New comparisons record `performance_policy: v14-near-v13-10pct-v1`.
The following candidate metrics must each be at most 1.10 times the matched
baseline:

- Keep-alive audible TTFA P50.
- Keep-alive audible TTFA P95.
- Complete-WAV RTF P50.

The previously generated 3% reports remain unchanged. This policy changes only
performance tolerances; it does not relax content, acoustic, provenance or human
qualification requirements. A comparison with different workloads is invalid
regardless of how favorable the numbers look.

## Measurement is not qualification

When no historical baseline is supplied, new automatic model evaluations may
run if their required runtime and ASR assets exist. Their comparison is marked
unavailable. A measured result is not a regression pass and cannot grant
artifact promotion on its own. First-use model qualification still requires
content, reference-speaker identity, streaming/runtime, human listening and
security evidence. In that mode the performance gate verifies completed,
finite measurements and explicitly reports that no historical comparison was
made; it does not claim v1.3 regression approval.

A comparison error occurring after measurement must preserve the completed
language measurements, audio, benchmark and `evaluation-report.json`.
Such a report records failure and the comparison reason instead of discarding
the measurements or inventing a successful qualification.

First-use qualification uses exact normalized ASR content (or an explicit
exact-audio review) and the existing absolute reference-speaker limit.
Software release qualification still uses its matched historical benchmark. An HTTP 200, successful
job execution or valid model package alone is not a release approval.

## Explicit runtime sampling policy

Evaluation accepts `semantic_sampling`: `legacy-topk-v1` or
`native-v2proplus-v1`. When omitted, it inherits the model package contract;
packages without a declared contract retain `legacy-topk-v1`. The native
contract requires full-vocabulary logits engines.
The report records the selected contract and its repetition penalty under
`runtime.sampling_policy`; the child API receives explicit environment values.
This allows candidate runtime implementations to be compared on the same
submitted workload without silently measuring a different implementation.
Newly converted packages declare the verified native contract; existing legacy
packages keep their original behavior.

Scheduler preflight uses the worker's configured import roots, including CLI
`--import-root`. HTTP preflight continues to use Studio's configured roots.
These trusted configuration sources do not become user-controlled job settings.


Explicit human content reviews may be supplied as
`content_evidence_artifact_id` to qualification composition. Reviews bind the
original evaluation checksum and the exact registered audio from the same job
and language case. Composition and promotion reverify those files; the
original automated report remains unchanged. Only the content gate can use
these decisions. Models explicitly declaring no expression profiles record
expression quality as not applicable, with no invented listening trials.
