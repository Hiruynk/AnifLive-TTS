# Studio browser regressions

These tests use Playwright 1.55.1 inside Docker. The matching browser image is:

`mcr.microsoft.com/playwright:v1.55.1-noble@sha256:2f29369043d81d6d69a815ceb80760f55e85f5020371ad06a4d996f18503ad1c`

## Fixture

Run `studio_ux_server.py` in the existing workstation Python Docker image,
with the repository's `src`, `webui` and `assets` mounted read-only under
`/repo`. Set `PYTHONPATH=/repo/src` and
`ANIFLIVE_TTS_WEBUI_TRUSTED_HOSTS=127.0.0.1,localhost,host.docker.internal`.
Publish container port 8000 to **127.0.0.1:9893** only and provide writable
temporary storage at `/tmp`.

The fixture owns only temporary test metadata and a synthetic audio upstream.
Do not attach the real workstation/data directory. Do not use a real Studio as
the write target. The mutation-capable checks require the
`X-Aniflive-UI-Fixture: synthetic` marker before proceeding.

## Checks

Mount a writable local QA artifact directory at `/qa` in the browser image.
Install the pinned `playwright@1.55.1` package there, then run the checked-in
scripts with that module available on Node's search path:

- `layout-check.cjs`: 66 module/viewport combinations and screenshots.
- `interaction-check.cjs`: desktop and touch-emulated mobile workflows,
  unchanged settings request body, save races and synthetic audio.
- `motion-material-check.cjs`: actual animation intermediate frames and
  reversal, reduced motion, and transparent compound input interiors.
- `state-check.cjs`: empty/large/error states, translations and standalone UI.

Artifacts are written under `/qa`. These tests do not substitute for the
required computer-use acceptance; simulated mobile coverage must not be
reported as physical-device testing.
Do not reset or restart the user's Studio or its settings to run these checks.

### Editorial layout checks

Set STUDIO_QA_BASE to the isolated fixture origin when it is not on port 9893.
The interaction, state, motion-material and editorial-contract checks reject
a non-synthetic fixture. editorial-contract-check.cjs measures dock animation
geometry, keyboard/reduced motion, stale-row preservation, first-read errors,
unknown progress and four zoom-equivalent reflow sizes. Its reflow checks do
not certify native browser zoom.


- progressive-table-check.cjs: lazy construction of 1,000 records, keyboard
  loading, focus preservation and frozen-dataset background import prevention.
- editorial-editing-check.cjs: expression draft/save races and two mobile
  keyboard geometry simulations.
- editorial-pwa-check.cjs: use the synthetic server's Docker network namespace
  so 127.0.0.1:8000 is a secure loopback origin for Service Worker testing;
  verifies cache version/cleanup, locale-only storage and cached offline content.
