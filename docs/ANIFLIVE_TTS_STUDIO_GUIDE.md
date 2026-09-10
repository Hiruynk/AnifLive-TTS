# AnifLive-TTS Studio Guide

[繁體中文](ANIFLIVE_TTS_STUDIO_GUIDE.zh-TW.md) · [简体中文](ANIFLIVE_TTS_STUDIO_GUIDE.zh-CN.md)

AnifLive-TTS Studio is the local production workstation included with AnifLive-TTS v1.4. It brings synthesis, expression references, dataset preparation, target-speaker extraction, V2ProPlus training, qualification, TensorRT packaging, and GPU job control into one interface. The classic AnifLive-TTS WebUI remains available separately.

## Start Studio

1. Start Docker Desktop and wait until the Linux engine is ready. Dataset neural workers, training, evaluation, and TensorRT builds run in Linux Docker rather than in the Windows host process.
2. Double-click `run_tts.bat` when you need the inference API. Wait for the health check to report ready before testing synthesis.
3. Double-click `run_studio.bat` to start Studio only.
4. Open `http://127.0.0.1:9891/` if the browser does not open automatically.
5. Keep each launcher terminal open. Closing one stops the service owned by that launcher.

Studio detects an API on ports `9880` or `9882`. When neither is available, data, project, and job-management modules still open, while inference-dependent actions remain unavailable. Studio stores workstation metadata under `data/workstation` by default.

For a non-default API address, set `ANIFLIVE_TTS_WEBUI_UPSTREAM` before starting Studio. This changes the synthesis upstream only; it does not move Studio from port `9891`.

Use the globe button in the upper-right corner to switch the complete interface between English, Traditional Chinese, and Simplified Chinese. The setting is shared across Studio pages and the embedded synthesis workstation.

## Interface Basics

- The left navigation follows the production lifecycle: Create, Data, Train, Deploy, then System. Dataset acquisition starts in **Datasets**, including target-speaker work.
- The globe changes the Studio interface language. Speech language is a separate setting inside Synthesis, Dataset transcription, and Evaluation.
- Styled selectors are not native browser menus. Select the field, choose an item from the floating panel, and confirm that the field label changes before continuing.
- File and folder fields support **Browse**, drag and drop, and permitted manual paths. A dropped item is accepted only when the field shows its resolved path. Paths must be inside a configured import root.
- Disabled commands show why they are unavailable. Usually a prerequisite project, artifact, component, API, or qualification gate has not been selected yet.
- Creation and queue actions open a transition state and then add the new record to the relevant list. Do not submit the same action again while its progress state is visible.

## Overview

Overview is the production entry point. It shows the active model, runtime readiness, project counts, current jobs, and recent work. Select a recent project to continue it, or open the module that owns the next part of the workflow. All voice acquisition begins in **Datasets**; Target Speaker Extraction is not a separate normal-user project type.

The full-screen motion field can be disabled under **Settings → Workstation behavior**. Reduced-motion system preferences are respected automatically.

## Synthesis

Synthesis contains the complete AnifLive-TTS v1.3 speech workstation inside Studio.

1. Select a voice model.
2. Enter text. The initial example `今日はいい天気ですね。` is safe to replace.
3. Select the speech language.
4. Highlight text to assign an expression reference to that exact span.
5. Review expression tags, per-segment language, pause, and ordering.
6. Select **Play now** to stream audio, or download the completed result.

During playback, the spoken text is highlighted in gold. Expression underlines and descriptions remain fixed to their annotated spans. With no highlighted expression, synthesis uses the model's neutral native delivery.

Segment settings can assign a different language or pause to each committed segment. Reorder segments by dragging the handle or using the accessible move controls. **Stop** cancels the active stream without changing the script.

## Expression Bank

Expression Bank manages package expressions and local reference drafts.

1. Open **Expressions** and select **New expression**.
2. Enter a stable profile ID, display name, model ID, reference audio path, language, emotion, and intensity.
3. Add one or more descriptions and optional VAD/prosody metadata.
4. Select **Analyze** to measure duration, onset, pitch, rate, and reference identity.
5. Save the draft.

A draft cannot become qualified through a manual status switch. Select passed evaluation evidence, then promote it. This preserves a traceable link between reference audio, measured metadata, and qualification.

## Dataset Factory

Dataset Factory is the single entry point for creating training data. Select **New dataset**, then choose one acquisition mode:

- **Audio / Video Collection** processes recordings that are already mostly one speaker.
- **Extract Target Speaker** finds one speaker in long or multi-speaker media from a short reference recording.
- **Existing GPT-SoVITS Dataset** imports one reviewed `.list` dataset without rebuilding filenames from transcript text.

Every project follows the same visible lifecycle: **Source → Speaker → Clean → Text → Review → Style → Ready**.

### Audio Or Video Collection

1. Add one or more permitted local audio, video, or folder paths.
2. Choose the managed Speech Recognition backend. SenseVoice Small is the five-language default; Faster Whisper remains available when its component is installed.
3. Select **Start preparation**. Linux Docker decodes media, writes canonical PCM, runs managed VAD, creates silence-safe clips, transcribes, and records signal-quality evidence.
4. Open each clip in the item inspector. Correct the transcript and language, confirm the speaker, add an expression when appropriate, and accept or reject it.

### Extract Target Speaker

1. Add the short target reference and one or more source recordings. A trained voice package is not required.
2. Select **Start preparation**. The Linux worker turns usable speech in the reference into a multi-region identity prototype, then runs managed FSMN-VAD, the workstation-owned TensorRT ERes2NetV2 verifier, two-pass Sortformer evidence, the purity router, MossFormer2 only for salvage candidates, and managed ASR.
3. Clean target clips remain unprocessed. Suspected contamination is routed to separation; uncertain results stay in Review; non-target clips are retained under Rejected.
4. Open **Advanced speaker extraction** only when you need per-stage evidence. It shows reference support, primary and sensitive diarization, the clean-SNR gate, offline execution and every job in the chain. It is an inspector, not a second workflow.

Target-speaker Dataset mode emits one stable `seg_000184_7f42a1.wav`-style clip per source range. It never concatenates the selected speaker and sends that file through VAD again. Each item retains source checksum, sample-exact bounds, target similarity, route, separation evidence, raw audio, optional processed audio, and ASR provenance.

The acquisition policy follows evidence-first ideas also found in the public
[Timbre](https://github.com/Etherll/Timbre) project: multi-reference identity,
diarization before separation, silence-validated cuts, forced-split quarantine,
verification fusion, retained rejects and resumable stage artifacts. AnifLive-TTS
uses its own generic TensorRT verifier, worker contracts and quality gates; no
Timbre source code or model asset is bundled.

### Review, Style And Freeze

1. Select **Review queue** and use the Studio player. Correct every ASR suggestion; suggestions never become authoritative training text by themselves.
2. Use `Space` to play and the visible Accept/Reject controls to decide each clip. Forced non-silence cuts, ambiguous identity, or failed post-separation gates remain in review.
3. Add an expression and intensity where appropriate. SenseVoice emotion output is only a suggestion; the human label is authoritative.
4. Select **Expression candidates** to rank reviewed clips by speaker identity, signal quality, usable duration, and annotation completeness. A selected candidate creates an Expression Bank draft, not an automatically qualified profile.
5. Verify each accepted clip's transcript and speaker separately. Editing either value invalidates its verification. Assign train/validation/test splits (default 85/10/5), then select **Freeze dataset**. Expression verification is required only when the project requires expressions.
6. Download **Manifest** or the evidence-only **Qualification report**. The qualification report lists missing training, engine, and production gates rather than guessing success.
7. Select **Continue to Training**. Studio creates the existing Training project from the frozen Dataset manifest; it does not duplicate training controls inside Dataset Factory.

Source media and generated files remain on disk. Studio records their lineage and review state rather than copying arbitrary files into browser storage.

## AI Components

Open **Settings → AI Components** before the first neural Dataset workflow.

- Install each required component explicitly, or install all missing required components together.
- Speaker Verification, SenseVoice Small, FSMN-VAD, and MossFormer2 must show **Ready** for the complete Target Speaker path.
- Every asset is revision-pinned, size-checked, SHA-256 checked, license-recorded, and assigned to the Linux worker runtime.
- Use **Export AI Components Bundle** on an online machine and **Import AI Components Bundle** on an offline workstation. Normal workers always run with `--network none`.
- DeepFilterNet is optional and dereverberation remains unavailable until a backend passes its Linux Docker quality gate.

## Model Training

Training uses GPT-SoVITS V2ProPlus inputs and runs in Linux Docker through the GPU job system.

1. Freeze a reviewed Dataset and select **Continue to Training**, or select a qualified frozen dataset when creating the training project.
2. Choose **Quick**, **Balanced**, **High Quality**, or **Advanced**. Advanced settings include stages, epochs, batch sizes, learning rates, checkpoint cadence, seed and gradient checkpointing.
3. Queue training and monitor status, loss, VRAM, GPU utilization, temperature, ETA, logs and checkpoints. Use **Jobs** for supported pause, resume, cancel and retry actions.
4. Wait for validation checkpoint selection. Saved GPT and SoVITS epochs are candidates; the last epoch is not automatically the winner.
5. Listen to the new blinded reference choices and record your decision. Choices from earlier runs do not apply to a new set. References come only from verified training clips.
6. After the checkpoint winner and reference are locked, the sealed test is evaluated once. A failed holdout stops this run; do not change the winner and reuse the same test.

Audio acceptance alone does not verify transcript or speaker identity. ASR remains a suggestion. Older frozen datasets without verification evidence cannot start new production training.

### Build A Runnable TensorRT Model

The workflow reaches engine building only after the reference lock and holdout pass. **Build production model** cannot bypass these prerequisites.

1. The Linux builder creates all nine TensorRT 11 engines. Inspect the build report and enqueue verification; an engine file alone is insufficient.
2. Wait for PyTorch → ONNX → TensorRT conversion parity. A parity failure blocks packaging.
3. Inspect the package manifest, checksums, runtime fingerprint and checkpoint/reference lineage in **Models**.
4. Complete formal multilingual and streaming evaluation, canonical benchmarks and required human listening evidence before qualification and promotion.

A built package remains qualification-pending until those checks pass. Failed selection can be retried from **Jobs** using the existing training results; it does not imply another training run is required.

The production chain is:

`frozen dataset → training.prepare → checkpoint.select → reference.select → human lock → holdout.evaluate → engine.prepare → conversion.parity → model.package → evaluation.prepare → qualification → promotion`

## Evaluation Lab

Evaluation Lab compares candidates and composes fail-closed production qualification.

1. Create or select an evaluation project.
2. Choose the audio language, a completed baseline run, and the upstream package dependency.
3. Queue evaluation, or queue the engine → package → evaluation chain.
4. Inspect multilingual CER/WER, speaker similarity, streaming parity, duration, TTFA, RTF, expression results, and long-form evidence.
5. Use blind A/B audio where available.
6. Compose qualification only after automated evaluation, long-form listening, expression listening, and security evidence are each selected.

Missing evidence stays visibly unavailable; Studio does not turn an incomplete result into a passing qualification.

## Models And TensorRT Engines

**Models** shows artifact lineage from dataset to checkpoint, expression bank, engine, package, and qualification. Select an artifact to inspect its file, checksum, parents, metrics, and promotion state.

**Engines** lists device-specific TensorRT artifacts and runtime compatibility. A verified package is tied to its TensorRT/CUDA/GPU fingerprint. Rebuild the engine when the target fingerprint changes instead of reusing a stale artifact.

Promotion requires passed qualification evidence. Selecting an artifact alone does not make it production-ready.

## GPU Jobs

Jobs serializes GPU-exclusive work and preserves dependencies.

- Choose the job type, project, priority, and prerequisite jobs.
- Inspect queue reasons, attempt number, dependencies, status, progress, and event logs.
- Use pause/resume/cancel/retry only when the current state exposes that action.
- Close Studio only after work has reached a safe checkpoint.

Docker-backed jobs require the configured Linux worker. If the worker is unavailable, jobs remain blocked with a reason instead of silently running neural work on Windows.

## Settings

Settings controls default speech/evaluation languages, session continuity, training preset, target-speaker threshold, refresh interval, and Overview motion. Select **Save settings** after changes.

**Import roots** define which local paths Studio may browse or accept by drag and drop. **Worker/runtime paths** select controlled directories used by Docker jobs; use the adjacent folder picker and verify the resolved value rather than typing an unverified path. **AI Components** manages pinned worker assets and offline bundles. These settings affect future jobs and do not rewrite completed artifacts.

The application section shows browser/PWA status. When installation is offered, **Install AnifLive-TTS Studio** adds the local web app without replacing the normal browser route.

### Clear Visible Records

Select **Clear visible records**, review the confirmation, then select **Clear interface history**. Studio records the current time and hides older projects, jobs, artifacts, qualifications, and local expression drafts from list views. It does not delete database rows, audio, model packages, checkpoints, reports, or files. New records appear normally after the cutoff.

## Keyboard And Accessibility

- Use `Tab` to reach controls.
- On any styled selector, use `Enter` or `Space` to open, arrow keys to move, `Home`/`End` to jump, `Enter` to choose, and `Escape` to close.
- Module shortcuts `1`–`9` open the main work areas and `0` opens Jobs when focus is not in a text field.
- Every status or expression is identified by text as well as color.

## Troubleshooting

- **Studio does not start:** run `run_studio.bat` from a terminal and read the retained log. Verify Python 3.10–3.12 and the required packages.
- **Port 9891 is in use:** stop the existing Studio process; the launcher deliberately does not terminate unknown processes.
- **Synthesis is offline:** start the AnifLive-TTS API and refresh Studio. Data workflows remain usable offline.
- **A voice does not appear in Synthesis:** confirm that it is a complete model package with all nine verified TensorRT engines, then restart or reload the API. Raw `.ckpt` and `.pth` files are not selectable runtime voices.
- **Play now remains disabled:** select an API-reported model, enter non-empty text, resolve invalid expression segments, and confirm that the API status is ready.
- **A selector opens but nothing changes:** choose an enabled row rather than the placeholder, then confirm the field label updated. If every row is disabled, create or complete the prerequisite shown in its status text.
- **Browse does not show a usable path:** choose a file or folder inside a configured import root, or add the parent directory under Settings and retry. Browser security prevents arbitrary filesystem discovery.
- **GPU job is blocked:** open Jobs and inspect its dependency/resource reason; verify Docker Desktop, NVIDIA container access, and worker configuration.
- **Preparation says a component is missing:** open **Settings → AI Components** and install or import the exact pinned component. The worker will not download assets while a job is running.
- **A file cannot be imported:** place it under a configured import root. Studio rejects path traversal and unapproved locations.
- **A model cannot be promoted:** inspect the qualification composer and supply every required evidence artifact.
