# FunASR, SenseVoice and FSMN-VAD Notice

AnifLive-TTS Studio can install these workstation components through an
explicit setup action. They are not stored in this source repository and are
never downloaded by a queued worker.

| Component | Pinned revision | License |
|---|---|---|
| FunASR Python runtime | 1.4.11 | MIT |
| FunAudioLLM/SenseVoiceSmall | 3847d57b6bdf2dd8875cb1508d2af43d80a16bf7 | FunASR Model Open Source License 1.1 |
| funasr/fsmn-vad | df20e6b30c653645fa4ff125cacfcabd1020a669 | Apache-2.0 |

The complete file inventory, immutable download URLs, byte counts and SHA-256
digests are recorded in `src/aniflive_tts/workstation_assets_lock.json`.

Upstream projects:

- https://github.com/modelscope/FunASR
- https://huggingface.co/FunAudioLLM/SenseVoiceSmall
- https://huggingface.co/funasr/fsmn-vad

Operators must review the upstream model and dataset terms before installing
or redistributing these components.
