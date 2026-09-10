# Third-Party Notices

Original AnifLive-TTS code is licensed under the PolyForm Noncommercial License
1.0.0. This does not replace or relicense upstream and third-party portions.
The release bundles retain the following upstream notices and licenses.

| Component | License | Use |
|---|---|---|
| GPT-SoVITS | MIT | Model architecture, multilingual frontend and inference conventions; license retained in `licenses/GPT-SoVITS-MIT.txt`, with relevant vendored changes recorded in `minimal_inference/ANIFLIVE_TTS_PROVENANCE.json` |
| GPT-SoVITS minimal inference | Apache-2.0 | ONNX export and TensorRT inference implementation; license retained in `minimal_inference/LICENSE`, with modified-file notices and provenance in `minimal_inference/ANIFLIVE_TTS_PROVENANCE.json` |
| GPT-SoVITS C++ | Apache-2.0 | Hot-path and persistent-buffer design reference; retained where applicable |
| ClearerVoice-Studio `MossFormer2_SS_16K` source | Apache-2.0 | Linux Docker speech-separation backend, fetched at source revision `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`; source archive SHA-256 `f8f8d2f2190b9909b51e91ce886d1c7efedb7349b3ff6d3ab521451165fbd8da`. The worker image retains the upstream license at `/opt/aniflive-tts/clearvoice-source/LICENSE`; repository notice is in `licenses/CLEARERVOICE-MOSSFORMER2-NOTICE.txt` |
| `alibabasglab/MossFormer2_SS_16K` checkpoint | Apache-2.0 model card metadata | External, operator-supplied two-speaker separation weights pinned to revision `407cb030cd66340918ebb6c8cc63b18f8592cdbe`; `last_best_checkpoint.pt` SHA-256 `00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a`. The 670,353,271-byte checkpoint is not bundled in the repository or release image |
| FunASR 1.4.11 | MIT | Offline Linux-worker runtime for the managed SenseVoice and FSMN-VAD components; no worker downloads models or code at execution time |
| `FunAudioLLM/SenseVoiceSmall` | FunASR Model Open Source License 1.1 | Managed five-language ASR and non-authoritative emotion/event suggestions, pinned to revision `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7`; exact file inventory and SHA-256 values are in `src/aniflive_tts/workstation_assets_lock.json` |
| `Systran/faster-whisper-small` | MIT | Optional managed multilingual CTranslate2 ASR backend, pinned to revision `536b0662742c02347bc0e980a01041f333bce120`; exact file inventory and SHA-256 values are in `src/aniflive_tts/workstation_assets_lock.json` |
| `funasr/fsmn-vad` | Apache-2.0 | Managed target-speaker voice-activity detection, pinned to revision `df20e6b30c653645fa4ff125cacfcabd1020a669`; exact file inventory and SHA-256 values are in `src/aniflive_tts/workstation_assets_lock.json` |
| NVIDIA NeMo-Speech.cpp 0.1.0 | Apache-2.0 | Pinned native CUDA runtime for isolated Sortformer diarization; the official release archive is checksum-verified during the worker build and retains its upstream licenses and notices. See `licenses/NVIDIA-NEMO-SPEECH-SORTFORMER-NOTICE.md` |
| `nvidia/diar_streaming_sortformer_4spk-v2` | CC BY 4.0 | Managed diarization and overlap-evidence model, pinned to revision `5240a64075176943f677d30fa2171c780229f341`; the Q8_0 GGUF fingerprint is locked in `src/aniflive_tts/workstation_assets_lock.json`. See `licenses/NVIDIA-NEMO-SPEECH-SORTFORMER-NOTICE.md` |
| NVIDIA TensorRT | NVIDIA Software License | TensorRT 11 runtime and engine builder |
| PyTorch | BSD-3-Clause | Tensor and CUDA interoperability |
| TorchCodec | BSD-3-Clause | Pinned Linux worker audio decoding backend used by torchaudio during speaker-vector preprocessing; license retained in `licenses/TORCHCODEC-BSD-3-CLAUSE.txt` |
| ONNX / ONNX Runtime | Apache-2.0 / MIT | Portable graph format and validation |
| g2p-en 2.1.0 | Apache-2.0 | Minimal English grapheme-to-phoneme runtime vendored without its unused Distance dependency; license retained beside the source, with modifications recorded in `minimal_inference/ANIFLIVE_TTS_PROVENANCE.json` |
| NLTK library | Apache-2.0 | English tokenization and data loading library |
| NLTK averaged perceptron taggers | MIT | `averaged_perceptron_tagger` and `averaged_perceptron_tagger_eng`; see `licenses/NLTK-DATA-NOTICES.md` |
| CMU Pronouncing Dictionary | CMU terms | Pronunciation data distributed through NLTK Data; see `licenses/CMUDICT-NOTICE.txt` |
| fast-langdetect | MIT | Language detection runtime; see `licenses/FAST-LANGDETECT-MIT.txt` |
| fastText `lid.176.bin` | CC BY-SA 3.0 | Language-identification model redistributed unchanged in the container image; see `licenses/FASTTEXT-LID-CC-BY-SA-3.0.txt` |
| inflect | MIT | English text normalization required by g2p-en |
| mecab-python3 / MeCab | BSD-3-Clause | Japanese tokenization and the Windows Korean G2P adapter; see `licenses/MECAB-BSD-3-CLAUSE.txt` |
| ipadic | NAIST / ICOT terms | Japanese dictionary data packaged for MeCab; see `licenses/IPADIC-NAIST-ICOT.txt` |
| python-mecab-ko | BSD-3-Clause | Korean morphological analysis on Windows and Linux |
| python-mecab-ko-dic | Apache-2.0 | Korean dictionary used by python-mecab-ko |
| Lucide 1.8.0 / Feather-derived icons | ISC / MIT | Offline WebUI icon runtime; licenses retained in `licenses/LUCIDE-ISC-MIT.txt` |
| [Mixkit: "Abstract Background in Grayscale"](https://mixkit.co/free-stock-video/abstract-background-in-grayscale-101010/) | [Mixkit Stock Video Free License](https://mixkit.co/license/#videoFree) | Full-screen local WebUI motion background from the [Mixkit contributor profile](https://mixkit.co/@mixkit/); source, hashes, and license record are retained in `webui/media/README.md` and `licenses/MIXKIT-STOCK-VIDEO-FREE-LICENSE.txt` |
| Docker CLI 29.8.0 | Apache-2.0 | Local Studio container management; see `licenses/docker-cli-LICENSE` |
| LibriSpeech | CC BY 4.0 | External validation reference only; not distributed |

Apache-2.0-derived files changed for this distribution carry a prominent
`Modified by AnifLive-TTS` notice. The machine-readable provenance record names
the upstream revisions, baseline file hashes, and modification categories; it
does not relicense any upstream work.

Model checkpoints, reference recordings and generated TensorRT engines are not
part of the AnifLive-TTS source distribution. Users are responsible for the
license and consent status of every model package they create.

The production Docker runtime intentionally excludes Distance, eunjeon and
chardet. They are not imported by the AnifLive-TTS serving path and are not
included in the release SBOM. This notice is informational and is not legal
advice.
