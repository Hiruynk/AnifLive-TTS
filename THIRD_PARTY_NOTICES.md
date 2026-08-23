# Third-Party Notices

Original AnifLive-TTS code is licensed under the PolyForm Noncommercial License
1.0.0. This does not replace or relicense upstream and third-party portions.
The release bundles retain the following upstream notices and licenses.

| Component | License | Use |
|---|---|---|
| GPT-SoVITS | MIT | Model architecture, multilingual frontend and inference conventions; license retained in `licenses/GPT-SoVITS-MIT.txt`, with relevant vendored changes recorded in `minimal_inference/ANIFLIVE_TTS_PROVENANCE.json` |
| GPT-SoVITS minimal inference | Apache-2.0 | ONNX export and TensorRT inference implementation; license retained in `minimal_inference/LICENSE`, with modified-file notices and provenance in `minimal_inference/ANIFLIVE_TTS_PROVENANCE.json` |
| GPT-SoVITS C++ | Apache-2.0 | Hot-path and persistent-buffer design reference; retained where applicable |
| NVIDIA TensorRT | NVIDIA Software License | TensorRT 11 runtime and engine builder |
| PyTorch | BSD-3-Clause | Tensor and CUDA interoperability |
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
