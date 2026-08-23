# AnifLive-TTS v1.0.0

## English

### Highlights

- The first-party TTS runtime for AnifEngine-Voice, built for low-latency, high-quality multilingual voice cloning.
- Eight neural inference stages run through FP16 TensorRT 11 with no PyTorch model fallback.
- Canonical, legacy GPT-SoVITS, and OpenAI-compatible APIs support Putonghua/Mandarin (`zh`), Cantonese (`yue`), English (`en`), Japanese (`ja`), and Korean (`ko`).
- v1 launches with GPT-SoVITS V2ProPlus model packaging and inference. More GPT-SoVITS model generations are planned without changing the public API contract.
- Docker images are provided for CUDA 12.8 and CUDA 12.1. Model packages and runtime caches live in mounted host directories, so recreating a container does not download or rebuild them.

### Distribution and Compatibility

- Container tags: `ghcr.io/hiruynk/aniflive-tts:1.0.0-cu128` and `ghcr.io/hiruynk/aniflive-tts:1.0.0-cu121`.
- The source bundle is generated from the release Git commit. Each image workflow produces an immutable digest, source-commit metadata, an image-derived SPDX SBOM, and BuildKit maximum-provenance attestation. Release evidence must match the release commit before use.
- Portable ONNX files move between supported hosts; TensorRT engines do not. Rebuild engines and run `validate --enqueue` inside the target Linux container and GPU before serving.
- The API request path performs no dependency installation, engine building, checkpoint loading, or network download.
- CUDA 12.1 engines pass build and load validation. RTX 5070 Ti requires CUDA 12.8, so CUDA 12.1 GPU end-to-end inference is not claimed on this machine.
- Original AnifLive-TTS code is licensed under PolyForm Noncommercial 1.0.0. Commercial use requires a separate written Commercial License from AnifEngine. Upstream and third-party components retain their respective licenses; see `LICENSING.md`, `THIRD_PARTY_NOTICES.md`, and the attached SBOMs.

## 繁體中文

### 重點

- AnifEngine-Voice 的第一方 TTS，專注於低延遲、高音質的多語言語音複製。
- 八個神經網絡推理階段均使用 FP16 TensorRT 11 執行，不會回退至 PyTorch 模型推理。
- 標準 API、GPT-SoVITS 舊版相容 API 及 OpenAI 相容 API 支援普通話（`zh`）、廣東話（`yue`）、英文（`en`）、日文（`ja`）及韓文（`ko`）。
- v1 首發支援 GPT-SoVITS V2ProPlus 模型封裝與推理，未來會在維持公開 API 規格的前提下支援更多 GPT-SoVITS 模型世代。
- 提供 CUDA 12.8 與 CUDA 12.1 Docker 映像。模型套件與快取保存在掛載的主機目錄，重新建立容器時毋須再次下載或建置。

### 發布與相容性

- 容器標籤：`ghcr.io/hiruynk/aniflive-tts:1.0.0-cu128` 及 `ghcr.io/hiruynk/aniflive-tts:1.0.0-cu121`。
- 原始碼套件由 release 的 Git commit 產生。每個映像 workflow 都會產出不可變 digest、來源 commit 資料、從實際映像擷取的 SPDX SBOM，以及 BuildKit 最高等級 provenance 證明；使用前必須確認整套證據與 release commit 一致。
- 可攜 ONNX 檔案可移至受支援主機，但 TensorRT 引擎不可跨環境通用。提供服務前，必須在目標 Linux 容器與 GPU 內重建引擎，並執行 `validate --enqueue`。
- API 請求期間不會安裝依賴、建立引擎、載入 checkpoint 或從網絡下載檔案。
- CUDA 12.1 引擎已通過建置與載入驗證。RTX 5070 Ti 最低需要 CUDA 12.8，因此本機不宣稱已完成 CUDA 12.1 GPU 端到端推理驗證。
- AnifLive-TTS 原創程式碼採用 PolyForm Noncommercial 1.0.0；商業使用須另行取得 AnifEngine 的書面商業授權。上游及第三方組件維持各自授權，詳見 `LICENSING.md`、`THIRD_PARTY_NOTICES.md` 與 release 附件中的 SBOM。
