# Docker 部署

**CUDA／TensorRT（Docker）** 提供 v2ProPlus（32 kHz）及 v5Turbo（48 kHz、FP16、4-step CFM、CFG 0）。模型、共享資源及快取使用主機綁定掛載；先準備模型套件，再啟動 API。

已測試主機為 RTX 5070 Ti，使用 CUDA 12.8、TensorRT 11 及 Triton。**Apple Silicon（macOS／MLX／Metal）** 請參閱 [macOS 安裝指南](MACOS_INSTALLATION_ZH.md)。

```powershell
Copy-Item .env.example .env
```

設定主機與容器的路徑對應：

```dotenv
ANIFLIVE_TTS_MODELS_DIR=D:/models/packages
ANIFLIVE_TTS_MODEL_PACKAGE=/data/models/my-v2proplus
ANIFLIVE_TTS_SHARED_HOST_DIR=D:/models/shared
ANIFLIVE_TTS_CACHE_HOST_DIR=D:/models/cache
ANIFLIVE_TTS_REPORTS_DIR=D:/models/reports
```

僅適用於 v2ProPlus 套件：先建立映像，再於實際提供服務的 Linux 容器內重建硬件
專屬引擎，並執行真實 enqueue 驗證：

```powershell
docker compose build
docker compose run --rm --entrypoint aniflive-tts aniflive-tts `
  model rebuild-engines --model-package /data/models/my-v2proplus
docker compose run --rm --entrypoint aniflive-tts aniflive-tts `
  validate --model-package /data/models/my-v2proplus `
  --shared-dir /data/shared --source-dir /app/minimal_inference --enqueue
.\scripts\run_docker.ps1 -CudaProfile cu128
```

## v5Turbo 套件

依照 [v5Turbo 建置指南](V1_5_V5_BUILD.md)準備 `gsv-v5turbo` 套件。上方 `model rebuild-engines` 用於 v2ProPlus；v5Turbo 使用專屬建置腳本。重新建置引擎後，先在目標 GPU 驗證再部署。

如使用 `cu126`，兩個 `docker compose` 命令都要加入
`-f docker-compose.yml -f docker-compose.cu126.yml`，並以
`-CudaProfile cu126` 啟動。

後續啟動會強制使用 `--pull never --no-build`。只刪除容器而保留映像及主機綁定
掛載時，不會重新下載、安裝或建立；刪除映像後才需再次明確建立。模型套件缺失或
引擎指紋不匹配時會回報 `ENGINE_REBUILD_REQUIRED`，不會在 API 啟動時動態修復。

正式發佈映像包含 OCI 來源版本資料；release workflow 亦會為每個 CUDA 配置輸出
不可變 digest、從映像擷取的 SPDX SBOM 與 BuildKit provenance 記錄。變更 Package
可見度或部署已發佈 tag 前，請依照[發佈映像驗證](RELEASE_VERIFICATION.md)核對。

CUDA 12.8 已在 RTX 5070 Ti 測試。CUDA 12.6 採用相同的 PyTorch 2.10 基線，已有原始碼及建置檢查，GPU 運行仍須在相容主機驗證。

模型／Reference 資料及路由憑證放在持久主機目錄與部署設定中。
