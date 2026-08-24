# Docker 部署

通用映像提供 `cu128` 與暫定的 `cu126` 配置。模型、共享資源、快取與報告
均使用主機綁定掛載。服務入口只會驗證及啟動 API，不會下載依賴或建立引擎。

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

先建立映像，再於實際提供服務的 Linux 容器內重建硬件專屬引擎，並執行真實
enqueue 驗證：

```powershell
docker compose build
docker compose run --rm --entrypoint aniflive-tts aniflive-tts `
  model rebuild-engines --model-package /data/models/my-v2proplus
docker compose run --rm --entrypoint aniflive-tts aniflive-tts `
  validate --model-package /data/models/my-v2proplus `
  --shared-dir /data/shared --source-dir /app/minimal_inference --enqueue
.\scripts\run_docker.ps1 -CudaProfile cu128
```

如使用 `cu126`，兩個 `docker compose` 命令都要加入
`-f docker-compose.yml -f docker-compose.cu126.yml`，並以
`-CudaProfile cu126` 啟動。

後續啟動會強制使用 `--pull never --no-build`。只刪除容器而保留映像及主機綁定
掛載時，不會重新下載、安裝或建立；刪除映像後才需再次明確建立。模型套件缺失或
引擎指紋不匹配時會回報 `ENGINE_REBUILD_REQUIRED`，不會在 API 啟動時動態修復。

正式發佈映像包含 OCI 來源版本資料；release workflow 亦會為每個 CUDA 配置輸出
不可變 digest、從映像擷取的 SPDX SBOM 與 BuildKit provenance 記錄。變更 Package
可見度或部署已發佈 tag 前，請依照[發佈映像驗證](RELEASE_VERIFICATION.md)核對。

CUDA 12.8 已在 RTX 5070 Ti 通過完整 GPU 端到端驗收。CUDA 12.6 相容配置採用
同一個已修補安全問題的 PyTorch 2.10 基線，但目前只完成原始碼及建置規則驗證；
發布或部署 `cu126` 標籤前，仍須在相容主機另行完成驗收。

角色資產與外部路由設定必須放在通用映像與原始碼發行包以外的私有部署覆蓋層。
