# macOS 原生安裝

**Backend：** Apple Silicon（macOS／MLX／Metal）。語音神經推理由 Rust／MLX／Metal worker 執行；Python 提供 HTTP API 及文字前處理。建議使用 FP16。Windows backend 維持 CUDA／TensorRT（Docker）。

## 硬件及 macOS 支援範圍

- 已驗證主機：Mac Studio M5 Max、40-core GPU、128 GB 統一記憶體、macOS 27.0。
- 其他 Apple Silicon 型號（M1–M5 各版本）未測，沒有已驗證的最低記憶體要求。
- Intel Mac 不支援。
- 隨附 MLX library 的部署下限為 macOS 26.2+；目前只在 macOS 27.0 完成驗證。

## 安裝預建 runtime

從專案發行資產下載 `AnifLive-TTS-v1.5.0-macos-arm64-runtime.zip` 及 `aniflive_tts-1.5.0-cp310-abi3-macosx_11_0_arm64.whl` SDK wheel。請在 macOS checkout 使用由你管理的環境設定檔：

```bash
export ANIFLIVE_TTS_MACOS_ENV_FILE="$HOME/.config/aniflive-tts/macos.env"
mkdir -p "$(dirname "$ANIFLIVE_TTS_MACOS_ENV_FILE")"
cp scripts/macos-release/macos-v5turbo.env.example "$ANIFLIVE_TTS_MACOS_ENV_FILE"
```

修改設定檔，指定現有、已授權及已轉換 v5Turbo 模型套件的**可寫副本**、上游前處理 source 及本機 Open JTalk 字典。Dynamic shared-base 套件可不含固定聲線，啟動後再登錄自己的 Voice 和 Reference。模型權重及已轉換資產須另行提供；runtime 壓縮檔和 SDK wheel 不包含它們。請保留未修改的原套件以便回退。

開始安裝前，請閱讀 [第三方授權聲明](../THIRD_PARTY_NOTICES.md)。以下命令使用現有的 `setup_shared_assets.py`，只下載並校驗文字前處理所需的 NLTK 和 fastText 資料，不會下載 GPT-SoVITS 模型權重。首次執行需要網路；已驗證的快取可重用。官方前處理原始碼須放在 `$ANIFLIVE_TTS_SOURCE_DIR/GPT_SoVITS/`，本機 Open JTalk 字典須包含 `sys.dic`。

以下例子假設兩個下載檔位於 `~/Downloads`：

```bash
source "$ANIFLIVE_TTS_MACOS_ENV_FILE"
python3.12 - "$ANIFLIVE_TTS_MACOS_DATA_ROOT" <<'PY'
from pathlib import Path
import shutil
import sys
from scripts.setup_shared_assets import NLTK_PACKAGES, _digest, _download_fasttext, _download_nltk

root = Path(sys.argv[1]).expanduser().resolve()
_download_nltk(root)
cache = root / "cache" / "nltk"
cache.mkdir(parents=True, exist_ok=True)
for package, spec in NLTK_PACKAGES.items():
    source = root / "nltk_data" / spec["subdir"] / f"{package}.zip"
    destination = cache / f"{package}.zip"
    if not destination.is_file() or _digest(destination) != spec["zip_sha256"]:
        shutil.copyfile(source, destination)
    if _digest(destination) != spec["zip_sha256"]:
        raise RuntimeError(f"NLTK cache checksum mismatch: {package}")
_download_fasttext(root / "cache")
PY
mkdir -p "$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT/download"
unzip "$HOME/Downloads/AnifLive-TTS-v1.5.0-macos-arm64-runtime.zip" -d "$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT/download"
ANIFLIVE_TTS_BUILD_WORKER=0 \
  ANIFLIVE_TTS_SDK_WHEEL="$HOME/Downloads/aniflive_tts-1.5.0-cp310-abi3-macosx_11_0_arm64.whl" \
  bash scripts/macos-release/install.sh
"$ANIFLIVE_TTS_VENV/bin/python" scripts/macos-qualification/fp16_package_refresh.py \
  "$ANIFLIVE_TTS_MODEL_PACKAGE" \
  "$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT/download/AnifLive-TTS-v1.5.0-macos-arm64-runtime"
```

安裝程式會安裝鎖定版本的 Python 相依套件及預建 SDK wheel；使用以上設定毋須 Rust 工具鏈。更新工具會替換套件副本中的 worker、MLX libraries 和授權檔，並重新產生套件 metadata 及 checksums；不會提供或轉換模型權重或聲線資料。Source 轉換工具見 `scripts/macos-v5turbo/convert_weights.py` 及 `convert_g2pw.py`；它們本身不會建立完整套件。

以上例子選擇 FP16。使用凍結 FP32 rollback 套件時，請在設定檔中設 `ANIFLIVE_TTS_PRECISION=FP32` 並使用獨立的 FP32 套件；它保留原有功能集。選擇策略前，請查詢 active worker 的 `expression.policies` 及 `expression.complete_policies`。

套件目錄格式見 `scripts/macos-release/model-package.template.json`。
套件包含已轉換的 G2PW 及 BERT 前處理資產時，設定 `native_mandarin: true`；
否則服務的 capability 回應不會列出普通話。

## 從 source 建置

若要自行建置 SDK 和 worker，請安裝 Rust 工具鏈、不設定 `ANIFLIVE_TTS_SDK_WHEEL`，然後執行：

```bash
ANIFLIVE_TTS_BUILD_WORKER=0 bash scripts/macos-release/install.sh
MACOSX_DEPLOYMENT_TARGET=26.2 bash scripts/macos-release/build.sh --output "$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT/build/native-worker"
"$ANIFLIVE_TTS_VENV/bin/python" scripts/macos-qualification/fp16_package_refresh.py "$ANIFLIVE_TTS_MODEL_PACKAGE" "$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT/build/native-worker"
```

使用預設設定時，安裝程式亦可直接建置 worker。

## 啟動及管理服務

請在 repository 根目錄執行。`start.sh` 會在前景執行，並於已設定的 `ANIFLIVE_TTS_PORT` 上綁定 loopback；服務執行時，請在另一個 terminal 執行 health、stop 及 restart 指令。

```bash
bash scripts/macos-release/start.sh
bash scripts/macos-release/health.sh
bash scripts/macos-release/stop.sh
bash scripts/macos-release/restart.sh
```

服務啟動或重新啟動後，可用 `health.sh` 檢查狀態。`stop.sh` 會停止 worker 並釋放本機 port。

## Runtime 功能

目前 FP16 worker 輸出 48 kHz 單聲道 PCM16，使用原生四步 CFM 及 CFG 0。支援完整 WAV、PCM 串流、官方同步／非同步 SDK，以及全部五種 Reference conditioning policy：`full-switch`、`identity-lock`、`semantic-style`、`acoustic-style`、`sv-only`。這些 policy 對應至 v5 可用的語義、風格及 CFM voice-anchor 元件；v5 沒有獨立的 v2ProPlus SV encoder。

已登錄 Voice 及 expression profiles 可由 `/v1/voices` 和 `/v1/expressions` 查詢；使用 `GET /v1/expressions?voice_profile=<id>` 可查詢指定聲線。SDK 請求以 `voice_profile=...` 選擇 Voice；Reference 會按需準備並快取。支援逐段 expression 及混合 policy。Generation 接受 `top_k` 1–50、`top_p` 位於 (0, 1]、大於 0 的 `temperature` 及大於或等於 -1 的整數 `seed`。v5 限定 `speed=1.0`、`noise_scale=0.5`、4 個 sample steps 及 CFG 0。相容介面接受 `auto`、`auto_yue` 及受支援的別名；canonical `ja` 使用日英混合前處理模式。

PCM 串流在 vocoder 階段增量輸出：自回歸生成到達 EOS 並完成完整 CFM 後才輸出第一段 PCM，之後逐個 vocoder window 輸出；不支援 EOS 前語義或聲學串流。取消完整 WAV 會令請求以 HTTP 409 結束，不回傳部分 WAV；服務可恢復並處理下一個請求。Session A 支援 append、Flush、Cancel 及恢復。Session 策略 B–F 仍屬研究功能，並非正式 API 承諾。v2ProPlus 及 Studio 仍屬 Windows 功能。

執行 `ANIFLIVE_TTS_WEBUI_DIR="$PWD/webui" "$ANIFLIVE_TTS_VENV/bin/aniflive-tts" webui --upstream "http://${ANIFLIVE_TTS_HOST}:${ANIFLIVE_TTS_PORT}"`，再開啟 `http://127.0.0.1:9890/`，即可使用共用 WebUI。Mac 瀏覽器播放尚未驗證；Studio workstation 功能仍只支援 Windows。

FP16 使用 PyTorch 2.9.1 MPS 執行 Reference DSP，例如重採樣、STFT 及 Mel 處理。語音神經推理維持在 Rust／MLX／Metal，沒有 CPU、PyTorch 或 MPS 神經 fallback。

性能口徑及比較結果請參閱[性能工程記錄](PERFORMANCE_ENGINEERING.md)、[平台能力矩陣](V1_5_PLATFORM_MATRIX.md)及[API 規格](API_ZH.md#macos-apple-silicon-行為)。
