# AnifLive-TTS v1.4.0

## English

AnifLive-TTS v1.4 adds the local Studio training workstation while preserving
the V2ProPlus FP16 TensorRT 11 inference path.

### Highlights

- Adds dataset preparation, review, training, checkpoint selection, conversion,
  evaluation and qualified model installation in Studio.
- Uses dataset-duration training recommendations and human listening to guide
  checkpoint selection.
- Supports epoch-boundary pause, real checkpoint continuation and managed GPU
  handoff between training/evaluation workers and inference.
- Stores semantic sampling with each newly exported model package and preserves
  the existing sampling behavior of older packages.
- Adds a Docker Studio launcher and supports named model folders on first use.

### Studio and WebUI

Studio and WebUI coexist. Studio provides the dataset, training, conversion,
evaluation and model-management workstation; run `run_studio_docker.bat` and
open `127.0.0.1:9891`. WebUI remains the lightweight model/expression/speech
interface; start the API, run `run_webui.bat` and open `127.0.0.1:9890`.
See the [Studio guide](https://github.com/Hiruynk/AnifLive-TTS/blob/main/docs/ANIFLIVE_TTS_STUDIO_GUIDE.md) for the complete workflow.

### Runtime

- API version: `1.4.0`
- Backend: TensorRT 11, FP16
- Languages: `zh`, `yue`, `en`, `ja`, `ko`
- Docker profiles: `cu128` by default; `cu126` is prepared as a compatibility
  build and has not been GPU-tested on the current acceptance machine.

Original AnifLive-TTS code remains licensed under PolyForm Noncommercial
1.0.0. Commercial use requires a separate written Commercial License from
Hiruynk; upstream and third-party components retain their respective licenses.

## 繁體中文

AnifLive-TTS v1.4 加入本機 Studio 訓練工作站，並保留既有 V2ProPlus FP16
TensorRT 11 推理路徑。

### 主要更新

- Studio 支援資料集準備、審核、訓練、檢查點選擇、轉換、評估及合格模型安裝。
- 根據資料集時長提供訓練建議，配合人工盲聽選擇檢查點。
- 支援在訓練回合邊界暫停、從真實檢查點續訓，以及訓練／評估與推理之間的 GPU 交接。
- 新匯出的模型套件會記錄語意採樣策略，舊套件保留原有採樣行為。
- 新增 Docker Studio 啟動器，首次使用亦支援具名模型資料夾。

### Studio 與 WebUI

Studio 與 WebUI 並存。Studio 提供資料集、訓練、轉換、評估及模型管理工作站，
執行 `run_studio_docker.bat` 後開啟 `127.0.0.1:9891`。
WebUI 保留模型選擇、情感控制及語音播放的輕量介面；先啟動 API，再執行
`run_webui.bat`，位址為 `127.0.0.1:9890`。
完整流程請見 [Studio 使用手冊](https://github.com/Hiruynk/AnifLive-TTS/blob/main/docs/ANIFLIVE_TTS_STUDIO_GUIDE.zh-TW.md)。

### 執行環境

- API 版本：`1.4.0`
- 後端：TensorRT 11、FP16
- 語言：`zh`、`yue`、`en`、`ja`、`ko`
- Docker 配置：預設 `cu128`；另準備 `cu126` 相容建置，未在本次驗收電腦執行 GPU 測試。

AnifLive-TTS 原創程式碼繼續採用 PolyForm Noncommercial 1.0.0；商業使用須
另行取得 Hiruynk 的書面商業授權，上游與第三方組件維持各自授權。
