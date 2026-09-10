# AnifLive-TTS v1.4.0

## English

AnifLive-TTS v1.4 adds the local Studio training workstation while preserving
the V2ProPlus FP16 TensorRT 11 inference path.

### Highlights

- Adds dataset preparation, review, training, checkpoint selection, conversion,
  evaluation and qualified model installation in Studio.
- Uses dataset-duration training recommendations and human listening to guide
  checkpoint selection; dataset length alone does not guarantee voice quality.
- Supports epoch-boundary pause, real checkpoint continuation and managed GPU
  handoff between training/evaluation workers and inference.
- Stores semantic sampling with each newly exported model package and preserves
  the existing sampling behavior of older packages.
- Adds a Docker Studio launcher and supports named model folders on first use.

### Quality and architecture decision

The Odette v2 workflow and a separate copy of approved training material were
used to exercise the pipeline. Human listening covers multilingual content,
repeated words, long sentences and expression playback. These checks guide
generic pipeline behavior; each newly trained voice still needs its own review.

The performance comparison retains the v1.3 Japanese sentence
「今日はいい天気ですね。」 and accepts comparable performance rather than
requiring identical timing. A model without a historical baseline records
measured performance without claiming a regression comparison.

### WebUI

Run `run_studio_docker.bat` to open Studio at `127.0.0.1:9891`.
The public source package excludes local login pages, accounts and passwords.
Existing local data is preserved when the same launcher is run again.

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
- 根據資料集時長提供訓練建議，配合人工盲聽選擇檢查點；音頻長度本身不能保證音質。
- 支援在訓練回合邊界暫停、從真實檢查點續訓，以及訓練／評估與推理之間的 GPU 交接。
- 新匯出的模型套件會記錄語意採樣策略，舊套件保留原有採樣行為。
- 新增 Docker Studio 啟動器，首次使用亦支援具名模型資料夾。

### 音質與架構決定

以 Odette v2 流程及一份獨立的已審核訓練素材副本進行實測，人工盲聽涵蓋多語內容、
重複字詞、長句及情感播放。修補反映於通用管線；每個新訓練的音色仍需個別驗收。

效能比較沿用 v1.3 的日文句子「今日はいい天気ですね。」，要求成績接近，
不要求每次計時完全一致。沒有歷史基準的模型只記錄實測效能，不會宣稱通過歷史回歸比較。

### WebUI

執行 `run_studio_docker.bat`，Studio 位址為 `127.0.0.1:9891`。
公版原始碼不包含本機登入頁、帳戶及密碼；再次執行同一啟動器會保留既有本機資料。

### 執行環境

- API 版本：`1.4.0`
- 後端：TensorRT 11、FP16
- 語言：`zh`、`yue`、`en`、`ja`、`ko`
- Docker 配置：預設 `cu128`；另準備 `cu126` 相容建置，未在本次驗收電腦執行 GPU 測試。

AnifLive-TTS 原創程式碼繼續採用 PolyForm Noncommercial 1.0.0；商業使用須
另行取得 Hiruynk 的書面商業授權，上游與第三方組件維持各自授權。
