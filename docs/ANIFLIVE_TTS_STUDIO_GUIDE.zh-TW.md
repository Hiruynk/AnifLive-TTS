# AnifLive-TTS Studio 使用說明

[English](ANIFLIVE_TTS_STUDIO_GUIDE.md) · [简体中文](ANIFLIVE_TTS_STUDIO_GUIDE.zh-CN.md)

AnifLive-TTS Studio 是 AnifLive-TTS v1.4 的本機語音製作工作站，把語音合成、情感參考、資料集處理、目標說話者擷取、V2ProPlus 訓練、評估、TensorRT 封裝與 GPU 工作管理集中在同一介面。原本的 AnifLive-TTS WebUI 仍會獨立保留。

## 啟動 Studio

1. 啟動 Docker Desktop，等待 Linux engine 就緒。資料集神經 worker、訓練、評估及 TensorRT build 只在 Linux Docker 執行，不在 Windows host process 執行。
2. 需要推理 API 時雙擊 `run_tts.bat`，等候 health check 顯示 ready。
3. 雙擊專案根目錄的 `run_studio.bat`，只啟動 Studio。
4. 若瀏覽器沒有自動開啟，前往 `http://127.0.0.1:9891/`。
5. 保持各 launcher 終端開啟；關閉視窗會停止該 launcher 所屬服務。

Studio 會偵測 `9880` 或 `9882` 的 API。兩者皆離線時，資料、專案和工作管理仍可使用，但依賴推理的功能會保持不可用。工作站中繼資料預設儲存在 `data/workstation`。

如需使用非預設 API 位址，請在啟動 Studio 前設定 `ANIFLIVE_TTS_WEBUI_UPSTREAM`。它只會改變語音合成 upstream，不會改變 Studio 的 `9891` port。

右上角地球按鈕可在 English、繁體中文、簡體中文之間切換，設定會同步套用所有 Studio 頁面及內嵌的語音合成工作站。

## 介面基本操作

- 左側導覽依照製作流程排列：Create、Data、Train、Deploy、System。包括目標說話者擷取在內的所有資料取得，都由 **資料集** 開始。
- 地球按鈕控制 Studio 介面語言；語音語言是語音合成、資料集轉錄與評估中的獨立設定。
- 風格化選單不是瀏覽器原生下拉框。點選欄位，在浮動面板選擇項目，並確認欄位標籤已更新才繼續。
- 檔案與資料夾欄位支援 **選擇**、拖放及允許的手動路徑。只有欄位顯示解析後路徑才代表已接收；路徑必須位於已設定的 import root。
- 不可用按鈕會顯示原因，通常是尚未選擇必要專案、artifact、AI 元件、API 或 qualification evidence。
- 建立或排程操作會先顯示過渡狀態，再把新紀錄加入清單；進度狀態仍在時不要重複送出同一操作。

## 總覽

總覽顯示使用中的模型、推理環境狀態、專案統計、執行中工作及最近工作。可直接開啟最近專案，或進入負責下一階段的模組。所有語音資料取得都從 **資料集** 開始；目標說話者擷取不是另一種普通使用者專案。

全畫面動態背景可在 **設定 → 工作站行為** 關閉；系統的減少動態效果設定也會自動套用。

## 語音合成

語音合成頁保留完整的 AnifLive-TTS v1.3 合成工作站功能。

1. 選擇音色模型。
2. 輸入文字；預設的 `今日はいい天気ですね。` 可以直接取代。
3. 選擇語音語言。
4. 反白指定文字，替該段選擇情感參考。
5. 檢查情感 tag、每段語言、停頓與順序。
6. 點擊 **立即播放** 串流播放，或下載完整音訊。

播放時，正在讀出的文字會變成金色；情感底線與描述固定在原本標註位置。完全沒有標註時，模型使用原生平靜表達。

分段設定可為每一個已確認片段指定不同語言與停頓。可拖曳把手或使用可存取的移動按鈕排序。**停止** 只取消目前串流，不會更改稿件。

## 情感資料庫

情感資料庫同時管理模型套件內的情感與本機參考草稿。

1. 開啟 **情感**，點擊 **新增情感**。
2. 輸入穩定的 profile ID、名稱、模型 ID、參考音訊路徑、語言、情感與強度。
3. 加入一或多行描述，以及選填的 VAD／韻律資料。
4. 點擊 **分析** 量測時長、起音、音高、語速和參考音訊識別資料。
5. 儲存草稿。

草稿不能靠手動切換狀態成為已驗證情感。必須選擇已通過的評估證據後再升級，保留參考音訊、量測資料與驗收結果之間的可追溯關係。

## 資料集工廠

資料集工廠是建立訓練資料的唯一入口。點擊 **新增資料集** 後選擇一種取得模式：

- **音訊／影片集合**：處理本身大致為單一說話者的錄音。
- **擷取目標說話者**：使用一段短參考音訊，從長錄音或多人媒體找出指定說話者。
- **現有 GPT-SoVITS 資料集**：匯入已審核的 `.list`，不會使用文字稿作檔名。

每個專案都沿用同一條可見流程：**來源 → 說話者 → 清理 → 文字 → 審核 → 風格 → 就緒**。

### 音訊或影片集合

1. 加入一或多個位於允許目錄內的音訊、影片或資料夾路徑。
2. 選擇受管理的語音辨識後端；五語預設是 SenseVoice Small，安裝相應元件後亦可使用 Faster Whisper。
3. 點擊 **開始準備**。Linux Docker 會解碼媒體、產生標準 PCM、執行受管理 VAD、以靜音安全位置切片、轉寫並記錄訊號品質。
4. 在片段檢查器修正文字和語言、確認說話者、按需要加入情感，再接受或拒絕。

### 擷取目標說話者

1. 加入目標短參考音訊和一或多段來源錄音；不需要已訓練的音色套件。
2. 點擊 **開始準備**。Linux worker 先把參考音訊中的有效語音建立成多區段身分原型，再依次執行受管理 FSMN-VAD、工作站專用 TensorRT ERes2NetV2 驗證、雙門檻 Sortformer 證據、純度路由、只用於可挽救候選的 MossFormer2，以及受管理 ASR。
3. 乾淨目標片段保留原音；疑似污染送往分離；不確定結果保留在審核；非目標片段保留在拒絕區。
4. 只有需要檢查每階段證據時才展開 **進階說話者擷取**；它會顯示參考支援數、主要與敏感 diarization、乾淨 SNR gate、離線狀態和整條工作鏈。它是檢查器，不是第二條流程。

目標說話者模式會為每個來源範圍產生 `seg_000184_7f42a1.wav` 一類穩定檔名，不會先串成一條 WAV 再重新 VAD。每個項目保留來源 checksum、sample 級範圍、目標相似度、路由、分離證據、原音、選填處理音訊與 ASR 來源。

此取得政策亦參考公開 [Timbre](https://github.com/Etherll/Timbre) 專案的證據優先思路：多參考身分、先 diarization 後分離、靜音驗證切點、強制切割隔離、驗證融合、保留拒絕片段及可續接的 stage 成品。AnifLive-TTS 使用自己的通用 TensorRT verifier、worker contract 與品質 gate，沒有捆綁 Timbre 原始碼或模型資產。

### 審核、風格與凍結

1. 點擊 **審核佇列** 並使用 Studio 播放器；每一段 ASR 建議都必須人工確認，不能自行成為正式訓練文字。
2. 使用 `Space` 播放，再用畫面上的接受／拒絕控制決定片段。非靜音強制切割、身分不明確或分離後 gate 失敗都會留在審核。
3. 按需要加入情感和強度。SenseVoice 情感只作建議，人工標籤才是正式資料。
4. 點擊 **情感候選**，按說話者身分、訊號品質、可用時長及標註完整度排序。採用候選只會建立情感資料庫草稿，不會自動成為已驗證 profile。
5. 分別核實每個已接受片段的逐字稿與說話者；修改內容後，原核實會失效。分配 train／validation／test（預設 85／10／5）後點擊 **凍結資料集**。只有要求情感標註的專案才需要核實情感。
6. 可下載 **Manifest** 或只根據證據生成的 **驗收報告**；缺少訓練、引擎或正式驗收時會逐項列出，不會猜測為成功。
7. 點擊 **繼續前往訓練**，Studio 會從已凍結 manifest 建立原有 Training 專案，不會在 Dataset Factory 重造一套訓練 UI。

來源媒體與生成檔案仍留在磁碟；Studio 只記錄來源關係與審核狀態，不會把任意檔案塞進瀏覽器儲存空間。

## AI 元件

第一次執行神經網絡資料流程前，開啟 **設定 → AI 元件**。

- 可逐個明確安裝必要元件，或一次安裝所有缺少的必要元件。
- 完整目標說話者流程需要 Speaker Verification、SenseVoice Small、FSMN-VAD 和 MossFormer2 全部顯示 **Ready**。
- 每個資產都固定 revision、大小、SHA-256、license 與 Linux worker runtime。
- 有網絡的電腦使用 **匯出 AI 元件套裝**，離線工作站使用 **匯入 AI 元件套裝**；正常 worker 永遠以 `--network none` 執行。
- DeepFilterNet 是選填元件；在有後端通過 Linux Docker 品質 gate 前，去混響保持不可用。

## 模型訓練

訓練使用 GPT-SoVITS V2ProPlus 輸入，透過 GPU 工作系統在 Linux Docker 執行。

1. 凍結已審核資料集後按 **繼續訓練**，或建立訓練專案時選取合格的 frozen dataset。
2. 選擇 **快速**、**平衡**、**高品質** 或 **進階**。進階設定包括階段、epoch、batch、learning rate、checkpoint 間隔、seed 與 gradient checkpointing。
3. 排入訓練並監察狀態、loss、VRAM、GPU 使用率、溫度、ETA、log 和 checkpoint。在 **工作** 頁使用 worker 支援的暫停、繼續、取消及重試功能。
4. 等待 validation checkpoint selection。保存的 GPT／SoVITS epoch 都是候選，最後一個 epoch 不會自動成為部署勝者。
5. 聆聽本次新產生的盲選 reference 音訊並提交決定；先前 run 的選擇不適用於新一組音訊。候選只來自已核實的 train 片段。
6. checkpoint 勝者及 reference 鎖定後，才首次開啟 sealed test，且只評估一次。holdout 失敗即停止該 run，不可更換勝者後重用同一 test。

接受音訊不等於核實逐字稿或說話者，ASR 仍只是建議。沒有核實證據的舊 frozen dataset 不可建立新的 production training。

### 建立可執行的 TensorRT 模型

只有 reference 鎖定及 holdout 通過後，流程才會進入引擎建置。**建立 production model** 不會略過這些條件。

1. Linux builder 建立全部九個 TensorRT 11 engine。檢查 build report 與 enqueue verification；只有 engine 檔案並不足夠。
2. 等待 PyTorch → ONNX → TensorRT conversion parity。parity 失敗會阻止封裝。
3. 在 **模型** 檢查 package manifest、checksum、runtime fingerprint 及 checkpoint／reference 來源關係。
4. 完成正式多語言與串流評估、canonical benchmark 及必要人工盲聽後，才可 qualification 與 promotion。

已建好的 package 在所有關卡通過前仍是 qualification-pending。selection 失敗可以在 **工作** 頁重用既有訓練成果重試，不代表必須重新訓練。

完整 production chain：

`frozen dataset → training.prepare → checkpoint.select → reference.select → human lock → holdout.evaluate → engine.prepare → conversion.parity → model.package → evaluation.prepare → qualification → promotion`

## 評估實驗室

評估實驗室比較候選模型，並以 fail-closed 方式組合正式驗收。

1. 建立或選擇評估專案。
2. 選擇音訊語言、已完成的基線與上游模型套件依賴。
3. 單獨排入評估，或排入 engine → package → evaluation 工作鏈。
4. 檢查五語 CER／WER、speaker similarity、streaming parity、時長、TTFA、RTF、情感及長句結果。
5. 有盲聽音訊時進行 A/B 比較。
6. 自動評估、長句盲聽、情感盲聽及安全證據全部選齊後，才可組合 qualification。

缺少證據時會明確保持不可用，不會把未完成結果包裝成通過。

## 模型與 TensorRT 引擎

**模型** 顯示資料集、checkpoint、情感資料庫、engine、package 和 qualification 的完整來源關係。選取成品可查看路徑、checksum、父成品、量測結果與升級狀態。

**引擎** 顯示特定裝置的 TensorRT 成品及相容性。已驗證套件綁定 TensorRT／CUDA／GPU fingerprint；目標環境更換後應重建，不要沿用過期 engine。

只有通過 qualification 的成品可以升級；單純選取成品不會令它成為 production-ready。

## GPU 工作

工作頁會序列化需要獨占 GPU 的流程並保存依賴關係。

- 選擇工作類型、專案、優先級與前置工作。
- 檢查等待原因、嘗試次數、依賴、狀態、進度及事件 log。
- 只有目前狀態容許時，才會顯示暫停／繼續／取消／重試。
- 長時間工作應到達安全 checkpoint 後才關閉 Studio。

Docker 工作需要已設定的 Linux worker。worker 不可用時，工作會保留在 blocked 狀態並顯示原因，不會暗中改到 Windows 執行神經網絡。

## 設定

設定頁控制預設語音／評估語言、session continuity、訓練 preset、目標說話者門檻、重新整理間隔及總覽背景。修改後點擊 **儲存設定**。

**Import roots** 決定 Studio 可瀏覽或接受拖放的本地路徑。**Worker/runtime paths** 選擇 Docker 工作使用的受控目錄；請使用旁邊的資料夾選擇器並核對解析後值，不要依賴未驗證的手動路徑。**AI 元件** 管理固定版本的 worker asset 及離線 bundle。設定只影響之後的工作，不會改寫已完成 artifact。

應用程式區顯示瀏覽器／PWA 狀態；系統提供安裝時，**安裝 AnifLive-TTS Studio** 可加入本機 Web App，不會取代一般瀏覽器網址。

### 清除介面紀錄

點擊 **清除介面紀錄**，閱讀確認內容後再點擊確認。Studio 只記錄目前時間，並在列表中隱藏較舊的專案、工作、成品、驗證與本機情感草稿；SQLite 資料列、音訊、模型套件、checkpoint、報告和檔案全部保留。清除後新建立的紀錄會正常顯示。

## 鍵盤與可存取操作

- 使用 `Tab` 移動至各控制項。
- 所有風格化選單都支援 `Enter`／空白鍵開啟、方向鍵移動、`Home`／`End` 跳轉、`Enter` 選擇與 `Escape` 關閉。
- 焦點不在文字輸入框時，`1`–`9` 可開啟主要模組，`0` 開啟工作頁。
- 狀態和情感不只靠顏色辨識，同時保留文字標籤。

## 疑難排解

- **Studio 無法啟動：** 從終端執行 `run_studio.bat` 並閱讀保留的 log；確認 Python 3.10–3.12 及必要套件。
- **9891 已被使用：** 停止現有 Studio；launcher 不會擅自終止未知程序。
- **語音合成離線：** 啟動 AnifLive-TTS API 後重新整理；資料工作仍可離線使用。
- **音色沒有出現在語音合成：** 確認它是包含九個已驗證 TensorRT engine 的完整 model package，然後重新載入或重啟 API。單獨 `.ckpt` 與 `.pth` 不能成為可選 runtime voice。
- **立即播放保持不可用：** 選擇 API 回報的模型、輸入非空文字、修正無效情感段落，並確認 API 狀態為 ready。
- **選單打開但選擇後沒有變化：** 請選可用項目而非 placeholder，再確認欄位標籤已更新。若所有項目都不可用，請先完成狀態文字所示的 prerequisite。
- **選擇器沒有返回可用路徑：** 選擇設定 import root 內的檔案或資料夾，或先在設定加入其上層目錄再重試；瀏覽器安全限制不允許任意探索檔案系統。
- **GPU 工作 blocked：** 到工作頁檢查依賴／資源原因，確認 Docker Desktop、NVIDIA container 及 worker 設定。
- **準備流程顯示缺少元件：** 到 **設定 → AI 元件** 安裝或匯入指定固定版本；worker 執行期間不會自行下載。
- **無法匯入檔案：** 把檔案放在允許的 import root；Studio 會拒絕路徑穿越與未批准位置。
- **無法升級模型：** 到 qualification composer 補齊所有必要證據。
