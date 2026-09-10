(() => {
  "use strict";

  const LOCALE_KEY = "aniflive.uiLocale";
  const SUPPORTED = Object.freeze(["en", "zh-Hant", "zh-Hans"]);
  const NATIVE_NAMES = Object.freeze({ en: "English", "zh-Hant": "繁體中文", "zh-Hans": "简体中文" });
  const TEXT = Object.freeze({
    "Inspect subject": ["檢查驗收對象", "检查验收对象"],
    "Promoting artifact…": ["正在晉升成品…", "正在晋升产物…"],
    "Promotion is in progress. Verification and copying may take some time.": ["正在晉升成品，驗證與複製可能需要一段時間。", "正在晋升产物，验证与复制可能需要一些时间。"],

    "Select required verified sources": ["選取必要的已驗證來源", "选择必要的已验证来源"],
    "Expression blind A/B · if applicable": ["表情盲測 A/B · 適用時提供", "表情盲测 A/B · 适用时提供"],
    "Expression evidence · if applicable": ["表情證據 · 適用時提供", "表情证据 · 适用时提供"],
    "Content review · optional": ["內容人工覆核 · 選填", "内容人工复核 · 选填"],
    "No content review selected · optional": ["未選取內容覆核 · 選填", "未选择内容复核 · 选填"],
    "Choose a different artifact for each selected source.": ["每個已選來源必須使用不同的成品。", "每个已选来源必须使用不同的产物。"],
    "Select the four required sources. The backend verifies whether expression evidence is applicable; content review is optional.": ["請選取四項必要來源。表情證據是否適用由後端驗證；內容人工覆核為選填。", "请选择四项必要来源。表情证据是否适用由后端验证；内容人工复核为选填。"],

    "queued": ["排隊中", "排队中"],
    "running": ["執行中", "运行中"],
    "missing": ["尚未安裝", "尚未安装"],
    "Missing": ["尚未安裝", "尚未安装"],
    "Blocked": ["已阻擋", "已阻止"],
    "Draft": ["草稿", "草稿"],
    "No TSE projects": ["沒有 TSE 專案", "没有 TSE 项目"],
    "ADVANCED DATA": ["進階資料", "高级数据"],
    "optional": ["選用", "可选"],
    "Show more rows": ["顯示更多資料列", "显示更多数据行"],
    "Pause job": ["暫停工作", "暂停任务"],
    "Resume job": ["恢復工作", "恢复任务"],
    "Cancel job": ["取消工作", "取消任务"],
    "Retry job": ["重試工作", "重试任务"],
    "Loading\u2026": ["讀取中…", "加载中…"],
    "Data could not be loaded.": ["資料讀取失敗。", "数据读取失败。"],
    "Refresh failed; previously loaded rows are shown.": ["更新失敗；目前保留先前載入的資料。", "更新失败；当前保留先前加载的数据。"],
    "Unknown": ["未知", "未知"],
    "Cancelled": ["已取消", "已取消"],
    "Paused": ["已暫停", "已暂停"],
    "Succeeded": ["執行成功", "执行成功"],
    "Running": ["執行中", "运行中"],
    "Failed": ["失敗", "失败"],
    "TRAINING": ["訓練", "训练"],
    "Queued": ["排隊中", "排队中"],
    "Entries shown": ["目前顯示筆數", "当前显示条数"],
    "Browse projects": ["瀏覽專案", "浏览项目"],
    "Selected project": ["已選專案", "已选项目"],
    "Dismiss notification": ["關閉通知", "关闭通知"],
    "Voice direction and metadata": ["聲音方向與進階資料", "声音方向与进阶资料"],
    "Advanced workstation settings": ["進階工作站設定", "进阶工作站设置"],
    "Benchmark and compute settings": ["效能測量與運算設定", "性能测量与运算设置"],
    "Unsaved changes": ["尚未儲存的修改", "尚未保存的修改"],

    "Overview": ["總覽", "总览"],
    "Create": ["建立", "创建"],
    "Train": ["訓練", "训练"],
    "Evaluate": ["評估", "评估"],
    "Deploy": ["部署", "部署"],
    "System": ["系統", "系统"],
    "Synthesis": ["語音合成", "语音合成"],
    "Speech Synthesis": ["語音合成工作站", "语音合成工作站"],
    "Segmented expression workspace": ["分段情感語音工作區", "分段情感语音工作区"],
    "Expressions": ["情感", "情感"],
    "Expression Bank": ["情感資料庫", "情感资料库"],
    "Datasets": ["資料集", "数据集"],
    "Dataset Factory": ["資料集工廠", "数据集工厂"],
    "Expression reference candidates": ["情感參考候選", "情感参考候选"],
    "STYLE REFERENCES": ["風格參考", "风格参考"],
    "Human-reviewed clips are ranked by identity, signal quality and usable duration.": ["已人工審核的片段會按音色一致性、訊號品質與可用時長排序。", "已人工审核的片段会按音色一致性、信号质量与可用时长排序。"],
    "Use as reference": ["用作參考", "用作参考"],
    "Preview": ["試聽", "试听"],
    "Recommended": ["推薦", "推荐"],
    "Install AI Components": ["安裝 AI 元件", "安装 AI 组件"],
    "AI Components ready": ["AI 元件已就緒", "AI 组件已就绪"],
    "Local .zip path": ["本機 .zip 路徑", "本地 .zip 路径"],
    "Browse": ["瀏覽", "浏览"],
    "Browse local path": ["瀏覽本機路徑", "浏览本地路径"],
    "Choose a local path": ["選擇本機路徑", "选择本地路径"],
    "Choose file": ["選擇檔案", "选择文件"],
    "Choose folder": ["選擇資料夾", "选择文件夹"],
    "Drop here or enter a path": ["拖放至此或輸入路徑", "拖放到此处或输入路径"],
    "Complete this field before continuing": ["請先完成此欄位", "请先完成此字段"],
    "Enter a value in the expected format": ["請使用此欄位要求的格式", "请使用此字段要求的格式"],
    "Use the format shown for this field": ["請按照欄位顯示的格式輸入", "请按照字段显示的格式输入"],
    "Choose a value within the permitted range": ["請選擇允許範圍內的數值", "请选择允许范围内的数值"],
    "Choose a valid step value": ["請選擇符合間隔要求的數值", "请选择符合步长要求的数值"],
    "Adjust the length of this value": ["請調整此內容的長度", "请调整此内容的长度"],
    "Check this field before continuing": ["請先檢查此欄位", "请先检查此字段"],
    "The browser withheld the local path; use Browse to select it securely": ["瀏覽器沒有提供本機路徑，請使用「瀏覽」安全選取。", "浏览器未提供本地路径，请使用“浏览”安全选择。"],
    "Downloads occur only from this setup action. Workers validate revision, inventory and SHA-256, then run offline.": ["只有此設定操作會進行下載。Worker 會驗證版本、檔案清單與 SHA-256，之後以離線模式執行。", "只有此设置操作会进行下载。Worker 会验证版本、文件清单与 SHA-256，之后以离线模式运行。"],
    "Target Speaker Extraction": ["目標說話者擷取", "目标说话者提取"],
    "Training": ["訓練", "训练"],
    "Model Training": ["模型訓練", "模型训练"],
    "Evaluation": ["評估", "评估"],
    "Evaluation Lab": ["評估實驗室", "评估实验室"],
    "Models": ["模型", "模型"],
    "Model Registry": ["模型註冊庫", "模型注册库"],
    "Engines": ["引擎", "引擎"],
    "TensorRT Engines": ["TensorRT 引擎", "TensorRT 引擎"],
    "Jobs": ["工作", "任务"],
    "GPU Jobs": ["GPU 工作", "GPU 任务"],
    "Settings": ["設定", "设置"],
    "More": ["更多", "更多"],
    "More modules": ["更多模組", "更多模块"],
    "More AnifLive-TTS Studio modules": ["更多 AnifLive-TTS Studio 模組", "更多 AnifLive-TTS Studio 模块"],
    "Open module navigation": ["開啟模組導覽", "打开模块导航"],
    "Close module navigation": ["關閉模組導覽", "关闭模块导航"],
    "Modules": ["模組", "模块"],
    "Change interface language": ["切換介面語言", "切换界面语言"],
    "Interface language": ["介面語言", "界面语言"],
    "LOCAL VOICE WORKSTATION": ["本機語音工作站", "本地语音工作站"],
    "RUNNING JOB": ["執行中工作", "运行中任务"],
    "PROJECTS": ["專案", "项目"],
    "RECENT WORK": ["最近工作", "最近任务"],
    "Continue where you left off": ["從上次進度繼續", "从上次进度继续"],
    "Start building a voice model": ["開始建立語音模型", "开始创建语音模型"],
    "New project": ["新增專案", "新建项目"],
    "ACTIVE PIPELINE": ["執行中流程", "运行中流程"],
    "Job flow": ["工作流程", "任务流程"],
    "Open jobs": ["開啟工作列表", "打开任务列表"],
    "Workstation status": ["工作站狀態", "工作站状态"],
    "Waiting for runtime": ["等待推理服務", "等待推理服务"],
    "Runtime unavailable": ["推理服務不可用", "推理服务不可用"],
    "Local metadata": ["本機中繼資料", "本地元数据"],
    "No projects yet": ["尚未建立專案", "尚未建立项目"],
    "No workstation activity": ["目前沒有工作站活動", "目前没有工作站活动"],
    "No jobs": ["目前沒有工作", "目前没有任务"],
    "No registered production artifacts": ["尚未註冊正式環境成品", "尚未注册生产环境产物"],
    "No compatible project": ["沒有相容的專案", "没有兼容的项目"],
    "No automated evaluation": ["沒有自動化評估", "没有自动化评估"],
    "Explicit long-form evidence required": ["需要明確的長句盲聽證據", "需要明确的长句盲听证据"],
    "Explicit expression evidence required": ["需要明確的情感盲聽證據", "需要明确的情感盲听证据"],
    "Explicit security evidence required": ["需要明確的安全驗證證據", "需要明确的安全验证证据"],
    "Reading the pinned component lock…": ["正在讀取固定版本的元件清單…", "正在读取固定版本的组件清单…"],
    "Not connected": ["未連線", "未连接"],
    "Unavailable": ["不可用", "不可用"],
    "None": ["無", "无"],
    "Offline": ["離線", "离线"],
    "Ready": ["就緒", "就绪"],
    "Connecting": ["連線中", "连接中"],
    "Idle": ["閒置", "空闲"],
    "CREATE": ["建立", "创建"],
    "DATA": ["資料", "数据"],
    "TRAIN": ["訓練", "训练"],
    "DEPLOY": ["部署", "部署"],
    "MODEL PACKAGE": ["模型套件", "模型包"],
    "Qualified runtime profiles": ["已驗證的推理設定檔", "已验证的推理配置"],
    "Package + local": ["模型套件與本機", "模型包与本地"],
    "New expression": ["新增情感", "新建情感"],
    "Refresh": ["重新整理", "刷新"],
    "Profile": ["設定檔", "配置"],
    "Languages": ["語言", "语言"],
    "Intensity": ["強度", "强度"],
    "Policy": ["策略", "策略"],
    "LOCAL WORKSTATION": ["本機工作站", "本地工作站"],
    "Expression drafts": ["情感草稿", "情感草稿"],
    "Name": ["名稱", "名称"],
    "Emotion": ["情感", "情感"],
    "Language": ["語言", "语言"],
    "Status": ["狀態", "状态"],
    "SELECTION": ["選取項目", "所选项目"],
    "No expression selected": ["未選取情感", "未选择情感"],
    "Select a package profile or local draft.": ["請選取模型套件設定檔或本機草稿。", "请选择模型包配置或本地草稿。"],
    "Profile ID": ["設定檔 ID", "配置 ID"],
    "Model ID": ["模型 ID", "模型 ID"],
    "Local reference path": ["本機參考音訊路徑", "本地参考音频路径"],
    "Reference transcript · optional": ["參考音訊文字稿 · 選填", "参考音频文本 · 选填"],
    "Qualification": ["驗證狀態", "验证状态"],
    "Draft · evidence required": ["草稿 · 需要驗證證據", "草稿 · 需要验证证据"],
    "Descriptions, one per line": ["描述，每行一項", "描述，每行一项"],
    "Voice direction": ["語音方向", "语音方向"],
    "Valence": ["愉悅度", "愉悦度"],
    "Arousal": ["喚醒度", "唤醒度"],
    "Dominance": ["支配度", "支配度"],
    "Prosody metadata (JSON object)": ["韻律中繼資料（JSON 物件）", "韵律元数据（JSON 对象）"],
    "MEASURED SIGNAL": ["量測訊號", "测量信号"],
    "Reference analysis": ["參考音訊分析", "参考音频分析"],
    "Analyze": ["分析", "分析"],
    "Measured reference waveform": ["量測參考音訊波形", "测量参考音频波形"],
    "Duration": ["時長", "时长"],
    "Onset": ["起音", "起音"],
    "Pitch": ["音高", "音高"],
    "Speaking rate": ["語速", "语速"],
    "Reference SHA": ["參考音訊 SHA", "参考音频 SHA"],
    "No measured analysis": ["尚無量測分析", "暂无测量分析"],
    "Evaluation evidence": ["評估證據", "评估证据"],
    "No passed evidence": ["沒有已通過的證據", "没有已通过的证据"],
    "Promote expression": ["升級情感設定", "升级情感配置"],
    "Qualified status can only be granted by a passed evaluation artifact.": ["只有通過評估的成品才能取得已驗證狀態。", "只有通过评估的产物才能取得已验证状态。"],
    "Delete": ["刪除", "删除"],
    "Cancel": ["取消", "取消"],
    "Save expression": ["儲存情感", "保存情感"],
    "New dataset": ["新增資料集", "新建数据集"],
    "No dataset projects": ["尚未建立資料集專案", "尚未创建数据集项目"],
    "Voice acquisition workflow": ["語音資料取得流程", "语音数据获取流程"],
    "Speaker": ["說話者", "说话者"],
    "Clean": ["清理", "清理"],
    "Text": ["文字", "文本"],
    "Review": ["審核", "审核"],
    "Style": ["風格", "风格"],
    "Audio / Video Collection": ["音訊／影片集合", "音频／视频集合"],
    "Extract Target Speaker": ["擷取目標說話者", "提取目标说话者"],
    "Existing GPT-SoVITS Dataset": ["既有 GPT-SoVITS 資料集", "现有 GPT-SoVITS 数据集"],
    "Standard dataset": ["音訊／影片集合", "音频／视频集合"],
    "Start preparation": ["開始準備", "开始准备"],
    "Import acquired clips": ["匯入取得的片段", "导入获取的片段"],
    "Salvaged": ["已救回", "已挽救"],
    "Rejected": ["已拒絕", "已拒绝"],
    "Advanced speaker extraction": ["進階說話者擷取", "高级说话者提取"],
    "Reference prototypes": ["參考音色原型", "参考音色原型"],
    "Primary diarization": ["主要說話者分段", "主要说话人分段"],
    "Sensitive scan": ["敏感度補充掃描", "敏感度补充扫描"],
    "Speaker evidence": ["說話者證據", "说话人证据"],
    "Clean SNR gate": ["乾淨音訊 SNR 門檻", "干净音频 SNR 门槛"],
    "Network": ["網絡", "网络"],
    "Lower-quality clips remain in review": ["較低品質片段會保留待人工審核", "较低质量片段会保留待人工审核"],
    "Linux Docker neural worker": ["Linux Docker 神經網絡工作器", "Linux Docker 神经网络工作器"],
    "Verified routing evidence appears here after the Linux Docker worker completes.": ["Linux Docker 工作器完成後，已驗證的路由證據會顯示在此。", "Linux Docker 工作器完成后，已验证的路由证据会显示在此。"],
    "Not reported": ["未記錄", "未记录"],
    "Review queue": ["審核佇列", "审核队列"],
    "Expression candidates": ["情感參考候選", "情感参考候选"],
    "Freeze dataset": ["凍結資料集", "冻结数据集"],
    "Continue to Training": ["繼續前往訓練", "继续前往训练"],
    "Qualification report": ["驗收報告", "验收报告"],
    "Qualification report exported · all gates passed": ["驗收報告已匯出 · 所有門檻通過", "验收报告已导出 · 所有门槛通过"],
    "PROJECT BROWSER": ["專案瀏覽器", "项目浏览器"],
    "Dataset projects": ["資料集專案", "数据集项目"],
    "ACTIVE DATASET": ["使用中資料集", "当前数据集"],
    "Select a project": ["請選取專案", "请选择项目"],
    "No source configured": ["尚未設定來源", "尚未设置来源"],
    "Sources": ["來源", "来源"],
    "Segments": ["片段", "片段"],
    "Accepted": ["已接受", "已接受"],
    "Assigned": ["已分配", "已分配"],
    "Process in Docker": ["在 Docker 中處理", "在 Docker 中处理"],
    "Ingest source": ["匯入來源", "导入来源"],
    "Import raw only": ["僅匯入原始檔", "仅导入原始文件"],
    "Import .list": ["匯入 .list", "导入 .list"],
    "Refresh dataset": ["重新整理資料集", "刷新数据集"],
    "Not queued": ["未排入佇列", "未加入队列"],
    "No Linux Docker processing result.": ["尚無 Linux Docker 處理結果。", "暂无 Linux Docker 处理结果。"],
    "Import verified segments": ["匯入已驗證片段", "导入已验证片段"],
    "Open job details": ["開啟工作詳細資料", "打开任务详情"],
    "DATASET LIFECYCLE": ["資料集生命週期", "数据集生命周期"],
    "Audio inventory": ["音訊清單", "音频清单"],
    "Assign split": ["分配資料分割", "分配数据划分"],
    "Manifest": ["清單檔", "清单文件"],
    "Dataset processing stages": ["資料集處理階段", "数据集处理阶段"],
    "Select a dataset project to begin.": ["請選取資料集專案以開始。", "请选择数据集项目以开始。"],
    "No dataset selected": ["未選取資料集", "未选择数据集"],
    "ITEM INSPECTOR": ["項目檢查器", "项目检查器"],
    "Dataset item": ["資料集項目", "数据集项目"],
    "Close dataset item inspector": ["關閉資料集項目檢查器", "关闭数据集项目检查器"],
    "Verified PCM waveform": ["已驗證 PCM 波形", "已验证 PCM 波形"],
    "Transcript": ["文字稿", "文本"],
    "Training annotation": ["訓練標註", "训练标注"],
    "Apply to derived resamples and segments": ["套用至衍生的重採樣與片段", "应用至衍生的重采样与片段"],
    "Optional label": ["選填標籤", "可选标签"],
    "Save annotation": ["儲存標註", "保存标注"],
    "Require expression classification before freeze": ["凍結前必須完成人工情感分類", "冻结前必须完成人工情感分类"],
    "Disable only when expression labels are outside this dataset's review scope.": ["僅在此資料集不包含情感審核時停用。", "仅在此数据集不包含情感审核时停用。"],
    "Human review contract": ["人工審核契約", "人工审核契约"],
    "Clip integrity": ["片段完整性", "片段完整性"],
    "Text + language": ["文字稿與語言", "文本与语言"],
    "Speaker identity": ["說話者身分", "说话人身份"],
    "Accept Audio": ["接受音訊", "接受音频"],
    "Reject Audio": ["拒絕音訊", "拒绝音频"],
    "Save": ["儲存", "保存"],
    "Save & Verify Transcript": ["儲存並核實文字稿", "保存并核实文本"],
    "Confirm Speaker": ["確認說話者", "确认说话人"],
    "Verify Expression": ["核實情感", "核实情感"],
    "Optional / Not required": ["選填／不要求", "可选／不要求"],
    "Required for this dataset": ["此資料集必須完成", "此数据集必须完成"],
    "Verified": ["已核實", "已核实"],
    "Pending": ["待處理", "待处理"],
    "Optional": ["選填", "可选"],
    "How would you like to create this dataset?": ["你想如何建立這個資料集？", "你想如何创建这个数据集？"],
    "Dataset acquisition mode": ["資料集取得方式", "数据集获取方式"],
    "Process existing mostly single-speaker recordings.": ["處理以單一說話者為主的既有錄音。", "处理以单一说话者为主的现有录音。"],
    "Find one speaker using a short reference recording.": ["使用一段短參考錄音找出指定說話者。", "使用一段短参考录音找出指定说话者。"],
    "Import one reviewed .list dataset.": ["匯入一份已審核的 .list 資料集。", "导入一份已审核的 .list 数据集。"],
    "Source media": ["來源媒體", "源媒体"],
    "Source audio / video": ["來源音訊／影片", "源音频／视频"],
    "GPT-SoVITS .list path": ["GPT-SoVITS .list 路徑", "GPT-SoVITS .list 路径"],
    "One local audio, video, folder or .list path per line": ["每行一個本機音訊、影片、資料夾或 .list 路徑", "每行一个本地音频、视频、文件夹或 .list 路径"],
    "One local audio, video or folder path per line": ["每行一個本機音訊、影片或資料夾路徑", "每行一个本地音频、视频或文件夹路径"],
    "One local long recording or folder per line": ["每行一個本機長錄音或資料夾路徑", "每行一个本地长录音或文件夹路径"],
    "One reviewed local .list path": ["一個已審核的本機 .list 路徑", "一个已审核的本地 .list 路径"],
    "Speech recognition": ["語音辨識", "语音识别"],
    "Dataset language": ["資料集語言", "数据集语言"],
    "Auto detect": ["自動偵測", "自动检测"],
    "Choose a language when the collection is known to prevent per-clip detection drift.": ["已知資料語言時請直接指定，避免逐片段偵測漂移。", "已知数据语言时请直接指定，避免逐片段检测漂移。"],
    "SenseVoice Small · recommended": ["SenseVoice Small · 建議", "SenseVoice Small · 推荐"],
    "Transcripts and detected styles remain suggestions until human review.": ["文字稿與偵測到的風格在人工審核前只屬建議。", "文本与检测到的风格在人工审核前仅作建议。"],
    "Target reference": ["目標參考音訊", "目标参考音频"],
    "Short clean reference audio": ["短而乾淨的參考音訊", "短而干净的参考音频"],
    "Target speaker detection": ["目標說話者偵測", "目标说话者检测"],
    "Strict": ["嚴格", "严格"],
    "High recall": ["高召回率", "高召回率"],
    "Deterministic audio quality": ["確定性音訊品質", "确定性音频质量"],
    "Recalculate": ["重新計算", "重新计算"],
    "Selected item stage state": ["所選項目的階段狀態", "所选项目的阶段状态"],
    "New TSE project": ["新增 TSE 專案", "新建 TSE 项目"],
    "Extraction sources": ["擷取來源", "提取来源"],
    "Reference": ["參考音訊", "参考音频"],
    "Source": ["來源", "来源"],
    "Select a TSE project": ["請選取 TSE 專案", "请选择 TSE 项目"],
    "Not configured": ["尚未設定", "尚未设置"],
    "Source audio": ["來源音訊", "源音频"],
    "Reference audio": ["參考音訊", "参考音频"],
    "V2ProPlus model package": ["V2ProPlus 模型套件", "V2ProPlus 模型包"],
    "MossFormer2 separation model": ["MossFormer2 分離模型", "MossFormer2 分离模型"],
    "Absolute local audio path": ["本機音訊的絕對路徑", "本地音频的绝对路径"],
    "Absolute local reference path": ["本機參考音訊的絕對路徑", "本地参考音频的绝对路径"],
    "Absolute local package directory": ["本機模型套件的絕對路徑", "本地模型包的绝对路径"],
    "External MossFormer2_SS_16K checkpoint directory": ["外部 MossFormer2_SS_16K checkpoint 目錄", "外部 MossFormer2_SS_16K checkpoint 目录"],
    "Verification": ["驗證", "验证"],
    "Target threshold": ["目標門檻", "目标阈值"],
    "Review margin": ["人工審查邊界", "人工审核边界"],
    "Overlap → separation + TensorRT identity gate": ["重疊 → 分離 + TensorRT 身分門檻", "重叠 → 分离 + TensorRT 身份门槛"],
    "Segmentation controls": ["分段控制", "分段控制"],
    "VAD frame · ms": ["VAD 影格 · ms", "VAD 帧 · ms"],
    "Minimum speech · ms": ["最短語音 · ms", "最短语音 · ms"],
    "Maximum gap · ms": ["最大間隔 · ms", "最大间隔 · ms"],
    "Context · ms": ["上下文 · ms", "上下文 · ms"],
    "Extraction gap · ms": ["擷取間隔 · ms", "提取间隔 · ms"],
    "Separation ambiguity": ["分離歧義", "分离歧义"],
    "Queue extraction": ["排入擷取工作", "加入提取任务"],
    "Re-run reviewed": ["重新執行已審查項目", "重新运行已审核项目"],
    "JOB EVENTS": ["工作事件", "任务事件"],
    "No extraction selected": ["未選取擷取工作", "未选择提取任务"],
    "No worker events": ["尚無 worker 事件", "暂无 worker 事件"],
    "Cancel run": ["取消執行", "取消运行"],
    "VERIFIED OUTPUT": ["已驗證輸出", "已验证输出"],
    "Target-speaker result": ["目標說話者結果", "目标说话者结果"],
    "TensorRT report pending": ["等待 TensorRT 報告", "等待 TensorRT 报告"],
    "Target audio player": ["目標音訊播放器", "目标音频播放器"],
    "Play target audio": ["播放目標音訊", "播放目标音频"],
    "Pause target audio": ["暫停目標音訊", "暂停目标音频"],
    "Target audio position": ["目標音訊播放位置", "目标音频播放位置"],
    "Mute target audio": ["將目標音訊靜音", "将目标音频静音"],
    "Unmute target audio": ["取消目標音訊靜音", "取消目标音频静音"],
    "Play": ["播放", "播放"],
    "Pause": ["暫停", "暂停"],
    "Mute": ["靜音", "静音"],
    "Unmute": ["取消靜音", "取消静音"],
    "Target audio": ["目標音訊", "目标音频"],
    "JSON report": ["JSON 報告", "JSON 报告"],
    "Time": ["時間", "时间"],
    "Similarity": ["相似度", "相似度"],
    "Overlap": ["重疊", "重叠"],
    "Separate": ["分離", "分离"],
    "Decision": ["判定", "判定"],
    "No report loaded": ["尚未載入報告", "尚未加载报告"],
    "New experiment": ["新增實驗", "新建实验"],
    "EXPERIMENTS": ["實驗", "实验"],
    "Training runs": ["訓練工作", "训练任务"],
    "Preset": ["預設組態", "预设配置"],
    "Progress": ["進度", "进度"],
    "Updated": ["更新時間", "更新时间"],
    "SELECTED EXPERIMENT": ["所選實驗", "所选实验"],
    "No experiment selected": ["未選取實驗", "未选择实验"],
    "Dataset": ["資料集", "数据集"],
    "Resume": ["續訓", "续训"],
    "Queue training": ["排入訓練工作", "加入训练任务"],
    "Build Production Model": ["建立 Production 模型", "构建 Production 模型"],
    "Continue production workflow": ["繼續成品工作流程", "继续成品工作流程"],
    "Production training lifecycle": ["成品訓練生命週期", "成品训练生命周期"],
    "Data": ["資料", "数据"],
    "Select": ["挑選", "选择"],
    "Reference": ["參考", "参考"],
    "Convert": ["轉換", "转换"],
    "Package": ["封裝", "封装"],
    "Waiting": ["等待中", "等待中"],
    "Complete": ["已完成", "已完成"],
    "Reviewed": ["已審核", "已审核"],
    "Dataset missing": ["缺少資料集", "缺少数据集"],
    "Human locked": ["人工鎖定", "人工锁定"],
    "Listening required": ["需要盲聽", "需要盲听"],
    "Holdout failed": ["留出測試失敗", "留出测试失败"],
    "Evidence missing": ["缺少證據", "缺少证据"],
    "Review reference A/B": ["審核參考音訊 A/B", "审核参考音频 A/B"],
    "Blind reference listening is required before the test holdout can open.": ["開啟留出測試前必須先完成參考音訊盲聽。", "开启留出测试前必须先完成参考音频盲听。"],
    "All candidates rejected": ["全部候選已拒絕", "全部候选已拒绝"],
    "BLIND REFERENCE REVIEW": ["參考音訊盲聽", "参考音频盲听"],
    "Choose the closest voice reference": ["選出最接近原聲的參考音訊", "选出最接近原声的参考音频"],
    "Preparing candidates": ["正在準備候選", "正在准备候选"],
    "Compare the same numbered sentence across A–E. Candidate identities and the automatic recommendation stay hidden until you decide.": ["依照相同編號逐一比較 A–E 的句子。作出決定前，候選身分與自動推薦都會保持隱藏。", "按照相同编号逐一比较 A–E 的句子。作出决定前，候选身份与自动推荐都会保持隐藏。"],
    "Loading verified audio…": ["正在載入已驗證音訊…", "正在加载已验证音频…"],
    "Test holdout remains sealed.": ["留出測試仍然封存。", "留出测试仍然封存。"],
    "Select one reference to lock it, use the automatic recommendation when there is no preference, or reject the complete set if none preserves the voice.": ["選擇一個參考音訊並鎖定；若無明顯偏好可採用自動推薦；若全部都無法保留原聲，請拒絕整組候選。", "选择一个参考音频并锁定；若无明显偏好可采用自动推荐；若全部都无法保留原声，请拒绝整组候选。"],
    "No preference · use automatic recommendation": ["無偏好 · 採用自動推薦", "无偏好 · 采用自动推荐"],
    "All references are poor": ["所有參考音訊都不合格", "所有参考音频都不合格"],
    "Confirm rejection of all A–E": ["確認拒絕全部 A–E", "确认拒绝全部 A–E"],
    "Candidate": ["候選", "候选"],
    "BLIND CANDIDATE": ["盲測候選", "盲测候选"],
    "Sentence": ["句子", "句子"],
    "Play blind reference": ["播放盲測音訊", "播放盲测音频"],
    "Pause blind reference": ["暫停盲測音訊", "暂停盲测音频"],
    "No verified blind candidates are available": ["沒有可用的已驗證盲測候選", "没有可用的已验证盲测候选"],
    "Verified evidence unavailable": ["已驗證證據不可用", "已验证证据不可用"],
    "Blind reference artifacts are incomplete": ["參考音訊盲測成品不完整", "参考音频盲测产物不完整"],
    "Loading verified audio": ["正在載入已驗證音訊", "正在加载已验证音频"],
    "blind candidates": ["個盲測候選", "个盲测候选"],
    "Reference locked; holdout evaluation queued": ["參考音訊已鎖定；留出測試已排入佇列", "参考音频已锁定；留出测试已加入队列"],
    "All references rejected; holdout remains sealed": ["全部參考音訊已拒絕；留出測試維持封存", "全部参考音频已拒绝；留出测试保持封存"],
    "Blind reference listening is ready": ["參考音訊盲聽已準備完成", "参考音频盲听已准备完成"],
    "Conversion parity failed. Packaging is blocked.": ["轉換一致性失敗，已阻止封裝。", "转换一致性失败，已阻止封装。"],
    "Test holdout failed. TensorRT conversion is blocked.": ["留出測試失敗，已阻止 TensorRT 轉換。", "留出测试失败，已阻止 TensorRT 转换。"],
    "Production validation is progressing through the locked job graph.": ["成品驗證正沿鎖定的工作依賴圖進行。", "成品验证正沿锁定的任务依赖图进行。"],
    "Training is complete; validation checkpoint selection is ready to start.": ["訓練已完成，可以開始驗證 checkpoint 挑選。", "训练已完成，可以开始验证 checkpoint 选择。"],
    "The production workflow is waiting for its next verified dependency.": ["成品工作流程正在等待下一項已驗證依賴。", "成品工作流程正在等待下一项已验证依赖。"],
    "TensorRT package is not built.": ["尚未建立 TensorRT 模型套件。", "尚未构建 TensorRT 模型包。"],
    "Select a configured experiment.": ["請選取已設定的實驗。", "请选择已配置的实验。"],
    "LINUX DOCKER WORKER": ["LINUX DOCKER WORKER", "LINUX DOCKER WORKER"],
    "No training run selected": ["未選取訓練工作", "未选择训练任务"],
    "Waiting for a real worker event": ["等待真實 worker 事件", "等待真实 worker 事件"],
    "Epoch": ["訓練週期", "训练轮次"],
    "Loss": ["損失值", "损失值"],
    "Checkpoint": ["檢查點", "检查点"],
    "Temperature": ["溫度", "温度"],
    "Worker events": ["Worker 事件", "Worker 事件"],
    "Verified outputs": ["已驗證輸出", "已验证输出"],
    "No ready checkpoint artifacts": ["沒有可用的 checkpoint 成品", "没有可用的 checkpoint 产物"],
    "New evaluation": ["新增評估", "新建评估"],
    "Verified audio comparison decks": ["已驗證音訊比較組", "已验证音频比较组"],
    "Candidate": ["候選版本", "候选版本"],
    "Baseline": ["基準版本", "基准版本"],
    "No ready audio artifact": ["沒有可用的音訊成品", "没有可用的音频产物"],
    "Choose a completed baseline run": ["請選擇已完成的基準工作", "请选择已完成的基准任务"],
    "QUALIFICATION PROJECTS": ["驗收專案", "验收项目"],
    "Evaluation workflows": ["評估工作流程", "评估工作流程"],
    "Quality": ["品質", "质量"],
    "Performance": ["效能", "性能"],
    "SELECTED WORKFLOW": ["所選流程", "所选流程"],
    "No evaluation selected": ["未選取評估", "未选择评估"],
    "Audio language": ["音訊語言", "音频语言"],
    "Baseline run": ["基準工作", "基准任务"],
    "No completed baseline run": ["沒有已完成的基準工作", "没有已完成的基准任务"],
    "Upstream package dependency": ["上游模型套件依賴", "上游模型包依赖"],
    "No package dependency": ["沒有模型套件依賴", "没有模型包依赖"],
    "Queue evaluation": ["排入評估工作", "加入评估任务"],
    "Queue engine → package → evaluation": ["排入引擎 → 套件 → 評估", "加入引擎 → 模型包 → 评估"],
    "Dependencies control scheduling. Evaluation consumes a verified model package output.": ["依賴關係控制排程；評估只使用已驗證的模型套件輸出。", "依赖关系控制调度；评估只使用已验证的模型包输出。"],
    "MEASURED RESULT": ["量測結果", "测量结果"],
    "No evaluation run selected": ["未選取評估工作", "未选择评估任务"],
    "Keep-alive audible TTFA P50": ["Keep-alive 可聽 TTFA P50", "Keep-alive 可听 TTFA P50"],
    "Keep-alive audible TTFA P95": ["Keep-alive 可聽 TTFA P95", "Keep-alive 可听 TTFA P95"],
    "Complete-WAV RTF P50": ["完整 WAV RTF P50", "完整 WAV RTF P50"],
    "Five-language content and parity": ["五語內容與一致性", "五语内容与一致性"],
    "Speaker": ["說話者", "说话者"],
    "Gate": ["門檻", "门槛"],
    "No measured evaluation report": ["尚無量測評估報告", "暂无测量评估报告"],
    "Measured gates": ["量測門檻", "测量门槛"],
    "Five languages": ["五語", "五语"],
    "Speaker identity": ["說話者身分", "说话者身份"],
    "Streaming parity": ["串流一致性", "流式一致性"],
    "Long-form continuity": ["長文連續性", "长文连续性"],
    "Expression quality": ["情感品質", "情感质量"],
    "TensorRT runtime": ["TensorRT 推理", "TensorRT 推理"],
    "TTFA and RTF": ["TTFA 與 RTF", "TTFA 与 RTF"],
    "Security": ["安全性", "安全性"],
    "Verified artifacts": ["已驗證成品", "已验证产物"],
    "Qualification evidence": ["驗收證據", "验收证据"],
    "IMPORTED EVIDENCE": ["已匯入證據", "已导入证据"],
    "No evidence selected": ["未選取證據", "未选择证据"],
    "FAIL-CLOSED COMPOSITION": ["失敗即關閉的證據組合", "失败即关闭的证据组合"],
    "Compose production qualification": ["組合正式環境驗收", "组合生产环境验收"],
    "Select all verified sources": ["請選取所有已驗證來源", "请选择所有已验证来源"],
    "Qualification subject": ["驗收對象", "验收对象"],
    "Automated evaluation": ["自動評估", "自动评估"],
    "Long-form blind A/B": ["長文盲聽 A/B", "长文盲听 A/B"],
    "Expression blind A/B": ["情感盲聽 A/B", "情感盲听 A/B"],
    "Security verification": ["安全驗證", "安全验证"],
    "Explicit evidence required": ["需要明確證據", "需要明确证据"],
    "Compose verified gates": ["組合已驗證門檻", "组合已验证门槛"],
    "Manual gates remain unavailable until separate blinded evidence artifacts are selected.": ["在選取獨立的盲測證據成品前，人工門檻維持不可用。", "在选择独立的盲测证据产物前，人工门槛保持不可用。"],
    "Artifact lineage": ["成品沿襲關係", "产物谱系"],
    "ARTIFACT REGISTRY": ["成品註冊庫", "产物注册库"],
    "Production lineage": ["正式環境沿襲關係", "生产环境谱系"],
    "Artifact": ["成品", "产物"],
    "Type": ["類型", "类型"],
    "Promotion": ["升級狀態", "升级状态"],
    "ARTIFACT EVIDENCE": ["成品證據", "产物证据"],
    "Select an artifact": ["請選取成品", "请选择产物"],
    "No verified file selected": ["未選取已驗證檔案", "未选择已验证文件"],
    "Open verified artifact": ["開啟已驗證成品", "打开已验证产物"],
    "No lineage selected": ["未選取沿襲關係", "未选择谱系"],
    "Passed qualification": ["已通過驗收", "已通过验收"],
    "Promote artifact": ["升級成品", "升级产物"],
    "Promotion is fail-closed until all required gates pass.": ["所有必要門檻通過前，成品升級會維持關閉。", "所有必要门槛通过前，产物升级保持关闭。"],
    "REGISTERED ENGINES": ["已註冊引擎", "已注册引擎"],
    "Verified engine artifacts": ["已驗證引擎成品", "已验证引擎产物"],
    "Engine": ["引擎", "引擎"],
    "Lineage": ["沿襲關係", "谱系"],
    "Queue job": ["排入工作", "加入任务"],
    "DEPENDENCY SPINE": ["依賴關係", "依赖关系"],
    "Work queue": ["工作佇列", "任务队列"],
    "Job / attempt": ["工作 / 嘗試", "任务 / 尝试"],
    "Type / priority": ["類型 / 優先級", "类型 / 优先级"],
    "Dependencies": ["依賴項目", "依赖项"],
    "Actions": ["操作", "操作"],
    "RUNNING": ["執行中", "运行中"],
    "QUEUED": ["佇列中", "队列中"],
    "Select a job": ["請選取工作", "请选择任务"],
    "No event stream selected": ["未選取事件串流", "未选择事件流"],
    "Local workstation": ["本機工作站", "本地工作站"],
    "AI COMPONENTS": ["AI 元件", "AI 组件"],
    "Managed offline workers": ["受管理的離線 Worker", "托管的离线 Worker"],
    "Offline component bundle": ["離線元件套件", "离线组件包"],
    "Import bundle": ["匯入套件", "导入包"],
    "Export ready bundle": ["匯出已就緒套件", "导出已就绪包"],
    "RUNTIME": ["推理環境", "推理环境"],
    "System contract": ["系統契約", "系统契约"],
    "Product": ["產品", "产品"],
    "Workstation build": ["工作站版本", "工作站版本"],
    "Backend": ["後端", "后端"],
    "DEFAULTS": ["預設值", "默认值"],
    "Workstation behavior": ["工作站行為", "工作站行为"],
    "Saved locally": ["已儲存於本機", "已保存于本地"],
    "Speech language": ["語音語言", "语音语言"],
    "Session continuity": ["語音工作階段連續性", "语音会话连续性"],
    "Training preset": ["訓練預設組態", "训练预设配置"],
    "Evaluation language": ["評估語言", "评估语言"],
    "TSE target threshold": ["TSE 目標門檻", "TSE 目标阈值"],
    "Refresh interval · seconds": ["重新整理間隔 · 秒", "刷新间隔 · 秒"],
    "Play the Overview motion field": ["播放總覽動態背景", "播放总览动态背景"],
    "Save settings": ["儲存設定", "保存设置"],
    "INSTALLATION": ["安裝", "安装"],
    "AnifLive-TTS Studio app": ["AnifLive-TTS Studio 應用程式", "AnifLive-TTS Studio 应用"],
    "Checking": ["檢查中", "检查中"],
    "Launch mode": ["啟動模式", "启动模式"],
    "Browser": ["瀏覽器", "浏览器"],
    "Offline shell": ["離線介面", "离线界面"],
    "Available": ["可用", "可用"],
    "Install AnifLive-TTS Studio": ["安裝 AnifLive-TTS Studio", "安装 AnifLive-TTS Studio"],
    "SPEECH SESSIONS": ["語音工作階段", "语音会话"],
    "Committed execution": ["已確認片段執行", "已确认片段执行"],
    "Input contract": ["輸入契約", "输入契约"],
    "Committed segments": ["已確認片段", "已确认片段"],
    "Context": ["上下文", "上下文"],
    "Turn orchestration": ["對話輪次編排", "对话轮次编排"],
    "INTERFACE HISTORY": ["介面紀錄", "界面记录"],
    "Return to a clean workspace view": ["回到乾淨的工作站畫面", "回到干净的工作站界面"],
    "Hide projects, jobs, artifacts, qualifications and local expression drafts created before this moment. Files and workstation data remain unchanged.": ["隱藏在此刻以前建立的專案、工作、成品、驗證與本機情感草稿。檔案與工作站資料不會變更。", "隐藏在此刻以前创建的项目、任务、产物、验证与本地情感草稿。文件与工作站数据不会更改。"],
    "All visible records are shown": ["目前顯示所有介面紀錄", "当前显示所有界面记录"],
    "Visible records cleared": ["介面紀錄已清除", "界面记录已清除"],
    "Workspace view is clean": ["工作站畫面已整理乾淨", "工作站界面已整理干净"],
    "Clear visible records": ["清除介面紀錄", "清除界面记录"],
    "Clear visible records?": ["要清除介面紀錄嗎？", "要清除界面记录吗？"],
    "This returns AnifLive-TTS Studio to a clean first-use view. It does not delete projects, jobs, model assets, audio, checkpoints or files.": ["這會讓 AnifLive-TTS Studio 回到乾淨的首次使用畫面，不會刪除專案、工作、模型資產、音訊、checkpoint 或檔案。", "这会让 AnifLive-TTS Studio 回到干净的首次使用界面，不会删除项目、任务、模型资产、音频、checkpoint 或文件。"],
    "Clear interface history": ["清除介面紀錄", "清除界面记录"],
    "Interface history cleared; files and assets were retained": ["介面紀錄已清除，檔案與資產均已保留", "界面记录已清除，文件与资产均已保留"],
    "LOCAL PROJECT": ["本機專案", "本地项目"],
    "Close": ["關閉", "关闭"],
    "Source path": ["來源路徑", "来源路径"],
    "Dataset path": ["資料集路徑", "数据集路径"],
    "Reference path": ["參考音訊路徑", "参考音频路径"],
    "Quick": ["快速", "快速"],
    "Balanced": ["平衡", "平衡"],
    "High Quality": ["高品質", "高质量"],
    "Advanced": ["進階", "高级"],
    "V2ProPlus training inputs": ["V2ProPlus 訓練輸入", "V2ProPlus 训练输入"],
    "Pretrained GPT checkpoint": ["預訓練 GPT checkpoint", "预训练 GPT checkpoint"],
    "Pretrained SoVITS generator G": ["預訓練 SoVITS 生成器 G", "预训练 SoVITS 生成器 G"],
    "Pretrained SoVITS discriminator D": ["預訓練 SoVITS 判別器 D", "预训练 SoVITS 判别器 D"],
    "Resume checkpoint · optional": ["續訓 checkpoint · 選填", "续训 checkpoint · 可选"],
    "Stage": ["階段", "阶段"],
    "GPT + SoVITS": ["GPT + SoVITS", "GPT + SoVITS"],
    "GPT only": ["僅 GPT", "仅 GPT"],
    "SoVITS only": ["僅 SoVITS", "仅 SoVITS"],
    "GPT epochs": ["GPT 訓練週期", "GPT 训练轮次"],
    "SoVITS epochs": ["SoVITS 訓練週期", "SoVITS 训练轮次"],
    "GPT batch": ["GPT 批次", "GPT 批次"],
    "SoVITS batch": ["SoVITS 批次", "SoVITS 批次"],
    "GPT learning rate": ["GPT 學習率", "GPT 学习率"],
    "SoVITS learning rate": ["SoVITS 學習率", "SoVITS 学习率"],
    "Save every epoch": ["每個週期儲存", "每轮保存"],
    "Gradient checkpointing": ["梯度 checkpointing", "梯度 checkpointing"],
    "Offline qualification inputs": ["離線驗收輸入", "离线验收输入"],
    "Shared runtime directory": ["共用推理環境目錄", "共享推理环境目录"],
    "Offline ASR model directory": ["離線 ASR 模型目錄", "离线 ASR 模型目录"],
    "Production baseline report · optional": ["正式環境基準報告 · 選填", "生产环境基准报告 · 可选"],
    "Benchmark sessions": ["Benchmark 工作階段", "Benchmark 会话"],
    "Warmups": ["預熱次數", "预热次数"],
    "Runs per workload": ["每種負載執行次數", "每种负载运行次数"],
    "Benchmark language": ["Benchmark 語言", "Benchmark 语言"],
    "ASR compute": ["ASR 計算精度", "ASR 计算精度"],
    "Request timeout · s": ["請求逾時 · 秒", "请求超时 · 秒"],
    "DEPENDENCY-AWARE QUEUE": ["依賴感知佇列", "依赖感知队列"],
    "Job type": ["工作類型", "任务类型"],
    "Project": ["專案", "项目"],
    "Priority": ["優先級", "优先级"],
    "No eligible dependencies": ["沒有可用依賴項目", "没有可用依赖项"],
    "Queue": ["排入佇列", "加入队列"],
    "Dataset inventory": ["資料集清單", "数据集清单"],
    "Dataset decode and segmentation": ["資料集解碼與分段", "数据集解码与分段"],
    "TSE prepare": ["TSE 準備", "TSE 准备"],
    "Training prepare": ["訓練準備", "训练准备"],
    "Evaluation prepare": ["評估準備", "评估准备"],
    "TensorRT engine prepare": ["TensorRT 引擎準備", "TensorRT 引擎准备"],
    "Model package": ["模型套件", "模型包"],
    "Data": ["資料", "数据"],
    "MODEL": ["模型", "模型"],
    "GPU": ["GPU", "GPU"],
    "JOBS": ["工作", "任务"],
    "DATASET": ["資料集", "数据集"],
    "SUCCEEDED": ["已成功", "已成功"],
    "FAILED": ["失敗", "失败"],
    "succeeded": ["已成功", "已成功"],
    "failed": ["失敗", "失败"],
    "No package expression profiles": ["沒有模型套件情感設定檔", "没有模型包情感配置"],
    "No local expression drafts": ["沒有本機情感草稿", "没有本地情感草稿"],
    "Open": ["開啟", "打开"],
    "Select": ["選取", "选择"],
    "Inspect": ["檢查", "检查"],
    "Inspecting": ["檢查中", "检查中"],
    "Annotated": ["已標註", "已标注"],
    "Quality mean": ["平均品質", "平均质量"],
    "No items": ["沒有項目", "没有项目"],
    "No PCM items": ["沒有 PCM 項目", "没有 PCM 项目"],
    "Voice activity": ["語音活動", "语音活动"],
    "Capability status unavailable": ["尚未取得能力狀態", "尚未取得能力状态"],
    "Not started": ["尚未開始", "尚未开始"],
    "Ingest": ["匯入", "导入"],
    "Resample": ["重新取樣", "重新采样"],
    "Energy VAD": ["能量 VAD", "能量 VAD"],
    "Segment": ["分段", "分段"],
    "Review": ["審查", "审核"],
    "Split": ["資料分割", "数据划分"],
    "Annotate": ["標註", "标注"],
    "No ingested items": ["沒有已匯入項目", "没有已导入项目"],
    "Decode": ["解碼", "解码"],
    "Denoise": ["降噪", "降噪"],
    "Dereverb": ["去混響", "去混响"],
    "The source is ready for deterministic Linux Docker processing or local ingest.": ["來源已可進行確定性的 Linux Docker 處理或本機匯入。", "来源已可进行确定性的 Linux Docker 处理或本地导入。"],
    "Deterministic first-audio-stream decode to mono PCM16 at 32 kHz.": ["將第一條音訊串流確定性解碼為 32 kHz 單聲道 PCM16。", "将第一条音频流确定性解码为 32 kHz 单声道 PCM16。"],
    "Optional conservative FFmpeg afftdn; disabled unless explicitly requested.": ["可選的保守 FFmpeg afftdn；只有明確啟用時才會執行。", "可选的保守 FFmpeg afftdn；只有明确启用时才会执行。"],
    "No redistribution-safe dereverb backend with pinned assets has passed the Linux Docker qualification contract.": ["目前沒有具固定資產且可安全再發佈的去混響後端通過 Linux Docker 驗收契約。", "目前没有具固定资产且可安全再分发的去混响后端通过 Linux Docker 验收契约。"],
    "Managed pinned assets transcribe checksum-verified segments inside the CUDA worker; every transcript still requires review.": ["受管理且固定版本的資產會在 CUDA worker 中轉錄通過 checksum 驗證的片段；每份文字稿仍需人工審核。", "受管理且固定版本的资产会在 CUDA worker 中转录通过 checksum 验证的片段；每份文本仍需人工审核。"],
    "Install component": ["安裝元件", "安装组件"],
    "Job queued": ["工作已排入佇列", "任务已加入队列"],
    "Job started": ["工作已開始", "任务已开始"],
    "Preparing Target Speaker Extraction worker manifest": ["正在準備目標說話者擷取 worker manifest", "正在准备目标说话者提取 worker manifest"],
    "Dispatching validated job to Linux Docker": ["正在將已驗證工作傳送至 Linux Docker", "正在将已验证任务发送至 Linux Docker"],
    "Linux Docker worker started": ["Linux Docker worker 已啟動", "Linux Docker worker 已启动"],
    "Linux Docker worker completed": ["Linux Docker worker 已完成", "Linux Docker worker 已完成"],
    "Docker artifacts registered": ["Docker 成品已註冊", "Docker 产物已注册"],
    "Job succeeded": ["工作已成功", "任务已成功"],
    "Target": ["目標", "目标"],
    "Clear": ["清晰", "清晰"],
    "No training experiments": ["沒有訓練實驗", "没有训练实验"],
    "VRAM": ["VRAM", "VRAM"],
    "ETA": ["預計時間", "预计时间"],
    "No ready candidate audio artifact": ["沒有可用的候選音訊成品", "没有可用的候选音频产物"],
    "No evaluation runs": ["沒有評估工作", "没有评估任务"],
    "Select a configured evaluation.": ["請選取已設定的評估。", "请选择已配置的评估。"],
    "Not measured by the current evaluation worker": ["目前的評估 worker 尚未量測", "当前评估 worker 尚未测量"],
    "Canonical benchmark artifact unavailable": ["Canonical benchmark 成品不可用", "Canonical benchmark 产物不可用"],
    "Full security qualification not emitted": ["尚未產生完整安全驗收結果", "尚未生成完整安全验收结果"],
    "No ready evaluation artifacts": ["沒有可用的評估成品", "没有可用的评估产物"],
    "VERIFIED ARTIFACTS": ["已驗證成品", "已验证产物"],
    "Evaluation artifact": ["評估成品", "评估产物"],
    "Subject": ["對象", "对象"],
    "Evidence": ["證據", "证据"],
    "No verified evaluation artifacts": ["沒有已驗證的評估成品", "没有已验证的评估产物"],
    "Not promoted": ["尚未升級", "尚未升级"],
    "Promoted": ["已升級", "已升级"],
    "Not reported": ["尚未回報", "尚未报告"],
    "Loaded · TensorRT": ["已載入 · TensorRT", "已加载 · TensorRT"],
    "Needs review": ["需要審查", "需要审核"],
    "Reject": ["拒絕", "拒绝"],
    "Target · accept": ["目標 · 接受", "目标 · 接受"],
    "Unavailable · worker does not emit structured loss": ["不可用 · worker 未提供結構化損失值", "不可用 · worker 未提供结构化损失值"],
    "No ready promotable artifact": ["沒有可升級的成品", "没有可升级的产物"],
    "No verified source": ["沒有已驗證來源", "没有已验证来源"],
    "Inspect gates": ["檢查門檻", "检查门槛"],
    "Inspect job events": ["檢查工作事件", "检查任务事件"],
    "Loading artifact…": ["正在載入成品…", "正在加载产物…"],
    "SYSTEM": ["系統", "系统"],
    "CHECKPOINT": ["檢查點", "检查点"],
    "EXPRESSION BANK": ["情感資料庫", "情感资料库"],
    "ENGINE": ["引擎", "引擎"],
    "EVALUATION": ["評估", "评估"],
    "READY": ["就緒", "就绪"],
    "No registered engine artifacts": ["沒有已註冊的引擎成品", "没有已注册的引擎产物"],
    "Item": ["項目", "项目"],
    "Kind": ["種類", "类型"],
    "Audio": ["音訊", "音频"],
    "Annotation": ["標註", "标注"],
    "Pipeline": ["處理流程", "处理流程"],
    "ready": ["就緒", "就绪"]
  });

  const textSources = new WeakMap();
  const attributeSources = new WeakMap();
  const LOCALIZED_ATTRIBUTES = Object.freeze(["aria-label", "title", "placeholder"]);
  let locale = detectLocale();

  function detectLocale() {
    let saved = "";
    try { saved = localStorage.getItem(LOCALE_KEY) || ""; } catch (_) {}
    if (SUPPORTED.includes(saved)) return saved;
    const browserLocale = (navigator.languages?.[0] || navigator.language || "").toLowerCase();
    if (browserLocale.startsWith("en")) return "en";
    if (browserLocale.includes("hans") || browserLocale.startsWith("zh-cn") || browserLocale.startsWith("zh-sg")) return "zh-Hans";
    return "zh-Hant";
  }

  function translateCount(value) {
    if (locale === "en") return value;
    let match = value.match(/^(\d+)\s+(projects?|profiles?|drafts?|jobs?|runs?|artifacts?|queued|running)$/i);
    const nouns = {
      project: ["個專案", "个项目"], projects: ["個專案", "个项目"],
      profile: ["個設定檔", "个配置"], profiles: ["個設定檔", "个配置"],
      draft: ["份草稿", "份草稿"], drafts: ["份草稿", "份草稿"],
      job: ["項工作", "个任务"], jobs: ["項工作", "个任务"],
      run: ["次執行", "次运行"], runs: ["次執行", "次运行"],
      artifact: ["個成品", "个产物"], artifacts: ["個成品", "个产物"],
      queued: ["項等待中", "个排队中"], running: ["項執行中", "个运行中"]
    };
    if (match) return `${match[1]} ${nouns[match[2].toLowerCase()][locale === "zh-Hant" ? 0 : 1]}`;
    match = value.match(/^(\d+) package · (\d+) local$/i);
    if (match) return locale === "zh-Hant"
      ? `${match[1]} 個套件 · ${match[2]} 個本機草稿`
      : `${match[1]} 个模型包 · ${match[2]} 个本地草稿`;
    match = value.match(/^(\d+)\s*\/\s*(\d+) sources selected$/i);
    if (match) return locale === "zh-Hant"
      ? `已選取 ${match[1]} / ${match[2]} 個來源`
      : `已选择 ${match[1]} / ${match[2]} 个来源`;
    match = value.match(/^(\d+) target · (\d+) review · (\d+) rejected$/i);
    if (match) return locale === "zh-Hant"
      ? `${match[1]} 個目標 · ${match[2]} 個待審 · ${match[3]} 個已拒絕`
      : `${match[1]} 个目标 · ${match[2]} 个待审 · ${match[3]} 个已拒绝`;
    match = value.match(/^Attempt (\d+) · (.+)$/i);
    if (match) return locale === "zh-Hant"
      ? `第 ${match[1]} 次嘗試 · ${match[2] === "initial" ? "初次執行" : match[2]}`
      : `第 ${match[1]} 次尝试 · ${match[2] === "initial" ? "初次运行" : match[2]}`;
    match = value.match(/^Priority ([+-]?\d+)$/i);
    if (match) return locale === "zh-Hant" ? `優先級 ${match[1]}` : `优先级 ${match[1]}`;
    match = value.match(/^Install (\d+) AI Components?$/);
    if (match) return locale === "zh-Hant" ? `安裝 ${match[1]} 個 AI 元件` : `安装 ${match[1]} 个 AI 组件`;
    match = value.match(/^(\d+) \/ (\d+) required ready$/);
    if (match) return locale === "zh-Hant" ? `必要元件 ${match[1]} / ${match[2]} 已就緒` : `必要组件 ${match[1]} / ${match[2]} 已就绪`;
    return value;
  }

  function t(value) {
    const source = String(value ?? "");
    if (locale === "en") return source;
    const translated = TEXT[source];
    if (translated) return translated[locale === "zh-Hant" ? 0 : 1];
    return translateCount(source);
  }

  function localizeTextNode(node) {
    if (!node?.parentElement || node.parentElement.closest("[data-i18n-static],script,style")) return;
    let source = textSources.get(node);
    if (source === undefined) {
      source = node.nodeValue;
      textSources.set(node, source);
    }
    const match = source.match(/^(\s*)([\s\S]*?)(\s*)$/);
    const next = `${match[1]}${t(match[2])}${match[3]}`;
    if (node.nodeValue !== next) node.nodeValue = next;
  }

  function localizeAttributes(element) {
    if (!(element instanceof Element) || element.closest("[data-i18n-static]")) return;
    let sources = attributeSources.get(element);
    if (!sources) {
      sources = {};
      attributeSources.set(element, sources);
    }
    for (const name of LOCALIZED_ATTRIBUTES) {
      if (!element.hasAttribute(name)) continue;
      if (!(name in sources)) sources[name] = element.getAttribute(name);
      const next = t(sources[name]);
      if (element.getAttribute(name) !== next) element.setAttribute(name, next);
    }
  }

  function localize(root = document.body) {
    if (!root) return;
    const elements = root instanceof Element ? [root, ...root.querySelectorAll("*")] : [...document.body.querySelectorAll("*")];
    for (const element of elements) {
      localizeAttributes(element);
      for (const child of element.childNodes) if (child.nodeType === Node.TEXT_NODE) localizeTextNode(child);
    }
  }

  function setLocalizedAttribute(element, name, source) {
    if (!(element instanceof Element) || !LOCALIZED_ATTRIBUTES.includes(name)) return;
    let sources = attributeSources.get(element);
    if (!sources) {
      sources = {};
      attributeSources.set(element, sources);
    }
    sources[name] = String(source);
    element.setAttribute(name, t(source));
  }

  function setLocalizedText(element, source) {
    if (!(element instanceof Element)) return;
    const canonical = String(source ?? "");
    const node = document.createTextNode(canonical);
    textSources.set(node, canonical);
    node.nodeValue = t(canonical);
    element.replaceChildren(node);
  }

  function updatePicker() {
    const button = document.getElementById("studioLocaleButton");
    const current = document.getElementById("studioLocaleCurrent");
    if (button) button.setAttribute("aria-label", t("Change interface language"));
    if (current) current.textContent = NATIVE_NAMES[locale];
    document.querySelectorAll(".studio-locale-option").forEach(option => {
      option.setAttribute("aria-selected", String(option.dataset.locale === locale));
    });
  }

  function setLocale(nextLocale, { persist = true, focus = false } = {}) {
    if (!SUPPORTED.includes(nextLocale)) return false;
    locale = nextLocale;
    if (persist) try { localStorage.setItem(LOCALE_KEY, locale); } catch (_) {}
    document.documentElement.lang = locale;
    localize(document.body);
    updatePicker();
    window.dispatchEvent(new CustomEvent("aniflive-tts:locale-changed", { detail: { locale } }));
    if (focus) document.getElementById("studioLocaleButton")?.focus();
    return true;
  }

  function closeMenu({ focus = false } = {}) {
    const button = document.getElementById("studioLocaleButton");
    const menu = document.getElementById("studioLocaleMenu");
    if (!button || !menu) return;
    menu.hidden = true;
    button.setAttribute("aria-expanded", "false");
    if (focus) button.focus();
  }

  function openMenu() {
    const button = document.getElementById("studioLocaleButton");
    const menu = document.getElementById("studioLocaleMenu");
    if (!button || !menu) return;
    menu.hidden = false;
    button.setAttribute("aria-expanded", "true");
    const selected = menu.querySelector(`[data-locale="${locale}"]`) || menu.querySelector(".studio-locale-option");
    selected?.focus();
  }

  function bindPicker() {
    const picker = document.getElementById("studioLocalePicker");
    const button = document.getElementById("studioLocaleButton");
    const menu = document.getElementById("studioLocaleMenu");
    if (!picker || !button || !menu) return;
    button.addEventListener("click", () => menu.hidden ? openMenu() : closeMenu());
    button.addEventListener("keydown", event => {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
        event.preventDefault();
        openMenu();
      }
    });
    menu.addEventListener("click", event => {
      const option = event.target.closest(".studio-locale-option");
      if (!option) return;
      setLocale(option.dataset.locale, { focus: true });
      closeMenu();
    });
    menu.addEventListener("keydown", event => {
      const options = [...menu.querySelectorAll(".studio-locale-option")];
      const currentIndex = options.indexOf(document.activeElement);
      if (event.key === "Escape") {
        event.preventDefault();
        closeMenu({ focus: true });
      } else if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        const nextIndex = event.key === "Home" ? 0 : event.key === "End" ? options.length - 1
          : (currentIndex + (event.key === "ArrowDown" ? 1 : -1) + options.length) % options.length;
        options[nextIndex]?.focus();
      } else if (["Enter", " "].includes(event.key)) {
        event.preventDefault();
        document.activeElement?.click();
      }
    });
    document.addEventListener("pointerdown", event => {
      if (!menu.hidden && !picker.contains(event.target)) closeMenu();
    });
  }

  const observer = new MutationObserver(records => {
    for (const record of records) {
      if (record.type === "childList") {
        for (const node of record.addedNodes) {
          if (node.nodeType === Node.TEXT_NODE) localizeTextNode(node);
          else if (node instanceof Element) localize(node);
        }
      } else if (record.type === "characterData") {
        localizeTextNode(record.target);
      } else if (record.type === "attributes") {
        localizeAttributes(record.target);
      }
    }
  });

  window.AnifLiveTTSStudioI18n = Object.freeze({
    t,
    localize,
    setLocalizedAttribute,
    setLocalizedText,
    getLocale: () => locale,
    setLocale,
    supportedLocales: SUPPORTED
  });

  bindPicker();
  setLocale(locale, { persist: false });
  observer.observe(document.body, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: LOCALIZED_ATTRIBUTES });
  window.addEventListener("storage", event => {
    if (event.key === LOCALE_KEY && SUPPORTED.includes(event.newValue)) setLocale(event.newValue, { persist: false });
  });
})();
