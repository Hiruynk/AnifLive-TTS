# AnifLive-TTS Studio 使用说明

[English](ANIFLIVE_TTS_STUDIO_GUIDE.md) · [繁體中文](ANIFLIVE_TTS_STUDIO_GUIDE.zh-TW.md)

AnifLive-TTS Studio 是 AnifLive-TTS v1.4 的本地语音制作工作站，把语音合成、情感参考、数据集处理、目标说话人提取、V2ProPlus 训练、评估、TensorRT 打包与 GPU 任务管理集中在同一界面。原有 AnifLive-TTS WebUI 仍独立保留。

## 启动 Studio

1. 启动 Docker Desktop，等待 Linux engine 就绪。数据集神经 worker、训练、评估和 TensorRT build 只在 Linux Docker 运行，不在 Windows host process 运行。
2. 需要推理 API 时双击 `run_tts.bat`，等待 health check 显示 ready。
3. 双击项目根目录的 `run_studio.bat`，只启动 Studio。
4. 如果浏览器没有自动打开，访问 `http://127.0.0.1:9891/`。
5. 保持各 launcher 终端开启；关闭窗口会停止该 launcher 所属服务。

Studio 会检测 `9880` 或 `9882` 的 API。两者都离线时，数据、项目和任务管理仍可使用，但依赖推理的操作会保持不可用。工作站元数据默认存放在 `data/workstation`。

如需使用非默认 API 地址，请在启动 Studio 前设置 `ANIFLIVE_TTS_WEBUI_UPSTREAM`。它只会改变语音合成 upstream，不会改变 Studio 的 `9891` 端口。

右上角地球按钮可在 English、繁體中文、简体中文之间切换，并同步更新所有 Studio 页面与内嵌语音合成工作站。

## 界面基本操作

- 左侧导航按照制作流程排列：Create、Data、Train、Deploy、System。包括目标说话人提取在内的所有数据获取都从 **数据集** 开始。
- 地球按钮控制 Studio 界面语言；语音语言是语音合成、数据集转写与评估中的独立设置。
- 风格化选择器不是浏览器原生下拉框。点击字段，在浮动面板选择项目，并确认字段标签已更新后再继续。
- 文件和文件夹字段支持 **选择**、拖放及允许的手动路径。只有字段显示解析后的路径才表示已接收；路径必须位于已配置的 import root。
- 不可用按钮会显示原因，通常是尚未选择必要项目、artifact、AI 组件、API 或 qualification evidence。
- 创建或排队操作会先显示过渡状态，再把新记录加入列表；进度状态仍可见时不要重复提交相同操作。

## 总览

总览显示当前模型、推理环境状态、项目统计、运行中任务与最近工作。可直接打开最近项目，或进入负责下一阶段的模块。所有语音数据获取都从 **数据集** 开始；目标说话人提取不是另一种普通用户项目。

全屏动态背景可在 **设置 → 工作站行为** 关闭；系统的减少动态效果设置也会自动生效。

## 语音合成

语音合成页保留完整的 AnifLive-TTS v1.3 合成工作站功能。

1. 选择音色模型。
2. 输入文字；默认的 `今日はいい天気ですね。` 可以直接替换。
3. 选择语音语言。
4. 高亮指定文字，为该段选择情感参考。
5. 检查情感标签、每段语言、停顿与顺序。
6. 点击 **立即播放** 进行流式播放，或下载完整音频。

播放时，当前读到的文字会变成金色；情感下划线与描述固定在标注位置。没有任何标注时，模型使用原生平静表达。

分段设置可为每个已确认片段指定不同语言和停顿。可拖动把手或使用无障碍移动按钮排序。**停止** 只取消当前流，不会更改稿件。

## 情感资料库

情感资料库同时管理模型包内的情感与本地参考草稿。

1. 打开 **情感**，点击 **新建情感**。
2. 输入稳定的 profile ID、名称、模型 ID、参考音频路径、语言、情感与强度。
3. 添加一行或多行描述，以及可选的 VAD／韵律数据。
4. 点击 **分析** 测量时长、起音、音高、语速与参考音频身份数据。
5. 保存草稿。

草稿不能通过手动切换状态变成已验证情感。必须选择已通过的评估证据再升级，以保留参考音频、测量数据与验收结果之间的可追溯关系。

## 数据集工厂

数据集工厂是创建训练数据的唯一入口。点击 **新建数据集** 后选择一种获取模式：

- **音频／视频集合**：处理本身基本为单一说话人的录音。
- **提取目标说话人**：使用一段短参考音频，从长录音或多人媒体中找到指定说话人。
- **现有 GPT-SoVITS 数据集**：导入已审核的 `.list`，不会使用文本作为文件名。

每个项目都使用同一条可见流程：**来源 → 说话人 → 清理 → 文本 → 审核 → 风格 → 就绪**。

### 音频或视频集合

1. 添加一个或多个允许目录内的音频、视频或文件夹路径。
2. 选择受管理的语音识别后端；五语默认是 SenseVoice Small，安装对应组件后也可使用 Faster Whisper。
3. 点击 **开始准备**。Linux Docker 会解码媒体、生成标准 PCM、运行受管理 VAD、在静音安全位置切片、转写并记录信号质量。
4. 在片段检查器修正文本和语言、确认说话人、按需添加情感，再接受或拒绝。

### 提取目标说话人

1. 添加目标短参考音频和一个或多个源录音；不需要已训练的音色包。
2. 点击 **开始准备**。Linux worker 先把参考音频中的有效语音建立为多区段身份原型，再依次运行受管理 FSMN-VAD、工作站专用 TensorRT ERes2NetV2 验证、双阈值 Sortformer 证据、纯度路由、仅用于可挽救候选的 MossFormer2，以及受管理 ASR。
3. 干净目标片段保留原音；疑似污染送去分离；不确定结果保留在审核；非目标片段保留在拒绝区。
4. 仅在需要检查每阶段证据时展开 **高级说话人提取**；它会显示参考支持数、主要与敏感 diarization、干净 SNR gate、离线状态和完整任务链。它是检查器，不是第二条流程。

目标说话人模式会为每个源范围生成 `seg_000184_7f42a1.wav` 一类稳定文件名，不会先拼接成一条 WAV 再重新 VAD。每个条目保留源 checksum、sample 级范围、目标相似度、路由、分离证据、原音、可选处理音频与 ASR 来源。

该获取策略也参考了公开 [Timbre](https://github.com/Etherll/Timbre) 项目的证据优先思路：多参考身份、先 diarization 后分离、静音验证切点、强制切分隔离、验证融合、保留拒绝片段以及可续接的 stage 产物。AnifLive-TTS 使用自己的通用 TensorRT verifier、worker contract 与质量 gate，没有捆绑 Timbre 源代码或模型资产。

### 审核、风格与冻结

1. 点击 **审核队列** 并使用 Studio 播放器；每一条 ASR 建议都必须人工确认，不能自行成为正式训练文本。
2. 使用 `Space` 播放，再用界面上的接受／拒绝控件决定片段。非静音强制切分、身份不明确或分离后 gate 失败都会留在审核。
3. 按需添加情感与强度。SenseVoice 情感仅作建议，人工标签才是正式数据。
4. 点击 **情感候选**，按说话人身份、信号质量、可用时长与标注完整度排序。采用候选只会创建情感资料库草稿，不会自动成为已验证 profile。
5. 分别核实每个已接受片段的逐字稿与说话人；修改内容后，原核实会失效。分配 train／validation／test（默认 85／10／5）后点击 **冻结数据集**。只有要求情感标注的项目才需要核实情感。
6. 可下载 **Manifest** 或只根据证据生成的 **验收报告**；缺少训练、引擎或正式验收时会逐项列出，不会猜测为成功。
7. 点击 **继续前往训练**，Studio 会从已冻结 manifest 创建原有 Training 项目，不会在 Dataset Factory 重造训练界面。

源媒体与生成文件保留在磁盘；Studio 记录来源关系与审核状态，不会把任意文件塞进浏览器存储。

## AI 组件

首次运行神经网络数据流程前，打开 **设置 → AI 组件**。

- 可逐个明确安装必要组件，或一次安装所有缺失的必要组件。
- 完整目标说话人流程需要 Speaker Verification、SenseVoice Small、FSMN-VAD 与 MossFormer2 全部显示 **Ready**。
- 每个资产都固定 revision、大小、SHA-256、license 与 Linux worker runtime。
- 联网电脑使用 **导出 AI 组件包**，离线工作站使用 **导入 AI 组件包**；正常 worker 始终以 `--network none` 运行。
- DeepFilterNet 是可选组件；在有后端通过 Linux Docker 质量 gate 前，去混响保持不可用。

## 模型训练

训练使用 GPT-SoVITS V2ProPlus 输入，通过 GPU 任务系统在 Linux Docker 运行。

1. 冻结已审核数据集后点击 **继续训练**，或新建训练项目时选择合格的 frozen dataset。
2. 选择 **快速**、**平衡**、**高质量** 或 **高级**。高级设置包括阶段、epoch、batch、learning rate、checkpoint 间隔、seed 与 gradient checkpointing。
3. 加入训练队列并监控状态、loss、VRAM、GPU 使用率、温度、ETA、日志和 checkpoint。在 **任务** 页使用 worker 支持的暂停、继续、取消及重试功能。
4. 等待 validation checkpoint selection。保存的 GPT／SoVITS epoch 都是候选，最后一个 epoch 不会自动成为部署胜者。
5. 聆听本次新生成的盲选 reference 音频并提交决定；之前 run 的选择不适用于新一组音频。候选只来自已核实的 train 片段。
6. checkpoint 胜者及 reference 锁定后，才首次打开 sealed test，且只评估一次。holdout 失败即停止该 run，不可更换胜者后重用同一 test。

接受音频不等于核实逐字稿或说话人，ASR 仍只是建议。没有核实证据的旧 frozen dataset 不可创建新的 production training。

### 构建可运行的 TensorRT 模型

只有 reference 锁定及 holdout 通过后，流程才会进入引擎构建。**构建 production model** 不会跳过这些条件。

1. Linux builder 构建全部九个 TensorRT 11 engine。检查 build report 与 enqueue verification；只有 engine 文件并不足够。
2. 等待 PyTorch → ONNX → TensorRT conversion parity。parity 失败会阻止打包。
3. 在 **模型** 检查 package manifest、checksum、runtime fingerprint 及 checkpoint／reference 来源关系。
4. 完成正式多语言与流式评估、canonical benchmark 及必要人工盲听后，才可 qualification 与 promotion。

已构建的 package 在所有关卡通过前仍是 qualification-pending。selection 失败可以在 **任务** 页复用已有训练结果重试，不代表必须重新训练。

完整 production chain：

`frozen dataset → training.prepare → checkpoint.select → reference.select → human lock → holdout.evaluate → engine.prepare → conversion.parity → model.package → evaluation.prepare → qualification → promotion`

## 评估实验室

评估实验室比较候选模型，并以 fail-closed 方式组合正式验收。

1. 新建或选择评估项目。
2. 选择音频语言、已完成的基线与上游模型包依赖。
3. 单独加入评估任务，或加入 engine → package → evaluation 工作链。
4. 检查五语 CER／WER、speaker similarity、streaming parity、时长、TTFA、RTF、情感与长句结果。
5. 有盲听音频时进行 A/B 比较。
6. 自动评估、长句盲听、情感盲听和安全证据全部选齐后，才可组合 qualification。

证据缺失时会明确保持不可用，不会把未完成结果包装成通过。

## 模型与 TensorRT 引擎

**模型** 显示数据集、checkpoint、情感资料库、engine、package 与 qualification 的完整来源关系。选择产物可查看路径、checksum、父产物、测量结果和升级状态。

**引擎** 显示设备专用 TensorRT 产物与运行时兼容性。已验证模型包绑定 TensorRT／CUDA／GPU fingerprint；目标环境变化后应重新构建，不应沿用过期 engine。

只有通过 qualification 的产物才能升级；仅选择产物不会让它成为 production-ready。

## GPU 任务

任务页会串行安排需要独占 GPU 的流程并保存依赖关系。

- 选择任务类型、项目、优先级与前置任务。
- 检查等待原因、尝试次数、依赖、状态、进度和事件日志。
- 只有当前状态允许时，才会显示暂停／继续／取消／重试。
- 长时间任务应到达安全 checkpoint 后再关闭 Studio。

Docker 任务需要已配置的 Linux worker。worker 不可用时，任务会保持 blocked 并显示原因，不会暗中改为在 Windows 运行神经网络。

## 设置

设置页控制默认语音／评估语言、session continuity、训练 preset、目标说话人阈值、刷新间隔与总览背景。修改后点击 **保存设置**。

**Import roots** 决定 Studio 可浏览或接受拖放的本地路径。**Worker/runtime paths** 选择 Docker 任务使用的受控目录；请使用相邻的文件夹选择器并核对解析后的值，不要依赖未经验证的手动路径。**AI 组件** 管理固定版本的 worker asset 和离线 bundle。设置只影响后续任务，不会重写已完成 artifact。

应用区域显示浏览器／PWA 状态；系统允许安装时，**安装 AnifLive-TTS Studio** 可添加本地 Web App，不会取代普通浏览器地址。

### 清除界面记录

点击 **清除界面记录**，阅读确认内容后再确认。Studio 只记录当前时间，并在列表中隐藏较旧的项目、任务、产物、验证和本地情感草稿；SQLite 数据行、音频、模型包、checkpoint、报告和文件全部保留。清除后新建的记录会正常显示。

## 键盘与无障碍操作

- 使用 `Tab` 移动到各控件。
- 所有风格化选择器都支持 `Enter`／空格打开、方向键移动、`Home`／`End` 跳转、`Enter` 选择和 `Escape` 关闭。
- 焦点不在文本输入框时，`1`–`9` 可打开主要模块，`0` 打开任务页。
- 状态与情感不仅依靠颜色识别，同时保留文字标签。

## 故障排查

- **Studio 无法启动：** 从终端运行 `run_studio.bat` 并阅读保留日志；确认 Python 3.10–3.12 及必要依赖。
- **9891 已被占用：** 停止现有 Studio；启动器不会擅自终止未知进程。
- **语音合成离线：** 启动 AnifLive-TTS API 后刷新；数据工作仍可离线使用。
- **音色没有出现在语音合成：** 确认它是包含九个已验证 TensorRT engine 的完整 model package，然后重新加载或重启 API。单独 `.ckpt` 与 `.pth` 不能成为可选 runtime voice。
- **立即播放保持不可用：** 选择 API 返回的模型、输入非空文本、修正无效情感分段，并确认 API 状态为 ready。
- **选择器打开但选择后没有变化：** 请选择可用项目而不是 placeholder，再确认字段标签已经更新。如果所有项目都不可用，请先完成状态文字所示的 prerequisite。
- **选择器没有返回可用路径：** 选择已配置 import root 内的文件或文件夹，或先在设置中加入其父目录再重试；浏览器安全限制不允许任意探索文件系统。
- **GPU 任务 blocked：** 到任务页检查依赖／资源原因，确认 Docker Desktop、NVIDIA container 与 worker 配置。
- **准备流程提示缺少组件：** 到 **设置 → AI 组件** 安装或导入指定固定版本；worker 运行期间不会自行下载。
- **无法导入文件：** 把文件放在允许的 import root；Studio 会拒绝路径穿越和未批准位置。
- **无法升级模型：** 到 qualification composer 补齐所有必要证据。
