# B：先验收当前融合，再开展分类头Dropout对照

## 最新状态：用户已完成本地验收

2026-10-17交付范围更新：先完成现有候选的第一版，Dropout训练继续暂停。通用运行步骤以[新运行说明](INFERENCE_RUNBOOK_20261017.md)为准，A新PR #10的文字回执及待核对的CPU容差差异见[冻结草案](RELEASE_FREEZE_DRAFT_20261017.md)；下面的Dropout流程保留为后续版本历史计划。

2026-10-17补充：用户确认模型ZIP已发送A，待接收确认及真实独立复现；不再处于待发送状态。[最新协议复核](DROPOUT_PROTOCOL_REVIEW_20261017.md)与[给C的B08报告材料](B08_REPORT_MATERIALS_FOR_C_20261017.md)已整理。本轮不启动训练，下面“权重稍后交A”等为此前交接记录。

当前`PASS_B_LOCAL_PENDING_A`：200条smoke及15086条dev通过，S=.963938808878，log loss=.148132168743，Brier=.034056804047。不要重复运行下方历史本地验收计划。实测/误差/耗时和下一步见[FUSION_LOCAL_ACCEPTANCE_20261016.md](FUSION_LOCAL_ACCEPTANCE_20261016.md)，必要证据见`b_fusion_local_acceptance/`。

用户选择本次只交代码、证据和模型清单；同一原始推理包保留本机`runs/b_fusion_acceptance_final/inference_bundle/`，权重稍后单独交A，尚未进入仓库。原bundle.json SHA保持`5775334edda46069d45f7eaade2222bea2c7393d19c6faae44a3ed4a256ab50a`。A直接按最新交接命令在本人独立环境复现，不必等官方未给出的细则；旧CSV复评不能替代模型复现。B07未完成，Dropout.2不启动，等待A对同一包真实回执。

已明确题面二分类/概率、S公式及短文本<100词；内部只继续临时空白计词/.5及p_machine。多语言计词实现、概率转类别和准确CSV模板仍未说明；示例日期不是实际截止时间。以下为准备当时的原计划与来源记录，其中“未推理/未上传”等描述只适用于准备阶段，最新状态以上段和本地验收交接为准。

## 准备阶段记录与可复用命令

本轮完成代码和必要CPU检查，不运行正式训练、XLM-R GPU推理或后台任务。继续“B准备代码、用户前台运行、结束后集中分析”。A PR #7已合main `0708c12`，C PR #8已合main `67dbf3f`，本地work/b-model已快进同步；原有未提交修改保留。本轮新增B代码尚未提交/推送。

## 已完成与当前边界

- C模型在B Python3.11.17 / NumPy2.4.6 / SciPy1.17.1 / scikit-learn1.9.1 / joblib1.6.0下，完整dev15086条CPU预测与C参考最大差`4.99960073341299e-11`，无警告，rtol1e-6/atol1e-8通过。该差异兼容参考CSV只有10位有效数字；不要求跨环境CSV字节完全相同。没有重训C。
- 当前候选仍B512/seed43/2000、分类头有效dropout0.1。统一入口仅输入正文及字符串sample_id，输出`0.7*p_C+0.3*p_B`；human0/machine1、原编号顺序，FP32/batch8、512token。不新增权重、阈值或校准器搜索。
- 已在本地`runs/b_fusion_acceptance_final/inference_bundle/`整理独立推理包，含一个B推理模型/分词器、C模型、推理/核验源码、三份参考概率和依赖清单。没有优化器、其他seed或训练环境。包的自校验通过，尚未运行该包的真实B预测；H05未完成。
- 官方提供的说明未规定计词、分类阈值或最终CSV模板，不把缺失细则作为开发阻塞。当前.5、空白计词<100及临时S仅作内部约定；最终输出保留machine概率，不先阈值化。模型仅依赖text，不依赖官方val未保证提供的灰盒字段。

## 现在由用户运行：当前候选融合验收

在仓库根目录使用自己的环境python：

```powershell
# 只核对配置、已接收模型、固定dev和磁盘；不推理或训练
python -X utf8 main_model/fusion_workflow.py --phase dry-run

# 前台顺序运行：独立包smoke_dev，再完整dev；不训练
python -X utf8 main_model/fusion_workflow.py --phase run

# 中断恢复：核验并跳过完成节点；失败节点使用新attempt
python -X utf8 main_model/fusion_workflow.py --phase resume

# 显式只重新做推理/评估，保留已有attempt；不训练
python -X utf8 main_model/fusion_workflow.py --phase evaluate

# 只重新整理小型分析包，不推理/训练
python -X utf8 main_model/fusion_workflow.py --phase bundle
```

默认输出是`runs/b_fusion_acceptance_final`，已准备好，无需再次prepare。不要使用历史准备目录`runs/b_fusion_acceptance`启动本轮；它保留为CPU检查来源。本轮按模型/参考/依赖/源码核验复用了其完整C CPU结果，没有重复35秒CPU推理，也没有覆盖已完成文件。

入口持有项目GPU锁；打包子进程继承锁，遇项目训练/推理冲突即退出，不杀进程、不轮询。每节点输出C.csv、B.csv、predictions.csv、预测清单、阶段状态和A word评估。对同一dev参考核对编号/顺序/唯一/行数/有限[0,1]、概率方向；C容差rtol1e-6/atol1e-8，B及融合沿用rtol1e-5/atol1e-6，并要求.5分类决策完全一致。每次新预测都用同样FP32/batch8/512设置。

保存：`unified_smoke/attempt-*`、`unified_dev/attempt-*`、根stages.json、C_cpu_acceptance.json、通过后acceptance.json；异常保存failure.json和失败阶段。`analysis/attempt-*.zip`收集阶段、指标、预测、SHA及必要失败尾部，不含权重/环境/正文/完整日志。程序非零退出即可停止，不依赖Codex在线监测。

完整dev原B推理实测约250秒，另有C CPU及加载/指纹时间；预计本轮数分钟，实际以日志为准。当前推理包约1.1GB量级，另需少量预测/分析空间。所有预备包和原runs均留本机，没有上传仓库。

## 本地通过后交A独立验收

首先提供最新`runs/b_fusion_acceptance_final/analysis/attempt-*.zip`供集中分析。通过时acceptance.json只标`PASS_B_LOCAL_PENDING_A`，不冒充A验收。

A取得同一推理包并安装其中requirements.txt（先按机器安装合适torch构建），使用A冻结的smoke_dev和完整dev。包不需要仓库、Codex、MCP或网络调用，A在自己的工作目录运行：

```powershell
python PATH_TO_BUNDLE/fusion_inference.py --bundle PATH_TO_BUNDLE --input PATH_TO_SMOKE_DEV.jsonl --output NEW_SMOKE_OUTPUT
python PATH_TO_BUNDLE/verify_fusion_output.py --bundle PATH_TO_BUNDLE --input PATH_TO_SMOKE_DEV.jsonl --predictions NEW_SMOKE_OUTPUT --output NEW_SMOKE_CHECK.json

python PATH_TO_BUNDLE/fusion_inference.py --bundle PATH_TO_BUNDLE --input PATH_TO_DEV.jsonl --output NEW_DEV_OUTPUT
python PATH_TO_BUNDLE/verify_fusion_output.py --bundle PATH_TO_BUNDLE --input PATH_TO_DEV.jsonl --predictions NEW_DEV_OUTPUT --output NEW_DEV_CHECK.json
```

默认CUDA；无CUDA可显式`--device cpu`，但完整XLM-R CPU推理可能很慢。这里不读取内部test或官方验证集。输出新目录，参考核验不覆盖报告。A应回传两个检查结果、实际环境/警告和GitHub回执，注明精确bundle.json SHA。推理包里的参考概率仅用于固定内部dev验收。

只有真实A验收完成后，在B本机`.local/fusion_acceptance/A_receipt.json`记录回执，不伪造PASS：字段需有`reviewer="A"`、`status="PASS"`、`bundle_sha256`（与dropout02_v2.json固定值相同）、`evidence_url`（实际GitHub回执链接）。该记录不由本轮自动生成。此门槛针对独立推理复现，不要求A补出官方未规定的细则。

## Dropout0.2：已准备，等待上述验收后再启动

你的解释成立：后半程出现“分类收益很小、部分错误更自信”的现象，但不能仅凭累计训练损失证明过拟合；AUC和类别取舍也在变化。因此0.2只是一项受控正则化候选，不宣布最优，不放弃已有2000步候选。

固定干预：仅XLM-R分类头dropout从有效0.1→0.2（该模块在分类头forward中调用两次）；编码器hidden和attention以及所有encoder Dropout模块保持0.1。微型CPU模型检查参数权重不被该设置改动，配置保存重载后分类头仍为0.2。新实验从固定原始revision重新初始化，不能从0.1检查点续训；恢复状态签名包含dropout值与实现指纹。

固定internal-v2 train/dev，不重划、不换版本；512token、无加权、seed42和43两项都执行，每项2000有效更新。其余已验证参数不变。各seed先50步资源门槛，通过后完整恢复同一轨迹继续；每500保存，500/1000/2000评估dev，1500只保存。资源失败停止，不自动换参数。已有0.1两seed六个评估节点全部复用，不补训或插值历史缺失。

```powershell
# 现在即可CPU dry-run；输出计划不代表门槛已通过
python -X utf8 main_model/local_round.py --config main_model/configs/dropout02_v2.json --output runs/b_v2_dropout02_round --phase dry-run

# 仅在当前融合B本地通过且A真实独立回执就绪后，用户自行启动
python -X utf8 main_model/local_round.py --config main_model/configs/dropout02_v2.json --output runs/b_v2_dropout02_round --phase run

# 中断恢复
python -X utf8 main_model/local_round.py --config main_model/configs/dropout02_v2.json --output runs/b_v2_dropout02_round --phase resume

# 只重评已有检查点，或只重画现有预测；不会启动训练
python -X utf8 main_model/local_round.py --config main_model/configs/dropout02_v2.json --output runs/b_v2_dropout02_round --phase evaluate
python -X utf8 main_model/local_round.py --config main_model/configs/dropout02_v2.json --output runs/b_v2_dropout02_round --phase plot
```

两个seed顺序运行，预计保护完整恢复状态合计约40–50GiB；程序要求至少50GiB空闲，建议60GiB以上。不删除旧实验腾空间。没有A真实回执、候选包不匹配或本地融合未通过时，run/resume在启动训练之前报错。

新代码继续绑定源码/数据/环境签名；因此旧0.1训练轨迹的严格恢复不能直接使用更改后的训练源码。旧权重/预测不受影响，旧运行的evaluate/plot/audit/bundle不要求当前训练源码等于历史源码，但仍核对保存状态和输入指纹。若确需继续旧训练，须另行恢复当时源码/环境，不能放宽签名绕过。历史源码哈希原样保留，不替换成新值。

## 自动比较与绘图

结束后分析包的`regularization_comparison/single_and_fixed_fusion.csv`包含两seed、两个dropout及所有实测预定节点的单B和固定.7C+.3B指标：全体/临时短文本F1、临时S、两类召回、错误数、高置信human误报、log loss、Brier、AUC和实际暴露/跳步。融合预测也保存；不新搜权重、校准或阈值，不只报告较好seed，不自动替换现有候选。

沿用nature-figure的Python matplotlib实现，新增按长度/seed/**dropout**区分轨迹，不把0.1和0.2当成一条前缀。图源CSV、300dpi PNG、可编辑PDF/SVG、1.5pt布局、字形大小和严格碰撞审计继续接入；指标/ROC/召回/混淆矩阵/概率质量/逐更新损失/速度显存均复用原流程。训练曲线为10次有效更新的逐更新CE滑动均值，不把累计损失与dev log loss直接相减解释泛化差。

准备阶段仅用真实0.1结果测试渲染和审计；没有0.2实测分数。新实验运行后的人工视觉复核仍为“运行后待检查”；自动通过不代表最终图表验收。图失败可独立plot/audit恢复，不触发重训。融合收益的完整对照在源表中，不能只看单B图。

## 本轮文件与未执行事项

主要新文件：fusion_inference.py、fusion_workflow.py、verify_fusion_output.py、dropout_protocol.py、dropout_results.py、dropout02_v2.json、test_fusion_dropout.py和requirements/fusion.txt。复用并小改train/training_engine/local_round/plot_local_round/gpu_guard；历史分析器将“当前源码是否仍相同”作为记录，不错误要求历史实验等于后续开发源码。

必要检查与包指纹见`b_fusion_dropout_preparation/checks.json`。指定环境依赖齐全，无需新增安装；12项CPU回归通过，包含微型合成正确性fixture，非比赛性能。

尚未执行：独立推理包的真实B/XLM-R预测、统一入口smoke/full-dev GPU验收、A独立复现、Dropout0.2正式训练/资源节点及其新性能/新图人工复核。内部test和官方val始终未读取或推理。下一步只由用户先运行当前融合验收，然后提供小分析包；不需要上传完整模型给Codex分析。
