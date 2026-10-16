# B：512 token前台运行说明

本文是512前台流程的历史准备说明；用户现已完成两seed运行，实测状态以[两seed结果](b_v2_length512_two_seed/README.md)和[后续分工](NEXT_ACTIONS_512_20261016.md)为准。下文“尚未执行”描述准备当时边界，不覆盖新验收证据。分支保持`work/b-model`；原有导师讨论稿记录和其他窗口修改保留。

## 已冻结的实验

- A已验收`internal-v2`：train119842 / dev15086；路径与SHA-256在`main_model/configs/length512_v2.json`固定。仅使用train/dev，不解析内部test或官方验证集。若出现新数据目录或冻结哈希变化，退出并报告，不自动切版本。
- 来源：`FacebookAI/xlm-roberta-base`，revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`；原始权重和tokenizer文件按已验收来源清单逐个核验。不重新安装模型、不重新下载权重。
- 512 token、seed42、无加权、目标2000次**有效**优化器更新；batch1、梯度累积16、lr2e-5恒定、AdamW wd0.01 / foreach=False / max_grad_norm1、FP16、梯度检查点、num_workers0、确定性模式。
- 先从固定原始预训练模型运行50步资源节点；核对损失有限、实际512长度、计数和完整恢复状态。通过后复用该50步，完整恢复到500，再1000、1500、2000。50步不推理dev、不作为性能结论。恢复失败或资源失败立即退出，不改参数重试。
- 每500有效更新保留完整恢复检查点，额外保留50节点；500/1000/2000为预定dev评估节点。1500只保存，不评估。全部节点在运行前固定，不按中途分数改预算、早停或挑非预定检查点。
- 保存前后均FP32 / batch8；每个评估节点在内存中预测后保存模型/分词器，释放该GPU模型，再由独立进程加载预测。严格核对ID顺序/唯一/数量、概率有限与[0,1]、human0/machine1映射及rtol1e-5/atol1e-6一致性。
- A脚本显式`--threshold 0.5 --short-by word --short-threshold 100`。短文本是**空白计词<100的临时口径**，并非已核实的官方多语言计词。
- 原256/seed42/43/2000结果复用；seed42另有已测500结果。256/seed42/1000、256/seed43/500和1000为**历史评估缺失**，不从旧检查点追加推理、不重训补齐、不插值。
- 512/seed43为显式可选配置，默认不执行；若选择，应从新输出目录初始化并在所有后续命令保留`--include-seed43`。

## 使用方法

在项目根目录使用当前环境的`python`，路径无需写进共享代码。用户指定解释器的完整PowerShell命令见本次回复；通用命令如下：

```powershell
# 只做CPU数据/配置/历史对照/依赖核验，附一次性进程和GPU资源快照
python -X utf8 main_model/local_round.py --phase dry-run

# 首次启动；输出目录必须不存在。整个实验在当前终端顺序执行
python -X utf8 main_model/local_round.py --phase run

# 中断后：只从已验证的完整节点恢复，跳过已完成训练/有效评估
python -X utf8 main_model/local_round.py --phase resume

# 只重跑现有512检查点的500/1000/2000评估；不启动训练
# 需上述三个检查点均存在。输出新attempt，保留原预测/模型。
python -X utf8 main_model/local_round.py --phase evaluate

# CPU：只重画现有已验收预测并自动审计；不训练、不GPU推理
python -X utf8 main_model/local_round.py --phase plot

# CPU：只审计最近一次已导出图，不重新绘图
python -X utf8 main_model/local_round.py --phase audit

# CPU：只重建分析包
python -X utf8 main_model/local_round.py --phase bundle
```

所有命令默认`--config main_model/configs/length512_v2.json --output runs/b_v2_length512_round`。更换输出目录时，每次恢复/评估/绘图都应显式传入同一`--output`。如果只是绘图/审计失败，优先`--phase plot`或`audit`；不需要重跑训练。

可选seed43（仅在确实要运行时手动使用）：

**2026-10-16补充：seed42已完成后只补seed43，使用独立目录与新参数：**

```powershell
python -X utf8 main_model/local_round.py --phase run --seed43-only --output runs/b_v2_length512_seed43
python -X utf8 main_model/local_round.py --phase resume --seed43-only --output runs/b_v2_length512_seed43
```

以上run/resume二选一，不是连续执行两次。该模式只安排43；CPU回归和dry-run已通过，训练核心不变。下列`--include-seed43`用于两个seed都还未运行的全新计划，会从头运行42和43；已有42时不要用它补43。

```powershell
python -X utf8 main_model/local_round.py --phase dry-run --include-seed43 --output runs/b_v2_length512_two_seeds
python -X utf8 main_model/local_round.py --phase run --include-seed43 --output runs/b_v2_length512_two_seeds
```

两个seed顺序执行，均各自先通过50步门槛。不能在已运行的单seed输出目录内临时加seed43。

## 锁、恢复与失败行为

- 前台入口持有项目级OS文件锁，直接调用新版`train.py`也遵守同一把锁。训练和评估子进程只继承父进程授权；锁在进程退出后由OS释放。旧锁文件可存在，不代表锁仍被占用，不需要删锁文件。
- 启动前和每个GPU阶段前，使用psutil检查本项目训练/预测进程，附nvidia-smi资源快照。发现冲突、Python进程权限不足或GPU信息不可读则报错退出。不杀进程、不轮询、不创建后台监控。其他软件的桌面显存使用会记录，不等同于本项目训练冲突。
- 模型、优化器、scaler、Python/NumPy/PyTorch/CUDA RNG、样本游标、累计尝试/有效更新/跳步、样本数、累计训练时间和峰值显存均随完整状态保存。新轨迹绑定训练源码哈希和数据/参数签名；不允许改长度、学习率或训练代码后继续同一轨迹。
- 评估在训练子进程退出后进行；后续训练再次从保存的随机状态/样本游标恢复，dev不参与梯度更新。GPU FP16跳步可能随长度改变，`sampling_audit.json`分别比较同一attempt前缀及同一有效更新终点的样本链，暴露不同时明确报告。
- Ctrl+C或异常时，当前未完成的梯度累积不冒充检查点。保留最后成功的原子检查点及各attempt日志/failure.json；恢复最多重做该检查点之后尚未持久化的训练。保存中残留的`.pending`不算有效节点，程序不会覆盖它。
- 已完成检查点/预测不会被清理或覆盖。新尝试使用新目录；损坏的完成检查点会明确阻塞，不自动回退掩盖问题。请保留错误与分析包交给Codex处理。
- 详细训练日志逐attempt落盘；终端每100有效更新显示简短进度及节点状态。没有按dev成绩触发的控制逻辑。

## 输出与磁盘

默认目录`runs/b_v2_length512_round/`：

```text
config.json, protocol.json, preflight.json, preflights/  固定协议/代码版本/dirty状态/依赖和数据指纹
stages.json, latest_outcome.json                        各阶段状态/本次命令结论
failure.json, failures/（异常时）                       历史失败原因；成功恢复后不删除
seed-42/resource_gate.json                             50步资源与完整状态检查
seed-42/training/to-*/attempt-*/                       训练日志、统计、模型/分词器、完整恢复状态
seed-42/evaluation/step-*/attempt-*/                   dev预测、独立重载验证、A指标、导出模型
sampling_audit.json                                   长度对照的抽样/暴露比较
figures/attempt-*/                                    图、图源CSV、布局/文字/碰撞审计
artifact_manifest.json                                预测/图/图源文件列表及SHA-256
bundles/attempt-*/                                    小型分析材料
bundles/attempt-*_analysis_bundle.zip                  发给Codex的分析包
```

每个完整XLM-R恢复点含约1.1GB模型和约2.2GB优化器状态，5个保护节点加3份推理导出模型，单seed预计约20–25GiB，失败尝试和重评会继续增加。建议预留30GiB以上；程序启动/恢复时要求每个seed至少25GiB空闲，双seed按50GiB检查。不自动删除旧运行腾空间。现有256对照只读取，不复制其完整权重。

分析包只收集预测、配置/版本/来源/指标/阶段状态、摘要、必要图表/图源和指纹；不收集完整权重、环境、原始正文或全部训练日志。大小取决于图和ROC源表，通常远小于模型目录，不设虚构的实测体积。

包中的`failure_details.json`收集小型异常记录及失败阶段日志尾部，不是全部运行日志。`failure.json`可能是恢复前的历史异常，当前状态以`latest_outcome.json`和`stages.json`为准。

## 绘图合同和依赖

Python matplotlib；复用现有导出逻辑和同一套A/分类指标，新增本轮节点汇总。输出全体/临时短文本宏F1、临时S、两类召回、最终节点混淆矩阵、ROC、可靠性、log loss/Brier、训练损失以及速度/显存对比。所有已通过节点的指标写入源表；ROC/可靠性/混淆矩阵主要比较2000步终点，不能把这种展示选择当成按分数挑模型。

图标签使用“长度 / seed / 有效步数”，图注明internal-v2、0.5阈值及临时空白计词规则。seed分别展示；同seed/长度历史日志确认前缀后只画最长损失轨迹，不重复叠画CE500前缀。历史缺失节点仅在记录中标缺失，不生成虚假点。

输出300dpi PNG、可编辑PDF/SVG、图源CSV。渲染时布局以1.5pt容差阻塞导出；随后执行PDF字形≥5pt与严格碰撞审计。源预检中与本报告合同不符的期刊尺寸/TIFF等启发式WARN，保存原报告并解释原因；真实几何FAIL、严格碰撞WARN、缺依赖或无法审计都不能标PASS。人工视觉复核始终**运行后待检查**，自动审计通过不代表最终图表验收。

依赖声明：`requirements/b-local-round.txt`；审计脚本已随代码放入`main_model/figure_audit/`，运行时不需要个人skill目录、Codex、MCP或联网。指定环境已按用户要求安装缺少的matplotlib/PyMuPDF/Pillow及其依赖，没有更换torch/numpy版本。

图的科学问题为“更长输入在固定有效更新预算下是否改善检测，代价和概率质量如何”；证据链依次是固定节点指标与类别取舍、终点错误与概率诊断、实际训练轨迹和资源约束。统计单位为固定dev样本；各seed是独立训练结果，不报告跨seed均值或伪造训练不确定性。

新引擎为准确计时增加每次attempt末尾CUDA同步，累计显存峰值会跨恢复段保留；旧256结果的计时边界略有区别。因此速度图属于操作性对照，应连同代码版本和50步资源实测解释，不能单凭一次速度差宣称长度造成精确比例的性能开销。

## 本轮必要检查与未验证事项

已完成目标Python语法检查、固定train/dev及原始模型来源哈希核验、三项历史对照预测与验收状态检查、必要CPU回归，以及使用已有256实测产物的绘图/自动审计试跑。CPU回归的微型模型仅用于工程正确性，不计比赛性能。

尚未执行：512的50步资源验证、正式训练、任一GPU评估、GPU FP16分段恢复实测、真实512图表自动审计和人工视觉复核、可选seed43、A跨机器复现。512可能OOM或与256有不同跳步；程序会停止记录，不自动调参数。GPU路径在本轮只静态检查，不能声称端到端GPU验收通过。

运行结束后优先提供最新的`*_analysis_bundle.zip`。若提前失败且包未生成，提供`stages.json`、`failure.json`、最新阶段日志末尾及`preflight.json`；不要上传完整权重或环境。阶段文件给出出错attempt和最后有效检查点位置。
