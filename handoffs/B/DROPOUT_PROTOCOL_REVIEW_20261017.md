# B：下一轮分类头Dropout协议复核

日期：2026-10-17（Asia/Shanghai）。结论：现有配置、训练入口和对比逻辑一致；CPU检查通过。仅完成准备复核，本轮不启动训练。A独立同包验收回执缺失，仍不可启动。

后续状态更新：用户已改为优先交付现有候选，本协议留作后续版本，不再作为第一版交付前置。A新PR #10报告通过，但机器报告与CPU容差差异仍待B复核，见[冻结草案](RELEASE_FREEZE_DRAFT_20261017.md)。下文“回执缺失”等为本轮早先CPU复核时的快照；即使验收问题关闭也不自动启动训练。

## 问题、干预与冻结范围

问题：分类头Dropout从有效0.1改为0.2后，在相同seed和预定预算下，能否改善开发集概率质量，并保持单B及固定融合的分类表现？当前现象提供实验动机，不证明过拟合或0.2必然有效。

| 项目 | 固定协议 | 核查位置 |
| --- | --- | --- |
| 唯一参数干预 | 分类头同一Dropout模块p=.2（forward调用两次）；encoder hidden/attention及encoder各Dropout保持.1 | main_model/dropout_protocol.py；train.py |
| 起点 | 每seed从原始XLM-R revision e73636d4f797dec63c3081bb6ed5c7b0bb3f2089重新开始；拒绝用.1轨迹续训成.2 | train_command、apply_classifier_dropout及训练签名 |
| 数据 | internal-v2固定train119842/dev15086；配置绑定train/dev、manifest、评估器SHA | dropout02_v2.json；local_round.preflight |
| 训练 | 512token，batch1/累积16，lr2e-5，无加权CE，AdamW/weight_decay .01/clip1，恒定lr，FP16/梯度检查点 | 配置、train.py、training_engine.py |
| seed与终点 | 必须42和43，不能seed43-only；各2000有效更新 | load_config拒绝单seed变体 |
| 节点 | 50资源门槛后恢复同轨迹；500/1000/1500/2000保存，500/1000/2000评估；1500不评估 | segment_steps与evaluation_steps |
| 对照 | 复用.1双seed六节点真实预测，逐项核对交接manifest SHA | comparison_baselines；preflight |
| 推理/评价 | FP32/batch8；p_machine≥.5；临时空白词数<100；固定.7C+.3B | 配置；dropout_results.generate |
| 恢复/保护 | 模型、优化器、scaler、RNG及游标完整恢复；绑定数据/配置/源码/Dropout签名；保护旧节点 | training_engine.signature与local_round |
| 资源 | run/resume至少50GiB空闲（25×2），建议60GiB；资源失败停止，不自动改参数 | local_round磁盘门槛和资源节点 |

train SHA：`5a5dca9db706e0faadc5c89cbcadc77dc585ce302f5e73b11b82213f40ab7ff6`；dev SHA：`1d80e7b6045e0914f224dcbc3afe5f56569f5bd995f742607389c26316476600`。C参考预测SHA：`031662988c4459b0cc30f1a8b45b89f396568b8b4b844789a88fc4362adfcabd`。

## 启动门槛与当前核查

1. B本地融合acceptance.json必须为PASS_B_LOCAL_PENDING_A，且bundle SHA为`5775334edda46069d45f7eaade2222bea2c7393d19c6faae44a3ed4a256ab50a`。当前满足。
2. A须实际在独立环境完成同包smoke/full-dev核验。用户已发送ZIP，仅更新发送事实；本机`.local/fusion_acceptance/A_receipt.json`尚不存在。真实回执到达后再记录reviewer=A、status=PASS、精确bundle SHA及真实GitHub evidence_url，不能将发送记录作为PASS。
3. 上述检查在run/resume创建输出目录及启动训练子进程前执行。回执JSON只校验字段与URL前缀，程序不会验证链接内容；仍须人工读取A证据并核对两份检查结果，不能依赖一份手填JSON自动证明验收。
4. 启动当天重新运行dry-run，确认数据、环境、源码及空间未变；本次CPU预检PASS不等于GPU资源门槛通过。按本次用户要求，即使后来收到A回执，本轮也不启动训练。

本次CPU dry-run通过：冻结数据、原模型文件、历史预测及依赖检查通过，缺失依赖/审计器均为空，可用空间约230.25GiB（当时快照）。7项test_fusion_dropout CPU回归全部通过，覆盖微型模型分类头修改与保存重载、固定双seed日程、缺B验收/缺A回执时拒绝启动、概率方向和包篡改检查。微型合成fixture仅为代码正确性检查，无真实XLM-R训练/GPU预测，不是性能实验。

## 比较与结果解释规则

对每个seed报告0.2减0.1的2000更新终点差，列出单B和固定融合的全体/临时短文本宏F1、临时S、human/machine召回、误报/漏报、p≥.99的人类误报、log loss、Brier和AUC。同时列出两seed终点均值及差值方向；两个seed不足以给出可靠seed分布。500/1000只用于预定诊断，不据此改选检查点或追加步数。

完整记录实际尝试、FP16跳步、处理行次和抽样轨迹；相同seed及有效更新预算不保证相同实际暴露，Dropout也会改变随机数消费与训练轨迹。不能将观测差异扩大成严格等曝光的机制证明。正则化比较仍复用同一dev，不能据此宣称未见生成器泛化。

当前协议没有数值化的自动换模阈值，也不自动替换候选。结果到齐后按双seed一致性、分类与概率质量取舍报告，再做联合冻结决定；若只有较好seed或中间节点改善，明确写证据不足。不得追加权重/阈值搜索，或用内部test/官方val筛选。完整覆盖与未见生成器实验仍是B06独立缺项，本轮Dropout对照不会自动补齐。

## 可复用入口与交付

这里只提供不训练的CPU预检命令；其余启动/恢复入口保留在[原运行说明](FUSION_DROPOUT_RUNBOOK_20261016.md)。指定独立dry-run报告路径，避免覆盖历史512准备报告。

```powershell
python -X utf8 main_model/local_round.py --config main_model/configs/dropout02_v2.json --output runs/b_v2_dropout02_round --phase dry-run --dry-run-report .local/b08_dropout_review_20261017/dry_run.json
python -m unittest discover -s main_model -p test_fusion_dropout.py -v
```

未来实测完成后复用regularization_comparison/single_and_fixed_fusion.csv及已有绘图/审计、精简分析包；人工图检另行完成，不重复训练来修图。当前不存在Dropout0.2实测结果，目标runs/b_v2_dropout02_round未创建。本轮未改训练代码、配置、依赖或原始实验产物；复核指纹见[b08_dropout_review_checks.json](b08_dropout_review_checks.json)。
