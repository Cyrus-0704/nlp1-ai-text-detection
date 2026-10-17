# 第一版模型冻结记录草案

日期：2026-10-17。状态：`DRAFT_PENDING_ACCEPTANCE_REVIEW_AND_TEAM_FREEZE`。本文件用于A/C审阅；不是最终冻结声明，也不授权训练、内部test评估或官方val推理。

## 本版候选与交付范围

沿用同一internal-v2数据和既有权重：XLM-R 512 token、seed43、2000有效更新、分类头Dropout0.1；与C字符TF-IDF/逻辑回归按概率`0.7*p_C+0.3*p_B`融合。内部human=0/machine=1、阈值0.5、临时空白计词<100；输出保留machine概率。FP32、batch8、输入原顺序。候选选择依据既有dev比较，不查看内部test再选择。

本版交付离线模型、预测及核验入口、依赖/运行说明、来源与指纹、真实评估记录和技术报告。已有有限预算结果可以形成工程交付；训练覆盖不足、seed波动、同dev选择偏差和未见生成器证据缺失必须保留。Dropout0.2、调度、更长训练及更多融合方案留后续独立版本，不自动启动。

## 固定身份与证据

| 项目 | 固定值/证据 |
| --- | --- |
| 数据 | internal-v2；train119842/dev15086，保持A划分 |
| train SHA-256 | `5a5dca9db706e0faadc5c89cbcadc77dc585ce302f5e73b11b82213f40ab7ff6` |
| dev SHA-256 | `1d80e7b6045e0914f224dcbc3afe5f56569f5bd995f742607389c26316476600` |
| 原始预训练来源 | FacebookAI/xlm-roberta-base；revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` |
| 已交A的bundle.json SHA-256 | `5775334edda46069d45f7eaade2222bea2c7393d19c6faae44a3ed4a256ab50a` |
| B model.safetensors SHA-256 | `47c9deae762ee43c7700045538bb9b9a388fc5b9c1b3e015db25f6874128e48e` |
| C joblib SHA-256 | `6d231819e61d24b2f9b8ee91b07b72f6e7c0f8c26d93c500d419d791211c46f8` |
| 发送ZIP SHA-256 | `f5b34bb2d87e884ce876152875f0692c6e8f8eb03de26de9c50af696886dd687`；用户确认已发送A |
| 包内各文件 | [模型清单](b_fusion_local_acceptance/model_inventory.json)；该文件为LF副本，其文件SHA不等于原bundle.json SHA |
| 已交付推理代码 | B提交6a8ce50及13774ad，经PR #9进入main e2a9544；精确包内源码以原bundle清单SHA为准 |
| 原训练代码 | [seed43训练协议](b_v2_length512_two_seed/seed43_protocol.json)；启动4ca7c85含未提交源码，不能仅凭Git commit声称重训可精确复现 |
| dev结果 | 全体宏F1 .968463、临时短文本 .953381、临时S .963939；[B08正文与源表](B08_REPORT_MATERIALS_FOR_C_20261017.md) |

本轮新增check_inference_io.py为包外标准库辅助工具，不覆盖已发A的17文件包、bundle.json或ZIP。不修改参考容差、模型权重、分词器或推理公式；最终发行清单应另收录辅助脚本及文档并保留原bundle身份。当前辅助文件指纹与验证结果见[本轮检查记录](release_preparation_checks_20261017.json)。

## A独立复现：已收到文字回执，数值证据待复核

远端[PR #10](https://github.com/Cyrus-0704/nlp1-ai-text-detection/pull/10)，head `90b6b291968fbafdd0ae48e44a3bee52a2701d8e`，本轮查看时OPEN。A声明17/17包文件校验通过，在Windows/Python3.11、torch2.13.0+cpu、Transformers4.57.6、scikit-learn1.9.1下完成smoke200和dev15086推理；C/融合在容差内，三分量0.5分类决策一致，dev分数与B一致。

需要关闭的一项差异：A报告B概率最大绝对差2.56e-5，原脚本要求`abs(actual-reference) <= 1e-6 + 1e-5*abs(reference)`，参考概率在[0,1]时允许误差最高1.1e-5。因此该最大差不能直接记为通过原B逐点allclose门槛。CPU/GPU浮点差异是A给出的解释，本轮没有原始预测/检查JSON可独立确认其分布或原因。

A的PR目前仅修改个人清单，正文与评论未附两份机器核验报告。请补充smoke/dev检查JSON、prediction_manifest、所用核验脚本SHA/命令和环境/警告；说明是否使用不同容差，以及2.56e-5对应的分量和检查范围。B对实际文件及预定容差复核后再关闭验收项；不因分类相同自动放宽容差、不伪造本地PASS回执、不要求无理由重跑全部推理。PR合并和数值问题解决分别记录。

## 冻结前必须确认

- [ ] A上述回执证据与容差差异核对完成；真实问题由B修复并重验受影响部分。
- [ ] A确认数据版本、清洗依据/影响及报告中的评测口径；未说明的官方细则继续明确列出，不自动重划数据。
- [ ] C接收B08并核对模型/基线/融合章节、来源与资源声明；既有H04和固定融合复核无需重复训练。
- [ ] 全队确认当前候选、数据、融合及内部口径并记录冻结日期/版本；本文件保持草案直至有真实确认。
- [ ] 冻结后由A做一次内部test评估；结果用于验收与报告，不回流反复选参。当前不执行。
- [ ] C统稿及匿名检查通过，模型来源/许可证与竞赛外部资源要求核验完毕；B最终发行包与验收版本一致。

给老师的离线交付不需要提前读取官方val。比赛最终提交另待准确CSV模板、编号/概率方向和真实截止日期；冻结后由A执行官方推理及格式验收，并整理result.csv、technical_report.pdf和承诺书。

## C接收本PR后的动作

从B08材料整合方法、实测结果与局限，核对本文冻结草案；将报告中的“test评估后决定最终方案”改为“冻结后一次验收”，将已完成的C跨环境核验按B现有证据更新。A的CPU容差差异关闭前，报告标注“已收到独立复现声明，机器核验证据待复核”。回复接收的B提交、合稿位置及待解决问题。B不代改C的报告或个人清单，本PR发布不等同C已验收、main已收到或全队已冻结。
