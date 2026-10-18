# A：数据、评估与交付验收

维护人：A。最后核对：2026-10-16。仅A更新本文件的状态、交接和工作记录；他人发现问题先反馈，不改写A的进度。

## 当前目标和前置条件

目标：让B/C使用同一套可信、可复现的真实训练数据与内部评估。

- 安装Git与Git LFS，拉取公开仓库代码和需要的数据；提交改动需获协作写权限，或Fork后提PR。原始数据可从已授权官方渠道获取并对照manifest校验。
- 已阅读 `TEAM_WORKPLAN.md` 的标签、数据接口、验证集限制和交接次序。
- 自己配置本机Python环境；只需基础数据处理依赖时使用 `requirements/base.txt`。不需要等待B训练或使用B的解释器路径。
- A01、A02 已完成；A03 脚本已交付、官方口径待核实；A04 进行中（B 的 dev 预测已复评，待 C）；A05 独立复现已完成（H05，融合包 dev 复现通过），最终官方推理待冻结。

## 任务清单

- [x] **A01 清洗与正文提取**：兼容实际字段和列表；展开人类/机器评审；区分论文摘要与评审正文；确认前缀匹配才剥离提示词；单侧为空不丢弃另一侧；记录过滤、重复和异常数量。保持原始文件不变。当前状态：完成。`data_tools/preprocess.py` 已全量跑通 95889 条记录，清洗与划分统计见 `data/processed/v2/manifest.json`（本机运行日志 `data/processed/v2/preprocess_log.txt` 不入库）：过滤过短 human 7101 / machine 93，提示词剥离 219，人类去重 72136，机器去重 14074，唯一正文指纹 130821；原始 `data.jsonl` SHA-256 与 `data/manifest.raw.json` 一致（未修改）。另按角色 C 的隐私要求，脚本已不再向控制台/日志输出 internal test 正文（仅保留 train/dev 示例），隐私修复后重跑，三个划分文件与原哈希逐字节一致。
- [x] **A02 固定划分与小样本**：按同原文/主题及重复正文关联分组，约80/10/10，seed=42；human=0、machine=1；给每个样本分配唯一字符串编号。交付完整train/dev/test及约1000/200真实小样，小样从对应集合抽取，两类均有，保持隔离。当前状态：完成（v2）。v1 因评审展开项跨 train/dev 正文重复未通过验收，已修复 `data_tools/preprocess.py`（所有将被导出的规范化正文全部绑定 group_id，写出时逐条实测交集）并重跑为 `internal-v2`：train 119842 / dev 15086 / test 14967，分组 33083，三集合两两 group_id 与规范化正文交集均为 0（独立脚本复核一致）；真实小样 `data/processed/v2/smoke_train.jsonl`（1000 条）/ `smoke_dev.jsonl`（200 条）已重新生成入库，详见 `data/processed/v2/manifest.json`。v1 已标记废弃。
- [ ] **A03 统一评估与官方口径**：交付全体宏F1、短文本宏F1、综合S及分项统计脚本；暂用0.5阈值，临时计词明确标注。核实官方阈值、计词、CSV编号/列名/概率方向、预训练资源规则及真实截止时间。当前状态：脚本 `data_tools/evaluate.py` 已交付（零 sklearn 依赖，输出全体/短文本宏F1、综合S、混淆矩阵、各类别召回率、短文本数量与占比，阈值默认 0.5，CSV 列名容错）。**团队统一短文本口径为 word 分词、<100 words**，脚本须显式传 `--short-by word --short-threshold 100`（默认 char 仅为本机回退，不是团队口径）。**官方阈值、官方计词方式、CSV 编号/列名/方向及真实截止时间仍待核实**；已按该口径完成 B 第二轮 dev 预测的复评（见 A04）。
- [ ] **A04 统一验收预测**：收到B/C同一数据版本预测后检查编号、顺序、数量与概率；使用统一脚本出结果表。完成报告的数据处理与评测章节。当前状态：进行中。已复评两批 dev 预测：① B 第二轮 256 长度（CE2000_s42 S=0.9047 / CE2000_s43 S=0.9077）；② B 512 长度两 seed 终点与固定融合（`handoffs/B/b_v2_length512_two_seed/`，L512_s42_u2000 S=0.9210 / L512_s43_u2000 S=0.9490 / 融合 C07_B03_L512_s43_u2000 S=0.9639，全部 Word 口径，7 个预测 CSV 格式与 manifest SHA 逐项核验通过，结果与 B 自评一致）。当前最优开发候选为 C .7 + B512 s43 .3 固定融合（S=0.9639）；融合与 C 基线的预测本体来自 C 分支交接，C 尚未单独 PR，待其合入后正式归档。
- [ ] **A05 独立复现与最终验收**：收到B冻结模型包后按说明在干净环境加载推理；问题回传B。方案冻结后执行官方最终推理并核验提交CSV；整理承诺书，检查匿名材料。当前状态：独立复现已完成（2026-10-17，H05）。对 `handoffs/shared_models/b_v2_fusion_s43_2000`（bundle.json SHA-256 `5775334edda46069d45f7eaade2222bea2c7393d19c6faae44a3ed4a256ab50a`，17/17 文件字节+哈希核验通过）在独立环境（Windows / Python 3.11 / torch 2.13.0+cpu / transformers 4.57.6 / scikit-learn 1.9.1，CPU FP32 batch8 length512）完成 smoke_dev（200 条）与全量 dev（15086 条）推理。数值核验：C 分量与融合预测均在容差内，B 分量最大绝对差 2.56e-5（CPU vs CUDA 浮点漂移），三个分量 0.5 阈值判定与参考 100% 一致；Word/.5/100 口径评估：全体 F1=0.9685、短文本 F1=0.9534、S=0.9639，与 B 自评（0.963939）一致。产物：`runs/a_fusion_smoke*`、`runs/a_fusion_dev*`（含 prediction_manifest/stages、check JSON 与 `a_evaluation_word.txt`，本机不入库）。全程仅使用 v2 smoke_dev/dev，未接触内部 test 与官方 val。官方最终推理与承诺书待 H03 冻结后进行。

## A → B/C 的首轮交付

推荐位置 `data/processed/v2/`；如已存在其他约定路径，先同步共享接口，不搬动他人文件。

```text
train.jsonl / dev.jsonl / test.jsonl
smoke_train.jsonl / smoke_dev.jsonl
manifest.json / cleaning_audit.md
```

六字段：`sample_id,group_id,text,label,generator,domain`。评估代码放自己的代码目录（建议 `data_tools/`），附明确调用命令。manifest列出版本、文件SHA-256、样本/类别数、来源、划分规则和seed，全部使用相对路径。

验收：文本非空、标签合法、编号唯一、两类均有、同原文/主题不跨集合、空白规范化重复正文不跨集合；B入口会检查sample_id/group_id/正文重叠，不关闭检查来绕过问题。

合并到main并推送LFS后通知接收方数据版本、commit、目录和命令；B/C先验收接口，再分别训练。

## 上传边界

上传：清洗/评估代码、冻结数据及manifest、必要审计摘要、报告章节、本任务清单。

不上传：本机Python/Conda环境、缓存、IDE设置、机器路径配置、访问凭据、临时中间数据、完整个人运行目录。交接数据放正式版本目录，临时文件放 `.local/` 或 `runs/`。不要用 `git add .` 代替逐项检查。

## 数据与口径状态（供 C 撰写报告引用）

- **数据冻结版本**：internal-v2（PR #3 已合入 main），划分与哈希见 `data/processed/v2/manifest.json`。
- **统一评估口径**：固定 **Word/.5/100**（word 分词、阈值 0.5、短文本 <100 words），`evaluate.py` 须显式传 `--threshold 0.5 --short-by word --short-threshold 100`。
- **未决口径（待官方最终确认）**：多语言计词细节、概率转类别阈值、官方提交 CSV 模板/列名/概率方向、真实截止时间。
- **H05 证据归档**：`handoffs/A/h05_numerical_check/`（复算脚本 `recheck.py`、smoke/dev 检查 JSON、C/B/融合轻量预测 CSV（仅 sample_id,p_machine）、含全部 SHA-256 的 `manifest.json` 与 README）；本目录即为 C 报告引用的评估证据路径。

## Issue 与分支交付规范（队长 2026-10-18 更新）

- **Issue 使用**：每个版本共用一个主 Issue；较长的独立任务再拆子 Issue，结论汇总回主 Issue（例如 V1 数值闭环、D42/7500 校准包验收各在主 Issue 下汇总）。
- **交付路径**：本人记录只维护在本文档；代码与证据一律通过本人分支 `work/a-data` 提交，经 PR 合入 main 后才算全队收到。

## 下一步与交接记录

下一步：核实官方短文本计词、阈值与预测 CSV 口径（A03）；等待 C 的基线与融合预测后按 A04 统一复评；H03 冻结后执行官方最终推理与承诺书整理（A05 剩余部分）；H05 数值差异（full-dev B 分量原判据超限 274 条、融合 4 条，0.5 决策差异均 0）保持「待共同闭环」，不强制标记 PASS。

| 日期 | 任务编号 | 状态/产物与commit | 验收结果 | 已交给谁/下一步 |
| --- | --- | --- | --- | --- |
| 2026-10-15 | A01—A03 | 交付 `data_tools/preprocess.py`、`data_tools/evaluate.py`、`data_tools/make_smoke.py`、`data/processed/v2/manifest.json`、smoke 小样、`handoffs/A/v2/README.md`（分支 `work/a-data`，已并入 main）；完整划分未入库，接收方按 manifest 命令本地生成并核对 SHA-256 | H01 已发出，待B/C验收 | 通知B/C一律使用 internal-v2 |
| 2026-10-16 | A01 | 修复 `data_tools/preprocess.py` 的 internal test 正文日志/控制台泄漏（角色C 隐私要求）：仅保留 train/dev 示例；修复后重跑 v2，三划分文件逐字节不变 | 自检通过（日志无 test 正文、哈希一致） | 随本次提交推送 |
| 2026-10-16 | A03/A04 | 按团队统一 Word 口径（`--short-by word --short-threshold 100`）复评 B 第二轮 dev 预测 CE2000_s42（S=0.9047）与 CE2000_s43（S=0.9077）；完成格式校验（15086 行、表头、编号顺序、概率范围） | 与 B 自评逐项一致 | 待 C 基线与融合预测 |
| 2026-10-16 | A04 | PR#4 合入 main 后复评 B 512 两 seed 交付：7 个预测 CSV 全部通过格式与 manifest SHA 核验；Word 口径下 L512_s42_u2000 S=0.9210、L512_s43_u2000 S=0.9490、固定融合 C07_B03_L512_s43_u2000 S=0.9639（当前最优候选）；与 B 自评一致。全队冻结前不运行内部 test 或官方验证集 | 通过 | 回执已出；待 C 分支 PR 合入与 H03 冻结 |
| 2026-10-17 | A05 | H05 独立复现完成并按队长要求归档：`handoffs/A/h05_numerical_check/`（复算脚本、smoke/dev 检查 JSON、C/B/融合轻量预测 CSV 仅两列、含 SHA-256 的 manifest）；原判据复算 full-dev B 超限 274 条、融合 4 条、决策差异 0；状态字段来源已澄清为 A 侧归纳标注 | 数值差异待与 B 共同闭环，不标 PASS | 证据 PR 已发；internal test 继续封存 |
| 2026-10-18 | A05 | V1 数值闭环收尾：换行符统一 LF（哈希与队出口径一致）、recheck.py 改读交付 CSV 并加严格异常断言、CPU/CUDA 表述改「可能原因，尚未证实」；V1 原包 CPU 重复推理与存档逐字节一致（B 4 条超限/决策差 0，形态稳定）。D42/7500 校准包：ZIP/bundle/calibrator 哈希核验通过，smoke accept 原判据失败即停（raw 超限 1/200、calibrated 5/200、决策差 0），full-dev 未执行，现场证据保留于 `runs/a_d42_calibrated_*`，证据回执已发主 Issue | V1 差异待共同闭环；D42 smoke 验收未通过待 B 指示 | 回执已发 issue #14；后续按「每版本一个主 Issue」汇总 |
