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
- **数据使用/再分发条款**：当前仅有「原始数据从已授权官方渠道获取并对照 manifest 校验」的既有口径，**暂无赛题方关于数据使用/再分发的最终明确授权依据**——**待赛题方或老师最终确认（作为发行待办项，不阻塞当前模型方案冻结）**。
- **H05 证据归档**：`handoffs/A/h05_numerical_check/`（复算脚本 `recheck.py`、smoke/dev 检查 JSON、C/B/融合轻量预测 CSV（仅 sample_id,p_machine）、含全部 SHA-256 的 `manifest.json` 与 README）；本目录即为 C 报告引用的评估证据路径。

## Issue 与分支交付规范（队长 2026-10-18 更新）

- **Issue 使用**：每个版本共用一个主 Issue；较长的独立任务再拆子 Issue，结论汇总回主 Issue（例如 V1 数值闭环、D42/7500 校准包验收各在主 Issue 下汇总）。
- **交付路径**：本人记录只维护在本文档；代码与证据一律通过本人分支 `work/a-data` 提交，经 PR 合入 main 后才算全队收到。

## 下一步与交接记录

下一步：核实官方短文本计词、阈值与预测 CSV 口径（A03）；等待 C 的基线与融合预测后按 A04 统一复评；H03 冻结后执行官方最终推理与承诺书整理（A05 剩余部分）；**V1 已正式冻结，冻结标识 V1-20261019（PR #16 已合并）**，A 侧清单状态为「CUDA原容差验收通过，CPU历史超限保留」；**知悉 B 侧已收录历史 test 证据（existing_V1_test_evidence_20261019，宏 F1=0.9681 等），A 侧不重复运行 test**；PR #12 剩余收尾（manifest README 条目修正、再分发条款缺口说明）已提交待合入；D42/7500 校准包维持「smoke 验收未通过待 B 指示」独立状态不变。

| 日期 | 任务编号 | 状态/产物与commit | 验收结果 | 已交给谁/下一步 |
| --- | --- | --- | --- | --- |
| 2026-10-15 | A01—A03 | 交付 `data_tools/preprocess.py`、`data_tools/evaluate.py`、`data_tools/make_smoke.py`、`data/processed/v2/manifest.json`、smoke 小样、`handoffs/A/v2/README.md`（分支 `work/a-data`，已并入 main）；完整划分未入库，接收方按 manifest 命令本地生成并核对 SHA-256 | H01 已发出，待B/C验收 | 通知B/C一律使用 internal-v2 |
| 2026-10-16 | A01 | 修复 `data_tools/preprocess.py` 的 internal test 正文日志/控制台泄漏（角色C 隐私要求）：仅保留 train/dev 示例；修复后重跑 v2，三划分文件逐字节不变 | 自检通过（日志无 test 正文、哈希一致） | 随本次提交推送 |
| 2026-10-16 | A03/A04 | 按团队统一 Word 口径（`--short-by word --short-threshold 100`）复评 B 第二轮 dev 预测 CE2000_s42（S=0.9047）与 CE2000_s43（S=0.9077）；完成格式校验（15086 行、表头、编号顺序、概率范围） | 与 B 自评逐项一致 | 待 C 基线与融合预测 |
| 2026-10-16 | A04 | PR#4 合入 main 后复评 B 512 两 seed 交付：7 个预测 CSV 全部通过格式与 manifest SHA 核验；Word 口径下 L512_s42_u2000 S=0.9210、L512_s43_u2000 S=0.9490、固定融合 C07_B03_L512_s43_u2000 S=0.9639（当前最优候选）；与 B 自评一致。全队冻结前不运行内部 test 或官方验证集 | 通过 | 回执已出；待 C 分支 PR 合入与 H03 冻结 |
| 2026-10-17 | A05 | H05 独立复现完成并按队长要求归档：`handoffs/A/h05_numerical_check/`（复算脚本、smoke/dev 检查 JSON、C/B/融合轻量预测 CSV 仅两列、含 SHA-256 的 manifest）；原判据复算 full-dev B 超限 274 条、融合 4 条、决策差异 0；状态字段来源已澄清为 A 侧归纳标注 | 数值差异待与 B 共同闭环，不标 PASS | 证据 PR 已发；internal test 继续封存 |
| 2026-10-18 | A05 | V1 数值闭环收尾：换行符统一 LF（哈希与队出口径一致）、recheck.py 改读交付 CSV 并加严格异常断言、CPU/CUDA 表述改「可能原因，尚未证实」；V1 原包 CPU 重复推理与存档逐字节一致（B 4 条超限/决策差 0，形态稳定）。D42/7500 校准包：ZIP/bundle/calibrator 哈希核验通过，smoke accept 原判据失败即停（raw 超限 1/200、calibrated 5/200、决策差 0），full-dev 未执行，现场证据保留于 `runs/a_d42_calibrated_*`，证据回执已发主 Issue | V1 差异待共同闭环；D42 smoke 验收未通过待 B 指示 | 回执已发 issue #14；后续按「每版本一个主 Issue」汇总 |
| 2026-10-19 | A05 | PR #15 V1 诊断包离线分析：LFS 载荷 39600 字节/SHA-256 校验通过，解压至独立目录 `runs/v1_diag_analysis_20261019/`，33 项成员 CRC+SHA-256 全部通过；64 条多环境对齐复算（6 模式 B、0.7C+0.3B 融合、决策差异）与 B `summary.json` 逐项一致；环境三分类比对与逐层差异分析完成（本地无独立 CUDA 环境） | 离线诊断，不改判 PASS；差异待 B 综合复核 | 离线分析回执待发主 Issue |
| 2026-10-19 | A05 | D42 smoke 物理证据补交（B 指出此前仅有哈希/路径）：归档 `raw.csv`、`predictions.csv`、`prediction_receipt.json`、`acceptance.json`、`verify_bundle.log`、`accept_smoke.log` 及新生成 `evidence_manifest.json` 至 `handoffs/A/d42_calibrated_smoke_evidence/`（原字节复制，复制后逐文件 SHA 复核一致）；打包 `d42_smoke_evidence_for_B.zip`（10217 B，SHA-256 `2f58f53cc0ed4b3521f504eb0b89a91ebc0abfd2aecabebfeaa2e92925bc37e3`） | 仅补交物理文件，不重跑推理、不改 D42 流程与容差 | 分支 work/a-data 推送，待 B 独立复算 |
| 2026-10-19 | A05 | V1 CUDA 证据补交（B 要求补物理文件）：归档 smoke 与 dev 两套 CUDA 运行证据（各含 C/B/predictions CSV、prediction_manifest、stages、verify_check、cuda_vs_cpu_compare）至 `handoffs/A/v1_cuda_evidence/`，原字节复制并逐文件 SHA 复核；打包 `v1_cuda_evidence_for_B.zip`（540557 B，SHA-256 `ca44d8ddc855288c6ab5e95244c756d6c667d1ee68afe75fe61000d61d521aa1`，15 成员）。运行命令/环境日志从未落盘，清单中如实登记为「缺失」 | 暂存字节保真核验 ALL_BYTES_FAITHFUL；未重跑推理、未训练、未触碰 D42 | 分支 work/a-data 推送，含两处文字澄清（torch 版本检测时点、逐位全等适用范围） |
| 2026-10-19 | A05 | V1 CUDA 复跑（用户指示：确认 CUDA 不可用根因为 torch CPU 版后换装 GPU 版重跑 V1 issue #14 数据）：根因证实为 `torch 2.14.0+cpu`；新建 Python 3.11 独立环境 `zyy-gpu`（对齐 B 验证环境与包 requirements：torch 2.13.0+cu130、transformers 4.57.6、scikit-learn 1.9.1、numpy 2.4.6 等，RTX 4060 sm_89），CUDA 驱动验证通过；重跑 V1 smoke（200）与全量 dev（15086）输出 `runs/v1_cuda_check_20261019/`、`runs/v1_cuda_dev_check_20261019/`（attempt-001 环境失败现场保留） | smoke：对 B 包内参考 PASS_REFERENCE_CHECK（B max 4.17e-7 < atol 1e-6），决策差 0；dev：对 B 参考逐位全等（B diff=0.0、融合 3.5e-11），对 A 本地 CPU 的差异模式与 H05 历史完全吻合（B 超限 274、融合 4、决策差 0）；C 分量两数据集均与 CPU 逐字节一致。设备相关浮点差异假设获本机复现证据，最终归因待 B 确认参考生成环境 | 原容差/口径未改，未读内部 test；回执待发 issue #14 |
| 2026-10-19 | A05 | **V1 候选冻结最终确认（B 最终确认要求，仅更新清单与状态，零推理/零训练/零新文件）**：①A侧同意冻结此候选——Bundle SHA `5775334edda46069d45f7eaade2222bea2c7393d19c6faae44a3ed4a256ab50a`（B512/seed43/2000/Dropout0.1，固定 0.7C+0.3B，machine=1，阈值 0.5，临时空白计词 <100）；internal-v2 原 train/dev SHA 不变（本日实测 train `5a5dca9db706e0faadc5c89cbcadc77dc585ce302f5e73b11b82213f40ab7ff6`、dev `1d80e7b6045e0914f224dcbc3afe5f56569f5bd995f742607389c26316476600` 与 `data/processed/v2/manifest.json` 完全一致，划分依据 manifest、PR #3 合入），沿用既有清洗与隔离依据，无重划数据、无新审计实验；②数据与评测口径沿用 internal-v2 + Word/.5/100；③内部 test（`data/processed/v2/test.jsonl`）**A侧未执行**（知悉 B 侧历史窗口已保存 `runs/b_internal_test_joint/attempt-001/` 记录，该记录先于本次全队冻结，不作本次新验收替代、不与本次冻结混同），A 侧严格封存不提前运行；④当前状态统一为「**CUDA原容差验收通过，CPU历史超限保留**」——CUDA 复跑（smoke PASS_REFERENCE_CHECK、dev 对 B 参考逐位一致）见 commit `46aec69` 与 `handoffs/A/v1_cuda_evidence/`，CPU 历史超限记录（full-dev B 274 条、融合 4 条，决策差 0）原样保留不作改写；⑤**隔离声明：PR #12 中涉及的 D42 材料与本次 V1 冻结结论严格隔离，D42 维持「smoke 验收未通过待 B 指示」独立状态，两个版本不一起改判** | 冻结确认完成；本行所在提交即此次确认对应 commit | 回执发主 Issue；后续按 H03 冻结流程推进 |
| 2026-10-19 | A05 | **PR #12 最终归档收尾（B 通报 V1 正式冻结后）**：①manifest.json README 条目最终修正完成并实测核验——2317 B/SHA `a5597e778202dd834f1cf85a24083ba7b98e681e2c5b2cfa0b668641b833f028`，与仓库 index LF blob 逐项一致，JSON 纯 LF（上一提交 `5970d32` 已完成，本提交复核确认）；②数据使用/再分发条款按 B 最终口径正式补齐：「暂无赛题方最终明确授权依据，待赛题方或老师最终确认（作为发行待办项，不阻塞当前模型方案冻结）」；③**知悉三方候选正式冻结标识 V1-20261019（PR #16 已合并）**；④**知悉 B 侧已收录历史 test 证据（existing_V1_test_evidence_20261019，宏 F1=0.9681 等），A 侧不重复运行 test**，A 侧状态维持「A侧未执行」；⑤D42 材料与 V1 冻结结论维持严格隔离 | 零推理/零训练/零触碰 D42/零新文件 | 本行所在提交即收尾对应 commit；等待 PR #12 后续合入 |
| 2026-10-20 | A05 | **D42/7500 + 固定校准包 CUDA 真实模型重载验收（B 正式授权，范围 smoke→完整 dev）**：新建日志目录 `runs/a_d42_calibrated_cuda_logs/attempt-001/`，按 B 指令分阶段执行且任何一步非零退出即停止——①verify-bundle exit=0（`PASS_PACKAGE_HASHES_AND_REFERENCE_FORMULA_ONLY`）；②smoke CUDA 预测（200 行，7.4s，device=cuda，threads=4/deterministic/sdpa/fp32/512/batch8）+ accept exit=0 → **PASS_MODEL_RELOAD_NUMERICS**（raw max 2.68e-6、calibrated 3.45e-7、formula 0.0，三组 numeric_failures=0）；③完整 dev CUDA 预测（15086 行，286.5s）+ accept exit=0 → **PASS_MODEL_RELOAD_NUMERICS**（raw/calibrated/formula 三组 max_abs_error **全为 0.0 逐位全等**，numeric_failures=0）。环境：py3.11.17 / torch 2.13.0+cu130 / CUDA 13.0 / transformers 4.57.6 / numpy 2.4.6 / RTX 4060 Laptop | 三阶段全部 exit=0，未放宽容差、未改任何参数、未重跑训练；未触碰内部 test | 证据归档 `handoffs/A/d42_calibrated_cuda_evidence/` + ZIP `d42_cuda_evidence_for_B.zip`（305300 B，SHA `6d6505d8e47df9b0fff3223777383a8449082a26b6124611089d7fa2cade0aaf`，15 成员，ALL_BYTES_FAITHFUL）；分支推送待 B 复算 |
