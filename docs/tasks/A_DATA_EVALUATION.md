# A：数据、评估与交付验收

维护人：A。最后核对：2026-10-15。仅A更新本文件的状态、交接和工作记录；他人发现问题先反馈，不改写A的进度。当前未收到A的实际交付反馈，未勾选表示“待确认/未验收”，不表示A没有开展工作。

## 当前目标和前置条件

目标：让B/C使用同一套可信、可复现的真实训练数据与内部评估。

- 安装Git与Git LFS，拉取公开仓库代码和需要的数据；提交改动需获协作写权限，或Fork后提PR。原始数据可从已授权官方渠道获取并对照manifest校验。
- 已阅读 `TEAM_WORKPLAN.md` 的标签、数据接口、验证集限制和交接次序。
- 自己配置本机Python环境；只需基础数据处理依赖时使用 `requirements/base.txt`。不需要等待B训练或使用B的解释器路径。
- A01—A03可立即推进；A04等待B/C预测；A05等待冻结模型包。

## 任务清单

- [x] **A01 清洗与正文提取**：兼容实际字段和列表；展开人类/机器评审；区分论文摘要与评审正文；确认前缀匹配才剥离提示词；单侧为空不丢弃另一侧；记录过滤、重复和异常数量。保持原始文件不变。当前状态：完成。`data_tools/preprocess.py` 已全量跑通 95889 条记录，统计见 `data/processed/v1/cleaning_audit.md`（过滤 human 7101 / machine 93，提示词剥离 219，人类去重 72136，机器去重 14074），原始 `data.jsonl` SHA-256 与 `data/manifest.raw.json` 一致（未修改）。
- [x] **A02 固定划分与小样本**：按同原文/主题及重复正文关联分组，约80/10/10，seed=42；human=0、machine=1；给每个样本分配唯一字符串编号。交付完整train/dev/test及约1000/200真实小样，小样从对应集合抽取，两类均有，保持隔离。当前状态：完成。完整划分已产出（train 119727 / dev 14998 / test 15170，分组 33199，跨集合 group_id 交集为 0）；真实小样 `data/processed/v1/smoke_train.jsonl`（200 条，human 73 / machine 127）、`smoke_dev.jsonl`（100 条，human 36 / machine 64）由 `data_tools/make_smoke.py` 分层抽取（seed=42，两类均有，为正式集子集），已随 v1 入库，详见 `data/processed/v1/manifest.json` 的 `smoke` 节。
- [ ] **A03 统一评估与官方口径**：交付全体宏F1、短文本宏F1、综合S及分项统计脚本；暂用0.5阈值，临时计词明确标注。核实官方阈值、计词、CSV编号/列名/概率方向、预训练资源规则及真实截止时间。当前状态：脚本 `data_tools/evaluate.py` 已交付（零 sklearn 依赖，输出全体/短文本宏F1、综合S、混淆矩阵、各类别召回率、短文本数量与占比，阈值默认 0.5，CSV 列名容错）；**官方阈值、计词口径、CSV 编号/列名/方向仍待核实**，当前短文本按字符<100 为临时口径。
- [ ] **A04 统一验收预测**：收到B/C同一数据版本预测后检查编号、顺序、数量与概率；使用统一脚本出结果表。完成报告的数据处理与评测章节。
- [ ] **A05 独立复现与最终验收**：收到B冻结模型包后按说明在干净环境加载推理；问题回传B。方案冻结后执行官方最终推理并核验提交CSV；整理承诺书，检查匿名材料。

## A → B/C 的首轮交付

推荐位置 `data/processed/v1/`；如已存在其他约定路径，先同步共享接口，不搬动他人文件。

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

## 下一步与交接记录

下一步：提交 smoke 小样与更新后的交接文档并推送到 `work/a-data`，向 main 提 PR；核实官方短文本计词、阈值与预测 CSV 口径（A03）；PR 合并后向 B/C 发出 H01 验收通知（数据版本 internal-v1）。

| 日期 | 任务编号 | 状态/产物与commit | 验收结果 | 已交给谁/下一步 |
| --- | --- | --- | --- | --- |
| 2026-10-15 | A01—A03 | 交付 `data_tools/preprocess.py`、`data_tools/evaluate.py`、`data/processed/v1/manifest.json`、`data/processed/v1/cleaning_audit.md`、`handoffs/A/v1/README.md`（分支 `work/a-data`，首个交付 PR 已提）；完整划分未入库，接收方可本地生成 | 待B/C验收 | PR 合并后通知B/C走H01 |
| 2026-10-15 | A02 | 补交 `data_tools/make_smoke.py` 与真实小样 `data/processed/v1/smoke_train.jsonl`（200条）/ `smoke_dev.jsonl`（100条），已校验为正式集子集且两类均有 | 待B/C验收 | 随本次提交推送后并入同一 PR 或新 PR |
