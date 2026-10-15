# 数据清洗与统计审计（internal-v1）

- 生成脚本：`data_tools/preprocess.py`
- 执行命令：`python data_tools/preprocess.py --data data.jsonl --out-dir data/processed/v1 --seed 42 --min-len 20`
- 原始数据：仓库根 `data.jsonl`（`raw-v1`，980438954 B，SHA-256 `a4d8d012…27d98`），**全程只读，未修改**
- 版本：`internal-v1`；权威机器可读清单见同目录 `manifest.json`

## 1. 规模与分组

| 指标 | 数值 |
| --- | --- |
| 有效记录数 | 95889 |
| 解析失败行数 | 0 |
| 唯一来源 ID 数 | 37179 |
| 唯一正文指纹数（人类） | 33773 |
| 因分组合并的记录对 | 62690 |
| 分组（group）总数 | 33199 |

## 2. 清洗统计

| 操作 | 数量 |
| --- | --- |
| 提示词模板剥离次数 | 219 |
| 过滤空/过短样本（human） | 7101 |
| 过滤空/过短样本（machine） | 93 |
| 人类去重（来源标识 + 正文） | 72136 |
| 机器去重（生成器 + 正文） | 14074 |

## 3. 各字段实际贡献的样本数

原始 JSONL 有 13 种记录形态、32 个字段并集。字段兼容优先级：人类 `human_text` > `text` > `abstract`；机器 `machine_abstract` > `machine_answer` > `machine_text`；评审类字段按 `^([A-Za-z0-9]+)_reviews$` 展开。

| 字段 | 贡献样本数 |
| --- | --- |
| `machine_text` | 80097 |
| `human_text` | 48707 |
| `machine_abstract` | 7416 |
| `text` | 4586 |
| `machine_answer` | 2804 |
| `bloom_reviews` | 2226 |
| `chatgpt_reviews` | 1272 |
| `davinci_reviews` | 1172 |
| `abstract` | 1156 |
| `human_reviews` | 459 |

## 4. 关键处理决策

- **提示词剥离**：仅当机器文本开头确实匹配对应提示词模板（`prompt`/`promt`，模板占位符用同记录字段填充）时才剥离，不做固定长度截断；全量真实命中 219 次。
- **人类去重**：判重条件为「来源标识 **且** 正文指纹都相同」，避免把同一论文的多条**不同**人类评审误删（如 PeerRead 的 `human_reviews` 保留 459 条）。缺失来源 ID 的样本以正文指纹兜底，不退化为「全部同一 ID」。
- **机器去重**：按「生成器名 + 正文指纹」判重，保留原始生成器名（`generator`），未做家族归一化。
- **分组**：并查集（DSU）先按 `(domain, 来源 ID)` 合并、再按人类正文指纹（小写 + 空白规范化）合并，使同一原文/主题及其机器回答落入同一组。来源 ID 优先级 `source_id/source_ID/question_id/answer_id/q_id/a_id/id`；**排除 `Unnamed: 0`**（pandas 索引泄漏列，跨形态会撞号）。
- **空文本**：一侧有效、一侧为空时保留有效侧；`min_text_len=20` 过滤空/过短文本。
- **原始保护**：`data.jsonl` 全程只读；本机副本 SHA-256 与 `data/manifest.raw.json` 记录完全一致，证明原始文件未被改动。

## 5. 划分结果（group-based，seed=42）

目标比例 80/10/10，按 `max(SPLITS, key=lambda k: target[k]-cur[k])` 贪心分配**组**，保证同组不拆。

| 集合 | 样本数 | 占比 | human | machine |
| --- | --- | --- | --- | --- |
| `train.jsonl` | 119727 | 79.87% | 43870 | 75857 |
| `dev.jsonl` | 14998 | 10.01% | 5444 | 9554 |
| `test.jsonl` | 15170 | 10.12% | 5594 | 9576 |
| 合计 | 149895 | 100% | 54908 | 94987 |

## 6. 完整性校验

- 分组跨集合泄露：**否**（train∩dev、train∩test、dev∩test 的 `group_id` 交集均为 0；149895 个 `sample_id` 全局唯一）。
- 原始数据是否被修改：**否**。
- 文本非空、`label ∈ {0,1}`、两类均有。

## 7. 已知限制

1. **短文本口径差异**：团队临时口径为「<100 词」，`data_tools/evaluate.py` 默认按「字符 <100」（`--short-by char`），需按官方核实结果切换并标注为临时口径。
2. **生成器未归一化**：`generator` 保留原始 `model` 名（`cohere` / `cohere-xlarge-nightly` / `davinci` / `text-davinci-003` 等仍分列），归因任务如需家族聚合需另行映射。
3. **domain 取自 `source`**，空串或缺失统一记为 `unknown`。
4. **类别不均衡**：machine 约占 63%，训练时需考虑类别加权/阈值调整。
5. **未生成真实小样**：本轮未产出 `smoke_train.jsonl` / `smoke_dev.jsonl`（A02 待补）。
6. **划分文件未入库**：`train/dev/test.jsonl`（约 352MB，走 Git LFS）本轮未上传，由接收方按 manifest 的 `generation.command` 本地生成并核对 SHA-256。

> 以上第 1、2、4、5 项与 `manifest.json` 的 `known_limitations` 保持一致。
