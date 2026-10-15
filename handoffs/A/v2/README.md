# A → B/C 交接说明（数据版本 internal-v2 / H01）

> **取代 internal-v1**：v1 验收发现 5 条人类正文（PeerRead/iclr_2017 评审展开项）跨 train/dev 重复。根因是分组时每条记录仅注册一条人类代表正文指纹，多文本展开项未全部绑定 group。v2 已将所有将被导出的规范化正文全部绑定入组，并在写出时逐条实测：三集合两两规范化正文指纹交集为 0。请一律使用 v2，停止使用 v1 任何划分与小样。

## 1. 基本信息

| 项 | 内容 |
| --- | --- |
| 交付人 | A（数据、评估与交付验收） |
| 用途 | H01：向 B/C 提供同一套可信、可复现的内部训练/调参/内部测试数据、真实小样与统一评估口径 |
| 接收人 | B（主模型）、C（基线/融合） |
| 数据版本 | `internal-v2`（取代 `internal-v1`） |
| 分支 | `work/a-data` |
| 原始数据 | 仓库根 `data.jsonl`（Git LFS，`raw-v1`，不可变，见 `data/manifest.raw.json`） |

完整划分（train/dev/test 合计约 352MB，走 Git LFS）本轮**不入库**，接收方按第 4 节命令本地生成并用 SHA-256 核对；**smoke 真实小样已入库**，可直接拉取使用。

## 2. 包含文件

正式交付物（已入库）：

| 相对路径 | 说明 | 样本量 | SHA-256 |
| --- | --- | --- | --- |
| `data/processed/v2/smoke_train.jsonl` | 训练小样（抽自 train） | 1000（human 366 / machine 634） | `42a99294125e06edfb6fd2792c0f33ab2363677e5392620e6f189206b96df94e` |
| `data/processed/v2/smoke_dev.jsonl` | 调参小样（抽自 dev） | 200（human 73 / machine 127） | `218b56ed0f19e59351853c13e2f11f1598c172decbc97b60fa1682e6e9d2883c` |
| `data/processed/v2/manifest.json` | 权威机器可读清单（版本/字段/分组/划分/评估/限制） | — | 见 git |
| `data_tools/preprocess.py` | 生成 internal-v2 的处理与划分脚本（含组隔离修复与实测核验） | — | 见 git |
| `data_tools/make_smoke.py` | smoke 小样分层抽样脚本 | — | 见 git |
| `data_tools/evaluate.py` | 统一评估脚本（零 sklearn 依赖） | — | 见 git |

完整划分（**未入库**，接收方本地生成后核对）：

| 相对路径 | 样本量 | human | machine | SHA-256 |
| --- | --- | --- | --- | --- |
| `data/processed/v2/train.jsonl` | 119842 | 43922 | 75920 | `5a5dca9db706e0faadc5c89cbcadc77dc585ce302f5e73b11b82213f40ab7ff6` |
| `data/processed/v2/dev.jsonl` | 15086 | 5495 | 9591 | `1d80e7b6045e0914f224dcbc3afe5f56569f5bd995f742607389c26316476600` |
| `data/processed/v2/test.jsonl` | 14967 | 5491 | 9476 | `49a0525da0f6a918f5bea2d722339f0eac13182bca581a6a84a0e234e8967378` |

合计 149895 样本，33083 个分组；`group_id` 跨集合交集为 0，规范化正文跨集合交集为 0（实测）。

## 3. 数据集规范（务必遵守）

- **六字段**：每行一个 JSON 对象，字段严格为 `sample_id, group_id, text, label, generator, domain`。
- **label 含义**：`human = 0`，`machine = 1`；人类 `generator` 统一为 `human`，机器保留原始模型名。
- **预测 CSV 格式**：必须包含 `sample_id` 和 `p_machine` 两列；`p_machine` 为 **machine（label=1）方向的预测概率**，取值 `[0,1]`，每个编号一条，顺序与输入一致。`evaluate.py` 对概率列名有容错（`p_machine/prob/score` 等均可），但请优先使用 `p_machine`。
- **评测阈值**：默认 `p_machine >= 0.5` 判为 machine（可用 `--threshold` 调整）。
- **短文本定义**：团队临时口径为「文本长度 **< 100 词**」（空白分词）；当前 `evaluate.py` 默认按「**字符数 < 100**」（`--short-by char --short-threshold 100`）实现，属临时口径，待官方核实后用 `--short-by word` 切换并在报告中标注。
- **综合得分**：`S = 0.7 × 全体MacroF1 + 0.3 × 短文本MacroF1`；脚本同时输出混淆矩阵、各类别召回率、短文本数量与占比。

## 4. 可复制命令

生成完整划分（需先拉取原始数据）：

```bash
git lfs pull --include="data.jsonl"
python data_tools/preprocess.py --data data.jsonl --out-dir data/processed/v2 --seed 42 --min-len 20
```

复现 smoke 小样（可选，已入库）：

```bash
python data_tools/make_smoke.py --train data/processed/v2/train.jsonl --dev data/processed/v2/dev.jsonl --out-dir data/processed/v2 --n-train 1000 --n-dev 200 --seed 42
```

本地评测（以 smoke_dev 为例，test 同理换成 `--gold data/processed/v2/test.jsonl`）：

```bash
python data_tools/evaluate.py --gold data/processed/v2/smoke_dev.jsonl --pred predictions.csv --threshold 0.5 --save-report report.txt
```

## 5. 验收检查

1. 生成/拉取的文件与 `manifest.json` 中 SHA-256、bytes、样本数一致。
2. 文本非空、`label ∈ {0,1}`、`sample_id` 全局唯一、各集合两类均有。
3. 同一 `group_id` 不跨集合；空白规范化后的重复正文不跨集合（v2 已在写出时逐条实测，脚本遇泄漏会直接终止）。
4. 模型入口按 `TEAM_WORKPLAN.md` 检查 `sample_id`/`group_id`/正文重叠，不关闭检查绕过泄漏。

## 6. 已知限制

1. 短文本口径待官方核实（当前为字符 <100 的临时口径）。
2. `generator` 未做家族归一化（`davinci`/`text-davinci-003` 等仍分列）。
3. 类别不均衡：machine 约占 63%，训练时建议类别加权或调阈值。
4. `domain` 取自 `source`，缺失记为 `unknown`。

仅本机调试产物（不入库）：`preprocess_log.txt`、`explore_output.txt`、完整划分本地副本。
