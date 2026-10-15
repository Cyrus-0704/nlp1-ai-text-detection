# A → B/C 交接说明（数据版本 internal-v1）

## 1. 基本信息

| 项 | 内容 |
| --- | --- |
| 交付人 | A（数据、评估与交付验收） |
| 用途 | 向 B/C 提供同一套可信、可复现的内部训练/调参/内部测试数据与统一评估口径 |
| 接收人 | B（主模型）、C（基线/融合） |
| 数据版本 | `internal-v1` |
| 分支 | `work/a-data` |
| 代码 commit | 本文件所在 commit（即本次交付 commit）；基线 `f3758d2`（origin/main） |
| 原始数据 | 仓库根 `data.jsonl`（Git LFS，`raw-v1`，不可变，见 `data/manifest.raw.json`） |

本轮只上传**脚本 / manifest / 清洗审计**；完整划分文件约 352MB 且 `*.jsonl` 走 Git LFS，为不占满团队 LFS 配额暂不入库，接收方按第 3 节命令本地生成并用 SHA-256 核对。

## 2. 文件清单

| 相对路径 | SHA-256 | 格式 | 说明 |
| --- | --- | --- | --- |
| `data/processed/v1/manifest.json` | `27696dc3a81cad9c0b45f5e866b0cf1852ad520ddf58d05412d424b9ecea622c` | JSON | 版本/字段/分组/划分/评估/已知限制的权威清单 |
| `data/processed/v1/cleaning_audit.md` | 见文件内 | Markdown | 清洗与过滤统计、去重与分组决策、划分完整性校验 |
| `data_tools/preprocess.py` | `fba43fe349f56b2bfc8f8f1f67285f02d7cd6c6bbf5362a769a0c46e053fd9e3` | Python | 生成 `internal-v1` 的处理与划分脚本（20867 B） |
| `data_tools/evaluate.py` | `f6a3c702b92ac395aeca43781df929d130858a4508cada6d5c610e218b16a209` | Python | 统一评估脚本，零 sklearn 依赖（13229 B） |

划分文件（**本轮未上传**，由接收方本地生成；尺寸与哈希见 `manifest.json` 的 `splits`）：

| 相对路径 | 样本量 | human | machine | bytes | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `data/processed/v1/train.jsonl` | 119727 | 43870 | 75857 | 281186299 | `ebf52482625cc69d73ea1a7b2f2fd3557acc944718f3f3b0c36eb1de9d6a2b6d` |
| `data/processed/v1/dev.jsonl` | 14998 | 5444 | 9554 | 35049138 | `4790283a83857538b7c799d143db9dfb4870449b5d20dd38bf66d2864ff4e14a` |
| `data/processed/v1/test.jsonl` | 15170 | 5594 | 9576 | 35785495 | `737a728de70e70bb7c17940e1bba59ae2f34aa864d1b32035dee33b32cc6850b` |

合计 149895 样本（human 54908 / machine 94987），分组 33199 个。六字段：`sample_id,group_id,text,label,generator,domain`，约定 `human=0`、`machine=1`。

## 3. 接收方可复制的命令

生成划分（在仓库根目录执行，需先有原始 `data.jsonl`）：

```bash
git lfs pull --include="data.jsonl"
python data_tools/preprocess.py --data data.jsonl --out-dir data/processed/v1 --seed 42 --min-len 20
```

统一评估（`predictions.csv` 由 B/C 产出）：

```bash
python data_tools/evaluate.py --gold data/processed/v1/test.jsonl --pred predictions.csv --threshold 0.5
```

- **概率方向**：`p_machine` 表示 machine（`label=1`）的概率，取值 `[0,1]`，`p_machine >= 0.5` 判为 machine。CSV 列 `sample_id,p_machine`，顺序与输入一致，每个编号一条。
- **评估口径**：`S = 0.7 × 全体MacroF1 + 0.3 × 短文本MacroF1`；额外输出混淆矩阵、各类别召回率、短文本数量与占比。
- **短文本口径**：团队临时口径为「<100 **词**」；`evaluate.py` 默认按「字符 < 100」（`--short-by char`），需按官方核实结果用 `--short-by word` 切换，并在报告中标注为临时口径。

## 4. 验收检查与调试产物

接收方请执行以下检查：

1. 生成文件与 `manifest.json` 的 `splits` 中 `sha256` / `bytes` / 样本数一致。
2. 文本非空、`label ∈ {0,1}`、`sample_id` 全局唯一、train/dev/test 两类均有。
3. 同一 `group_id` 不跨集合；空白规范化后的重复正文不跨集合（`manifest.json` 记录 `cross_split_group_leakage=false`）。
4. 模型入口按 `TEAM_WORKPLAN.md` 检查 `sample_id`/`group_id`/正文重叠，不关闭检查绕过泄漏。

仅本机调试产物（不入库、不作为验收依据）：本机生成的 `train/dev/test.jsonl` 副本、`preprocess_log.txt`、`explore_output.txt`。正式审计以 `data/processed/v1/cleaning_audit.md` 为准。
