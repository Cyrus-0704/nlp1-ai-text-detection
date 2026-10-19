# V1 运行证据归档（H05 推理运行记录补齐）

归档日期：2026-10-18。交付人：角色 A。用途：补齐 V1（bundle SHA-256 `5775334edda46069d45f7eaade2222bea2c7393d19c6faae44a3ed4a256ab50a`）H05 复现与 CPU 重复性验证的原始运行记录。共 9 个文件，均为原字节复制（复制后逐文件与 runs/ 源文件 SHA-256 比对，全部一致），未重跑推理、未重存 CSV、未改换行符。

## 目录与文件

| 本目录 | 来源 | 文件 | 字节 | SHA-256 |
| --- | --- | --- | --- | --- |
| `v1_runs_smoke/` | `runs/a_fusion_smoke/` | prediction_manifest.json | 754 | `2157c1c888312d243392f2baf621be0ba34db2c5c82a72ecdbccf772e4ce5782` |
| `v1_runs_smoke/` | `runs/a_fusion_smoke/` | stages.json | 251 | `fc55f398f355b788a5f684075c591b84ac32eda5e669a3586b16377464b17467` |
| `v1_runs_dev/` | `runs/a_fusion_dev/` | prediction_manifest.json | 762 | `bda2c604818a6f5f728479134a5de41d886b12c8732e583370c92354c18614cc` |
| `v1_runs_dev/` | `runs/a_fusion_dev/` | stages.json | 250 | `f7e8549132fd64752078d00996d9491bc27ef4cfc541e325f8a17a07ce65846c` |
| `v1_runs_smoke_repeat/` | `runs/v1_smoke_repeat/` | prediction_manifest.json | 754 | `2157c1c888312d243392f2baf621be0ba34db2c5c82a72ecdbccf772e4ce5782` |
| `v1_runs_smoke_repeat/` | `runs/v1_smoke_repeat/` | stages.json | 249 | `5c0b80f0127f2917f6cebb374210fb3d29e84fc4b015a1fdf9264155d88b88dd` |
| `v1_runs_smoke_repeat/` | `runs/v1_smoke_repeat/` | C.csv | 6275 | `bef7aa31365557129cb7e495e0dd22244a1a83c18960fdaa6b9c31a046169964` |
| `v1_runs_smoke_repeat/` | `runs/v1_smoke_repeat/` | B.csv | 6310 | `a04b5b2b6d3dcc3e02fc124f98702171cad729d72d85e8e78dd72b56d3bbcfd3` |
| `v1_runs_smoke_repeat/` | `runs/v1_smoke_repeat/` | predictions.csv | 6287 | `99113f9fdc7abd4208e2c5c4c23b9481373cdb1f06f1547dff081458f2a8d9c2` |

三份 CSV 仅含 `sample_id,p_machine` 两列（编号与概率，无正文）；repeat 三 CSV 哈希与 `v1_runs_smoke/` 首次运行产物完全一致（CPU 推理逐字节可复现）。

## threads 口径澄清

- **代码侧**：包内 `fusion_inference.py` 第 85 行在推理前显式执行 `torch.set_num_threads(4); torch.use_deterministic_algorithms(True)`，即历史所有 V1 推理运行均为 4 线程 + 确定性算法。
- **运行侧**：此前回执/环境查询中出现的 `threads=24` 是另开 Python 进程查询时的系统/环境默认值（未调用 set_num_threads），**不代表推理时的实际线程数**；历史运行未另行实测线程值，推理实际生效值以代码侧的 4 为准。

## 与 D42 的隔离

本归档仅涉及 V1 旧包运行记录；D42/7500 校准包任务（`runs/a_d42_calibrated_*`）保持独立，未受影响。
