# H05 数值核验证据归档（角色 A → 全队）

归档日期：2026-10-17。交付人：角色 A（数据与评估负责人）。用途：供 B 复核 H05 独立环境复现的数值核验证据，供 C 撰写报告引用。

## 内容清单

| 文件 | 说明 |
| --- | --- |
| `recheck.py` | 原判据复算脚本（`abs(actual-ref) > atol + rtol*abs(ref)`；C: rtol=1e-6/atol=1e-8；B 与融合: rtol=1e-5/atol=1e-6）。不重跑推理，仅比对轻量预测 CSV 与 bundle 内 references CSV |
| `smoke_check.json` / `dev_check.json` | A 侧补充分析脚本生成的核验汇总（含 bundle/input SHA、三分量 max_abs_difference、容差、决策一致性、status 标注）。status 标签 `PASS_DECISIONS_IDENTICAL_WITH_FP_TOLERANCE_NOTE` 为 A 侧归纳标注，非 B 原生 `verify_fusion_output.py` 输出 |
| `smoke_{C,B,predictions}.csv`（各 200 行） | smoke_dev 复核用轻量预测，仅 `sample_id,p_machine` 两列，无正文 |
| `dev_{C,B,predictions}.csv`（各 15086 行） | full-dev 复核用轻量预测，仅 `sample_id,p_machine` 两列，无正文 |
| `manifest.json` | 本目录全部文件的 bytes + SHA-256 |

## 复算命令（在仓库根目录执行）

```bash
python handoffs/A/h05_numerical_check/recheck.py
```

依赖：仅需 numpy；需本机存在 `data/processed/v2/{smoke_dev,dev}.jsonl`、`runs/a_fusion_{smoke,dev}/` 与 bundle 目录。若仅复核 CSV 一致性，可将脚本中 `run_dir` 指向本目录下的轻量 CSV。

## 关键结论

- C/B/融合在 smoke_dev（200 条）与 full_dev（15086 条）上 0.5 分类决策均 100% 一致。
- 原判据超限：full-dev 上 B 分量 274 条（1.82%）、融合 4 条（0.027%）、C 0 条；smoke_dev 上 B 4 条、其余 0 条。
- 复现环境：Windows 11 / Python 3.11 / torch 2.13.0+cpu / transformers 4.57.6 / CPU FP32；差异归因为 CPU vs CUDA 跨硬件浮点累积。

## 未决事项

- B 分量与融合的原判据超限条数保持「数值差异待共同闭环」状态，不强制标记 PASS。
- 官方口径（多语言计词、阈值、CSV 模板/方向、截止时间）仍未确认，见 `docs/tasks/A_DATA_EVALUATION.md`。
- internal test 继续封存，待方案与验收正式确认后一次性正式评估。
