"""H05 数值复核: 按原判据公式复算 C/B/融合超限条数与决策一致性.

判据(严格不变): abs(actual - reference) > atol + rtol * abs(reference)
  C: rtol=1e-6, atol=1e-8; B 与融合(predictions): rtol=1e-5, atol=1e-6

默认直接读取本脚本所在目录交付的轻量预测 CSV
(smoke_C.csv / smoke_B.csv / smoke_predictions.csv 及 dev_*.csv,
仅 sample_id,p_machine 两列), 与 bundle 内 references CSV 比对.
可用 --actual-dir 指向其他预测目录(如 runs/a_fusion_smoke).

任何异常(缺行/重复编号/乱序/非有限概率)均明确报错退出(非零),
不篡改容差, 不自动判 PASS.
用法: python handoffs/A/h05_numerical_check/recheck.py [--actual-dir DIR]
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

BUNDLE = "handoffs/shared_models/b_v2_fusion_s43_2000"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SETS = (
    ("smoke", "data/processed/v2/smoke_dev.jsonl"),
    ("dev", "data/processed/v2/dev.jsonl"),
)
TOL = {"C": (1e-6, 1e-8), "B": (1e-5, 1e-6), "predictions": (1e-5, 1e-6)}


def fail(msg):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(2)


def read_csv_strict(path):
    if not os.path.isfile(path):
        fail(f"文件不存在: {path}")
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["sample_id", "p_machine"]:
            fail(f"{path} 表头应为 sample_id,p_machine, 实际 {reader.fieldnames}")
        rows = list(reader)
    ids, vals = [], []
    for lineno, r in enumerate(rows, start=2):
        sid = r["sample_id"]
        try:
            v = float(r["p_machine"])
        except (TypeError, ValueError):
            fail(f"{path} 第 {lineno} 行概率无法解析: {r['p_machine']!r}")
        if not math.isfinite(v):
            fail(f"{path} 第 {lineno} 行概率非有限值(NaN/Inf): {sid}")
        if not 0.0 <= v <= 1.0:
            fail(f"{path} 第 {lineno} 行概率越界[0,1]: {sid}={v}")
        ids.append(sid)
        vals.append(v)
    if len(set(ids)) != len(ids):
        fail(f"{path} 存在重复 sample_id")
    return ids, np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actual-dir", default=HERE,
                    help="实际预测 CSV 目录(默认: 本脚本所在交付目录)")
    ap.add_argument("--bundle", default=os.path.join(REPO_ROOT, BUNDLE))
    args = ap.parse_args()

    ref = {}
    for comp in ("C", "B", "predictions"):
        r_ids, r = read_csv_strict(os.path.join(args.bundle, "references", f"{comp}.csv"))
        ref[comp] = dict(zip(r_ids, r))

    worst = 0
    for name, gold_rel in SETS:
        gold_path = os.path.join(REPO_ROOT, gold_rel)
        gold_ids = [json.loads(l)["sample_id"] for l in open(gold_path, encoding="utf-8") if l.strip()]
        print(f"=== {name} ({len(gold_ids)} 条) ===")
        for comp in ("C", "B", "predictions"):
            a_ids, a = read_csv_strict(os.path.join(args.actual_dir, f"{name}_{comp}.csv"))
            if len(a_ids) != len(gold_ids):
                fail(f"{name}/{comp}: 行数 {len(a_ids)} 与金标 {len(gold_ids)} 不一致(缺行/多行)")
            if a_ids != gold_ids:
                fail(f"{name}/{comp}: sample_id 列表或顺序与金标不一致(乱序)")
            missing = [i for i in gold_ids if i not in ref[comp]]
            if missing:
                fail(f"{name}/{comp}: {len(missing)} 个编号在参考 CSV 中缺失, 首例 {missing[0]}")
            exp = np.array([ref[comp][i] for i in gold_ids])
            rtol, atol = TOL[comp]
            viol = int((np.abs(a - exp) > atol + rtol * np.abs(exp)).sum())
            dec_diff = int(np.sum((a >= 0.5) != (exp >= 0.5)))
            worst = max(worst, viol, dec_diff)
            print(
                f"{comp}: rtol={rtol} atol={atol} 超限={viol} 条 "
                f"max_abs_diff={np.abs(a - exp).max():.6e} 决策差异={dec_diff} 条"
            )
    sys.exit(1 if worst else 0)


if __name__ == "__main__":
    main()
