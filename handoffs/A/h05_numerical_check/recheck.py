"""H05 补充核验: 按原判据公式复算 C/B/融合超限条数与决策一致性.

判据: abs(actual - reference) > atol + rtol * abs(reference)
  C: rtol=1e-6, atol=1e-8; B 与融合: rtol=1e-5, atol=1e-6
不重跑推理, 仅比对 runs/ 下已保存预测与 bundle 内 references CSV.
用法: python runs/a_h05_recheck.py
"""
import csv
import json
import numpy as np

BUNDLE = "handoffs/shared_models/b_v2_fusion_s43_2000"
SETS = (
    ("smoke_dev", "data/processed/v2/smoke_dev.jsonl", "runs/a_fusion_smoke"),
    ("full_dev", "data/processed/v2/dev.jsonl", "runs/a_fusion_dev"),
)
TOL = {"C": (1e-6, 1e-8), "B": (1e-5, 1e-6), "predictions": (1e-5, 1e-6)}


def read_csv(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig", newline="")))
    ids = [r["sample_id"] for r in rows]
    vals = np.array([float(r["p_machine"]) for r in rows])
    return ids, vals


def main():
    ref = {}
    for comp in ("C", "B", "predictions"):
        r_ids, r = read_csv(f"{BUNDLE}/references/{comp}.csv")
        ref[comp] = dict(zip(r_ids, r))
        print(f"reference {comp}: {len(r_ids)} rows")

    for name, gold_path, run_dir in SETS:
        gold_ids = [json.loads(l)["sample_id"] for l in open(gold_path, encoding="utf-8") if l.strip()]
        print(f"=== {name} ({len(gold_ids)} 条) ===")
        for comp in ("C", "B", "predictions"):
            a_ids, a = read_csv(f"{run_dir}/{comp}.csv")
            id_ok = (a_ids == gold_ids) and all(i in ref[comp] for i in gold_ids)
            exp = np.array([ref[comp][i] for i in gold_ids])
            rtol, atol = TOL[comp]
            viol = int((np.abs(a - exp) > atol + rtol * np.abs(exp)).sum())
            dec_diff = int(np.sum((a >= 0.5) != (exp >= 0.5)))
            print(
                f"{comp}: id_order_exact={id_ok} rtol={rtol} atol={atol} "
                f"violations={viol} max_abs_diff={np.abs(a - exp).max():.6e} "
                f"decision_diffs={dec_diff}"
            )


if __name__ == "__main__":
    main()
