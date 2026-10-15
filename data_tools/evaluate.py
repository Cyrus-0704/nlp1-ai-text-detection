# -*- coding: utf-8 -*-
"""
evaluate.py —— "大模型生成文本检测与生成器归因" 竞赛 团队内部评估脚本

输入
----
1. gold  : 标准测试集 (JSONL), 须含 sample_id 与 label (0=human, 1=machine),
           同时含 text (用于短文本子集切分). 例如我们生成的 test.jsonl.
2. pred  : 模型预测 CSV, 须含 sample_id 与 p_machine (机器生成的预测概率, [0,1]).

核心指标
--------
- 全体样本 Macro-F1
- 短文本子集 Macro-F1   (短文本定义: 字符数或词数 < 阈值, 可配置)
- 综合得分 = 0.7 * 全体宏F1 + 0.3 * 短文本宏F1

额外输出
--------
- 混淆矩阵 (二分类 2×2, 含计数与百分比)
- 各类别召回率 (Recall)
- 短文本样本数量与占比

用法示例
--------
# 基本用法
python evaluate.py --gold test.jsonl --pred predict.csv

# 自定义阈值与短文本切分方式
python evaluate.py --gold test.jsonl --pred predict.csv \
    --threshold 0.5 --short-threshold 100 --short-by char

# 将控制台报告另存为文件
python evaluate.py --gold test.jsonl --pred predict.csv --save-report eval.txt

依赖: 仅 numpy (项目环境已具备). 预测 CSV 用 Python 标准库 csv 读取.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

# Windows 控制台中文输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LABELS = (0, 1)                              # human / machine
LABEL_NAMES = {0: "human", 1: "machine"}


# ----------------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------------
def load_gold(path):
    """读标准测试集 JSONL, 返回 {sample_id: (label, text)}"""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = str(d.get("sample_id", "")).strip()
            if not sid:
                continue
            lab = d.get("label")
            try:
                lab = int(lab)
            except (TypeError, ValueError):
                continue
            if lab not in LABELS:
                continue
            text = d.get("text", "") or ""
            out[sid] = (lab, text)
    return out


def load_pred(path):
    """读预测 CSV, 返回 {sample_id: float p_machine}, 跳过缺失/非法行"""
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return out
        # 兼容 p_machine / prob / probas / score / pred 等常见列名
        cand_cols = ["p_machine", "prob", "probas", "score", "pred", "prediction",
                     "p_machine_prob", "machine_probability"]
        field_lower = {c.lower(): c for c in reader.fieldnames}
        p_col = None
        for c in cand_cols:
            if c.lower() in field_lower:
                p_col = field_lower[c.lower()]
                break
        sid_col = None
        for c in ("sample_id", "sid", "id", "SampleId", "ID"):
            if c.lower() in field_lower:
                sid_col = field_lower[c.lower()]
                break
        if p_col is None or sid_col is None:
            raise ValueError(
                "预测 CSV 需包含 sample_id 与 p_machine 列. 实际列: %s"
                % list(reader.fieldnames)
            )
        for row in reader:
            sid = str(row.get(sid_col, "")).strip()
            if not sid or sid in out:
                continue
            raw = row.get(p_col, "")
            try:
                p = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isnan(p):
                continue
            out[sid] = p
    return out


def align(gold, pred):
    """按 sample_id 对齐, 同时返回:
    ys_true : int8 array, shape (N,)
    ys_prob : float array, shape (N,)
    texts   : list of str, shape (N,)
    miss_in_pred : 金标中存在但预测缺失的 sample_id 数
    extra_in_pred: 预测中存在但金标缺失的 sample_id 数
    """
    common = [sid for sid in gold.keys() if sid in pred]
    if not common:
        raise SystemExit("[错误] 金标与预测无任何 sample_id 交集, 无法评估")
    common.sort()                               # 保证可复现顺序

    ys_true = np.array([gold[s][0] for s in common], dtype=np.int8)
    ys_prob = np.array([pred[s] for s in common], dtype=np.float64)
    texts = [gold[s][1] for s in common]

    miss_in_pred = len(gold) - len(common)
    extra_in_pred = len(pred) - len(common)
    return ys_true, ys_prob, texts, miss_in_pred, extra_in_pred


# ----------------------------------------------------------------------------
# 指标计算 (纯 numpy 手写, 零额外依赖)
# ----------------------------------------------------------------------------
def confusion_matrix(ys_true, ys_pred):
    """二分类 2×2 混淆矩阵, 行=真实, 列=预测, 顺序 [0, 1]."""
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(ys_true, ys_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1(ys_true, ys_pred, cls):
    """计算类别 cls 的 (precision, recall, f1)."""
    tp = int(np.sum((ys_true == cls) & (ys_pred == cls)))
    fp = int(np.sum((ys_true != cls) & (ys_pred == cls)))
    fn = int(np.sum((ys_true == cls) & (ys_pred != cls)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def macro_f1(ys_true, ys_pred):
    """二分类 Macro-F1 = (F1_0 + F1_1) / 2."""
    return 0.5 * (
        precision_recall_f1(ys_true, ys_pred, 0)[2]
        + precision_recall_f1(ys_true, ys_pred, 1)[2]
    )


# ----------------------------------------------------------------------------
# 短文本子集切分
# ----------------------------------------------------------------------------
def split_short(texts, threshold, by):
    """返回布尔掩码: True 表示短文本."""
    by = by.lower()
    if by == "char":
        lens = np.array([len(t) for t in texts], dtype=np.int64)
    elif by == "word":
        lens = np.array([len(t.split()) for t in texts], dtype=np.int64)
    else:
        raise ValueError("--short-by 仅支持 char 或 word")
    return lens < threshold, lens


# ----------------------------------------------------------------------------
# 报告渲染
# ----------------------------------------------------------------------------
def render_report(opts, ys_true, ys_prob, texts, miss_in_pred, extra_in_pred):
    ys_pred = (ys_prob >= opts.threshold).astype(np.int8)
    cm = confusion_matrix(ys_true, ys_pred)
    N = len(ys_true)

    # 全体指标
    mF1_all = macro_f1(ys_true, ys_pred)
    per_cls = {c: precision_recall_f1(ys_true, ys_pred, c) for c in LABELS}

    # 短文本子集
    short_mask, lens = split_short(texts, opts.short_threshold, opts.short_by)
    N_short = int(short_mask.sum())
    short_ratio = N_short / N if N > 0 else 0.0
    if N_short > 0:
        ys_true_s = ys_true[short_mask]
        ys_pred_s = ys_pred[short_mask]
        mF1_short = macro_f1(ys_true_s, ys_pred_s)
    else:
        mF1_short = float("nan")

    score = 0.7 * mF1_all + 0.3 * (mF1_short if not np.isnan(mF1_short) else 0.0)

    # 格式化
    lines = []
    lines.append("=" * 72)
    lines.append("大模型生成文本检测 —— 团队内部评估报告")
    lines.append("=" * 72)
    lines.append("金标文件              : %s" % os.path.abspath(opts.gold))
    lines.append("预测文件              : %s" % os.path.abspath(opts.pred))
    lines.append("分类阈值(threshold)   : %.3f" % opts.threshold)
    lines.append("短文本阈值            : %d (按 %s)" % (opts.short_threshold, opts.short_by))
    lines.append("综合分公式            : 0.7 * MacroF1_all + 0.3 * MacroF1_short")
    lines.append("")

    lines.append("[对齐情况]")
    lines.append("  金标总数            : %d" % len(ys_true))
    lines.append("  金标有但预测缺失     : %d" % miss_in_pred)
    lines.append("  预测有但金标缺失     : %d" % extra_in_pred)
    lines.append("  实际用于评估样本数   : %d" % N)
    lines.append("")

    # 混淆矩阵
    lines.append("[混淆矩阵] (行=真实, 列=预测; 括号内为行内百分比)")
    lines.append("                         预测 human(0)  预测 machine(1)")
    for t in LABELS:
        row_counts = cm[t]
        row_total = int(row_counts.sum()) or 1
        pcts = [row_counts[p] / row_total * 100 for p in LABELS]
        tag = "真实 %s(%d)" % (LABEL_NAMES[t], t)
        lines.append("  %-22s %8d (%5.1f%%)  %10d (%5.1f%%)" % (
            tag, row_counts[0], pcts[0], row_counts[1], pcts[1]))
    grand_total = int(cm.sum())
    col_sums = cm.sum(axis=0)
    lines.append("  %-22s %8d              %10d" % ("列合计", col_sums[0], col_sums[1]))
    lines.append("")

    # 各类别指标
    lines.append("[各类别指标]")
    lines.append("  %-8s %8s %8s %8s %8s" % ("类别", "P", "R", "F1", "support"))
    for c in LABELS:
        P, R, F1 = per_cls[c]
        sup = int(np.sum(ys_true == c))
        lines.append("  %-8s %8.4f %8.4f %8.4f %8d" % (
            "%s(%d)" % (LABEL_NAMES[c], c), P, R, F1, sup))
    lines.append("  %-8s %8s %8s %8.4f %8d" % ("Macro", "", "", mF1_all, N))
    lines.append("")

    # 召回率单独醒目
    lines.append("[各类别召回率 (Recall)]")
    for c in LABELS:
        R = per_cls[c][1]
        lines.append("  Recall(%s) = %.4f" % (LABEL_NAMES[c], R))
    lines.append("")

    # 短文本子集
    lines.append("[短文本子集]")
    lines.append("  定义                : 文本%s < %d" % (
        "字符数" if opts.short_by == "char" else "词数", opts.short_threshold))
    lines.append("  短文本样本数        : %d / %d  (%.2f%%)" % (N_short, N, short_ratio * 100))
    if N_short > 0:
        lines.append("  子集 Macro-F1       : %.4f" % mF1_short)
        lines.append("  子集类别 Recall     : human=%.4f, machine=%.4f" % (
            precision_recall_f1(ys_true_s, ys_pred_s, 0)[1],
            precision_recall_f1(ys_true_s, ys_pred_s, 1)[1]))
    else:
        lines.append("  (无短文本样本, 子集宏F1跳过, 按 0 计入综合分)")
    lines.append("")

    # 最终分数
    lines.append("=" * 72)
    lines.append("[最终分数]")
    lines.append("  全体样本 Macro-F1   : %.4f" % mF1_all)
    if not np.isnan(mF1_short):
        lines.append("  短文本 Macro-F1     : %.4f" % mF1_short)
    else:
        lines.append("  短文本 Macro-F1     : N/A (按 0 计入)")
    lines.append("  -------")
    lines.append("  综合得分 SCORE      : %.4f" % score)
    lines.append("=" * 72)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="大模型生成文本检测 —— 团队内部评估脚本")
    ap.add_argument("--gold", required=True,
                    help="标准测试集 JSONL (含 sample_id / label / text)")
    ap.add_argument("--pred", required=True,
                    help="模型预测 CSV (含 sample_id / p_machine)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="分类阈值, 默认 0.5")
    ap.add_argument("--short-threshold", type=int, default=100,
                    help="短文本阈值, 默认 100")
    ap.add_argument("--short-by", choices=("char", "word"), default="char",
                    help="短文本按字符数(char,默认)还是词数(word)判断")
    ap.add_argument("--save-report", default=None,
                    help="可选: 将控制台报告另存为指定文件")
    opts = ap.parse_args()

    if not (0.0 <= opts.threshold <= 1.0):
        raise SystemExit("[错误] threshold 须在 [0,1] 区间")
    if opts.short_threshold <= 0:
        raise SystemExit("[错误] short-threshold 须为正整数")

    if not os.path.exists(opts.gold):
        raise SystemExit("[错误] 找不到金标文件: %s" % opts.gold)
    if not os.path.exists(opts.pred):
        raise SystemExit("[错误] 找不到预测文件: %s" % opts.pred)

    gold = load_gold(opts.gold)
    pred = load_pred(opts.pred)
    ys_true, ys_prob, texts, miss_in_pred, extra_in_pred = align(gold, pred)

    report = render_report(opts, ys_true, ys_prob, texts, miss_in_pred, extra_in_pred)
    print(report)

    if opts.save_report:
        with open(opts.save_report, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print("\n[信息] 报告已另存为: %s" % os.path.abspath(opts.save_report))


if __name__ == "__main__":
    main()
