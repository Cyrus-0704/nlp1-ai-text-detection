import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from run import read_jsonl, sha256_file, write_predictions
from validate_predictions import validate


def load_probabilities(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return np.asarray([float(row["p_machine"]) for row in csv.DictReader(handle)])


def score_probabilities(labels, probabilities, short_mask, threshold):
    predictions = (probabilities >= threshold).astype(np.int8)
    macro_all = f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)
    macro_short = f1_score(
        labels[short_mask], predictions[short_mask],
        labels=[0, 1], average="macro", zero_division=0,
    )
    recall = [
        float(np.mean(predictions[labels == label] == label))
        for label in (0, 1)
    ]
    return {
        "macro_f1_all": float(macro_all),
        "macro_f1_short": float(macro_short),
        "score_s": float(0.7 * macro_all + 0.3 * macro_short),
        "human_recall": recall[0],
        "machine_recall": recall[1],
    }


def error_overlap(labels, c_probabilities, peer_probabilities, short_mask, threshold):
    c_wrong = (c_probabilities >= threshold) != labels
    peer_wrong = (peer_probabilities >= threshold) != labels

    def counts(mask):
        return {
            "samples": int(mask.sum()),
            "both_correct": int((mask & ~c_wrong & ~peer_wrong).sum()),
            "c_only_correct": int((mask & ~c_wrong & peer_wrong).sum()),
            "peer_only_correct": int((mask & c_wrong & ~peer_wrong).sum()),
            "both_wrong": int((mask & c_wrong & peer_wrong).sum()),
        }

    return {
        "all": counts(np.ones(len(labels), dtype=bool)),
        "short": counts(short_mask),
    }


def read_group_ids(path):
    groups = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            group_id = row.get("group_id")
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(f"{path}:{line_number}: group_id must be a non-empty string")
            groups.append(group_id)
    return groups


def bootstrap_delta_s(labels, c_probabilities, blend_probabilities, short_mask, groups, threshold, replicates, seed):
    rows_by_group = defaultdict(list)
    for row_index, group_id in enumerate(groups):
        rows_by_group[group_id].append(row_index)
    unique_groups = list(rows_by_group)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)

    for replicate in range(replicates):
        selected = rng.integers(0, len(unique_groups), size=len(unique_groups))
        indices = np.fromiter(
            (row_index for group_index in selected for row_index in rows_by_group[unique_groups[group_index]]),
            dtype=np.int64,
        )
        sampled_short = short_mask[indices]
        c_score = score_probabilities(
            labels[indices], c_probabilities[indices], sampled_short, threshold
        )["score_s"]
        blend_score = score_probabilities(
            labels[indices], blend_probabilities[indices], sampled_short, threshold
        )["score_s"]
        deltas[replicate] = blend_score - c_score

    lower, upper = np.percentile(deltas, [2.5, 97.5])
    return {
        "unit": "group_id, sampled with replacement",
        "replicates": replicates,
        "seed": seed,
        "delta_s_ci_95": [float(lower), float(upper)],
        "fraction_delta_s_positive": float(np.mean(deltas > 0)),
        "caveat": "Conditional on models and a blend weight selected on this same dev; not an independent test or training-seed interval.",
    }


def write_comparison(path, records):
    fields = [
        "peer", "system", "c_weight", "macro_f1_all", "macro_f1_short", "score_s",
        "human_recall", "machine_recall",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run(args):
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise ValueError(f"output directory already exists; use a new run path: {output_dir}")

    rows = read_jsonl(args.gold, require_labels=True)
    groups = read_group_ids(args.gold)
    if len(groups) != len(rows):
        raise ValueError("group_id count does not match the validated dev sample count")
    sample_ids = [row["sample_id"] for row in rows]
    labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
    if args.short_by == "word":
        lengths = np.asarray([len(row["text"].split()) for row in rows])
    else:
        lengths = np.asarray([len(row["text"]) for row in rows])
    short_mask = lengths < args.short_threshold
    if not short_mask.any():
        raise ValueError("short-text subset is empty")

    validate(args.gold, args.c_pred)
    c_probabilities = load_probabilities(args.c_pred)
    c_metrics = score_probabilities(labels, c_probabilities, short_mask, args.threshold)
    results = [{"peer": "", "system": "C", "c_weight": 1.0, **c_metrics}]
    pair_reports = {}
    best_fusions = {}

    for peer_spec in args.peer:
        if "=" not in peer_spec:
            raise ValueError("each --peer must use NAME=CSV_PATH")
        peer_name, peer_path = peer_spec.split("=", 1)
        if not peer_name or not peer_path:
            raise ValueError("each --peer must use NAME=CSV_PATH")

        validate(args.gold, peer_path)
        peer_probabilities = load_probabilities(peer_path)
        peer_metrics = score_probabilities(labels, peer_probabilities, short_mask, args.threshold)
        results.append({"peer": peer_name, "system": peer_name, "c_weight": 0.0, **peer_metrics})
        pair_reports[peer_name] = {
            "peer_metrics": peer_metrics,
            "error_overlap": error_overlap(
                labels, c_probabilities, peer_probabilities, short_mask, args.threshold
            ),
        }

        candidates = []
        for step in range(1, 10):
            c_weight = step / 10
            blended = c_weight * c_probabilities + (1 - c_weight) * peer_probabilities
            metrics = score_probabilities(labels, blended, short_mask, args.threshold)
            row = {
                "peer": peer_name,
                "system": "blend",
                "c_weight": c_weight,
                **metrics,
            }
            results.append(row)
            candidates.append((metrics["score_s"], metrics["macro_f1_all"], c_weight, blended))

        best = max(candidates, key=lambda item: (item[0], item[1], -abs(item[2] - 0.5)))
        best_path = output_dir / f"fusion_c{best[2]:.1f}_{peer_name}.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_predictions(best_path, rows, best[3])
        bootstrap = bootstrap_delta_s(
            labels,
            c_probabilities,
            best[3],
            short_mask,
            groups,
            args.threshold,
            args.bootstrap_replicates,
            args.seed,
        )
        best_fusions[peer_name] = {
            "c_weight": best[2],
            "peer_weight": 1 - best[2],
            "metrics": score_probabilities(labels, best[3], short_mask, args.threshold),
            "delta_s_vs_c_bootstrap": bootstrap,
            "prediction_path": best_path.name,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "comparison.csv"
    write_comparison(comparison_path, results)
    summary = {
        "gold_sha256": sha256_file(args.gold),
        "c_prediction_sha256": sha256_file(args.c_pred),
        "threshold": args.threshold,
        "short_text_rule": f"{args.short_by} < {args.short_threshold} (provisional unless confirmed by A)",
        "fusion_grid": "C weight 0.1 through 0.9 in 0.1 increments; peer weight is 1 - C weight",
        "selection_note": "Best fusion selected on this dev set only; treat as exploratory and report all candidates.",
        "c_metrics": c_metrics,
        "pairs": pair_reports,
        "best_fusions": best_fusions,
        "comparison_csv": comparison_path.name,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Short-text rule: {summary['short_text_rule']}")
    print(f"C: all={c_metrics['macro_f1_all']:.6f}, short={c_metrics['macro_f1_short']:.6f}, S={c_metrics['score_s']:.6f}")
    for peer_name, report in pair_reports.items():
        peer_metrics = report["peer_metrics"]
        best = best_fusions[peer_name]
        overlap = report["error_overlap"]["all"]
        print(
            f"{peer_name}: all={peer_metrics['macro_f1_all']:.6f}, "
            f"short={peer_metrics['macro_f1_short']:.6f}, S={peer_metrics['score_s']:.6f}; "
            f"C-only-correct={overlap['c_only_correct']}, peer-only-correct={overlap['peer_only_correct']}; "
            f"best dev blend C={best['c_weight']:.1f}/peer={best['peer_weight']:.1f}, "
            f"S={best['metrics']['score_s']:.6f}, "
            f"group-bootstrap deltaS 95%CI={best['delta_s_vs_c_bootstrap']['delta_s_ci_95']}"
        )
    print(f"Saved comparison artifacts to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare C baseline with same-dev peer predictions")
    parser.add_argument("--gold", required=True, help="fixed dev JSONL")
    parser.add_argument("--c-pred", required=True, help="C prediction CSV")
    parser.add_argument("--peer", action="append", required=True, help="NAME=CSV_PATH; repeat for each peer model")
    parser.add_argument("--output-dir", required=True, help="new, unused local result directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--short-by", choices=("word", "char"), default="word")
    parser.add_argument("--short-threshold", type=int, default=100)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        raise SystemExit("threshold must be within [0, 1]")
    if args.short_threshold <= 0:
        raise SystemExit("short threshold must be positive")
    if args.bootstrap_replicates <= 0:
        raise SystemExit("bootstrap replicates must be positive")
    try:
        run(args)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()