import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, log_loss

from run import read_jsonl, sha256_file
from validate_predictions import validate


def load_prediction_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["sample_id", "p_machine"]:
            raise ValueError(f"{path}: expected columns sample_id,p_machine")
        rows = list(reader)
    ids = [row["sample_id"] for row in rows]
    probabilities = np.asarray([float(row["p_machine"]) for row in rows], dtype=np.float64)
    return ids, probabilities


def metrics(labels, probabilities, short_mask, threshold):
    predicted = (probabilities >= threshold).astype(np.int8)
    short_f1 = f1_score(
        labels[short_mask], predicted[short_mask],
        labels=[0, 1], average="macro", zero_division=0,
    )
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))
    return {
        "macro_f1_all": float(f1_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "macro_f1_short": float(short_f1),
        "score_s": float(0.7 * f1_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0) + 0.3 * short_f1),
        "human_false_positive": fp,
        "machine_false_negative": fn,
        "total_errors": fp + fn,
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def error_complementarity(labels, c_probabilities, b_probabilities, threshold):
    c_wrong = (c_probabilities >= threshold) != labels
    b_wrong = (b_probabilities >= threshold) != labels
    return {
        "samples": int(len(labels)),
        "c_only_correct": int(np.sum(~c_wrong & b_wrong)),
        "b_only_correct": int(np.sum(c_wrong & ~b_wrong)),
        "both_correct": int(np.sum(~c_wrong & ~b_wrong)),
        "both_wrong": int(np.sum(c_wrong & b_wrong)),
    }


def run(args):
    rows = read_jsonl(args.gold, require_labels=True)
    expected_ids = [row["sample_id"] for row in rows]
    labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
    for path in (args.c_prediction, args.b_prediction, args.reported_fusion):
        validate(args.gold, path)

    c_ids, c_probabilities = load_prediction_csv(args.c_prediction)
    b_ids, b_probabilities = load_prediction_csv(args.b_prediction)
    fusion_ids, fusion_probabilities = load_prediction_csv(args.reported_fusion)
    if c_ids != expected_ids or b_ids != expected_ids or fusion_ids != expected_ids:
        raise ValueError("prediction IDs or row order do not match the fixed dev input")

    expected_fusion = args.c_weight * c_probabilities + (1.0 - args.c_weight) * b_probabilities
    max_abs_error = float(np.max(np.abs(expected_fusion - fusion_probabilities)))
    if max_abs_error > args.probability_tolerance:
        raise ValueError(
            f"reported fusion does not match fixed weights: max abs error {max_abs_error:.12g} "
            f"> tolerance {args.probability_tolerance:.12g}"
        )

    if args.short_by == "word":
        lengths = np.asarray([len(row["text"].split()) for row in rows])
    else:
        lengths = np.asarray([len(row["text"]) for row in rows])
    short_mask = lengths < args.short_threshold
    if not short_mask.any():
        raise ValueError("short-text subset is empty")

    summary = {
        "dataset_version": args.dataset_version,
        "gold_sha256": sha256_file(args.gold),
        "prediction_sha256": {
            "c": sha256_file(args.c_prediction),
            "b": sha256_file(args.b_prediction),
            "reported_fusion": sha256_file(args.reported_fusion),
        },
        "rows": len(rows),
        "threshold": args.threshold,
        "short_text_rule": f"{args.short_by} < {args.short_threshold} (provisional unless confirmed by A)",
        "fusion": {
            "c_weight": args.c_weight,
            "b_weight": 1.0 - args.c_weight,
            "max_probability_abs_error_vs_reported_file": max_abs_error,
            "probability_tolerance": args.probability_tolerance,
            "weight_search_performed": False,
        },
        "metrics": {
            "c": metrics(labels, c_probabilities, short_mask, args.threshold),
            "b": metrics(labels, b_probabilities, short_mask, args.threshold),
            "fixed_fusion": metrics(labels, fusion_probabilities, short_mask, args.threshold),
        },
        "error_complementarity": error_complementarity(
            labels, c_probabilities, b_probabilities, args.threshold
        ),
        "interpretation": "Same-dev development results; the fixed blend was not retuned here and is not an independent or official evaluation.",
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for name, result in summary["metrics"].items():
        print(
            f"{name}: all={result['macro_f1_all']:.6f}, short={result['macro_f1_short']:.6f}, "
            f"S={result['score_s']:.6f}, log_loss={result['log_loss']:.6f}, "
            f"brier={result['brier_score']:.6f}, errors={result['total_errors']}"
        )
    print(f"C/B error complementarity: {summary['error_complementarity']}")
    print(f"Fixed blend max probability difference: {max_abs_error:.12g}")
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Verify a fixed C/B probability blend without weight search")
    parser.add_argument("--gold", required=True, help="fixed internal dev JSONL")
    parser.add_argument("--c-prediction", required=True)
    parser.add_argument("--b-prediction", required=True)
    parser.add_argument("--reported-fusion", required=True)
    parser.add_argument("--dataset-version", default="internal-v2")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--c-weight", type=float, default=0.7)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--short-by", choices=("word", "char"), default="word")
    parser.add_argument("--short-threshold", type=int, default=100)
    parser.add_argument("--probability-tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    if not math.isfinite(args.c_weight) or not 0 <= args.c_weight <= 1:
        raise SystemExit("c-weight must be finite and within [0, 1]")
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        raise SystemExit("threshold must be finite and within [0, 1]")
    if args.short_threshold <= 0 or args.probability_tolerance < 0:
        raise SystemExit("short threshold must be positive and tolerance non-negative")
    try:
        run(args)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()