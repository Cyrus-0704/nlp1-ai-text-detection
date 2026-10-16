import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from run import read_jsonl, sha256_file
from validate_predictions import validate


def load_probabilities(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return np.asarray([float(row["p_machine"]) for row in csv.DictReader(handle)])


def calculate_slice(rows, labels, predictions, indexes, kind, name):
    selected = np.asarray(indexes, dtype=np.int64)
    selected_labels = labels[selected]
    selected_predictions = predictions[selected]
    human_mask = selected_labels == 0
    machine_mask = selected_labels == 1
    human_recall = (
        float(np.mean(selected_predictions[human_mask] == 0)) if human_mask.any() else None
    )
    machine_recall = (
        float(np.mean(selected_predictions[machine_mask] == 1)) if machine_mask.any() else None
    )
    macro_f1 = None
    if human_mask.any() and machine_mask.any():
        macro_f1 = float(
            f1_score(selected_labels, selected_predictions, labels=[0, 1], average="macro", zero_division=0)
        )
    false_positive = int(np.sum((selected_labels == 0) & (selected_predictions == 1)))
    false_negative = int(np.sum((selected_labels == 1) & (selected_predictions == 0)))
    size = len(selected)
    return {
        "slice_type": kind,
        "slice": name,
        "samples": size,
        "human_support": int(human_mask.sum()),
        "machine_support": int(machine_mask.sum()),
        "false_positive_human_as_machine": false_positive,
        "false_negative_machine_as_human": false_negative,
        "error_rate": float((false_positive + false_negative) / size) if size else None,
        "human_recall": human_recall,
        "machine_recall": machine_recall,
        "macro_f1": macro_f1,
        "macro_f1_note": "only reported when both classes are present",
    }


def grouped_slices(rows, field):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            value = "unknown"
        groups[value].append(index)
    return groups


def write_csv(path, records):
    fields = [
        "slice_type", "slice", "samples", "human_support", "machine_support",
        "false_positive_human_as_machine", "false_negative_machine_as_human",
        "error_rate", "human_recall", "machine_recall", "macro_f1", "macro_f1_note",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run(args):
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise ValueError(f"output directory already exists; use a new analysis path: {output_dir}")

    rows = read_jsonl(args.gold, require_labels=True)
    validate(args.gold, args.pred)
    probabilities = load_probabilities(args.pred)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
    predictions = (probabilities >= args.threshold).astype(np.int8)
    if args.short_by == "word":
        lengths = np.asarray([len(row["text"].split()) for row in rows])
    else:
        lengths = np.asarray([len(row["text"]) for row in rows])
    short_mask = lengths < args.short_threshold

    slices = [
        calculate_slice(rows, labels, predictions, np.arange(len(rows)), "overall", "all"),
        calculate_slice(rows, labels, predictions, np.flatnonzero(short_mask), "length", "short"),
        calculate_slice(rows, labels, predictions, np.flatnonzero(~short_mask), "length", "not_short"),
    ]
    for field, kind in (("domain", "domain"), ("generator", "generator")):
        for name, indexes in grouped_slices(rows, field).items():
            slices.append(calculate_slice(rows, labels, predictions, indexes, kind, name))

    slices.sort(key=lambda row: (row["slice_type"], -row["samples"], row["slice"]))
    output_dir.mkdir(parents=True)
    csv_path = output_dir / "error_slices.csv"
    json_path = output_dir / "error_summary.json"
    write_csv(csv_path, slices)
    summary = {
        "dataset_version": args.dataset_version,
        "gold_sha256": sha256_file(args.gold),
        "prediction_sha256": sha256_file(args.pred),
        "threshold": args.threshold,
        "short_text_rule": f"{args.short_by} < {args.short_threshold} (provisional unless confirmed by A)",
        "samples": len(rows),
        "overall": slices[0] if slices[0]["slice_type"] == "overall" else next(
            row for row in slices if row["slice_type"] == "overall"
        ),
        "slice_count": len(slices),
        "privacy_note": "No raw text or sample IDs are written to this report.",
        "interpretation_note": "Small domain/generator slices are descriptive and are not independent generalization estimates.",
        "slices_csv": csv_path.name,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    overall = next(row for row in slices if row["slice_type"] == "overall")
    short = next(row for row in slices if row["slice_type"] == "length" and row["slice"] == "short")
    print(
        f"Samples={len(rows)}; errors={overall['false_positive_human_as_machine'] + overall['false_negative_machine_as_human']}; "
        f"short={short['samples']} with macro-F1={short['macro_f1']:.6f}; "
        f"saved {csv_path} and {json_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze baseline errors by text length, domain, and generator")
    parser.add_argument("--gold", required=True, help="fixed internal dev JSONL")
    parser.add_argument("--pred", required=True, help="prediction CSV with sample_id,p_machine")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output-dir", required=True, help="new, unused local analysis directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--short-by", choices=("word", "char"), default="word")
    parser.add_argument("--short-threshold", type=int, default=100)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        raise SystemExit("threshold must be within [0, 1]")
    if args.short_threshold <= 0:
        raise SystemExit("short threshold must be positive")
    try:
        run(args)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()