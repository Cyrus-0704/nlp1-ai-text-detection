import argparse
import csv
import math
from pathlib import Path

from run import read_jsonl


def validate(gold_path, prediction_path):
    expected_ids = [row["sample_id"] for row in read_jsonl(gold_path, require_labels=False)]
    with Path(prediction_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["sample_id", "p_machine"]:
            raise ValueError("prediction CSV columns must be exactly sample_id,p_machine")
        rows = list(reader)

    prediction_ids = [row["sample_id"] for row in rows]
    if len(prediction_ids) != len(expected_ids):
        raise ValueError(f"row count mismatch: expected {len(expected_ids)}, got {len(prediction_ids)}")
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("prediction CSV contains duplicate sample_id values")
    if prediction_ids != expected_ids:
        raise ValueError("prediction sample_id values or order do not match the input JSONL")

    for row_number, row in enumerate(rows, start=2):
        try:
            probability = float(row["p_machine"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"row {row_number}: p_machine is not numeric") from error
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"row {row_number}: p_machine must be finite and within [0, 1]")

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Strictly validate baseline prediction CSV alignment")
    parser.add_argument("--gold-jsonl", required=True, help="input JSONL containing sample_id and text")
    parser.add_argument("--pred-csv", required=True, help="CSV with sample_id,p_machine")
    args = parser.parse_args()
    try:
        count = validate(args.gold_jsonl, args.pred_csv)
    except (OSError, ValueError) as error:
        raise SystemExit(f"FAIL: {error}") from error
    print(f"PASS: {count} predictions are uniquely aligned and probabilities are valid")


if __name__ == "__main__":
    main()