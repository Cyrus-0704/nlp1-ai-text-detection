import argparse
import csv
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def read_jsonl(path, require_labels):
    rows = []
    seen_ids = set()
    required_fields = {"sample_id", "text"}
    if require_labels:
        required_fields.add("label")

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict) or not required_fields.issubset(row):
                raise ValueError(f"{path}:{line_number}: missing required fields")

            sample_id = row["sample_id"]
            text = row["text"]
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{path}:{line_number}: sample_id must be a non-empty string")
            if sample_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id {sample_id!r}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number}: text must be non-empty")

            item = {"sample_id": sample_id, "text": text}
            for field in ("group_id", "generator", "domain"):
                if field in row:
                    item[field] = row[field]
            if require_labels:
                try:
                    label = int(row["label"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: label must be 0 or 1") from error
                if label not in (0, 1):
                    raise ValueError(f"{path}:{line_number}: label must be 0 or 1")
                item["label"] = label

            rows.append(item)
            seen_ids.add(sample_id)

    if not rows:
        raise ValueError(f"{path}: no samples found")
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_model(args):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(args.min_ngram, args.max_ngram),
                    min_df=args.min_df,
                    max_features=args.max_features,
                    dtype=np.float32,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=args.c,
                    class_weight="balanced",
                    max_iter=args.max_iter,
                    solver="liblinear",
                    random_state=args.seed,
                ),
            ),
        ]
    )


def write_predictions(path, rows, probabilities):
    if len(rows) != len(probabilities):
        raise ValueError("prediction count does not match sample count")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("machine probabilities must be finite and within [0, 1]")

    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "p_machine"])
        writer.writerows(
            (row["sample_id"], format(float(probability), ".10g"))
            for row, probability in zip(rows, probabilities)
        )


def run_train(args):
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise ValueError(f"output directory already exists; use a new run path: {output_dir}")

    train_rows = read_jsonl(args.train, require_labels=True)
    dev_rows = read_jsonl(args.dev, require_labels=False)
    labels = [row["label"] for row in train_rows]
    label_counts = Counter(labels)
    if set(label_counts) != {0, 1}:
        raise ValueError("training data must contain both labels 0 and 1")

    model = build_model(args)
    start_time = time.perf_counter()
    model.fit([row["text"] for row in train_rows], labels)
    fit_seconds = time.perf_counter() - start_time

    machine_column = list(model.named_steps["classifier"].classes_).index(1)
    probabilities = model.predict_proba([row["text"] for row in dev_rows])[:, machine_column]

    output_dir.mkdir(parents=True)
    model_path = output_dir / "baseline.joblib"
    prediction_path = output_dir / "dev_predictions.csv"
    metadata_path = output_dir / "run.json"
    joblib.dump(model, model_path)
    write_predictions(prediction_path, dev_rows, probabilities)

    metadata = {
        "dataset_version": args.dataset_version,
        "git_revision": git_revision(),
        "python_version": __import__("sys").version.split()[0],
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "train_path": str(Path(args.train)),
        "train_sha256": sha256_file(args.train),
        "train_samples": len(train_rows),
        "train_label_counts": {str(label): count for label, count in sorted(label_counts.items())},
        "dev_path": str(Path(args.dev)),
        "dev_sha256": sha256_file(args.dev),
        "dev_samples": len(dev_rows),
        "model": {
            "vectorizer": "TfidfVectorizer",
            "analyzer": "char",
            "ngram_range": [args.min_ngram, args.max_ngram],
            "min_df": args.min_df,
            "max_features": args.max_features,
            "classifier": "LogisticRegression",
            "C": args.c,
            "class_weight": "balanced",
            "max_iter": args.max_iter,
            "solver": "liblinear",
            "seed": args.seed,
        },
        "fit_seconds": round(fit_seconds, 3),
        "artifacts": {
            "model": model_path.name,
            "dev_predictions": prediction_path.name,
        },
        "evaluation_status": "not evaluated by this command",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model, dev predictions, and run metadata to {output_dir}")
    print("Evaluate predictions with data_tools/evaluate.py using the agreed dev protocol.")


def run_predict(args):
    rows = read_jsonl(args.input, require_labels=False)
    model = joblib.load(args.model)
    machine_column = list(model.named_steps["classifier"].classes_).index(1)
    probabilities = model.predict_proba([row["text"] for row in rows])[:, machine_column]
    write_predictions(args.output, rows, probabilities)
    print(f"Saved {len(rows)} predictions to {args.output}")


def build_parser():
    parser = argparse.ArgumentParser(description="Character TF-IDF + logistic regression baseline")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="fit on A's train split and predict A's dev split")
    train.add_argument("--train", required=True, help="A six-field train JSONL")
    train.add_argument("--dev", required=True, help="A six-field dev JSONL")
    train.add_argument("--output-dir", required=True, help="new, unused experiment directory")
    train.add_argument("--dataset-version", required=True, help="for example: internal-v2")
    train.add_argument("--min-ngram", type=int, default=2)
    train.add_argument("--max-ngram", type=int, default=5)
    train.add_argument("--min-df", type=int, default=2)
    train.add_argument("--max-features", type=int, default=300_000)
    train.add_argument("--c", type=float, default=1.0)
    train.add_argument("--max-iter", type=int, default=1000)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(func=run_train)

    predict = commands.add_parser("predict", help="load a saved baseline and predict JSONL")
    predict.add_argument("--model", required=True, help="saved baseline.joblib")
    predict.add_argument("--input", required=True, help="JSONL with sample_id and text")
    predict.add_argument("--output", required=True, help="new prediction CSV path")
    predict.set_defaults(func=run_predict)
    return parser


def main():
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()