"""从正式 train/dev 抽取真实小样(smoke samples), 供 B/C 首轮快速验收。

- 分层抽样: 按 label 分层, 保持与原集合相同的类别比例, 两类均有。
- 六字段原样保留, 行序打乱; 与正式划分同源, 天然满足组隔离。
- 用法:
    python data_tools/make_smoke.py --train data/processed/v1/train.jsonl \
        --dev data/processed/v1/dev.jsonl --out-dir data/processed/v1 \
        --n-train 1000 --n-dev 200 --seed 42
"""
import argparse
import hashlib
import json
import os
import random


def load_by_label(path):
    """读取 JSONL, 按 label 分桶, 返回 {label: [line_str, ...]}。"""
    buckets = {0: [], 1: []}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            lab = json.loads(line)["label"]
            if lab not in buckets:
                raise ValueError(f"非法 label={lab} in {path}")
            buckets[lab].append(line)
    return buckets


def stratified_sample(buckets, n, rng):
    """按原比例分层抽取 n 条, 保证每类至少 1 条。"""
    total = sum(len(v) for v in buckets.values())
    picked = []
    remain = n
    labels = sorted(buckets.keys())
    for i, lab in enumerate(labels):
        if i == len(labels) - 1:
            k = remain
        else:
            k = max(1, round(n * len(buckets[lab]) / total))
            k = min(k, remain - (len(labels) - i - 1))  # 给剩余类别至少各留1条
        k = min(k, len(buckets[lab]))
        picked.extend(rng.sample(buckets[lab], k))
        remain -= k
    rng.shuffle(picked)
    return picked


def write_jsonl(lines, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for ln in lines:
            f.write(ln + "\n")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def report(name, lines, path):
    n_h = sum(1 for ln in lines if json.loads(ln)["label"] == 0)
    n_m = len(lines) - n_h
    size = os.path.getsize(path)
    digest = sha256_of(path)
    print(f"{name}: {len(lines)} 条 (human={n_h}, machine={n_m}), "
          f"{size} bytes, sha256={digest}")
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser(description="抽取 smoke 真实小样")
    ap.add_argument("--train", default="data/processed/v1/train.jsonl")
    ap.add_argument("--dev", default="data/processed/v1/dev.jsonl")
    ap.add_argument("--out-dir", default="data/processed/v1")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-dev", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    for src, n, name in ((args.train, args.n_train, "smoke_train"),
                         (args.dev, args.n_dev, "smoke_dev")):
        buckets = load_by_label(src)
        lines = stratified_sample(buckets, n, rng)
        out = os.path.join(args.out_dir, f"{name}.jsonl")
        write_jsonl(lines, out)
        report(f"{name}.jsonl", lines, out)


if __name__ == "__main__":
    main()
