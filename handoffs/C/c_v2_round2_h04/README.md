# C → B H04 Model Interface

Status: C-side package and same-environment reference check complete; B Python 3.11 load/reproduction remains pending. This is a development handoff, not a frozen competition model.

## C Model Artifact

- Artifact: `handoffs/C/c_v2_round1/baseline.joblib` (11,931,952 bytes).
- SHA-256: `6d231819e61d24b2f9b8ee91b07b72f6e7c0f8c26d93c500d419d791211c46f8`.
- Current inference source: `baseline/run.py`, SHA-256 `c722f96e6500b0361d25ebaa53813f70b7dad802643717f37bb227cdba692416`.
- Training configuration: character TF-IDF, `analyzer=char`, n-grams 2–5, `min_df=2`, `max_features=300000`, `dtype=float32`; logistic regression `C=1`, `class_weight=balanced`, `max_iter=1000`, `solver=liblinear`, seed 42.
- Class order from the saved artifact is `[0, 1]`; machine is class 1, so `predict_proba(... )[:, 1]` is `p_machine`.
- Serialized with Python 3.14.0, NumPy 2.4.6, scikit-learn 1.9.1, and joblib 1.6.0. Shared dependencies are declared in `requirements/base.txt`.

## Exact Reproduction Caveat

The C training run's metadata recorded base revision `7231172ba8e99dfe9dead7b40bf0c47e24ba3f21`, but C's source tree had uncommitted changes when training ran and no source fingerprint was recorded. The `baseline/run.py` hash above identifies the current inference implementation; it must not be presented as the exact historical training-source hash. This limitation cannot be repaired retroactively from the saved model alone.

## Input, Output, and Fusion Contract

1. Input is A's fixed `internal-v2` JSONL. The model consumes only each row's `text`; text is not otherwise normalized or altered before TF-IDF. `sample_id` is preserved as a string, including leading zeroes.
2. C prediction CSV columns are exactly `sample_id,p_machine`, in the original input order, one row per ID. The reference full-dev prediction is `handoffs/C/c_v2_round1/baseline_dev_predictions.csv`, SHA-256 `031662988c4459b0cc30f1a8b45b89f396568b8b4b844789a88fc4362adfcabd`.
3. B and C outputs must be aligned by exact string `sample_id`, after confirming the same v2 dev hash and row order. Do not align by row position alone unless the ID/order check has passed.
4. Fixed fusion is on probabilities, not logits: `p_machine = 0.7 * p_C + 0.3 * p_B512_s43_u2000`. The B input is `handoffs/B/b_v2_length512_two_seed/L512_s43_u2000_dev_predictions.csv`, SHA-256 `36179ab16a91967cb04f3a12503d2f06751115e4d6838fa35ab6fc914dc591e4`. The reported fixed fusion CSV is B's `C07_B03_L512_s43_u2000_dev_predictions.csv`, SHA-256 `a1d42378fb56d1c58813cd3a39e42a351c04bdda9a95fc328723c941fc1ffea9`.
5. The reported fusion matched the stated formula across all 15,086 rows with maximum absolute probability difference `1.11e-16`. C-side same-dev verification: all macro-F1 0.968463, temporary short macro-F1 0.953381, temporary S 0.963939, log-loss 0.148132, Brier 0.034057; human false positives 240, machine false negatives 200. Short text remains whitespace-token count `<100`, provisional pending A.

## C-Side Load Check and B Action

In C's Python 3.14.0 / scikit-learn 1.9.1 environment, loading the saved joblib and predicting the full v2 dev reproduced the reference CSV SHA exactly. Python 3.11 was not installed on C's machine, so cross-version compatibility is not claimed.

B: load the artifact in the actual Python 3.11 inference environment, verify the `classes_` order and `p_machine` column, then predict the frozen smoke_dev and full dev. Compare rows, IDs, order, probability range, and values to C's reference within a declared tolerance. If loading or predictions differ, report the exact dependency/runtime issue; do not silently retrain or alter preprocessing.

## Data and Evaluation Provenance

- Train SHA-256: `5a5dca9db706e0faadc5c89cbcadc77dc585ce302f5e73b11b82213f40ab7ff6`.
- Dev SHA-256: `1d80e7b6045e0914f224dcbc3afe5f56569f5bd995f742607389c26316476600`.
- Evaluator: `data_tools/evaluate.py`, threshold 0.5, temporary `--short-by word --short-threshold 100`.
- No new fusion weight/threshold search was run in this H04 recheck. The fixed blend and all reported metrics are same-dev development evidence, not an independent or official evaluation.
- B's model source/revision is recorded in the B 512 handoff; B remains responsible for supplying the integrated inference package and its own source/runtime manifest.

Reproduction command for C's baseline inference:

```powershell
python baseline/run.py predict `
  --model handoffs/C/c_v2_round1/baseline.joblib `
  --input .local/processed/v2/dev.jsonl `
  --output runs/h04_c_baseline_dev_predictions.csv
```