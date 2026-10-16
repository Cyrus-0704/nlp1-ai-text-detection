# C Internal-v2 Round 1 Handoff

Status: exploratory development handoff; not a frozen model or official score. The model training run recorded base revision `7231172ba8e99dfe9dead7b40bf0c47e24ba3f21`, but the training worktree had uncommitted changes and did not record the exact source-file fingerprint. The current inference source fingerprint is documented separately in the H04 handoff; do not treat it as a retroactive proof of the exact training source snapshot.

## Data and Evaluation

- Dataset: A `internal-v2`; train 119842 rows, dev 15086 rows.
- Train SHA-256: `5a5dca9db706e0faadc5c89cbcadc77dc585ce302f5e73b11b82213f40ab7ff6`.
- Dev SHA-256: `1d80e7b6045e0914f224dcbc3afe5f56569f5bd995f742607389c26316476600`.
- Train/dev hashes and byte counts match A's manifest. This handoff contains no test predictions or test metrics.
- Threshold: 0.5. Short text: whitespace-token count `<100`, provisional pending A's confirmation of the official rule.

## C Baseline

Character TF-IDF n-grams 2-5, `min_df=2`, `max_features=300000`; balanced logistic regression with `C=1`, `max_iter=1000`, seed 42. TF-IDF and classifier were fitted on train only. Fit time: 530.635 seconds.

The C dev prediction CSV has 15086 rows, exactly matches the dev sample IDs in input order, and contains finite `p_machine` probabilities in `[0,1]`. SHA-256: `031662988c4459b0cc30f1a8b45b89f396568b8b4b844789a88fc4362adfcabd`.

Temporary dev metrics: all macro-F1 0.946766, short macro-F1 0.918468, S 0.938276. These are not official scores.

## B Prediction Comparison

B predictions were read from branch commit `4ca7c858c43fea44a5863eb528df2734b736795b` in an isolated detached worktree; B code was not merged. All three prediction files passed strict row count, sample ID order, uniqueness, and probability validation against the same v2 dev.

The exploratory grid tried C weights 0.1 through 0.9. The best same-dev point for each B candidate was C/B=0.7/0.3. The highest temporary S was 0.957972 for C + B CE2000 seed 43. Its 500-replicate group bootstrap interval for delta S versus C was [0.014761, 0.025716]. Because the weight was selected and evaluated on the same dev, this interval is conditional and is not an independent test. Do not freeze or describe it as unbiased generalization performance without team review and a later frozen evaluation.

The `baseline.joblib` artifact was serialized with Python 3.14.0 and scikit-learn 1.9.1. B must verify loading and reproducing predictions in B's runtime before integrating it.

## Files

- `baseline_dev_predictions.csv`: C baseline probabilities on v2 dev.
- `baseline_run.json`: data hashes, model parameters, package versions, and fit time.
- `baseline_evaluation_word_temp.txt`: A evaluator report using the temporary whitespace-token short-text rule.
- `baseline_error_slices.csv` and `baseline_error_summary.json`: aggregate length/domain/generator error statistics; no raw text or sample IDs.
- `comparison.csv` and `comparison.json`: full coarse-grid and error-overlap results, with group-bootstrap intervals.
- `fusion_c0.7_CE2000_s42.csv` and `fusion_c0.7_CE2000_s43.csv`: exploratory dev predictions for the two 2000-step B candidates.
- `baseline.joblib`: C vectorizer and classifier; test cross-runtime compatibility before integration.

## Reproduction

From the repository root, after generating and verifying A's internal-v2 train/dev splits:

```powershell
python baseline/run.py train `
  --train .local/processed/v2/train.jsonl `
  --dev .local/processed/v2/dev.jsonl `
  --dataset-version internal-v2 `
  --output-dir runs/c02_internal_v2_char25
```

Then evaluate with A's script using the currently provisional convention:

```powershell
python data_tools/evaluate.py `
  --gold .local/processed/v2/dev.jsonl `
  --pred runs/c02_internal_v2_char25/dev_predictions.csv `
  --threshold 0.5 --short-by word --short-threshold 100
```

## Note for A

The fixed preprocessing script prints two example texts from each generated split, including internal test examples. This run did not use internal test for model fitting, tuning, evaluation, or error analysis, and no test-derived results are included here. Please suppress internal-test example printing in future preprocessing output and confirm the official short-text counting convention before C freezes a fusion decision.