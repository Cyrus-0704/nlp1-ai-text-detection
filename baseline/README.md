# C Baseline

This CPU baseline uses character TF-IDF features and binary logistic regression. It consumes A's fixed JSONL splits and never creates a new split.

## Train and predict dev

From the repository root, with the project environment active:

```powershell
python baseline/run.py train `
  --train data/processed/v2/train.jsonl `
  --dev data/processed/v2/dev.jsonl `
  --dataset-version internal-v2 `
  --output-dir runs/c_baseline_v2_char25
```

The command fits the vectorizer and classifier on train only. It writes `baseline.joblib`, `dev_predictions.csv` (`sample_id,p_machine`), and `run.json` with data hashes, parameters, package versions, class counts, and fit time. The output directory must be new so experiments are not overwritten.

## Predict with a saved model

The input JSONL must contain `sample_id` and `text`:

```powershell
python baseline/run.py predict `
  --model runs/c_baseline_v2_char25/baseline.joblib `
  --input path/to/input.jsonl `
  --output runs/c_baseline_v2_char25/predictions.csv
```

Evaluate dev predictions with A's script, explicitly selecting the agreed short-text convention. Until A confirms the official multilingual word-count rule, `--short-by word` is a temporary whitespace-token convention, not an official metric.

```powershell
python baseline/validate_predictions.py `
  --gold-jsonl data/processed/v2/dev.jsonl `
  --pred-csv runs/c_baseline_v2_char25/dev_predictions.csv

python data_tools/evaluate.py `
  --gold data/processed/v2/dev.jsonl `
  --pred runs/c_baseline_v2_char25/dev_predictions.csv `
  --threshold 0.5 --short-by word --short-threshold 100
```

Generate privacy-preserving error slices by length, domain, and generator. The report contains aggregate counts and metrics only; it does not export text or sample IDs.

```powershell
python baseline/analyze.py `
  --gold .local/processed/v2/dev.jsonl `
  --pred runs/c02_internal_v2_char25/dev_predictions.csv `
  --dataset-version internal-v2 `
  --output-dir runs/c02_internal_v2_error_analysis
```

Keep model outputs and full experiment runs local. Only copy artifacts required for review into `handoffs/C/<experiment-id>/`.

## Compare with B and test simple fusion

After receiving B predictions generated on the exact same dev version, validate them and run a predeclared coarse grid of probability blends. The default short-text count is whitespace-token based and provisional until A confirms the official rule.

```powershell
python baseline/compare.py `
  --gold .local/processed/v2/dev.jsonl `
  --c-pred runs/c02_internal_v2_char25/dev_predictions.csv `
  --peer CE500=.local/b_handoff/handoffs/B/b_v2_training_round2/CE500_dev_predictions.csv `
  --peer CE2000_s42=.local/b_handoff/handoffs/B/b_v2_training_round2/CE2000_s42_dev_predictions.csv `
  --peer CE2000_s43=.local/b_handoff/handoffs/B/b_v2_training_round2/CE2000_s43_dev_predictions.csv `
  --output-dir runs/c03_internal_v2_comparison
```

The output includes the full candidate grid, error-overlap counts, and one best-dev blend CSV per peer model. The selected weight is exploratory because it was selected on dev; report the full comparison and do not treat it as an independent performance estimate.

For a predeclared fixed blend, use this verifier instead of the grid-search command. It checks the reported CSV against the exact formula and reports classification/probability metrics and C/B error overlap without searching weights or thresholds.

```powershell
python baseline/verify_fixed_fusion.py `
  --gold .local/processed/v2/dev.jsonl `
  --c-prediction handoffs/C/c_v2_round1/baseline_dev_predictions.csv `
  --b-prediction handoffs/B/b_v2_length512_two_seed/L512_s43_u2000_dev_predictions.csv `
  --reported-fusion handoffs/B/b_v2_length512_two_seed/C07_B03_L512_s43_u2000_dev_predictions.csv `
  --output-json runs/c06_v2_fixed_fusion_verification/summary.json `
  --c-weight 0.7 --threshold 0.5 --short-by word --short-threshold 100
```