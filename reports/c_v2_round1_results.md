# C Internal-v2 Round 1 Results

All scores below are internal development results on the same 15,086-row `internal-v2` dev split. The threshold is 0.5. Short text is temporarily defined as whitespace-token count below 100; A has not confirmed that this is the official multilingual word-count rule. These are not official validation scores.

## Single Models

| Model | All macro-F1 | Short macro-F1 | Temporary S |
| --- | ---: | ---: | ---: |
| C character TF-IDF + balanced logistic regression | 0.946766 | 0.918468 | 0.938276 |
| B CE500 | 0.882048 | 0.876565 | 0.880403 |
| B CE2000 seed 42 | 0.902358 | 0.910128 | 0.904689 |
| B CE2000 seed 43 | 0.892437 | 0.943283 | 0.907691 |

C configuration: character n-grams 2-5, `min_df=2`, `max_features=300000`, logistic-regression `C=1`, balanced class weights, `max_iter=1000`, seed 42. TF-IDF and classifier were fitted on train only. Fit time was 530.635 seconds. Train/dev byte counts and SHA-256 matched A's `internal-v2` manifest.

## Exploratory Probability Blends

The tested grid used C probability weights 0.1 through 0.9 in increments of 0.1. The best point for each B candidate was C/B = 0.7/0.3:

| Blend | All macro-F1 | Short macro-F1 | Temporary S | Conditional group-bootstrap 95% CI for delta S vs C |
| --- | ---: | ---: | ---: | ---: |
| C + B CE500 | 0.957702 | 0.941617 | 0.952877 | [0.010006, 0.020066] |
| C + B CE2000 seed 42 | 0.960841 | 0.935215 | 0.953153 | [0.008961, 0.021217] |
| C + B CE2000 seed 43 | 0.961422 | 0.949922 | 0.957972 | [0.014761, 0.025716] |

The bootstrap resampled `group_id` 500 times with seed 42. It is conditional on models and a weight selected on this same dev set; it is not an independent estimate, does not capture training-seed uncertainty, and does not justify freezing the fusion. B's 2000-step models cover about 26.7% of train rows, so neither is a full-data final model.

## Error Notes

C made 319 human-to-machine false positives and 428 machine-to-human false negatives. Under the temporary short-text rule, 81 of 1,841 short samples were false positives and 38 were false negatives; short-subset macro-F1 was 0.918468. The lower short-subset human recall (0.8451 versus machine recall 0.9712) is a useful follow-up target.

Descriptive domain slices with at least several hundred samples included `RuATD_2022` (1,202 samples, macro-F1 0.8749) and `PeerRead/iclr_2017/` (1,266, macro-F1 0.9009). Generator slices contain only one true class, so macro-F1 is not meaningful there; use the class-specific miss rate instead. Among larger machine-generator slices, `dolly-v2` had 34 misses out of 216 (15.7%), while `gpt-3.5-turbo` had 130/2,660 (4.9%) and `text-davinci-003` had 105/1,640 (6.4%). Slice observations are descriptive, not held-out generator-generalization claims.

The C/B errors are not identical: C-only-correct versus B-only-correct counts were 1,326/494 (CE500), 1,116/557 (CE2000 seed 42), and 1,247/575 (CE2000 seed 43). This supports testing a blend, but not treating the dev-selected result as unbiased performance.

## Limitations and Next Decision

- A must confirm official short-text counting, probability threshold, and output CSV specification. Recompute comparisons under the confirmed rule.
- Ask B to verify loading the C `joblib` artifact in B's runtime before integration; it was serialized with Python 3.14 and scikit-learn 1.9.1.
- The A preprocessing script prints two example texts from each split, including internal test examples. This run did not use test for model fitting, tuning, evaluation, or analysis; ask A to suppress test-example printing in future preprocessing runs.
- Results were generated from uncommitted C working-tree code. Do not treat the current `git_revision` in run metadata as a pin for the exact source snapshot; rerun after the C code is committed if a strictly pinned reproduction is required.