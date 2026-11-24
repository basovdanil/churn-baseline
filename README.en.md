# churn-baseline

<p align="center">
  <img src="https://raw.githubusercontent.com/basovdanil/basovdanil/main/assets/tools.png"
       alt="Eight tools with tests and exit codes" width="100%">
</p>

[![tests](https://github.com/basovdanil/churn-baseline/actions/workflows/tests.yml/badge.svg)](https://github.com/basovdanil/churn-baseline/actions/workflows/tests.yml)

Churn prediction on an imbalanced dataset (26,5 % churn): stratified cross-validation of three
models, a decision threshold tuned on out-of-fold predictions, and full-base scoring. Logistic
regression wins on AUC — **0,8446 ± 0,0132** at threshold 0,57.

The interesting part is not the AUC: with 26,5 % positives the default 0,5 threshold is the wrong
tool, so the threshold is chosen on OOF probabilities and the precision/recall trade-off is reported.

## Quick start

```
python churn.py fetch      # download the dataset and verify its sha256
python churn.py train      # CV of three models, work/metrics.json + model.pkl
python churn.py predict --holdout 0.2
python churn.py check
```

## Results

| Model | AUC | Precision | Accuracy | Threshold |
| --- | --- | --- | --- | --- |
| LogisticRegression (balanced) | **0,8446 ± 0,0132** | **0,6540** | 0,7453 | 0,57 |
| RandomForest, 400 trees | 0,8411 ± 0,0123 | 0,6477 | 0,7796 | 0,42 |
| Gradient boosting | 0,8288 ± 0,0091 | 0,6310 | 0,7599 | 0,33 |

## Exit codes

`0` done, `2` bad arguments, `3` file missing, `4` data or artifact error.

## Tests

```
python -m unittest discover -s tests
```

19 tests: CSV parsing, imbalance handling, threshold search, metric arithmetic, artifact checks.

## Scope

Public Telco dataset (7 043 rows, verified by sha256). The model is a baseline, not a production
scorer: no feature store, no monitoring.

---

More projects: [github.com/basovdanil](https://github.com/basovdanil) · write-ups: [portfolio](https://basovdanil.github.io/portfolio/)
