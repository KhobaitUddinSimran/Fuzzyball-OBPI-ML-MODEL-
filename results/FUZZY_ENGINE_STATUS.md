# Fuzzy Engine Status

## Completed

- Built `src/obpi/fuzzy/membership.py` with trapezoidal membership functions.
- Built `src/obpi/fuzzy/engine.py` with Mamdani-style SISO rule firing, centroid defuzzification, and `[0, 1]` range correction.
- Added batch scoring in `src/obpi/fuzzy/scoring.py`.
- Added CLI entrypoint: `python3 -m obpi.fuzzy.score`.
- Added synthetic development data in `data/sample/sample_metrics.csv`.
- Added unit tests in `tests/test_fuzzy.py`.

## Verified

```text
python3 -m unittest discover -s tests
Ran 12 tests
OK
```

The sample scorer writes `results/sample_obpi_scores.csv`.

## Dataset Dependency

The real StatsBomb-derived metric dataset is not required to continue coding the fuzzy engine, but it is required before claiming calibrated or validated OBPI results.

Required final input shape:

```text
player_id, match_id, minutes, M1, M2, M3, M4, M5, M6, M7, M8, M9
```

All metric columns should be numeric and normalized to `[0, 1]`.

Real data becomes necessary for:

- empirical membership calibration,
- OBPI vs minutes sanity checks,
- real player ranking inspection,
- ML validation,
- SHAP importance,
- ablation,
- benchmark correlation,
- expert correlation.

## Week 6 Progress

Implemented:

- `src/obpi/ml/validation.py`
- `create_labels(obpi_scores)` using top/bottom quartiles.
- `prepare_labeled_data(metrics_df)` for M1-M9 plus `obpi_score`.
- Logistic-regression baseline with `StandardScaler`.
- RBF SVM grid search with `StandardScaler`.
- Stratified cross-validation evaluation with accuracy, ROC-AUC, and class-1 recall.
- Optional `train_xgboost()` wrapper that requires the `xgboost` dependency.
- `validate(metrics_df)` report API for later FastAPI integration.
- `tests/test_ml.py`.

Verified:

```text
python3 -m unittest discover -s tests
Ran 19 tests
OK
```

Synthetic report output:

- `results/cv_results.json`

Important: the current `cv_results.json` is generated from synthetic sample data only. It proves the validation code runs; it does not prove real tactical signal.
