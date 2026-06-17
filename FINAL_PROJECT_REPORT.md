# Final Project Report: Off-Ball Positional Intelligence Model

## Title

Quantifying Off-Ball Positional Intelligence in Attacking Midfielders Using StatsBomb 360 Data, Fuzzy Logic, and Machine Learning Validation

## Executive Summary

This project develops an Off-Ball Positional Intelligence (OBPI) scoring pipeline for attacking midfielders. The objective is to measure tactical qualities that are often visible in football analysis but difficult to capture with conventional event statistics, such as space creation, receiving intelligence, off-ball movement, temporal control, and communication through movement.

The final pipeline uses StatsBomb open data and StatsBomb 360 freeze-frame data to compute nine engineered metrics for attacking-midfield roles. These metrics are normalized and passed through a Mamdani-style fuzzy scoring layer to produce a scalar OBPI score in the range 0 to 1. The resulting score is validated internally through machine learning classification, robustness checks, explainability analysis, aggregate-player validation, and externally against FIFA 23 player ratings.

The latest validation run confirms that the engineered metric space cleanly separates high-OBPI and low-OBPI player-match samples. XGBoost achieved a 5-fold cross-validation accuracy of 0.9927 on the balanced extreme-quartile validation set. Aggregate-player validation also remained strong, with Logistic Regression reaching 0.9920 accuracy after collapsing repeated player-match rows into one row per player.

External validation against FIFA 23 ratings matched 230 of 252 OBPI players, a 91.3 percent match rate. FIFA correlations were weak, with FIFA overall almost uncorrelated with OBPI. This suggests that OBPI is not simply reproducing general player-quality ratings. Instead, it appears to capture a distinct off-ball tactical construct.

## Research Problem

Traditional football analytics often focuses on on-ball actions: passes, shots, carries, duels, pressures, and expected threat. However, attacking midfielders frequently create value before they touch the ball. They manipulate defensive shape, position themselves between lines, offer passing lanes, time runs, pause play, and communicate intent through movement.

The research problem is therefore:

Can off-ball positional intelligence for attacking midfielders be quantified using event and freeze-frame data, and can the resulting metric be validated as a meaningful tactical signal?

This project answers the question by building a complete OBPI pipeline and validating it through both internal discrimination tests and external benchmark comparison.

## Objectives

The project had five main objectives:

1. Build a reproducible data pipeline for StatsBomb open data and 360 freeze-frame data.
2. Engineer a nine-metric OBPI framework for attacking-midfield roles.
3. Combine the metrics into a fuzzy OBPI score using interpretable fuzzy membership logic.
4. Validate the score through model accuracy, robustness, explainability, and aggregate-player testing.
5. Compare OBPI with an external benchmark, using FIFA 23 player ratings as a commercial player-quality reference.

## Data Sources

### StatsBomb Open Data

The core football data comes from StatsBomb open data. The local acquisition pipeline downloads competition metadata, matches, events, lineups, and available 360 freeze-frame files.

Raw data is intentionally kept local and ignored by Git because it is large. Processed and validation-ready artifacts are stored in the repository.

Relevant pipeline scripts:

- `scripts/download_statsbomb_mens_open_data.py`
- `scripts/preprocess_statsbomb_open_data.py`
- `scripts/process_interim_metrics.py`

### StatsBomb 360 Freeze-Frame Data

The final validation subset uses attacking-midfield player-match rows with 360 coverage. This matters because several OBPI metrics depend on visible teammate/opponent positions, passing lanes, pressure distances, and space structure.

Current processed coverage:

| Item | Value |
|---|---:|
| Processed player-match rows | 549 |
| Processed players | 251 |
| Processed matches | 216 |
| Rows with 360 data | 549 |
| Manifest matches scanned | 2,650 |
| Matches with 360 files | 300 |
| Matches with freeze frames | 293 |
| Freeze-frame events | 976,692 |

### FIFA 23 Ratings

External benchmark validation uses the Kaggle FIFA 23 Complete Player Dataset:

https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset

The raw Kaggle CSV is kept local-only. The repository stores only the small matched benchmark files:

- `data/external/fifa_ratings.csv`
- `data/external/fifa_ratings_match_audit.csv`
- `results/fifa_external_validation.json`
- `results/FIFA_EXTERNAL_VALIDATION.md`

## Target Population

The current research subset focuses on attacking-midfield roles with 360 coverage.

| Position | Rows |
|---|---:|
| Center Attacking Midfield | 289 |
| Right Attacking Midfield | 135 |
| Left Attacking Midfield | 125 |

This target population fits the research purpose because the model is designed around attacking midfielders, not all outfield roles.

## OBPI Metric Framework

The OBPI framework is built from nine metrics grouped into four tactical dimensions.

### Spatial Manipulation

**M1: Screening Coefficient (SC)**  
Measures defensive displacement caused by off-ball positioning or screening movement.

**M7: Space Creation Index (SCI)**  
Measures change in usable space around the player, using freeze-frame spatial structure.

### Movement Quality

**M2: Off-Ball Impact Run Coefficient (OIRC)**  
Measures directness and tactical value of off-ball runs toward dangerous areas.

**M4: Off-Ball Runs per 90 (OBR90)**  
Measures volume of qualifying off-ball runs standardized per 90 minutes.

### Receiving Intelligence

**M3: Best Receiving Position Coefficient (BRPC)**  
Measures whether receipts occur with pressure separation and open passing lanes.

**M5: Receipts Between the Lines (RBTL)**  
Measures how often the player receives in valuable between-line or half-space zones.

**M6: Receipts Under Pressure (RUP)**  
Measures ability or tendency to receive while under defensive pressure.

### Temporal Control and Communication

**M8: La Pausa Coefficient (LPC)**  
Measures pause actions that improve the next passing or threat outcome.

**M9: Call-for-Ball Index (CBI)**  
Measures movement that signals availability and opens a receiving lane.

## Methodology

### Data Acquisition

The project downloads men's StatsBomb open data into a raw local folder. Raw files are not committed to Git because of their size. The acquisition stage collects:

- `competitions.json`
- match files
- event files
- lineup files
- 360 freeze-frame files where available

### Preprocessing

The preprocessing step converts JSON data into analysis-ready parquet tables. It creates competition, match, player-match, and event-manifest tables, and partitions events by match.

When 360 freeze-frame files are available, they are merged into the event-level processing flow and tracked in the manifest.

### Metric Processing

The metric processing stage filters the attacking-midfielder subset and computes the nine OBPI metrics. The current validation subset uses a 360-backed sample with a bounded frame cap.

Current metric-processing artifact:

- `data/processed/player_match_metrics.parquet`

### Fuzzy OBPI Scoring

The fuzzy layer consumes normalized M1-M9 metrics and produces a scalar OBPI score. The fuzzy engine uses low, medium, and high membership regions for metric interpretation, then combines metric evidence into a final score.

Current scored artifact:

- `data/processed/player_obpi_scores.parquet`

OBPI score distribution:

| Statistic | Value |
|---|---:|
| Count | 549 |
| Mean | 0.5040 |
| Standard deviation | 0.1194 |
| Minimum | 0.1111 |
| 25th percentile | 0.4180 |
| Median | 0.5013 |
| 75th percentile | 0.5895 |
| Maximum | 0.8178 |

### Training Preparation

For machine learning validation, the model uses the top and bottom OBPI quartiles:

- Top quartile: high OBPI class
- Bottom quartile: low OBPI class
- Middle 50 percent: discarded

This creates a balanced discriminant-validation dataset.

| Item | Value |
|---|---:|
| Input scored rows | 549 |
| Prepared rows | 274 |
| Class 0 count | 137 |
| Class 1 count | 137 |
| Cross-validation folds | 5 |

Important note: these labels are derived from OBPI quartiles. They are useful for testing whether the engineered metric space separates high and low OBPI examples, but they are not independent ground-truth labels.

## Model Validation

The validation layer trains three classifiers on all nine OBPI metrics:

- Logistic Regression
- Support Vector Machine
- XGBoost

The purpose is not to replace the fuzzy score with a black-box model. The models are used as validation instruments to test whether M1-M9 separate high and low OBPI samples.

### Cross-Validation Accuracy

| Model | Accuracy Mean | Accuracy Std | ROC-AUC Mean | Recall Class 1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9890 | 0.0090 | 1.0000 | 0.9854 |
| SVM | 0.9927 | 0.0090 | 1.0000 | 0.9854 |
| XGBoost | 0.9927 | 0.0089 | 1.0000 | 0.9929 |

Best model by accuracy: XGBoost.

The high accuracy shows that the metric space strongly separates the top and bottom OBPI quartiles.

## Aggregate-Player Validation

To test whether the signal survives beyond repeated player-match rows, the project also collapses player-match rows into aggregate player rows.

| Level | Samples | Best Model | Best Accuracy |
|---|---:|---|---:|
| Player-match | 274 | XGBoost | 0.9927 |
| Aggregate player | 126 | Logistic Regression | 0.9920 |

The aggregate-player result supports the view that the OBPI signal is not only a match-row artifact. It remains separable when player-match rows are collapsed into player-level summaries.

## Robustness Validation

A frame-cap sensitivity test was run at 25, 50, and 75 freeze frames per match. This checks whether model accuracy is sensitive to the number of freeze frames sampled.

| Frame Cap | Rows | Prepared Rows | Logistic Accuracy | SVM Accuracy | XGBoost Accuracy |
|---:|---:|---:|---:|---:|---:|
| 25 | 549 | 274 | 0.9890 | 0.9927 | 0.9927 |
| 50 | 549 | 274 | 0.9855 | 0.9927 | 0.9927 |
| 75 | 549 | 274 | 0.9854 | 0.9854 | 0.9927 |

The XGBoost accuracy range across these settings is extremely small, approximately 0.00007. This indicates that the validation result is stable across the tested frame caps.

## Explainability

Explainability artifacts were generated using XGBoost metric weights, permutation importance, and SHAP values.

### Permutation Importance Ranking

| Rank | Metric | Importance |
|---:|---|---:|
| 1 | M9: Call-for-Ball Index | 0.0119 |
| 2 | M7: Space Creation Index | 0.0056 |
| 3 | M6: Receipts Under Pressure | 0.0036 |
| 4 | M2: Off-Ball Impact Run Coefficient | 0.0024 |
| 5 | M5: Receipts Between the Lines | 0.0007 |
| 6 | M4: Off-Ball Runs per 90 | 0.0001 |
| 7 | M3: Best Receiving Position Coefficient | 0.0000 |
| 8 | M1: Screening Coefficient | 0.0000 |
| 9 | M8: La Pausa Coefficient | 0.0000 |

### XGBoost Metric Weights

| Metric | Weight |
|---|---:|
| M9 | 0.4888 |
| M7 | 0.2306 |
| M6 | 0.1496 |
| M2 | 0.0970 |
| M5 | 0.0298 |
| M4 | 0.0039 |
| M3 | 0.0002 |
| M1 | 0.0000 |
| M8 | 0.0000 |

The highest-ranked features are M9, M7, M6, and M2. This suggests that signaling availability, creating space, receiving under pressure, and impactful off-ball runs are the most discriminative components in the current validation subset.

The low importance of M1 and M8 should not be interpreted as permanent evidence that these concepts are unimportant. It may reflect the current open-data limitations, sparse event coverage for certain behaviors, and the difficulty of detecting subtle temporal actions from event and freeze-frame data alone.

## External FIFA Benchmark Validation

The project validates OBPI against FIFA 23 ratings as an external commercial player-quality benchmark.

The matching process uses exact name matching, token containment, fuzzy name matching, and team/nationality context to map StatsBomb player names to FIFA player rows.

| Item | Value |
|---|---:|
| OBPI players | 252 |
| Matched FIFA players | 230 |
| Unmatched players | 22 |
| Match rate | 0.913 |
| Exact matches | 196 |
| Token-containment matches | 31 |
| Fuzzy matches | 3 |

### FIFA Correlation Results

| FIFA Benchmark | Spearman rho | p-value | n |
|---|---:|---:|---:|
| FIFA shooting | 0.1024 | 0.1214 | 230 |
| FIFA pace | 0.0925 | 0.1620 | 230 |
| FIFA potential | 0.0719 | 0.2778 | 230 |
| FIFA physic | 0.0510 | 0.4417 | 230 |
| FIFA reactions | 0.0129 | 0.8452 | 230 |
| FIFA overall | 0.0005 | 0.9943 | 230 |
| FIFA dribbling | -0.0116 | 0.8616 | 230 |
| FIFA vision | -0.1076 | 0.1036 | 230 |
| FIFA passing | -0.1249 | 0.0586 | 230 |
| FIFA defending | -0.1456 | 0.0273 | 230 |

The FIFA validation shows weak correlations between OBPI and FIFA attributes. FIFA overall is almost exactly uncorrelated with OBPI.

This is an important result. It means the OBPI model is not simply reconstructing general player quality. Instead, it captures a narrower tactical signal related to off-ball behavior.

## Results Summary

The project is complete as a working technical and validation pipeline.

Main findings:

1. The pipeline successfully computes nine OBPI metrics for a 360-backed attacking-midfielder subset.
2. Fuzzy scoring produces a stable OBPI score distribution.
3. The top and bottom OBPI quartiles are highly separable using M1-M9.
4. XGBoost reached 0.9927 cross-validation accuracy on the balanced validation set.
5. Aggregate-player validation remained strong at 0.9920 best accuracy.
6. Frame-cap sensitivity tests show stable XGBoost performance across 25, 50, and 75 frame caps.
7. Explainability identifies M9, M7, M6, and M2 as the strongest current drivers.
8. FIFA validation is complete, but correlations are weak, supporting the claim that OBPI measures a distinct tactical construct.

## Key Figures

The repository includes generated figures that can be used in a presentation or written submission:

- `results/figures/model_accuracy.png`
- `results/figures/match_vs_aggregate_accuracy.png`
- `results/figures/frame_cap_sensitivity.png`
- `results/figures/permutation_importance.png`
- `results/figures/shap_summary.png`
- `results/figures/metric_weights.png`
- `results/figures/metric_distributions.png`
- `results/figures/obpi_distribution.png`
- `results/figures/benchmark_correlations.png`

## Limitations

The project has several important limitations.

First, the validation labels are internally derived from OBPI quartiles. This is appropriate for testing discriminative separation, but it is not the same as validating against independent expert labels.

Second, the external FIFA benchmark is a general player-quality rating system. It does not directly measure off-ball intelligence. Weak correlation with FIFA should therefore be interpreted carefully. It does not invalidate OBPI; it suggests construct difference.

Third, StatsBomb open data and 360 data are not complete for every match or competition. The current validation subset is limited to rows with 360 coverage and attacking-midfield role labels.

Fourth, some subtle concepts, especially La Pausa and screening behavior, are difficult to infer perfectly from event and freeze-frame data without full tracking data.

Fifth, the high internal model accuracy should not be overstated as real-world prediction accuracy. It measures separation between OBPI-defined high and low classes, not prediction of independent match outcomes.

## Future Work

Recommended next steps:

1. Collect independent expert ratings for a sample of attacking midfielders and compare them with OBPI.
2. Validate OBPI against role-specific scouting reports rather than broad FIFA attributes.
3. Expand the data sample when more 360 or tracking data becomes available.
4. Improve M1 and M8 detection with richer temporal tracking or video-derived annotations.
5. Test OBPI season-level stability across multiple competitions and years.
6. Compare OBPI with other public analytics metrics such as expected threat, progressive receptions, possession value, and packing-style measures.
7. Build player comparison dashboards for scouting use.

## Reproducibility

The core validation can be rerun with:

```bash
python3 scripts/prepare_training_data.py
python3 scripts/train_validation_models.py --include-xgboost
python3 scripts/compare_match_aggregate_validation.py --include-xgboost
python3 scripts/run_fifa_ratings_validation.py
python3 scripts/run_research_validation_audit.py --external-report results/fifa_external_validation.json
```

Focused validation tests can be run with:

```bash
python3 -m pytest --no-cov \
  tests/test_training_preparation.py \
  tests/test_model_training.py \
  tests/test_fifa_ratings_validation.py \
  tests/test_research_validation.py \
  tests/test_ablation_correlation.py
```

Latest focused test result:

```text
15 passed
```

## Repository Artifacts

Important artifacts:

| Artifact | Purpose |
|---|---|
| `data/processed/player_match_metrics.parquet` | Processed raw M1-M9 metric values |
| `data/processed/player_obpi_scores.parquet` | Normalized M1-M9 plus OBPI score |
| `data/processed/training_prepared.parquet` | Balanced validation frame |
| `results/cv_results.json` | Main model validation results |
| `results/VALIDATION_REPORT.md` | Human-readable model validation report |
| `results/match_vs_aggregate_validation.json` | Match vs aggregate validation |
| `results/FIFA_EXTERNAL_VALIDATION.md` | FIFA benchmark validation report |
| `results/research_validation_audit.json` | Consolidated research audit |
| `results/RESEARCH_VALIDATION.md` | Human-readable research validation summary |
| `results/explainability_report.json` | Metric importance and SHAP availability |

## Conclusion

The OBPI project successfully implements a complete pipeline for quantifying off-ball positional intelligence in attacking midfielders. It starts from StatsBomb open data and 360 freeze frames, computes nine tactical metrics, produces a fuzzy OBPI score, and validates the score through multiple internal and external checks.

The strongest result is the model validation: all nine metrics together separate high and low OBPI samples with very high cross-validation accuracy. The aggregate-player test confirms that the signal remains strong after collapsing repeated match rows. The robustness test shows stable accuracy across freeze-frame sampling caps. Explainability identifies call-for-ball movement, space creation, pressure receiving, and off-ball impact runs as the strongest current drivers.

The FIFA benchmark validation adds an important research nuance. OBPI does not strongly correlate with FIFA overall or most FIFA attributes. This supports the interpretation that OBPI is measuring a distinct tactical quality rather than simply reproducing generic player ratings.

Overall, the project is complete as a technical pipeline and validation framework. The main remaining research improvement would be to collect independent expert off-ball intelligence ratings and use them as direct external ground truth.
