# Research Validation Audit

## Data Coverage

- processed_player_match_rows: 549
- processed_players: 251
- processed_matches: 216
- rows_with_360_data: 549
- frame_count_mean: 24.825136612021858
- frame_count_min: 1.0
- frame_count_max: 25.0
- manifest_matches: 2650
- matches_with_three_sixty_file: 300
- matches_with_freeze_frames: 293
- freeze_frame_events: 976692

## Target Population

- Center Attacking Midfield: 289
- Right Attacking Midfield: 135
- Left Attacking Midfield: 125
- note: Current validation subset is attacking-midfield roles with 360 coverage.

## Model Validation

- samples: 274
- class_counts: {'0': 137, '1': 137}
- best_accuracy_model: xgboost
- logistic accuracy: 0.9890
- logistic roc_auc: 1.0000
- logistic recall_class_1: 0.9854
- svm accuracy: 0.9927
- svm roc_auc: 1.0000
- svm recall_class_1: 0.9854
- xgboost accuracy: 0.9927
- xgboost roc_auc: 1.0000
- xgboost recall_class_1: 0.9929

## Explainability

- model_name: xgboost
- shap_available: True
- permutation_top_5: ['M9', 'M7', 'M6', 'M2', 'M5']
- mean_abs_shap_top_5: [('M9', 1.5035311660729926), ('M6', 1.25260427429927), ('M7', 1.18237962235073), ('M2', 1.019667589989051), ('M5', 0.9793859519393068)]

## Validity Status

- pipeline_validation: complete
- construct_validation: internal_obpi_extreme_quartile_only
- robustness_validation: complete
- external_validation: complete
- interpretation: Use current scores as 360-enriched internal validation evidence. Do not present them as final independent convergent validity unless external_validation is complete.

## Robustness Validation

- status: complete
- xgboost_accuracy_range: 6.734006734010478e-05
- cap 25: rows=549, samples=274, xgboost_accuracy=0.9927272727272728
- cap 50: rows=549, samples=274, xgboost_accuracy=0.9926599326599327
- cap 75: rows=549, samples=274, xgboost_accuracy=0.9927272727272728

## External Validation

- status: complete

## Match vs Aggregate Validation

- match_level_best_model: xgboost
- match_level_samples: 274
- aggregate_level_best_model: logistic
- aggregate_level_samples: 126

## Next Validation Requirements

