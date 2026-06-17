# Frame-Cap Sensitivity

This report reruns the 360-backed attacking-midfield validation subset with different per-match freeze-frame caps.

| Frame cap | Rows | Players | Matches | Samples | XGBoost accuracy | SVM accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 549 | 251 | 216 | 274 | 0.9927 | 0.9927 |
| 50 | 549 | 251 | 216 | 274 | 0.9927 | 0.9927 |
| 75 | 549 | 251 | 216 | 274 | 0.9927 | 0.9854 |

## Interpretation

Stable accuracy across caps supports robustness to 360 sampling. Large swings mean the metric processor should be rerun with a larger cap for final research reporting.
