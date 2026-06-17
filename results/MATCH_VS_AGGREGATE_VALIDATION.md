# Match vs Aggregate Validation

Aggregate validation tests whether the signal persists when repeated player-match rows are collapsed to one row per player.

## Summary

| Level | Samples | Best model | Best accuracy |
|---|---:|---|---:|
| Player-match | 274 | xgboost | 0.9927 |
| Aggregate player | 126 | logistic | 0.9920 |
