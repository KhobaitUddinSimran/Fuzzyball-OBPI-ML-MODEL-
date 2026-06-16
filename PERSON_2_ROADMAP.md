# Person 2 Roadmap: Fuzzy Logic & ML Validation Lead

> Your day-by-day execution plan for the Mamdani Fuzzy Inference System, ML validation layer, and statistical analysis.  
> **Branch:** `fuzzy-ml-validation`  
> **Timeline:** 6 weeks (Weeks 5–10)  
> **Deliverable:** Fuzzy engine producing OBPI scores in [0,1], SVM/XGBoost CV accuracy > 75%, SHAP importance vector, correlation report against expert ratings.

---

## Dependencies & Handoff from Person 1

Before you start Week 5, Person 1 must complete:
- [ ] `data/processed/` contains at least 5 match metric files (Parquet format)
- [ ] `notebooks/01_metric_debugging.ipynb` proves all 9 metrics work on real data
- [ ] GitHub Issue #10 exists with sample `metrics.parquet` attached

**Your input:** A DataFrame with columns `[player_id, match_id, minutes, M1, M2, ..., M9]` per match.

---

## Week 5: Fuzzy Inference Engine (Phase 3)

**Goal:** Build the Mamdani FIS that aggregates 9 crisp metrics into a single OBPI score [0,1].

| Day | Task | Files | Details |
|-----|------|-------|---------|
| **Day 29** | Pull `main` into your branch | `git checkout fuzzy-ml-validation && git merge main` | Sync latest from Person 1's merged PRs |
| | Inspect Person 1's sample data | `pd.read_parquet('data/processed/3794687_metrics.parquet')` | Understand value ranges, distributions, outliers |
| **Day 30** | Compute empirical percentiles | `notebooks/02_membership_tuning.ipynb` | P20, P40, P50, P60, P80 per metric across all sample matches |
| | Design trapezoidal membership functions | `src/obpi/fuzzy/membership.py` — `build_membership_functions(metric_values)` | `trapmf(x; 0, 0, P20, P50)` for Low, etc. |
| **Day 31** | Build Mamdani ControlSystem | `src/obpi/fuzzy/engine.py` — `FuzzyEngine` class | 9 Antecedents, 1 Consequent, 27 SISO rules |
| | Test on single player | `tests/test_fuzzy.py` | Feed [0.3, 0.5, 0.7, ...] → expect score in [0.15, 0.85] |
| **Day 32** | Implement defuzzification + range correction | `engine.py` — `defuzzify()` then `correct_range()` | Linear scaling from [0.15, 0.85] → [0, 1] |
| | Validate output range | `tests/test_fuzzy.py` | Assert all outputs in [0, 1] |
| **Day 33** | Run fuzzy engine on full sample dataset | `notebooks/02_membership_tuning.ipynb` | Plot membership functions, 2D surface responses (e.g., M7 vs M8) |
| | Sanity-check: OBPI vs minutes played | Scatter plot — elite AMs should cluster high regardless of minutes | |
| **Day 34** | Edge-case testing | `tests/test_fuzzy.py` | All metrics at 0 → score ≈ 0. All at 1 → score ≈ 1. Mixed → middle. |
| **Day 35** | **PR to `main`** | "Week 5: Mamdani fuzzy engine with data-calibrated membership functions" | **PR #6** |

**Key code to implement:**
```python
# src/obpi/fuzzy/membership.py
from skfuzzy import membership as mf

def build_membership_functions(values: np.ndarray):
    p20, p40, p50, p60, p80 = np.percentile(values, [20, 40, 50, 60, 80])
    return {
        'Low':    lambda x: mf.trapmf(x, [0, 0, p20, p50]),
        'Medium': lambda x: mf.trapmf(x, [p20, p40, p60, p80]),
        'High':   lambda x: mf.trapmf(x, [p50, p80, 1, 1])
    }

# src/obpi/fuzzy/engine.py
from skfuzzy import control as ctrl

class FuzzyEngine:
    def __init__(self, metric_names: list[str], membership_funcs: dict):
        self.antecedents = {m: ctrl.Antecedent(np.arange(0, 1.01, 0.01), m) for m in metric_names}
        self.consequent = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'obpi')
        # ... build rules ...

    def compute(self, metrics: dict) -> float:
        sim = ctrl.ControlSystemSimulation(self.control_system)
        sim.input(metrics)
        sim.compute()
        return self._correct_range(sim.output['obpi'])
```

---

## Week 6: ML Validation Suite (Phase 4, Part 1)

**Goal:** Prove the 9 engineered dimensions carry independent tactical signal via SVM + XGBoost.

| Day | Task | Files | Details |
|-----|------|-------|---------|
| **Day 36** | Label construction (no leakage) | `src/obpi/ml/validation.py` — `create_labels(obpi_scores)` | Sort by OBPI, top 25% = Class 1, bottom 25% = Class 0, discard middle 50% |
| | Test label distribution | `tests/test_ml.py` | Assert Class 1 and Class 0 each = 25% of total |
| **Day 37** | Preprocessing pipeline | `validation.py` — `StandardScaler` on all 9 metrics | Mandatory before SVM/XGBoost due to different native scales |
| **Day 38** | SVM (RBF) grid search | `validation.py` — `train_svm(X, y)` | Grid search `C` ∈ [0.1, 1, 10, 100], `gamma` ∈ ['scale', 'auto', 0.001, 0.01] |
| | Stratified 5-Fold CV | `sklearn.model_selection.StratifiedKFold(n_splits=5)` | Record accuracy per fold |
| **Day 39** | XGBoost Bayesian optimization | `validation.py` — `train_xgboost(X, y)` | `optuna` or `scikit-optimize` over depth, learning_rate, subsample |
| | Logistic regression baseline | `validation.py` — `train_logistic(X, y)` | If SVM/XGB don't outperform, features lack signal |
| **Day 40** | Evaluation report | `validation.py` — `evaluate(models, X, y)` | Accuracy, ROC-AUC, Recall@Class1 per model |
| | Target check | Assert `accuracy > 0.75`, `roc_auc > 0.80` | If not met, debug with Person 1 — may need metric refinement |
| **Day 41** | Results persistence | `results/cv_results.json`, `results/roc_curves.png` | Save all CV tables and plots |
| **Day 42** | **PR to `main`** | "Week 6: SVM + XGBoost validation suite with stratified CV" | **PR #7** |

**Key code to implement:**
```python
# src/obpi/ml/validation.py
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def create_labels(obpi_scores: pd.Series) -> pd.Series:
    q75, q25 = obpi_scores.quantile([0.75, 0.25])
    labels = pd.Series(index=obpi_scores.index, dtype=int)
    labels[obpi_scores >= q75] = 1
    labels[obpi_scores <= q25] = 0
    labels[(obpi_scores > q25) & (obpi_scores < q75)] = np.nan
    return labels.dropna().astype(int)

def train_svm(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01]}
    grid = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=5, scoring='roc_auc')
    grid.fit(X_scaled, y)
    return grid.best_estimator_, grid.best_score_
```

---

## Week 7: Explainability + SHAP (Phase 4, Part 2)

**Goal:** Derive the metric importance vector and feed it back into fuzzy aggregation weights.

| Day | Task | Files | Details |
|-----|------|-------|---------|
| **Day 43** | SHAP computation | `src/obpi/ml/explainability.py` — `compute_shap(xgb_model, X)` | `shap.TreeExplainer` on best XGBoost model |
| | Aggregate SHAP per metric | Mean absolute SHAP value per metric column | |
| **Day 44** | Visualize SHAP | `results/shap_summary.png`, `results/shap_beeswarm.png` | "LPC contributes +0.18 to High classifications..." |
| **Day 45** | Derive importance vector | `explainability.py` — `get_metric_weights(shap_values)` | Normalize SHAP importances to sum to 1.0 |
| | Feed back to fuzzy engine | Update `engine.py` to accept optional `metric_weights` | Weighted average of fuzzy outputs instead of uniform |
| **Day 46** | Re-run fuzzy with SHAP weights | Compare OBPI scores: uniform vs weighted | Document change in rankings |
| **Day 47** | Permutation importance (sanity check) | `explainability.py` — `sklearn.inspection.permutation_importance` | Validate SHAP findings |
| **Day 48** | Write `results/VALIDATION_REPORT.md` | Summary: CV accuracy, ROC-AUC, SHAP rankings, permutation check | |
| **Day 49** | **PR to `main`** | "Week 7: SHAP explainability + metric weight feedback into fuzzy engine" | **PR #8** |

**Key code to implement:**
```python
# src/obpi/ml/explainability.py
import shap

def compute_shap(model, X: pd.DataFrame) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return pd.DataFrame(shap_values, columns=X.columns)

def get_metric_weights(shap_df: pd.DataFrame) -> dict:
    importances = shap_df.abs().mean().sort_values(ascending=False)
    return (importances / importances.sum()).to_dict()
```

---

## Week 8: Phase 6 Support + Statistical Analysis

**Goal:** Support Person 3's dashboard and prepare for expert validation correlation analysis.

| Day | Task | Details |
|-----|------|---------|
| **Day 50** | Provide SHAP weights to Person 3 | Share `results/shap_summary.png` + JSON weights for dashboard radar chart |
| **Day 51** | Create `obpi.ml.validate()` API | `validation.py` — single function: `validate(metrics_df) → report_dict` |
| | Person 3 will call this from FastAPI `POST /analyze` | |
| **Day 52** | Ablation study prep | `src/obpi/ml/ablation.py` — `run_ablation(metrics_df)` stub | Train on 8 metrics, record accuracy drop |
| **Day 53** | Proxy data correlation prep | `src/obpi/ml/correlation.py` — `spearman_correlation(obpi_scores, proxy_scores)` | Expect Person 3 to provide proxy ratings |
| **Day 54** | Documentation | Update `README.md` with fuzzy + ML sections | How to run validation, interpret SHAP |
| **Day 55** | Review Person 3's FastAPI PR | Check that `POST /analyze` correctly calls `obpi.ml.validate()` | |
| **Day 56** | **PR to `main`** | "Week 8: ML API + ablation/correlation prep + documentation" | **PR #9** |

---

## Week 9: Ablation + Comparative Benchmarking

**Goal:** Prove unique signal per metric; compare OBPI against existing benchmarks.

| Day | Task | Files | Details |
|-----|------|-------|---------|
| **Day 57** | Run ablation study | `src/obpi/ml/ablation.py` | Remove each metric, train on remaining 8, record `ΔA = A_full - A_-i` |
| | Identify redundant metrics | Any `ΔA < 2%` flagged | Discuss with team before removing |
| **Day 58** | Ablation visualization | `results/ablation_plot.png` | Bar chart of accuracy drop per metric |
| **Day 59** | Collect benchmark data | Manual: StatsBomb xThreat, Football Benchmark ratings for same players | |
| **Day 60** | Comparative correlation | `correlation.py` — `compare_benchmarks(obpi, benchmarks)` | Spearman ρ between OBPI and each benchmark |
| **Day 61** | Orthogonal variance test | PCA on [OBPI, xThreat, Benchmark] | If OBPI loads on separate component → incremental value proven |
| **Day 62** | Write `results/ABLATION_BENCHMARK.md` | Report `ΔA_i` per metric, benchmark correlations, PCA results | |
| **Day 63** | **PR to `main`** | "Week 9: Ablation study + comparative benchmarking" | **PR #10** |

---

## Week 10: Expert Validation Correlation + Paper Figures

**Goal:** Correlate OBPI against expert panel ratings; generate publication-ready figures.

| Day | Task | Files | Details |
|-----|------|-------|---------|
| **Day 64** | Receive expert ratings from Person 3 | CSV: `expert_ratings.csv` [player_id, expert_1, expert_2, ..., median] | |
| **Day 65** | Compute Spearman correlation | `correlation.py` — `expert_correlation(obpi, expert_median)` | Target: ρ > 0.65, p < 0.001 |
| **Day 66** | Inter-rater reliability | `correlation.py` — `cronbach_alpha(expert_ratings_df)` | Target: α > 0.70 |
| **Day 67** | Convergent validity report | `results/CONVERGENT_VALIDITY.md` | Both proxy (Option A) and expert (Option B) correlations |
| **Day 68** | Paper figures | `results/figures/` | 1. Metric distributions, 2. Fuzzy membership functions, 3. SHAP summary, 4. ROC curves, 5. Ablation plot, 6. Expert correlation scatter |
| **Day 69** | Final `results/` packaging | Zip: all CSVs, PNGs, JSONs, Markdown reports | Ready for Person 3's paper LaTeX |
| **Day 70** | **Final PR to `main`** | "Week 10: Expert validation correlation + paper figures" | **PR #11** |

---

## Daily Rhythm

```
09:00  Pull latest main, merge into fuzzy-ml-validation
09:30  Write code / run experiments
12:00  Check in with Person 1 if metric questions arise
14:00  Continue implementation / analyze results
17:00  Run full test suite + ruff + mypy
17:30  Commit with descriptive message
18:00  Push to fuzzy-ml-validation branch
```

---

## Critical Handoffs

| When | What | To Whom |
|------|------|---------|
| End of Week 5 | SHAP metric weights (JSON) | Person 3 (for dashboard radar chart) |
| End of Week 6 | `obpi.ml.validate()` function signature | Person 3 (for `POST /analyze` endpoint) |
| End of Week 7 | Updated fuzzy engine with SHAP weights | Person 1 (for re-running pipeline) |
| End of Week 9 | Ablation results | Person 1 (to discuss redundant metrics) |
| End of Week 10 | All figures + correlation reports | Person 3 (for LaTeX paper) |

---

## Files You Will Create

| File | Purpose |
|------|---------|
| `src/obpi/fuzzy/membership.py` | Data-driven trapezoidal membership functions |
| `src/obpi/fuzzy/engine.py` | Mamdani ControlSystem builder + defuzzification |
| `src/obpi/ml/validation.py` | SVM/XGBoost/LR training, stratified CV, labeling |
| `src/obpi/ml/explainability.py` | SHAP + permutation importance |
| `src/obpi/ml/ablation.py` | Leave-one-metric-out sensitivity analysis |
| `src/obpi/ml/correlation.py` | Spearman correlation, Cronbach's alpha |
| `tests/test_fuzzy.py` | Fuzzy engine unit tests |
| `tests/test_ml.py` | ML validation tests |
| `notebooks/02_membership_tuning.ipynb` | Membership function visualization + 2D surfaces |
| `results/cv_results.json` | Cross-validation tables |
| `results/roc_curves.png` | ROC curves per model |
| `results/shap_summary.png` | SHAP bar plot |
| `results/shap_beeswarm.png` | SHAP beeswarm plot |
| `results/ablation_plot.png` | Ablation accuracy drops |
| `results/VALIDATION_REPORT.md` | Week 6–7 summary |
| `results/ABLATION_BENCHMARK.md` | Week 9 summary |
| `results/CONVERGENT_VALIDITY.md` | Week 10 summary |
| `results/figures/` | Publication-ready PNGs |

---

## Success Criteria (End of Week 10)

- [ ] `pytest` passes with ≥ 80% coverage
- [ ] `ruff` and `mypy` report zero errors
- [ ] Fuzzy engine outputs OBPI in [0, 1] for any valid 9-metric input
- [ ] SVM 5-Fold CV accuracy > 75%, ROC-AUC > 0.80
- [ ] SHAP importance vector identifies top 3 discriminating metrics
- [ ] Ablation study shows `ΔA_i >= 2%` for at least 7 of 9 metrics
- [ ] Expert correlation: Spearman ρ > 0.65, p < 0.001, Cronbach α > 0.70
- [ ] 6 PRs merged to `main`, each reviewed by at least one teammate

---

## What Comes Before & After You

| Phase | Owner | Your Interface |
|-------|-------|---------------|
| **Phase 1–2** (Data + 9 Metrics) | Person 1 | Provides `metrics.parquet` DataFrame |
| **Phase 3–4** (Fuzzy + ML) | **You** | Consumes metrics, produces OBPI + validation |
| **Phase 5** (API + Dashboard) | Person 3 | Uses your `obpi.ml.validate()` and SHAP weights |
| **Phase 6** (Paper) | Person 3 | Uses your figures + correlation reports |

---

**Start now:** `git checkout fuzzy-ml-validation && git merge main` and inspect Person 1's sample data.
