# Fuzzyball-OBPI-ML-MODEL-
# OBPI Model: Quantifying Off-Ball Positional Intelligence in Attacking Midfielders

An advanced sports data science framework that uses **StatsBomb 360 Freeze-Frame Data**, **Mamdani Fuzzy Inference Systems (FIS)**, and **Support Vector Machines (SVM)** to mathematically grade space creation, temporal control (*La Pausa*), and positioning in elite attacking midfielders.

---

## 🏗️ 1. Pipeline Architecture
### Dataset Filters
* **Position:** AM, CAM, #10 | **Minutes:** ≥ 900 min/season | **Window:** 2013/14 – 2023/24
* **Broadcast Fix:** Voronoi clipping bound dynamically to the `ConvexHull` of visible players to fix camera blind spots.
* **Sampling Fix:** Metrics omit tracking velocity vectors if event time delta $\Delta t > 1.5\text{s}$ to fix sparse sampling errors.

---

## 📐 2. The 9-Metric Framework & Formulas

### Dimension A: Spatial Manipulation
* **M1: Screening Coefficient (SC)**
  $$SC = \frac{N_{\text{shifts} > 1.5m}}{\text{Total Off-Ball Possessions in Screening Zone}}$$
  *Derivation:* Uses an adjusted shift metric to isolate movement from a globally dropping defensive block:
  $$\Delta C_{\text{adj}} = \|\text{mean}(D_{\text{local, after}}) - \text{mean}(D_{\text{local, before}})\|_2 - \|\text{mean}(D_{\text{all, after}}) - \text{mean}(D_{\text{all, before}})\|_2$$

* **M7: Space Creation Index (SCI)**
  $$\text{SCI} = \frac{\sum (A_{\text{after}, k} - A_{\text{before}, k})}{N_{\text{actions}}}$$
  *Derivation:* Net change in usable pitch territory using `scipy.spatial.Voronoi` clipped tightly to the frame's visible convex hull.

### Dimension B: Movement Quality
* **M2: Off-Ball Impact Run Coefficient (OIRC)**
  $$\text{OIRC} = \frac{\sum (\Delta C_{k} \times RD_{k})}{N_{\text{runs}}}$$
  *Derivation:* $RD_{k}$ is the Run Directness Score, calculated via cosine similarity to opponent goal center:
  $$RD_{k} = \max\left(0, \frac{v_{\text{run}} \cdot v_{\text{goal}}}{\|v_{\text{run}}\| \times \|v_{\text{goal}}\|}\right)$$

* **M4: Off-Ball Runs per 90 (OBR90)**
  $$\text{OBR90} = \left(\frac{N_{\text{runs}}}{T_{\text{minutes}}}\right) \times 90$$
  *Derivation:* Volume of out-of-possession sprints where inferred velocity $v_k > 2.5\text{m/s}$ for $\ge 0.4\text{s}$.

### Dimension C: Receiving Intelligence
* **M3: Best Receiving Position Coefficient (BRPC)**
  $$\text{BRPC} = \frac{N_{(\text{pressure} > 5m) \cap (\text{lane open})}}{N_{\text{total receipts}}}$$
  *Derivation:* Proportion of receipts where nearest defender $d > 5\text{m}$ and a $45^\circ$ forward passing lane cone is completely open.

* **M5: Receipts Between the Lines (RBTL)**
  $$\text{RBTL} = \frac{N_{\text{receipts in half-spaces}}}{N_{\text{total receipts}}}$$

* **M6: Receipts Under Pressure (RUP)**
  $$\text{RUP} = \frac{N_{\text{receipts with under\_pressure = True}}}{N_{\text{total receipts}}}$$

### Dimension D: Temporal Control & Communication
* **M8: La Pausa Coefficient (LPC)**
  $$\text{LPC} = \frac{N_{\text{pause actions that improve next pass xT}}}{N_{\text{total pause opportunities}}}$$
  *Derivation:* Actions where ball retention $\ge 1.2\text{s}$, player velocity drops $< 0.5\text{m/s}$, local defender decelerates, and subsequent pass output increases Expected Threat ($xT$).

* **M9: Call-for-Ball Index (CBI)**
  $$\text{CBI} = \frac{N_{\text{call-for-ball movements opening a lane}}}{N_{\text{receipt opportunities}}}$$
  *Derivation:* Run vector directed within $30^\circ$ of ball carrier opening a lane with a $\ge 1.5\text{m}$ defender safety buffer.

---

## ⚙️ 3. Fuzzy Logic & Model Specifications

### Stage 3: Mamdani FIS Engine
1. **Fuzzification:** Crisp metrics mapped to triangular membership functions (`trimf`):
   $$\mu_{\text{Low}}(x) = \text{trimf}(x; 0, 0, 0.4), \quad \mu_{\text{Med}}(x) = \text{trimf}(x; 0.2, 0.5, 0.8), \quad \mu_{\text{High}}(x) = \text{trimf}(x; 0.6, 1.0, 1.0)$$
2. **Rule Base:** Parallel firing mapping metric state to matching performance output thresholds.
3. **Defuzzification:** Discrete Centroid method (Center of Gravity) translates shapes back to a scalar score:
   $$x^* = \frac{\sum_{i} [\mu_{\text{agg}}(x_i) \cdot x_i]}{\sum_{i} \mu_{\text{agg}}(x_i)}$$

### Stage 4: Hyperplane Validation Layer (SVM)
The machine learning layer works strictly as a **discriminant validation instrument** to prove the engineered feature dimensions cleanly isolate player classes.
* **No Label Leakage:** Composite scores are sorted. Top 25% are labeled Class 1 (High), bottom 25% are labeled Class 0 (Low). The middle 50% is discarded to prevent mathematical tautology.
* **Variance Restoration:** Passing defuzzified metrics through a `StandardScaler` is mandatory to correct for the range compression ($[0.15, 0.85]$) inherent to centroid defuzzification.
* **Optimization:** Radial Basis Function (RBF) Kernel handles non-linear relationships across the 9D space. High Stratified 5-Fold cross-validation accuracy confirms tactical signal dominance over data noise.

---

## 🛠️ 4. Setup & Python Control Engine Initialization

```bash
pip install statsbombpy scikit-fuzzy scipy scikit-learn pandas numpy
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

universe = np.arange(0, 1.01, 0.01)
metric_in = ctrl.Antecedent(universe, 'metric_input')
score_out = ctrl.Consequent(universe, 'score_output')

for var in [metric_in, score_out]:
    var['Low'] = fuzz.trimf(universe, [0.0, 0.0, 0.4])
    var['Medium'] = fuzz.trimf(universe, [0.2, 0.5, 0.8])
    var['High'] = fuzz.trimf(universe, [0.6, 1.0, 1.0])

rule1 = ctrl.Rule(metric_in['Low'], score_out['Low'])
rule2 = ctrl.Rule(metric_in['Medium'], score_out['Medium'])
rule3 = ctrl.Rule(metric_in['High'], score_out['High'])

obpi_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem([rule1, rule2, rule3]))
```

---

## Current Implementation Status

### Fuzzy engine

The repository now includes a first working fuzzy scoring layer under `src/obpi/fuzzy/`.

Implemented:
- `MembershipFunction` and trapezoidal `Low` / `Medium` / `High` membership functions.
- Data-calibrated percentile memberships using P20, P40, P50, P60, and P80.
- `FuzzyEngine` for M1-M9 inputs with centroid defuzzification and range correction to `[0, 1]`.
- Uniform metric aggregation today, with a `metric_weights` hook ready for SHAP-derived weights later.
- DataFrame and CSV scoring helpers.
- Synthetic development data in `data/sample/sample_metrics.csv`.

Run the sample scorer:

```bash
PYTHONPATH=src python3 -m obpi.fuzzy.score data/sample/sample_metrics.csv results/sample_obpi_scores.csv
```

Run tests:

```bash
python3 -m unittest discover -s tests
```

### When real data becomes necessary

Synthetic data is enough for engine development, CLI wiring, and edge-case tests. Real processed metric data becomes necessary at these points:

- Final membership calibration: use real M1-M9 distributions to set P20/P40/P50/P60/P80.
- Week 5 sanity checks: plot OBPI against minutes and inspect player rankings.
- Week 6 ML validation: SVM/XGBoost scores are only meaningful on real metric rows.
- Week 7 SHAP: feature importance from synthetic data should not be interpreted.
- Week 9 ablation and benchmarking: requires real metric signal and external benchmark data.
- Week 10 expert correlation: requires expert panel ratings matched to OBPI rows.

### Week 6 validation

The validation suite lives in `src/obpi/ml/validation.py`.

Implemented:
- `create_labels(obpi_scores)`: top 25% = class 1, bottom 25% = class 0, middle 50% discarded.
- `train_logistic(X, y)`: standardized logistic-regression baseline.
- `train_svm(X, y)`: standardized RBF SVM grid search.
- `train_xgboost(X, y)`: optional wrapper requiring the `xgboost` dependency.
- `validate(metrics_df)`: report API returning class counts and cross-validation metrics.

Example:

```python
import pandas as pd
from obpi.ml.validation import validate

metrics_df = pd.read_csv("results/sample_obpi_scores.csv")
report = validate(metrics_df, include_xgboost=False)
```

`results/cv_results.json` currently uses synthetic sample data and is only a pipeline smoke test.
