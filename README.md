# Fuzzyball-OBPI-ML-MODEL-
# OBPI Model: Quantifying Off-Ball Positional Intelligence in Attacking Midfielders

An advanced sports data science framework that uses **StatsBomb 360 Freeze-Frame Data**, **Mamdani Fuzzy Inference Systems (FIS)**, and **Support Vector Machines (SVM)** to mathematically grade space creation, temporal control (*La Pausa*), and positioning in elite attacking midfielders.

Final write-up: [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md)

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

### Installation

```bash
pip install -r requirements.txt
```

### Download full men's StatsBomb open data

To pull every men's senior competition listed in the official open-data index,
including competition metadata plus all linked match, event, and lineup files:

```bash
python scripts/download_statsbomb_mens_open_data.py --skip-existing --include-360
```

By default, files are stored in:

```text
data/raw/statsbomb_open_data/
  competitions.json
  matches/
  events/
  lineups/
  three-sixty/
```

Use `--output-dir` if you want to place the dataset somewhere else.

### Preprocess downloaded open data

Once the raw dataset is available locally, convert it into interim parquet
tables and per-match event partitions:

```bash
python3 scripts/preprocess_statsbomb_open_data.py
```

This writes:

```text
data/interim/
  competitions.parquet
  matches.parquet
  player_matches.parquet
  events_manifest.parquet
  events_by_match/
```

If `data/raw/statsbomb_open_data/three-sixty/` is present, the preprocessor
automatically merges standalone StatsBomb 360 frames into the per-match event
parquet files and records coverage in `events_manifest.parquet`.

To compute metrics only for matches with 360 coverage:

```bash
python3 scripts/process_interim_metrics.py --require-360
```

For the attacking-midfielder validation subset used by the current research
pipeline, use the 360 subset with a bounded, evenly sampled frame list:

```bash
python3 scripts/process_interim_metrics.py \
  --require-360 \
  --position-keyword "Attacking Midfield" \
  --max-frames-per-match 25
```

### Running the pipeline

```bash
python -m obpi.pipeline --match-id 3794686 --verbose
```

CLI flags:
- `--match-id` (required): StatsBomb match identifier.
- `--tier`: Data tier — `open` (default) or `api`.
- `--config`: Path to custom YAML config (default: `config/default.yaml`).
- `--output`: Cache directory for Parquet files (default: `data/processed`).
- `--verbose`: Enable debug logging.

### Training the xT model

```bash
python -m obpi.ml.xt_trainer
```

This downloads open-data shots with 360 frames, trains a logistic-regression xG model, smooths the resulting 12×8 grid with a Gaussian kernel, and saves it to `data/processed/xt_grid_12x8.npy`. The `XTModel` class loads this grid automatically when present; otherwise it falls back to a synthetic ramp.

### Configuration

All hard-coded thresholds have been moved to `config/default.yaml`:

```yaml
movement:
  v_threshold: 2.5
  duration_threshold: 0.4
  max_dt: 1.5

receiving:
  proximity_threshold: 2.5
  pressure_radius: 5.0
  cone_angle: 45.0
  cone_length: 15.0

temporal:
  min_dt: 1.2
  max_vel: 0.5
  angle_threshold: 30.0
  lane_buffer: 1.5
```

Override any value by passing a custom YAML file via `--config`.

### Validation

```python
from obpi.validation.checks import validate

result = validate(df)
assert result["valid"]  # schema, finiteness, and range checks
print(result["summary"])  # per-metric mean / std / min / max
```

### Logging

Structured logging is configured via `obpi.utils.logger.setup_logging`. When `--verbose` is passed, debug-level logs include match-level event/frame counts and cache hit/miss messages.

### Legacy fuzzy snippet

```python
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

### Fuzzy aggregation as downstream scoring

The M1-M9 metrics engine remains the source of truth. Fuzzy aggregation is an
additive downstream step that consumes an existing metrics DataFrame:

```python
from obpi.pipeline import compute_all_metrics, run_fuzzy_pipeline

metrics_df = compute_all_metrics(match_id=3794686)
scored_df = run_fuzzy_pipeline(metrics_df)
```

`run_fuzzy_pipeline()` supports both the pipeline metric schema
(`M1_SC`, `M2_OIRC`, ..., `M9_CBI`) and normalized Person 2 columns
(`M1`, `M2`, ..., `M9`). It writes an OBPI score column in `[0, 1]` without
modifying or replacing the existing metrics pipeline.
