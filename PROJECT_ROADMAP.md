# OBPI-ML: Project Roadmap & Technical Narrative

A step-by-step technical blueprint for building the Off-Ball Positional Intelligence (OBPI) engine from an empty repository to a published, validated sports analytics system.

---

## Data Source & Field Availability Audit

### StatsBomb Open Data: What Is Actually Free

StatsBomb's open data repository (`https://github.com/statsbomb/open-data`) provides two distinct data products:

| Data Product | Free? | Where Stored |
|-------------|-------|-------------|
| **Event data** (passes, shots, receipts, carries) | Yes | `data/events/{match_id}.json` |
| **360 freeze-frame coordinates** (player x,y) | **Yes, selected matches** | `data/three-sixty/{match_id}.json` |
| **360 metrics** (line-breaking passes, etc.) | **No** | Subscription API only |

**Critical clarification:** The raw 360 freeze-frame player coordinates (`teammate`, `actor`, `keeper`, `x`, `y`) are free for selected competitions (World Cups, Euros, Champions League, some domestic leagues). The derived "360 metrics" are paid-only. We only need the raw coordinates.

The `competitions.json` metadata file indicates 360 availability via the `match_available_360` field. The pipeline filters to these matches automatically.

### Metric-by-Metric Data Availability

| Metric | Required Data | Open Data? | Notes |
|--------|-------------|-----------|-------|
| **M5 RBTL** | `ball_receipt` events, `location` [x,y] | Yes | Standard event data |
| **M6 RUP** | `ball_receipt` events, `under_pressure` flag | Yes | Flag present on most events |
| **M3 BRPC** | Receipt events + 360 frames (opponent positions) | **Yes (360 matches)** | Lane geometry from `x,y` coordinates |
| **M7 SCI** | 360 frames before/after action + `visible_area` | **Yes (360 matches)** | Voronoi on `teammate`/`x`/`y` + ConvexHull clip |
| **M1 SC** | 360 frames before/after (defender positions) | **Yes (360 matches)** | Local vs global defender shift |
| **M4 OBR90** | Event timestamps + inferred velocity | Yes (approx.) | Discrete events, not 25fps tracking. Velocity inferred from `Δx/Δt` with `Δt > 1.5s` gate |
| **M2 OIRC** | Event-to-event displacement vectors | Yes (approx.) | Same discrete limitation as OBR90 |
| **M9 CBI** | Ball carrier position + player run vector | Yes (approx.) | Run vector from discrete event points |
| **M8 LPC** | Receipt-to-action Δt + xT values | **Partial** | Timestamps: yes. **Pass xT: NO** — we build our own 12×8 grid xT model |

### What We Must Build Ourselves

1. **Expected Threat (xT) Model** (`src/obpi/utils/xt_model.py`)
   - StatsBomb open data does not include xT on passes (only `shot_statsbomb_xg` on shots).
   - We implement Karun Singh's original 12×8 grid formulation using only event data.
   - This is standard practice across the industry.

2. **Velocity Inference Layer** (`src/obpi/utils/kinematics.py`)
   - No continuous tracking. We approximate velocity from event-to-event displacement.
   - The `Δt > 1.5s` gate from the prospectus explicitly handles this limitation.
   - We will validate our approximations against **Metrica Sports open tracking data** (sample matches) to quantify error.

3. **Before/After Frame Proxy**
   - Spatial metrics (SCI, SC) use the 360 frame attached to the event immediately before and after the off-ball action.
   - This is the standard event-data approach; continuous tracking would be ideal but is not required.

### Dataset Window

The StatsBomb open data covers **2018–2024** (World Cups, Euros, Champions Leagues, some domestic seasons). The original prospectus specified 2013/14–2023/24, but the open data does not extend to 2013. The study is framed as "validated on 2018–2024 data" with historical extension as future work.

---

## Phase 1: Infrastructure & Scaffolding

**Goal:** Transform the empty repository into a production-grade Python monorepo with automated testing, linting, type safety, and a reproducible data access layer.

### 1.1 Directory Structure

We adopt the `src/` layout, the modern Python standard:

```
Fuzzyball-OBPI-ML-MODEL-/
├── .github/workflows/ci.yml     # GitHub Actions: pytest, ruff, mypy
├── data/
│   ├── raw/                     # Downloaded StatsBomb JSONs (gitignored)
│   └── processed/               # Parquet caches of computed metrics
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_membership_tuning.ipynb
│   └── 03_validation_results.ipynb
├── src/obpi/
│   ├── __init__.py
│   ├── pipeline.py              # Main orchestrator CLI entrypoint
│   ├── data/
│   │   ├── loader.py            # StatsBomb open-data / API wrapper
│   │   └── preprocessor.py      # ConvexHull clipping, sampling fixes
│   ├── metrics/
│   │   ├── spatial.py           # M1 (SC), M7 (SCI)
│   │   ├── movement.py          # M2 (OIRC), M4 (OBR90)
│   │   ├── receiving.py         # M3 (BRPC), M5 (RBTL), M6 (RUP)
│   │   └── temporal.py          # M8 (LPC), M9 (CBI)
│   ├── fuzzy/
│   │   ├── engine.py            # Mamdani ControlSystem builder
│   │   └── membership.py        # Data-driven membership functions
│   ├── ml/
│   │   ├── validation.py        # SVM + XGBoost stratified CV
│   │   └── explainability.py    # SHAP & permutation importance
│   └── utils/
│       ├── geometry.py          # Voronoi, convex hull, lane cones
│       ├── kinematics.py        # Velocity inference from discrete events
│       ├── units.py             # Unit conversion, normalizers
│       └── xt_model.py          # Expected Threat 12x8 grid model
├── tests/                       # Full pytest suite
├── api/main.py                  # FastAPI application
├── dashboard/                   # React frontend (Phase 5)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml               # Build system, ruff & mypy config
├── LICENSE
└── README.md
```

### 1.2 Dependencies

**Core stack (`requirements.txt`):**
- `pandas>=2.0.0`, `numpy>=1.24.0`, `pyarrow>=12.0.0`
- `statsbombpy>=1.0.0`
- `scipy>=1.10.0`, `shapely>=2.0.0`
- `scikit-fuzzy>=0.4.2`
- `scikit-learn>=1.3.0`, `xgboost>=2.0.0`, `shap>=0.42.0`
- `fastapi>=0.100.0`, `uvicorn[standard]>=0.23.0`, `pydantic>=2.0.0`

**Dev stack (`requirements-dev.txt`):**
- `pytest>=7.4.0`, `pytest-cov>=4.1.0`
- `ruff>=0.1.0`, `mypy>=1.5.0`, `pre-commit>=3.4.0`

### 1.3 Tooling Configuration

- **`pyproject.toml`:** Configures `ruff` for PEP 8 + Google docstrings, `mypy` in `strict` mode, and `pytest` with 80% coverage threshold.
- **`.github/workflows/ci.yml`:** Triggers on push/PR. Runs lint, type check, and full test suite on Ubuntu.

### 1.4 Data Access Layer (`src/obpi/data/loader.py`)

Abstraction over StatsBomb tiers:
```python
from obpi.data.loader import StatsBombLoader
loader = StatsBombLoader(tier="open")  # or "api"
frames = loader.get_freeze_frames(match_id=3794687)
```
This lets the entire pipeline run on free data during development and switch to paid 360 data seamlessly.

### 1.5 Preprocessing (`src/obpi/data/preprocessor.py`)

**Broadcast Blind Spot Fix:** Clips Voronoi diagrams to the `ConvexHull` of visible players, preventing overestimation of space due to missing off-camera defenders.

**Sparse Sampling Fix:** If `Δt > 1.5s` between frames, velocity-based metrics (OIRC, OBR90) discard finite-difference approximations and fall back to event-based proxies.

### 1.6 Test Fixtures (`tests/conftest.py`)

A `SyntheticMatchGenerator` creates fake but geometrically plausible match scenarios (e.g., a midfielder receiving between two dummy defenders) so every metric can be unit-tested without real data or an internet connection.

**Deliverable:** `pytest` passes, `ruff` is clean, and `python -m obpi.pipeline --help` works.

---

## Phase 2: The 9-Metric Engine

**Goal:** Implement the complete mathematical framework for all 9 dimensions, from simple event counting to complex spatiotemporal behavioral derivations.

### 2.1 Movement Quality (Easiest First)

#### M4: Off-Ball Runs per 90 (OBR90)
```
OBR90 = (N_runs / T_minutes) * 90
```
- Detect a "run" as contiguous frames where inferred velocity `v_k > 2.5 m/s` for `≥ 0.4 s`.
- Skip velocity inference if `Δt > 1.5s`.
- Exclude set-piece contexts.

#### M2: Off-Ball Impact Run Coefficient (OIRC)
```
OIRC = Σ(ΔC_k * RD_k) / N_runs
RD_k = max(0, cos_similarity(v_run, v_goal))
```
- `RD_k` is Run Directness: cosine similarity between the run vector and the vector to the opponent goal center.
- This penalizes sideways/backward runs and rewards directly threatening movement.

### 2.2 Receiving Intelligence

#### M5: Receipts Between the Lines (RBTL)
```
RBTL = N_receipts_in_half_spaces / N_total_receipts
```
- Half-spaces = vertical corridors between CB-FB channels, bounded by defensive midfield line and back line.
- Implemented via `shapely.geometry.Polygon.contains()` point-in-polygon tests.

#### M6: Receipts Under Pressure (RUP)
```
RUP = N_receipts_under_pressure / N_total_receipts
```
- Uses StatsBomb's `under_pressure` flag.
- Fallback: if flag missing, impute pressure by checking if any opponent is within `2.5 m` in the 360 freeze-frame.

#### M3: Best Receiving Position Coefficient (BRPC)
```
BRPC = N_(pressure>5m AND lane_open) / N_total_receipts
```
- **Pressure check:** nearest opponent `d > 5 m`.
- **Lane check:** a `45°` forward cone (length `15 m`) must not intersect any opponent point (buffered by `0.5 m`).

### 2.3 Spatial Manipulation (Hardest)

#### M7: Space Creation Index (SCI)
```
SCI = Σ(A_after_k - A_before_k) / N_actions
```
- Computes Voronoi tessellation (`scipy.spatial.Voronoi`) before and after each off-ball action.
- Clips to `ConvexHull` of visible players (Phase 1 fix).
- Sums attacking-team cell areas.
- Infinite edges are bounded to pitch dimensions `[0, 120] x [0, 80]` before clipping.

#### M1: Screening Coefficient (SC)
```
SC = N_(adjusted_shift > 1.5m) / N_total_screening_zone_possessions
ΔC_adj = ||mean(D_local_after) - mean(D_local_before)||
         - ||mean(D_all_after) - mean(D_all_before)||
```
- Measures defender displacement in a `10m x 10m` local box around the player.
- **Subtracts the global defensive shift** to isolate movement caused by the target player, not a globally dropping block.

### 2.4 Temporal Control & Communication

#### M8: La Pausa Coefficient (LPC)
```
LPC = N_pause_actions_improving_xT / N_total_pause_opportunities
```
Implemented as a Finite State Machine:
1. A pause opportunity = `ball_receipt` where the next action is by the same player.
2. Valid pause: `Δt ≥ 1.2s`, player velocity `< 0.5 m/s`, nearest defender decelerates.
3. Successful pause: post-pause pass `xT` > baseline immediate-pass `xT`.
- Uses a lightweight 12x8 grid `xT` model (Karun Singh formulation) or StatsBomb's own `xT` values.

#### M9: Call-for-Ball Index (CBI)
```
CBI = N_call-for-ball_movements_opening_lane / N_receipt_opportunities
```
- Receipt opportunity = frame where player is within `15 m` of ball carrier and closing distance.
- Call-for-ball run: movement vector aligned within `30°` of the ball carrier-to-player vector.
- Lane is open if the line segment from ball carrier to player has no opponent within `1.5 m`.

**Deliverable:** `obpi.metrics` computes all 9 metrics from raw match JSON; `pytest` validates each against synthetic scenarios.

---

## Phase 3: Mamdani Fuzzy Inference Layer

**Goal:** Aggregate the 9 crisp metrics into a single composite OBPI score using a data-calibrated fuzzy logic system.

### 3.1 Why Fuzzy Logic?

Deterministic metrics (xG, xT) cannot capture subjective constructs like "positional intelligence." Fuzzy logic encodes expert rules like *"If Space Creation is High AND La Pausa is High, then OBPI is Very High"* in an interpretable way. Coaches can read and challenge the rule base.

### 3.2 Data-Driven Membership Functions

Instead of the generic `trimf` from the prospectus:
- Low: `[0.0, 0.0, 0.4]`
- Medium: `[0.2, 0.5, 0.8]`
- High: `[0.6, 1.0, 1.0]`

We compute empirical percentiles from the dataset and use **trapezoidal functions (`trapmf`)**:
- Low: `trapmf(x; 0, 0, P20, P50)`
- Medium: `trapmf(x; P20, P40, P60, P80)`
- High: `trapmf(x; P50, P80, 1.0, 1.0)`

This ensures "High" means "better than 80% of players," not "greater than 0.6."

### 3.3 Rule Base & Aggregation

Parallel SISO rules per metric:
```
IF metric_i IS Low    THEN score_i IS Low
IF metric_i IS Medium THEN score_i IS Medium
IF metric_i IS High   THEN score_i IS High
```

Aggregate 9 fuzzy outputs using a weighted average. Initially uniform weights; weights are updated from SHAP importance in Phase 4.

### 3.4 Defuzzification & Range Correction

**Discrete Centroid (Center of Gravity):**
```
x* = Σ[μ_agg(x_i) * x_i] / Σ[μ_agg(x_i)]
```

**Known issue:** Centroid compresses output to `[0.15, 0.85]`. We restore variance via post-hoc linear scaling or a learned sigmoid calibration.

**Deliverable:** `obpi.fuzzy` takes a 9D metric vector and returns an OBPI score in `[0, 1]`. A notebook visualizes membership functions and 2D surface responses.

---

## Phase 4: ML Validation & Discriminant Layer

**Goal:** Prove the 9 engineered dimensions carry independent tactical signal.

### 4.1 Labeling (No Leakage)

1. Compute OBPI for every player.
2. Sort by OBPI.
3. Label **top 25%** = Class 1 (High), **bottom 25%** = Class 0 (Low).
4. **Discard middle 50%** to maximize separability and prevent tautology.

### 4.2 Preprocessing

Apply `StandardScaler` before ML input:
- Fuzzy layer compresses ranges to `[0.15, 0.85]`.
- Metrics have different native scales (e.g., SCI in m², OBR90 as count).
- Mandatory for SVM RBF and XGBoost convergence.

### 4.3 Model Suite

| Model | Role | Notes |
|-------|------|-------|
| **SVM (RBF)** | Primary discriminant | Grid search `C` and `gamma`; 5-Fold Stratified CV |
| **XGBoost** | Non-linear comparison | Bayesian opt over depth, learning rate, subsample |
| **Logistic Regression** | Linear baseline | If SVM/XGBoost do not outperform, features lack signal |

### 4.4 Evaluation

- **Stratified 5-Fold CV Accuracy:** Target `> 75%`.
- **ROC-AUC:** Target `> 0.80`.
- **Recall for Class 1:** Prioritized to catch all elite players, even with some false positives.

### 4.5 Explainability (SHAP)

Compute SHAP values for XGBoost to derive the **metric importance vector**:
> "LPC contributes +0.18 to High classifications, while OBR90 only contributes +0.04. Quality of pauses is more discriminating than quantity of runs."

This vector feeds back into Phase 3 to weight fuzzy aggregation.

**Deliverable:** `results/` contains CV tables, ROC curves, and SHAP plots. `obpi.ml.validate(metrics_df)` returns the classification report.

---

## Phase 5: API, Dashboard & Industrialization

**Goal:** Package the pipeline into a deployable web service with interactive visualizations.

### 5.1 FastAPI Backend (`api/main.py`)

**Endpoints:**
- `POST /analyze` - Accepts `match_id` + `player_id`, returns OBPI score, CI, percentile, 9 metrics, and SHAP breakdown.
- `GET /players?match_id=` - All players in a match with OBPI scores.
- `GET /health` - Service status and model version.

### 5.2 React Dashboard (`dashboard/`)

**Views:**
- **Player Profile:** Search bar, OBPI score with percentile badge, radar chart of 9 metrics.
- **Pitch Visualization:** Interactive pitch with toggles for Voronoi, run vectors, receipt heatmaps, La Pausa moments. Timeline slider for frame-by-frame scrubbing.
- **Comparative Analysis:** Side-by-side player comparison with parallel coordinates plot and auto-generated insights.

### 5.3 Docker & Deployment

**Multi-stage Dockerfile:**
1. Builder stage: compile wheels.
2. Runtime stage: non-root user, minimal image.

**`docker-compose.yml`** orchestrates `api`, `dashboard` (nginx), and optional `redis` cache.

**Deploy to Render:** Connect GitHub repo, auto-deploy on `main` branch pushes.

**Deliverable:** A live URL where a scout can search a match, click a player, and see a full OBPI breakdown with pitch diagrams.

---

## Phase 6: Expert Validation, Ablation & Publication

**Goal:** Validate against human judgment, prove unique signal per metric, and publish.

### 6.1 Convergent Validation Strategy (Dual Track)

We employ **two independent validation methods** to establish convergent validity. If both methods correlate strongly with OBPI, the construct is robust.

#### Option A: Public Scouting Reports (Large N, Weak Signal)

**Sources:**
- **Football Benchmark** player profiles (numerical ratings for "off-ball movement," "positioning")
- **The Athletic** scouting summaries (textual descriptions of off-ball work; NLP sentiment scoring)
- **Transfermarkt** "strengths/weaknesses" tags (categorical; mapped to 1–10 rubric)
- **Sofascore** tactical ratings (where available)

**Processing:**
1. Normalize all sources to a 1–10 scale.
2. Average across sources per player to reduce individual source bias.
3. Target: **n ≈ 200** attacking midfielders with proxy ratings.

**Statistical use:** Spearman correlation between proxy average and OBPI score.

#### Option B: Expert Panel (Small N, Strong Signal)

**Recruitment:**
- 3–5 amateur tactical analysts from Reddit /r/footballtactics, Twitter/X analytics community, or Discord.
- Must have published at least some tactical analysis (thread, blog, video) to qualify.

**Protocol:**
1. Provide each expert with 10–15 anonymized video clips (60–90 seconds each) of attacking midfielders.
2. Randomize order; no player names visible (tactical cam or blurred jerseys) to prevent halo bias.
3. Rubric: Rate 1–10 on "off-ball positional intelligence" with sub-dimensions (space creation, timing, receiving quality).
4. Aggregate expert scores via median (robust to outliers).

**Statistical use:**
- Spearman correlation between expert median and OBPI.
- **Inter-rater reliability:** Cronbach's alpha or ICC(2,k). Target `α > 0.70` to prove the construct is well-defined.

#### Convergent Validity Test

| Approach | Sample Size | Expected ρ | Purpose |
|----------|------------|------------|---------|
| **A: Proxy Ground Truth** | ~200 players | ~0.65–0.75 | Large-sample external validation |
| **B: Expert Panel** | ~50 players | ~0.70–0.80 | Direct, controlled human judgment |

**Success criteria:**
- Both correlations `ρ > 0.65` and `p < 0.001`.
- Correlations are consistent across methods (within `±0.10`).
- If both converge, construct validity is established.
- If they diverge, we investigate which construct each measures and refine OBPI.

**Practical sequencing:**
- Run **A first** (1–2 days of manual extraction) to validate the pipeline immediately.
- Launch **B in parallel** (1–2 weeks for recruitment + rating) for the final paper.
- Report both in the methodology section.

### 6.2 Sensitivity Analysis (Ablation)

1. Train classifier on all 9 metrics. Record accuracy `A_full`.
2. For each metric `M_i`, remove it, train on remaining 8. Record `A_-i`.
3. Compute `ΔA_i = A_full - A_-i`.
4. Metrics with `ΔA_i < 2%` are flagged as redundant.

**Expected:** Spatial (SCI, SC) and temporal (LPC) metrics should show highest `ΔA`, proving they capture missing signal.

### 6.3 Comparative Benchmarking

Compare OBPI against:
- StatsBomb xThreat
- Football Benchmark "Off-the-ball runs"
- SciSports "Offensive Positioning"

If OBPI explains orthogonal variance, incremental value is proven.

### 6.4 LaTeX Paper & Open Source

**Target:** *Journal of Sports Analytics*, MIT SSAC, or arXiv preprint.

**Structure:** Abstract, Introduction (asymmetry problem), Literature Review, Methodology (9 metrics + fuzzy + SVM), Results (CV, SHAP, expert ρ, ablation), Discussion (limitations), Conclusion.

**Release:** MIT License, reproducibility script for StatsBomb open data, Medium/Substack article with animated pitch diagrams.

**Deliverable:** Published preprint, live dashboard, and an extensible open-source community.

---

## Technology Stack Summary

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Data | StatsBombpy, Pandas, PyArrow |
| Geometry | SciPy, Shapely |
| Fuzzy Logic | scikit-fuzzy |
| ML | scikit-learn, XGBoost, SHAP |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | React, D3.js |
| DevOps | Docker, Docker Compose, GitHub Actions |
| Deployment | Render |

---

## Immediate Kick-Off Checklist

### Phase 1: Scaffolding
- [ ] Create directory skeleton (include `kinematics.py`, `xt_model.py` in `utils/`)
- [ ] Write `requirements.txt` and `pyproject.toml`
- [ ] Set up `statsbombpy` with open-data compatibility layer
- [ ] Implement `SyntheticMatchGenerator` in `tests/conftest.py`
- [ ] Write `src/obpi/utils/xt_model.py` (12×8 Expected Threat model)
- [ ] Write `src/obpi/utils/kinematics.py` (velocity inference with `Δt` gate)
- [ ] Implement 360 frame availability filter in `src/obpi/data/loader.py`

### Phase 2: Metrics Engine (Start Here)
- [ ] Start `src/obpi/metrics/movement.py` with OBR90 (M4) and OIRC (M2)
- [ ] Write first unit tests for M4 and M2 against synthetic data
- [ ] Build `receiving.py` with RBTL (M5), RUP (M6), BRPC (M3)
- [ ] Build `spatial.py` with SCI (M7) and SC (M1) using Voronoi + ConvexHull
- [ ] Build `temporal.py` with LPC (M8) and CBI (M9)

### Phase 6 Prep (Parallel Track)
- [ ] Create proxy ground truth spreadsheet template (Football Benchmark, The Athletic, etc.)
- [ ] Draft expert panel rubric and recruitment message
- [ ] Download Metrica Sports open tracking data (sample matches for velocity validation)
- [ ] Open a GitHub issue to track Phase 2 progress

**Next Step:** Switch to Code mode and execute Phase 1 scaffolding + Phase 2 metric engine implementation.
