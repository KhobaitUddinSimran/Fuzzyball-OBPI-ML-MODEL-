# OBPI-ML: 3-Person Work Breakdown

> Clear role assignments, branch ownership, and deliverable timelines for the collaborative OBPI engine build.

---

## Team Overview

| Person | Role | GitHub Branch | Primary Phases | Secondary Support |
|--------|------|---------------|----------------|-------------------|
| **Person 1** | Data & Core Metrics Engineer | `data-metrics-engine` | Phase 1 (Scaffolding), Phase 2 (9-Metric Engine) | Phase 5 (API data endpoints) |
| **Person 2** | Fuzzy Logic & ML Validation Lead | `fuzzy-ml-validation` | Phase 3 (Fuzzy Inference), Phase 4 (ML Validation) | Phase 6 (Statistical analysis) |
| **Person 3** | API, Dashboard & Validation Coordinator | `api-dashboard-pub` | Phase 5 (API + Dashboard), Phase 6 (Expert Validation + Paper) | Phase 1 (CI/CD setup) |

---

## Collaboration Rules

### Branch Strategy
- Each person owns their branch and works exclusively on it.
- No direct pushes to `main`. All changes go through **Pull Requests** with at least 1 reviewer.
- Merge `main` into your branch weekly to stay synchronized.
- When a phase is complete, open a PR to `main`, tag both teammates for review.

### Code Review Protocol
- **Person 1** reviews PRs touching `src/obpi/metrics/` or `src/obpi/data/`.
- **Person 2** reviews PRs touching `src/obpi/fuzzy/` or `src/obpi/ml/`.
- **Person 3** reviews PRs touching `api/` or `dashboard/`.
- Any PR modifying `requirements.txt` or `pyproject.toml` requires **all 3** to review.

### Communication
- **GitHub Issues** for task tracking. Create an issue per metric/module.
- **PR descriptions** must include: what changed, why, and a screenshot/test output if applicable.
- **Weekly sync:** Every Sunday, a 15-minute check-in issue is posted summarizing blockers and next-week goals.

---

## Person 1: Data & Core Metrics Engineer

**Branch:** `data-metrics-engine`

### Phase 1: Infrastructure (Week 1)
- [ ] Create full directory skeleton including `kinematics.py` and `xt_model.py`
- [ ] Write `requirements.txt` and `requirements-dev.txt`
- [ ] Configure `pyproject.toml` (ruff, mypy, pytest)
- [ ] Set up GitHub Actions CI (`.github/workflows/ci.yml`)
- [ ] Build `SyntheticMatchGenerator` in `tests/conftest.py`
- [ ] Implement `StatsBombLoader` in `src/obpi/data/loader.py`
- [ ] Implement 360 frame availability filter
- [ ] Build `src/obpi/data/preprocessor.py` (ConvexHull clip, Δt gate)

### Phase 2: The 9-Metric Engine (Weeks 2–4)
- [ ] `src/obpi/utils/xt_model.py` — Expected Threat 12×8 grid model
- [ ] `src/obpi/utils/kinematics.py` — Velocity inference with Δt gate
- [ ] `src/obpi/utils/geometry.py` — Voronoi, lane cones, polygon ops
- [ ] `src/obpi/metrics/movement.py` — **M4 OBR90** + **M2 OIRC**
- [ ] `src/obpi/metrics/receiving.py` — **M5 RBTL** + **M6 RUP** + **M3 BRPC**
- [ ] `src/obpi/metrics/spatial.py` — **M7 SCI** + **M1 SC**
- [ ] `src/obpi/metrics/temporal.py` — **M8 LPC** + **M9 CBI**
- [ ] Unit tests for every metric (`tests/test_metrics_*.py`)
- [ ] Notebook: `notebooks/01_metric_debugging.ipynb` (visual validation)

### Phase 5 Support
- [ ] Assist Person 3 with `POST /analyze` endpoint data pipeline
- [ ] Ensure FastAPI can call the metric engine via `pipeline.py`

**Deliverables:** All 9 metrics pass `pytest`. A notebook shows each metric computed on a real StatsBomb open-data match.

---

## Person 2: Fuzzy Logic & ML Validation Lead

**Branch:** `fuzzy-ml-validation`

### Phase 3: Mamdani Fuzzy Inference (Week 5)
- [ ] `src/obpi/fuzzy/membership.py` — Percentile-driven `trapmf` generators
- [ ] `src/obpi/fuzzy/engine.py` — Mamdani ControlSystem builder
- [ ] Notebook: `notebooks/02_membership_tuning.ipynb` (distribution plots + function shapes)
- [ ] Range compression correction (post-hoc scaling/sigmoid)
- [ ] End-to-end test: input 9 metrics → output OBPI score

### Phase 4: ML Validation (Weeks 6–7)
- [ ] `src/obpi/ml/validation.py` — SVM (RBF) + XGBoost + Logistic Regression pipeline
- [ ] Quartile labeling strategy (top 25% / bottom 25%, discard middle 50%)
- [ ] `StandardScaler` preprocessing
- [ ] Stratified 5-Fold CV with accuracy, ROC-AUC, recall
- [ ] `src/obpi/ml/explainability.py` — SHAP summary plots + metric importance vector
- [ ] Notebook: `notebooks/03_validation_results.ipynb` (CV tables, ROC curves, SHAP plots)

### Phase 6 Support
- [ ] Statistical analysis support for correlation studies
- [ ] Generate all figures for the LaTeX paper

**Deliverables:** SVM/XGBoost achieve >75% CV accuracy. SHAP plots identify top 3 discriminating metrics. A notebook reproduces the full validation pipeline.

---

## Person 3: API, Dashboard & Validation Coordinator

**Branch:** `api-dashboard-pub`

### Phase 5: Industrialization (Weeks 6–8)
- [ ] `api/main.py` — FastAPI with three endpoints (`POST /analyze`, `GET /players`, `GET /health`)
- [ ] Pydantic request/response models
- [ ] `docker/Dockerfile` (multi-stage build)
- [ ] `docker-compose.yml` (api + dashboard + redis)
- [ ] React dashboard scaffolding (`dashboard/`)
- [ ] Player Profile view (search, OBPI score, radar chart)
- [ ] Pitch Visualization view (Voronoi toggle, run vectors, heatmaps, timeline)
- [ ] Comparative Analysis view (side-by-side, parallel coordinates)
- [ ] Deploy to Render

### Phase 6: Validation & Publication (Weeks 8–10)
- [ ] Create proxy ground truth spreadsheet template
- [ ] Extract ratings from Football Benchmark, The Athletic, Transfermarkt for ~50 attacking midfielders
- [ ] Draft expert panel rubric and recruitment message
- [ ] Coordinate 3–5 expert analysts for rating study
- [ ] Compute Spearman correlations (proxy vs OBPI, expert vs OBPI)
- [ ] Inter-rater reliability analysis (Cronbach's alpha)
- [ ] LaTeX paper draft (overleaf or local)
- [ ] Medium/Substack technical article with animated diagrams
- [ ] Final README polish + open-source release

**Deliverables:** A live dashboard URL. A completed LaTeX paper draft. Published preprint or conference submission.

---

## Shared Responsibilities

### Everyone
- [ ] Run `ruff` and `mypy` before every commit
- [ ] Write docstrings for every public function (Google style)
- [ ] Keep the `README.md` updated as the project evolves
- [ ] Attend weekly sync (Sunday check-in issue)

### Person 1 + Person 2
- [ ] Person 1 delivers the 9-metric DataFrame; Person 2 consumes it for fuzzy + ML
- [ ] Jointly agree on metric value ranges and edge cases before fuzzy calibration

### Person 2 + Person 3
- [ ] Person 2 provides SHAP metric weights; Person 3 displays them in the dashboard
- [ ] Person 3 provides the proxy ratings; Person 2 runs the correlation analysis

### Person 1 + Person 3
- [ ] Person 1 ensures the data loader works in Docker; Person 3 configures the container
- [ ] Person 1 provides sample match outputs; Person 3 uses them for dashboard demos

---

## Timeline Summary

| Week | Person 1 | Person 2 | Person 3 |
|------|----------|----------|----------|
| 1 | Phase 1: Scaffolding | — | CI/CD review |
| 2 | M4, M2 + tests | — | — |
| 3 | M5, M6, M3 + tests | — | — |
| 4 | M7, M1, M8, M9 + tests | — | — |
| 5 | Notebook debugging | Phase 3: Fuzzy engine | — |
| 6 | Support | Phase 4: ML validation | Phase 5: FastAPI |
| 7 | Support | ML validation + SHAP | React dashboard |
| 8 | Support | Support | Dashboard + Deploy |
| 9 | Support | Statistical support | Proxy data collection |
| 10 | Support | Paper figures | Expert panel + paper |

---

## First Action Items (This Week)

1. **Person 1:** Create the directory skeleton and push to `data-metrics-engine`. Open a PR to `main`.
2. **Person 2:** Review the PR, then pull `main` into `fuzzy-ml-validation` and stub the fuzzy engine.
3. **Person 3:** Review the PR, then pull `main` into `api-dashboard-pub` and set up the FastAPI skeleton.

**Goal by end of Week 1:** `main` has the repo scaffold + CI. Each branch has the first module stubbed and passing `pytest`.
