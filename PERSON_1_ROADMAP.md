# Person 1 Roadmap: Data & Core Metrics Engineer

> Your day-by-day execution plan for building the OBPI data pipeline and 9-metric engine.  
> **Branch:** `data-metrics-engine`  
> **Timeline:** 5 weeks (Weeks 1–5)  
> **Deliverable:** All 9 metrics pass `pytest` + a notebook proving each metric on real StatsBomb open-data.

---

## Week 1: Infrastructure & Scaffolding

**Goal:** Transform the empty repo into a production-grade Python monorepo with automated testing, type safety, and a reproducible data layer.

| Day | Task | Files / Commands | PR? |
|-----|------|-----------------|-----|
| **Day 1** | Create full directory skeleton | `mkdir -p src/obpi/{data,metrics,fuzzy,ml,utils} tests notebooks data/{raw,processed} .github/workflows docker` | — |
| | Write `requirements.txt` | `pandas>=2.0.0`, `numpy>=1.24.0`, `pyarrow>=12.0.0`, `statsbombpy>=1.0.0`, `scipy>=1.10.0`, `shapely>=2.0.0`, `scikit-fuzzy>=0.4.2`, `scikit-learn>=1.3.0`, `xgboost>=2.0.0`, `shap>=0.42.0`, `fastapi>=0.100.0`, `uvicorn[standard]>=0.23.0`, `pydantic>=2.0.0` | — |
| | Write `requirements-dev.txt` | `pytest>=7.4.0`, `pytest-cov>=4.1.0`, `ruff>=0.1.0`, `mypy>=1.5.0`, `pre-commit>=3.4.0` | — |
| **Day 2** | Configure `pyproject.toml` | Set `ruff` (PEP 8 + Google docstrings), `mypy` strict, `pytest` 80% coverage | — |
| | Set up GitHub Actions CI | `.github/workflows/ci.yml` — run `ruff`, `mypy`, `pytest` on Ubuntu for every push/PR | — |
| **Day 3** | Stub `src/obpi/__init__.py`, `pipeline.py` | `pipeline.py` = CLI entrypoint: `python -m obpi.pipeline --help` | — |
| | Build `SyntheticMatchGenerator` | `tests/conftest.py` — fake match events + 360 frames for offline unit tests | — |
| **Day 4** | Implement `StatsBombLoader` | `src/obpi/data/loader.py` — `get_freeze_frames(match_id)`, `get_events(match_id)`, `get_competitions()` | — |
| | Implement 360 availability filter | `loader.py` method: filter `match_available_360 == True` from `competitions.json` | — |
| **Day 5** | Build `preprocessor.py` | `src/obpi/data/preprocessor.py` — `ConvexHullClipper`, `DeltaTGate` | — |
| | Write first test | `tests/test_data_loader.py` — assert loader returns non-empty DataFrame for a known open-data match | — |
| **Day 6** | Polish + lint | Run `ruff .`, `mypy src/`, `pytest` — fix all errors | — |
| **Day 7** | **PR to `main`** | Open PR: "Week 1: Repo scaffold + data layer". Tag Person 2 & 3 for review. | **PR #1** |

**Verification by end of Week 1:**
```bash
ruff check src/ tests/       # clean
mypy src/                   # clean
pytest tests/               # passes
python -m obpi.pipeline --help  # works
```

---

## Week 2: Movement Quality Metrics (M4, M2)

**Goal:** Implement the two movement-quality metrics using velocity inference. These are the easiest to validate because they rely on event timestamps and displacement vectors.

| Day | Task | Details |
|-----|------|---------|
| **Day 8** | Build `kinematics.py` | `src/obpi/utils/kinematics.py` — `infer_velocity(event_df, player_id)` with `Δt > 1.5s` gate |
| | Build `geometry.py` stub | `src/obpi/utils/geometry.py` — `run_directness(v_run, v_goal)` using cosine similarity |
| **Day 9** | Implement **M4 OBR90** | `src/obpi/metrics/movement.py` — `compute_obr90(events, player_id)` |
| | Test M4 | `tests/test_metrics_movement.py` — synthetic player with 3 runs in 90 min → expect `OBR90 = 3.0` |
| **Day 10** | Implement **M2 OIRC** | `movement.py` — `compute_oirc(events, player_id)` using `ΔC_k * RD_k` sum |
| | Test M2 | Same test file — verify directness-weighted run coefficient |
| **Day 11** | Validate on real data | Run M4 & M2 on StatsBomb match `3794687` (2018 World Cup, has 360) |
| | Sanity-check output | Are values in expected range? Any negative OIRC? Debug if so. |
| **Day 12** | Document edge cases | Docstrings for: set-piece exclusion, `Δt` gate fallback, missing velocity |
| **Day 13** | Refactor + add units | `src/obpi/utils/units.py` — `meters_per_second_to_kmh()`, normalizers |
| **Day 14** | **PR to `main`** | "Week 2: M4 + M2 movement metrics with tests". Tag reviewers. | **PR #2** |

**Key formulas to implement:**
```python
# M4: OBR90
runs = detect_runs(events, v_threshold=2.5, duration_threshold=0.4)
OBR90 = (len(runs) / minutes_played) * 90

# M2: OIRC
RD_k = max(0, np.dot(v_run, v_goal) / (np.linalg.norm(v_run) * np.linalg.norm(v_goal)))
OIRC = sum(delta_c * rd for delta_c, rd in zip(delta_Cs, RDs)) / len(runs)
```

---

## Week 3: Receiving Intelligence (M5, M6, M3)

**Goal:** Implement the three receiving metrics. M5 and M6 are event-based; M3 requires 360 freeze-frame geometry.

| Day | Task | Details |
|-----|------|---------|
| **Day 15** | Implement **M5 RBTL** | `src/obpi/metrics/receiving.py` — `compute_rbtl(events, player_id)` |
| | Define half-space polygons | In `geometry.py`: `get_half_spaces(defensive_line_y, back_line_y)` |
| | Test M5 | `tests/test_metrics_receiving.py` — 2 receipts in half-spaces / 4 total → `RBTL = 0.5` |
| **Day 16** | Implement **M6 RUP** | `receiving.py` — `compute_rup(events, player_id)` |
| | Add fallback | If `under_pressure` flag missing, check 360 frame for opponent within `2.5m` |
| | Test M6 | Same test file |
| **Day 17** | Implement **M3 BRPC** | `receiving.py` — `compute_brpc(events, frames, player_id)` |
| | Lane geometry | `geometry.py` — `forward_cone(origin, angle=45, length=15)` + `is_lane_open()` |
| **Day 18** | Test M3 | Synthetic receipt with defender at 6m + open cone → count = 1 |
| | Test M3 (blocked lane) | Defender inside cone → count = 0 |
| **Day 19** | Validate all 3 on real data | Run on match `3794687`, inspect distributions |
| **Day 20** | Refactor shared logic | Extract `get_receipt_events()`, `get_nearest_opponent()` into `preprocessor.py` |
| **Day 21** | **PR to `main`** | "Week 3: M5, M6, M3 receiving metrics with tests". Tag reviewers. | **PR #3** |

**Key formulas to implement:**
```python
# M5: RBTL
half_space_poly = get_half_space_polygon(def_line, back_line)
RBTL = sum(1 for r in receipts if half_space_poly.contains(Point(r['location']))) / len(receipts)

# M6: RUP
RUP = sum(1 for r in receipts if r.get('under_pressure', False)) / len(receipts)

# M3: BRPC
for receipt in receipts:
    pressure_ok = nearest_opponent_dist(frames, receipt) > 5.0
    lane_ok = is_lane_open(receipt['location'], cone=forward_cone(), opponents=frames)
    if pressure_ok and lane_ok: count += 1
BRPC = count / len(receipts)
```

---

## Week 4: Spatial + Temporal Metrics (M7, M1, M8, M9)

**Goal:** Implement the four hardest metrics. Spatial uses Voronoi; temporal uses the xT model.

| Day | Task | Details |
|-----|------|---------|
| **Day 22** | Build `xt_model.py` | `src/obpi/utils/xt_model.py` — Karun Singh 12×8 grid using only event data |
| | Test xT | Assert that a pass from xG=0.05 zone to xG=0.15 zone returns higher xT |
| **Day 23** | Implement **M7 SCI** | `src/obpi/metrics/spatial.py` — `compute_sci(frames_before, frames_after, player_id)` |
| | Voronoi + ConvexHull clip | `geometry.py` — `voronoi_area(players, clip_to_convex_hull=True)` |
| | Test M7 | Synthetic: 3 teammates gain 10m² each → `SCI = 10.0` |
| **Day 24** | Implement **M1 SC** | `spatial.py` — `compute_sc(frames_before, frames_after, player_id)` |
| | Adjusted shift | Local box mean vs global mean, subtract global shift |
| | Test M1 | Synthetic: local defenders shift 2m, global shifts 0.5m → adjusted = 1.5m > 1.5m threshold → count = 1 |
| **Day 25** | Implement **M8 LPC** | `src/obpi/metrics/temporal.py` — `compute_lpc(events, xt_model, player_id)` |
| | FSM logic | Receipt → same-player next action, `Δt >= 1.2s`, velocity < 0.5, defender decelerates, pass xT improves |
| | Test M8 | Synthetic: pause improves xT by 0.03 → count as successful |
| **Day 26** | Implement **M9 CBI** | `temporal.py` — `compute_cbi(events, frames, player_id)` |
| | Alignment check | Run vector within 30° of ball-carrier→player vector |
| | Lane check | No opponent within 1.5m of passing line |
| | Test M9 | Synthetic: aligned run, open lane → count = 1 |
| **Day 27** | Validate M7–M9 on real data | Run on match `3794687`, check for NaNs, infinities, impossible values |
| **Day 28** | **PR to `main`** | "Week 4: M7, M1, M8, M9 spatial + temporal metrics". Tag reviewers. | **PR #4** |

**Key formulas to implement:**
```python
# M7: SCI
cells_before = voronoi_cells(frames_before, team='attacking')
cells_after = voronoi_cells(frames_after, team='attacking')
SCI = sum(area_after - area_before) / n_actions

# M1: SC
local_before = mean(defenders_in_box(player, 10x10, frames_before))
local_after = mean(defenders_in_box(player, 10x10, frames_after))
global_before = mean(all_defenders(frames_before))
global_after = mean(all_defenders(frames_after))
ΔC_adj = abs(local_after - local_before) - abs(global_after - global_before)
SC = count(ΔC_adj > 1.5) / n_screening_possessions

# M8: LPC
for receipt in same_player_receipts:
    if Δt >= 1.2 and velocity < 0.5 and defender_decelerates and next_pass_xT > immediate_xT:
        count += 1
LPC = count / n_pause_opportunities

# M9: CBI
for opportunity in receipt_opportunities:
    if angle(run_vector, ball_to_player) < 30 and lane_open(opponents, buffer=1.5):
        count += 1
CBI = count / n_receipt_opportunities
```

---

## Week 5: Integration, Notebook & Handoff

**Goal:** Combine all 9 metrics into a single pipeline, debug visually, and prepare for Person 2's fuzzy engine.

| Day | Task | Details |
|-----|------|---------|
| **Day 29** | Build `pipeline.py` orchestrator | `src/obpi/pipeline.py` — `run_match(match_id) → DataFrame[player_id, M1..M9]` |
| | Add caching | Save computed metrics to `data/processed/{match_id}_metrics.parquet` |
| **Day 30** | Integration test | `tests/test_pipeline.py` — run full pipeline on synthetic match, assert 9 columns exist |
| **Day 31** | Create `notebooks/01_metric_debugging.ipynb` | Load real match, display pitch diagrams with: Voronoi cells, run vectors, receipt heatmap, La Pausa moments |
| **Day 32** | Notebook: M4 + M2 visuals | Histogram of OBR90, scatter OIRC vs minutes played |
| **Day 33** | Notebook: M5 + M6 + M3 visuals | Half-space receipt map, pressure receipt density, BRPC by zone |
| **Day 34** | Notebook: M7 + M1 + M8 + M9 visuals | Before/after Voronoi, screening heatmap, pause timeline, call-for-ball arrows |
| **Day 35** | **Final PR to `main`** | "Week 5: Full pipeline + debugging notebook". Tag reviewers. | **PR #5** |
| | Open GitHub Issue #10 | "Deliver 9-metric DataFrame to Person 2" — attach sample `metrics.parquet` |

**Handoff checklist for Person 2:**
- [ ] `data/processed/` contains at least 5 match metric files
- [ ] `notebooks/01_metric_debugging.ipynb` runs end-to-end
- [ ] `pytest tests/` passes with 80%+ coverage
- [ ] All 9 metrics have docstrings + edge-case notes
- [ ] `pipeline.py` has a `compute_all_metrics(match_id)` function that returns a clean DataFrame

---

## Daily Rhythm

```
09:00  Pull latest main, merge into your branch
09:30  Write code for today's task
12:00  Run tests, fix failures
14:00  Continue implementation / write tests
17:00  Run full test suite + ruff + mypy
17:30  Commit with descriptive message
18:00  Push to data-metrics-engine branch
```

---

## How to Get Unblocked

| Problem | Solution |
|---------|----------|
| **StatsBomb 360 not downloading** | Check `match_available_360` in `competitions.json`. Use match `3794687` (2018 World Cup) as a known working test case. |
| **Voronoi produces infinite cells** | Bound to pitch `[0,120]×[0,80]` before clipping to ConvexHull. See `scipy.spatial.Voronoi` + `shapely` intersection. |
| **xT model giving negative values** | Ensure you use the correct 12×8 grid and normalize by shot probability. Reference Karun Singh's original blog post. |
| **Velocity inference looks wrong** | Validate against Metrica Sports open tracking data. Compare your `Δx/Δt` approximation to ground-truth 25fps vectors. |
| **Metric values seem unrealistic** | Run on 3–5 real matches and plot distributions. If any metric is outside [0, reasonable_max], check for unit errors (m vs yards). |

---

## Files You Will Create

| File | Purpose |
|------|---------|
| `src/obpi/data/loader.py` | StatsBomb open-data wrapper |
| `src/obpi/data/preprocessor.py` | ConvexHull clip, Δt gate, event filters |
| `src/obpi/utils/geometry.py` | Voronoi, lane cones, polygons, half-spaces |
| `src/obpi/utils/kinematics.py` | Velocity inference with Δt gate |
| `src/obpi/utils/xt_model.py` | 12×8 Expected Threat model |
| `src/obpi/utils/units.py` | Unit conversions, normalizers |
| `src/obpi/metrics/movement.py` | **M4 OBR90**, **M2 OIRC** |
| `src/obpi/metrics/receiving.py` | **M5 RBTL**, **M6 RUP**, **M3 BRPC** |
| `src/obpi/metrics/spatial.py` | **M7 SCI**, **M1 SC** |
| `src/obpi/metrics/temporal.py` | **M8 LPC**, **M9 CBI** |
| `src/obpi/pipeline.py` | Main orchestrator CLI |
| `tests/conftest.py` | SyntheticMatchGenerator |
| `tests/test_data_*.py` | Data layer tests |
| `tests/test_metrics_*.py` | Metric unit tests |
| `tests/test_pipeline.py` | Integration test |
| `notebooks/01_metric_debugging.ipynb` | Visual validation on real matches |
| `requirements.txt` | Core dependencies |
| `requirements-dev.txt` | Dev dependencies |
| `pyproject.toml` | ruff, mypy, pytest config |
| `.github/workflows/ci.yml` | GitHub Actions CI |

---

## Success Criteria (End of Week 5)

- [ ] `pytest` passes with ≥ 80% coverage
- [ ] `ruff` and `mypy` report zero errors
- [ ] `python -m obpi.pipeline --match-id 3794687` outputs a DataFrame with columns `[player_id, M1, M2, ..., M9]`
- [ ] `notebooks/01_metric_debugging.ipynb` renders pitch diagrams for all 9 metrics
- [ ] 5 PRs merged to `main`, each reviewed by at least one teammate
- [ ] GitHub Issue #10 created for handoff to Person 2

---

## What Comes Next (Weeks 6–10)

| Week | Your Role |
|------|-----------|
| 6 | **Support Person 2** — answer questions about metric ranges, edge cases, DataFrame schema. Assist Person 3 with `POST /analyze` endpoint. |
| 7 | **Support Person 3** — ensure FastAPI can call `pipeline.py`. Provide sample match outputs for dashboard demos. |
| 8–10 | **Light support** — review PRs, fix bugs found during dashboard/validation integration. No new features unless asked. |

---

**Start now:** `git checkout data-metrics-engine` and create the directory skeleton.
