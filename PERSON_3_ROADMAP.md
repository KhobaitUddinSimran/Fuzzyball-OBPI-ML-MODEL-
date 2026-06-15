# Person 3 Roadmap: API, Dashboard & Validation Coordinator

> Your day-by-day execution plan for the FastAPI backend, React dashboard, Docker deployment, expert validation coordination, and publication.  
> **Branch:** `api-dashboard-pub`  
> **Timeline:** 5 weeks (Weeks 6–10)  
> **Deliverable:** Live deployed dashboard URL + LaTeX paper draft with all figures.

---

## Dependencies & Handoffs

Before you start Week 6, you need:
- **From Person 1 (end of Week 5):** `pipeline.py` works in Docker, sample `metrics.parquet` files exist
- **From Person 2 (end of Week 5):** Fuzzy engine outputs OBPI scores for sample data
- **From Person 2 (end of Week 7):** SHAP weights JSON + `obpi.ml.validate()` function signature

---

## Week 6: FastAPI Backend (Phase 5, Part 1)

**Goal:** Build the REST API that exposes the full OBPI pipeline to the dashboard.

| Day | Task | Files | Details |
|-----|------|-------|---------|
| **Day 36** | Pull `main` into your branch | `git checkout api-dashboard-pub && git merge main` | Sync all merged PRs |
| | Set up FastAPI skeleton | `api/main.py` — `FastAPI()` app with `/health` | |
| **Day 37** | Implement `GET /health` | `api/routers/health.py` | Returns service status + model version + dependency versions |
| **Day 38** | Implement `GET /players?match_id=` | `api/routers/players.py` | Returns all players in a match with OBPI scores |
| | Integrate Person 1's loader | `from obpi.data.loader import StatsBombLoader` | |
| **Day 39** | Implement `POST /analyze` | `api/routers/analyze.py` | Accepts `{match_id, player_id}`, returns `{obpi, ci_95, percentile, metrics: {M1..M9}, shap_breakdown}` |
| | Integrate Person 2's validation | `from obpi.ml.validation import validate` | Call after computing metrics |
| **Day 40** | Pydantic models | `api/schemas.py` | `AnalyzeRequest`, `AnalyzeResponse`, `PlayerSummary` |
| **Day 41** | Error handling | `api/exceptions.py` | 404 for missing match, 422 for invalid player_id, 500 with trace_id |
| **Day 42** | **PR to `main`** | "Week 6: FastAPI backend with /analyze, /players, /health endpoints" | **PR #7** |

**Key code to implement:**
```python
# api/main.py
from fastapi import FastAPI
from api.routers import health, players, analyze

app = FastAPI(title="OBPI API", version="1.0.0")
app.include_router(health.router)
app.include_router(players.router, prefix="/players")
app.include_router(analyze.router, prefix="/analyze")

# api/routers/analyze.py
from obpi.pipeline import run_match
from obpi.ml.validation import validate
from obpi.ml.explainability import get_shap_breakdown

@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    metrics_df = run_match(request.match_id)
    player = metrics_df[metrics_df.player_id == request.player_id].iloc[0]
    obpi = fuzzy_engine.compute(player[METRICS].to_dict())
    shap = get_shap_breakdown(player[METRICS])
    return AnalyzeResponse(
        obpi=obpi,
        percentile=percentile_rank(obpi),
        metrics=player[METRICS].to_dict(),
        shap_breakdown=shap
    )
```

---

## Week 7: React Dashboard (Phase 5, Part 2)

**Goal:** Build the interactive frontend with pitch visualizations.

| Day | Task | Files | Details |
|-----|------|-------|---------|
| **Day 43** | Scaffold React app | `dashboard/` — `create-react-app` or Vite + TypeScript | |
| | Install dependencies | `recharts`, `d3`, `react-router-dom`, `axios` | |
| **Day 44** | Player Profile view | `dashboard/src/components/PlayerProfile.tsx` | Search bar, OBPI score badge, percentile |
| **Day 45** | Radar chart of 9 metrics | `dashboard/src/components/RadarChart.tsx` | Recharts radar, normalized to [0,1] |
| **Day 46** | Pitch Visualization view | `dashboard/src/components/PitchView.tsx` | Interactive pitch: toggles for Voronoi, run vectors, receipt heatmap |
| | D3.js pitch renderer | `dashboard/src/lib/pitchRenderer.ts` | StatsBomb pitch dimensions [0,120]×[0,80] |
| **Day 47** | Timeline slider | `dashboard/src/components/Timeline.tsx` | Frame-by-frame scrubbing for La Pausa moments |
| **Day 48** | Comparative Analysis view | `dashboard/src/components/CompareView.tsx` | Side-by-side player comparison, parallel coordinates plot |
| **Day 49** | Auto-generated insights | `dashboard/src/lib/insights.ts` | "Player X has higher LPC than 85% of AMs" |
| **Day 50** | **PR to `main`** | "Week 7: React dashboard with pitch viz, radar charts, player comparison" | **PR #8** |

---

## Week 8: Docker + Deployment (Phase 5, Part 3)

**Goal:** Containerize and deploy to Render.

| Day | Task | Files | Details |
|-----|------|-------|---------|
| **Day 51** | Multi-stage Dockerfile | `docker/Dockerfile` | Stage 1: builder (pip install + npm build), Stage 2: runtime (non-root, minimal) |
| **Day 52** | docker-compose.yml | `docker/docker-compose.yml` | `api` (FastAPI + Uvicorn), `dashboard` (nginx), `redis` (optional cache) |
| **Day 53** | Person 1 integration | Ensure `obpi.pipeline` works inside container | Test: `docker-compose up`, hit `GET /health` |
| **Day 54** | Environment config | `.env.example` — `STATSBOMB_TIER`, `REDIS_URL`, `MODEL_VERSION` | |
| **Day 55** | Render deployment | Connect GitHub repo to Render | Auto-deploy on `main` pushes |
| **Day 56** | **PR to `main`** | "Week 8: Docker + Render deployment" | **PR #9** |

---

## Week 9: Proxy Data Collection + Expert Panel Prep (Phase 6, Part 1)

**Goal:** Gather external validation data and recruit expert panel.

| Day | Task | Details |
|-----|------|---------|
| **Day 57** | **Option A: Proxy data** | Create spreadsheet template for Football Benchmark, The Athletic, Sofascore, Transfermarkt ratings |
| **Day 58** | Extract ratings manually | Target: n ≈ 200 attacking midfielders, normalize to 1–10 scale |
| **Day 59** | Save proxy data | `data/validation/proxy_ratings.csv` [player_name, source, rating, url] |
| **Day 60** | **Option B: Expert panel** | Draft recruitment message for Reddit r/footballtactics, Twitter/X analytics community |
| **Day 61** | Create rating rubric | 1–10 scale with sub-dimensions (space creation, timing, receiving quality) |
| **Day 62** | Prepare clip set | Identify 10–15 anonymized 60–90s clips from StatsBomb open-data matches |
| **Day 63** | **PR to `main`** | "Week 9: Validation data collection + expert panel prep" | **PR #10** |

---

## Week 10: Paper Draft + Final Polish (Phase 6, Part 2)

**Goal:** Write LaTeX paper and finalize dashboard.

| Day | Task | Details |
|-----|------|---------|
| **Day 64** | Receive Person 2's figures | `results/figures/` — 6 PNGs ready | |
| **Day 65** | LaTeX skeleton | `paper/main.tex` — Abstract, Intro, Literature Review, Methodology, Results, Discussion |
| **Day 66** | Write Methodology section | 9 metrics + fuzzy + SVM + validation protocol |
| **Day 67** | Write Results section | CV tables, SHAP, ablation, expert correlation, benchmark comparison |
| **Day 68** | Write Discussion | Limitations (discrete data, velocity approximation), future work (continuous tracking) |
| **Day 69** | Final dashboard polish | Responsive design, loading states, error boundaries |
| **Day 70** | **Final PR to `main`** | "Week 10: LaTeX paper draft + dashboard polish" | **PR #11** |

---

## Daily Rhythm

```
09:00  Pull latest main, merge into api-dashboard-pub
09:30  Frontend or backend work
12:00  Test API endpoints with curl/Postman
14:00  Continue implementation
17:00  Run lint, build checks
17:30  Commit with descriptive message
18:00  Push to api-dashboard-pub branch
```

---

## Critical Handoffs

| When | What | To/From Whom |
|------|------|--------------|
| End of Week 6 | `POST /analyze` API spec | From Person 2's `validate()` signature |
| End of Week 7 | SHAP weights for radar chart | From Person 2 |
| End of Week 8 | Live Render URL | To team for testing |
| End of Week 9 | Proxy ratings CSV | To Person 2 for correlation |
| End of Week 10 | All figures + paper draft | From Person 2 |

---

## Files You Will Create

| File | Purpose |
|------|---------|
| `api/main.py` | FastAPI app entrypoint |
| `api/routers/health.py` | `/health` endpoint |
| `api/routers/players.py` | `/players?match_id=` endpoint |
| `api/routers/analyze.py` | `/analyze` endpoint |
| `api/schemas.py` | Pydantic request/response models |
| `api/exceptions.py` | Custom HTTP exceptions |
| `dashboard/src/App.tsx` | React root |
| `dashboard/src/components/PlayerProfile.tsx` | Player search + OBPI display |
| `dashboard/src/components/RadarChart.tsx` | 9-metric radar |
| `dashboard/src/components/PitchView.tsx` | Interactive pitch |
| `dashboard/src/components/Timeline.tsx` | Frame scrubber |
| `dashboard/src/components/CompareView.tsx` | Side-by-side comparison |
| `dashboard/src/lib/pitchRenderer.ts` | D3 pitch drawing |
| `dashboard/src/lib/insights.ts` | Auto-generated text insights |
| `docker/Dockerfile` | Multi-stage build |
| `docker/docker-compose.yml` | Local orchestration |
| `data/validation/proxy_ratings.csv` | External validation data |
| `paper/main.tex` | LaTeX paper |

---

## Success Criteria (End of Week 10)

- [ ] FastAPI `/analyze` returns OBPI + metrics + SHAP in < 2s
- [ ] Dashboard renders on mobile and desktop
- [ ] Live URL on Render works for demo matches
- [ ] Proxy data: n ≥ 200 players with ratings
- [ ] Expert panel: 3–5 analysts recruited, rubric finalized
- [ ] LaTeX paper: complete draft with all figures
- [ ] 6 PRs merged to `main`, each reviewed by at least one teammate

---

## What Comes Before & After You

| Phase | Owner | Your Interface |
|-------|-------|---------------|
| **Phase 1–2** (Data + Metrics) | Person 1 | Provides Docker-compatible pipeline + sample data |
| **Phase 3–4** (Fuzzy + ML) | Person 2 | Provides `validate()` + SHAP weights |
| **Phase 5–6** (API + Dashboard + Paper) | **You** | Builds and deploys everything visible |

---

**Start now:** `git checkout api-dashboard-pub && git merge main` and scaffold the FastAPI app.
