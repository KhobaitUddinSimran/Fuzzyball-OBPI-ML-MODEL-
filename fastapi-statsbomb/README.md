# Fuzzyball StatsBomb API

Small FastAPI service for fetching football data from StatsBomb through `statsbombpy`.

## Setup

```bash
cd fastapi-statsbomb
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The Next.js app already points to this service through:

```env
MODEL_API_BASE_URL=http://localhost:8000
```

## Main Endpoints

- `GET /health`
- `GET /events/fifa-world-cup/years`
- `GET /events/fifa-world-cup/dates?year=YYYY`
- `GET /matches?event=fifa-world-cup&year=YYYY`
- `GET /matches/{match_id}`
- `GET /matches/{match_id}/eligible-players`
- `GET /matches/{match_id}/events`
- `GET /matches/{match_id}/lineups`

StatsBomb open-data IDs used here:

- FIFA World Cup competition: `43`
- Available World Cup seasons are read from `sb.competitions()`
