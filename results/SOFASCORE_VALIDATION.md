# SofaScore External Benchmark Validation

- status: pending_sofascore_ratings
- source: SofaScore
- access_probe: {'status': 'http_error', 'url': 'https://www.sofascore.com/api/v1/search/all?q=Lionel+Messi&page=0', 'http_status': 403, 'body': '{"error": {"code": 403, "reason": "Forbidden" }}'}
- reason: No local SofaScore ratings CSV is available, and the public SofaScore web API probe did not return usable rating data.
- template_output: data/external/sofascore_ratings_template.csv
- next_step: Fill the template with SofaScore ratings from an allowed export or manual collection, save it as data/external/sofascore_ratings.csv, then rerun this script.
