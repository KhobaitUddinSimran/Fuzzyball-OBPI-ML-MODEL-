"""Download all men's non-youth StatsBomb open data files.

This script pulls the official open-data repository structure needed for
validation work:

- ``competitions.json``
- ``matches/<competition_id>/<season_id>.json`` for every men's competition
- ``events/<match_id>.json`` for all discovered matches
- ``lineups/<match_id>.json`` for all discovered matches

Files are written under ``data/raw/statsbomb_open_data`` by default.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "statsbomb_open_data"
)
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
MAX_WORKERS = 8


def fetch_bytes(url: str) -> bytes:
    """Fetch raw bytes with retry support."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return response.read()
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            time.sleep(attempt)
    assert last_error is not None
    raise last_error


def download_json(url: str, destination: Path) -> Any:
    """Fetch a JSON payload from a URL and save it locally."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = fetch_bytes(url)
    destination.write_bytes(payload)
    return json.loads(payload.decode("utf-8"))


def download_file(url: str, destination: Path) -> None:
    """Fetch a single raw file to disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(fetch_bytes(url))


def is_target_competition(row: dict[str, Any]) -> bool:
    """Keep men's senior competitions only."""
    return row.get("competition_gender") == "male" and not row.get(
        "competition_youth",
        False,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI parser for the downloader."""
    parser = argparse.ArgumentParser(
        description=(
            "Download all men's StatsBomb open-data competitions, matches, "
            "events, and lineups."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store the downloaded files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not re-download files that already exist locally.",
    )
    return parser


def maybe_download_json(url: str, destination: Path, skip_existing: bool) -> Any:
    """Load cached JSON when available or download it."""
    if skip_existing and destination.exists():
        return json.loads(destination.read_text(encoding="utf-8"))
    return download_json(url, destination)


def maybe_download_file(url: str, destination: Path, skip_existing: bool) -> None:
    """Skip an existing file when requested."""
    if skip_existing and destination.exists():
        return
    download_file(url, destination)


def download_match_assets(
    match_id: int,
    output_dir: Path,
    skip_existing: bool,
) -> tuple[bool, bool]:
    """Download event and lineup files for a single match."""
    events_url = f"{BASE_URL}/events/{match_id}.json"
    events_path = output_dir / "events" / f"{match_id}.json"
    lineups_url = f"{BASE_URL}/lineups/{match_id}.json"
    lineups_path = output_dir / "lineups" / f"{match_id}.json"

    event_ok = False
    lineup_ok = False

    try:
        maybe_download_file(events_url, events_path, skip_existing)
        event_ok = True
    except urllib.error.URLError as exc:
        print(f"Skipping events for match {match_id}: {exc}", file=sys.stderr)

    try:
        maybe_download_file(lineups_url, lineups_path, skip_existing)
        lineup_ok = True
    except urllib.error.URLError as exc:
        print(f"Skipping lineups for match {match_id}: {exc}", file=sys.stderr)

    return event_ok, lineup_ok


def main() -> int:
    """Run the full men's open-data download."""
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()

    competitions_url = f"{BASE_URL}/competitions.json"
    competitions_path = output_dir / "competitions.json"

    try:
        competitions = maybe_download_json(
            competitions_url,
            competitions_path,
            args.skip_existing,
        )
    except urllib.error.URLError as exc:
        print(f"Failed to download competitions index: {exc}", file=sys.stderr)
        return 1

    target_competitions = [row for row in competitions if is_target_competition(row)]
    print(f"Found {len(target_competitions)} men's competition-seasons.")

    match_ids: set[int] = set()
    match_file_count = 0
    for row in target_competitions:
        competition_id = row["competition_id"]
        season_id = row["season_id"]
        matches_url = f"{BASE_URL}/matches/{competition_id}/{season_id}.json"
        matches_path = output_dir / "matches" / str(competition_id) / f"{season_id}.json"

        try:
            matches = maybe_download_json(matches_url, matches_path, args.skip_existing)
        except urllib.error.URLError as exc:
            print(
                f"Skipping matches for competition {competition_id}, season {season_id}: {exc}",
                file=sys.stderr,
            )
            continue

        match_file_count += 1
        for match in matches:
            match_ids.add(int(match["match_id"]))

    print(f"Downloaded {match_file_count} match files covering {len(match_ids)} matches.")

    event_count = 0
    lineup_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(download_match_assets, match_id, output_dir, args.skip_existing)
            for match_id in sorted(match_ids)
        ]
        for future in concurrent.futures.as_completed(futures):
            event_ok, lineup_ok = future.result()
            event_count += int(event_ok)
            lineup_count += int(lineup_ok)

    print(f"Downloaded {event_count} event files and {lineup_count} lineup files.")
    print(f"Saved dataset under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
