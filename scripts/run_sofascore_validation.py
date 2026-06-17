"""Attempt SofaScore benchmark validation for OBPI scores.

SofaScore does not provide a documented open bulk-export API for player ratings.
This script tries the public web endpoint used for player search. If access is
blocked, it writes a reproducible pending report and a CSV template that can be
filled from an allowed SofaScore export/manual collection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Validate OBPI against SofaScore ratings when accessible."
    )
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=Path("data/processed/player_obpi_scores.parquet"),
        help="Player-match OBPI score parquet.",
    )
    parser.add_argument(
        "--ratings-path",
        type=Path,
        default=Path("data/external/sofascore_ratings.csv"),
        help="Optional CSV with player_id/player_name plus sofascore_rating.",
    )
    parser.add_argument(
        "--template-output",
        type=Path,
        default=Path("data/external/sofascore_ratings_template.csv"),
        help="Template CSV for SofaScore ratings.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/sofascore_validation.json"),
        help="Destination JSON report.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("results/SOFASCORE_VALIDATION.md"),
        help="Destination Markdown report.",
    )
    parser.add_argument(
        "--probe-player",
        default="Lionel Messi",
        help="Player name used to probe SofaScore endpoint access.",
    )
    return parser


def main() -> int:
    """Run SofaScore validation or write a pending access report."""
    import pandas as pd

    from obpi.ml.correlation import compare_benchmarks

    args = build_parser().parse_args()
    scored_df = pd.read_parquet(args.scores_path)
    aggregate = _aggregate_scores(scored_df)

    if args.ratings_path.exists():
        ratings = pd.read_csv(args.ratings_path)
        merged = _merge_ratings(aggregate, ratings)
        benchmark_columns = [
            column
            for column in ratings.columns
            if column not in {"player_id", "player_name"}
        ]
        report = {
            "status": "complete" if len(merged) >= 3 else "insufficient_overlap",
            "ratings_path": str(args.ratings_path),
            "overlap_rows": int(len(merged)),
            "matched_players": int(merged["player_id"].nunique())
            if "player_id" in merged
            else int(len(merged)),
            "benchmark_validation": compare_benchmarks(
                merged["obpi_mean"],
                merged[benchmark_columns].apply(pd.to_numeric, errors="coerce"),
            ).to_dict(orient="records")
            if len(merged) >= 3 and benchmark_columns
            else [],
        }
        _write_report(report, args.output_json, args.output_markdown)
        print(f"sofascore_validation: {report['status']}")
        print(f"overlap_rows: {report['overlap_rows']}")
        return 0

    access = _probe_sofascore(args.probe_player)
    _write_template(aggregate, args.template_output)
    report = {
        "status": "pending_sofascore_ratings",
        "source": "SofaScore",
        "access_probe": access,
        "reason": (
            "No local SofaScore ratings CSV is available, and the public "
            "SofaScore web API probe did not return usable rating data."
        ),
        "ratings_path_expected": str(args.ratings_path),
        "template_output": str(args.template_output),
        "template_rows": int(min(len(aggregate), 100)),
        "required_columns": ["player_id", "player_name", "sofascore_rating"],
        "next_step": (
            "Fill the template with SofaScore ratings from an allowed export or "
            "manual collection, save it as data/external/sofascore_ratings.csv, "
            "then rerun this script."
        ),
    }
    _write_report(report, args.output_json, args.output_markdown)
    print("sofascore_validation: pending_sofascore_ratings")
    print(f"template_output: {args.template_output}")
    print(f"access_status: {access['status']}")
    return 0


def _aggregate_scores(scored_df: Any) -> Any:
    grouped = scored_df.groupby(["player_id", "player_name"], dropna=False)
    aggregate = grouped.agg(
        obpi_mean=("obpi", "mean"),
        obpi_matches=("match_id", "nunique"),
    ).reset_index()
    return aggregate.sort_values("obpi_mean", ascending=False, ignore_index=True)


def _probe_sofascore(player_name: str) -> dict[str, Any]:
    url = "https://www.sofascore.com/api/v1/search/all?" + urlencode(
        {"q": player_name, "page": 0}
    )
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/125 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.sofascore.com/",
            "Origin": "https://www.sofascore.com",
        },
    )
    try:
        with urlopen(request, timeout=15) as response:
            body = response.read(5000).decode("utf-8", errors="replace")
            return {
                "status": "accessible",
                "url": url,
                "http_status": int(response.status),
                "sample": body[:500],
            }
    except HTTPError as exc:
        body = exc.read(500).decode("utf-8", errors="replace")
        return {
            "status": "http_error",
            "url": url,
            "http_status": int(exc.code),
            "body": body,
        }
    except URLError as exc:
        return {
            "status": "url_error",
            "url": url,
            "reason": str(exc.reason),
        }


def _write_template(aggregate: Any, output_path: Path) -> None:
    template = aggregate.head(100)[["player_id", "player_name", "obpi_mean"]].copy()
    template["sofascore_rating"] = ""
    template["sofascore_notes"] = ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)


def _merge_ratings(aggregate: Any, ratings: Any) -> Any:
    if "player_id" in ratings.columns:
        return aggregate.merge(ratings, on="player_id", how="inner")
    if "player_name" in ratings.columns:
        return aggregate.merge(ratings, on="player_name", how="inner")
    raise ValueError("SofaScore ratings CSV must include player_id or player_name")


def _write_report(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SofaScore External Benchmark Validation",
        "",
        f"- status: {report['status']}",
    ]
    if report["status"] == "complete":
        lines.extend(
            [
                f"- ratings_path: {report['ratings_path']}",
                f"- overlap_rows: {report['overlap_rows']}",
                f"- matched_players: {report['matched_players']}",
                "",
                "## Benchmark Correlations",
                "",
            ]
        )
        for item in report["benchmark_validation"]:
            lines.append(f"- {item}")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            f"- source: {report['source']}",
            f"- access_probe: {report['access_probe']}",
            f"- reason: {report['reason']}",
            f"- template_output: {report['template_output']}",
            f"- next_step: {report['next_step']}",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
