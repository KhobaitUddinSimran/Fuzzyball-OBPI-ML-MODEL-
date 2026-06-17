"""Validate aggregate OBPI scores against FIFA 23 player ratings."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset"
)
DEFAULT_KAGGLE_CACHE = Path(
    "~/Library/Caches/kagglehub/datasets/"
    "stefanoleone992/fifa-23-complete-player-dataset/versions/1/male_players.csv"
).expanduser()
LINUX_KAGGLE_CACHE = Path(
    "~/.cache/kagglehub/datasets/"
    "stefanoleone992/fifa-23-complete-player-dataset/versions/1/male_players.csv"
).expanduser()
FIFA_COLUMNS = [
    "player_id",
    "fifa_version",
    "fifa_update",
    "fifa_update_date",
    "short_name",
    "long_name",
    "player_positions",
    "overall",
    "potential",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
    "mentality_vision",
    "movement_reactions",
    "club_name",
    "league_name",
    "nationality_name",
]
BENCHMARK_COLUMNS = [
    "fifa_overall",
    "fifa_potential",
    "fifa_passing",
    "fifa_dribbling",
    "fifa_vision",
    "fifa_reactions",
    "fifa_shooting",
    "fifa_pace",
    "fifa_physic",
    "fifa_defending",
]


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Match aggregate OBPI players to FIFA 23 ratings and run benchmark "
            "correlations."
        )
    )
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=Path("data/processed/player_obpi_scores.parquet"),
        help="Scored OBPI parquet file.",
    )
    parser.add_argument(
        "--fifa-source",
        type=Path,
        default=None,
        help=(
            "Path to Kaggle male_players.csv. Defaults to FIFA_PLAYERS_CSV or the "
            "local kagglehub cache."
        ),
    )
    parser.add_argument(
        "--fifa-version",
        type=int,
        default=23,
        help="FIFA game version to use.",
    )
    parser.add_argument(
        "--fifa-update",
        type=int,
        default=None,
        help="Specific FIFA update number. Defaults to the latest update for the version.",
    )
    parser.add_argument(
        "--min-fuzzy-confidence",
        type=float,
        default=0.90,
        help="Minimum difflib confidence for non-exact name matches.",
    )
    parser.add_argument(
        "--ratings-output",
        type=Path,
        default=Path("data/external/fifa_ratings.csv"),
        help="Small external ratings CSV for generic validation workflows.",
    )
    parser.add_argument(
        "--match-audit-output",
        type=Path,
        default=Path("data/external/fifa_ratings_match_audit.csv"),
        help="Auditable matched-player CSV with match metadata.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/fifa_external_validation.json"),
        help="Destination JSON report.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("results/FIFA_EXTERNAL_VALIDATION.md"),
        help="Destination Markdown report.",
    )
    return parser


def main() -> int:
    """Run FIFA benchmark validation."""
    import pandas as pd

    from obpi.ml.correlation import compare_benchmarks

    args = build_parser().parse_args()
    fifa_source = resolve_fifa_source(args.fifa_source)
    scored_df = pd.read_parquet(args.scores_path)
    aggregate = aggregate_scores(scored_df)
    fifa_df = load_fifa_players(fifa_source, args.fifa_version, args.fifa_update)
    matched = match_obpi_to_fifa(
        aggregate,
        fifa_df,
        min_fuzzy_confidence=args.min_fuzzy_confidence,
    )

    ratings = matched[
        ["player_id", "player_name", *BENCHMARK_COLUMNS]
    ].sort_values("player_name", ignore_index=True)
    ratings.to_csv(_ensure_parent(args.ratings_output), index=False)
    matched.to_csv(_ensure_parent(args.match_audit_output), index=False)

    benchmark_df = matched[BENCHMARK_COLUMNS].apply(pd.to_numeric, errors="coerce")
    validation = compare_benchmarks(matched["obpi_mean"], benchmark_df)
    report = build_report(
        aggregate=aggregate,
        fifa_df=fifa_df,
        matched=matched,
        validation=validation,
        fifa_source=fifa_source,
        ratings_output=args.ratings_output,
        match_audit_output=args.match_audit_output,
        fifa_version=args.fifa_version,
        min_fuzzy_confidence=args.min_fuzzy_confidence,
    )
    write_report(report, args.output_json, args.output_markdown)
    print("fifa_external_validation: complete")
    print(f"matched_players: {report['matched_players']}/{report['obpi_players']}")
    print(f"best_benchmark: {report['benchmark_validation'][0]}")
    return 0


def resolve_fifa_source(path: Path | None) -> Path:
    """Resolve the FIFA CSV source path."""
    candidates = [
        path,
        Path(os.environ["FIFA_PLAYERS_CSV"]) if "FIFA_PLAYERS_CSV" in os.environ else None,
        LINUX_KAGGLE_CACHE,
        DEFAULT_KAGGLE_CACHE,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates if candidate is not None)
    raise FileNotFoundError(
        "FIFA male_players.csv was not found. Download the Kaggle dataset from "
        f"{KAGGLE_DATASET_URL} and rerun with --fifa-source. Searched: {searched}"
    )


def aggregate_scores(scored_df: Any) -> Any:
    """Aggregate player-match OBPI rows to one row per StatsBomb player."""
    return (
        scored_df.groupby(["player_id", "player_name"], dropna=False)
        .agg(
            obpi_mean=("obpi", "mean"),
            obpi_median=("obpi", "median"),
            obpi_matches=("match_id", "nunique"),
            obpi_team_names=("team_name", lambda values: "; ".join(sorted(set(values)))),
        )
        .reset_index()
        .sort_values("obpi_mean", ascending=False, ignore_index=True)
    )


def load_fifa_players(source_path: Path, fifa_version: int, fifa_update: int | None) -> Any:
    """Load the selected FIFA version/update from the Kaggle player CSV."""
    import pandas as pd

    fifa_df = pd.read_csv(source_path, usecols=FIFA_COLUMNS)
    fifa_df = fifa_df[fifa_df["fifa_version"].eq(fifa_version)].copy()
    if fifa_df.empty:
        raise ValueError(f"no FIFA rows found for fifa_version={fifa_version}")

    selected_update = fifa_update if fifa_update is not None else int(fifa_df["fifa_update"].max())
    fifa_df = fifa_df[fifa_df["fifa_update"].eq(selected_update)].copy()
    if fifa_df.empty:
        raise ValueError(
            f"no FIFA rows found for fifa_version={fifa_version}, fifa_update={selected_update}"
        )
    return fifa_df.reset_index(drop=True)


def match_obpi_to_fifa(
    aggregate: Any,
    fifa_df: Any,
    min_fuzzy_confidence: float,
) -> Any:
    """Name-match OBPI aggregate rows to FIFA ratings."""
    import pandas as pd

    prepared_fifa = _prepare_fifa_lookup(fifa_df)
    lookup = _build_lookup(prepared_fifa)
    keys = list(lookup.keys())
    rows = []
    for _, obpi_row in aggregate.iterrows():
        match = _match_one_player(
            obpi_row,
            prepared_fifa,
            lookup,
            keys,
            min_fuzzy_confidence,
        )
        if match is None:
            continue
        fifa_row, match_type, confidence = match
        rows.append(
            {
                **obpi_row.to_dict(),
                "fifa_player_id": int(fifa_row["player_id"]),
                "fifa_short_name": fifa_row["short_name"],
                "fifa_long_name": fifa_row["long_name"],
                "fifa_positions": fifa_row["player_positions"],
                "fifa_club_name": fifa_row["club_name"],
                "fifa_league_name": fifa_row["league_name"],
                "fifa_nationality_name": fifa_row["nationality_name"],
                "fifa_version": int(fifa_row["fifa_version"]),
                "fifa_update": int(fifa_row["fifa_update"]),
                "fifa_update_date": fifa_row["fifa_update_date"],
                "fifa_overall": fifa_row["overall"],
                "fifa_potential": fifa_row["potential"],
                "fifa_pace": fifa_row["pace"],
                "fifa_shooting": fifa_row["shooting"],
                "fifa_passing": fifa_row["passing"],
                "fifa_dribbling": fifa_row["dribbling"],
                "fifa_defending": fifa_row["defending"],
                "fifa_physic": fifa_row["physic"],
                "fifa_vision": fifa_row["mentality_vision"],
                "fifa_reactions": fifa_row["movement_reactions"],
                "match_type": match_type,
                "match_confidence": confidence,
            }
        )
    return pd.DataFrame(rows).sort_values("obpi_mean", ascending=False, ignore_index=True)


def build_report(
    aggregate: Any,
    fifa_df: Any,
    matched: Any,
    validation: Any,
    fifa_source: Path,
    ratings_output: Path,
    match_audit_output: Path,
    fifa_version: int,
    min_fuzzy_confidence: float,
) -> dict[str, Any]:
    """Build a serializable FIFA validation report."""
    match_type_counts = Counter(matched["match_type"])
    unmatched = sorted(set(aggregate["player_name"]) - set(matched["player_name"]))
    return {
        "status": "complete" if len(matched) >= 3 else "insufficient_overlap",
        "benchmark_source": "FIFA 23 complete player dataset",
        "benchmark_source_url": KAGGLE_DATASET_URL,
        "source_file": fifa_source.name,
        "source_file_note": "Raw Kaggle CSV is intentionally local-only and not committed.",
        "fifa_version": int(fifa_version),
        "fifa_update": int(matched["fifa_update"].iloc[0]) if len(matched) else None,
        "fifa_update_date": str(matched["fifa_update_date"].iloc[0]) if len(matched) else None,
        "fifa_rows_available": int(len(fifa_df)),
        "obpi_players": int(len(aggregate)),
        "matched_players": int(len(matched)),
        "unmatched_players": int(len(unmatched)),
        "match_rate": float(len(matched) / len(aggregate)) if len(aggregate) else 0.0,
        "match_type_counts": dict(match_type_counts),
        "min_fuzzy_confidence": float(min_fuzzy_confidence),
        "ratings_output": str(ratings_output),
        "match_audit_output": str(match_audit_output),
        "benchmark_columns": BENCHMARK_COLUMNS,
        "benchmark_validation": validation.to_dict(orient="records"),
        "unmatched_sample": unmatched[:25],
        "interpretation": (
            "FIFA ratings are an external commercial player-quality benchmark, not an "
            "expert panel and not event-level ground truth. Spearman rho measures "
            "convergent validity between aggregate OBPI and FIFA attributes on the "
            "matched subset."
        ),
    }


def write_report(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    """Persist JSON and Markdown FIFA validation reports."""
    _ensure_parent(json_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _ensure_parent(markdown_path).write_text(_markdown(report), encoding="utf-8")


def normalize_name(value: object) -> str:
    """Normalize names for cross-provider matching."""
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKD", str(value).lower())
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", ascii_text)
    cleaned = re.sub(r"\b(jr|junior|sr|senior)\b", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def token_sort_name(value: object) -> str:
    """Return a token-sorted normalized name."""
    return " ".join(sorted(normalize_name(value).split()))


def _prepare_fifa_lookup(fifa_df: Any) -> Any:
    prepared = fifa_df.copy()
    for column in ["long_name", "short_name"]:
        prepared[f"{column}_norm"] = prepared[column].map(normalize_name)
        prepared[f"{column}_token"] = prepared[column].map(token_sort_name)
        prepared[f"{column}_token_set"] = prepared[f"{column}_norm"].map(
            lambda value: frozenset(value.split())
        )
    prepared["club_name_norm"] = prepared["club_name"].map(normalize_name)
    prepared["nationality_name_norm"] = prepared["nationality_name"].map(normalize_name)
    return prepared


def _build_lookup(fifa_df: Any) -> dict[str, set[int]]:
    lookup: dict[str, set[int]] = {}
    for idx, fifa_row in fifa_df.iterrows():
        for column in [
            "long_name_norm",
            "short_name_norm",
            "long_name_token",
            "short_name_token",
        ]:
            key = fifa_row[column]
            if key:
                lookup.setdefault(key, set()).add(int(idx))
    return lookup


def _match_one_player(
    obpi_row: Any,
    fifa_df: Any,
    lookup: dict[str, set[int]],
    keys: list[str],
    min_fuzzy_confidence: float,
) -> tuple[Any, str, float] | None:
    candidate_scores: dict[int, tuple[float, str]] = {}
    player_name = obpi_row["player_name"]
    exact_keys = [normalize_name(player_name), token_sort_name(player_name)]
    for key in exact_keys:
        candidates = lookup.get(key, set())
        for idx in candidates:
            _set_candidate_score(candidate_scores, idx, 1.0, "exact_name")

    fuzzy_candidates: dict[str, float] = {}
    for key in exact_keys:
        for candidate in get_close_matches(key, keys, n=5, cutoff=min_fuzzy_confidence):
            fuzzy_candidates[candidate] = max(
                fuzzy_candidates.get(candidate, 0.0),
                SequenceMatcher(None, key, candidate).ratio(),
            )

    for key, confidence in fuzzy_candidates.items():
        for idx in lookup.get(key, set()):
            _set_candidate_score(candidate_scores, idx, float(confidence), "fuzzy_name")

    query_tokens = frozenset(exact_keys[0].split())
    if len(query_tokens) >= 2:
        for idx, fifa_row in fifa_df.iterrows():
            if query_tokens.issubset(fifa_row["long_name_token_set"]) or query_tokens.issubset(
                fifa_row["short_name_token_set"]
            ):
                _set_candidate_score(candidate_scores, idx, 0.94, "token_containment")

    if not candidate_scores:
        return None

    team_context = _extract_team_context(obpi_row)
    ranked = sorted(
        (
            (
                _has_team_context_match(fifa_df.loc[idx], team_context),
                confidence,
                float(fifa_df.loc[idx, "overall"]),
                idx,
                match_type,
            )
            for idx, (confidence, match_type) in candidate_scores.items()
        ),
        reverse=True,
    )
    if len(ranked) > 1 and ranked[0][:3] == ranked[1][:3]:
        return None
    _, confidence, _, idx, match_type = ranked[0]
    return fifa_df.loc[idx], match_type, float(confidence)


def _set_candidate_score(
    candidate_scores: dict[int, tuple[float, str]],
    idx: int,
    confidence: float,
    match_type: str,
) -> None:
    current = candidate_scores.get(idx)
    if current is None or confidence > current[0]:
        candidate_scores[idx] = (confidence, match_type)


def _extract_team_context(obpi_row: Any) -> set[str]:
    raw_value = str(obpi_row.get("obpi_team_names", ""))
    return {normalize_name(team_name) for team_name in raw_value.split("; ") if team_name}


def _has_team_context_match(fifa_row: Any, team_context: set[str]) -> bool:
    if not team_context:
        return False
    return (
        fifa_row["club_name_norm"] in team_context
        or fifa_row["nationality_name_norm"] in team_context
    )


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# FIFA External Validation",
        "",
        f"- status: {report['status']}",
        f"- benchmark_source: {report['benchmark_source']}",
        f"- benchmark_source_url: {report['benchmark_source_url']}",
        f"- fifa_version: {report['fifa_version']}",
        f"- fifa_update: {report['fifa_update']}",
        f"- fifa_update_date: {report['fifa_update_date']}",
        f"- obpi_players: {report['obpi_players']}",
        f"- matched_players: {report['matched_players']}",
        f"- unmatched_players: {report['unmatched_players']}",
        f"- match_rate: {report['match_rate']:.3f}",
        f"- match_type_counts: {report['match_type_counts']}",
        "",
        "## Benchmark Validation",
        "",
    ]
    for item in report["benchmark_validation"]:
        lines.append(
            "- "
            f"{item['benchmark']}: "
            f"rho={item['spearman_rho']:.4f}, "
            f"p={item['p_value']:.4g}, "
            f"n={item['n']}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "## Outputs",
            "",
            f"- ratings_output: {report['ratings_output']}",
            f"- match_audit_output: {report['match_audit_output']}",
        ]
    )
    if report["unmatched_sample"]:
        lines.extend(["", "## Unmatched Sample", ""])
        for name in report["unmatched_sample"]:
            lines.append(f"- {name}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
