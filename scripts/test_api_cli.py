"""Terminal CLI for testing the OBPI API endpoints.

Usage (PYTHONPATH must include ./src and project root):
    PYTHONPATH=./src:. python scripts/test_api_cli.py health
    PYTHONPATH=./src:. python scripts/test_api_cli.py players --match-id 3794686
    PYTHONPATH=./src:. python scripts/test_api_cli.py analyze --match-id 3794686 --player-id 1001
    PYTHONPATH=./src:. python scripts/test_api_cli.py leaderboard --match-id 3794686
    PYTHONPATH=./src:. python scripts/test_api_cli.py compare --match-id 3794686 --ids 1001,1002
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ─── Narrative helpers ─────────────────────────────────────────────────────

_ARCHETYPE_DESCRIPTIONS: dict[str, str] = {
    "tempo_controller": "Dictates the rhythm of play through smart pauses and well-timed off-ball calls.",
    "space_creator": "Excels at drawing defenders away and opening lanes for teammates.",
    "safe_receiver": "Reliable under pressure, consistently offering a secure passing option.",
    "balanced": "Solid across all areas with no glaring weakness.",
    "runner": "High volume of off-ball sprints and dynamic runs in behind the defence.",
    "high_obpi": "Elite off-ball intelligence overall.",
    "low_obpi": "Struggles to influence the game without the ball.",
}


def _obpi_label(score: float) -> str:
    if score >= 0.7:
        return "Elite off-ball performance"
    if score >= 0.55:
        return "Strong off-ball showing"
    if score >= 0.45:
        return "Average off-ball contribution"
    if score >= 0.3:
        return "Below-par off-ball game"
    return "Poor off-ball involvement"


def _percentile_word(pct: float) -> str:
    if pct >= 95:
        return "one of the best"
    if pct >= 80:
        return "well above average"
    if pct >= 60:
        return "above average"
    if pct >= 40:
        return "around average"
    if pct >= 20:
        return "below average"
    return "near the bottom"


def _dimension_bar(val: float, width: int = 20) -> str:
    filled = int(min(val / 50, 1.0) * width) if val > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _strengths(metrics: dict[str, float], top_n: int = 2) -> list[str]:
    sorted_m = sorted(metrics.items(), key=lambda kv: kv[1], reverse=True)
    out: list[str] = []
    for name, val in sorted_m[:top_n]:
        label = name.replace("M1_", "").replace("M2_", "").replace("M3_", "").replace("M4_", "").replace("M5_", "").replace("M6_", "").replace("M7_", "").replace("M8_", "").replace("M9_", "")
        out.append(f"{label} ({val:.2f})")
    return out


def _weaknesses(metrics: dict[str, float], bottom_n: int = 2) -> list[str]:
    sorted_m = sorted(metrics.items(), key=lambda kv: kv[1])
    out: list[str] = []
    for name, val in sorted_m[:bottom_n]:
        if val == 0.0:
            continue
        label = name.replace("M1_", "").replace("M2_", "").replace("M3_", "").replace("M4_", "").replace("M5_", "").replace("M6_", "").replace("M7_", "").replace("M8_", "").replace("M9_", "")
        out.append(f"{label} ({val:.2f})")
    return out


def _narrative_analyze(p: dict[str, Any]) -> str:
    lines: list[str] = []
    obpi = p["obpi_score"]
    pct = p["percentile"]
    arch = p["archetype"]
    dims = p["dimensions"]
    metrics = p["metrics"]

    lines.append(f"\n  What the engine sees:")
    lines.append(f"  {'─'*52}")
    lines.append(f"  {p['player_name']} recorded an OBPI of {obpi:.3f} — {_obpi_label(obpi).lower()}.")
    lines.append(f"  This places them in the {pct:.0f}th percentile, meaning they were")
    lines.append(f"  {_percentile_word(pct)} performers in this match.")
    lines.append(f"")
    lines.append(f"  Archetype: '{arch}' — {_ARCHETYPE_DESCRIPTIONS.get(arch, 'No description available.')}")
    lines.append(f"")

    # Dimension narrative
    lines.append(f"  Dimension breakdown:")
    lines.append(f"  {'─'*52}")
    for dim, val in dims.items():
        bar = _dimension_bar(val)
        lines.append(f"    {dim.title():<12} {bar}  {val:.2f}")

    # Strengths / weaknesses
    strengths = _strengths(metrics, top_n=2)
    weaknesses = _weaknesses(metrics, bottom_n=2)
    if strengths:
        lines.append(f"")
        lines.append(f"  Strengths: {', '.join(strengths)}")
    if weaknesses:
        lines.append(f"  Areas to improve: {', '.join(weaknesses)}")

    # SHAP insight
    shap = p.get("shap", {})
    top_shap = sorted(shap.items(), key=lambda kv: kv[1], reverse=True)[:1]
    if top_shap and top_shap[0][1] != 0.0:
        lines.append(f"")
        lines.append(f"  Key driver: {top_shap[0][0]} pushes the OBPI score {'up' if top_shap[0][1] > 0 else 'down'} the most.")

    lines.append(f"")
    lines.append(f"  {'─'*52}")
    return "\n".join(lines)


def _narrative_players(data: dict[str, Any]) -> str:
    players = data["players"]
    if not players:
        return ""
    top = players[0]
    bottom = players[-1]
    lines: list[str] = [
        f"\n  Match snapshot:",
        f"  {'─'*52}",
        f"  Top performer: {top['player_name']} (OBPI {top['obpi_score']:.3f}) — {top['archetype']}.",
        f"  Lowest rated: {bottom['player_name']} (OBPI {bottom['obpi_score']:.3f}) — {bottom['archetype']}.",
        f"  {'─'*52}",
    ]
    return "\n".join(lines)


def _narrative_leaderboard(data: dict[str, Any]) -> str:
    entries = data["entries"]
    if not entries:
        return ""
    top = entries[0]
    avg = sum(e["obpi_score"] for e in entries) / len(entries)
    lines: list[str] = [
        f"\n  Leaderboard snapshot:",
        f"  {'─'*52}",
        f"  {top['player_name']} leads with an OBPI of {top['obpi_score']:.3f} ({top['percentile']:.0f}th percentile).",
        f"  Average OBPI in this view: {avg:.3f}.",
        f"  {'─'*52}",
    ]
    return "\n".join(lines)


def _narrative_compare(data: dict[str, Any]) -> str:
    players = data["players"]
    deltas = data["dimension_deltas"]
    if len(players) < 2:
        return ""
    p1, p2 = players[0], players[1]
    winner = p1 if p1["obpi_score"] > p2["obpi_score"] else p2
    loser = p2 if winner is p1 else p1
    margin = abs(p1["obpi_score"] - p2["obpi_score"])
    lines: list[str] = [
        f"\n  Head-to-head story:",
        f"  {'─'*52}",
        f"  {winner['player_name']} comes out on top (OBPI {winner['obpi_score']:.3f} vs {loser['obpi_score']:.3f}).",
        f"  The gap is {margin:.3f} OBPI points.",
    ]
    for dim, val in deltas.items():
        if val == 0.0:
            continue
        leader = p1["player_name"] if val > 0 else p2["player_name"]
        lines.append(f"  {leader} dominates in {dim.title()} by {abs(val):.2f}.")
    lines.append(f"  {'─'*52}")
    return "\n".join(lines)


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, default=str))


# ─── Commands ──────────────────────────────────────────────────────────────


def cmd_health(_args: argparse.Namespace) -> int:
    """Test GET /health endpoint."""
    resp = client.get("/health")
    print(f"Status: {resp.status_code}")
    _print_json(resp.json())
    return 0


def cmd_players(args: argparse.Namespace) -> int:
    """Test GET /players endpoint."""
    resp = client.get(f"/players?match_id={args.match_id}")
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"\nMatch: {data['match_id']} | Players: {data['count']}\n")
        print(f"{'Rank':<6}{'ID':<8}{'Name':<22}{'OBPI':<8}{'Pctile':<8}{'Archetype':<18}")
        print("-" * 70)
        for idx, p in enumerate(data["players"], 1):
            print(
                f"{idx:<6}{p['player_id']:<8}{p['player_name']:<22}"
                f"{p['obpi_score']:<8.3f}{p['percentile']:<8.1f}{p['archetype']:<18}"
            )
        print(_narrative_players(data))
    else:
        _print_json(resp.json())
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Test POST /analyze endpoint."""
    payload = {
        "match_id": args.match_id,
        "player_id": args.player_id,
        "tier": args.tier,
    }
    resp = client.post("/analyze", json=payload)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        p = resp.json()
        print(f"\n{'='*60}")
        print(f"  {p['player_name']}  |  Match #{p['match_id']}")
        print(f"  OBPI: {p['obpi_score']:.3f}  |  Percentile: {p['percentile']:.1f}")
        print(f"  Archetype: {p['archetype']}  |  Minutes: {p['minutes']:.0f}")
        print(f"{'='*60}")
        print(_narrative_analyze(p))
        print(f"\n  {'Metric':<12}{'Value':<10}")
        print(f"  {'-'*22}")
        for m, v in p["metrics"].items():
            print(f"  {m:<12}{v:.3f}")
        print(f"\n  {'Weight':<12}{'Value':<10}")
        print(f"  {'-'*22}")
        for m, v in p["metric_weights"].items():
            print(f"  {m:<12}{v:.3f}")
    else:
        _print_json(resp.json())
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    """Test GET /leaderboard endpoint."""
    url = f"/leaderboard?match_id={args.match_id}"
    if args.limit:
        url += f"&limit={args.limit}"
    if args.archetype:
        url += f"&archetype={args.archetype}"
    resp = client.get(url)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"\nMatch: {data['match_id']} | Entries: {data['count']}\n")
        print(f"{'Rank':<6}{'ID':<8}{'Name':<22}{'OBPI':<8}{'Pctile':<8}{'Archetype':<18}")
        print("-" * 70)
        for e in data["entries"]:
            print(
                f"{e['rank']:<6}{e['player_id']:<8}{e['player_name']:<22}"
                f"{e['obpi_score']:<8.3f}{e['percentile']:<8.1f}{e['archetype']:<18}"
            )
        print(_narrative_leaderboard(data))
    else:
        _print_json(resp.json())
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Test POST /compare endpoint."""
    ids = [int(x) for x in args.ids.split(",")]
    payload = {"match_id": args.match_id, "player_ids": ids}
    resp = client.post("/compare", json=payload)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        players = data["players"]
        print(f"\n{'='*70}")
        print(f"  COMPARISON  |  Match #{args.match_id}")
        print(f"{'='*70}")
        for p in players:
            print(f"\n  {p['player_name']}")
            print(f"    OBPI: {p['obpi_score']:.3f}  |  Percentile: {p['percentile']:.1f}")
            print(f"    Archetype: {p['archetype']}")
            for dim, val in p["dimensions"].items():
                print(f"    {dim.title():<12}: {val:.3f}")
        print(_narrative_compare(data))
        print("\n  Dimension Deltas:")
        for dim, val in data["dimension_deltas"].items():
            sign = "+" if val > 0 else ""
            print(f"    {dim.title():<12}: {sign}{val:.3f}")
        print(f"\n  Insight: {data['auto_insight']}")
    else:
        _print_json(resp.json())
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to the appropriate test command."""
    parser = argparse.ArgumentParser(
        prog="test_api_cli",
        description="Terminal CLI for testing the OBPI API endpoints",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # health
    sub.add_parser("health", help="GET /health")

    # players
    p_players = sub.add_parser("players", help="GET /players?match_id=")
    p_players.add_argument("--match-id", type=int, required=True)

    # analyze
    p_analyze = sub.add_parser("analyze", help="POST /analyze")
    p_analyze.add_argument("--match-id", type=int, required=True)
    p_analyze.add_argument("--player-id", type=int, required=True)
    p_analyze.add_argument("--tier", type=str, default="open", choices=["open", "api"])

    # leaderboard
    p_board = sub.add_parser("leaderboard", help="GET /leaderboard")
    p_board.add_argument("--match-id", type=int, required=True)
    p_board.add_argument("--limit", type=int, default=None)
    p_board.add_argument("--archetype", type=str, default=None)

    # compare
    p_comp = sub.add_parser("compare", help="POST /compare")
    p_comp.add_argument("--match-id", type=int, required=True)
    p_comp.add_argument("--ids", type=str, required=True, help="Comma-separated player IDs")

    args = parser.parse_args(argv)
    cmd_map = {
        "health": cmd_health,
        "players": cmd_players,
        "analyze": cmd_analyze,
        "leaderboard": cmd_leaderboard,
        "compare": cmd_compare,
    }
    return cmd_map[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
