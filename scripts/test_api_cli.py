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

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, default=str))


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
        data = resp.json()
        p = data
        print(f"\n{'='*60}")
        print(f"  {p['player_name']}  |  Match #{p['match_id']}")
        print(f"  OBPI: {p['obpi_score']:.3f}  |  Percentile: {p['percentile']:.1f}")
        print(f"  Archetype: {p['archetype']}  |  Minutes: {p['minutes']:.0f}")
        print(f"{'='*60}")
        print(f"\n  {'Dimension':<12}{'Score':<10}")
        print(f"  {'-'*22}")
        for dim, val in p["dimensions"].items():
            bar = "█" * int(val * 10)
            print(f"  {dim.title():<12}{val:.3f}  {bar}")
        print(f"\n  {'Metric':<12}{'Value':<10}")
        print(f"  {'-'*22}")
        for m, v in p["metrics"].items():
            print(f"  {m:<12}{v:.3f}")
        print(f"\n  {'SHAP':<12}{'Value':<10}")
        print(f"  {'-'*22}")
        for m, v in p["shap"].items():
            sign = "+" if v > 0 else ""
            print(f"  {m:<12}{sign}{v:.3f}")
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
