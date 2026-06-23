"use client";

import Link from "next/link";
import { Play } from "lucide-react";
import { PlayerObpiData } from "./PlayerObpiData";

export function AnalyzePlayerPanel({ match, player }) {
  const analysisHref = `/matches/${match.match_id}/players/${player.player_id}/analysis`;

  return (
    <div className="space-y-6">
      <section className="card p-5">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Info label="Player" value={player.player_name} />
          <Info label="Team" value={player.team_name} />
          <Info label="Position" value={player.position} />
          <Info label="Match" value={`${match.home_team} vs ${match.away_team}`} />
          <Info label="Minutes" value={player.minutes ? `${player.minutes}` : "N/A"} />
          <Info label="Match ID" value={match.match_id} />
          <Info label="Player ID" value={player.player_id} />
          <Info label="360 Data" value={match.has_360 ? "Available" : "Not available"} />
        </div>
      </section>

      <section className="card p-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-sm font-semibold text-white">Run analysis</h2>
            <p className="mt-1 text-sm text-muted">
              Send this match and player to the OBPI pipeline.
            </p>
          </div>
          <Link
            href={analysisHref}
            className="inline-flex items-center justify-center gap-2 rounded-md bg-sky-400 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-sky-300 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <Play size={16} />
            Analyze OBPI
          </Link>
        </div>
      </section>

      <PlayerObpiData metrics={player.obpi_metrics} />
    </div>
  );
}

function Info({ label, value }) {
  return (
    <div className="rounded-md border border-slate-700 bg-slate-900 p-3">
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-1 text-sm font-semibold text-white">{value}</div>
    </div>
  );
}
