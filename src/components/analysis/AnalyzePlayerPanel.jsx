"use client";

import { useState } from "react";
import { Play } from "lucide-react";
import { AnalysisLoadingState } from "./AnalysisLoadingState";
import { AnalysisResult } from "./AnalysisResult";
import { ErrorState } from "@/components/ui-custom/ErrorState";

export function AnalyzePlayerPanel({ match, player }) {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function analyzePlayer() {
    setLoading(true);
    setError("");

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          match_id: Number(match.match_id),
          player_id: Number(player.player_id),
          tier: "open"
        })
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Unable to analyze player");
      setAnalysis(data);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

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
        </div>

        <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-center">
          <button
            type="button"
            onClick={analyzePlayer}
            disabled={loading}
            className="inline-flex items-center justify-center gap-2 rounded-md bg-sky-400 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-sky-300 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <Play size={16} />
            {loading ? "Analyzing OBPI" : "Analyze OBPI"}
          </button>
          <p className="text-sm text-muted">Click Analyze OBPI to send this player-match data to the model service.</p>
        </div>
      </section>

      {loading ? <AnalysisLoadingState /> : null}
      {error ? <ErrorState title="Analysis failed" message={error} /> : null}
      {analysis ? <AnalysisResult analysis={analysis} /> : null}
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
