"use client";

import { useState } from "react";
import { Play } from "lucide-react";
import { AnalysisLoadingState } from "@/components/analysis/AnalysisLoadingState";
import { AnalysisResult } from "@/components/analysis/AnalysisResult";
import { PageContainer } from "@/components/layout/PageContainer";
import { ErrorState } from "@/components/ui-custom/ErrorState";

const analyzeButtonClass = [
  "inline-flex min-h-10 items-center justify-center gap-2 rounded-md",
  "bg-sky-400 px-4 py-2 text-sm font-semibold text-slate-950",
  "hover:bg-sky-300 disabled:cursor-not-allowed disabled:opacity-60"
].join(" ");

export default function AnalyzePage() {
  const [matchId, setMatchId] = useState("");
  const [playerId, setPlayerId] = useState("");
  const [tier, setTier] = useState("open");
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submitAnalysis(event) {
    event.preventDefault();
    setError("");
    setAnalysis(null);

    const numericMatchId = Number(matchId);
    const numericPlayerId = Number(playerId);

    if (!Number.isInteger(numericMatchId) || numericMatchId <= 0) {
      setError("Enter a valid numeric match ID.");
      return;
    }

    if (!Number.isInteger(numericPlayerId) || numericPlayerId <= 0) {
      setError("Enter a valid numeric player ID.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          match_id: numericMatchId,
          player_id: numericPlayerId,
          tier
        })
      });

      const data = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(data?.error || data?.detail || "Unable to analyze player.");
      }

      setAnalysis(data);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <PageContainer
      eyebrow="Direct analysis"
      title="Analyze OBPI"
      subtitle="Enter a StatsBomb match ID and player ID to run the OBPI pipeline directly."
    >
      <section className="card p-5">
        <form
          onSubmit={submitAnalysis}
          className="grid gap-4 lg:grid-cols-[1fr_1fr_180px_auto] lg:items-end"
        >
          <label className="text-sm font-medium text-slate-200">
            Match ID
            <input
              value={matchId}
              onChange={(event) => setMatchId(event.target.value)}
              className="control mt-2 w-full"
              inputMode="numeric"
              placeholder="3794686"
            />
          </label>

          <label className="text-sm font-medium text-slate-200">
            Player ID
            <input
              value={playerId}
              onChange={(event) => setPlayerId(event.target.value)}
              className="control mt-2 w-full"
              inputMode="numeric"
              placeholder="1001"
            />
          </label>

          <label className="text-sm font-medium text-slate-200">
            Tier
            <select
              value={tier}
              onChange={(event) => setTier(event.target.value)}
              className="control mt-2 w-full"
            >
              <option value="open">open</option>
            </select>
          </label>

          <button
            type="submit"
            disabled={loading}
            className={analyzeButtonClass}
          >
            <Play size={16} />
            {loading ? "Analyzing" : "Analyze OBPI"}
          </button>
        </form>
      </section>

      {loading ? <AnalysisLoadingState /> : null}
      {error ? <ErrorState title="Analysis failed" message={error} /> : null}
      {analysis ? <AnalysisResult analysis={analysis} /> : null}
    </PageContainer>
  );
}
