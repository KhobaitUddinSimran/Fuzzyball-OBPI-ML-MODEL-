"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import { AnalysisResult } from "@/components/analysis/AnalysisResult";
import { PipelineProgress } from "@/components/analysis/PipelineProgress";
import { PageContainer } from "@/components/layout/PageContainer";
import { ErrorState } from "@/components/ui-custom/ErrorState";

const stages = [
  "Preparing player request",
  "Fetching StatsBomb events",
  "Computing M1-M9 metrics",
  "Scoring OBPI model",
  "Rendering analysis"
];

export default function GuidedAnalysisRunPage() {
  const { matchId, playerId } = useParams();
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState("");
  const [activeStage, setActiveStage] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    let progressTimer;

    async function runAnalysis() {
      setLoading(true);
      setError("");
      setAnalysis(null);
      setActiveStage(0);

      progressTimer = window.setInterval(() => {
        setActiveStage((current) => Math.min(current + 1, stages.length - 2));
      }, 1400);

      try {
        const response = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            match_id: Number(matchId),
            player_id: Number(playerId),
            tier: "open"
          })
        });

        const data = await response.json().catch(() => null);
        if (!response.ok) {
          throw new Error(data?.error || data?.detail || "Unable to analyze player");
        }

        if (!cancelled) {
          setActiveStage(stages.length - 1);
          setAnalysis(data);
        }
      } catch (requestError) {
        if (!cancelled) {
          setError(requestError.message);
        }
      } finally {
        window.clearInterval(progressTimer);
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    runAnalysis();

    return () => {
      cancelled = true;
      if (progressTimer) {
        window.clearInterval(progressTimer);
      }
    };
  }, [matchId, playerId]);

  return (
    <PageContainer
      eyebrow="OBPI pipeline"
      title="Analysis Run"
      subtitle={`Match ${matchId} / Player ${playerId}`}
    >
      <div className="mb-4">
        <Link
          href={`/matches/${matchId}/players/${playerId}`}
          className="inline-flex items-center gap-2 text-sm font-semibold text-sky-300 hover:text-sky-200"
        >
          <ArrowLeft size={16} />
          Back to player
        </Link>
      </div>

      {loading ? <PipelineProgress activeIndex={activeStage} stages={stages} /> : null}
      {error ? <ErrorState title="Analysis failed" message={error} /> : null}
      {analysis ? <AnalysisResult analysis={analysis} /> : null}
    </PageContainer>
  );
}
