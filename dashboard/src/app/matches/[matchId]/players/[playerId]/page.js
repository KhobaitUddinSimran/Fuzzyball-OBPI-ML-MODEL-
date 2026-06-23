"use client";

import { useParams } from "next/navigation";
import { PageContainer } from "@/components/layout/PageContainer";
import { AnalyzePlayerPanel } from "@/components/analysis/AnalyzePlayerPanel";
import { LoadingState } from "@/components/ui-custom/LoadingState";
import { ErrorState } from "@/components/ui-custom/ErrorState";
import { useWorldCupMatch } from "@/hooks/useWorldCupData";
import { useEligiblePlayers } from "@/hooks/useMatchPlayers";

export default function PlayerAnalysisPage() {
  const { matchId, playerId } = useParams();
  const { match, loading: matchLoading, error: matchError } = useWorldCupMatch(matchId);
  const { players, loading: playersLoading, error: playersError } = useEligiblePlayers(matchId);
  const loading = matchLoading || playersLoading;
  const player = players.find((item) => String(item.player_id) === String(playerId));
  const error = matchError || playersError || (!loading && !player ? "This player is not eligible for OBPI analysis in this match" : "");

  return (
    <PageContainer eyebrow="Step 4" title="Analyze Player" subtitle="Run OBPI analysis only when you are ready to send the player-match request to the model service.">
      {loading ? <LoadingState label="Loading player details" /> : null}
      {error ? <ErrorState title="Could not load player" message={error} /> : null}
      {!loading && !error && match && player ? <AnalyzePlayerPanel match={match} player={player} /> : null}
    </PageContainer>
  );
}
