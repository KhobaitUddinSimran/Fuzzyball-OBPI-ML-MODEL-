"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { PageContainer } from "@/components/layout/PageContainer";
import { AnalyzePlayerPanel } from "@/components/analysis/AnalyzePlayerPanel";
import { LoadingState } from "@/components/ui-custom/LoadingState";
import { ErrorState } from "@/components/ui-custom/ErrorState";

export default function PlayerAnalysisPage() {
  const { matchId, playerId } = useParams();
  const [match, setMatch] = useState(null);
  const [player, setPlayer] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    setLoading(true);
    setError("");

    Promise.all([
      fetch(`/api/matches/${matchId}`).then((response) => {
        if (!response.ok) throw new Error("Unable to load match details");
        return response.json();
      }),
      fetch(`/api/matches/${matchId}/eligible-players`).then((response) => {
        if (!response.ok) throw new Error("Unable to load eligible players");
        return response.json();
      })
    ])
      .then(([matchData, players]) => {
        const selectedPlayer = players.find((item) => String(item.player_id) === String(playerId));
        if (!selectedPlayer) throw new Error("This player is not eligible for OBPI analysis in this match");
        setMatch(matchData);
        setPlayer(selectedPlayer);
      })
      .catch((requestError) => setError(requestError.message))
      .finally(() => setLoading(false));
  }, [matchId, playerId]);

  return (
    <PageContainer eyebrow="Step 4" title="Analyze Player" subtitle="Run OBPI analysis only when you are ready to send the player-match request to the model service.">
      {loading ? <LoadingState label="Loading player details" /> : null}
      {error ? <ErrorState title="Could not load player" message={error} /> : null}
      {!loading && !error && match && player ? <AnalyzePlayerPanel match={match} player={player} /> : null}
    </PageContainer>
  );
}
