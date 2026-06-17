"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { PageContainer } from "@/components/layout/PageContainer";
import { MatchHeader } from "@/components/matches/MatchHeader";
import { TeamTabs } from "@/components/matches/TeamTabs";
import { EligiblePlayersSection } from "@/components/matches/EligiblePlayersSection";
import { LoadingState } from "@/components/ui-custom/LoadingState";
import { ErrorState } from "@/components/ui-custom/ErrorState";

export default function MatchDetailPage() {
  const { matchId } = useParams();
  const [match, setMatch] = useState(null);
  const [players, setPlayers] = useState([]);
  const [selectedTeamId, setSelectedTeamId] = useState(null);
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
      .then(([matchData, playerData]) => {
        setMatch(matchData);
        setPlayers(playerData);
        setSelectedTeamId(matchData.teams.home.team_id);
      })
      .catch((requestError) => setError(requestError.message))
      .finally(() => setLoading(false));
  }, [matchId]);

  return (
    <PageContainer eyebrow="Step 3" title="Select Player" subtitle="Review the match context, then choose an eligible player for OBPI analysis.">
      {loading ? <LoadingState label="Loading match details" /> : null}
      {error ? <ErrorState title="Could not load match" message={error} /> : null}
      {!loading && !error && match ? (
        <div className="space-y-6">
          <MatchHeader match={match} />
          <TeamTabs match={match} selectedTeamId={selectedTeamId} onSelectTeam={setSelectedTeamId} />
          <EligiblePlayersSection players={players} selectedTeamId={selectedTeamId} matchId={matchId} />
        </div>
      ) : null}
    </PageContainer>
  );
}
