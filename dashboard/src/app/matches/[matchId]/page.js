"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { PageContainer } from "@/components/layout/PageContainer";
import { MatchHeader } from "@/components/matches/MatchHeader";
import { StatsBomb360Preview } from "@/components/matches/StatsBomb360Preview";
import { TeamTabs } from "@/components/matches/TeamTabs";
import { EligiblePlayersSection } from "@/components/matches/EligiblePlayersSection";
import { LoadingState } from "@/components/ui-custom/LoadingState";
import { ErrorState } from "@/components/ui-custom/ErrorState";
import { useWorldCupMatch } from "@/hooks/useWorldCupData";
import { useEligiblePlayers } from "@/hooks/useMatchPlayers";

export default function MatchDetailPage() {
  const { matchId } = useParams();
  const [selectedTeamId, setSelectedTeamId] = useState(null);
  const { match, loading: matchLoading, error: matchError } = useWorldCupMatch(matchId);
  const { players, loading: playersLoading, error: playersError } = useEligiblePlayers(matchId);
  const loading = matchLoading || playersLoading;
  const error = matchError || playersError;

  useEffect(() => {
    if (match?.teams?.home?.team_id && !selectedTeamId) {
      setSelectedTeamId(match.teams.home.team_id);
    }
  }, [match, selectedTeamId]);

  return (
    <PageContainer eyebrow="Step 3" title="Select Player" subtitle="Review the match context, then choose an eligible player for OBPI analysis.">
      {loading ? <LoadingState label="Loading match details" /> : null}
      {error ? <ErrorState title="Could not load match" message={error} /> : null}
      {!loading && !error && match ? (
        <div className="space-y-6">
          <MatchHeader match={match} />
          <StatsBomb360Preview matchId={matchId} has360={match.has_360} />
          <TeamTabs match={match} selectedTeamId={selectedTeamId} onSelectTeam={setSelectedTeamId} />
          <EligiblePlayersSection players={players} selectedTeamId={selectedTeamId} matchId={matchId} />
        </div>
      ) : null}
    </PageContainer>
  );
}
