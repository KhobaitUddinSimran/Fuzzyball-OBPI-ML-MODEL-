import { EmptyState } from "@/components/ui-custom/EmptyState";
import { PlayerCard } from "./PlayerCard";

export function EligiblePlayersSection({ players, selectedTeamId, matchId }) {
  const teamPlayers = players.filter((player) => String(player.team_id) === String(selectedTeamId));

  return (
    <section className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold text-white">Players available for OBPI analysis</h2>
        <p className="mt-1 text-sm text-muted">Project scope: attacking midfield / advanced midfield roles only.</p>
      </div>
      {teamPlayers.length ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {teamPlayers.map((player) => (
            <PlayerCard key={player.player_id} player={player} matchId={matchId} />
          ))}
        </div>
      ) : (
        <EmptyState title="No eligible players" message="No eligible attacking-midfield players are available for this team." />
      )}
    </section>
  );
}
