import Link from "next/link";
import { UserRound } from "lucide-react";

export function PlayerCard({ player, matchId }) {
  return (
    <div className="card p-4">
      <div className="flex items-start gap-3">
        <span className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-slate-950 text-sky-300">
          <UserRound size={19} />
        </span>
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-base font-semibold text-white">{player.player_name}</h3>
          <div className="mt-1 flex items-center gap-2 text-sm text-muted">
            {player.has_obpi_data ? <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" title="Enough OBPI data" /> : null}
            <span>{player.team_name}</span>
          </div>
          <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-300">
            <span className="rounded-md border border-slate-700 px-2 py-1">{player.position}</span>
            {player.minutes ? <span className="rounded-md border border-slate-700 px-2 py-1">{player.minutes} minutes</span> : null}
          </div>
        </div>
      </div>
      <Link href={`/matches/${matchId}/players/${player.player_id}`} className="mt-4 inline-flex w-full justify-center rounded-md bg-sky-400 px-3 py-2 text-sm font-semibold text-slate-950 hover:bg-sky-300">
        View Player
      </Link>
    </div>
  );
}
