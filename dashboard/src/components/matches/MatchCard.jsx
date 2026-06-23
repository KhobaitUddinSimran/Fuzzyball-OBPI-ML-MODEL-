import Link from "next/link";
import { Clock, MapPin } from "lucide-react";

export function MatchCard({ match }) {
  const score = match.home_score !== undefined && match.away_score !== undefined ? `${match.home_score} - ${match.away_score}` : "Score unavailable";

  return (
    <Link href={`/matches/${match.match_id}`} className="card block p-5 transition hover:border-sky-400 hover:bg-slate-800">
      <div className="text-xs uppercase tracking-wide text-sky-300">{match.competition}</div>
      <div className="mt-3 grid grid-cols-[1fr_auto_1fr] items-center gap-3">
        <div className="text-lg font-semibold text-white">{match.home_team}</div>
        <div className="rounded-md bg-slate-950 px-3 py-2 text-center text-sm font-semibold text-slate-100">{score}</div>
        <div className="text-right text-lg font-semibold text-white">{match.away_team}</div>
      </div>
      <div className="mt-4 flex flex-wrap gap-2 text-xs text-muted">
        {match.has_360 ? <span className="rounded-md border border-emerald-500/60 bg-emerald-500/10 px-2 py-1 text-emerald-300">360 available</span> : null}
        {match.stage ? <span className="rounded-md border border-slate-700 px-2 py-1">{match.stage}</span> : null}
        {match.kickoff_time ? (
          <span className="inline-flex items-center gap-1 rounded-md border border-slate-700 px-2 py-1">
            <Clock size={13} />
            {match.kickoff_time}
          </span>
        ) : null}
        {match.stadium ? (
          <span className="inline-flex items-center gap-1 rounded-md border border-slate-700 px-2 py-1">
            <MapPin size={13} />
            {match.stadium}
          </span>
        ) : null}
      </div>
      <div className="mt-4 text-xs text-slate-400">Match ID: {match.match_id}</div>
    </Link>
  );
}
