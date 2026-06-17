export function MatchHeader({ match }) {
  const score = match.home_score !== undefined && match.away_score !== undefined ? `${match.home_score} - ${match.away_score}` : "Score unavailable";

  return (
    <section className="card p-5">
      <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="text-sm uppercase tracking-wide text-sky-300">{match.competition}</div>
          <h1 className="mt-2 text-3xl font-semibold text-white">
            {match.home_team} vs {match.away_team}
          </h1>
          <div className="mt-3 flex flex-wrap gap-2 text-sm text-muted">
            <span>{match.date}</span>
            {match.stage ? <span>{match.stage}</span> : null}
            {match.kickoff_time ? <span>{match.kickoff_time}</span> : null}
            {match.stadium ? <span>{match.stadium}</span> : null}
          </div>
        </div>
        <div className="rounded-md border border-slate-600 bg-slate-950 px-6 py-4 text-center">
          <div className="text-xs uppercase tracking-wide text-muted">Score</div>
          <div className="mt-1 text-3xl font-semibold text-white">{score}</div>
          <div className="mt-1 text-xs text-slate-400">Match ID: {match.match_id}</div>
        </div>
      </div>
    </section>
  );
}
