"use client";

export function TeamTabs({ match, selectedTeamId, onSelectTeam }) {
  const teams = [match.teams.home, match.teams.away];

  return (
    <div className="flex flex-wrap gap-2">
      {teams.map((team) => {
        const active = String(team.team_id) === String(selectedTeamId);
        return (
          <button
            key={team.team_id}
            type="button"
            onClick={() => onSelectTeam(team.team_id)}
            className={`rounded-md border px-4 py-2 text-sm font-medium ${active ? "border-sky-400 bg-sky-400/10 text-sky-200" : "border-slate-700 bg-slate-800 text-slate-300 hover:border-slate-500"}`}
          >
            {team.team_name}
          </button>
        );
      })}
    </div>
  );
}
