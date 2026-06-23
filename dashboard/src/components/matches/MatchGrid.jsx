import { MatchCard } from "./MatchCard";
import { EmptyState } from "@/components/ui-custom/EmptyState";

export function MatchGrid({ matches }) {
  if (!matches?.length) {
    return <EmptyState title="No matches found" message="No FIFA World Cup matches are available for this date." />;
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {matches.map((match) => (
        <MatchCard key={match.match_id} match={match} />
      ))}
    </div>
  );
}
