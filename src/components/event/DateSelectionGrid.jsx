import { DateCard } from "./DateCard";
import { EmptyState } from "@/components/ui-custom/EmptyState";

export function DateSelectionGrid({ dates }) {
  if (!dates?.length) {
    return <EmptyState title="No match dates" message="No FIFA World Cup dates are available yet." />;
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {dates.map((date) => (
        <DateCard key={date.date} date={date} />
      ))}
    </div>
  );
}
