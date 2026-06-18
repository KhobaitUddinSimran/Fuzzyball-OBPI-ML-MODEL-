"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Search } from "lucide-react";
import { EmptyState } from "@/components/ui-custom/EmptyState";

export function YearSearchForm({ years }) {
  const router = useRouter();
  const [selectedYear, setSelectedYear] = useState(years?.[0]?.year || "");

  if (!years?.length) {
    return <EmptyState title="No World Cup years" message="No FIFA World Cup seasons are available from StatsBomb yet." />;
  }

  function submitSearch(event) {
    event.preventDefault();
    if (!selectedYear) return;
    router.push(`/matches?year=${encodeURIComponent(selectedYear)}`);
  }

  return (
    <form onSubmit={submitSearch} className="card max-w-xl space-y-4 p-5">
      <div>
        <label htmlFor="world-cup-year" className="text-sm font-medium text-slate-200">
          FIFA World Cup year
        </label>
        <select id="world-cup-year" className="control mt-2 w-full" value={selectedYear} onChange={(event) => setSelectedYear(event.target.value)}>
          {years.map((item) => (
            <option key={`${item.year}-${item.season_id}`} value={item.year}>
              {item.label || item.year}
            </option>
          ))}
        </select>
      </div>

      <button type="submit" className="inline-flex items-center justify-center gap-2 rounded-md bg-sky-400 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-sky-300">
        <Search size={16} />
        Search Matches
      </button>
    </form>
  );
}
