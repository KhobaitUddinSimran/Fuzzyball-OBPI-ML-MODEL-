import Link from "next/link";
import { CalendarDays } from "lucide-react";

export function DateCard({ date }) {
  return (
    <Link href={`/matches?date=${date.date}`} className="card block p-5 transition hover:border-sky-400 hover:bg-slate-800">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-lg font-semibold text-white">{date.label}</div>
          <div className="mt-2 text-sm text-muted">{date.date}</div>
        </div>
        <span className="inline-flex h-10 w-10 items-center justify-center rounded-md bg-sky-400/10 text-sky-300">
          <CalendarDays size={20} />
        </span>
      </div>
      <div className="mt-5 text-sm text-slate-300">{date.match_count} matches available</div>
    </Link>
  );
}
