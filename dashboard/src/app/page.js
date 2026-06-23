"use client";

import { useState } from "react";
import Link from "next/link";
import { RefreshCw } from "lucide-react";
import { PageContainer } from "@/components/layout/PageContainer";
import { YearSearchForm } from "@/components/event/YearSearchForm";
import { LoadingState } from "@/components/ui-custom/LoadingState";
import { ErrorState } from "@/components/ui-custom/ErrorState";
import { useStatsBomb360Availability, useWorldCupYears } from "@/hooks/useWorldCupData";

const directAnalyzeClass = [
  "inline-flex items-center justify-center rounded-md bg-sky-400",
  "px-3 py-2 text-sm font-semibold text-slate-950 hover:bg-sky-300"
].join(" ");

const refreshButtonClass = [
  "inline-flex items-center justify-center gap-2 rounded-md border border-slate-600",
  "px-3 py-2 text-sm font-semibold text-slate-100",
  "hover:border-sky-400 hover:text-sky-300",
  "disabled:cursor-not-allowed disabled:opacity-60"
].join(" ");

export default function HomePage() {
  const { years, loading, error } = useWorldCupYears();
  const { availability, refresh, loading: refreshLoading, error: refreshError } = useStatsBomb360Availability();
  const [refreshMessage, setRefreshMessage] = useState("");

  async function refresh360Ids() {
    const result = await refresh();
    if (!result) {
      setRefreshMessage("");
      return;
    }
    setRefreshMessage(`Cached ${result.count} StatsBomb 360 match IDs from ${result.source}.`);
  }

  return (
    <PageContainer
      eyebrow="Step 1"
      title="Fuzzyball OBPI Dashboard"
      subtitle="Analyze off-ball positional intelligence in FIFA World Cup matches."
    >
      <section className="space-y-4">
        <div>
          <h2 className="text-xl font-semibold text-white">Select a World Cup Year</h2>
          <p className="mt-1 text-sm text-muted">Choose a FIFA World Cup year from StatsBomb, then search for matches.</p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Link
            href="/analyze"
            className={directAnalyzeClass}
          >
            Analyze by Match and Player ID
          </Link>
          <button
            type="button"
            onClick={refresh360Ids}
            disabled={refreshLoading}
            className={refreshButtonClass}
          >
            <RefreshCw size={16} className={refreshLoading ? "animate-spin" : ""} />
            {refreshLoading ? "Refreshing 360 IDs" : "Refresh 360 frame IDs"}
          </button>
          {refreshMessage ? <span className="text-sm text-emerald-300">{refreshMessage}</span> : null}
          {!refreshMessage && availability ? <span className="text-sm text-muted">{availability.count} 360 match IDs cached.</span> : null}
        </div>
        {refreshError ? <ErrorState title="Could not refresh 360 IDs" message={refreshError} /> : null}
        {loading ? <LoadingState label="Loading World Cup years" /> : null}
        {error ? <ErrorState title="Could not load years" message={error} /> : null}
        {!loading && !error ? <YearSearchForm years={years} /> : null}
      </section>
    </PageContainer>
  );
}
