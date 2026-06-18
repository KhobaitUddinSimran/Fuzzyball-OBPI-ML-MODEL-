"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { PageContainer } from "@/components/layout/PageContainer";
import { MatchGrid } from "@/components/matches/MatchGrid";
import { LoadingState } from "@/components/ui-custom/LoadingState";
import { ErrorState } from "@/components/ui-custom/ErrorState";

export default function MatchesPage() {
  return (
    <Suspense fallback={<LoadingState label="Loading matches" />}>
      <MatchesPageContent />
    </Suspense>
  );
}

function MatchesPageContent() {
  const searchParams = useSearchParams();
  const year = searchParams.get("year");
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(Boolean(year));
  const [error, setError] = useState("");

  useEffect(() => {
    if (!year) return;

    setLoading(true);
    setError("");
    fetch(`/api/matches?year=${encodeURIComponent(year)}`)
      .then((response) => {
        if (!response.ok) throw new Error("Unable to load matches for this year");
        return response.json();
      })
      .then(setMatches)
      .catch((requestError) => setError(requestError.message))
      .finally(() => setLoading(false));
  }, [year]);

  return (
    <PageContainer
      eyebrow="Step 2"
      title={year ? `FIFA World Cup ${year} Matches` : "Select a World Cup year"}
      subtitle="Choose the FIFA World Cup match you want to inspect."
      actions={<Link href="/" className="rounded-md border border-slate-600 px-3 py-2 text-sm text-slate-200 hover:border-sky-400 hover:text-sky-300">Back to years</Link>}
    >
      {!year ? <ErrorState title="Missing year" message="Return to the first screen, choose a FIFA World Cup year, and click Search." /> : null}
      {loading ? <LoadingState label="Loading matches" /> : null}
      {error ? <ErrorState title="Could not load matches" message={error} /> : null}
      {!loading && !error && year ? <MatchGrid matches={matches} /> : null}
    </PageContainer>
  );
}
