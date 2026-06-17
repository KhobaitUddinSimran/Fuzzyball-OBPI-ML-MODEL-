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
  const date = searchParams.get("date");
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(Boolean(date));
  const [error, setError] = useState("");

  useEffect(() => {
    if (!date) return;

    setLoading(true);
    setError("");
    fetch(`/api/matches?date=${encodeURIComponent(date)}`)
      .then((response) => {
        if (!response.ok) throw new Error("Unable to load matches for this date");
        return response.json();
      })
      .then(setMatches)
      .catch((requestError) => setError(requestError.message))
      .finally(() => setLoading(false));
  }, [date]);

  return (
    <PageContainer
      eyebrow="Step 2"
      title={date ? `Matches on ${date}` : "Select a match date"}
      subtitle="Choose the FIFA World Cup match you want to inspect."
      actions={<Link href="/" className="rounded-md border border-slate-600 px-3 py-2 text-sm text-slate-200 hover:border-sky-400 hover:text-sky-300">Back to dates</Link>}
    >
      {!date ? <ErrorState title="Missing date" message="Return to the date screen and select a FIFA World Cup date." /> : null}
      {loading ? <LoadingState label="Loading matches" /> : null}
      {error ? <ErrorState title="Could not load matches" message={error} /> : null}
      {!loading && !error && date ? <MatchGrid matches={matches} /> : null}
    </PageContainer>
  );
}
