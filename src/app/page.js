"use client";

import { useEffect, useState } from "react";
import { PageContainer } from "@/components/layout/PageContainer";
import { DateSelectionGrid } from "@/components/event/DateSelectionGrid";
import { LoadingState } from "@/components/ui-custom/LoadingState";
import { ErrorState } from "@/components/ui-custom/ErrorState";

export default function HomePage() {
  const [dates, setDates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/events/fifa-world-cup/dates")
      .then((response) => {
        if (!response.ok) throw new Error("Unable to load match dates");
        return response.json();
      })
      .then(setDates)
      .catch((requestError) => setError(requestError.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <PageContainer
      eyebrow="Step 1"
      title="Fuzzyball OBPI Dashboard"
      subtitle="Analyze off-ball positional intelligence in FIFA World Cup matches."
    >
      <section className="space-y-4">
        <div>
          <h2 className="text-xl font-semibold text-white">Select a Match Date</h2>
          <p className="mt-1 text-sm text-muted">Choose a FIFA World Cup date to view available matches.</p>
        </div>
        {loading ? <LoadingState label="Loading match dates" /> : null}
        {error ? <ErrorState title="Could not load dates" message={error} /> : null}
        {!loading && !error ? <DateSelectionGrid dates={dates} /> : null}
      </section>
    </PageContainer>
  );
}
