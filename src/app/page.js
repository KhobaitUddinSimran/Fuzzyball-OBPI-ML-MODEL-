"use client";

import { useEffect, useState } from "react";
import { PageContainer } from "@/components/layout/PageContainer";
import { YearSearchForm } from "@/components/event/YearSearchForm";
import { LoadingState } from "@/components/ui-custom/LoadingState";
import { ErrorState } from "@/components/ui-custom/ErrorState";

export default function HomePage() {
  const [years, setYears] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/events/fifa-world-cup/years")
      .then((response) => {
        if (!response.ok) throw new Error("Unable to load World Cup years");
        return response.json();
      })
      .then(setYears)
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
          <h2 className="text-xl font-semibold text-white">Select a World Cup Year</h2>
          <p className="mt-1 text-sm text-muted">Choose a FIFA World Cup year from StatsBomb, then search for matches.</p>
        </div>
        {loading ? <LoadingState label="Loading World Cup years" /> : null}
        {error ? <ErrorState title="Could not load years" message={error} /> : null}
        {!loading && !error ? <YearSearchForm years={years} /> : null}
      </section>
    </PageContainer>
  );
}
