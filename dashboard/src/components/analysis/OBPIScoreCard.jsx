import { formatPercentile, formatScore } from "@/lib/utils/formatters";
import { ScoreBandBadge } from "@/components/ui-custom/ScoreBandBadge";
import { OBPIStyleBadge } from "./OBPIStyleBadge";

export function OBPIScoreCard({ analysis }) {
  const style = analysis.obpi_style || formatArchetype(analysis.archetype);

  return (
    <section className="card p-5">
      <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="text-sm uppercase tracking-wide text-sky-300">OBPI Analysis Result</div>
          <h2 className="mt-2 text-3xl font-semibold text-white">{formatScore(analysis.obpi_score)}</h2>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <ScoreBandBadge score={analysis.obpi_score} />
            <OBPIStyleBadge style={style} />
          </div>
          <p className="mt-3 text-sm text-muted">
            Scores are normalized relative to players in this match.
          </p>
        </div>
        <div className="rounded-md border border-slate-700 bg-slate-950 px-5 py-4">
          <div className="text-xs uppercase tracking-wide text-muted">Percentile</div>
          <div className="mt-1 text-2xl font-semibold text-white">
            {analysis.percentile === null || analysis.percentile === undefined
              ? "N/A"
              : formatPercentile(analysis.percentile)}
          </div>
        </div>
      </div>
    </section>
  );
}

function formatArchetype(archetype) {
  if (!archetype) return "";
  return String(archetype)
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
