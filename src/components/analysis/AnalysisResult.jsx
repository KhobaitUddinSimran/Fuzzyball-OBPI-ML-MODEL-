import { MetricRadar } from "@/components/charts/MetricRadar";
import { ShapBarChart } from "@/components/charts/ShapBarChart";
import { OBPIScoreCard } from "./OBPIScoreCard";
import { DimensionCards } from "./DimensionCards";
import { MetricBreakdownTable } from "./MetricBreakdownTable";

export function AnalysisResult({ analysis }) {
  return (
    <div className="space-y-6">
      <OBPIScoreCard analysis={analysis} />
      <DimensionCards scores={analysis.dimensions} />
      <div className="grid gap-6 lg:grid-cols-2">
        <MetricRadar metrics={analysis.metrics} />
        <ShapBarChart shap={analysis.shap} />
      </div>
      <MetricBreakdownTable metrics={analysis.metrics} />
      <section className="card p-5">
        <h2 className="text-sm font-semibold text-white">Analysis Summary</h2>
        <p className="mt-2 text-sm leading-6 text-muted">{analysis.summary || fallbackSummary(analysis)}</p>
      </section>
    </div>
  );
}

function fallbackSummary(analysis) {
  const dimensions = Object.entries(analysis.dimensions || {}).sort((a, b) => b[1] - a[1]);
  const strongest = dimensions[0]?.[0] || "available";
  return `${analysis.player_name} returned an OBPI score of ${analysis.obpi_score}. The strongest returned dimension is ${strongest}.`;
}
