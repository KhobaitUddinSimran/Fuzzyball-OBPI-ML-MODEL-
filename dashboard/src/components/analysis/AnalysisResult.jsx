import { MetricRadar } from "@/components/charts/MetricRadar";
import { ShapBarChart } from "@/components/charts/ShapBarChart";
import { formatScore } from "@/lib/utils/formatters";
import { OBPIScoreCard } from "./OBPIScoreCard";
import { DimensionCards } from "./DimensionCards";
import { MetricBreakdownTable } from "./MetricBreakdownTable";

export function AnalysisResult({ analysis }) {
  const radarMetrics = analysis.fuzzy_metrics || analysis.normalized_metrics || analysis.metrics;
  const shapValues = analysis.explainability?.shap_values || analysis.shap;
  const explainabilityWeights =
    analysis.explainability?.metric_weights || analysis.metric_weights;

  return (
    <div className="space-y-6">
      <OBPIScoreCard analysis={analysis} />
      <AnalysisSummary analysis={analysis} />
      <DimensionCards scores={analysis.dimensions} />
      <DataQualityPanel dataQuality={analysis.data_quality} />
      <div className="grid gap-6 lg:grid-cols-2">
        <MetricRadar metrics={radarMetrics} />
        <ShapBarChart shap={shapValues} />
      </div>
      <MetricBreakdownTable
        rawMetrics={analysis.raw_metrics || analysis.metrics}
        normalizedMetrics={analysis.normalized_metrics || analysis.metrics}
        fuzzyMetrics={analysis.fuzzy_metrics || analysis.metrics}
        metricStatus={analysis.metric_status}
      />
      <MetricWeightsTable
        weights={explainabilityWeights}
        model={analysis.explainability?.model}
      />
    </div>
  );
}

function AnalysisSummary({ analysis }) {
  return (
    <section className="card border-sky-400/60 bg-sky-400/10 p-5">
      <h2 className="text-base font-semibold text-white">Analysis Summary</h2>
      <p className="mt-3 text-sm leading-6 text-sky-50">
        {formatSummary(analysis.summary) || fallbackSummary(analysis)}
      </p>
      <p className="mt-3 text-sm leading-6 text-sky-100/80">
        This result is a player-match OBPI estimate based on available StatsBomb
        event and 360 data.
      </p>
    </section>
  );
}

function formatSummary(summary) {
  if (!summary) return "";
  return String(summary).replace(/-?\d+\.\d+/g, (value) => formatScore(value));
}

function fallbackSummary(analysis) {
  const dimensions = Object.entries(analysis.dimensions || {}).sort((a, b) => b[1] - a[1]);
  const strongest = dimensions[0]?.[0] || "available";
  return (
    `${analysis.player_name} returned an OBPI score of ${formatScore(analysis.obpi_score)}. ` +
    `The strongest returned dimension is ${strongest}.`
  );
}

function DataQualityPanel({ dataQuality }) {
  if (!dataQuality) return null;
  const warnings = dataQuality.warnings || [];
  const unavailable = dataQuality.unavailable_metrics || [];

  return (
    <section className="card p-5">
      <h2 className="text-sm font-semibold text-white">Data Quality</h2>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <QualityItem label="360 data" value={dataQuality.has_360 ? "Available" : "Unavailable"} />
        <QualityItem label="Events" value={dataQuality.events_loaded ? "Loaded" : "Missing"} />
        <QualityItem
          label="Player events with 360 frames"
          value={dataQuality.joined_360_frames ?? 0}
        />
        <QualityItem label="Minutes" value={dataQuality.minutes_available ? "Available" : "Estimated"} />
      </div>
      {unavailable.length ? (
        <p className="mt-3 text-sm text-amber-200">
          Unavailable metrics: {unavailable.join(", ")}
        </p>
      ) : null}
      {warnings.length ? (
        <div className="mt-3 space-y-1 text-sm text-muted">
          {warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function QualityItem({ label, value }) {
  const displayValue = typeof value === "number" ? formatScore(value) : value;

  return (
    <div className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2">
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-1 text-sm font-semibold text-white">{displayValue}</div>
    </div>
  );
}

function MetricWeightsTable({ weights, model }) {
  const entries = Object.entries(weights || {});
  if (!entries.length) return null;

  return (
    <section className="card p-5">
      <h2 className="text-sm font-semibold text-white">Explainability Weights</h2>
      {model ? <p className="mt-1 text-sm text-muted">Model: {model}</p> : null}
      {model === "uniform" ? (
        <p className="mt-2 text-sm text-amber-200">
          Explainability fallback used. SHAP/model explanation was not available
          for this run.
        </p>
      ) : null}
      <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {entries.map(([metric, value]) => (
          <div
            key={metric}
            className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2"
          >
            <div className="text-xs uppercase tracking-wide text-muted">{metric}</div>
            <div className="mt-1 text-sm font-semibold text-white">
              {formatScore(value)}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
