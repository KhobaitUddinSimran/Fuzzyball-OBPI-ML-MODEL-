import { metricGroups, metricLabels } from "@/lib/types/obpi";
import { formatScore } from "@/lib/utils/formatters";

export function MetricBreakdownTable({
  rawMetrics,
  normalizedMetrics,
  fuzzyMetrics,
  metricStatus
}) {
  return (
    <div className="card overflow-hidden">
      <div className="border-b border-slate-700 p-4">
        <h2 className="text-sm font-semibold text-white">Metric Breakdown</h2>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[900px] text-left text-sm">
          <thead className="bg-slate-900 text-xs uppercase tracking-wide text-muted">
            <tr>
              <th className="px-4 py-3">Dimension</th>
              <th className="px-4 py-3">Metric</th>
              <th className="px-4 py-3 text-right">Raw Value</th>
              <th className="px-4 py-3 text-right">Match-Normalized</th>
              <th className="px-4 py-3 text-right">Fuzzy Score</th>
              <th className="px-4 py-3">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700">
            {Object.entries(metricGroups).flatMap(([dimension, keys]) =>
              keys.map((key, index) => (
                <tr key={key}>
                  <td className="px-4 py-3 text-slate-300">{index === 0 ? dimension : ""}</td>
                  <td className="px-4 py-3 text-white">{metricLabels[key]}</td>
                  <td className="px-4 py-3 text-right font-semibold text-slate-200">
                    {formatNullable(rawMetrics?.[key])}
                  </td>
                  <td className="px-4 py-3 text-right font-semibold text-sky-200">
                    {formatNullable(normalizedMetrics?.[key])}
                  </td>
                  <td className="px-4 py-3 text-right font-semibold text-emerald-200">
                    {formatNullable(fuzzyMetrics?.[key])}
                  </td>
                  <td className="px-4 py-3 text-xs text-muted">
                    <div className="font-semibold text-slate-200">
                      {displayStatus({
                        raw: rawMetrics?.[key],
                        normalized: normalizedMetrics?.[key],
                        fuzzy: fuzzyMetrics?.[key],
                        status: metricStatus?.[key]?.status,
                        reason: metricStatus?.[key]?.reason
                      })}
                    </div>
                    {metricStatus?.[key]?.reason ? (
                      <div className="mt-1 max-w-72 leading-5">
                        {metricStatus[key].reason}
                      </div>
                    ) : null}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function formatNullable(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }
  return formatScore(value);
}

function formatStatus(status) {
  if (!status) return "Available";
  return status
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function displayStatus({ raw, normalized, fuzzy, status, reason }) {
  if (
    isClose(raw, 0) &&
    isClose(normalized, 0.5) &&
    isClose(fuzzy, 0.5) &&
    mentionsNeutralFallback(reason)
  ) {
    return "Neutral fallback";
  }

  return formatStatus(status);
}

function isClose(value, target) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return false;
  }
  return Math.abs(Number(value) - target) < 0.005;
}

function mentionsNeutralFallback(reason = "") {
  const text = String(reason).toLowerCase();
  return (
    text.includes("no opportunity") ||
    text.includes("no receipt opportunities") ||
    text.includes("no variation") ||
    text.includes("same value") ||
    text.includes("constant")
  );
}
