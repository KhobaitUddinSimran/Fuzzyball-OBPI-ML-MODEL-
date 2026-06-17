import { metricGroups, metricLabels } from "@/lib/types/obpi";
import { formatScore } from "@/lib/utils/formatters";

export function MetricBreakdownTable({ metrics }) {
  return (
    <div className="card overflow-hidden">
      <div className="border-b border-slate-700 p-4">
        <h2 className="text-sm font-semibold text-white">Metric Breakdown</h2>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead className="bg-slate-900 text-xs uppercase tracking-wide text-muted">
            <tr>
              <th className="px-4 py-3">Dimension</th>
              <th className="px-4 py-3">Metric</th>
              <th className="px-4 py-3 text-right">Value</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700">
            {Object.entries(metricGroups).flatMap(([dimension, keys]) =>
              keys.map((key, index) => (
                <tr key={key}>
                  <td className="px-4 py-3 text-slate-300">{index === 0 ? dimension : ""}</td>
                  <td className="px-4 py-3 text-white">{metricLabels[key]}</td>
                  <td className="px-4 py-3 text-right font-semibold text-sky-200">{formatScore(metrics?.[key])}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
