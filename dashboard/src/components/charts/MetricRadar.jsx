"use client";

import { PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Tooltip } from "recharts";
import { metricLabels } from "@/lib/types/obpi";
import { formatScore } from "@/lib/utils/formatters";

export function MetricRadar({ metrics, comparison }) {
  const data = Object.entries(metricLabels).map(([key, label]) => ({
    metric: label.replace(/^M(\d) /, "M$1\n"),
    A: metrics?.[key] ?? 0,
    B: comparison?.[key] ?? 0
  }));

  return (
    <div className="card h-96 p-4">
      <h2 className="text-sm font-semibold text-white">9-Metric Radar</h2>
      <p className="mb-3 mt-1 text-xs text-muted">
        Scores are normalized relative to players in this match.
      </p>
      <ResponsiveContainer width="100%" height="84%">
        <RadarChart data={data}>
          <PolarGrid stroke="#475569" />
          <PolarAngleAxis dataKey="metric" tick={{ fill: "#cbd5e1", fontSize: 10 }} />
          <PolarRadiusAxis angle={30} domain={[0, 1]} tick={{ fill: "#94a3b8", fontSize: 10 }} />
          <Tooltip
            formatter={(value) => formatScore(value)}
            contentStyle={{ background: "#1e293b", border: "1px solid #475569" }}
          />
          <Radar name="Player A" dataKey="A" stroke="#38bdf8" fill="#38bdf8" fillOpacity={comparison ? 0.18 : 0.35} />
          {comparison ? <Radar name="Player B" dataKey="B" stroke="#fb923c" fill="#fb923c" fillOpacity={0.18} /> : null}
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
