"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export function ShapBarChart({ shap }) {
  if (!shap) {
    return <div className="card p-4 text-sm text-muted">Feature importance is not available for this analysis result.</div>;
  }

  const data = Object.entries(shap)
    .map(([metric, value]) => ({ metric, value }))
    .sort((a, b) => b.value - a.value);

  return (
    <div className="card h-96 p-4">
      <h2 className="mb-3 text-sm font-semibold text-white">Feature Importance</h2>
      <ResponsiveContainer width="100%" height="90%">
        <BarChart data={data} layout="vertical" margin={{ left: 20 }}>
          <CartesianGrid stroke="#334155" />
          <XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 11 }} />
          <YAxis type="category" dataKey="metric" tick={{ fill: "#cbd5e1", fontSize: 11 }} width={50} />
          <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #475569" }} />
          <Bar dataKey="value" fill="#22c55e" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
