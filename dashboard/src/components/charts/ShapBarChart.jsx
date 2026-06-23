"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export function ShapBarChart({ shap }) {
  const data = Object.entries(shap || {})
    .map(([metric, value]) => ({ metric, value }))
    .filter((item) => Number.isFinite(Number(item.value)) && Number(item.value) !== 0)
    .sort((a, b) => b.value - a.value);

  if (!data.length) {
    return (
      <div className="card flex h-96 flex-col justify-center p-5 text-sm text-muted">
        <h2 className="text-sm font-semibold text-white">Feature Importance</h2>
        <p className="mt-2">
          No SHAP/model feature importance values are available for this run.
        </p>
      </div>
    );
  }

  return (
    <div className="card h-96 p-4">
      <h2 className="mb-3 text-sm font-semibold text-white">Feature Importance</h2>
      <ResponsiveContainer width="100%" height="90%">
        <BarChart data={data} layout="vertical" margin={{ left: 20 }}>
          <CartesianGrid stroke="#334155" />
          <XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 11 }} />
          <YAxis type="category" dataKey="metric" tick={{ fill: "#cbd5e1", fontSize: 11 }} width={50} />
          <Tooltip
            formatter={(value) => Number(value || 0).toFixed(2)}
            contentStyle={{ background: "#1e293b", border: "1px solid #475569" }}
          />
          <Bar dataKey="value" fill="#22c55e" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
