import { formatScore } from "@/lib/utils/formatters";

const dimensions = [
  ["spatial", "Spatial"],
  ["movement", "Movement"],
  ["receiving", "Receiving"],
  ["temporal", "Temporal"]
];

export function DimensionCards({ scores }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {dimensions.map(([key, label]) => (
        <div key={key} className="card p-4">
          <div className="text-sm text-muted">{label}</div>
          <div className="mt-2 text-2xl font-semibold text-white">{formatScore(scores?.[key])}</div>
        </div>
      ))}
    </div>
  );
}
