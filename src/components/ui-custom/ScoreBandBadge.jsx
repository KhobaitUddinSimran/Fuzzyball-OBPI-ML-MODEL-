import { getScoreBand, getScoreBandColor } from "@/lib/utils/scoreBands";

export function ScoreBandBadge({ score }) {
  return (
    <span className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-medium ${getScoreBandColor(score)}`}>
      {getScoreBand(score)}
    </span>
  );
}
