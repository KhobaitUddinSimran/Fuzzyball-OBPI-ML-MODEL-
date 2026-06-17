export function getScoreBand(score) {
  if (score >= 0.75) return "Elite";
  if (score >= 0.5) return "Developing";
  if (score >= 0.25) return "Below Average";
  return "Low Impact";
}

export function getScoreBandColor(score) {
  if (score >= 0.75) return "text-elite border-elite/60 bg-elite/10";
  if (score >= 0.5) return "text-mid border-mid/60 bg-mid/10";
  return "text-low border-low/60 bg-low/10";
}
