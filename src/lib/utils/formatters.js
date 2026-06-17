export function formatScore(score) {
  return Number(score ?? 0).toFixed(2);
}

export function formatPercentile(percentile) {
  return `${Math.round(Number(percentile ?? 0))}th`;
}
