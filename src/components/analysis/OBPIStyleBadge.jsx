export function OBPIStyleBadge({ style }) {
  if (!style) return null;

  return <span className="inline-flex rounded-md border border-sky-400 bg-sky-400/10 px-3 py-1 text-sm font-medium text-sky-200">{style}</span>;
}
