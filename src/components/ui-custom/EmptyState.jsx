export function EmptyState({ title = "Nothing to show", message = "No players match the current filters." }) {
  return (
    <div className="card p-6 text-sm text-muted">
      <div className="font-semibold text-slate-200">{title}</div>
      <div className="mt-1">{message}</div>
    </div>
  );
}
