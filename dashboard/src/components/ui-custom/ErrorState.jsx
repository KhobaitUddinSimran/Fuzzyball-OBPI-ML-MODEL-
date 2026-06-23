export function ErrorState({ title = "Something went wrong", message }) {
  return (
    <div className="card border-low/60 bg-low/10 p-6 text-sm text-red-100">
      <div className="font-semibold text-red-50">{title}</div>
      {message ? <div className="mt-1">{message}</div> : null}
    </div>
  );
}
