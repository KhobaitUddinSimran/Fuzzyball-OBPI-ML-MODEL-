const defaultStages = [
  "Preparing request",
  "Loading StatsBomb data",
  "Computing OBPI metrics",
  "Running fuzzy model",
  "Building explanation"
];

export function PipelineProgress({ activeIndex = 0, stages = defaultStages }) {
  const progress = Math.min(100, Math.max(8, ((activeIndex + 1) / stages.length) * 100));

  return (
    <section className="card border-sky-400/60 bg-sky-400/10 p-5">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-sm font-semibold text-white">Running OBPI pipeline</h2>
          <p className="mt-1 text-sm text-sky-100">{stages[activeIndex] || stages.at(-1)}</p>
        </div>
        <div className="text-sm font-semibold text-sky-100">{Math.round(progress)}%</div>
      </div>

      <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full rounded-full bg-sky-300 transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div className="mt-4 grid gap-2 sm:grid-cols-5">
        {stages.map((stage, index) => {
          const isDone = index < activeIndex;
          const isActive = index === activeIndex;
          const markerClass = isDone || isActive ? "bg-sky-300" : "bg-slate-700";
          const textClass = isDone || isActive ? "text-sky-100" : "text-muted";

          return (
            <div key={stage} className="flex items-center gap-2 sm:block">
              <div className={`h-2.5 w-2.5 rounded-full sm:mb-2 ${markerClass}`} />
              <div className={`text-xs leading-5 ${textClass}`}>{stage}</div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
