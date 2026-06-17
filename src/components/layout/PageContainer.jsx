export function PageContainer({ eyebrow, title, subtitle, children, actions }) {
  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          {eyebrow ? <div className="mb-2 text-sm font-medium uppercase tracking-wide text-sky-300">{eyebrow}</div> : null}
          <h1 className="text-3xl font-semibold text-white sm:text-4xl">{title}</h1>
          {subtitle ? <p className="mt-2 max-w-3xl text-sm leading-6 text-muted sm:text-base">{subtitle}</p> : null}
        </div>
        {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
      </div>
      {children}
    </div>
  );
}
