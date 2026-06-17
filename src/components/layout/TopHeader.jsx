"use client";

import Link from "next/link";
import { ArrowLeft, Trophy } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

export function TopHeader() {
  const pathname = usePathname();
  const router = useRouter();
  const searchParams = useSearchParams();
  const date = searchParams.get("date");
  const canGoBack = pathname !== "/";

  return (
    <header className="sticky top-0 z-30 border-b border-slate-700 bg-slate-950/90 backdrop-blur">
      <div className="mx-auto flex min-h-16 w-full max-w-7xl flex-col gap-3 px-4 py-3 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
        <div className="flex items-center gap-3">
          {canGoBack ? (
            <button
              type="button"
              onClick={() => router.back()}
              className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-slate-600 text-slate-200 hover:border-sky-400 hover:text-sky-300"
              aria-label="Go back"
            >
              <ArrowLeft size={18} />
            </button>
          ) : null}

          <Link href="/" className="flex items-center gap-3">
            <span className="inline-flex h-10 w-10 items-center justify-center rounded-md bg-sky-400/15 text-sky-300">
              <Trophy size={20} />
            </span>
            <div>
              <div className="text-base font-semibold text-white">Fuzzyball OBPI</div>
              <div className="text-xs text-muted">FIFA World Cup match analysis</div>
            </div>
          </Link>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-300">
          <Step active={pathname === "/"}>1. Select date</Step>
          <Step active={pathname === "/matches" && Boolean(date)}>2. Select match</Step>
          <Step active={pathname.startsWith("/matches/") && !pathname.includes("/players/")}>3. Select player</Step>
          <Step active={pathname.includes("/players/")}>4. Analyze player</Step>
        </div>
      </div>
    </header>
  );
}

function Step({ active, children }) {
  return (
    <span className={`rounded-md border px-2.5 py-1 ${active ? "border-sky-400 bg-sky-400/10 text-sky-200" : "border-slate-700 bg-slate-800 text-muted"}`}>
      {children}
    </span>
  );
}
