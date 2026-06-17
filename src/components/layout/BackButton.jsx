"use client";

import { ArrowLeft } from "lucide-react";
import { useRouter } from "next/navigation";

export function BackButton({ label = "Back" }) {
  const router = useRouter();

  return (
    <button type="button" onClick={() => router.back()} className="inline-flex items-center gap-2 rounded-md border border-slate-600 px-3 py-2 text-sm text-slate-200 hover:border-sky-400 hover:text-sky-300">
      <ArrowLeft size={16} />
      {label}
    </button>
  );
}
