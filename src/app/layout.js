import "./globals.css";
import { Suspense } from "react";
import { TopHeader } from "@/components/layout/TopHeader";

export const metadata = {
  title: "Fuzzyball OBPI Dashboard",
  description: "Football scouting dashboard for Off-Ball Positional Intelligence"
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen bg-background text-slate-100">
          <Suspense fallback={null}>
            <TopHeader />
          </Suspense>
          <main className="min-h-screen">
            <div className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-8">{children}</div>
          </main>
        </div>
      </body>
    </html>
  );
}
