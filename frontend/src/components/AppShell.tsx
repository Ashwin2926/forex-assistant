"use client";

import { useEffect, useState, type ReactNode } from "react";
import { usePathname } from "next/navigation";
import { AuthGuard } from "@/components/AuthGuard";
import { AuthNav } from "@/components/AuthNav";
import { SideNav } from "@/components/SideNav";
import { SyncNowButton } from "@/components/SyncNowButton";

// Sidebar on desktop, slide-over drawer on mobile -- the old layout was a single fixed
// w-56 <aside> with zero handling below that width, which just clipped/wrapped on a
// phone screen. Below md (768px) it collapses into a top bar with a menu button instead.
export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Close the drawer on navigation rather than leaving it open over the new page.
  useEffect(() => {
    setDrawerOpen(false);
  }, [pathname]);

  return (
    <div className="flex min-h-full w-full flex-col md:flex-row">
      {/* Mobile top bar -- md:hidden so it disappears entirely once the desktop sidebar
          takes over. */}
      <header className="flex items-center justify-between border-b px-4 py-3 md:hidden"
        style={{ borderColor: "var(--border)", background: "var(--surface)" }}
      >
        <button
          onClick={() => setDrawerOpen(true)}
          aria-label="Open navigation"
          className="btn-secondary px-2.5 py-1.5"
        >
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
            <path d="M2 4.5h14M2 9h14M2 13.5h14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </button>
        <span className="text-sm font-semibold tracking-tight">Forex Trading Assistant</span>
        <div className="w-9" />
      </header>

      {/* Mobile drawer + backdrop */}
      {drawerOpen && (
        <div className="fixed inset-0 z-40 md:hidden">
          <div
            className="absolute inset-0 bg-black/40"
            onClick={() => setDrawerOpen(false)}
            aria-hidden="true"
          />
          <aside
            className="absolute inset-y-0 left-0 flex w-72 max-w-[85vw] flex-col border-r"
            style={{ borderColor: "var(--border)", background: "var(--surface)" }}
          >
            <div className="flex items-center justify-between px-4 py-4">
              <span className="text-sm font-semibold tracking-tight">Forex Trading Assistant</span>
              <button
                onClick={() => setDrawerOpen(false)}
                aria-label="Close navigation"
                className="btn-secondary px-2 py-1"
              >
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                  <path d="M1 1l12 12M13 1L1 13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
              </button>
            </div>
            <div className="flex-1 overflow-y-auto px-2">
              <SideNav />
            </div>
            <div className="flex flex-col gap-3 border-t px-3 py-4" style={{ borderColor: "var(--border)" }}>
              <SyncNowButton />
              <AuthNav />
            </div>
          </aside>
        </div>
      )}

      {/* Desktop sidebar -- hidden below md, fixed-width column above it. */}
      <aside
        className="sticky top-0 hidden h-screen w-56 shrink-0 flex-col border-r md:flex"
        style={{ borderColor: "var(--border)", background: "var(--bg)" }}
      >
        <div className="px-4 py-4">
          <span className="whitespace-nowrap text-sm font-semibold tracking-tight">
            Forex Trading Assistant
          </span>
        </div>
        <div className="flex-1 overflow-y-auto px-2">
          <SideNav />
        </div>
        <div className="flex flex-col gap-3 border-t px-3 py-4" style={{ borderColor: "var(--border)" }}>
          <SyncNowButton />
          <AuthNav />
        </div>
      </aside>

      <div className="flex min-h-full flex-1 flex-col">
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-6 sm:py-8">
          <AuthGuard>{children}</AuthGuard>
        </main>
        <footer
          className="border-t px-4 py-3 text-xs sm:px-6"
          style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}
        >
          Build {process.env.NEXT_PUBLIC_VERCEL_GIT_COMMIT_SHA?.slice(0, 7) ?? "dev"}
        </footer>
      </div>
    </div>
  );
}
