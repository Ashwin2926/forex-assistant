import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { AuthGuard } from "@/components/AuthGuard";
import { AuthNav } from "@/components/AuthNav";
import { SideNav } from "@/components/SideNav";
import { SyncNowButton } from "@/components/SyncNowButton";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});
  
const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});  

export const metadata: Metadata = {
  title: "Forex Trading Assistant",
  description: "Rule-based forex signals with full reasoning, backtested before they're trusted.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="flex min-h-full bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
        {/* Side nav replaces the old top nav (which had grown to 8 links and was wrapping
            onto multiple broken lines on narrower viewports) -- fixed-width column, its own
            scroll for the link list so a future 9th/10th page doesn't push the footer
            controls off screen. */}
        {/* fixed, not just a flex sibling -- stays in place while the main content scrolls
            independently, instead of scrolling away with the page on a tall page. */}
        <aside className="fixed inset-y-0 left-0 flex w-56 shrink-0 flex-col border-r border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950">
          <div className="px-4 py-4">
            <span className="whitespace-nowrap text-sm font-semibold tracking-tight">
              Forex Trading Assistant
            </span>
          </div>
          <div className="flex-1 overflow-y-auto px-2">
            <SideNav />
          </div>
          <div className="flex flex-col gap-3 border-t border-zinc-200 px-3 py-4 dark:border-zinc-800">
            <SyncNowButton />
            <AuthNav />
          </div>
        </aside>
        {/* ml-56 matches the fixed sidebar's own width, taking its place in normal flow now
            that the sidebar itself is out of it. */}
        <div className="ml-56 flex min-h-full flex-1 flex-col">
          <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-8">
            <AuthGuard>{children}</AuthGuard>
          </main>
          <footer className="border-t border-zinc-200 px-6 py-3 text-xs text-zinc-400 dark:border-zinc-800 dark:text-zinc-600">
            {/* NEXT_PUBLIC_VERCEL_GIT_COMMIT_SHA is auto-injected by Vercel at build time
                (requires "Automatically expose System Environment Variables" in project
                settings) -- lets a deploy be confirmed as picking up the latest push
                without needing a throwaway commit each time. */}
            Build {process.env.NEXT_PUBLIC_VERCEL_GIT_COMMIT_SHA?.slice(0, 7) ?? "dev"}
          </footer>
        </div>
      </body>
    </html>
  );
}
