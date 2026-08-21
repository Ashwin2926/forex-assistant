import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import { AuthGuard } from "@/components/AuthGuard";
import { AuthNav } from "@/components/AuthNav";
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
      <body className="min-h-full flex flex-col bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
        <header className="border-b border-zinc-200 dark:border-zinc-800">
          <div className="mx-auto flex max-w-5xl items-center gap-6 px-6 py-4">
            <span className="text-sm font-semibold tracking-tight">
              Forex Trading Assistant
            </span>
            <nav className="flex gap-4 text-sm text-zinc-500 dark:text-zinc-400">
              <Link href="/" className="hover:text-zinc-900 dark:hover:text-zinc-100">
                Signal Feed
              </Link>
              <Link href="/trading-signals" className="hover:text-zinc-900 dark:hover:text-zinc-100">
                Trading Signals
              </Link>
              <Link href="/chart" className="hover:text-zinc-900 dark:hover:text-zinc-100">
                Chart
              </Link>
              <Link href="/backtest" className="hover:text-zinc-900 dark:hover:text-zinc-100">
                Backtesting
              </Link>
              <Link href="/consensus" className="hover:text-zinc-900 dark:hover:text-zinc-100">
                Consensus
              </Link>
              <Link href="/paper-trade" className="hover:text-zinc-900 dark:hover:text-zinc-100">
                Paper Trading
              </Link>
            </nav>
            <AuthNav />
          </div>
        </header>
        <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-8">
          <AuthGuard>{children}</AuthGuard>
        </main>
      </body>
    </html>
  );
}
