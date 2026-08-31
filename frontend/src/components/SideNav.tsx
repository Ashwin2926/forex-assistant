"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Signal Feed" },
  { href: "/trading-signals", label: "Trading Signals" },
  { href: "/chart", label: "Chart" },
  { href: "/backtest", label: "Backtesting" },
  { href: "/consensus", label: "Consensus" },
  { href: "/paper-trade", label: "Paper Trading" },
  { href: "/ml", label: "ML" },
  { href: "/rl", label: "RL Agent" },
];

export function SideNav() {
  const pathname = usePathname();
  return (
    <nav className="flex flex-col gap-0.5">
      {LINKS.map(({ href, label }) => {
        // Exact match only for "/" (every other route starts with "/") -- otherwise "/"
        // would highlight as active on every page.
        const active = href === "/" ? pathname === "/" : pathname === href || pathname?.startsWith(`${href}/`);
        return (
          <Link
            key={href}
            href={href}
            className={`rounded-md px-3 py-2 text-sm transition-colors ${
              active
                ? "bg-zinc-200 font-medium text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100"
                : "text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-900 dark:hover:text-zinc-100"
            }`}
          >
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
