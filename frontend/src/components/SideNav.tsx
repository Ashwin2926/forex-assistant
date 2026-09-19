"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/trading-signals", label: "Trading Signals" },
  { href: "/chart", label: "Chart" },
  { href: "/candles", label: "Candles" },
  { href: "/backtest", label: "Backtesting" },
  { href: "/consensus", label: "Consensus" },
  { href: "/paper-trade", label: "Paper Trading" },
  { href: "/ml", label: "ML" },
  { href: "/rl", label: "PPO Agent" },
];

export function SideNav() {
  const pathname = usePathname();
  return (
    <nav className="flex flex-col gap-0.5">
      {LINKS.map(({ href, label }) => {
        const active = pathname === href || pathname?.startsWith(`${href}/`);
        return (
          <Link
            key={href}
            href={href}
            className="rounded-md px-3 py-2 text-sm transition-colors"
            style={
              active
                ? { background: "var(--accent-bg)", color: "var(--accent)", fontWeight: 500 }
                : { color: "var(--text-muted)" }
            }
            onMouseEnter={(e) => {
              if (!active) e.currentTarget.style.background = "var(--surface-sunken)";
            }}
            onMouseLeave={(e) => {
              if (!active) e.currentTarget.style.background = "transparent";
            }}
          >
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
