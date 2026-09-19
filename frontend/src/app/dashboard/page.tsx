"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { DailyPairSignals, DailySignalsResponse } from "@/lib/types";
import { DirectionBadge, StatusBadge } from "@/components/Badges";

// Auto-refresh cadence -- signals resolve and new ones fire continuously through the day
// (create_rl_signal/score_pending_rl_signals both run on keep-fresh.yml's own cron), so
// this page should reflect that without the user having to manually reload.
const REFRESH_MS = 60_000;

export default function DashboardPage() {
  const [data, setData] = useState<DailySignalsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    try {
      setError(null);
      setData(await api.getDailySignals());
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load today's signals. Is the API running?");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    const id = setInterval(load, REFRESH_MS);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex flex-col gap-8">
      <section>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">Dashboard</h1>
            <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
              Today&apos;s trading day, one trained RL policy per pair/interval, checked
              continuously as each new candle closes. A pair/interval with no policy has
              genuinely found no edge yet — that&apos;s an honest answer, not an error, and
              nothing here is ever forced.
            </p>
          </div>
          {data && (
            <span className="shrink-0 text-xs text-zinc-500 dark:text-zinc-400">{data.date} UTC</span>
          )}
        </div>
      </section>

      {error && (
        <p className="rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950 dark:text-rose-300">
          {error}
        </p>
      )}
      {loading && !error && <p className="text-sm text-zinc-500">Loading…</p>}

      {data && !loading && (
        <div className="flex flex-col gap-6">
          {data.pairs.map((pair) => (
            <PairCard key={pair.pair} pair={pair} />
          ))}
        </div>
      )}
    </div>
  );
}

function PairCard({ pair }: { pair: DailyPairSignals }) {
  const totalSignals = pair.intervals.reduce((sum, iv) => sum + iv.signals.length, 0);
  const activeIntervals = pair.intervals.filter((iv) => iv.has_policy).length;

  return (
    <section className="card p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">{pair.pair}</h2>
        <span className="text-xs text-zinc-500 dark:text-zinc-400">
          {activeIntervals}/{pair.intervals.length} intervals trained · {totalSignals} signal{totalSignals === 1 ? "" : "s"} today
        </span>
      </div>

      <div className="mt-3 flex flex-col gap-4">
        {pair.intervals.map((iv) => (
          <div key={iv.interval}>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-medium text-zinc-600 dark:text-zinc-300">{iv.interval}</span>
              {!iv.has_policy && (
                <span className="text-xs text-zinc-400 dark:text-zinc-500">— no edge yet, not trading</span>
              )}
            </div>
            {iv.has_policy && iv.signals.length === 0 && (
              <p className="mt-1 text-xs text-zinc-400 dark:text-zinc-500">Nothing fired today so far.</p>
            )}
            {iv.signals.length > 0 && (
              <div className="mt-1 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
                <table className="w-full text-left text-xs">
                  <thead className="table-head uppercase">
                    <tr>
                      <th className="px-3 py-1.5">Time</th>
                      <th className="px-3 py-1.5">Direction</th>
                      <th className="px-3 py-1.5">Entry</th>
                      <th className="px-3 py-1.5">Target</th>
                      <th className="px-3 py-1.5">Stop</th>
                      <th className="px-3 py-1.5">Confidence</th>
                      <th className="px-3 py-1.5">Size</th>
                      <th className="px-3 py-1.5">Status</th>
                      <th className="px-3 py-1.5">Outcome</th>
                    </tr>
                  </thead>
                  <tbody>
                    {iv.signals.map((s, i) => (
                      <tr key={s.signal_id ?? i} className="border-t border-zinc-100 dark:border-zinc-800">
                        <td className="num px-3 py-1.5">
                          {new Date(s.timestamp).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}
                        </td>
                        <td className="px-3 py-1.5"><DirectionBadge direction={s.direction} /></td>
                        <td className="num px-3 py-1.5">{s.entry_price.toFixed(5)}</td>
                        <td className="num px-3 py-1.5">{s.target_price.toFixed(5)}</td>
                        <td className="num px-3 py-1.5">{s.stop_price.toFixed(5)}</td>
                        <td className="num px-3 py-1.5">{s.confidence_pct.toFixed(0)}%</td>
                        <td className={`px-3 py-1.5 ${s.size_tier === "LARGE" ? "text-amber-600 dark:text-amber-400" : ""}`}>
                          {s.size_tier}
                        </td>
                        <td className="px-3 py-1.5"><StatusBadge status={s.status} /></td>
                        <td className="num px-3 py-1.5">
                          {s.outcome_pct_move != null
                            ? `${s.outcome_pct_move >= 0 ? "+" : ""}${s.outcome_pct_move.toFixed(3)}%`
                            : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
