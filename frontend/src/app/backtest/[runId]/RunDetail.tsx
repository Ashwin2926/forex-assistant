"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { BacktestRun, Signal } from "@/lib/types";
import { DirectionBadge, StatusBadge } from "@/components/Badges";

export default function RunDetail({ runId }: { runId: string }) {
  const [run, setRun] = useState<BacktestRun | null>(null);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [runData, signalData] = await Promise.all([
          api.getBacktestRun(runId),
          api.getBacktestRunSignals(runId, { status: statusFilter || undefined, limit: 200 }),
        ]);
        setRun(runData);
        setSignals(signalData);
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Failed to load run.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [runId, statusFilter]);

  if (loading) return <p className="text-sm text-zinc-500">Loading…</p>;
  if (error) {
    return (
      <p className="rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950 dark:text-rose-300">
        {error}
      </p>
    );
  }
  if (!run) return null;

  return (
    <div className="flex flex-col gap-8">
      <div>
        <Link href="/backtest" className="text-xs text-zinc-500 underline underline-offset-2">
          ← All runs
        </Link>
        <h1 className="mt-2 text-xl font-semibold tracking-tight">
          {run.pair} · {run.interval} · {run.profile}
        </h1>
        <p className="mt-1 font-mono text-xs text-zinc-500">{run.run_id}</p>
      </div>

      <section className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Hit rate" value={run.hit_rate_pct != null ? `${run.hit_rate_pct}%` : "—"} />
        <Stat label="Signals" value={`${run.directional_signals} (${run.hold_signals} hold)`} />
        <Stat label="Hit / Miss / Expired" value={`${run.hits} / ${run.misses} / ${run.expired}`} />
        <Stat label="Candles evaluated" value={String(run.candles_evaluated)} />
        <Stat label="Avg win" value={run.avg_win_pct != null ? `+${run.avg_win_pct}%` : "—"} />
        <Stat label="Avg loss" value={run.avg_loss_pct != null ? `${run.avg_loss_pct}%` : "—"} />
        <Stat label="Avg confidence (hit)" value={run.avg_confidence_hit != null ? `${run.avg_confidence_hit}%` : "—"} />
        <Stat label="Avg confidence (miss)" value={run.avg_confidence_miss != null ? `${run.avg_confidence_miss}%` : "—"} />
      </section>

      <section>
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
          Rule performance — which rules earn their vote
        </h2>
        <div className="mt-3 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
          <table className="w-full text-left text-sm">
            <thead className="bg-zinc-50 text-xs uppercase text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
              <tr>
                <th className="px-4 py-2">Rule</th>
                <th className="px-4 py-2">Fired</th>
                <th className="px-4 py-2">Agreed with direction</th>
                <th className="px-4 py-2">Hit rate when agreed</th>
              </tr>
            </thead>
            <tbody>
              {run.rule_stats.map((rs) => (
                <tr key={rs.rule} className="border-t border-zinc-100 dark:border-zinc-800">
                  <td className="px-4 py-2 font-mono text-xs">{rs.rule}</td>
                  <td className="px-4 py-2">{rs.fired_count}</td>
                  <td className="px-4 py-2">{rs.directional_signals_agreed}</td>
                  <td className="px-4 py-2">
                    {rs.hit_rate_when_agreed_pct != null ? `${rs.hit_rate_when_agreed_pct}%` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Labeled signals</h2>
          <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} className="select">
            <option value="">All statuses</option>
            <option value="hit">Hit</option>
            <option value="miss">Miss</option>
            <option value="expired">Expired</option>
          </select>
        </div>
        <ul className="mt-3 flex flex-col gap-2">
          {signals.map((s) => (
            <li
              key={s._id}
              className="flex items-center justify-between rounded-lg border border-zinc-200 bg-white px-4 py-2 text-sm dark:border-zinc-800 dark:bg-zinc-900"
            >
              <div className="flex items-center gap-3">
                <DirectionBadge direction={s.direction} />
                <span className="text-xs text-zinc-500">{new Date(s.timestamp).toLocaleString()}</span>
                <span className="text-xs">conf {s.confidence}%</span>
              </div>
              <div className="flex items-center gap-3 text-xs text-zinc-500">
                <span>
                  entry {s.price_at_signal.toFixed(5)} → {s.outcome_price?.toFixed(5)} ({s.outcome_pct_move?.toFixed(4)}%)
                </span>
                <span>{s.candles_to_outcome} candles</span>
                <StatusBadge status={s.status} />
              </div>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-3 dark:border-zinc-800 dark:bg-zinc-900">
      <p className="text-xs text-zinc-500">{label}</p>
      <p className="mt-1 text-lg font-semibold">{value}</p>
    </div>
  );
}
