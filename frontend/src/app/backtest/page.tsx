"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ApiError } from "@/lib/api";
import type { BacktestRun } from "@/lib/types";

export default function BacktestPage() {
  const [runs, setRuns] = useState<BacktestRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadRuns() {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listBacktestRuns({ limit: 50 });
      setRuns(data);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load backtest runs. Is the API running?");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadRuns();
  }, []);

  const chartData = [...runs]
    .filter((r) => r.hit_rate_pct != null)
    .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
    .map((r) => ({
      label: new Date(r.created_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }),
      hit_rate: r.hit_rate_pct,
      pair: r.pair,
    }));

  return (
    <div className="flex flex-col gap-8">
      <section>
        <h1 className="text-xl font-semibold tracking-tight">Backtest runs</h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          History of every SMC consensus and PPO agent backtest run. Trigger a new consensus
          backtest from the <Link href="/consensus" className="underline underline-offset-2">Consensus page</Link>,
          or an RL evaluation from the <Link href="/rl" className="underline underline-offset-2">PPO Agent page</Link> —
          both land here.
        </p>
      </section>

      {chartData.length > 1 && (
        <section className="card p-4">
          <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Hit-rate over runs</h2>
          <div className="mt-3 h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-zinc-200 dark:stroke-zinc-800" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} unit="%" />
                <Tooltip />
                <Line type="monotone" dataKey="hit_rate" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} name="Hit rate %" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      <section>
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Run history</h2>
        {error && (
          <p className="mt-4 rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {error}
          </p>
        )}
        {loading && !error && <p className="mt-4 text-sm text-zinc-500">Loading…</p>}
        {!loading && !error && runs.length === 0 && (
          <p className="mt-4 text-sm text-zinc-500">
            No backtests run yet. Run one from the Consensus or PPO Agent page.
          </p>
        )}
        <div className="mt-4 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
          <table className="w-full text-left text-sm">
            <thead className="table-head text-xs uppercase">
              <tr>
                <th className="px-4 py-2">Run</th>
                <th className="px-4 py-2">Pair</th>
                <th className="px-4 py-2">Profile</th>
                <th className="px-4 py-2">Signals</th>
                <th className="px-4 py-2">Hit rate</th>
                <th className="px-4 py-2">Avg win / loss</th>
                <th className="px-4 py-2">When</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id} className="border-t border-zinc-100 dark:border-zinc-800">
                  <td className="px-4 py-2">
                    <Link href={`/backtest/${r.run_id}`} className="font-mono text-xs underline underline-offset-2">
                      {r.run_id}
                    </Link>
                  </td>
                  <td className="px-4 py-2">{r.pair} · {r.interval}</td>
                  <td className="px-4 py-2">{r.profile}</td>
                  <td className="px-4 py-2">{r.directional_signals} ({r.hold_signals} hold)</td>
                  <td className="px-4 py-2">
                    {r.hit_rate_pct != null ? `${r.hit_rate_pct}%` : "—"}
                    <span className="ml-1 text-xs text-zinc-400">
                      ({r.hits}/{r.misses}/{r.expired})
                    </span>
                  </td>
                  <td className="px-4 py-2 text-xs">
                    {r.avg_win_pct != null ? `+${r.avg_win_pct}%` : "—"} / {r.avg_loss_pct != null ? `${r.avg_loss_pct}%` : "—"}
                  </td>
                  <td className="px-4 py-2 text-xs text-zinc-500">{new Date(r.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
