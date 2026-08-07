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
import { INTERVALS, PAIRS, PROFILES, type BacktestRun, type OptimizeRankBy, type OptimizeResult, type Profile } from "@/lib/types";

export default function BacktestPage() {
  const [runs, setRuns] = useState<BacktestRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [pair, setPair] = useState<string>(PAIRS[0]);
  const [interval, setInterval_] = useState<string>("1h");
  const [profile, setProfile] = useState<Profile>("swing");
  const [targetAtrMult, setTargetAtrMult] = useState(1.5);
  const [stopAtrMult, setStopAtrMult] = useState(1.0);
  const [maxLookforward, setMaxLookforward] = useState(20);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const [optPair, setOptPair] = useState<string>(PAIRS[0]);
  const [optInterval, setOptInterval] = useState<string>("15min");
  const [optProfile, setOptProfile] = useState<Profile>("intraday");
  const [trainFrac, setTrainFrac] = useState(0.7);
  const [minSignals, setMinSignals] = useState(20);
  const [rankBy, setRankBy] = useState<OptimizeRankBy>("expectancy");
  const [optimizing, setOptimizing] = useState(false);
  const [optimizeError, setOptimizeError] = useState<string | null>(null);
  const [optimizeResult, setOptimizeResult] = useState<OptimizeResult | null>(null);

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

  async function handleRunBacktest() {
    setRunning(true);
    setRunError(null);
    try {
      await api.runBacktest(pair, interval, profile, {
        target_atr_mult: targetAtrMult,
        stop_atr_mult: stopAtrMult,
        max_lookforward: maxLookforward,
      });
      await loadRuns();
    } catch (e) {
      setRunError(e instanceof ApiError ? e.message : "Backtest failed.");
    } finally {
      setRunning(false);
    }
  }

  async function handleRunOptimize() {
    setOptimizing(true);
    setOptimizeError(null);
    setOptimizeResult(null);
    try {
      const result = await api.runOptimize(optPair, optInterval, optProfile, {
        train_frac: trainFrac,
        min_directional_signals: minSignals,
        rank_by: rankBy,
      });
      setOptimizeResult(result);
      await loadRuns();
    } catch (e) {
      setOptimizeError(e instanceof ApiError ? e.message : "Optimize failed.");
    } finally {
      setOptimizing(false);
    }
  }

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
        <h1 className="text-xl font-semibold tracking-tight">Backtesting</h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          Replay the rule engine bar-by-bar over stored history. No live signal is trusted
          until its ruleset has a track record here.
        </p>
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Run a backtest</h2>
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <Field label="Pair">
            <select value={pair} onChange={(e) => setPair(e.target.value)} className="select">
              {PAIRS.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </Field>
          <Field label="Interval">
            <select value={interval} onChange={(e) => setInterval_(e.target.value)} className="select">
              {INTERVALS.map((i) => <option key={i} value={i}>{i}</option>)}
            </select>
          </Field>
          <Field label="Profile">
            <select value={profile} onChange={(e) => setProfile(e.target.value as Profile)} className="select">
              {PROFILES.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </Field>
          <Field label="Target (x ATR)">
            <input
              type="number" step="0.1" value={targetAtrMult}
              onChange={(e) => setTargetAtrMult(Number(e.target.value))}
              className="select w-20"
            />
          </Field>
          <Field label="Stop (x ATR)">
            <input
              type="number" step="0.1" value={stopAtrMult}
              onChange={(e) => setStopAtrMult(Number(e.target.value))}
              className="select w-20"
            />
          </Field>
          <Field label="Max lookforward">
            <input
              type="number" step="1" value={maxLookforward}
              onChange={(e) => setMaxLookforward(Number(e.target.value))}
              className="select w-24"
            />
          </Field>
          <button onClick={handleRunBacktest} disabled={running} className="btn-primary">
            {running ? "Running…" : "Run backtest"}
          </button>
        </div>
        {runError && (
          <p className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {runError}
          </p>
        )}
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Optimize (train/test validated)</h2>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          Grid-searches RuleConfig on the first <code className="font-mono">train_frac</code> of
          history, picks the best hit-rate there, then re-scores that exact config on the
          untouched remaining tail. A real improvement survives on data it never saw during
          tuning — a large drop from train to test means the &quot;winner&quot; was curve-fit noise, not signal.
        </p>
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <Field label="Pair">
            <select value={optPair} onChange={(e) => setOptPair(e.target.value)} className="select">
              {PAIRS.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </Field>
          <Field label="Interval">
            <select value={optInterval} onChange={(e) => setOptInterval(e.target.value)} className="select">
              {INTERVALS.map((i) => <option key={i} value={i}>{i}</option>)}
            </select>
          </Field>
          <Field label="Profile">
            <select value={optProfile} onChange={(e) => setOptProfile(e.target.value as Profile)} className="select">
              {PROFILES.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </Field>
          <Field label="Train fraction">
            <input
              type="number" step="0.05" min="0.1" max="0.9" value={trainFrac}
              onChange={(e) => setTrainFrac(Number(e.target.value))}
              className="select w-20"
            />
          </Field>
          <Field label="Min signals">
            <input
              type="number" step="1" value={minSignals}
              onChange={(e) => setMinSignals(Number(e.target.value))}
              className="select w-24"
            />
          </Field>
          <Field label="Rank by">
            <select value={rankBy} onChange={(e) => setRankBy(e.target.value as OptimizeRankBy)} className="select">
              <option value="expectancy">Expectancy (recommended)</option>
              <option value="hit_rate">Hit rate</option>
            </select>
          </Field>
          <button onClick={handleRunOptimize} disabled={optimizing} className="btn-primary">
            {optimizing ? "Optimizing…" : "Run optimize"}
          </button>
        </div>
        {optimizeError && (
          <p className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {optimizeError}
          </p>
        )}

        {optimizeResult && (
          <div className="mt-4 flex flex-col gap-4">
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Ranked by <span className="font-medium">{optimizeResult.rank_by === "expectancy" ? "expectancy" : "hit rate"}</span>.
              Both metrics are shown below — they can disagree (a lower hit-rate config can still
              have better expected value if its wins are bigger relative to its losses).
            </p>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-2">
              <TrainTestCard label="Train" run={optimizeResult.train} accent="zinc" />
              <TrainTestCard label="Test (out-of-sample)" run={optimizeResult.test} accent="emerald" />
            </div>

            <div>
              <p className="text-xs font-medium text-zinc-500 dark:text-zinc-400">Winning config</p>
              <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 rounded-md bg-zinc-50 px-3 py-2 text-xs font-mono dark:bg-zinc-800">
                <span>EMA {optimizeResult.winning_config.ema_fast}/{optimizeResult.winning_config.ema_slow}</span>
                <span>RSI {optimizeResult.winning_config.rsi_oversold}/{optimizeResult.winning_config.rsi_overbought}</span>
                <span>target/stop {optimizeResult.winning_config.target_atr_mult}x/{optimizeResult.winning_config.stop_atr_mult}x ATR</span>
                <span>vol_threshold {optimizeResult.winning_config.volatility_threshold_pct}%</span>
                <span>
                  session_filter {optimizeResult.winning_config.session_filter_enabled
                    ? `${optimizeResult.winning_config.session_start_hour_utc}:00-${optimizeResult.winning_config.session_end_hour_utc}:00 UTC`
                    : "off"}
                </span>
              </div>
            </div>

            <div>
              <p className="text-xs font-medium text-zinc-500 dark:text-zinc-400">All candidates (train slice)</p>
              <div className="mt-1 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
                <table className="w-full text-left text-xs">
                  <thead className="bg-zinc-50 uppercase text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
                    <tr>
                      <th className="px-3 py-1.5">EMA</th>
                      <th className="px-3 py-1.5">RSI</th>
                      <th className="px-3 py-1.5">Target/Stop</th>
                      <th className="px-3 py-1.5">Session</th>
                      <th className="px-3 py-1.5">Signals</th>
                      <th className="px-3 py-1.5">Hit rate</th>
                      <th className="px-3 py-1.5">Expectancy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...optimizeResult.candidates_evaluated]
                      .sort((a, b) => {
                        const key = optimizeResult.rank_by === "expectancy" ? "expectancy_pct" : "hit_rate_pct";
                        return (b[key] ?? -Infinity) - (a[key] ?? -Infinity);
                      })
                      .map((c, i) => {
                        const isWinner = JSON.stringify(c.config) === JSON.stringify(optimizeResult.winning_config);
                        return (
                          <tr
                            key={i}
                            className={`border-t border-zinc-100 dark:border-zinc-800 ${isWinner ? "bg-emerald-50 dark:bg-emerald-950" : ""}`}
                          >
                            <td className="px-3 py-1.5 font-mono">{c.config.ema_fast}/{c.config.ema_slow}</td>
                            <td className="px-3 py-1.5 font-mono">{c.config.rsi_oversold}/{c.config.rsi_overbought}</td>
                            <td className="px-3 py-1.5 font-mono">{c.config.target_atr_mult}x/{c.config.stop_atr_mult}x</td>
                            <td className="px-3 py-1.5">{c.config.session_filter_enabled ? "on" : "off"}</td>
                            <td className="px-3 py-1.5">{c.directional_signals}</td>
                            <td className="px-3 py-1.5">{c.hit_rate_pct != null ? `${c.hit_rate_pct}%` : "—"}</td>
                            <td className="px-3 py-1.5">
                              {c.expectancy_pct != null ? `${c.expectancy_pct >= 0 ? "+" : ""}${c.expectancy_pct}%` : "—"}
                              {isWinner && <span className="ml-1 text-emerald-700 dark:text-emerald-400">← winner</span>}
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </section>

      {chartData.length > 1 && (
        <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
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
          <p className="mt-4 text-sm text-zinc-500">No backtests run yet. Use the form above.</p>
        )}
        <div className="mt-4 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
          <table className="w-full text-left text-sm">
            <thead className="bg-zinc-50 text-xs uppercase text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
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

function TrainTestCard({ label, run, accent }: { label: string; run: BacktestRun; accent: "zinc" | "emerald" }) {
  const border = accent === "emerald" ? "border-emerald-200 dark:border-emerald-900" : "border-zinc-200 dark:border-zinc-800";
  const bg = accent === "emerald" ? "bg-emerald-50 dark:bg-emerald-950" : "bg-zinc-50 dark:bg-zinc-800";
  return (
    <div className={`rounded-lg border ${border} ${bg} p-3`}>
      <p className="text-xs text-zinc-500 dark:text-zinc-400">{label}</p>
      <div className="mt-1 flex items-baseline gap-3">
        <p className="text-2xl font-semibold">{run.hit_rate_pct != null ? `${run.hit_rate_pct}%` : "—"}</p>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          hit rate · expectancy{" "}
          <span className={run.expectancy_pct != null && run.expectancy_pct >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}>
            {run.expectancy_pct != null ? `${run.expectancy_pct >= 0 ? "+" : ""}${run.expectancy_pct}%` : "—"}
          </span>
        </p>
      </div>
      <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
        {run.directional_signals} signals ({run.hits}/{run.misses}/{run.expired}) · {run.candles_evaluated} candles
      </p>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-zinc-500">
      {label}
      {children}
    </label>
  );
}
