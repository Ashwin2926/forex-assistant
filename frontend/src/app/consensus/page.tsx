"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { INTERVALS, PAIRS, type BacktestRun, type ConsensusBacktestResult, type ConsensusSignal, type StrategyCall } from "@/lib/types";

export default function ConsensusPage() {
  const [pair, setPair] = useState<string>(PAIRS[0]);
  const [interval, setInterval_] = useState<string>("1h");

  const [checking, setChecking] = useState(false);
  const [checkError, setCheckError] = useState<string | null>(null);
  const [strategyCalls, setStrategyCalls] = useState<StrategyCall[] | null>(null);
  const [consensus, setConsensus] = useState<ConsensusSignal | null>(null);
  const [checked, setChecked] = useState(false);

  const [trainFrac, setTrainFrac] = useState(0.7);
  const [backtesting, setBacktesting] = useState(false);
  const [backtestError, setBacktestError] = useState<string | null>(null);
  const [backtestResult, setBacktestResult] = useState<ConsensusBacktestResult | null>(null);

  const [recent, setRecent] = useState<ConsensusSignal[]>([]);
  const [recentLoading, setRecentLoading] = useState(true);

  async function loadRecent() {
    setRecentLoading(true);
    try {
      setRecent(await api.listConsensusSignals({ limit: 20 }));
    } catch {
      // non-critical section of the page -- a failed load here shouldn't block the rest of it
    } finally {
      setRecentLoading(false);
    }
  }

  useEffect(() => {
    loadRecent();
  }, []);

  async function handleCheck() {
    setChecking(true);
    setCheckError(null);
    try {
      const result = await api.generateConsensus(pair, interval);
      setConsensus(result.consensus);
      setStrategyCalls(result.strategy_calls);
      setChecked(true);
      await loadRecent();
    } catch (e) {
      setCheckError(e instanceof ApiError ? e.message : "Consensus check failed.");
    } finally {
      setChecking(false);
    }
  }

  async function handleBacktest() {
    setBacktesting(true);
    setBacktestError(null);
    setBacktestResult(null);
    try {
      setBacktestResult(await api.runConsensusBacktest(pair, interval, { train_frac: trainFrac }));
    } catch (e) {
      setBacktestError(e instanceof ApiError ? e.message : "Consensus backtest failed.");
    } finally {
      setBacktesting(false);
    }
  }

  return (
    <div className="flex flex-col gap-8">
      <section>
        <h1 className="text-xl font-semibold tracking-tight">Multi-strategy consensus</h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          Five independent strategies (trend, Bollinger mean-reversion, support/resistance,
          candlestick patterns, stochastic+ADX) each analyze the same candles. A consensus
          only fires when at least 4 of 5 agree on direction <em>and</em> their entry/exit
          prices land within half an ATR of each other — agreeing on direction alone isn&apos;t
          enough to call it the same trade. This is additive to the existing intraday/swing
          engine, not a replacement for it, and hasn&apos;t been validated the way that engine
          has — run the backtest below before trusting anything it says.
        </p>
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Check consensus now</h2>
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
          <button onClick={handleCheck} disabled={checking} className="btn-primary">
            {checking ? "Checking…" : "Check consensus"}
          </button>
        </div>
        {checkError && (
          <p className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {checkError}
          </p>
        )}

        {checked && (
          <div className="mt-4 flex flex-col gap-4">
            {consensus ? (
              <div className={`rounded-lg border p-4 ${consensus.direction === "BUY" ? "border-emerald-200 bg-emerald-50 dark:border-emerald-900 dark:bg-emerald-950" : "border-rose-200 bg-rose-50 dark:border-rose-900 dark:bg-rose-950"}`}>
                <p className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
                  {consensus.agreeing_count}/5 strategies agree
                </p>
                <p className="mt-1 text-lg font-semibold">
                  {consensus.direction} at {consensus.entry_price.toFixed(5)}, exit at {consensus.target_price.toFixed(5)}
                </p>
                <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                  Stop {consensus.stop_price.toFixed(5)}
                </p>
              </div>
            ) : (
              <p className="rounded-md bg-zinc-50 px-3 py-2 text-sm text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400">
                No consensus right now — see how each strategy called it below.
              </p>
            )}

            <div className="overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
              <table className="w-full text-left text-xs">
                <thead className="bg-zinc-50 uppercase text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
                  <tr>
                    <th className="px-3 py-1.5">Strategy</th>
                    <th className="px-3 py-1.5">Direction</th>
                    <th className="px-3 py-1.5">Entry</th>
                    <th className="px-3 py-1.5">Exit</th>
                  </tr>
                </thead>
                <tbody>
                  {strategyCalls?.map((c) => (
                    <tr key={c.strategy} className="border-t border-zinc-100 dark:border-zinc-800">
                      <td className="px-3 py-1.5 font-mono">{c.strategy}</td>
                      <td className={`px-3 py-1.5 font-medium ${c.direction === "BUY" ? "text-emerald-600 dark:text-emerald-400" : c.direction === "SELL" ? "text-rose-600 dark:text-rose-400" : "text-zinc-400"}`}>
                        {c.direction}
                      </td>
                      <td className="px-3 py-1.5">{c.entry_price.toFixed(5)}</td>
                      <td className="px-3 py-1.5">{c.target_price != null ? c.target_price.toFixed(5) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Backtest the consensus mechanism</h2>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          Same train/test discipline as the single-strategy optimizer — a real edge should
          survive on data the mechanism never saw. Consensus signals are rare by design, so
          don&apos;t expect a large sample size here.
        </p>
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
          <Field label="Train fraction">
            <input
              type="number" step="0.05" min="0.1" max="0.9" value={trainFrac}
              onChange={(e) => setTrainFrac(Number(e.target.value))}
              className="select w-20"
            />
          </Field>
          <button onClick={handleBacktest} disabled={backtesting} className="btn-primary">
            {backtesting ? "Running…" : "Run consensus backtest"}
          </button>
        </div>
        {backtestError && (
          <p className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {backtestError}
          </p>
        )}
        {backtestResult && (
          <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-2">
            <TrainTestCard label="Train" run={backtestResult.train} accent="zinc" />
            <TrainTestCard label="Test (out-of-sample)" run={backtestResult.test} accent="emerald" />
          </div>
        )}
      </section>

      <section>
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Recent consensus signals</h2>
        {recentLoading && <p className="mt-4 text-sm text-zinc-500">Loading…</p>}
        {!recentLoading && recent.length === 0 && (
          <p className="mt-4 text-sm text-zinc-500">No consensus signals yet — check one above.</p>
        )}
        {recent.length > 0 && (
          <div className="mt-4 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
            <table className="w-full text-left text-sm">
              <thead className="bg-zinc-50 text-xs uppercase text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
                <tr>
                  <th className="px-4 py-2">Pair</th>
                  <th className="px-4 py-2">Direction</th>
                  <th className="px-4 py-2">Entry</th>
                  <th className="px-4 py-2">Exit</th>
                  <th className="px-4 py-2">Agree</th>
                  <th className="px-4 py-2">Status</th>
                  <th className="px-4 py-2">When</th>
                </tr>
              </thead>
              <tbody>
                {recent.map((c) => (
                  <tr key={c._id ?? `${c.pair}-${c.timestamp}`} className="border-t border-zinc-100 dark:border-zinc-800">
                    <td className="px-4 py-2">{c.pair} · {c.interval}</td>
                    <td className={`px-4 py-2 font-medium ${c.direction === "BUY" ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                      {c.direction}
                    </td>
                    <td className="px-4 py-2">{c.entry_price.toFixed(5)}</td>
                    <td className="px-4 py-2">{c.target_price.toFixed(5)}</td>
                    <td className="px-4 py-2">{c.agreeing_count}/5</td>
                    <td className="px-4 py-2">{c.status}</td>
                    <td className="px-4 py-2 text-xs text-zinc-500">{new Date(c.timestamp).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
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
        {run.directional_signals} consensus signals ({run.hits}/{run.misses}/{run.expired}) · {run.candles_evaluated} candles evaluated
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
