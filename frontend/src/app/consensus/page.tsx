"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { INTERVALS, PAIRS, type BacktestRun, type ConsensusBacktestResult, type ConsensusSignal, type StrategyCall } from "@/lib/types";

interface GridCell {
  pair: string;
  interval: string;
  consensus: ConsensusSignal | null;
  strategyCalls: StrategyCall[];
  error?: string;
}

interface BacktestGridCell {
  pair: string;
  interval: string;
  result: ConsensusBacktestResult | null;
  error?: string;
}

export default function ConsensusPage() {
  const [grid, setGrid] = useState<GridCell[]>([]);
  const [gridLoading, setGridLoading] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null); // "pair-interval" key of the row showing its strategy calls

  const [pair, setPair] = useState<string>(PAIRS[0]);
  const [interval, setInterval_] = useState<string>("1h");
  const [trainFrac, setTrainFrac] = useState(0.7);
  const [backtesting, setBacktesting] = useState(false);
  const [backtestError, setBacktestError] = useState<string | null>(null);
  const [backtestResult, setBacktestResult] = useState<ConsensusBacktestResult | null>(null);

  const [backtestGrid, setBacktestGrid] = useState<BacktestGridCell[]>([]);
  const [backtestGridProgress, setBacktestGridProgress] = useState(0);
  const [backtestGridRunning, setBacktestGridRunning] = useState(false);

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

  async function checkAll() {
    setGridLoading(true);
    const combos = PAIRS.flatMap((p) => INTERVALS.map((i) => ({ pair: p, interval: i })));
    const results = await Promise.all(
      combos.map(async ({ pair: p, interval: i }): Promise<GridCell> => {
        try {
          const result = await api.generateConsensus(p, i);
          return { pair: p, interval: i, consensus: result.consensus, strategyCalls: result.strategy_calls };
        } catch (e) {
          return { pair: p, interval: i, consensus: null, strategyCalls: [], error: e instanceof ApiError ? e.message : "Failed" };
        }
      }),
    );
    // Firing consensus rows first (the actually useful ones), then everything else in a
    // stable pair/interval order.
    results.sort((a, b) => {
      const aFired = a.consensus ? 1 : 0;
      const bFired = b.consensus ? 1 : 0;
      if (aFired !== bFired) return bFired - aFired;
      return 0;
    });
    setGrid(results);
    setGridLoading(false);
    await loadRecent();
  }

  useEffect(() => {
    checkAll();
    loadRecent();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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

  async function runAllBacktests() {
    setBacktestGridRunning(true);
    setBacktestGrid([]);
    setBacktestGridProgress(0);
    const combos = PAIRS.flatMap((p) => INTERVALS.map((i) => ({ pair: p, interval: i })));
    const results: BacktestGridCell[] = [];
    // Sequential, not Promise.all like the live grid above -- each call replays every
    // candle bar-by-bar across every strategy, far heavier than a single-candle live
    // check, and this hits the same backend instance 20x in a row.
    for (const { pair: p, interval: i } of combos) {
      try {
        const result = await api.runConsensusBacktest(p, i, { train_frac: trainFrac });
        results.push({ pair: p, interval: i, result });
      } catch (e) {
        results.push({ pair: p, interval: i, result: null, error: e instanceof ApiError ? e.message : "Failed" });
      }
      setBacktestGridProgress(results.length);
      setBacktestGrid([...results]);
    }
    setBacktestGridRunning(false);
  }

  const firedCount = grid.filter((c) => c.consensus).length;
  const sortedBacktestGrid = [...backtestGrid].sort((a, b) => {
    const aSignals = a.result?.test.directional_signals ?? -1;
    const bSignals = b.result?.test.directional_signals ?? -1;
    return bSignals - aSignals;
  });

  return (
    <div className="flex flex-col gap-8">
      <section>
        <h1 className="text-xl font-semibold tracking-tight">Multi-strategy consensus</h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          Seven independent strategies (trend, Bollinger mean-reversion, support/resistance,
          candlestick patterns, stochastic+ADX, volume-confirmed momentum, and a liquidity-sweep
          approximation of Smart Money Concepts) each analyze the same candles independently —
          no shared state between them. A consensus only fires when a weighted majority agree on
          direction <em>and</em> their entry/exit prices land within half an ATR of each other.
          Additive to the existing intraday/swing engine, not a replacement — run the backtest
          below before trusting anything it says.
        </p>
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
            All pairs &times; all intervals {grid.length > 0 && `(${firedCount} consensus firing)`}
          </h2>
          <button onClick={checkAll} disabled={gridLoading} className="btn-primary">
            {gridLoading ? "Checking 20 combinations…" : "Refresh all"}
          </button>
        </div>

        {grid.length > 0 && (
          <div className="mt-4 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
            <table className="w-full text-left text-xs">
              <thead className="bg-zinc-50 uppercase text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
                <tr>
                  <th className="px-3 py-1.5">Pair</th>
                  <th className="px-3 py-1.5">Interval</th>
                  <th className="px-3 py-1.5">Result</th>
                  <th className="px-3 py-1.5">Entry</th>
                  <th className="px-3 py-1.5">Exit</th>
                  <th className="px-3 py-1.5">Stop</th>
                  <th className="px-3 py-1.5"></th>
                </tr>
              </thead>
              <tbody>
                {grid.map((cell) => {
                  const key = `${cell.pair}-${cell.interval}`;
                  return (
                    <>
                      <tr
                        key={key}
                        className={`border-t border-zinc-100 dark:border-zinc-800 ${cell.consensus ? (cell.consensus.direction === "BUY" ? "bg-emerald-50 dark:bg-emerald-950" : "bg-rose-50 dark:bg-rose-950") : ""}`}
                      >
                        <td className="px-3 py-1.5 font-mono">{cell.pair}</td>
                        <td className="px-3 py-1.5 font-mono">{cell.interval}</td>
                        <td className="px-3 py-1.5">
                          {cell.error ? (
                            <span className="text-zinc-400">{cell.error}</span>
                          ) : cell.consensus ? (
                            <span className={`font-semibold ${cell.consensus.direction === "BUY" ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                              {cell.consensus.direction} · {cell.consensus.agreeing_count}/{cell.strategyCalls.length} agree
                            </span>
                          ) : (
                            <span className="text-zinc-400">No consensus</span>
                          )}
                        </td>
                        <td className="px-3 py-1.5">{cell.consensus ? cell.consensus.entry_price.toFixed(5) : "—"}</td>
                        <td className="px-3 py-1.5">{cell.consensus ? cell.consensus.target_price.toFixed(5) : "—"}</td>
                        <td className="px-3 py-1.5">{cell.consensus ? cell.consensus.stop_price.toFixed(5) : "—"}</td>
                        <td className="px-3 py-1.5">
                          <button
                            onClick={() => setExpanded(expanded === key ? null : key)}
                            className="text-zinc-500 underline underline-offset-2 hover:text-zinc-900 dark:hover:text-zinc-100"
                          >
                            {expanded === key ? "Hide" : "Show"} calls
                          </button>
                        </td>
                      </tr>
                      {expanded === key && (
                        <tr key={`${key}-detail`} className="border-t border-zinc-100 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-800">
                          <td colSpan={7} className="px-3 py-2">
                            <div className="flex flex-wrap gap-x-6 gap-y-1">
                              {cell.strategyCalls.map((c) => (
                                <span key={c.strategy} className="font-mono">
                                  {c.strategy}:{" "}
                                  <span className={c.direction === "BUY" ? "text-emerald-600 dark:text-emerald-400" : c.direction === "SELL" ? "text-rose-600 dark:text-rose-400" : "text-zinc-400"}>
                                    {c.direction}
                                  </span>
                                  {c.target_price != null && ` @ ${c.entry_price.toFixed(5)} → ${c.target_price.toFixed(5)}`}
                                </span>
                              ))}
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  );
                })}
              </tbody>
            </table>
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

        <div className="mt-6 border-t border-zinc-100 pt-4 dark:border-zinc-800">
          <div className="flex items-center justify-between">
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Or run the same train/test backtest across all 4 pairs &times; 5 intervals at
              once, using the train fraction above.
            </p>
            <button onClick={runAllBacktests} disabled={backtestGridRunning} className="btn-primary shrink-0">
              {backtestGridRunning ? `Running ${backtestGridProgress}/20…` : "Run all backtests"}
            </button>
          </div>

          {backtestGrid.length > 0 && (
            <div className="mt-4 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
              <table className="w-full text-left text-xs">
                <thead className="bg-zinc-50 uppercase text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
                  <tr>
                    <th className="px-3 py-1.5">Pair</th>
                    <th className="px-3 py-1.5">Interval</th>
                    <th className="px-3 py-1.5">Train hit rate</th>
                    <th className="px-3 py-1.5">Train expectancy</th>
                    <th className="px-3 py-1.5">Test hit rate</th>
                    <th className="px-3 py-1.5">Test expectancy</th>
                    <th className="px-3 py-1.5">Test signals</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedBacktestGrid.map((cell) => {
                    const key = `${cell.pair}-${cell.interval}`;
                    const test = cell.result?.test;
                    const train = cell.result?.train;
                    const sameSign = test && train && test.expectancy_pct != null && train.expectancy_pct != null
                      && Math.sign(test.expectancy_pct) === Math.sign(train.expectancy_pct) && test.expectancy_pct > 0;
                    return (
                      <tr key={key} className={`border-t border-zinc-100 dark:border-zinc-800 ${sameSign ? "bg-emerald-50 dark:bg-emerald-950" : ""}`}>
                        <td className="px-3 py-1.5 font-mono">{cell.pair}</td>
                        <td className="px-3 py-1.5 font-mono">{cell.interval}</td>
                        {cell.error ? (
                          <td className="px-3 py-1.5 text-zinc-400" colSpan={5}>{cell.error}</td>
                        ) : (
                          <>
                            <td className="px-3 py-1.5">{train?.hit_rate_pct != null ? `${train.hit_rate_pct}%` : "—"}</td>
                            <td className={`px-3 py-1.5 ${train?.expectancy_pct != null && train.expectancy_pct >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                              {train?.expectancy_pct != null ? `${train.expectancy_pct >= 0 ? "+" : ""}${train.expectancy_pct}%` : "—"}
                            </td>
                            <td className="px-3 py-1.5">{test?.hit_rate_pct != null ? `${test.hit_rate_pct}%` : "—"}</td>
                            <td className={`px-3 py-1.5 ${test?.expectancy_pct != null && test.expectancy_pct >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                              {test?.expectancy_pct != null ? `${test.expectancy_pct >= 0 ? "+" : ""}${test.expectancy_pct}%` : "—"}
                            </td>
                            <td className="px-3 py-1.5">{test?.directional_signals ?? "—"}</td>
                          </>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="px-3 py-2 text-xs text-zinc-400 dark:text-zinc-600">
                Highlighted rows: test expectancy is positive and the same sign as train —
                the closest this view gets to "worth trusting," and even then only with a
                large enough test-signal count to mean anything.
              </p>
            </div>
          )}
        </div>
      </section>

      <section>
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Recent consensus signals</h2>
        {recentLoading && <p className="mt-4 text-sm text-zinc-500">Loading…</p>}
        {!recentLoading && recent.length === 0 && (
          <p className="mt-4 text-sm text-zinc-500">No consensus signals yet.</p>
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
                    <td className="px-4 py-2">{c.agreeing_count}/{c.strategy_calls.length}</td>
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
