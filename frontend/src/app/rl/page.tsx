"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { INTERVALS, PAIRS, type BacktestRun, type RLPolicy, type RLSignal, type Signal } from "@/lib/types";
import { StatusBadge } from "@/components/Badges";

// Same formula as trading-signals/page.tsx, duplicated rather than shared -- this project
// keeps sizing logic local to whichever page displays it, not in lib/. RL signals are meant
// to be executed manually on whatever broker you actually have (Deriv isn't available in
// every region), so a lot size needs to travel with the signal on its own.
function usdPerUnit(pair: string, entryPrice: number): number {
  return pair === "USD/JPY" ? 1 / entryPrice : 1;
}

function calculateLotSize(pair: string, entryPrice: number, stopPrice: number, riskAmountUsd: number): number {
  const stopDistance = Math.abs(entryPrice - stopPrice);
  if (stopDistance === 0) return 0;
  const units = riskAmountUsd / (stopDistance * usdPerUnit(pair, entryPrice));
  return units / 100_000;
}

interface TrainAllCell {
  pair: string;
  interval: string;
  evaluation: BacktestRun | null;
  policy: RLPolicy | null;
  error?: string;
}

interface GenerateAllCell {
  pair: string;
  interval: string;
  signal: RLSignal | null;
  qValues: Record<string, number> | null;
  error?: string;
}

export default function RLPage() {
  const [accountBalance, setAccountBalance] = useState(10000);
  const [riskPercent, setRiskPercent] = useState(1);

  const [pair, setPair] = useState<string>(PAIRS[0]);
  const [interval, setInterval_] = useState<string>("1h");
  const [episodes, setEpisodes] = useState(100);
  const [trainFrac, setTrainFrac] = useState(0.7);
  const [training, setTraining] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainResult, setTrainResult] = useState<{ policy: RLPolicy; evaluation: BacktestRun } | null>(null);

  const [policies, setPolicies] = useState<RLPolicy[]>([]);
  const [policiesLoading, setPoliciesLoading] = useState(true);

  const [signalPair, setSignalPair] = useState<string>(PAIRS[0]);
  const [signalInterval, setSignalInterval] = useState<string>("1h");
  const [generating, setGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [generated, setGenerated] = useState<{ signal: RLSignal | null; q_values: Record<string, number> } | null>(null);

  const [recentSignals, setRecentSignals] = useState<RLSignal[]>([]);
  const [recentLoading, setRecentLoading] = useState(true);

  const [trainAllResults, setTrainAllResults] = useState<TrainAllCell[]>([]);
  const [trainAllProgress, setTrainAllProgress] = useState(0);
  const [trainAllRunning, setTrainAllRunning] = useState(false);
  const [expandedWeights, setExpandedWeights] = useState<string | null>(null);

  const [expandedHistoryWeights, setExpandedHistoryWeights] = useState<string | null>(null);
  const [expandedHistoryTrades, setExpandedHistoryTrades] = useState<string | null>(null);
  const [tradeLogs, setTradeLogs] = useState<Record<string, Signal[]>>({});
  const [tradeLogsLoading, setTradeLogsLoading] = useState<string | null>(null);

  const [generateAllResults, setGenerateAllResults] = useState<GenerateAllCell[]>([]);
  const [generateAllProgress, setGenerateAllProgress] = useState(0);
  const [generateAllRunning, setGenerateAllRunning] = useState(false);

  async function loadPolicies() {
    setPoliciesLoading(true);
    try {
      setPolicies(await api.listRLPolicies({ limit: 20 }));
    } catch {
      // non-critical section -- a failed load here shouldn't block the rest of the page
    } finally {
      setPoliciesLoading(false);
    }
  }

  async function loadRecentSignals() {
    setRecentLoading(true);
    try {
      setRecentSignals(await api.listRLSignals({ limit: 20 }));
    } catch {
      // non-critical section
    } finally {
      setRecentLoading(false);
    }
  }

  useEffect(() => {
    loadPolicies();
    loadRecentSignals();
  }, []);

  async function handleTrain() {
    setTraining(true);
    setTrainError(null);
    try {
      const result = await api.trainRLPolicy(pair, interval, { episodes, train_frac: trainFrac });
      setTrainResult(result);
      await loadPolicies();
    } catch (e) {
      setTrainError(e instanceof ApiError ? e.message : "Training failed.");
    } finally {
      setTraining(false);
    }
  }

  async function handleGenerate() {
    setGenerating(true);
    setGenerateError(null);
    setGenerated(null);
    try {
      const result = await api.generateRLSignal(signalPair, signalInterval);
      setGenerated(result);
      await loadRecentSignals();
    } catch (e) {
      setGenerateError(e instanceof ApiError ? e.message : "Signal generation failed.");
    } finally {
      setGenerating(false);
    }
  }

  async function handleTrainAll() {
    setTrainAllRunning(true);
    setTrainAllResults([]);
    setTrainAllProgress(0);
    const combos = PAIRS.flatMap((p) => INTERVALS.map((i) => ({ pair: p, interval: i })));
    const results: TrainAllCell[] = [];
    // Sequential, not Promise.all -- same reasoning as the consensus/ML "run all" grids:
    // each call is a real training run (many episodes over that interval's full candle
    // history), and 20 of those hitting the same backend instance at once would be far
    // heavier than 20 concurrent live checks. Each cell catches its own error and the loop
    // keeps going, so one pair/interval timing out (5min/15min are the likeliest candidates
    // -- see PROGRESS.md) doesn't block the other 19.
    for (const { pair: p, interval: i } of combos) {
      try {
        const result = await api.trainRLPolicy(p, i, { episodes, train_frac: trainFrac });
        results.push({ pair: p, interval: i, evaluation: result.evaluation, policy: result.policy });
      } catch (e) {
        results.push({ pair: p, interval: i, evaluation: null, policy: null, error: e instanceof ApiError ? e.message : "Failed" });
      }
      setTrainAllProgress(results.length);
      setTrainAllResults([...results]);
    }
    setTrainAllRunning(false);
    await loadPolicies();
  }

  async function handleGenerateAll() {
    setGenerateAllRunning(true);
    setGenerateAllResults([]);
    setGenerateAllProgress(0);
    const combos = PAIRS.flatMap((p) => INTERVALS.map((i) => ({ pair: p, interval: i })));
    const results: GenerateAllCell[] = [];
    for (const { pair: p, interval: i } of combos) {
      try {
        const result = await api.generateRLSignal(p, i);
        results.push({ pair: p, interval: i, signal: result.signal, qValues: result.q_values });
      } catch (e) {
        results.push({ pair: p, interval: i, signal: null, qValues: null, error: e instanceof ApiError ? e.message : "Failed" });
      }
      setGenerateAllProgress(results.length);
      setGenerateAllResults([...results]);
    }
    setGenerateAllRunning(false);
    await loadRecentSignals();
  }

  async function handleShowTrades(policy: RLPolicy) {
    const key = policy.policy_id;
    if (expandedHistoryTrades === key) {
      setExpandedHistoryTrades(null);
      return;
    }
    setExpandedHistoryTrades(key);
    if (tradeLogs[policy.eval_run_id]) return; // already cached
    setTradeLogsLoading(policy.eval_run_id);
    try {
      const signals = await api.getBacktestRunSignals(policy.eval_run_id);
      setTradeLogs((prev) => ({ ...prev, [policy.eval_run_id]: signals }));
    } catch {
      setTradeLogs((prev) => ({ ...prev, [policy.eval_run_id]: [] }));
    } finally {
      setTradeLogsLoading(null);
    }
  }

  return (
    <div className="flex flex-col gap-8">
      <section>
        <h1 className="text-xl font-semibold tracking-tight">RL agent (v1)</h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          A linear Q-learning agent that learns how to weight the same 7 strategies consensus
          uses, instead of a fixed vote threshold — one independent policy per pair <em>and</em>{" "}
          interval, trained on historical candle replay (not live signal outcomes). Every trade
          it takes uses a fixed 1.5:1 target:stop ATR ratio, so it can never risk more than it
          stands to gain. Additive and separate from the regular signal feed, consensus, and
          the ML classifier — doesn&apos;t touch any of them. No broker execution here —
          Deriv isn&apos;t available in every region, so signals are meant to be sized (below)
          and traded manually on whatever broker you actually have.
        </p>
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Position sizing</h2>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          Applied to every signal below — lot size = (account balance &times; risk %) &divide;
          (stop distance in price &times; USD value per unit), same formula and same 4-pair
          quote-currency assumptions as the Trading Signals page.
        </p>
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <Field label="Account balance (USD)">
            <input
              type="number" step="100" min="0" value={accountBalance}
              onChange={(e) => setAccountBalance(Number(e.target.value))}
              className="select w-32"
            />
          </Field>
          <Field label="Risk per trade (%)">
            <input
              type="number" step="0.1" min="0.1" max="10" value={riskPercent}
              onChange={(e) => setRiskPercent(Number(e.target.value))}
              className="select w-24"
            />
          </Field>
        </div>
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Train a policy</h2>
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
          <Field label="Episodes">
            <input
              type="number" step="10" min="10" value={episodes}
              onChange={(e) => setEpisodes(Number(e.target.value))}
              className="select w-24"
            />
          </Field>
          <Field label="Train fraction">
            <input
              type="number" step="0.05" min="0.1" max="0.9" value={trainFrac}
              onChange={(e) => setTrainFrac(Number(e.target.value))}
              className="select w-20"
            />
          </Field>
          <button onClick={handleTrain} disabled={training} className="btn-primary">
            {training ? "Training…" : "Train"}
          </button>
        </div>
        {trainError && (
          <p className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {trainError}
          </p>
        )}
        {trainResult && (
          <div className="mt-4 flex flex-col gap-4">
            <div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Test-slice evaluation (a real backtest run, greedy/no exploration) — directly
                comparable to any other approach&apos;s expectancy on the same pair/interval.
              </p>
              <div className="mt-2 max-w-xs">
                <TrainTestCard run={trainResult.evaluation} />
              </div>
            </div>
            <div>
              <p className="text-xs font-medium text-zinc-500 dark:text-zinc-400">What it learned to value</p>
              <p className="mt-1 text-xs text-zinc-400 dark:text-zinc-500">
                Reward/punishment isn&apos;t logged trade-by-trade — it&apos;s baked directly
                into these weights during training. For each action, a positive weight on a
                strategy&apos;s vote means agreeing with that strategy got reinforced
                (led to reward); negative means it got punished (led to loss).
              </p>
              <WeightsTable policy={trainResult.policy} />
            </div>
          </div>
        )}

        <div className="mt-6 border-t border-zinc-100 pt-4 dark:border-zinc-800">
          <div className="flex items-center justify-between">
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Or train all 4 pairs &times; 5 intervals at once (using the episodes/train
              fraction above) — same thing the cron does once daily, run on demand.
            </p>
            <button onClick={handleTrainAll} disabled={trainAllRunning} className="btn-primary shrink-0">
              {trainAllRunning ? `Training ${trainAllProgress}/20…` : "Train all"}
            </button>
          </div>
          {trainAllResults.length > 0 && (
            <div className="mt-4 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
              <table className="w-full text-left text-xs">
                <thead className="bg-zinc-50 uppercase text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
                  <tr>
                    <th className="px-3 py-1.5">Pair</th>
                    <th className="px-3 py-1.5">Interval</th>
                    <th className="px-3 py-1.5">Test hit rate</th>
                    <th className="px-3 py-1.5">Test expectancy</th>
                    <th className="px-3 py-1.5">Trades</th>
                    <th className="px-3 py-1.5">Holds</th>
                    <th className="px-3 py-1.5"></th>
                  </tr>
                </thead>
                <tbody>
                  {trainAllResults.map((cell) => {
                    const key = `${cell.pair}-${cell.interval}`;
                    const run = cell.evaluation;
                    const isExpanded = expandedWeights === key;
                    return (
                      <>
                        <tr key={key} className="border-t border-zinc-100 dark:border-zinc-800">
                          <td className="px-3 py-1.5 font-mono">{cell.pair}</td>
                          <td className="px-3 py-1.5 font-mono">{cell.interval}</td>
                          {cell.error ? (
                            <td className="px-3 py-1.5 text-zinc-400" colSpan={5}>{cell.error}</td>
                          ) : (
                            <>
                              <td className="px-3 py-1.5">{run?.hit_rate_pct != null ? `${run.hit_rate_pct}%` : "—"}</td>
                              <td className={`px-3 py-1.5 ${run?.expectancy_pct != null && run.expectancy_pct >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                                {run?.expectancy_pct != null ? `${run.expectancy_pct >= 0 ? "+" : ""}${run.expectancy_pct}%` : "—"}
                              </td>
                              <td className="px-3 py-1.5">{run?.directional_signals ?? "—"}</td>
                              <td className="px-3 py-1.5">{run?.hold_signals ?? "—"}</td>
                              <td className="px-3 py-1.5">
                                <button
                                  onClick={() => setExpandedWeights(isExpanded ? null : key)}
                                  className="text-zinc-500 underline underline-offset-2 hover:text-zinc-900 dark:hover:text-zinc-100"
                                >
                                  {isExpanded ? "Hide" : "Show"} weights
                                </button>
                              </td>
                            </>
                          )}
                        </tr>
                        {isExpanded && cell.policy && (
                          <tr key={`${key}-weights`} className="border-t border-zinc-100 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-800">
                            <td colSpan={7} className="px-3 py-2">
                              <WeightsTable policy={cell.policy} />
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
        </div>
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Generate a signal</h2>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          Loads the latest trained policy for this pair/interval and picks its greedy action on
          the current candle. Only BUY/SELL get stored — a HOLD just shows the q_values below.
        </p>
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <Field label="Pair">
            <select value={signalPair} onChange={(e) => setSignalPair(e.target.value)} className="select">
              {PAIRS.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </Field>
          <Field label="Interval">
            <select value={signalInterval} onChange={(e) => setSignalInterval(e.target.value)} className="select">
              {INTERVALS.map((i) => <option key={i} value={i}>{i}</option>)}
            </select>
          </Field>
          <button onClick={handleGenerate} disabled={generating} className="btn-primary">
            {generating ? "Generating…" : "Generate"}
          </button>
        </div>
        {generateError && (
          <p className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {generateError}
          </p>
        )}
        {generated && (
          <div className="mt-4 rounded-lg border border-zinc-200 p-4 dark:border-zinc-800">
            {generated.signal ? (
              <>
                <p className={`text-lg font-semibold ${generated.signal.direction === "BUY" ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                  {generated.signal.direction} at {generated.signal.entry_price.toFixed(5)}
                </p>
                <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                  Target {generated.signal.target_price.toFixed(5)} · Stop {generated.signal.stop_price.toFixed(5)}
                </p>
                <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                  Lot size {calculateLotSize(
                    generated.signal.pair, generated.signal.entry_price, generated.signal.stop_price,
                    accountBalance * (riskPercent / 100),
                  ).toFixed(2)} (risking ${(accountBalance * (riskPercent / 100)).toFixed(0)})
                </p>
              </>
            ) : (
              <p className="text-lg font-semibold text-zinc-400">HOLD</p>
            )}
            <QValueRow qValues={generated.q_values} />
          </div>
        )}

        <div className="mt-6 border-t border-zinc-100 pt-4 dark:border-zinc-800">
          <div className="flex items-center justify-between">
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Or generate for all 4 pairs &times; 5 intervals at once.
            </p>
            <button onClick={handleGenerateAll} disabled={generateAllRunning} className="btn-primary shrink-0">
              {generateAllRunning ? `Generating ${generateAllProgress}/20…` : "Generate all"}
            </button>
          </div>
          {generateAllResults.length > 0 && (
            <div className="mt-4 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
              <table className="w-full text-left text-xs">
                <thead className="bg-zinc-50 uppercase text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
                  <tr>
                    <th className="px-3 py-1.5">Pair</th>
                    <th className="px-3 py-1.5">Interval</th>
                    <th className="px-3 py-1.5">Direction</th>
                    <th className="px-3 py-1.5">Entry</th>
                    <th className="px-3 py-1.5">Exit</th>
                    <th className="px-3 py-1.5">Lot size</th>
                  </tr>
                </thead>
                <tbody>
                  {generateAllResults.map((cell) => {
                    const key = `${cell.pair}-${cell.interval}`;
                    const s = cell.signal;
                    const riskAmountUsd = accountBalance * (riskPercent / 100);
                    return (
                      <tr
                        key={key}
                        className={`border-t border-zinc-100 dark:border-zinc-800 ${s ? (s.direction === "BUY" ? "bg-emerald-50 dark:bg-emerald-950" : "bg-rose-50 dark:bg-rose-950") : ""}`}
                      >
                        <td className="px-3 py-1.5 font-mono">{cell.pair}</td>
                        <td className="px-3 py-1.5 font-mono">{cell.interval}</td>
                        {cell.error ? (
                          <td className="px-3 py-1.5 text-zinc-400" colSpan={4}>{cell.error}</td>
                        ) : !s ? (
                          <td className="px-3 py-1.5 text-zinc-400" colSpan={4}>HOLD</td>
                        ) : (
                          <>
                            <td className={`px-3 py-1.5 font-semibold ${s.direction === "BUY" ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                              {s.direction}
                            </td>
                            <td className="px-3 py-1.5">{s.entry_price.toFixed(5)}</td>
                            <td className="px-3 py-1.5">{s.target_price.toFixed(5)}</td>
                            <td className="px-3 py-1.5 font-mono">
                              {calculateLotSize(s.pair, s.entry_price, s.stop_price, riskAmountUsd).toFixed(2)}
                            </td>
                          </>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      <section>
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Recent RL signals</h2>
        {recentLoading && <p className="mt-4 text-sm text-zinc-500">Loading…</p>}
        {!recentLoading && recentSignals.length === 0 && (
          <p className="mt-4 text-sm text-zinc-500">No RL signals yet.</p>
        )}
        {recentSignals.length > 0 && (
          <div className="mt-4 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
            <table className="w-full text-left text-sm">
              <thead className="bg-zinc-50 text-xs uppercase text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
                <tr>
                  <th className="px-4 py-2">Pair</th>
                  <th className="px-4 py-2">Direction</th>
                  <th className="px-4 py-2">Entry</th>
                  <th className="px-4 py-2">Exit</th>
                  <th className="px-4 py-2">Stop</th>
                  <th className="px-4 py-2">Lot size</th>
                  <th className="px-4 py-2">Risk / reward</th>
                  <th className="px-4 py-2">Status</th>
                  <th className="px-4 py-2">When</th>
                </tr>
              </thead>
              <tbody>
                {recentSignals.map((s) => {
                  const riskAmountUsd = accountBalance * (riskPercent / 100);
                  const lots = calculateLotSize(s.pair, s.entry_price, s.stop_price, riskAmountUsd);
                  const rewardUsd = riskAmountUsd * (Math.abs(s.target_price - s.entry_price) / Math.abs(s.entry_price - s.stop_price));
                  return (
                    <tr key={s._id ?? `${s.pair}-${s.timestamp}`} className="border-t border-zinc-100 dark:border-zinc-800">
                      <td className="px-4 py-2">{s.pair} · {s.interval}</td>
                      <td className={`px-4 py-2 font-medium ${s.direction === "BUY" ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                        {s.direction}
                      </td>
                      <td className="px-4 py-2">{s.entry_price.toFixed(5)}</td>
                      <td className="px-4 py-2">{s.target_price.toFixed(5)}</td>
                      <td className="px-4 py-2">{s.stop_price.toFixed(5)}</td>
                      <td className="px-4 py-2 font-mono">{lots.toFixed(2)}</td>
                      <td className="px-4 py-2 text-xs">
                        <span className="text-rose-600 dark:text-rose-400">-${riskAmountUsd.toFixed(0)}</span>
                        {" / "}
                        <span className="text-emerald-600 dark:text-emerald-400">+${rewardUsd.toFixed(0)}</span>
                      </td>
                      <td className="px-4 py-2"><StatusBadge status={s.status} /></td>
                      <td className="px-4 py-2 text-xs text-zinc-500">{new Date(s.timestamp).toLocaleString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section>
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Training history</h2>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          Read down a given pair/interval&apos;s rows over successive trainings to see whether
          hit rate/expectancy is actually improving, not just whichever number is newest.
        </p>
        {policiesLoading && <p className="mt-4 text-sm text-zinc-500">Loading…</p>}
        {!policiesLoading && policies.length === 0 && (
          <p className="mt-4 text-sm text-zinc-500">No trained policies yet — train one above.</p>
        )}
        {policies.length > 0 && (
          <div className="mt-4 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
            <table className="w-full text-left text-sm">
              <thead className="bg-zinc-50 text-xs uppercase text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
                <tr>
                  <th className="px-4 py-2">Pair</th>
                  <th className="px-4 py-2">Episodes</th>
                  <th className="px-4 py-2">Train fraction</th>
                  <th className="px-4 py-2">Test hit rate</th>
                  <th className="px-4 py-2">Test expectancy</th>
                  <th className="px-4 py-2">When</th>
                  <th className="px-4 py-2"></th>
                </tr>
              </thead>
              <tbody>
                {policies.map((p) => {
                  const evaluation = p.evaluation;
                  const weightsOpen = expandedHistoryWeights === p.policy_id;
                  const tradesOpen = expandedHistoryTrades === p.policy_id;
                  const trades = tradeLogs[p.eval_run_id];
                  return (
                    <>
                      <tr key={p.policy_id} className="border-t border-zinc-100 dark:border-zinc-800">
                        <td className="px-4 py-2">{p.pair} · {p.interval}</td>
                        <td className="px-4 py-2">{p.episodes}</td>
                        <td className="px-4 py-2">{p.train_frac}</td>
                        <td className="px-4 py-2">{evaluation?.hit_rate_pct != null ? `${evaluation.hit_rate_pct}%` : "—"}</td>
                        <td className={`px-4 py-2 ${evaluation?.expectancy_pct != null && evaluation.expectancy_pct >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                          {evaluation?.expectancy_pct != null ? `${evaluation.expectancy_pct >= 0 ? "+" : ""}${evaluation.expectancy_pct}%` : "—"}
                        </td>
                        <td className="px-4 py-2 text-xs text-zinc-500">{new Date(p.created_at).toLocaleString()}</td>
                        <td className="px-4 py-2 text-xs whitespace-nowrap">
                          <button
                            onClick={() => setExpandedHistoryWeights(weightsOpen ? null : p.policy_id)}
                            className="text-zinc-500 underline underline-offset-2 hover:text-zinc-900 dark:hover:text-zinc-100"
                          >
                            {weightsOpen ? "Hide" : "Show"} weights
                          </button>
                          {" · "}
                          <button
                            onClick={() => handleShowTrades(p)}
                            className="text-zinc-500 underline underline-offset-2 hover:text-zinc-900 dark:hover:text-zinc-100"
                          >
                            {tradesOpen ? "Hide" : "Show"} trades
                          </button>
                        </td>
                      </tr>
                      {weightsOpen && (
                        <tr key={`${p.policy_id}-weights`} className="border-t border-zinc-100 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-800">
                          <td colSpan={7} className="px-4 py-2">
                            <WeightsTable policy={p} />
                          </td>
                        </tr>
                      )}
                      {tradesOpen && (
                        <tr key={`${p.policy_id}-trades`} className="border-t border-zinc-100 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-800">
                          <td colSpan={7} className="px-4 py-2">
                            {tradeLogsLoading === p.eval_run_id && <p className="text-xs text-zinc-500">Loading…</p>}
                            {trades && trades.length === 0 && <p className="text-xs text-zinc-500">No test-slice trades (all HOLD).</p>}
                            {trades && trades.length > 0 && (
                              <div className="overflow-x-auto">
                                <table className="w-full text-left text-xs">
                                  <thead className="text-zinc-500 dark:text-zinc-400">
                                    <tr>
                                      <th className="px-2 py-1">Direction</th>
                                      <th className="px-2 py-1">Entry</th>
                                      <th className="px-2 py-1">Outcome</th>
                                      <th className="px-2 py-1">Move</th>
                                      <th className="px-2 py-1">Status</th>
                                      <th className="px-2 py-1">When</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {trades.map((t, idx) => (
                                      <tr key={t._id ?? idx} className="border-t border-zinc-200 dark:border-zinc-700">
                                        <td className={`px-2 py-1 font-medium ${t.direction === "BUY" ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                                          {t.direction}
                                        </td>
                                        <td className="px-2 py-1 font-mono">{t.price_at_signal.toFixed(5)}</td>
                                        <td className="px-2 py-1 font-mono">{t.outcome_price != null ? t.outcome_price.toFixed(5) : "—"}</td>
                                        <td className={`px-2 py-1 ${t.outcome_pct_move != null && t.outcome_pct_move >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                                          {t.outcome_pct_move != null ? `${t.outcome_pct_move >= 0 ? "+" : ""}${t.outcome_pct_move.toFixed(4)}%` : "—"}
                                        </td>
                                        <td className="px-2 py-1"><StatusBadge status={t.status} /></td>
                                        <td className="px-2 py-1 text-zinc-500">{new Date(t.timestamp).toLocaleString()}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            )}
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
    </div>
  );
}

const ACTION_DISPLAY_ORDER = ["BUY", "SELL", "HOLD"];

function WeightsTable({ policy }: { policy: RLPolicy }) {
  const actions = ACTION_DISPLAY_ORDER.filter((a) => a in policy.weights);
  return (
    <div className="mt-2 overflow-x-auto rounded-md border border-zinc-200 dark:border-zinc-800">
      <table className="w-full text-left text-xs">
        <thead className="bg-zinc-50 uppercase text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
          <tr>
            <th className="px-3 py-1.5">Feature</th>
            {actions.map((a) => <th key={a} className="px-3 py-1.5">{a}</th>)}
          </tr>
        </thead>
        <tbody>
          {policy.feature_names.map((feature, i) => (
            <tr key={feature} className="border-t border-zinc-100 dark:border-zinc-800">
              <td className="px-3 py-1.5 font-mono">{feature}</td>
              {actions.map((a) => {
                const value = policy.weights[a][i];
                return (
                  <td key={a} className={`px-3 py-1.5 font-mono ${value >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                    {value >= 0 ? "+" : ""}{value.toFixed(4)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function QValueRow({ qValues }: { qValues: Record<string, number> }) {
  return (
    <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 rounded-md bg-zinc-50 px-3 py-2 text-xs font-mono dark:bg-zinc-800">
      {Object.entries(qValues).map(([action, value]) => (
        <span key={action}>
          {action}: <span className={value >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}>{value.toFixed(4)}</span>
        </span>
      ))}
    </div>
  );
}

function TrainTestCard({ run }: { run: BacktestRun }) {
  return (
    <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 dark:border-emerald-900 dark:bg-emerald-950">
      <p className="text-xs text-zinc-500 dark:text-zinc-400">Test (out-of-sample)</p>
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
        {run.directional_signals} trades ({run.hits}/{run.misses}/{run.expired}) · {run.hold_signals} holds ·
        {" "}{run.candles_evaluated} candles evaluated
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
