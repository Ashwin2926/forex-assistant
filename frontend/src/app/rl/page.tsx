"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { PAIRS, type BacktestRun, type PaperTrade, type RLPolicy, type RLSignal } from "@/lib/types";
import { StatusBadge } from "@/components/Badges";

const RL_INTERVAL = "1h"; // v1 is scoped to 1h only -- see PROGRESS.md

// Same formula as trading-signals/page.tsx, duplicated rather than shared -- this project
// keeps sizing logic local to whichever page displays it, not in lib/. Relevant here
// specifically because RL signals are meant to be executed manually on whatever broker you
// actually have (see PROGRESS.md -- no Deriv access in this region), so a lot size needs to
// travel with the signal regardless of the paper-trade endpoint.
function usdPerUnit(pair: string, entryPrice: number): number {
  return pair === "USD/JPY" ? 1 / entryPrice : 1;
}

function calculateLotSize(pair: string, entryPrice: number, stopPrice: number, riskAmountUsd: number): number {
  const stopDistance = Math.abs(entryPrice - stopPrice);
  if (stopDistance === 0) return 0;
  const units = riskAmountUsd / (stopDistance * usdPerUnit(pair, entryPrice));
  return units / 100_000;
}

export default function RLPage() {
  const [accountBalance, setAccountBalance] = useState(10000);
  const [riskPercent, setRiskPercent] = useState(1);

  const [pair, setPair] = useState<string>(PAIRS[0]);
  const [episodes, setEpisodes] = useState(100);
  const [trainFrac, setTrainFrac] = useState(0.7);
  const [training, setTraining] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainResult, setTrainResult] = useState<{ policy: RLPolicy; evaluation: BacktestRun } | null>(null);

  const [policies, setPolicies] = useState<RLPolicy[]>([]);
  const [policiesLoading, setPoliciesLoading] = useState(true);

  const [signalPair, setSignalPair] = useState<string>(PAIRS[0]);
  const [generating, setGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [generated, setGenerated] = useState<{ signal: RLSignal | null; q_values: Record<string, number> } | null>(null);

  const [recentSignals, setRecentSignals] = useState<RLSignal[]>([]);
  const [recentLoading, setRecentLoading] = useState(true);

  const [tradePair, setTradePair] = useState<string>(PAIRS[0]);
  const [trading, setTrading] = useState(false);
  const [tradeError, setTradeError] = useState<string | null>(null);
  const [tradeResult, setTradeResult] = useState<{ signal: RLSignal | null; paper_trade: PaperTrade | null; note?: string } | null>(null);

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
      const result = await api.trainRLPolicy(pair, RL_INTERVAL, { episodes, train_frac: trainFrac });
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
      const result = await api.generateRLSignal(signalPair, RL_INTERVAL);
      setGenerated(result);
      await loadRecentSignals();
    } catch (e) {
      setGenerateError(e instanceof ApiError ? e.message : "Signal generation failed.");
    } finally {
      setGenerating(false);
    }
  }

  async function handlePaperTrade() {
    setTrading(true);
    setTradeError(null);
    setTradeResult(null);
    try {
      setTradeResult(await api.paperTradeRL(tradePair, RL_INTERVAL));
    } catch (e) {
      setTradeError(e instanceof ApiError ? e.message : "Paper trade failed.");
    } finally {
      setTrading(false);
    }
  }

  return (
    <div className="flex flex-col gap-8">
      <section>
        <h1 className="text-xl font-semibold tracking-tight">RL agent (v1)</h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          A linear Q-learning agent that learns how to weight the same 7 strategies consensus
          uses, instead of a fixed vote threshold — one independent policy per pair, trained
          on historical candle replay (not live signal outcomes). Every trade it takes uses a
          fixed 1.5:1 target:stop ATR ratio, so it can never risk more than it stands to gain.
          Scoped to the 1h interval only for now. Additive and separate from the regular
          signal feed, consensus, and the ML classifier — doesn&apos;t touch any of them.
        </p>
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Position sizing</h2>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          Applied to every signal below — lot size = (account balance &times; risk %) &divide;
          (stop distance in price &times; USD value per unit), same formula and same 4-pair
          quote-currency assumptions as the Trading Signals page. For executing on whatever
          broker you actually have access to, not the (currently unusable) Deriv paper-trade
          button.
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
          <div className="mt-4">
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Test-slice evaluation (a real backtest run, greedy/no exploration) — directly
              comparable to any other approach&apos;s expectancy on the same pair/interval.
            </p>
            <div className="mt-2 max-w-xs">
              <TrainTestCard run={trainResult.evaluation} />
            </div>
          </div>
        )}
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Generate a signal</h2>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          Loads the latest trained policy and picks its greedy action on the current candle.
          Only BUY/SELL get stored — a HOLD just shows the q_values below.
        </p>
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <Field label="Pair">
            <select value={signalPair} onChange={(e) => setSignalPair(e.target.value)} className="select">
              {PAIRS.map((p) => <option key={p} value={p}>{p}</option>)}
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
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Paper trade (Deriv demo account)</h2>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          Manual only — not run by the cron. Generates a signal the same way as above and, if
          directional, executes it on the Deriv <strong>demo</strong> account.
          <strong> Not usable if Deriv isn&apos;t available in your region</strong> — in that
          case, manually execute the signals from the table below on whatever broker you
          actually have (use the lot size shown there). This button will also fail on the
          existing Deriv API auth issue, separately from the region question.
        </p>
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <Field label="Pair">
            <select value={tradePair} onChange={(e) => setTradePair(e.target.value)} className="select">
              {PAIRS.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </Field>
          <button onClick={handlePaperTrade} disabled={trading} className="btn-secondary">
            {trading ? "Placing…" : "Paper trade"}
          </button>
        </div>
        {tradeError && (
          <p className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {tradeError}
          </p>
        )}
        {tradeResult && (
          <div className="mt-4 text-sm">
            {tradeResult.note && <p className="text-zinc-500">{tradeResult.note}</p>}
            {tradeResult.paper_trade && (
              <p>
                {tradeResult.paper_trade.direction} · stake {tradeResult.paper_trade.stake} ·
                status {tradeResult.paper_trade.status}
                {tradeResult.paper_trade.error && ` — ${tradeResult.paper_trade.error}`}
              </p>
            )}
          </div>
        )}
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
                  <th className="px-4 py-2">When</th>
                </tr>
              </thead>
              <tbody>
                {policies.map((p) => (
                  <tr key={p.policy_id} className="border-t border-zinc-100 dark:border-zinc-800">
                    <td className="px-4 py-2">{p.pair} · {p.interval}</td>
                    <td className="px-4 py-2">{p.episodes}</td>
                    <td className="px-4 py-2">{p.train_frac}</td>
                    <td className="px-4 py-2 text-xs text-zinc-500">{new Date(p.created_at).toLocaleString()}</td>
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
