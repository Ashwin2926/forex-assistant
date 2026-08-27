"use client";

import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { INTERVALS, PAIRS, type BacktestRun, type RLAccuracy, type RLMemorySummary, type RLPolicy, type RLSignal, type RLTrainAllJob, type Signal } from "@/lib/types";
import { StatusBadge } from "@/components/Badges";

// How often to poll GET /rl/train-all/{job_id} while a batch run is in progress -- the job
// itself takes ~15 minutes total (20 combos x ~40-50s each), so this just needs to be
// frequent enough to feel live, not frequent enough to matter for load.
const TRAIN_ALL_POLL_MS = 4000;

interface GenerateAllCell {
  pair: string;
  interval: string;
  signal: RLSignal | null;
  qValues: Record<string, number> | null;
  memory?: RLMemorySummary | null;
  error?: string;
}

type Trend = "up" | "down" | null;

// Tracks whether a percentage is currently rising or falling compared to its own last value
// -- not a multi-tick history, just "did it move since the last time this changed." Used by
// both confidence widgets: training confidence updates on every poll while a job runs,
// overall accuracy updates on the 60s refetch above.
function useTrend(value: number | null): Trend {
  const prevRef = useRef<number | null>(null);
  const [trend, setTrend] = useState<Trend>(null);
  useEffect(() => {
    if (value == null) return;
    if (prevRef.current != null && value !== prevRef.current) {
      setTrend(value > prevRef.current ? "up" : "down");
    }
    prevRef.current = value;
  }, [value]);
  return trend;
}

function TrendArrow({ trend }: { trend: Trend }) {
  if (!trend) return null;
  return trend === "up" ? (
    <span className="ml-1 align-middle text-base text-emerald-600 dark:text-emerald-400" title="Up since the last update">▲</span>
  ) : (
    <span className="ml-1 align-middle text-base text-rose-600 dark:text-rose-400" title="Down since the last update">▼</span>
  );
}

export default function RLPage() {
  const [pair, setPair] = useState<string>(PAIRS[0]);
  const [interval, setInterval_] = useState<string>("1h");
  const [episodes, setEpisodes] = useState(200);
  const [trainFrac, setTrainFrac] = useState(0.7);
  const [startingBalance, setStartingBalance] = useState(50);
  const [training, setTraining] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainResult, setTrainResult] = useState<{ policy: RLPolicy; evaluation: BacktestRun } | null>(null);

  const [policies, setPolicies] = useState<RLPolicy[]>([]);
  const [policiesLoading, setPoliciesLoading] = useState(true);

  const [currentBalance, setCurrentBalance] = useState(50);

  const [recentSignals, setRecentSignals] = useState<RLSignal[]>([]);
  const [recentLoading, setRecentLoading] = useState(true);

  const [overallAccuracy, setOverallAccuracy] = useState<RLAccuracy | null>(null);

  const [trainAllJob, setTrainAllJob] = useState<RLTrainAllJob | null>(null);
  const [trainAllStartError, setTrainAllStartError] = useState<string | null>(null);
  const trainAllPollGuard = useRef<string | null>(null); // job_id currently being polled, to avoid a stray double-poll

  const [expandedHistoryWeights, setExpandedHistoryWeights] = useState<string | null>(null);
  const [expandedHistoryTrades, setExpandedHistoryTrades] = useState<string | null>(null);
  const [tradeLogs, setTradeLogs] = useState<Record<string, Signal[]>>({});
  const [tradeLogsLoading, setTradeLogsLoading] = useState<string | null>(null);

  const [generateAllResults, setGenerateAllResults] = useState<GenerateAllCell[]>([]);
  const [generateAllProgress, setGenerateAllProgress] = useState(0);
  const [generateAllRunning, setGenerateAllRunning] = useState(false);
  const [expandedQValues, setExpandedQValues] = useState<string | null>(null);

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

  async function loadOverallAccuracy() {
    try {
      setOverallAccuracy(await api.getRLAccuracy({ limit: 200 }));
    } catch {
      // non-critical section
    }
  }

  // Trains sequentially server-side now (see app/main.py's run_train_all_job) -- this just
  // polls GET /rl/train-all/{job_id} until status flips to "done". Recurses via setTimeout
  // rather than setInterval so a slow poll response can't overlap the next one.
  async function pollTrainAllJob(jobId: string) {
    trainAllPollGuard.current = jobId;
    let job: RLTrainAllJob;
    try {
      job = await api.getTrainAllRLJob(jobId);
    } catch {
      return; // transient network hiccup -- next mount or manual "Train all" click recovers
    }
    if (trainAllPollGuard.current !== jobId) return; // superseded by a newer job
    setTrainAllJob(job);
    if (job.status === "running") {
      setTimeout(() => pollTrainAllJob(jobId), TRAIN_ALL_POLL_MS);
    } else {
      await loadPolicies();
    }
  }

  useEffect(() => {
    loadPolicies();
    loadRecentSignals();
    loadOverallAccuracy();
    // Rehydrate an in-progress "Train all" job on load/reload -- the job itself lives
    // server-side now, so a reload should resume watching it, not lose track of it.
    api.getLatestTrainAllRLJob().then((job) => {
      if (job && job.status === "running") {
        setTrainAllJob(job);
        pollTrainAllJob(job.job_id);
      } else if (job) {
        setTrainAllJob(job);
      }
    }).catch(() => {});
  }, []);

  // Overall accuracy only changes as live signals resolve (roughly the cron's own cadence),
  // not on every render -- refetch periodically so the trend arrow next to it (useTrend
  // below) has something real to compare against over the course of a session, not just a
  // single static snapshot from page load.
  useEffect(() => {
    const id = setInterval(loadOverallAccuracy, 60_000);
    return () => clearInterval(id);
  }, []);

  async function handleTrain() {
    setTraining(true);
    setTrainError(null);
    try {
      const result = await api.trainRLPolicy(pair, interval, { episodes, train_frac: trainFrac, starting_balance: startingBalance });
      setTrainResult(result);
      await loadPolicies();
    } catch (e) {
      setTrainError(e instanceof ApiError ? e.message : "Training failed.");
    } finally {
      setTraining(false);
    }
  }

  async function handleTrainAll() {
    setTrainAllStartError(null);
    try {
      const job = await api.startTrainAllRL({ episodes, train_frac: trainFrac, starting_balance: startingBalance });
      setTrainAllJob(job);
      pollTrainAllJob(job.job_id);
    } catch (e) {
      setTrainAllStartError(e instanceof ApiError ? e.message : "Couldn't start training.");
    }
  }

  const [cancelling, setCancelling] = useState(false);

  async function handleCancelTrainAll() {
    if (!trainAllJob) return;
    setCancelling(true);
    try {
      // Cancel takes effect immediately server-side now (not just a flag the loop picks up
      // between combos -- a hung combo would never come back around to check it), so the
      // response already reflects status: "cancelled". No need to wait for the next poll.
      const job = await api.cancelTrainAllRLJob(trainAllJob.job_id);
      trainAllPollGuard.current = null; // stop any in-flight poll loop from overwriting this
      setTrainAllJob(job);
      await loadPolicies();
    } catch (e) {
      setTrainAllStartError(e instanceof ApiError ? e.message : "Couldn't stop training.");
    } finally {
      setCancelling(false);
    }
  }

  const trainAllRunning = trainAllJob?.status === "running";

  // Live "confidence" while a Train all job runs -- test-slice hit rate averaged across the
  // pair/intervals completed so far, weighted by each one's own trade count (directional_signals)
  // rather than a flat per-combo average, so a combo with 300 test trades isn't drowned out by
  // one with 5. hit_rate_pct is itself hits/directional_signals*100, so hits is recoverable.
  const trainingConfidence = (() => {
    const cells = (trainAllJob?.results ?? []).filter(
      (c) => c.ok && c.hit_rate_pct != null && c.directional_signals != null && c.directional_signals > 0,
    );
    const totalTrades = cells.reduce((sum, c) => sum + (c.directional_signals ?? 0), 0);
    if (totalTrades === 0) return null;
    const totalHits = cells.reduce((sum, c) => sum + (c.hit_rate_pct! / 100) * c.directional_signals!, 0);
    return { pct: (totalHits / totalTrades) * 100, trades: totalTrades, combosDone: cells.length };
  })();
  const trainingConfidenceTrend = useTrend(trainingConfidence?.pct ?? null);
  const overallAccuracyTrend = useTrend(overallAccuracy?.directional_hit_rate_pct ?? null);

  async function handleGenerateAll() {
    setGenerateAllRunning(true);
    setGenerateAllResults([]);
    setGenerateAllProgress(0);
    const combos = PAIRS.flatMap((p) => INTERVALS.map((i) => ({ pair: p, interval: i })));
    const results: GenerateAllCell[] = [];
    for (const { pair: p, interval: i } of combos) {
      try {
        const result = await api.generateRLSignal(p, i, currentBalance);
        results.push({ pair: p, interval: i, signal: result.signal, qValues: result.q_values, memory: result.memory ?? null });
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
        <h1 className="text-xl font-semibold tracking-tight">RL agent (v2)</h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          A linear Q-learning agent that learns how to weight the same 7 strategies consensus
          uses, instead of a fixed vote threshold — <strong>and how much to risk on each
          trade</strong>, choosing between a SMALL (1%) or LARGE (3%) size against a
          compounding account balance, trained to maximize log-growth (the same objective the
          Kelly criterion is built on, which also naturally punishes oversized bets that risk
          ruin). One independent policy per pair <em>and</em> interval, trained on historical
          candle replay starting from a chosen balance (default $50) — not live signal
          outcomes. Every trade still uses a fixed 1.5:1 target:stop ATR ratio regardless of
          size tier, so it can never risk more than it stands to gain. Additive and separate
          from the regular signal feed, consensus, and the ML classifier — doesn&apos;t touch
          any of them. No broker execution here — Deriv isn&apos;t available in every region,
          so signals are sized against your real balance (below) and traded manually on
          whatever broker you actually have.
        </p>
      </section>

      <section className="flex flex-wrap gap-3">
        <div className="min-w-[220px] flex-1 rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
            AI confidence {trainAllRunning ? "(training live)" : "(last training run)"}
          </p>
          {trainingConfidence == null ? (
            <p className="mt-1 text-sm text-zinc-400">
              {trainAllRunning ? "Waiting on the first pair/interval to finish…" : "No training run yet — click \"Train all\" below."}
            </p>
          ) : (
            <>
              <p className={`mt-1 text-2xl font-semibold ${trainingConfidence.pct >= 40 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                {trainingConfidence.pct.toFixed(1)}%
                <TrendArrow trend={trainingConfidenceTrend} />
              </p>
              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                test-slice hit rate across {trainingConfidence.trades} trades
                {" "}({trainingConfidence.combosDone}/{trainAllJob?.total ?? 20} pair/intervals{trainAllRunning ? " so far" : ""})
              </p>
            </>
          )}
        </div>
        <div className="min-w-[220px] flex-1 rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-xs font-medium text-zinc-500 dark:text-zinc-400">Overall trading accuracy (live signals)</p>
          {!overallAccuracy || overallAccuracy.total_resolved === 0 ? (
            <p className="mt-1 text-sm text-zinc-400">No resolved live RL signals yet.</p>
          ) : (
            <>
              <p className={`mt-1 text-2xl font-semibold ${(overallAccuracy.directional_hit_rate_pct ?? 0) >= 40 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                {overallAccuracy.directional_hit_rate_pct ?? "—"}%
                <TrendArrow trend={overallAccuracyTrend} />
              </p>
              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                {overallAccuracy.total_hits}/{(overallAccuracy.total_hits + overallAccuracy.total_misses)} decided
                {" "}(hit vs. miss only, every pair &amp; interval combined)
              </p>
              <p className="mt-1 text-xs text-zinc-400 dark:text-zinc-500">
                {overallAccuracy.resolution_breakdown_pct.expired}% of all signals expired
                (timed out — neither hit nor miss) rather than resolved
                {overallAccuracy.total_superseded > 0 && (
                  <>; +{overallAccuracy.total_superseded} superseded (agent changed its mind first)</>
                )}
              </p>
              <p className="mt-1 text-xs text-zinc-400 dark:text-zinc-500">
                blended hit-rate incl. expired as non-wins: {overallAccuracy.total_hit_rate_pct}%
              </p>
            </>
          )}
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
          <Field label="Starting balance ($)">
            <input
              type="number" step="10" min="1" value={startingBalance}
              onChange={(e) => setStartingBalance(Number(e.target.value))}
              className="select w-24"
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
              fraction above) — same thing the cron does once daily, run on demand. Runs
              server-side (~15 min total) once started, so it&apos;s safe to close this tab —
              reopening the page picks the same run back up.
            </p>
            <div className="flex shrink-0 items-center gap-2">
              <button onClick={handleTrainAll} disabled={trainAllRunning} className="btn-primary shrink-0">
                {trainAllRunning
                  ? `Training ${trainAllJob?.completed ?? 0}/${trainAllJob?.total ?? 20}…`
                  : trainAllJob?.status === "cancelled"
                    ? "Train all (last run stopped)"
                    : "Train all"}
              </button>
              {trainAllRunning && (
                <button
                  onClick={handleCancelTrainAll}
                  disabled={cancelling}
                  className="rounded-md border border-rose-300 px-3 py-1.5 text-xs font-medium text-rose-700 hover:bg-rose-50 disabled:opacity-50 dark:border-rose-800 dark:text-rose-300 dark:hover:bg-rose-950"
                >
                  {cancelling ? "Stopping…" : "Stop"}
                </button>
              )}
            </div>
          </div>
          {trainAllStartError && (
            <p className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950 dark:text-rose-300">
              {trainAllStartError}
            </p>
          )}
          {trainAllJob && trainAllJob.results.length > 0 && (
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
                    <th className="px-3 py-1.5">Ending balance</th>
                  </tr>
                </thead>
                <tbody>
                  {trainAllJob.results.map((cell) => {
                    const key = `${cell.pair}-${cell.interval}`;
                    return (
                      <tr key={key} className="border-t border-zinc-100 dark:border-zinc-800">
                        <td className="px-3 py-1.5 font-mono">{cell.pair}</td>
                        <td className="px-3 py-1.5 font-mono">{cell.interval}</td>
                        {!cell.ok ? (
                          <td className="px-3 py-1.5 text-zinc-400" colSpan={5}>{cell.error}</td>
                        ) : (
                          <>
                            <td className="px-3 py-1.5">{cell.hit_rate_pct != null ? `${cell.hit_rate_pct}%` : "—"}</td>
                            <td className={`px-3 py-1.5 ${cell.expectancy_pct != null && cell.expectancy_pct >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                              {cell.expectancy_pct != null ? `${cell.expectancy_pct >= 0 ? "+" : ""}${cell.expectancy_pct}%` : "—"}
                            </td>
                            <td className="px-3 py-1.5">{cell.directional_signals ?? "—"}</td>
                            <td className="px-3 py-1.5">{cell.hold_signals ?? "—"}</td>
                            <td className={`px-3 py-1.5 font-mono ${cell.total_return_pct != null && cell.total_return_pct >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                              {cell.ending_balance != null ? `$${cell.ending_balance}` : "—"}
                            </td>
                          </>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="border-t border-zinc-100 px-3 py-2 text-xs text-zinc-400 dark:border-zinc-800">
                Weights for each freshly-trained policy are in Training history below.
              </p>
            </div>
          )}
        </div>
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Generate signals</h2>
            <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
              Loads each pair/interval&apos;s latest trained policy and picks its greedy
              action on the current candle, sized against the balance below — all 4 pairs
              &times; 5 intervals at once. Only BUY/SELL get stored; a HOLD row shows as HOLD.
            </p>
          </div>
          <button onClick={handleGenerateAll} disabled={generateAllRunning} className="btn-primary shrink-0">
            {generateAllRunning ? `Generating ${generateAllProgress}/20…` : "Generate all"}
          </button>
        </div>
        <div className="mt-3">
          <Field label="Your current balance ($)">
            <input
              type="number" step="10" min="1" value={currentBalance}
              onChange={(e) => setCurrentBalance(Number(e.target.value))}
              className="select w-32"
            />
          </Field>
        </div>
        <div className="mt-4">
          {generateAllResults.length > 0 && (
            <div className="mt-4 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
              <table className="w-full text-left text-xs">
                <thead className="bg-zinc-50 uppercase text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
                  <tr>
                    <th className="px-3 py-1.5">Pair</th>
                    <th className="px-3 py-1.5">Interval</th>
                    <th className="px-3 py-1.5">Direction</th>
                    <th className="px-3 py-1.5">Confidence</th>
                    <th className="px-3 py-1.5">Memory</th>
                    <th className="px-3 py-1.5">Entry</th>
                    <th className="px-3 py-1.5">Exit</th>
                    <th className="px-3 py-1.5">Size</th>
                    <th className="px-3 py-1.5">Units</th>
                    <th className="px-3 py-1.5"></th>
                  </tr>
                </thead>
                <tbody>
                  {generateAllResults.map((cell) => {
                    const key = `${cell.pair}-${cell.interval}`;
                    const s = cell.signal;
                    const qOpen = expandedQValues === key;
                    const confidence = s ? actionConfidence(cell.qValues, s.size_tier ? `${s.direction}_${s.size_tier}` : s.direction) : null;
                    return (
                      <>
                        <tr
                          key={key}
                          className={`border-t border-zinc-100 dark:border-zinc-800 ${s ? (s.direction === "BUY" ? "bg-emerald-50 dark:bg-emerald-950" : "bg-rose-50 dark:bg-rose-950") : ""}`}
                        >
                          <td className="px-3 py-1.5 font-mono">{cell.pair}</td>
                          <td className="px-3 py-1.5 font-mono">{cell.interval}</td>
                          {cell.error ? (
                            <td className="px-3 py-1.5 text-zinc-400" colSpan={8}>{cell.error}</td>
                          ) : (
                            <>
                              {!s ? (
                                <td className="px-3 py-1.5 text-zinc-400" colSpan={6}>HOLD</td>
                              ) : (
                                <>
                                  <td className={`px-3 py-1.5 font-semibold ${s.direction === "BUY" ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                                    {s.direction}
                                  </td>
                                  <td className="px-3 py-1.5 font-mono">{confidence != null ? `${confidence.toFixed(0)}%` : "—"}</td>
                                  <td className="px-3 py-1.5 font-mono" title={cell.memory?.avg_pct_move != null ? `avg pct move: ${cell.memory.avg_pct_move}%, ${cell.memory.expired_pct}% expired` : undefined}>
                                    {cell.memory && cell.memory.cases_found > 0
                                      ? `${cell.memory.hit_rate_pct != null ? `${cell.memory.hit_rate_pct.toFixed(0)}%` : "—"} (${cell.memory.cases_found})`
                                      : "no history"}
                                  </td>
                                  <td className="px-3 py-1.5">{s.entry_price.toFixed(5)}</td>
                                  <td className="px-3 py-1.5">{s.target_price.toFixed(5)}</td>
                                  <td className={`px-3 py-1.5 font-mono ${s.size_tier === "LARGE" ? "text-amber-600 dark:text-amber-400" : ""}`}>
                                    {s.size_tier ?? "—"}
                                  </td>
                                </>
                              )}
                              <td className="px-3 py-1.5 font-mono">{s?.position_size_units != null ? s.position_size_units.toFixed(0) : "—"}</td>
                              <td className="px-3 py-1.5">
                                {cell.qValues && (
                                  <button
                                    onClick={() => setExpandedQValues(qOpen ? null : key)}
                                    className="text-zinc-500 underline underline-offset-2 hover:text-zinc-900 dark:hover:text-zinc-100"
                                  >
                                    {qOpen ? "Hide" : "Show"} q-values
                                  </button>
                                )}
                              </td>
                            </>
                          )}
                        </tr>
                        {qOpen && cell.qValues && (
                          <tr key={`${key}-q`} className="border-t border-zinc-100 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-800">
                            <td colSpan={10} className="px-3 py-2">
                              <QValueRow qValues={cell.qValues} />
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
                  <th className="px-4 py-2">Confidence</th>
                  <th className="px-4 py-2">Entry</th>
                  <th className="px-4 py-2">Exit</th>
                  <th className="px-4 py-2">Stop</th>
                  <th className="px-4 py-2">Size</th>
                  <th className="px-4 py-2">Units</th>
                  <th className="px-4 py-2">Balance at signal</th>
                  <th className="px-4 py-2">Status</th>
                  <th className="px-4 py-2">When</th>
                </tr>
              </thead>
              <tbody>
                {recentSignals.map((s) => {
                  const confidence = actionConfidence(s.q_values, s.size_tier ? `${s.direction}_${s.size_tier}` : s.direction);
                  return (
                  <tr key={s._id ?? `${s.pair}-${s.timestamp}`} className="border-t border-zinc-100 dark:border-zinc-800">
                    <td className="px-4 py-2">{s.pair} · {s.interval}</td>
                    <td className={`px-4 py-2 font-medium ${s.direction === "BUY" ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                      {s.direction}
                    </td>
                    <td className="px-4 py-2 font-mono">{confidence != null ? `${confidence.toFixed(0)}%` : "—"}</td>
                    <td className="px-4 py-2">{s.entry_price.toFixed(5)}</td>
                    <td className="px-4 py-2">{s.target_price.toFixed(5)}</td>
                    <td className="px-4 py-2">{s.stop_price.toFixed(5)}</td>
                    <td className={`px-4 py-2 font-mono ${s.size_tier === "LARGE" ? "text-amber-600 dark:text-amber-400" : ""}`}>{s.size_tier ?? "—"}</td>
                    <td className="px-4 py-2 font-mono">{s.position_size_units != null ? s.position_size_units.toFixed(0) : "—"}</td>
                    <td className="px-4 py-2 font-mono">{s.balance_at_signal != null ? `$${s.balance_at_signal}` : "—"}</td>
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
                  <th className="px-4 py-2">Ending balance</th>
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
                        <td className={`px-4 py-2 font-mono ${evaluation?.total_return_pct != null && evaluation.total_return_pct >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                          {evaluation?.ending_balance != null ? `$${evaluation.ending_balance}` : "—"}
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
                          <td colSpan={8} className="px-4 py-2">
                            <WeightsTable policy={p} />
                          </td>
                        </tr>
                      )}
                      {tradesOpen && (
                        <tr key={`${p.policy_id}-trades`} className="border-t border-zinc-100 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-800">
                          <td colSpan={8} className="px-4 py-2">
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
                                      <th className="px-2 py-1">Size</th>
                                      <th className="px-2 py-1">Balance</th>
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
                                        <td className={`px-2 py-1 font-mono ${t.size_tier === "LARGE" ? "text-amber-600 dark:text-amber-400" : ""}`}>{t.size_tier ?? "—"}</td>
                                        <td className="px-2 py-1 font-mono">{t.balance_at_signal != null ? `$${t.balance_at_signal}` : "—"}</td>
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

// Per-trade "confidence" -- a softmax over the stored q_values, read at the chosen action.
// Q-values aren't probabilities (they're expected log-growth per trade, a reward scale that's
// naturally tiny -- real observed values sit around +/-0.001 to 0.005), so a PLAIN softmax on
// the raw values is useless: exp(0.002) vs exp(0.0005) are within a fraction of a percent of
// each other regardless of which action actually "won" by a meaningful margin, which is
// exactly why every trade was showing ~20% (1/5 actions, i.e. indistinguishable-from-uniform).
// Z-score normalizing the q-values first (mean 0, unit variance) before softmax fixes this --
// it's the *relative* spread between actions in units of their own standard deviation that
// should drive confidence, not their absolute tiny scale. If every action is genuinely tied
// (std ~0, real if a policy's weights haven't diverged from init yet), this correctly reports
// a plain 1/n uniform split instead of blowing up dividing by ~zero.
// Returns null if qValues is missing or doesn't contain the chosen action (e.g. a pre-v2
// signal using the old BUY/SELL action names against q_values that were never restructured).
function actionConfidence(qValues: Record<string, number> | null | undefined, action: string | null | undefined): number | null {
  if (!qValues || !action || !(action in qValues)) return null;
  const entries = Object.values(qValues);
  const n = entries.length;
  if (n === 0) return null;
  const mean = entries.reduce((a, b) => a + b, 0) / n;
  const variance = entries.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  if (std < 1e-9) return 100 / n; // genuinely tied -- report the honest uniform split
  const scaled = entries.map((v) => (v - mean) / std);
  const max = Math.max(...scaled);
  const exps = scaled.map((v) => Math.exp(v - max));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const chosenScaled = (qValues[action] - mean) / std;
  const chosenExp = Math.exp(chosenScaled - max);
  return sumExp > 0 ? (chosenExp / sumExp) * 100 : null;
}

// Covers both the current 5-action space and pre-v2 policies (plain BUY/SELL) still in the DB.
const ACTION_DISPLAY_ORDER = ["BUY_LARGE", "BUY_SMALL", "BUY", "SELL_SMALL", "SELL_LARGE", "SELL", "HOLD"];

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
      {run.starting_balance != null && run.ending_balance != null && (
        <p className="mt-2 border-t border-emerald-200 pt-2 text-sm dark:border-emerald-900">
          Started at <span className="font-mono">${run.starting_balance}</span> → grew to{" "}
          <span className="font-mono font-semibold">${run.ending_balance}</span>{" "}
          <span className={run.total_return_pct != null && run.total_return_pct >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}>
            ({run.total_return_pct != null && run.total_return_pct >= 0 ? "+" : ""}{run.total_return_pct}%)
          </span>
        </p>
      )}
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
