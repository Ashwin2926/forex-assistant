"use client";

import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { INTERVALS, PAIRS, type BacktestRun, type PPOPolicy, type PPOTrainDiagnostics, type RLAccuracy, type RLInsights, type RLLearningCurve, type RLLearningVerdict, type RLMemorySummary, type RLSignal, type RLTrainAllJob, type Signal } from "@/lib/types";
import { DirectionBadge, StatusBadge } from "@/components/Badges";

// How often to poll GET /rl/train-all/{job_id} while a train-all run is in progress.
const TRAIN_ALL_POLL_MS = 4000;

interface GenerateAllCell {
  pair: string;
  interval: string;
  signal: RLSignal | null;
  qValues: Record<string, number> | null;
  memory?: RLMemorySummary | null;
  memoryOverride?: string | null;
  excludedReason?: string | null;
  mlBlockedReason?: string | null;
  error?: string;
}

type Trend = "up" | "down" | null;

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

function SectionToggle({ open, onToggle, title }: { open: boolean; onToggle: () => void; title: string }) {
  return (
    <button onClick={onToggle} className="flex items-center gap-2 text-left">
      <span className={`inline-block text-xs text-zinc-400 transition-transform ${open ? "rotate-90" : ""}`}>▶</span>
      <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">{title}</h2>
    </button>
  );
}

const VERDICT_LABEL: Record<RLLearningVerdict["status"], string> = {
  improving: "Improving",
  declining: "Declining",
  flat: "Holding steady",
  not_enough_data: "Not enough data yet",
};
const VERDICT_COLOR: Record<RLLearningVerdict["status"], string> = {
  improving: "text-emerald-600 dark:text-emerald-400",
  declining: "text-rose-600 dark:text-rose-400",
  flat: "text-zinc-500 dark:text-zinc-400",
  not_enough_data: "text-zinc-400 dark:text-zinc-500",
};

function VerdictCard({ title, verdict, sampleUnit, minN }: { title: string; verdict: RLLearningVerdict; sampleUnit: string; minN: number }) {
  return (
    <div className="min-w-[220px] flex-1 card p-4">
      <p className="text-xs font-medium text-zinc-500 dark:text-zinc-400">{title}</p>
      <p className={`mt-1 text-xl font-semibold ${VERDICT_COLOR[verdict.status]}`}>{VERDICT_LABEL[verdict.status]}</p>
      {verdict.first_half_rate_pct != null && verdict.second_half_rate_pct != null ? (
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          {verdict.first_half_rate_pct}% → {verdict.second_half_rate_pct}%{" "}
          (earlier {verdict.first_half_n} {sampleUnit} vs later {verdict.second_half_n})
        </p>
      ) : (
        <p className="mt-1 text-xs text-zinc-400">Needs at least {minN} {sampleUnit} in each half of the window.</p>
      )}
    </div>
  );
}

// PPO's q_values ARE real action probabilities (they sum to 1) -- unlike the retired linear
// policy's raw Q-values, no softmax/normalization is needed to turn them into a "confidence"
// percentage, just read the chosen action's own probability directly.
function actionProbability(qValues: Record<string, number> | null | undefined, action: string | null | undefined): number | null {
  if (!qValues || !action || !(action in qValues)) return null;
  return qValues[action] * 100;
}

export default function RLPage() {
  const [pair, setPair] = useState<string>(PAIRS[0]);
  const [interval, setInterval_] = useState<string>("1h");
  const [totalTimesteps, setTotalTimesteps] = useState(50_000);
  const [trainFrac, setTrainFrac] = useState(0.7);
  const [startingBalance, setStartingBalance] = useState(50);
  const [training, setTraining] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainResult, setTrainResult] = useState<{ policy: PPOPolicy; evaluation: BacktestRun; poc_diagnostics: PPOTrainDiagnostics } | null>(null);

  const [policies, setPolicies] = useState<PPOPolicy[]>([]);
  const [policiesLoading, setPoliciesLoading] = useState(true);

  const [currentBalance, setCurrentBalance] = useState(50);

  const [recentSignals, setRecentSignals] = useState<RLSignal[]>([]);
  const [recentLoading, setRecentLoading] = useState(true);

  const [overallAccuracy, setOverallAccuracy] = useState<RLAccuracy | null>(null);
  const [learningCurve, setLearningCurve] = useState<RLLearningCurve | null>(null);
  const [insights, setInsights] = useState<RLInsights | null>(null);

  const [insightsOpen, setInsightsOpen] = useState(true);
  const [trainOpen, setTrainOpen] = useState(false);
  const [generateOpen, setGenerateOpen] = useState(false);
  const [recentSignalsOpen, setRecentSignalsOpen] = useState(false);
  const [trainingHistoryOpen, setTrainingHistoryOpen] = useState(false);

  const [trainAllJob, setTrainAllJob] = useState<RLTrainAllJob | null>(null);
  const [trainAllStartError, setTrainAllStartError] = useState<string | null>(null);
  const trainAllPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

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

  async function loadLearningCurve() {
    try {
      setLearningCurve(await api.getRLLearningCurve(30));
    } catch {
      // non-critical section
    }
  }

  async function loadInsights() {
    try {
      setInsights(await api.getRLInsights());
    } catch {
      // non-critical section
    }
  }

  useEffect(() => {
    loadPolicies();
    loadRecentSignals();
    loadOverallAccuracy();
    loadLearningCurve();
    loadInsights();
  }, []);

  useEffect(() => {
    const id = setInterval(loadOverallAccuracy, 60_000);
    return () => clearInterval(id);
  }, []);

  // Rehydrates the last train-all run (server-persisted, GitHub-Actions-backed -- see
  // POST /rl/train-all) on mount/reload so trainingConfidence below has real data
  // immediately instead of staying empty until someone manually clicks "Train all" again
  // in this exact tab. Resumes polling if that run is still in progress (e.g. the page was
  // reloaded mid-run).
  useEffect(() => {
    (async () => {
      try {
        const job = await api.getLatestTrainAllRLJob();
        if (job) {
          setTrainAllJob(job);
          if (job.status === "running") pollTrainAllJob(job.job_id);
        }
      } catch {
        // non-critical section
      }
    })();
    return () => {
      if (trainAllPollRef.current) clearInterval(trainAllPollRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleTrain() {
    setTraining(true);
    setTrainError(null);
    try {
      const result = await api.trainRLPolicy(pair, interval, { total_timesteps: totalTimesteps, train_frac: trainFrac, starting_balance: startingBalance });
      setTrainResult(result);
      await loadPolicies();
    } catch (e) {
      setTrainError(e instanceof ApiError ? e.message : "Training failed.");
    } finally {
      setTraining(false);
    }
  }

  // Server-side job (POST /rl/train-all), backed by a GitHub Actions run
  // (.github/workflows/train-rl.yml) rather than FastAPI's BackgroundTasks -- the earlier
  // client-side loop this replaced existed specifically because BackgroundTasks didn't
  // reliably survive on FastAPI Cloud (see keep-fresh.yml's own history: the in-process
  // APScheduler died the same way). Training now actually executes on a GitHub Actions
  // runner instead of inside the backend process, so this can go back to "start a job,
  // poll it" without that risk -- and unlike the client-side version, this survives the tab
  // being closed, and rehydrates on reload (see the mount effect above).
  function pollTrainAllJob(jobId: string) {
    if (trainAllPollRef.current) clearInterval(trainAllPollRef.current);
    trainAllPollRef.current = setInterval(async () => {
      try {
        const job = await api.getTrainAllRLJob(jobId);
        setTrainAllJob(job);
        if (job.status !== "running") {
          if (trainAllPollRef.current) clearInterval(trainAllPollRef.current);
          trainAllPollRef.current = null;
          await loadPolicies();
        }
      } catch {
        // transient poll failure -- try again next tick rather than aborting the whole poll
      }
    }, TRAIN_ALL_POLL_MS);
  }

  async function handleTrainAll() {
    setTrainAllStartError(null);
    try {
      const job = await api.startTrainAllRL({ total_timesteps: totalTimesteps, train_frac: trainFrac, starting_balance: startingBalance });
      setTrainAllJob(job);
      pollTrainAllJob(job.job_id);
    } catch (e) {
      setTrainAllStartError(e instanceof ApiError ? e.message : "Failed to start training.");
    }
  }

  const [cancelling, setCancelling] = useState(false);

  async function handleCancelTrainAll() {
    if (!trainAllJob) return;
    setCancelling(true);
    try {
      const job = await api.cancelTrainAllRLJob(trainAllJob.job_id);
      setTrainAllJob(job);
      if (trainAllPollRef.current) clearInterval(trainAllPollRef.current);
      trainAllPollRef.current = null;
    } catch (e) {
      setTrainAllStartError(e instanceof ApiError ? e.message : "Failed to cancel.");
    } finally {
      setCancelling(false);
    }
  }

  const trainAllRunning = trainAllJob?.status === "running";

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

  const [scoringRL, setScoringRL] = useState(false);
  const [scoreResult, setScoreResult] = useState<string | null>(null);

  async function handleScoreRL() {
    setScoringRL(true);
    setScoreResult(null);
    try {
      const result = await api.scoreRLSignals();
      setScoreResult(
        `${result.hit} hit, ${result.miss} miss, ${result.expired} expired, ${result.still_pending} still pending`
        + (result.skipped_no_data > 0 ? `, ${result.skipped_no_data} skipped (no data yet)` : ""),
      );
      await Promise.all([loadRecentSignals(), loadOverallAccuracy(), loadPolicies(), loadLearningCurve(), loadInsights()]);
    } catch (e) {
      setScoreResult(e instanceof ApiError ? e.message : "Scoring failed.");
    } finally {
      setScoringRL(false);
    }
  }

  async function handleGenerateAll() {
    setGenerateAllRunning(true);
    setGenerateAllResults([]);
    setGenerateAllProgress(0);
    const combos = PAIRS.flatMap((p) => INTERVALS.map((i) => ({ pair: p, interval: i })));
    const results: GenerateAllCell[] = [];
    for (const { pair: p, interval: i } of combos) {
      try {
        const result = await api.generateRLSignal(p, i, currentBalance);
        results.push({
          pair: p, interval: i, signal: result.signal, qValues: result.q_values,
          memory: result.memory ?? null, memoryOverride: result.memory_override ?? null,
          excludedReason: result.excluded_reason ?? null, mlBlockedReason: result.ml_blocked_reason ?? null,
        });
      } catch (e) {
        results.push({ pair: p, interval: i, signal: null, qValues: null, error: e instanceof ApiError ? e.message : "Failed" });
      }
      setGenerateAllProgress(results.length);
      setGenerateAllResults([...results]);
    }
    setGenerateAllRunning(false);
    await loadRecentSignals();
  }

  async function handleShowTrades(policy: PPOPolicy) {
    const key = policy.policy_id;
    if (expandedHistoryTrades === key) {
      setExpandedHistoryTrades(null);
      return;
    }
    setExpandedHistoryTrades(key);
    if (tradeLogs[policy.eval_run_id]) return;
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
        <h1 className="text-xl font-semibold tracking-tight">RL agent (PPO)</h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          A PPO agent (stable-baselines3, via a gymnasium.Env adapter) that learns how to
          weight the same strategies consensus uses, instead of a fixed vote threshold —{" "}
          <strong>and how much to risk on each trade</strong>, choosing between a SMALL (1%) or
          LARGE (3%) size against a compounding account balance, trained to maximize
          log-growth (the same objective the Kelly criterion is built on, which also naturally
          punishes oversized bets that risk ruin). One independent policy per pair{" "}
          <em>and</em> interval, trained on historical candle replay starting from a chosen
          balance (default $50) — not live signal outcomes. Every trade still uses a fixed
          1.5:1 target:stop ATR ratio regardless of size tier, so it can never risk more than
          it stands to gain. This project&apos;s RL agent was linear Q-learning before PPO
          replaced it — every policy here is a fresh PPO model, with no warm-start carried
          over from that prior algorithm. Additive and separate from the regular signal feed,
          consensus, and the ML classifier — doesn&apos;t touch any of them. No broker
          execution here — signals are sized against your real balance (below) and traded
          manually on whatever broker you actually have.
        </p>
      </section>

      <section className="flex flex-wrap gap-3">
        <div className="min-w-[220px] flex-1 card p-4">
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
        <div className="min-w-[220px] flex-1 card p-4">
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

      <section className="card p-4">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Is it getting smarter?</h2>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          Compares the earlier half of the last 30 days against the later half, pooling real
          counts within each half rather than averaging daily percentages. Unlike the retired
          linear policy, PPO has no warm start — each retrain is an independent from-scratch
          run, so a rising trend here reflects genuinely improving training data/setup, not
          accumulated weight refinement.
        </p>
        {!learningCurve ? (
          <p className="mt-4 text-sm text-zinc-400">Loading…</p>
        ) : (
          <div className="mt-4 flex flex-wrap gap-3">
            <VerdictCard title="Live trading" verdict={learningCurve.verdict.live} sampleUnit="trades" minN={15} />
            <VerdictCard title="Training quality" verdict={learningCurve.verdict.training} sampleUnit="policy runs" minN={20} />
          </div>
        )}
      </section>

      <section className="card p-4">
        <div className="flex items-center justify-between">
          <div>
            <SectionToggle open={insightsOpen} onToggle={() => setInsightsOpen((o) => !o)} title="What to improve" />
            {insightsOpen && (
              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                Deterministic findings from data already collected here — no AI judgment call,
                every line traces back to a specific number. Critical first.
              </p>
            )}
          </div>
          {insightsOpen && (
            <button
              onClick={loadInsights}
              className="shrink-0 rounded-md border border-zinc-300 px-2.5 py-1 text-xs font-medium text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
            >
              Refresh
            </button>
          )}
        </div>
        {insightsOpen && (!insights ? (
          <p className="mt-4 text-sm text-zinc-400">Loading…</p>
        ) : (
          <div className="mt-4 flex flex-col gap-2">
            {insights.findings.map((f, i) => {
              const style = {
                critical: "border-rose-200 bg-rose-50 dark:border-rose-900 dark:bg-rose-950",
                warning: "border-amber-200 bg-amber-50 dark:border-amber-900 dark:bg-amber-950",
                good: "border-emerald-200 bg-emerald-50 dark:border-emerald-900 dark:bg-emerald-950",
              }[f.severity];
              const badge = {
                critical: "text-rose-700 dark:text-rose-300",
                warning: "text-amber-700 dark:text-amber-300",
                good: "text-emerald-700 dark:text-emerald-300",
              }[f.severity];
              return (
                <div key={i} className={`rounded-md border p-3 ${style}`}>
                  <p className="text-sm font-medium text-zinc-800 dark:text-zinc-100">
                    <span className={`mr-2 text-xs font-semibold uppercase ${badge}`}>{f.severity}</span>
                    {f.title}
                    {f.pair && f.interval && (
                      <span className="ml-2 font-mono text-xs font-normal text-zinc-500 dark:text-zinc-400">
                        {f.pair} · {f.interval}
                      </span>
                    )}
                  </p>
                  <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-300">{f.detail}</p>
                </div>
              );
            })}
          </div>
        ))}
      </section>

      <section className="card p-4">
        <SectionToggle open={trainOpen} onToggle={() => setTrainOpen((o) => !o)} title="Train a policy" />
        {trainOpen && (
        <>
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
          <Field label="Total timesteps">
            <input
              type="number" step="1000" min="1000" value={totalTimesteps}
              onChange={(e) => setTotalTimesteps(Number(e.target.value))}
              className="select w-28"
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
        <p className="mt-2 text-xs text-zinc-400">
          No warm start — every training run starts a fresh PPO model from total_timesteps
          real environment steps, it doesn&apos;t continue a prior run&apos;s weights.
        </p>
        {trainError && (
          <p className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {trainError}
          </p>
        )}
        {trainResult && (
          <div className="mt-4 flex flex-col gap-4">
            <div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Test-slice evaluation (a real backtest run, greedy/deterministic) — directly
                comparable to any other approach&apos;s expectancy on the same pair/interval.
              </p>
              <div className="mt-2 max-w-xs">
                <TrainTestCard run={trainResult.evaluation} />
              </div>
            </div>
            <div>
              <p className="text-xs font-medium text-zinc-500 dark:text-zinc-400">Proof-of-concept diagnostics</p>
              <p className="mt-1 text-xs text-zinc-400 dark:text-zinc-500">
                Worth checking on every real training run, not just once — a policy that fails
                these isn&apos;t ready to trust live.
              </p>
              <div className="mt-2 grid grid-cols-2 gap-3 sm:grid-cols-3">
                <StatTile
                  label="Beats random?"
                  value={trainResult.poc_diagnostics.beats_random ? "Yes" : "No"}
                  accent={trainResult.poc_diagnostics.beats_random ? "emerald" : "rose"}
                />
                <StatTile label="Model size" value={`${(trainResult.poc_diagnostics.model_size_bytes / 1024).toFixed(0)} KB`} />
                <StatTile label="Save+load latency" value={`${trainResult.poc_diagnostics.save_load_latency_ms.toFixed(1)} ms`} />
              </div>
              {trainResult.poc_diagnostics.random_baseline_total_return_pct != null && (
                <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
                  Random baseline total return: {trainResult.poc_diagnostics.random_baseline_total_return_pct}%
                </p>
              )}
            </div>
          </div>
        )}

        <div className="mt-6 border-t border-zinc-100 pt-4 dark:border-zinc-800">
          <div className="flex items-center justify-between">
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Or train all 4 pairs &times; 5 intervals at once (using the settings above) —
              same thing the cron does once daily, run on demand. Runs server-side on GitHub
              Actions (usually ~30-35 minutes total — that runner's CPU is slower for this
              than the backend's own) — safe to close this tab or navigate away; reopening
              this page picks the run back up.
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
                <thead className="table-head uppercase">
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
                Full details for each freshly-trained policy are in Training history below.
              </p>
            </div>
          )}
        </div>
        </>
        )}
      </section>

      <section className="card p-4">
        <div className="flex items-center justify-between">
          <div>
            <SectionToggle open={generateOpen} onToggle={() => setGenerateOpen((o) => !o)} title="Generate signals" />
            {generateOpen && (
              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                Loads each pair/interval&apos;s latest trained policy and picks its greedy
                (deterministic) action on the current candle, sized against the balance below
                — all 4 pairs &times; 5 intervals at once. Only BUY/SELL get stored; a HOLD
                row shows as HOLD.
              </p>
            )}
          </div>
          {generateOpen && (
            <button onClick={handleGenerateAll} disabled={generateAllRunning} className="btn-primary shrink-0">
              {generateAllRunning ? `Generating ${generateAllProgress}/20…` : "Generate all"}
            </button>
          )}
        </div>
        {generateOpen && (
        <>
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
                <thead className="table-head uppercase">
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
                    const confidence = s ? actionProbability(cell.qValues, s.size_tier ? `${s.direction}_${s.size_tier}` : s.direction) : null;
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
                                <td
                                  className={`px-3 py-1.5 ${cell.excludedReason || cell.mlBlockedReason ? "text-amber-600 dark:text-amber-400" : "text-zinc-400"}`}
                                  colSpan={6}
                                  title={cell.excludedReason ?? cell.mlBlockedReason ?? cell.memoryOverride ?? undefined}
                                >
                                  {cell.excludedReason ? "Excluded (losing edge)" : cell.mlBlockedReason ? "HOLD (ML blocked)" : cell.memoryOverride ? "HOLD (memory override)" : "HOLD"}
                                </td>
                              ) : (
                                <>
                                  <td className={`px-3 py-1.5 font-semibold ${s.direction === "BUY" ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                                    {s.direction}
                                  </td>
                                  <td className="px-3 py-1.5 font-mono">{confidence != null ? `${confidence.toFixed(0)}%` : "—"}</td>
                                  <td
                                    className="px-3 py-1.5 font-mono"
                                    title={cell.memory ? (
                                      `avg pct move: ${cell.memory.avg_pct_move ?? "—"}%, ${cell.memory.expired_pct ?? "—"}% expired -- `
                                      + `policy's overall record: ${cell.memory.policy_hit_rate_pct ?? "—"}% over ${cell.memory.policy_decided_trades} decided trades`
                                    ) : undefined}
                                  >
                                    {cell.memory && cell.memory.cases_found > 0
                                      ? `${cell.memory.hit_rate_pct != null ? `${cell.memory.hit_rate_pct.toFixed(0)}%` : "—"} (${cell.memory.cases_found})`
                                      : "no history"}
                                  </td>
                                  <td className="px-3 py-1.5">{s.entry_price.toFixed(5)}</td>
                                  <td className="px-3 py-1.5">{s.target_price.toFixed(5)}</td>
                                  <td
                                    className={`px-3 py-1.5 font-mono ${s.size_tier === "LARGE" ? "text-amber-600 dark:text-amber-400" : ""} ${s.memory_override ? "underline decoration-dotted" : ""}`}
                                    title={s.memory_override ?? undefined}
                                  >
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
                                    {qOpen ? "Hide" : "Show"} probabilities
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
        </>
        )}
      </section>

      <section>
        <div className="flex items-center justify-between">
          <SectionToggle open={recentSignalsOpen} onToggle={() => setRecentSignalsOpen((o) => !o)} title="Recent RL signals" />
          {recentSignalsOpen && (
            <div className="flex items-center gap-2">
              <button
                onClick={handleScoreRL}
                disabled={scoringRL}
                title="Checks every pending RL signal against candles that have arrived since it fired, resolving hit/miss/expired where enough real data now exists -- doesn't generate or train anything, just resolves outcomes."
                className="rounded-md border border-zinc-300 px-2.5 py-1 text-xs font-medium text-zinc-600 hover:bg-zinc-100 disabled:opacity-50 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
              >
                {scoringRL ? "Scoring…" : "Score now"}
              </button>
              {scoreResult && <span className="text-xs text-zinc-500 dark:text-zinc-400">{scoreResult}</span>}
            </div>
          )}
        </div>
        {recentSignalsOpen && recentLoading && <p className="mt-4 text-sm text-zinc-500">Loading…</p>}
        {recentSignalsOpen && !recentLoading && recentSignals.length === 0 && (
          <p className="mt-4 text-sm text-zinc-500">No RL signals yet.</p>
        )}
        {recentSignalsOpen && recentSignals.length > 0 && (
          <div className="mt-4 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
            <table className="w-full text-left text-sm">
              <thead className="table-head text-xs uppercase">
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
                  const confidence = actionProbability(s.q_values, s.size_tier ? `${s.direction}_${s.size_tier}` : s.direction);
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
        <SectionToggle open={trainingHistoryOpen} onToggle={() => setTrainingHistoryOpen((o) => !o)} title="Training history" />
        {trainingHistoryOpen && (
          <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
            Read down a given pair/interval&apos;s rows over successive trainings to see whether
            hit rate/expectancy is actually improving, not just whichever number is newest.
            Unlike the retired linear policy there's no weight table to show here — PPO's
            policy network has no per-feature weight, see the action-probability distribution
            in Generate signals/Recent RL signals instead.
          </p>
        )}
        {trainingHistoryOpen && policiesLoading && <p className="mt-4 text-sm text-zinc-500">Loading…</p>}
        {trainingHistoryOpen && !policiesLoading && policies.length === 0 && (
          <p className="mt-4 text-sm text-zinc-500">No trained policies yet — train one above.</p>
        )}
        {trainingHistoryOpen && policies.length > 0 && (
          <div className="mt-4 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
            <table className="w-full text-left text-sm">
              <thead className="table-head text-xs uppercase">
                <tr>
                  <th className="px-4 py-2">Pair</th>
                  <th className="px-4 py-2">Timesteps</th>
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
                  const tradesOpen = expandedHistoryTrades === p.policy_id;
                  const trades = tradeLogs[p.eval_run_id];
                  return (
                    <>
                      <tr key={p.policy_id} className="border-t border-zinc-100 dark:border-zinc-800">
                        <td className="px-4 py-2">{p.pair} · {p.interval}</td>
                        <td className="px-4 py-2">{p.total_timesteps.toLocaleString()}</td>
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
                            onClick={() => handleShowTrades(p)}
                            className="text-zinc-500 underline underline-offset-2 hover:text-zinc-900 dark:hover:text-zinc-100"
                          >
                            {tradesOpen ? "Hide" : "Show"} trades
                          </button>
                        </td>
                      </tr>
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

function QValueRow({ qValues }: { qValues: Record<string, number> }) {
  return (
    <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 rounded-md bg-zinc-50 px-3 py-2 text-xs font-mono dark:bg-zinc-800">
      {Object.entries(qValues)
        .sort(([, a], [, b]) => b - a)
        .map(([action, value]) => (
          <span key={action}>
            {action}: <span className="text-sky-600 dark:text-sky-400">{(value * 100).toFixed(1)}%</span>
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

function StatTile({ label, value, accent }: { label: string; value: string; accent?: "emerald" | "rose" }) {
  const color = accent === "emerald" ? "text-emerald-600 dark:text-emerald-400" : accent === "rose" ? "text-rose-600 dark:text-rose-400" : "";
  return (
    <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-3 dark:border-zinc-800 dark:bg-zinc-800">
      <p className="text-xs text-zinc-500 dark:text-zinc-400">{label}</p>
      <p className={`mt-1 text-xl font-semibold ${color}`}>{value}</p>
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
