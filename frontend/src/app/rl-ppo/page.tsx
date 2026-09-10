"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { INTERVALS, PAIRS, type BacktestRun, type PPOPolicy, type PPOTrainDiagnostics, type RLMemorySummary, type RLSignal } from "@/lib/types";
import { DirectionBadge } from "@/components/Badges";

export default function RLPPOPage() {
  const [pair, setPair] = useState<string>(PAIRS[0]);
  const [interval, setInterval_] = useState<string>("5min");
  const [totalTimesteps, setTotalTimesteps] = useState(50_000);
  const [trainFrac, setTrainFrac] = useState(0.7);
  const [startingBalance, setStartingBalance] = useState(50);

  const [training, setTraining] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainResult, setTrainResult] = useState<{ policy: PPOPolicy; evaluation: BacktestRun; poc_diagnostics: PPOTrainDiagnostics } | null>(null);

  const [policies, setPolicies] = useState<PPOPolicy[]>([]);
  const [policiesLoading, setPoliciesLoading] = useState(true);

  const [balance, setBalance] = useState(50);
  const [generating, setGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [generateResult, setGenerateResult] = useState<{
    signal: RLSignal | null; q_values: Record<string, number>;
    memory?: RLMemorySummary; memory_override?: string | null;
    excluded_reason?: string | null; ml_blocked_reason?: string | null;
  } | null>(null);

  const [recentSignals, setRecentSignals] = useState<RLSignal[]>([]);
  const [recentLoading, setRecentLoading] = useState(true);

  async function loadPolicies() {
    setPoliciesLoading(true);
    try {
      setPolicies(await api.listPPOPolicies({ limit: 20 }));
    } catch {
      // non-critical section -- a failed load here shouldn't block the rest of the page
    } finally {
      setPoliciesLoading(false);
    }
  }

  async function loadRecentSignals() {
    setRecentLoading(true);
    try {
      // GET /rl/signals has no algo filter server-side (it's a small, shared trade log) --
      // filtering client-side here rather than adding one for a single beta page's sake.
      const all = await api.listRLSignals({ limit: 50 });
      setRecentSignals(all.filter((s) => s.algo === "ppo"));
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
      const result = await api.trainPPOPolicy(pair, interval, {
        total_timesteps: totalTimesteps, train_frac: trainFrac, starting_balance: startingBalance,
      });
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
    setGenerateResult(null);
    try {
      const result = await api.generatePPOSignal(pair, interval, balance);
      setGenerateResult(result);
      if (result.signal) await loadRecentSignals();
    } catch (e) {
      setGenerateError(e instanceof ApiError ? e.message : "Couldn't generate a signal.");
    } finally {
      setGenerating(false);
    }
  }

  return (
    <div className="flex flex-col gap-8">
      <section>
        <h1 className="text-xl font-semibold tracking-tight">PPO Agent (beta)</h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          A second, independent RL agent (stable-baselines3&apos;s PPO via a gymnasium.Env
          adapter) trained against the exact same replay mechanics as the RL Agent page&apos;s
          linear Q-learning policy — same state features, same reward, same three-gate live
          safety stack. Additive, not a replacement: nothing here touches the linear policies,
          and a pair/interval only starts using PPO once you train it below <em>and</em>{" "}
          generate signals from this page instead of the RL Agent page. Every check on a
          freshly-trained policy below (&quot;beats random&quot;, model size, save/load
          latency) is a proof-of-concept sanity check, not a guarantee — review the evaluation
          numbers before trusting it live.
        </p>
      </section>

      <section className="card p-4">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Train a PPO policy</h2>
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
            {training ? "Training…" : "Train PPO"}
          </button>
        </div>
        <p className="mt-2 text-xs text-zinc-400">
          Unlike the linear policy, PPO has no warm start — every training run starts a fresh
          model rather than continuing the last one. This can take a while (real environment
          steps, not epsilon-greedy episodes); a few thousand timesteps is a quick check, tens
          of thousands is closer to a real attempt.
        </p>
        {trainError && (
          <p className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {trainError}
          </p>
        )}
        {trainResult && (
          <div className="mt-4 flex flex-col gap-3">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <StatTile label="Hit rate" value={trainResult.evaluation.hit_rate_pct != null ? `${trainResult.evaluation.hit_rate_pct}%` : "—"} />
              <StatTile label="Total return" value={trainResult.evaluation.total_return_pct != null ? `${trainResult.evaluation.total_return_pct}%` : "—"} />
              <StatTile label="Directional / hold" value={`${trainResult.evaluation.directional_signals} / ${trainResult.evaluation.hold_signals}`} />
              <StatTile
                label="Beats random?"
                value={trainResult.poc_diagnostics.beats_random ? "Yes" : "No"}
                accent={trainResult.poc_diagnostics.beats_random ? "emerald" : "rose"}
              />
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Random baseline:{" "}
              {trainResult.poc_diagnostics.random_baseline_total_return_pct != null
                ? `${trainResult.poc_diagnostics.random_baseline_total_return_pct}%`
                : "—"}{" "}
              total return · model size {(trainResult.poc_diagnostics.model_size_bytes / 1024).toFixed(0)} KB
              · save+load {trainResult.poc_diagnostics.save_load_latency_ms.toFixed(1)} ms
            </p>
            <p className="text-xs font-mono text-zinc-400">policy_id: {trainResult.policy.policy_id}</p>
          </div>
        )}
      </section>

      <section>
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Trained PPO policies</h2>
        {policiesLoading && <p className="mt-4 text-sm text-zinc-500">Loading…</p>}
        {!policiesLoading && policies.length === 0 && (
          <p className="mt-4 text-sm text-zinc-500">No PPO policies trained yet — train one above.</p>
        )}
        {policies.length > 0 && (
          <div className="mt-4 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
            <table className="w-full text-left text-xs">
              <thead className="table-head uppercase">
                <tr>
                  <th className="px-3 py-1.5">Pair</th>
                  <th className="px-3 py-1.5">Interval</th>
                  <th className="px-3 py-1.5">Hit rate</th>
                  <th className="px-3 py-1.5">Total return</th>
                  <th className="px-3 py-1.5">Directional / hold</th>
                  <th className="px-3 py-1.5">Timesteps</th>
                  <th className="px-3 py-1.5">When</th>
                </tr>
              </thead>
              <tbody>
                {policies.map((p) => (
                  <tr key={p.policy_id} className="border-t border-zinc-100 dark:border-zinc-800">
                    <td className="px-3 py-1.5 font-mono">{p.pair}</td>
                    <td className="px-3 py-1.5 font-mono">{p.interval}</td>
                    <td className="px-3 py-1.5">{p.evaluation?.hit_rate_pct != null ? `${p.evaluation.hit_rate_pct}%` : "—"}</td>
                    <td
                      className={`px-3 py-1.5 font-semibold ${
                        (p.evaluation?.total_return_pct ?? 0) >= 0
                          ? "text-emerald-600 dark:text-emerald-400"
                          : "text-rose-600 dark:text-rose-400"
                      }`}
                    >
                      {p.evaluation?.total_return_pct != null ? `${p.evaluation.total_return_pct}%` : "—"}
                    </td>
                    <td className="px-3 py-1.5">{p.evaluation ? `${p.evaluation.directional_signals} / ${p.evaluation.hold_signals}` : "—"}</td>
                    <td className="px-3 py-1.5">{p.total_timesteps.toLocaleString()}</td>
                    <td className="px-3 py-1.5 text-zinc-500">{new Date(p.created_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="card p-4">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Generate a live PPO signal</h2>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          Same three-gate safety stack as the RL Agent page (live-exclusion, ML quality gate,
          case-memory), fed by PPO&apos;s action instead of a Q-policy argmax. Shares the same
          one-open-position-per-pair/interval lock — if the linear agent already has an open
          position here, this returns that pending signal as-is instead of deciding again.
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
          <Field label="Current balance ($)">
            <input
              type="number" step="10" min="1" value={balance}
              onChange={(e) => setBalance(Number(e.target.value))}
              className="select w-28"
            />
          </Field>
          <button onClick={handleGenerate} disabled={generating} className="btn-primary">
            {generating ? "Deciding…" : "Generate signal"}
          </button>
        </div>
        {generateError && (
          <p className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {generateError}
          </p>
        )}
        {generateResult && (
          <div className="mt-4 rounded-lg border border-zinc-200 p-4 dark:border-zinc-800">
            {generateResult.signal ? (
              <>
                <p className="flex items-center gap-2 text-lg font-semibold">
                  <DirectionBadge direction={generateResult.signal.direction} />
                  {generateResult.signal.entry_price.toFixed(5)}
                  {generateResult.signal.size_tier && (
                    <span className="text-xs font-normal text-zinc-400">({generateResult.signal.size_tier})</span>
                  )}
                </p>
                <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                  Target {generateResult.signal.target_price.toFixed(5)} · Stop {generateResult.signal.stop_price.toFixed(5)}
                  {generateResult.signal.position_size_units != null && ` · ${generateResult.signal.position_size_units} units`}
                </p>
                {generateResult.signal.memory_override && (
                  <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">{generateResult.signal.memory_override}</p>
                )}
              </>
            ) : (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                {generateResult.excluded_reason ?? generateResult.ml_blocked_reason ?? "HOLD — no trade this bar."}
              </p>
            )}
            <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 rounded-md bg-zinc-50 px-3 py-2 text-xs font-mono dark:bg-zinc-800">
              <span className="w-full text-[11px] font-sans text-zinc-400">Action probabilities (PPO&apos;s own, not softmax-derived):</span>
              {Object.entries(generateResult.q_values)
                .sort(([, a], [, b]) => b - a)
                .map(([action, p]) => (
                  <span key={action}>
                    {action}: <span className="text-sky-600 dark:text-sky-400">{(p * 100).toFixed(1)}%</span>
                  </span>
                ))}
            </div>
            {generateResult.memory && (
              <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
                Memory: {generateResult.memory.cases_found} similar case(s)
                {generateResult.memory.hit_rate_pct != null && `, ${generateResult.memory.hit_rate_pct}% hit rate`}
              </p>
            )}
          </div>
        )}
      </section>

      <section>
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Recent PPO signals</h2>
        {recentLoading && <p className="mt-4 text-sm text-zinc-500">Loading…</p>}
        {!recentLoading && recentSignals.length === 0 && (
          <p className="mt-4 text-sm text-zinc-500">No PPO signals yet — generate one above.</p>
        )}
        {recentSignals.length > 0 && (
          <div className="mt-4 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
            <table className="w-full text-left text-xs">
              <thead className="table-head uppercase">
                <tr>
                  <th className="px-3 py-1.5">Pair</th>
                  <th className="px-3 py-1.5">Interval</th>
                  <th className="px-3 py-1.5">Direction</th>
                  <th className="px-3 py-1.5">Entry</th>
                  <th className="px-3 py-1.5">Status</th>
                  <th className="px-3 py-1.5">When</th>
                </tr>
              </thead>
              <tbody>
                {recentSignals.map((s) => (
                  <tr key={s.signal_id ?? `${s.pair}-${s.timestamp}`} className="border-t border-zinc-100 dark:border-zinc-800">
                    <td className="px-3 py-1.5 font-mono">{s.pair}</td>
                    <td className="px-3 py-1.5 font-mono">{s.interval}</td>
                    <td className="px-3 py-1.5"><DirectionBadge direction={s.direction} /></td>
                    <td className="px-3 py-1.5">{s.entry_price.toFixed(5)}</td>
                    <td className="px-3 py-1.5">{s.status}</td>
                    <td className="px-3 py-1.5 text-zinc-500">{new Date(s.timestamp).toLocaleString()}</td>
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
