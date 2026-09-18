import type { BacktestRebuildJob, BacktestRun, CandleCatchupJob, CandlePoint, ConsensusBacktestResult, ConsensusCheckResult, ConsensusSignal, MLPrediction, MLTrainResult, PaperTrade, PaperTradeAccount, PaperTradeResult, PPOPolicy, PPOTrainDiagnostics, RLAccuracy, RLInsights, RLLearningCurve, RLMemorySummary, RLSignal, RLTrainAllJob, RunAllFlowsJob, Signal } from "./types";
import { clearToken, getToken } from "./auth";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "https://forex-assistant.fastapicloud.dev";

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getToken();
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...init?.headers,
    },
    cache: "no-store",
  });
  if (res.status === 401 && path !== "/auth/login") {
    clearToken();
    if (typeof window !== "undefined" && window.location.pathname !== "/login") {
      window.location.href = "/login";
    }
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(body.detail ?? `Request failed: ${res.status}`, res.status);
  }
  return res.json();
}

export const api = {
  login(username: string, password: string) {
    return request<{ access_token: string; token_type: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
  },

  ingest(interval: string) {
    return request<Record<string, string>>(`/ingest/${interval}`, { method: "POST" });
  },

  // Catches every pair x interval's candles up to now, from wherever ingestion last left
  // off -- runs as a background job (real gaps across 20 combos, rate-limit-paced, can take
  // well over a minute), same job-doc-plus-polling pattern as startRebuildBacktests below.
  startCandleCatchup() {
    return request<CandleCatchupJob>("/ingest/catch-up", { method: "POST" });
  },

  getCandleCatchupJob(jobId: string) {
    return request<CandleCatchupJob>(`/ingest/catch-up/${jobId}`);
  },

  getLatestCandleCatchupJob() {
    return request<CandleCatchupJob | null>("/ingest/catch-up-latest");
  },

  cancelCandleCatchupJob(jobId: string) {
    return request<CandleCatchupJob>(`/ingest/catch-up/${jobId}/cancel`, { method: "POST" });
  },

  getCandles(pair: string, interval: string, profile: string, limit = 250) {
    const qs = new URLSearchParams({ pair, profile, limit: String(limit) });
    return request<CandlePoint[]>(`/candles/${interval}?${qs.toString()}`);
  },

  listBacktestRuns(params: { pair?: string; profile?: string; limit?: number } = {}) {
    const qs = new URLSearchParams();
    if (params.pair) qs.set("pair", params.pair);
    if (params.profile) qs.set("profile", params.profile);
    if (params.limit) qs.set("limit", String(params.limit));
    const query = qs.toString();
    return request<BacktestRun[]>(`/backtest/runs${query ? `?${query}` : ""}`);
  },

  getBacktestRun(runId: string) {
    return request<BacktestRun>(`/backtest/runs/${runId}`);
  },

  getBacktestRunSignals(runId: string, params: { status?: string; limit?: number } = {}) {
    const qs = new URLSearchParams();
    if (params.status) qs.set("status", params.status);
    if (params.limit) qs.set("limit", String(params.limit));
    const query = qs.toString();
    return request<Signal[]>(`/backtest/runs/${runId}/signals${query ? `?${query}` : ""}`);
  },

  generateConsensus(pair: string, interval: string) {
    // pair goes in the query string, not the path — a literal '/' in a path segment
    // breaks Starlette's routing even when percent-encoded.
    return request<ConsensusCheckResult>(
      `/consensus/${interval}?pair=${encodeURIComponent(pair)}`,
      { method: "POST" },
    );
  },

  listConsensusSignals(params: { pair?: string; limit?: number } = {}) {
    const qs = new URLSearchParams();
    if (params.pair) qs.set("pair", params.pair);
    if (params.limit) qs.set("limit", String(params.limit));
    const query = qs.toString();
    return request<ConsensusSignal[]>(`/consensus${query ? `?${query}` : ""}`);
  },

  runConsensusBacktest(
    pair: string,
    interval: string,
    opts: { train_frac?: number; max_lookforward?: number } = {},
  ) {
    const qs = new URLSearchParams({ pair });
    if (opts.train_frac !== undefined) qs.set("train_frac", String(opts.train_frac));
    if (opts.max_lookforward !== undefined) qs.set("max_lookforward", String(opts.max_lookforward));
    return request<ConsensusBacktestResult>(
      `/consensus/backtest/${interval}?${qs.toString()}`,
      { method: "POST" },
    );
  },

  trainMLModel(trainFrac?: number) {
    const qs = trainFrac !== undefined ? `?train_frac=${trainFrac}` : "";
    return request<MLTrainResult>(`/ml/train${qs}`, { method: "POST" });
  },

  listMLRuns(limit = 20) {
    return request<MLTrainResult[]>(`/ml/runs?limit=${limit}`);
  },

  // Rebuilds qualifying backtest data for the ML classifier (see
  // app/services/ml_training_data.py) on a GitHub Actions runner, then retrains -- same
  // job-doc-plus-polling pattern as startTrainAllRL below.
  startRebuildBacktests(maxLookforward?: number) {
    const qs = maxLookforward !== undefined ? `?max_lookforward=${maxLookforward}` : "";
    return request<BacktestRebuildJob>(`/backtest/rebuild-all${qs}`, { method: "POST" });
  },

  getRebuildBacktestsJob(jobId: string) {
    return request<BacktestRebuildJob>(`/backtest/rebuild-all/${jobId}`);
  },

  getLatestRebuildBacktestsJob() {
    return request<BacktestRebuildJob | null>(`/backtest/rebuild-all-latest`);
  },

  cancelRebuildBacktestsJob(jobId: string) {
    return request<BacktestRebuildJob>(`/backtest/rebuild-all/${jobId}/cancel`, { method: "POST" });
  },

  predictML(pair: string, interval: string, profile: string) {
    // pair goes in the query string, not the path — a literal '/' in a path segment
    // breaks Starlette's routing even when percent-encoded.
    return request<MLPrediction>(
      `/ml/predict/${interval}/${profile}?pair=${encodeURIComponent(pair)}`,
      { method: "POST" },
    );
  },

  // PPO (app/services/ppo_engine.py) -- this project's RL agent. Linear Q-learning was
  // retired; POST /rl/train and POST /rl/signal are PPO-only now, not one of two options.
  trainRLPolicy(
    pair: string,
    interval: string,
    opts: {
      total_timesteps?: number; train_frac?: number; max_lookforward?: number; starting_balance?: number;
      random_seed?: number; target_atr_mult?: number; stop_atr_mult?: number;
    } = {},
  ) {
    const qs = new URLSearchParams({ pair });
    if (opts.total_timesteps !== undefined) qs.set("total_timesteps", String(opts.total_timesteps));
    if (opts.train_frac !== undefined) qs.set("train_frac", String(opts.train_frac));
    if (opts.max_lookforward !== undefined) qs.set("max_lookforward", String(opts.max_lookforward));
    if (opts.starting_balance !== undefined) qs.set("starting_balance", String(opts.starting_balance));
    if (opts.random_seed !== undefined) qs.set("random_seed", String(opts.random_seed));
    // Both required together -- see POST /rl/train/{interval}'s own docstring.
    if (opts.target_atr_mult !== undefined) qs.set("target_atr_mult", String(opts.target_atr_mult));
    if (opts.stop_atr_mult !== undefined) qs.set("stop_atr_mult", String(opts.stop_atr_mult));
    return request<{ policy: PPOPolicy; evaluation: BacktestRun; poc_diagnostics: PPOTrainDiagnostics }>(
      `/rl/train/${interval}?${qs.toString()}`,
      { method: "POST" },
    );
  },

  startTrainAllRL(opts: { total_timesteps?: number; train_frac?: number; starting_balance?: number } = {}) {
    const qs = new URLSearchParams();
    if (opts.total_timesteps !== undefined) qs.set("total_timesteps", String(opts.total_timesteps));
    if (opts.train_frac !== undefined) qs.set("train_frac", String(opts.train_frac));
    if (opts.starting_balance !== undefined) qs.set("starting_balance", String(opts.starting_balance));
    const query = qs.toString();
    return request<RLTrainAllJob>(`/rl/train-all${query ? `?${query}` : ""}`, { method: "POST" });
  },

  getTrainAllRLJob(jobId: string) {
    return request<RLTrainAllJob>(`/rl/train-all/${jobId}`);
  },

  getLatestTrainAllRLJob() {
    return request<RLTrainAllJob | null>(`/rl/train-all-latest`);
  },

  cancelTrainAllRLJob(jobId: string) {
    return request<RLTrainAllJob>(`/rl/train-all/${jobId}/cancel`, { method: "POST" });
  },

  listRLPolicies(params: { pair?: string; interval?: string; limit?: number } = {}) {
    const qs = new URLSearchParams();
    if (params.pair) qs.set("pair", params.pair);
    if (params.interval) qs.set("interval", params.interval);
    if (params.limit) qs.set("limit", String(params.limit));
    const query = qs.toString();
    return request<PPOPolicy[]>(`/rl/policies${query ? `?${query}` : ""}`);
  },

  generateRLSignal(pair: string, interval: string, balance?: number) {
    // pair goes in the query string, not the path — a literal '/' in a path segment
    // breaks Starlette's routing even when percent-encoded.
    const qs = new URLSearchParams({ pair });
    if (balance !== undefined) qs.set("balance", String(balance));
    // memory is absent when the action was HOLD (no trade, nothing to look up memory against)
    // -- see create_rl_signal's early return in app/main.py. memory_override at the top level
    // (not on `signal`) only appears when memory_gate blocked the trade to HOLD outright --
    // when it only downsized LARGE->SMALL instead, the reason lives on signal.memory_override.
    // q_values holds PPO's action-probability distribution (real probabilities, sum to 1) --
    // see RLSignal.algo's own comment.
    return request<{
      signal: RLSignal | null; q_values: Record<string, number>;
      memory?: RLMemorySummary; memory_override?: string | null;
      // Present when this pair/interval's latest training run showed a clear losing edge --
      // live signals are withheld (not just downsized like memory_override) until a retrain
      // clears it. Training itself is unaffected; this re-checks fresh on every call.
      excluded_reason?: string | null;
      // Present when the ML classifier's hit-probability gate blocked this specific direction.
      ml_blocked_reason?: string | null;
    }>(
      `/rl/signal/${interval}?${qs.toString()}`,
      { method: "POST" },
    );
  },

  getRLAccuracy(params: { pair?: string; interval?: string; limit?: number } = {}) {
    const qs = new URLSearchParams();
    if (params.pair) qs.set("pair", params.pair);
    if (params.interval) qs.set("interval", params.interval);
    if (params.limit) qs.set("limit", String(params.limit));
    const query = qs.toString();
    return request<RLAccuracy>(`/rl/accuracy${query ? `?${query}` : ""}`);
  },

  getRLLearningCurve(days?: number) {
    const qs = days !== undefined ? `?days=${days}` : "";
    return request<RLLearningCurve>(`/rl/learning-curve${qs}`);
  },

  getRLInsights() {
    return request<RLInsights>(`/rl/insights`);
  },

  listRLSignals(params: { pair?: string; limit?: number } = {}) {
    const qs = new URLSearchParams();
    if (params.pair) qs.set("pair", params.pair);
    if (params.limit) qs.set("limit", String(params.limit));
    const query = qs.toString();
    return request<RLSignal[]>(`/rl/signals${query ? `?${query}` : ""}`);
  },

  scoreRLSignals(maxLookforward?: number) {
    const qs = maxLookforward !== undefined ? `?max_lookforward=${maxLookforward}` : "";
    return request<{ hit: number; miss: number; expired: number; still_pending: number; skipped_no_data: number }>(
      `/rl/score${qs}`, { method: "POST" },
    );
  },

  getPaperTradeAccount() {
    return request<PaperTradeAccount>("/paper-trade/account");
  },

  executePaperTrade(
    pair: string,
    interval: string,
    profile: string,
    opts: { stake?: number; multiplier?: number; target_atr_mult?: number; stop_atr_mult?: number } = {},
  ) {
    const qs = new URLSearchParams({ pair });
    if (opts.stake !== undefined) qs.set("stake", String(opts.stake));
    if (opts.multiplier !== undefined) qs.set("multiplier", String(opts.multiplier));
    if (opts.target_atr_mult !== undefined) qs.set("target_atr_mult", String(opts.target_atr_mult));
    if (opts.stop_atr_mult !== undefined) qs.set("stop_atr_mult", String(opts.stop_atr_mult));
    return request<PaperTradeResult>(
      `/paper-trade/${interval}/${profile}?${qs.toString()}`,
      { method: "POST" },
    );
  },

  listOpenPaperTrades() {
    return request<PaperTrade[]>("/paper-trade/open");
  },

  listPaperTradeHistory(params: { status?: string; limit?: number } = {}) {
    const qs = new URLSearchParams();
    if (params.status) qs.set("status", params.status);
    if (params.limit) qs.set("limit", String(params.limit));
    const query = qs.toString();
    return request<PaperTrade[]>(`/paper-trade/history${query ? `?${query}` : ""}`);
  },

  // Manually replicates one keep-fresh.yml cron cycle (ingest, retrain RL, check consensus,
  // generate RL signals, score, retrain ML) -- for when the GitHub Actions cron has gone
  // quiet for a while. Runs as a background job (can take well over a minute) -- poll
  // getRunAllFlowsJob for status/results, same pattern as RL "train all".
  startRunAllFlows() {
    return request<{ job_id: string; status: string }>("/ops/run-all-flows", { method: "POST" });
  },

  getRunAllFlowsJob(jobId: string) {
    return request<RunAllFlowsJob>(`/ops/run-all-flows/${jobId}`);
  },

  cancelRunAllFlows(jobId: string) {
    return request<RunAllFlowsJob>(`/ops/run-all-flows/${jobId}/cancel`, { method: "POST" });
  },
};

export { ApiError };
