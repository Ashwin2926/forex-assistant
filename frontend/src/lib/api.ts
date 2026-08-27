import type { BacktestRun, CandlePoint, ConsensusBacktestResult, ConsensusCheckResult, ConsensusSignal, MLPrediction, MLTrainResult, OptimizeRankBy, OptimizeResult, PaperTrade, PaperTradeAccount, PaperTradeResult, RLAccuracy, RLMemorySummary, RLPolicy, RLSignal, RLTrainAllJob, RuleConfig, RunAllFlowsResult, Signal, SignalAccuracy } from "./types";
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

  generateSignal(pair: string, interval: string, profile: string) {
    // pair goes in the query string, not the path — a literal '/' in a path segment
    // breaks Starlette's routing even when percent-encoded.
    return request<Signal>(
      `/signals/${interval}/${profile}?pair=${encodeURIComponent(pair)}`,
      { method: "POST" },
    );
  },

  listSignals(params: { pair?: string; limit?: number } = {}) {
    const qs = new URLSearchParams();
    if (params.pair) qs.set("pair", params.pair);
    if (params.limit) qs.set("limit", String(params.limit));
    const query = qs.toString();
    return request<Signal[]>(`/signals${query ? `?${query}` : ""}`);
  },

  scoreSignals(maxLookforward?: number) {
    const qs = maxLookforward !== undefined ? `?max_lookforward=${maxLookforward}` : "";
    return request<{ hit: number; miss: number; expired: number; still_pending: number; skipped_no_data: number }>(
      `/signals/score${qs}`, { method: "POST" },
    );
  },

  getSignalAccuracy(params: { pair?: string; profile?: string; limit?: number } = {}) {
    const qs = new URLSearchParams();
    if (params.pair) qs.set("pair", params.pair);
    if (params.profile) qs.set("profile", params.profile);
    if (params.limit) qs.set("limit", String(params.limit));
    const query = qs.toString();
    return request<SignalAccuracy>(`/signals/accuracy${query ? `?${query}` : ""}`);
  },

  getCandles(pair: string, interval: string, profile: string, limit = 250) {
    const qs = new URLSearchParams({ pair, profile, limit: String(limit) });
    return request<CandlePoint[]>(`/candles/${interval}?${qs.toString()}`);
  },

  runBacktest(
    pair: string,
    interval: string,
    profile: string,
    opts: { target_atr_mult?: number; stop_atr_mult?: number; max_lookforward?: number; config?: Partial<RuleConfig> } = {},
  ) {
    const qs = new URLSearchParams({ pair });
    if (opts.target_atr_mult !== undefined) qs.set("target_atr_mult", String(opts.target_atr_mult));
    if (opts.stop_atr_mult !== undefined) qs.set("stop_atr_mult", String(opts.stop_atr_mult));
    if (opts.max_lookforward !== undefined) qs.set("max_lookforward", String(opts.max_lookforward));
    return request<BacktestRun>(
      `/backtest/${interval}/${profile}?${qs.toString()}`,
      { method: "POST", body: opts.config ? JSON.stringify(opts.config) : undefined },
    );
  },

  runOptimize(
    pair: string,
    interval: string,
    profile: string,
    opts: {
      train_frac?: number; target_atr_mult?: number; stop_atr_mult?: number;
      max_lookforward?: number; min_directional_signals?: number; rank_by?: OptimizeRankBy; configs?: RuleConfig[];
    } = {},
  ) {
    const qs = new URLSearchParams({ pair });
    if (opts.train_frac !== undefined) qs.set("train_frac", String(opts.train_frac));
    if (opts.target_atr_mult !== undefined) qs.set("target_atr_mult", String(opts.target_atr_mult));
    if (opts.stop_atr_mult !== undefined) qs.set("stop_atr_mult", String(opts.stop_atr_mult));
    if (opts.max_lookforward !== undefined) qs.set("max_lookforward", String(opts.max_lookforward));
    if (opts.min_directional_signals !== undefined) qs.set("min_directional_signals", String(opts.min_directional_signals));
    if (opts.rank_by !== undefined) qs.set("rank_by", opts.rank_by);
    return request<OptimizeResult>(
      `/backtest/optimize/${interval}/${profile}?${qs.toString()}`,
      { method: "POST", body: opts.configs ? JSON.stringify(opts.configs) : undefined },
    );
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

  predictML(pair: string, interval: string, profile: string) {
    // pair goes in the query string, not the path — a literal '/' in a path segment
    // breaks Starlette's routing even when percent-encoded.
    return request<MLPrediction>(
      `/ml/predict/${interval}/${profile}?pair=${encodeURIComponent(pair)}`,
      { method: "POST" },
    );
  },

  trainRLPolicy(
    pair: string,
    interval: string,
    opts: { episodes?: number; train_frac?: number; max_lookforward?: number; starting_balance?: number } = {},
  ) {
    const qs = new URLSearchParams({ pair });
    if (opts.episodes !== undefined) qs.set("episodes", String(opts.episodes));
    if (opts.train_frac !== undefined) qs.set("train_frac", String(opts.train_frac));
    if (opts.max_lookforward !== undefined) qs.set("max_lookforward", String(opts.max_lookforward));
    if (opts.starting_balance !== undefined) qs.set("starting_balance", String(opts.starting_balance));
    return request<{ policy: RLPolicy; evaluation: BacktestRun }>(
      `/rl/train/${interval}?${qs.toString()}`,
      { method: "POST" },
    );
  },

  startTrainAllRL(opts: { episodes?: number; train_frac?: number; starting_balance?: number } = {}) {
    const qs = new URLSearchParams();
    if (opts.episodes !== undefined) qs.set("episodes", String(opts.episodes));
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
    return request<RLPolicy[]>(`/rl/policies${query ? `?${query}` : ""}`);
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
    return request<{
      signal: RLSignal | null; q_values: Record<string, number>;
      memory?: RLMemorySummary; memory_override?: string | null;
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

  // Manually replicates one keep-fresh.yml cron cycle (ingest, generate signals, score,
  // retrain ML) -- for when the GitHub Actions cron has gone quiet for a while. Excludes RL
  // training (see POST /rl/train-all separately) -- see the endpoint's own docstring for why.
  // Takes a while (many sequential ingest/signal/score calls across every pair/interval) --
  // callers should show a loading state, not assume this resolves quickly.
  runAllFlows() {
    return request<RunAllFlowsResult>("/ops/run-all-flows", { method: "POST" });
  },
};

export { ApiError };
