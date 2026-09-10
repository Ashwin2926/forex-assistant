// Mirrors app/models/schemas.py — keep in sync with the backend.

export type Direction = "BUY" | "SELL" | "HOLD";
export type Profile = "intraday";
export type SignalStatus = "pending" | "hit" | "miss" | "expired" | "superseded";
export type SignalSource = "live" | "backtest";

export interface SignalReason {
  rule: string;
  passed: boolean;
  detail: string;
  // The single most decision-relevant number behind this rule's verdict (RSI reading,
  // normalized EMA spread %, MACD histogram, ATR%, or session hour) — absent on signals
  // generated before this field existed.
  value?: number | null;
}

export interface Signal {
  _id?: string;
  pair: string;
  profile: Profile;
  interval: string;
  timestamp: string;
  direction: Direction;
  confidence: number;
  reasons: SignalReason[];
  price_at_signal: number;
  status: SignalStatus;
  outcome_price?: number | null;
  outcome_timestamp?: string | null;
  outcome_pct_move?: number | null;
  source: SignalSource;
  run_id?: string | null;
  target_price?: number | null;
  stop_price?: number | null;
  candles_to_outcome?: number | null;
  // RL sizing-aware trade log only (see rl_engine.train_rl_policy) -- null/absent for every
  // other Signal.
  size_tier?: "SMALL" | "LARGE" | null;
  risk_fraction?: number | null;
  balance_at_signal?: number | null;
  position_size_units?: number | null;
}

export interface StrategyCall {
  strategy: string; // "market_structure" | "order_blocks" | "fair_value_gap" | "liquidity_sweep" | "supply_demand"
  direction: Direction;
  entry_price: number;
  target_price: number | null;
  stop_price: number | null;
  reasons: SignalReason[];
}

export interface ConsensusSignal {
  _id?: string;
  pair: string;
  interval: string;
  timestamp: string;
  direction: "BUY" | "SELL";
  entry_price: number;
  target_price: number;
  stop_price: number;
  agreeing_count: number;
  strategy_calls: StrategyCall[];
  status: SignalStatus;
  outcome_price?: number | null;
  outcome_timestamp?: string | null;
  outcome_pct_move?: number | null;
  candles_to_outcome?: number | null;
  source: SignalSource;
  run_id?: string | null;
}

export interface ConsensusCheckResult {
  consensus: ConsensusSignal | null;
  strategy_calls: StrategyCall[];
}

export interface RuleConfig {
  ema_fast: number;
  ema_slow: number;
  rsi_period: number;
  rsi_oversold: number;
  rsi_overbought: number;
  macd_fast: number;
  macd_slow: number;
  macd_signal: number;
  atr_period: number;
  volatility_threshold_pct: number;
  session_filter_enabled: boolean;
  session_start_hour_utc: number;
  session_end_hour_utc: number;
  target_atr_mult: number;
  stop_atr_mult: number;
}

export interface RuleStat {
  rule: string;
  fired_count: number;
  directional_signals_agreed: number;
  hits_when_agreed: number;
  hit_rate_when_agreed_pct: number | null;
}

export interface BacktestRun {
  _id?: string;
  run_id: string;
  pair: string;
  interval: string;
  profile: Profile | "consensus"; // "consensus" from POST /consensus/backtest/{interval}, reusing this model
  created_at: string;
  rule_config: RuleConfig;
  target_atr_mult: number;
  stop_atr_mult: number;
  max_lookforward: number;
  candles_evaluated: number;
  total_signals: number;
  hold_signals: number;
  directional_signals: number;
  hits: number;
  misses: number;
  expired: number;
  hit_rate_pct: number | null;
  avg_win_pct: number | null;
  avg_loss_pct: number | null;
  expectancy_pct: number | null;
  avg_confidence_hit: number | null;
  avg_confidence_miss: number | null;
  rule_stats: RuleStat[];
  // RL sizing-aware eval runs only (profile "rl") -- null/absent for every other BacktestRun.
  starting_balance?: number | null;
  ending_balance?: number | null;
  total_return_pct?: number | null;
}

export interface MLCalibrationBucket {
  range_label: string;
  count: number;
  actual_hit_rate_pct: number;
}

export interface MLTrainResult {
  _id?: string;
  run_id: string;
  created_at: string;
  train_samples: number;
  test_samples: number;
  train_accuracy: number;
  test_accuracy: number;
  test_precision: number | null;
  test_recall: number | null;
  feature_importances: Record<string, number>;
  test_calibration: MLCalibrationBucket[];
}

export interface RLPolicy {
  _id?: string;
  policy_id: string;
  pair: string;
  interval: string;
  created_at: string;
  episodes: number;
  train_frac: number;
  weights: Record<string, number[]>;
  feature_names: string[];
  eval_run_id: string;
  // The balance this policy was trained against -- predict-time needs this same reference
  // point to compute the live balance_log_ratio feature correctly. Absent on policies trained
  // before position sizing existed (GET /rl/policies returns raw Mongo docs, not validated
  // through the RLPolicy Pydantic model, so an old doc really can be missing this key).
  starting_balance?: number | null;
  // Only present from GET /rl/policies (enriched server-side from the linked BacktestRun) --
  // absent on the RLPolicy returned directly by POST /rl/train.
  evaluation?: {
    hit_rate_pct: number | null;
    expectancy_pct: number | null;
    directional_signals: number;
    hold_signals: number;
    starting_balance: number | null;
    ending_balance: number | null;
    total_return_pct: number | null;
  } | null;
}

// Signal Stack v2 phase 3 -- this project's second RL agent, PPO (stable-baselines3), trained
// against the exact same replay mechanics RLPolicy's linear Q-learning uses. Separate model/
// collection/page from RLPolicy on purpose (see app/models/schemas.py's PPOPolicy) -- additive
// alongside the linear agent, not a replacement for it. model_bytes (the serialized neural
// network) never appears here -- the backend excludes it from every response that returns this.
export interface PPOPolicy {
  _id?: string;
  policy_id: string;
  pair: string;
  interval: string;
  created_at: string;
  feature_names: string[];
  eval_run_id: string;
  starting_balance: number;
  total_timesteps: number;
  // Only present from GET /rl/ppo/policies (enriched server-side from the linked BacktestRun)
  // -- absent on the PPOPolicy returned directly by POST /rl/train-ppo.
  evaluation?: {
    hit_rate_pct: number | null;
    expectancy_pct: number | null;
    directional_signals: number;
    hold_signals: number;
    starting_balance: number | null;
    ending_balance: number | null;
    total_return_pct: number | null;
  } | null;
}

// Proof-of-concept sanity checks POST /rl/train-ppo returns alongside every real training
// run, not just ppo_engine.py's own synthetic-data smoke test -- see that endpoint's docstring.
export interface PPOTrainDiagnostics {
  beats_random: boolean;
  model_size_bytes: number;
  save_load_latency_ms: number;
  random_baseline_total_return_pct: number | null;
}

export interface RLTrainAllCell {
  pair: string;
  interval: string;
  ok: boolean;
  error?: string | null;
  policy_id?: string | null;
  hit_rate_pct?: number | null;
  expectancy_pct?: number | null;
  directional_signals?: number | null;
  hold_signals?: number | null;
  starting_balance?: number | null;
  ending_balance?: number | null;
  total_return_pct?: number | null;
}

// A "train every pair x interval" batch run, tracked server-side (see app/main.py's
// run_train_all_job) so the ~15-minute-total run survives the triggering browser tab being
// closed, backgrounded, or losing connectivity -- the page just starts a job and polls
// GET /rl/train-all/{job_id} instead of holding 20 sequential fetches open itself.
export interface RLTrainAllJob {
  job_id: string;
  status: "running" | "done" | "cancelled";
  created_at: string;
  finished_at?: string | null;
  episodes: number;
  train_frac: number;
  starting_balance: number;
  total: number;
  completed: number;
  results: RLTrainAllCell[];
  cancel_requested: boolean;
}

export interface RLSignal {
  _id?: string;
  // Absent on any signal generated before this field existed -- GET /rl/signals returns raw
  // Mongo docs, not validated through the RLSignal Pydantic model.
  signal_id?: string;
  pair: string;
  interval: string;
  timestamp: string;
  direction: "BUY" | "SELL";
  entry_price: number;
  target_price: number;
  stop_price: number;
  q_values: Record<string, number>;
  policy_id: string;
  // Which RL agent produced this signal -- "linear_q" (rl_policies_collection) or "ppo"
  // (ppo_policies_collection, see PPOPolicy). Absent on any signal generated before PPO
  // existed; treat a missing value the same as "linear_q" (that's the backend's own default).
  // For a PPO signal, q_values holds the policy's action-probability distribution instead of
  // actual Q-values -- same shape, different number underneath (see PPOPolicy's own comment).
  algo?: "linear_q" | "ppo";
  // The state vector this decision was made from -- see app/services/case_memory.py. Absent
  // on pre-existing signals the same way size_tier etc. can be, below.
  state?: number[];
  // Set when case_memory.memory_gate() downsized or blocked this trade against the raw
  // Q-policy action -- null/absent means memory had nothing to say or agreed with the policy.
  memory_override?: string | null;
  // What the agent chose to risk, and against what balance -- absent on RLSignal documents
  // created before sizing existed (GET /rl/signals returns raw Mongo docs, not validated
  // through the RLSignal Pydantic model, so an old doc really can be missing these keys).
  size_tier?: "SMALL" | "LARGE" | null;
  risk_fraction?: number | null;
  balance_at_signal?: number | null;
  position_size_units?: number | null;
  status: SignalStatus;
  outcome_price?: number | null;
  outcome_timestamp?: string | null;
  outcome_pct_move?: number | null;
  candles_to_outcome?: number | null;
  source: SignalSource;
}

// "Have we seen a state like this before, and how did it turn out" -- a k-nearest-neighbor
// lookup against past resolved RLSignal state vectors (app/services/case_memory.py),
// returned alongside every POST /rl/signal/{interval} response. hit_rate_pct excludes
// "expired" neighbors from its denominator (same reasoning as RLAccuracy.directional_hit_rate_pct)
// -- null when there aren't enough resolved cases yet, not a fabricated number.
export interface RLMemorySummary {
  cases_found: number;
  hit_rate_pct: number | null;
  avg_pct_move: number | null;
  expired_pct: number | null;
  avg_distance: number | null;
  nearest: Array<{
    signal_id?: string;
    status?: SignalStatus;
    outcome_pct_move?: number | null;
    distance: number;
  }>;
  // This policy's own OVERALL resolved-trade record for this pair/interval (both directions,
  // same math as RLAccuracy.directional_hit_rate_pct) -- distinct from hit_rate_pct above,
  // which is scoped to the narrow same-direction nearest-neighbor slice. See
  // case_memory.memory_gate's docstring for why a policy can look weak locally while still
  // being strong overall, and why both numbers matter together.
  policy_hit_rate_pct: number | null;
  policy_decided_trades: number;
}

// Signal's normal fields plus the ML classifier's advisory hit probability -- returned by
// POST /ml/predict/{interval}/{profile}, which does NOT insert into the signals collection.
export type MLPrediction = Signal & { ml_hit_probability: number | null };

export type OptimizeRankBy = "expectancy" | "hit_rate";

export interface OptimizeCandidate {
  config: RuleConfig;
  hit_rate_pct: number | null;
  expectancy_pct: number | null;
  directional_signals: number;
}

export interface OptimizeResult {
  winning_config: RuleConfig;
  rank_by: OptimizeRankBy;
  train: BacktestRun;
  test: BacktestRun;
  candidates_evaluated: OptimizeCandidate[];
}

export interface ConsensusBacktestResult {
  train: BacktestRun;
  test: BacktestRun;
}

export type PaperTradeStatus = "open" | "won" | "lost" | "error";

export interface PaperTrade {
  _id?: string;
  pair: string;
  interval: string;
  profile: Profile;
  direction: "BUY" | "SELL";
  deriv_symbol: string;
  signal_price: number;
  target_price: number;
  stop_price: number;
  stake: number;
  multiplier: number;
  currency: string;
  take_profit_amount: number;
  stop_loss_amount: number;
  contract_id?: number | null;
  entry_spot?: number | null;
  buy_price?: number | null;
  status: PaperTradeStatus;
  pnl?: number | null;
  sell_price?: number | null;
  is_virtual: boolean;
  deriv_loginid: string;
  opened_at: string;
  closed_at?: string | null;
  error?: string | null;
}

export interface PaperTradeAccount {
  loginid: string;
  is_virtual: boolean;
  currency: string;
  balance: string;
}

export interface PaperTradeResult {
  signal: Signal;
  paper_trade: PaperTrade | null;
  note?: string;
}

export interface RLAccuracy {
  pair: string | null;
  interval: string | null;
  // Rolling window, capped at the request's `limit` (default 100) -- plateaus there, not a running total.
  sample_size: number;
  hits: number;
  misses: number;
  expired: number;
  hit_rate_pct: number | null;
  // Real uncapped counts/rate across every resolved live RL signal matching the same filter.
  total_resolved: number;
  total_hits: number;
  total_misses: number;
  total_expired: number;
  total_hit_rate_pct: number | null;
  // Signals the agent superseded (changed its mind) before label_outcome ever judged them --
  // not counted toward hit_rate_pct/total_hit_rate_pct, surfaced separately for transparency.
  total_superseded: number;
  // total_hits / (total_hits + total_misses) -- excludes "expired" (timed out, neither win nor
  // loss) from the denominator entirely, unlike total_hit_rate_pct. On a fast/unconverged
  // interval where most signals expire, total_hit_rate_pct collapses toward 0 for reasons
  // unrelated to directional accuracy; this is the number that actually answers "when the
  // agent commits to a hit-or-miss outcome, how often is it right."
  directional_hit_rate_pct: number | null;
  resolution_breakdown_pct: {
    hit: number | null;
    miss: number | null;
    expired: number | null;
    superseded: number | null;
  };
}

// One day's point on the "is the agent getting smarter" trend -- GET /rl/learning-curve.
// Two independent series: live_directional_hit_rate_pct (real resolved trades, grouped by
// the day they resolved) and avg_training_hit_rate_pct/avg_training_return_pct (that day's
// training runs' own test-slice evaluation, averaged across whichever pair/intervals got
// (re)trained). Either can be null for a day with no data in that series.
export interface RLLearningCurvePoint {
  date: string;
  live_directional_hit_rate_pct: number | null;
  live_decided_trades: number;
  avg_training_hit_rate_pct: number | null;
  avg_training_return_pct: number | null;
  policies_trained: number;
}

// A plain verdict, not a chart to interpret -- compares the earlier half of the window
// against the later half, pooling raw counts within each half (not averaging daily
// percentages, which would let a 1-trade day sway the result as much as an 18-trade day).
// "not_enough_data" when either half falls short of the pooled-sample minimum -- a
// confident-sounding verdict built on a handful of trades would be actively misleading.
export interface RLLearningVerdict {
  status: "improving" | "declining" | "flat" | "not_enough_data";
  first_half_rate_pct: number | null;
  second_half_rate_pct: number | null;
  first_half_n: number;
  second_half_n: number;
}

export interface RLLearningCurve {
  days: number;
  points: RLLearningCurvePoint[];
  verdict: { live: RLLearningVerdict; training: RLLearningVerdict };
}

// GET /rl/insights -- deterministic, explainable "what to improve" findings synthesized from
// data already collected elsewhere (no LLM call). pair/interval are null for a system-wide
// finding (e.g. the overall learning-curve verdict, reused directly from RLLearningCurve).
export interface RLInsightFinding {
  severity: "good" | "warning" | "critical";
  pair: string | null;
  interval: string | null;
  title: string;
  detail: string;
}

export interface RLInsights {
  generated_at: string;
  findings: RLInsightFinding[];
}

export interface SignalAccuracy {
  pair: string | null;
  profile: Profile | null;
  // Rolling window, capped at the request's `limit` (default 100) -- plateaus there, not a running total.
  sample_size: number;
  hits: number;
  misses: number;
  expired: number;
  hit_rate_pct: number | null;
  // Real uncapped counts/rate across every resolved live signal matching the same filter.
  total_resolved: number;
  total_hits: number;
  total_misses: number;
  total_expired: number;
  total_hit_rate_pct: number | null;
  // see RLAccuracy.directional_hit_rate_pct -- same reasoning, excludes "expired" from the denominator.
  directional_hit_rate_pct: number | null;
}

// Results section of a finished RunAllFlowsJob -- the manual "catch up now" job, for when
// the GitHub Actions cron has gone quiet for a while. Loosely typed on purpose: each
// section's per-key value is either "ok", an "error: ..." string, or (for ingest/score/
// ml_train) the underlying endpoint's own result shape passed through unchanged -- this is a
// manual diagnostic/ops action, not something other UI logic needs to depend on the exact
// shape of.
export interface RunAllFlowsResult {
  ingest: Record<string, unknown>;
  // Per pair/interval: "error: ..." on failure, otherwise a small summary object (not "ok"
  // like the other sections -- retraining is worth surfacing hit_rate_pct/total_return_pct
  // inline, not just pass/fail).
  rl_training: Record<string, string | { policy_id: string; hit_rate_pct: number | null; total_return_pct: number | null }>;
  signals: Record<string, string>;
  consensus: Record<string, string>;
  rl_signals: Record<string, string>;
  score: { signals?: unknown; consensus?: unknown; rl?: unknown };
  ml_train: unknown;
  fatal_error?: string;
}

// POST /ops/run-all-flows runs as a background job (the full sequence can take well over a
// minute -- Cloudflare's proxy has a ~100s timeout, the same 524 this project already hit
// with a single /rl/train call) -- poll GET /ops/run-all-flows/{job_id} for status/results,
// same pattern as RLTrainAllJob.
export interface RunAllFlowsJob {
  job_id: string;
  status: "running" | "done" | "cancelled";
  created_at: string;
  finished_at?: string | null;
  results: RunAllFlowsResult | Record<string, never>;
  current_step?: string | null;
  completed_steps: number;
  total_steps: number;
  cancel_requested: boolean;
}

export interface CandlePoint {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  ema_fast: number | null;
  ema_slow: number | null;
  rsi: number | null;
  macd: number | null;
  macd_signal: number | null;
  macd_hist: number | null;
  atr: number | null;
}

export const PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"] as const;
export const INTERVALS = ["5min", "15min", "1h", "4h", "1day"] as const;
export const PROFILES: Profile[] = ["intraday"];
