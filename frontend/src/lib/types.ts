// Mirrors app/models/schemas.py — keep in sync with the backend.

export type Direction = "BUY" | "SELL" | "HOLD";
export type Profile = "intraday" | "swing";
export type SignalStatus = "pending" | "hit" | "miss" | "expired";
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
}

export interface StrategyCall {
  strategy: string; // "trend" | "bollinger" | "support_resistance" | "candlestick" | "stoch_adx"
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
}

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

export interface SignalAccuracy {
  pair: string | null;
  profile: Profile | null;
  sample_size: number; // rolling window, capped at the request's `limit` (default 100) -- plateaus there, not a running total
  total_resolved: number; // real uncapped count of every resolved live signal matching the same filter
  hits: number;
  misses: number;
  expired: number;
  hit_rate_pct: number | null;
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
export const PROFILES: Profile[] = ["intraday", "swing"];
