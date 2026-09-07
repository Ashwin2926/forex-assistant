# Architecture

How the system actually works end to end, and where each piece lives. `README.md` covers
local setup/auth/paper-trading; this document covers the signal-generation/learning pipeline
itself — the part that's grown well past what the README describes. `PROGRESS.md` is the
dated history of *how* it got here; this is the current-state reference.

## What this is

Three independent ways of turning candles into a BUY/SELL/HOLD call, all reading the same 7
technical strategies, all logging their outcomes to the same kind of history, and all feeding
back into each other:

1. **Rule engine** (`signal_engine.py`) — fixed, hand-tuned vote-counting logic. Fully
   deterministic and explainable.
2. **Consensus** (`consensus.py`) — an additive layer on top of the same 7 strategies: fires
   only when a weighted majority independently agree, on price levels close enough together
   to average.
3. **RL agent** (`rl_engine.py`) — a learned linear Q-policy, one per pair/interval, that
   decides direction *and* position size from the same underlying signals, trained by replaying
   history and adjusted by everything else in this document (ATR tuning, ML-informed features,
   case memory).

A fourth piece, the **ML classifier** (`ml_model.py`), doesn't generate its own signals — it's
a single shared model that scores how likely a signal (from any source) is to actually hit,
and its output is now wired into both the rule engine's live gate and RL's own state vector.

## Data flow at a glance

```
Twelve Data ── ingest ──> candles (MongoDB)
                              │
                    add_all_indicators (EMA/RSI/MACD/ATR/Bollinger/Stochastic/ADX/volume)
                              │
                    7 strategies (strategies.py) ──> StrategyCall per strategy
                    ┌─────────┼─────────────────────────┐
                    ▼         ▼                         ▼
              rule engine  consensus                RL agent (state = votes +
              (apply_rules  (weighted                raw indicators + ML scores
               + decide)     majority)                + account balance)
                    │         │                         │
              ML quality  (no ML gate)             live-exclusion gate
              gate (60%+)                          -> memory gate (cross-pair)
                    │                               -> ML quality gate (60%+)
                    ▼         ▼                         ▼
              Signal    ConsensusSignal              RLSignal
                    └─────────┴─────────────────────────┘
                              │
                    outcome_scoring.py: check real candles as they arrive
                    -> status: hit / miss / expired (/ superseded, RL only)
                              │
              ┌───────────────┴───────────────────┐
              ▼                                    ▼
   ML classifier retrains on              RL's next training run
   every resolved rule-based               warm-starts from this
   signal (pooled, all pairs)              policy + a frozen ML
                                            snapshot fit on pre-split
                                            resolved signals
```

The loop closes: live outcomes train the ML classifier, the ML classifier's calibrated
opinion gates *and* informs RL, RL's own outcomes feed case memory and the next day's
warm-started retrain.

## Deployment & orchestration

- **Backend**: FastAPI on **FastAPI Cloud**, auto-deploys on every push to `master` (no
  separate deploy step). `https://forex-assistant.fastapicloud.dev`.
- **Frontend**: Next.js on **Vercel**, same repo, deploys from the same push.
- **Database**: MongoDB (`app/core/database.py` — collections listed below).
- **Data source**: Twelve Data, free tier (800 calls/day). Rate-limited in bursts (~8/min) —
  scripted multi-call ingests should expect occasional 429s and retry with a short backoff,
  not fire all pairs/intervals back-to-back.
- **The actual production heartbeat is `.github/workflows/keep-fresh.yml`**, a GitHub Actions
  cron — *not* an in-process scheduler (an earlier APScheduler-based approach silently died
  whenever the FastAPI Cloud instance scaled to zero between requests). Current schedule:
  - `*/20 * * * *` — ingest + generate + consensus + RL-signal for 5min/15min, every 20 min.
  - `11 0 * * *` — once-daily RL training pass across whichever intervals are active (see
    `RL_INTERVALS` below) plus 1day-cadence steps.
  - Scoring steps (`/signals/score`, `/consensus/score`, `/rl/score`) and ML retrain
    (`/ml/train`) run **unconditionally on every firing**, regardless of which schedule
    triggered it — so already-open trades keep resolving even when their own interval's
    generation step isn't currently active.
  - 1h/4h were previously on their own schedule entries; check the live file for whether
    they're currently active — this project has paused and resumed swing/longer intervals
    more than once (see `PROGRESS.md`), and the schedule entries are the source of truth for
    what's actually running right now, not this document.
- **Auth**: every route except `/` and `/auth/login` requires a bearer token (user JWT) or an
  `X-Service-Token` header matching `AUTH_SECRET_KEY2`/`AUTH_SECRET_KEY3` (service-to-service —
  the cron uses one, direct API/ops access uses the other). See `app/core/auth.py`.

## MongoDB collections

| collection | written by | holds |
|---|---|---|
| `candles` | `data_fetcher.py` | OHLCV bars per pair/interval |
| `signals` | `create_signal` | rule-engine `Signal` docs (live + backtest) |
| `consensus_signals` | `create_consensus_signal` | `ConsensusSignal` docs |
| `rl_signals` | `create_rl_signal` | `RLSignal` docs — live RL decisions + outcomes |
| `rl_policies` | `train_rl_policy` (via `/rl/train`) | one doc per trained policy (weights, feature_names, eval link) |
| `backtest_runs` | `run_backtest`, `train_rl_policy` | one summary doc per backtest/training eval (`profile` field distinguishes rule/consensus/RL) |
| `backtest_signals` | same | per-trade detail for a `backtest_runs` run, linked by `run_id` |
| `ml_runs` | `train_hit_classifier` (via `/ml/train`) | ML classifier training history + calibration table |

## The 7 strategies (`app/services/strategies.py`)

Each strategy is an independent function returning a `StrategyCall` (direction, entry/target/
stop when directional, `reasons`, and a `strength` in roughly `[0, 1]` — how strong *this*
bar's reading is, not just a flat vote). Consensus and RL both consume `strength`; the rule
engine's own vote-counting (`apply_rules`) does too, via `rule_strengths`.

| strategy | reads |
|---|---|
| `trend` (`trend_ema`) | fast/slow EMA cross, spread normalized by ATR% |
| `bollinger` | price vs. Bollinger bands |
| `support_resistance` | swing-level pivots (`patterns.py`'s `find_swing_levels` — unrelated to the removed "swing" trading *profile*, same name, different concept) |
| `candlestick` | candlestick pattern recognition |
| `stoch_adx` | stochastic oscillator + ADX trend strength |
| `volume_momentum` | volume vs. its moving average, rate-of-change |
| `smart_money` | order-block / wick-body style smart-money concepts (fires rarely — see RL's Adagrad note below) |

## Rule engine (`app/services/signal_engine.py`)

- `apply_rules(latest, prev, config)` — the core decision logic, isolated from indicator
  computation so the live engine and the backtester evaluate identical code. Returns
  `(reasons, bullish_votes, bearish_votes, total_rules, rule_votes, rule_strengths)`.
- `decide(...)` — turns votes into `(direction, confidence)`. Confidence is the *sum of
  agreeing rules' strengths* over `total_rules`, not a flat vote count — two signals with the
  same vote count can report different confidence depending on how strong each vote was.
- **`PROFILE_DEFAULTS`** used to have two entries (`"intraday"`, `"swing"`); `"swing"` was
  removed (see `PROGRESS.md`, 2026-09-07) after repeated tuning rounds never got it past
  breakeven. **Every interval, including 1h/4h/1day, now runs the single validated
  `"intraday"` config** (EMA 9/21, session filter 12:00-16:00 UTC, `volatility_threshold_pct
  =0.02`) — `default_config_for(profile, pair)` only knows `"intraday"` now. This means
  1h/4h/1day signals are also forced to HOLD outside the session window, which they never
  were under the old swing config — a deliberate side effect of the unification, not an
  oversight.
- `compute_atr_target_stop(entry_price, atr, direction, target_atr_mult, stop_atr_mult)` —
  shared by the rule engine, backtester, and RL for turning an ATR reading into absolute
  target/stop prices.
- `label_outcome(future_candles, direction, target_price, stop_price, max_lookforward)` — walks
  forward through already-known candles and returns whichever of hit/miss/expired comes
  first. This one function is what both the backtester (historical replay) and
  `outcome_scoring.py` (live resolution) use, so a backtested hit-rate and a live one mean
  the same thing.

## Consensus (`app/services/consensus.py`)

A separate, additive signal source over the same 7 `StrategyCall`s: fires only when a
weighted majority (`STRATEGY_WEIGHTS`, `REQUIRED_WEIGHT_FRACTION`) agree on direction *and*
their individual entry/target/stop levels land within `PROXIMITY_ATR_MULT` of each other
(averaged into one `ConsensusSignal`). No ML gate on this path currently. Most calls are
expected to return "no consensus" — that's normal, not a sign anything's broken.

## ML classifier (`app/services/ml_model.py`, `ml_features.py`)

Not reinforcement learning — a plain `LogisticRegression`, retrained fresh on **every**
currently-resolved rule-based `Signal` (pooled across all 4 pairs and every interval, one
shared model, not one per pair/interval).

- `extract_features(signal_dict)` — maps a stored `Signal` document to a flat feature vector:
  `ema_spread_pct, rsi, macd_hist, atr_pct, session_hour, confidence, profile_intraday,
  direction_buy`, plus one-hot `pair_*`/`interval_*` columns (added specifically so the shared
  model can tell e.g. EUR/USD 5min apart from USD/JPY 1h).
- `fit_hit_classifier(signals)` — fits on everything given, no holdout; factored out
  specifically so a caller (RL, see below) can fit once and score many inputs against it,
  rather than paying the fit cost per prediction.
- `train_hit_classifier(signals, train_frac)` — the honest, chronologically-split version
  (`POST /ml/train`) used to report real out-of-sample `test_accuracy`/calibration, never
  used to make a live decision.
- `predict_hit_probability(resolved_signals, features)` — live-only path: fits fresh on
  everything currently resolved, scores one new signal. Below `MIN_TRAIN_SIGNALS +
  MIN_TEST_SIGNALS` returns `None` (not enough data), never a fabricated number.
- **Calibration is the thing that actually matters here**, more than raw accuracy: `GET
  /ml/runs`'s `test_calibration` buckets predicted probability against actual hit rate. A
  well-behaved model's 70%+ bucket should actually hit around 70% of the time — check this
  before trusting `ml_hit_probability` for anything, a model can have decent accuracy while
  being badly mis-calibrated.

## RL agent (`app/services/rl_engine.py`)

The most complex piece. One independent `LinearQPolicy` per pair/interval (currently 4 pairs
× however many intervals `RL_INTERVALS` covers — check `main.py` for the live value).

**Actions** (`ACTIONS`): `HOLD, BUY_SMALL, BUY_LARGE, SELL_SMALL, SELL_LARGE` — direction and
position size in one learned decision. Two size tiers, not three, deliberately: every extra
action means fewer training samples per action.

**State** (`RL_FEATURE_NAMES`, 19 features per decision):
1. 7 strategy-vote features (`{strategy}_vote`, direction × strength, from the same
   `StrategyCall`s everything else uses)
2. `atr_pct`
3. 8 raw continuous indicator features (`raw_adx_norm`, `raw_rsi_centered`,
   `raw_stoch_spread`, `raw_macd_hist_pct`, `raw_bb_width_pct`, `raw_volume_ratio`,
   `raw_roc_pct`, `raw_stoch_k_norm`) — added because `LinearQPolicy` is strictly linear and
   can't reconstruct information a vote already discarded (e.g. a strategy that only ever
   gates on ADX > 20 throws away *how much* above 20)
4. `ml_hit_probability_buy`, `ml_hit_probability_sell` — the ML classifier's opinion of a
   hypothetical BUY and a hypothetical SELL at this exact bar (two scores, not one, because
   the classifier's own `direction_buy` feature needs a direction and RL hasn't picked one
   yet when this is computed). Computed via `compute_ml_scores`, which calls
   `signal_engine.apply_rules` directly against the already-computed indicator frame — never
   `generate_signal` (that recomputes every indicator from scratch, which would be
   O(n²·episodes) inside the training loop).
5. `balance_log_ratio` — `log(balance / starting_balance)`, the only state feature that isn't
   pure market data; lets the policy condition its sizing on how the account is actually
   doing.

**Reward**: `log(new_balance / balance)` per trade — the Kelly-criterion-standard objective
for compounding growth, which also naturally and severely penalizes ruin without a bolted-on
penalty (`RUIN_REWARD` only guards the literal balance≤0 edge `log` can't represent).

**Training** (`train_rl_policy`): epsilon-greedy Q-learning with Adagrad (per-weight adaptive
learning rate — the 7 strategies fire at very different rates, e.g. `trend` on most bars vs.
`smart_money` on ~4%, so a flat learning rate would let frequent strategies dominate through
repetition alone). Chronological train/test split, single-position-at-a-time (the training
walk only makes its next decision once the current simulated trade resolves — this discipline
is why live inference now also waits for a pending signal to resolve before re-deciding,
see below). Warm-starts from the pair/interval's last persisted policy by default (`reset=
true` for a fresh zero-initialized run).

**Two sweepable/overridable pieces, both exposed as `POST /rl/train/{interval}` query
params for candidate testing without a code change + redeploy:**
- `target_atr_mult`/`stop_atr_mult` — per-interval defaults in `RL_ATR_MULTS_BY_INTERVAL`
  (plus `RL_ATR_MULT_PAIR_OVERRIDES` for the rare pair-specific exception), always keeping
  the enforced ≥1.5:1 target:stop ratio. A per-interval sweep found 4h genuinely benefits
  from a wider band; 5min/15min/1day showed no stable, trustworthy improvement (see next
  point for why "stable" matters).
- `learning_rate`/`epsilon_min` — `LEARNING_RATE`/`EPSILON_MIN` overrides, same
  sweeping-without-a-redeploy pattern.

**`random_seed` (important)**: nothing in this module seeds Python's `random` by default, so
identical training config + identical data can still converge to meaningfully different
policies purely from exploration-path luck — confirmed directly (one sweep reading of +59%
on a config replayed to -27%/-48% under the same inputs). Normal training (cron, Train all)
deliberately stays unseeded — that's what lets warm-started retraining keep discovering
better weights over time. Pass an explicit `random_seed` int only when comparing two runs
against each other and isolating a real effect from luck actually matters; replay a few times
before trusting any single reading, seeded or not.

**Case memory** (`app/services/case_memory.py`): a k-nearest-neighbor lookup — "have we seen
a state like this before, and how did it turn out" — separate from the learned Q-weights,
pooled across **every** pair/interval (not just the current one), the same pooling logic
behind giving each thin policy access to what every other pair/interval has learned.
`memory_gate` can downgrade a trade's size tier or override it to HOLD based on this, layered
on top of the raw Q-policy action, same philosophy as the fixed 1.5:1 target:stop floor.

**Lookahead-bias-safe ML snapshot**: RL's frozen `ml_hit_probability_buy/sell` classifier
(fit inside `train_rl_policy`) only uses rule-based signals resolved *before* that specific
training run's own train/test split boundary — not "everything resolved right now" (which
would leak future information into historical replay the way live `/ml/predict` legitimately
can afford to, since it really is live).

## Live signal generation & the layered safety gates

`create_rl_signal` (`POST /rl/signal/{interval}`) stacks three independent checks on top of
the raw Q-policy action, in order:

1. **Live-exclusion gate** — a policy whose latest training eval shows a clear losing edge
   (`STRONG_LOSS_RETURN_PCT`) doesn't get to trade live at all, even though it keeps training
   normally in the background. Self-correcting: re-checked fresh on every call against the
   *latest* eval, so the next retrain that clears the bar re-enables live signals
   automatically, no manual flip needed.
2. **ML quality gate** (`GOOD_SIGNAL_ML_THRESHOLD = 0.6`) — the chosen direction's
   `ml_hit_probability_{buy,sell}` (already computed as part of the state) must clear 60%,
   backed by the calibration table showing that bucket boundary is roughly where "the model
   says good" and "actually good" start agreeing. Fails **open** (lets the signal through)
   when there isn't enough resolved history for a real number — absence of data isn't
   evidence of a bad trade.
3. **Case-memory gate** — see above.

The same `GOOD_SIGNAL_ML_THRESHOLD` gate also applies to `create_signal` (rule-based path),
computed live via `predict_hit_probability` rather than RL's frozen-snapshot version.
Blocked signals are recorded, not silently dropped: `Signal.ml_override` /
`RLSignal`'s response-level `ml_blocked_reason` / `excluded_reason` / `memory_override`
explain exactly which gate fired and why.

**Single-position-at-a-time live discipline**: `create_rl_signal` will not re-decide while a
pending signal for that pair/interval genuinely hasn't resolved yet — it checks
`resolve_rl_signal_real_outcome` first and returns the still-open signal as-is if nothing's
changed. This mirrors training's own discipline and is what fixed a supersede-churn bug
where re-deciding on every cron tick was retiring the vast majority of signals before they
ever got a real chance to resolve.

## Outcome resolution (`app/services/outcome_scoring.py`)

- `label_outcome` (shared with the backtester) walks forward through real candles as they
  arrive and resolves `pending` → `hit`/`miss`/`expired`.
- `LIVE_MAX_LOOKFORWARD_BY_INTERVAL` gives live resolution a much longer window than
  training's tight `max_lookforward=20` (which exists only for tractability inside a bounded
  training loop) — a live signal costs nothing to just check again next cron cycle.
- **`GET /rl/resolution-stats`** — diagnostic for a high expired rate: compares resolved
  trades' actual `candles_to_outcome` against the window ceiling, to tell whether "too many
  expired" is a window problem (resolved trades cluster near the ceiling → widen the window)
  or a band problem (resolved trades finish fast, expired ones just never reach the band →
  the ATR multiples are the real lever). Read this before changing either number.
- `superseded` (RL only) — a status distinct from `expired`, for when a newer decision
  disagrees with a still-pending one that genuinely hasn't resolved. Excluded from
  hit-rate math the same way `expired` is (neither is a real win/loss).

## Admin / ops endpoints worth knowing about

| endpoint | does |
|---|---|
| `POST /rl/reset?confirm=true` | wipes every RL policy + eval history for a clean restart, **keeps** genuine hit/miss/expired outcome history. `confirm=false` (default) is a dry-run showing counts only. |
| `POST /rl/train-all`, `POST /ops/run-all-flows` | background jobs (poll `GET .../{job_id}`) that retrain every combo `RL_INTERVALS` covers, or run the full ingest→train→signal→score pipeline in one call |
| `GET /rl/insights` | rule-based findings (no LLM) — losing edges, strongest performers, stability warnings, expired-rate warnings, a system-wide learning-curve verdict |
| `GET /rl/accuracy`, `GET /signals/accuracy` | rolling + all-time hit-rate, directional hit-rate (excludes expired from the denominator — the number that actually answers "when it commits, how often is it right") |

## Current state / known open questions

See `PROGRESS.md` for the dated history. As of the most recent entries: the separate
`"swing"` rule profile has been removed (every interval now runs the intraday config), RL was
fully reset and is retraining fresh, and 5min/15min show a high (~80-90%) expired rate on
GBP/USD/USD/JPY/AUD/USD that `GET /rl/resolution-stats` confirms is a band problem (resolved
trades finish in single-digit-to-low-20s candles against a 150-200 candle window, not a
window problem) — an earlier per-interval ATR sweep found no stable improvement for
5min/15min specifically (unlike 4h, which did improve), so this may be a genuine
characteristic of these pairs' price action at that speed rather than a quick parameter fix.
