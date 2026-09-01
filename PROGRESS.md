# Progress log

Running log of infrastructure/backend/frontend work on this project, most recent first.
Ruleset tuning history (backtest sweeps, per-pair overrides) lives in the README and
`signal_engine.py` instead — this file is for deploys, bugs, and ops.

## 2026-09-01

**Fixed the real driver of the RL supersede/expired mess, then ran a per-interval ATR sweep
that surfaced a training-determinism gap.**

Investigated why `GET /rl/accuracy` showed only ~13% of RL signals ever reaching a clean
hit/miss (57.7% expired, 29.3% superseded):

- **Supersede churn was a live-inference bug, not a policy-quality problem.**
  `create_rl_signal` recomputed `entry_price` from the current live price on every cron call
  and required an exact match against the pending signal to avoid retiring it — since price
  ticks constantly, that match almost never held, so nearly every call superseded the still-
  open signal and inserted a new one, even when direction hadn't changed. This directly
  violated the single-position-at-a-time discipline the policy was actually *trained* under
  (`_take_action_sized` only decides again once a trade resolves). Fixed: the endpoint now
  waits for a pending signal to genuinely resolve (`resolve_rl_signal_real_outcome`) before
  making a new decision. `_supersede_pending_rl_signal` became dead code and was removed.
- Added `GET /rl/resolution-stats` (candles-to-outcome distribution vs. each interval's
  `LIVE_MAX_LOOKFORWARD_BY_INTERVAL` ceiling) to check whether the separate high-expired rate
  was a window problem or an ATR-band problem before touching either.
- **ATR multiplier sweep** (5 intervals x 4 pairs x 5 target:stop grid points, `persist=false`
  so live policies stayed untouched): found the effect is genuinely pair/interval-specific,
  not a uniform "widen everything." 4h showed a real, reproducible improvement for AUD/USD
  and EUR/USD; 1day showed the opposite for EUR/USD and GBP/USD (this project's two strongest
  policies — widening made both worse). `RL_ATR_MULTS_BY_PROFILE` (2 buckets) became
  `RL_ATR_MULTS_BY_INTERVAL` (5 buckets), only 4h's value actually changed (target=3.75,
  stop=2.5).
- **Discovered `LinearQPolicy.epsilon_greedy` has no seeded RNG anywhere in `rl_engine.py`** —
  identical data and identical ATR values can converge to meaningfully different policies
  purely from exploration-path luck. Caught this because GBP/USD 4h's real persisted retrain
  (-39%) contradicted the sweep's single reading for that exact config (+59%). Replaying it
  5x at the new value and 3x at the old value showed both clustering around a similarly bad
  ~-40% — no stable edge either way for that combo. Added `RL_ATR_MULT_PAIR_OVERRIDES` to
  keep GBP/USD 4h at the original 1.5/1.0 rather than forcing it onto a default that didn't
  actually help. Net result: critical "losing edge" findings in `GET /rl/insights` dropped
  from 12 to 9, with none of the 5 already-strong policies (EUR/USD 1day/5min, GBP/USD
  1day/5min, USD/JPY 1h) touched.

## 2026-08-27 (cont.)

**Bug: every live `/rl/signal` call where the policy actually chose to trade was 500ing.**
User reported generating signals and getting "all HOLD and failed all of them." Some were
genuine HOLD decisions, the rest were a crash: `memory_gate` (added earlier today, see
below) got called in `create_rl_signal` but was never added to the `case_memory` import at
the top of `main.py` — only `memory_summary` was. Every non-HOLD decision hit
`NameError: name 'memory_gate' is not defined` before it could return a signal, surfacing
as a bare 500 with no detail (production error handling correctly hid the traceback from
the client, which is why it needed a temporary diagnostic try/except, commit `4034529`, to
actually see the `NameError` before fixing it for real in `7f60171`). Live-verified across
5min/15min/1day x 3 pairs after the fix: all HTTP 200, no more 500s.

## 2026-08-27

**Traced "0.9% accuracy, not learning" to a misleading metric (not a broken model), then
closed a real gap: RL training now warm-starts and carries a case-based memory that gates
live trades and triggers its own retrains on live degradation.**

User reported the RL dashboard's "Overall trading accuracy" showing 0.9% and asked what
`superseded` meant, believing it should count as a real win/loss. Investigated live via the
deployed backend (`X-Service-Token` auth) before changing anything:

- **The 0.9% figure was arithmetically correct but the wrong question.** Summed across all
  5 intervals: `hits=3, misses=4, expired=325, total=332` → 3/332 = 0.9%. But 325 of those
  332 are `expired` (never touched target *or* stop) — a non-event, not a loss. Among the 7
  that actually resolved decisively, real accuracy is 3/7 = 43%. `superseded` (11-16/interval)
  was already correctly excluded from this math — it wasn't the pollutant suspected.
- **Every 5min/15min RL policy showed genuine negative expectancy** in its own backtest eval
  (all 4 pairs, -52% to -87% total_return on $50 start) — confirmed via `GET /rl/policies`,
  not assumed. Root cause traced to `rl_engine.py`: `LinearQPolicy` is strictly linear over
  just 9 features, 7 of which are already-lossy `[0,1]` "strength" scalars each strategy
  computes from much richer raw values (raw RSI, ADX, stochastic, MACD hist, Bollinger width,
  volume ratio) before discarding them.
- **The ML confidence classifier was training on genuinely stale data** — `keep-fresh.yml`
  fires `/ml/train` every 20 min unconditionally, but a signal takes 100min-5hr to resolve;
  3 consecutive runs came back byte-identical (no new resolved rows arrived between them).

Fixes shipped (4 commits, `a9252ed`..`93ba2be`, all pushed to `master`):

1. **Accuracy metric** (`a9252ed`): `/rl/accuracy` and `/signals/accuracy` now also report
   `directional_hit_rate_pct` (hits/(hits+misses), excludes `expired` from the denominator)
   and `resolution_breakdown_pct`. RL page shows the directional rate as the headline now,
   not the old blended number. **Live-verified**: `directional_hit_rate_pct: 42.9` matched
   the by-hand calculation exactly.
2. **ML classifier** (`a9252ed`): added `pair_*`/`interval_*` one-hot features (previously
   one shared model couldn't tell EUR/USD 5min apart from USD/JPY 1h) and
   `class_weight="balanced"`. `/ml/train` now skips refitting (`skipped: true`) when the
   resolved count hasn't changed since the last run; `force=true` bypasses this.
   **Live-verified**: recall improved 0.25→0.375, calibration went from inverted
   (0-40% bucket showing a *higher* actual hit rate than 40-60%) to properly monotonic
   (70-100% bucket correctly highest at 75%).
3. **Richer RL state** (`a9252ed`): `compute_strategy_vote_states` now also feeds 6 raw
   continuous readings (adx, rsi-centered, stoch spread, macd_hist%, bb width%, volume
   ratio) alongside the existing 7 vote features. `RL_TARGET_ATR_MULT`/`RL_STOP_ATR_MULT`
   made profile-aware (`RL_ATR_MULTS_BY_PROFILE`) — structural only, values unchanged, real
   tuning deferred. **Live-verified** via 8 retrains (both fast intervals, all 4 pairs): 6/8
   improved, still negative everywhere but meaningfully less so (5min EUR/USD: -53%→-23%
   return, 46% hit rate vs 25.6% before). Real, partial improvement — not a fix.
4. **Warm-started training** (`da74e6d`): `RLPolicy` now persists `sum_sq_grad` (Adagrad's
   accumulator) and `warm_started_from`. `train_rl_policy` continues from the prior policy's
   weights AND optimizer state instead of zero-initializing every call, with a much lower
   `EPSILON_START_RESUME` (0.2 vs the fresh-run 1.0) — starting a continuation at full random
   exploration would apply real TD updates driven by noise on top of already-converged
   weights, undoing prior learning before epsilon decays back down. Falls back to a fresh run
   automatically if the prior policy's `feature_names` don't match (same guard added to
   `choose_action` for live inference, closing a latent silent-truncation bug this session's
   own feature-schema change would otherwise have hit).
5. **Live scoring window** (`da74e6d`): `/signals/score`, `/consensus/score`, `/rl/score` no
   longer share backtest/training's tight `max_lookforward=20` (a training-tractability
   constraint that doesn't apply live). `LIVE_MAX_LOOKFORWARD_BY_INTERVAL` gives each
   interval a realistic real-world holding window (200 candles at 5min down to 30 at 1day)
   before falling back to `expired`.
6. **Case-based memory** (`e31bc85`, new `app/services/case_memory.py`): every `RLSignal`
   now stores its exact state vector (`state`) and a stable `signal_id`. `memory_summary()`
   does a k-nearest-neighbor lookup over past resolved states (same pair/interval/direction)
   and reports their real hit rate — returned alongside every `POST /rl/signal/{interval}`
   response. New `GET /rl/signals/{signal_id}/explain` finds a resolved signal's nearest
   neighbor with a *disagreeing* outcome and reports the top feature differences.
7. **Closed the loop** (`93ba2be`): `memory_gate()` now *acts* on the memory lookup instead
   of only reporting it — downgrades a chosen `LARGE` size to `SMALL`, or overrides the whole
   trade to `HOLD`, when locally-similar past states have a poor hit rate. Explicitly
   considers the policy's own OVERALL record too (`policy_hit_rate_pct`, always surfaced,
   not just when gating fires) — a policy performing well overall only gets downsized, never
   fully blocked, by one thin/noisy local neighborhood; only a policy with no strong overall
   record backing it up gets blocked outright. Separately, `POST /rl/score` now also checks
   every pair/interval's live hit rate against its policy's own training-time backtest claim,
   and triggers a warm-started retrain in the background the moment live performance falls
   15pts+ behind, instead of waiting for the once-daily scheduled slot.

**Also found, in passing**: `keep-fresh.yml`'s cron is measurably less regular than its
declared `*/20 * * * *` schedule — pulled 100 real run timestamps, median gap 23.6 min but
45/99 gaps exceeded 25 min and one hit 3.5 hours. This is a known GitHub Actions limitation
(scheduled workflows aren't guaranteed to fire on time under load), not a config bug, and
doesn't by itself explain the expired-rate finding above (a late cycle delays resolution,
it doesn't change the hit/miss/expired split, since scoring counts real candles elapsed, not
wall-clock deadlines).

**Not yet independently live-verified**: items 4-7 above (warm-start, live scoring window,
case memory, gating, auto-retrain) — pushed and unit-tested locally (a pinned-down local
venv with pandas/pydantic/scikit-learn/motor), but the live end-to-end check was handed off
to the user to run manually rather than completed in-session. Confirmed this session that
FastAPI Cloud now auto-redeploys on push to `master` (see corrected note in `CLAUDE.md` —
supersedes the 2026-08-18 CLI-only finding).

## 2026-08-24 (cont.)

**RL follow-ups, same day: widened to all intervals, added position sizing, dropped the
Deriv integration entirely.**

- User confirmed no Deriv access in their region at all -- the RL paper-trade endpoint
  (`POST /rl/paper-trade/{interval}`) could never actually execute a trade, so it was
  removed outright (endpoint, frontend button, `paperTradeRL` api wrapper), not left as a
  known-broken feature. The original, pre-existing Paper Trading page/`paper_trading.py`
  system is untouched -- this was scoped to RL's own integration only, confirmed explicitly
  with the user before deleting anything.
- Added a lot-size calculator to the RL page instead (same formula as
  `trading-signals/page.tsx`) -- since execution is manual on whatever broker is actually
  available, every RL signal now carries account-balance/risk-%-driven lot size and $
  risk/reward alongside entry/target/stop.
- Widened RL from 1h-only to all 5 intervals -- `/rl/train`/`/rl/signal` were already
  interval-general (only the cron was 1h-gated), so this was mostly a cron change: training
  stays gated to once daily across every interval (heavy -- many episodes over full candle
  history), generate+score follow each interval's own natural cadence, same split already
  used for ingest/generate/consensus. One independent policy per pair *per interval* now,
  not just per pair. **Unverified risk, flagged not fixed**: 5min/15min candle history is
  much larger than 1h's, so those `/rl/train` calls could take meaningfully longer per call
  (possibly tens of seconds) -- worth confirming this doesn't hit a request timeout once
  live, not assumed safe.
- Fixed a real gap: a pending RL signal previously only resolved via `/rl/score`'s normal
  walk-forward, which can take up to ~20 hours (default `max_lookforward`) before the signal
  is stale. Since RL signals are meant to be traded manually and soon, generating a new
  signal that disagrees with whatever's still pending for that pair/interval now
  immediately marks the old one `expired` (updated in place, never deleted -- full history
  stays visible) instead of leaving it to resolve on its own hours later.
- Pushed as `f0fe5e1` (plus `7c2e7b3` for lot sizing and `b079367` for the stale-expiry fix,
  both same day). **Not yet deployed/verified.**

## 2026-08-24

**RL v1: a linear Q-learning agent that trades the 7 existing strategies itself, rather
than just grading a signal someone else generated.** Third independent signal source
(`RLSignal`, mirrors `ConsensusSignal`'s shape), sitting alongside the rule engine and
consensus — additive, not a replacement for either.

- Also this session, before RL: lowered consensus's `REQUIRED_WEIGHT_FRACTION` 0.8 -> 0.6
  (4-of-5 -> 3-of-5) after confirming it fired too rarely to be useful; a follow-up "run all
  backtests" across all pairs/intervals showed the looser threshold mostly produced small,
  consistent **negative** expectancy at real sample sizes (44-67 test signals) — a genuine
  finding, not noise, and a real caveat on the current threshold. Also added `smart_money`
  as a 7th consensus strategy (a mechanized, narrow liquidity-sweep approximation of ICT/
  Smart Money Concepts — wick-dominant rejection at a swing level, target the opposite
  swing level, stop just beyond the sweep's own extreme) and fixed a real bug where
  `/ml/predict` was returning a fabricated `ml_hit_probability` for HOLD predictions (HOLD
  never appears in the training set at all — there's no trade to hit or miss).
- **State**: not raw indicators — the same 7 strategies' current votes (+1/-1/0 per
  strategy), plus `atr_pct`. This is the literal mechanization of "use the existing
  strategies to generate signals": the agent learns how to weight/combine them, an adaptive
  version of what consensus's fixed threshold already does by hand. Computed once per bar
  before training starts (strategy outputs don't depend on the policy), not recomputed on
  every one of `episodes` passes.
- **Action space is BUY/SELL/HOLD only** — no position sizing in v1. **Risk:reward is
  enforced structurally**: every trade uses a fixed 1.5:1 target:stop ATR ratio
  (`RL_TARGET_ATR_MULT`/`RL_STOP_ATR_MULT` in `rl_engine.py`), deliberately not sourced from
  `RuleConfig`/`SWING_PAIR_OVERRIDES` (some of those, e.g. swing's global 0.5/1.25 default,
  actually risk more than they target) — the user's "never risk more than the gain"
  requirement is guaranteed by construction, not left for reward-driven discovery to
  (maybe) find eventually.
- **One independent policy per pair** (not shared/pooled) — matches "each strategy runs on
  its own" confirmed earlier for the consensus strategies' independence. Scoped to the 1h
  interval only for v1.
- Training's test-slice evaluation is a real `BacktestRun` (`profile="rl"`), not a new
  result shape — directly comparable to every other approach via the existing
  `GET /backtest/runs?pair=X&profile=rl`.
- Cron: training (many epsilon-greedy episodes over full candle history — genuinely heavier
  than the ML classifier's sub-second refit) is gated to once daily; signal generation +
  live scoring run hourly (cheap — a handful of strategy evaluations on the latest candle).
  **Paper-trade execution deliberately stays manual/on-demand, not cron-wired** — same
  "human stays in the loop for anything execution-adjacent" pattern the existing Paper
  Trading page already follows.
- **Known, deliberately deferred dependency**: `POST /rl/paper-trade/{interval}` reuses
  `execute_paper_trade` unchanged, so it inherits the same broken Deriv auth (invalid
  token / wrong auth flow in `deriv_client.py`) that paused paper trading earlier in this
  project. Confirmed with the user this plan does NOT fix that — train/signal/score have
  zero Deriv dependency and should be validated first; the Deriv auth fix is a separate,
  later task.
- New frontend `app/rl/page.tsx`: train (pair/episodes/train_frac -> evaluation card),
  generate signal (q_values + entry/target/stop), paper trade (explicitly labeled with the
  Deriv-dependency caveat), recent signals, training history. Nav link added.
- Pushed as `7339936` (RL) and several commits before it (threshold/smart_money/ML-HOLD-fix).
  **Not yet deployed/verified** — needs a FastAPI Cloud CLI deploy and a Vercel deploy, then
  `POST /rl/train/1h?pair=EUR%2FUSD` + `POST /rl/signal/1h?pair=EUR%2FUSD` curl checks and a
  `workflow_dispatch` to confirm the new cron steps go green.

## 2026-08-22

**ML v1: supervised hit/miss classifier, not reinforcement learning — deliberately
corrected scope after being asked to "confirm we are using reinforced training."** RL
needs an environment/reward-shaping scheme and far more data than the ~238 resolved
signals on hand; a binary classifier on rule-level features is the standard, appropriate
first step at this sample size, and is exactly what `SignalReason.value` +
`outcome_pct_move`/`status` (added the day before) were already shaped for.

- `app/services/ml_features.py`: `extract_features()` maps each signal's `reasons[].rule`
  to a canonical feature via `FEATURE_RULE_MAP` (`ema_spread_pct`, `rsi`, `macd_hist`,
  `atr_pct`, `session_hour`, plus `confidence`/`profile_intraday`/`direction_buy`) —
  the single place encoding lives, so training and prediction can't drift apart.
- `app/services/ml_model.py`: `train_hit_classifier()` fits `scikit-learn`
  `LogisticRegression` (chosen over a tree ensemble/anything deep-learning-shaped —
  far more resistant to overfitting ~166 train rows across 8 features, and its
  coefficients are directly interpretable, matching this project's existing
  "explainable" ethos). Chronological, not random, train/test split — same
  lookahead-bias discipline as the backtester's `eval_start_index`. No model
  persistence in v1: retraining fresh on every call is single-digit milliseconds at
  this row count, so versioning/staleness handling is deferred until it's actually slow.
- New `ml_runs_collection` + `POST /ml/train`, `GET /ml/runs`,
  `POST /ml/predict/{interval}/{profile}` (reuses `generate_signal`, does not insert
  into `signals_collection` — purely advisory, zero effect on the existing signal feed).
- **User explicitly required this run automatically, not just on-demand**: added a
  "Retrain ML model" step to `keep-fresh.yml`, right after "Score pending live signals"
  so each cycle trains on whatever just resolved. Unconditional every firing, like
  scoring — retraining costs no Twelve Data quota.
- New frontend `app/ml/page.tsx`: train button showing train/test accuracy +
  precision/recall side by side plus feature coefficients (which features the model
  actually leaned on); a pair/interval/profile predict form showing the generated
  signal's entry/target/stop alongside `ml_hit_probability`; a training-run history
  table backed by `GET /ml/runs`. Nav link added to `layout.tsx`.
- Backend pushed as `3c68dd2`, frontend as `f66aec2`. **Not yet deployed/verified** —
  needs a FastAPI Cloud CLI deploy (dashboard restart doesn't rebuild from git) and a
  Vercel deploy, then `POST /ml/train` + `POST /ml/predict/...` curl checks and a
  `workflow_dispatch` to confirm the new cron step goes green.
- **Explicitly out of scope for v1**: consensus-signal ML (0 live consensus samples so
  far), model persistence, anything RL.

## 2026-08-21

**Built a multi-strategy consensus system, on top of (not replacing) the intraday/swing
engine — checkpoint: pausing further build here, waiting on live data to accumulate before
trusting or extending it further.**

- Five independent strategies now exist in `app/services/strategies.py`: `call_trend`
  (wraps the existing EMA/RSI/MACD engine unmodified), `call_bollinger` (mean-reversion off
  the bands, target is the middle band itself rather than a generic ATR multiple),
  `call_support_resistance` (breakout/bounce off recent swing highs/lows,
  `app/services/patterns.py`), `call_candlestick` (engulfing/hammer/shooting-star/doji,
  also in `patterns.py`), `call_stoch_adx` (stochastic crossover gated by ADX as a
  trend-strength filter, not a direction source). New indicators (`bollinger_bands`,
  `stochastic`, `adx`) added to `indicators.py`.
- `app/services/consensus.py`'s `check_consensus()` fires only when >= 4 of 5 agree on
  direction AND their entry/target prices land within `PROXIMITY_ATR_MULT` (0.5, an
  unvalidated starting guess, same caveat as `TYPICAL_SPREAD_PRICE`) of each other —
  agreeing on direction alone isn't treated as agreeing on the same trade.
  `run_consensus_backtest` in `backtester.py` replays this bar-by-bar with the same
  `label_outcome`/`spread_cost_pct` real-cost accounting as everything else here.
  `POST /consensus/{interval}`, `GET /consensus`, `POST /consensus/backtest/{interval}`,
  `POST /consensus/score` (outcome resolution, mirrors `/signals/score` —
  `score_pending_consensus_signals` in `outcome_scoring.py`) are the new endpoints.
- Two real bugs caught and fixed before this went live: (1) `POST /consensus/score` was
  declared *after* `POST /consensus/{interval}` — Starlette matches routes in declaration
  order, and `{interval}` is a single dynamic segment that matched the literal path
  `/consensus/score` first (`interval="score"`), so every score call actually hit
  `create_consensus_signal` and 422'd on a missing `pair` param. Fixed by moving `/score`
  above the dynamic route, documented in the docstring so it doesn't regress. (2) Almost
  shipped without any way to *resolve* pending consensus signals at all — caught during the
  cron-wiring step, before anything ran unattended, not after.
- `keep-fresh.yml`: added "Check consensus" + "Score pending consensus signals" steps on
  the same per-interval cadence as the existing generate/score steps — reuses
  already-ingested candles, zero extra Twelve Data calls. Verified end-to-end via manual
  `workflow_dispatch`: all 16 steps green.
- **Validation backtest, 1h, all 4 pairs, train/test split** (via
  `POST /consensus/backtest/1h`): consensus signals are rare by construction (7-13 per
  pair across ~3,500 train candles). EUR/USD (+0.036%/+0.0098% expectancy) and GBP/USD
  (+0.0089%/+0.0477%) came back same-sign positive — the first same-sign positive result
  this project has found, across any methodology, since spread cost modeling was added.
  AUD/USD came back same-sign negative (-0.0306%/-0.0229%), consistent with every other
  methodology tried on this pair — reinforces confidence in the approach rather than
  undermining it. USD/JPY flipped sign (-0.0874% train / +0.021% test) and should be
  discounted per the usual rule. Also notable: Bollinger never once agreed with the
  consensus direction on any pair, train or test — plausibly real, not a bug: mean-reversion
  and the other four (mostly trend/momentum) strategies are philosophically opposed, so a
  strong move they all agree on is often exactly when Bollinger says the opposite.
- New frontend: `app/consensus/page.tsx` rebuilt to check all 4 pairs x 5 intervals (20
  combinations) at once instead of one pair/interval picked at a time, firing-consensus
  rows sorted to the top. New `app/trading-signals/page.tsx` — the page to check for
  actionable trades specifically, filtered to only fired consensus signals, with a
  per-signal lot size computed client-side from a configurable account balance + risk %
  (standard forex position-sizing formula; only correctly handles this project's exact 4
  pairs' quote-currency structure, not a general multi-currency calculator — noted
  explicitly on the page itself, and is a practical stand-in, not a validated-edge claim).
- **Why paused here**: 7-13 signals/pair is nowhere near enough to trust EUR/USD and
  GBP/USD's positive result, or to fully write off USD/JPY's flip as noise. The cron now
  accumulates real outcomes unattended on every cycle (both regular and consensus signals,
  with `SignalReason.value`/`StrategyCall` structured features and `outcome_pct_move`
  already in the exact shape a future ML pass needs) — the useful next step is mostly
  passive: let it run, then revisit the consensus backtest and the ML angle once there's
  an actual sample size, not build further on an 8-signal foundation.
- **Discussed, not built**: using news/economic-calendar data as a signal factor.
  Recommended treating it as a volatility *gate* first (don't trade around high-impact
  releases — cheap to source, easy to backtest as a filter) rather than attempting
  news-sentiment-as-a-directional-signal, which has real look-ahead-bias risk since
  historical news timing/sentiment data is much less available than price data.

## 2026-08-18 (cont. 2)

**Auth re-enabled; cron split by interval speed to fit Twelve Data quota; signals now
carry structured numeric features for a future ML pass.**

- Re-enabled `AuthMiddleware` (`app/main.py`) and the frontend `AuthGuard` redirect,
  commit `aa5252b` — both had been left open since the debugging session above. Cron
  unaffected: it authenticates via `X-Service-Token`, checked before the JWT path in
  `AuthMiddleware`, and `AUTH_SECRET_KEY2` already matched `AUTOMATION_TOKEN`.
- Found a second, separate env var gap right after re-enabling: `AUTH_SECRET_KEY` (the
  JWT *signing* key, not `AUTH_SECRET_KEY2`) was missing from FastAPI Cloud entirely —
  login succeeded (doesn't need it) but every subsequent request 401'd instantly
  (`verify_token` fails closed when unset), so the frontend looked like it was logging in
  then immediately logging back out. Set `AUTH_SECRET_KEY` alongside `AUTH_SECRET_KEY2`,
  fixed.
- Expanded `keep-fresh.yml` to all 5 intervals (5min/15min/1h/4h/1day) and added signal
  *generation* to the cron (previously frontend-only — `api.generateSignal`, only fired
  when someone loaded a page). This uncovered a real quota problem: 5 intervals x 4 pairs
  x 96 runs/day (the original single `*/15` schedule) = 1,920 Twelve Data calls/day, more
  than double the free tier's 800/day cap. Fixed by giving each interval its own cron
  entry at its natural cadence instead of refreshing everything every 15 min — a 4h candle
  only closes every 4 hours, a 1day candle once a day. `github.event.schedule` gates which
  ingest/generate steps run on each firing (`workflow_dispatch` still runs everything, for
  manual testing). Landed on 5min+15min every 20 min (576/day) + 1h hourly (96/day) + 4h
  every 4h (24/day) + 1day daily (4/day) = ~700/day, using the spare headroom on the two
  intervals intraday's signal quality depends on most rather than leaving it unused.
- Added `SignalReason.value: Optional[float]` (commit `367ac99`) — the single most
  decision-relevant number behind each rule's verdict (RSI reading, EMA spread normalized
  by price %, MACD histogram, ATR%, session hour), alongside the existing human-readable
  `detail` string. `detail` was fine for a person reading it but useless as a feature
  without re-parsing. Optional/defaulted so old stored signals still validate; `backtester.py`
  needed no changes since it just passes `apply_rules()`'s reasons through unmodified.
  Frontend: `types.ts` updated for parity, and `page.tsx`'s signal feed now shows each
  rule's value next to its detail text (commit `0763c30`).
- **Why this matters going in to tomorrow**: paired with `outcome_pct_move` (already
  stored per signal), every signal generated from here on is close to a ready-made labeled
  row — rule-level numeric features in, actual outcome out — without having to
  reconstruct features from raw candles after the fact once the ML pass starts.

## Next up: ML angle (starting 2026-08-19)

Rule-based engine stays as the trusted baseline (per README's existing backtest
discipline — train/test split, reject anything that flips sign) — the plan is to treat
ML as a separate, additive angle, not a replacement, at least until it's proven out with
the same rigor. Not yet scoped: model type, how much history to backfill/require before
training is meaningful, whether it predicts direction (classification, matching
hit/miss) or magnitude (regression, matching outcome_pct_move), and how a model's output
would sit alongside the existing rule-vote confidence score rather than silently
replacing it.

## 2026-08-18 (cont.)

**Found the actual reason `keep-fresh.yml` never worked: invalid YAML, not auth.** After
the deploy fix above, the cron was still failing on every run — GitHub's UI showed
`Invalid workflow file ... error in your yaml syntax on line 28`. Every `run:` step with
an inline `-H "X-Service-Token: $AUTOMATION_TOKEN"` was an unquoted plain scalar
containing a bare `: ` (colon-space), which YAML disallows outside quotes/block-scalars —
classic "mapping values are not allowed here" territory once GitHub's parser hits it.
This line was introduced in commit `0696744` back on 2026-08-10, meaning **the cron has
never successfully executed a single step since auth was added** — not 401s as assumed,
every run failed at YAML validation before any code ran at all. Also explains other loose
threads from earlier: the fallback `.github/workflows/keep-fresh.yml` display name instead
of `"Keep candle data fresh"`, `workflow_dispatch` rejecting dispatch with "workflow does
not have that trigger", and the Jobs API returning empty — all downstream of GitHub never
fully parsing the file. The "push"-triggered failing runs we kept seeing weren't a `push:`
trigger (the file only ever declared `schedule`/`workflow_dispatch`) — they were GitHub's
synthetic error-report run, fired on every push to the branch regardless of which files
changed, for as long as the workflow stayed invalid.
- Fix: converted every affected `run:` line to YAML's `|` block-literal style, which isn't
  subject to the same colon restriction (commit `0a237ca`).
- Same commit adds two new steps, `Generate intraday signals (15min)` and
  `Generate swing signals (1h)`, looping `POST /signals/{interval}/{profile}` over all 4
  pairs — previously signal *generation* only happened when the frontend loaded a page and
  called `api.generateSignal(...)`; ingestion and scoring ran unattended but nothing new
  ever got created without a human visiting the site.
- Verified via manual `workflow_dispatch` (run `#27`) — all 7 steps green: ingest, both
  generate steps, score. First confirmed-working end-to-end run since auth was added.

## 2026-08-18

**A week of stale/unresolved pending signals — root cause was FastAPI Cloud dashboard
"redeploy" never actually rebuilding from git, not the auth mismatch it looked like at
first.** `keep-fresh.yml` had been 401ing on every run since 2026-08-11 (see that entry);
turned out the cron's *schedule trigger itself* had gone dormant for the full week too —
GitHub Actions API showed zero runs, scheduled or manual, between 2026-08-11 04:58 and
today. It resumed on its own once new commits landed today.

- Rotated `AUTOMATION_TOKEN` (GitHub secret) / `AUTH_SECRET_KEY2` (FastAPI Cloud env var)
  to a fresh matching value first — didn't fix it, which was the real clue. Confirmed
  byte-for-byte the GitHub secret and FastAPI Cloud value matched, and the env var change
  had triggered a fresh container boot (clean startup log, no crash) — yet requests kept
  401ing regardless. Deployment IDs kept changing every few minutes on their own (`466f5902`
  → `858b6f95` → `f00c234f` → `402f9e9d` → `158ac579` → `25711bd4`, all within ~40 min) with
  **zero build logs on any of them** — every dashboard "redeploy"/restart was just
  restarting the same already-built image, never pulling latest git. That's why nothing
  we changed (env vars, two code pushes) ever took effect no matter how long we waited.
- To isolate the deploy problem from auth itself, temporarily disabled `AuthMiddleware`
  (`app/main.py`, commit `b6d3498`) and the frontend's login-redirect (`AuthGuard.tsx`,
  commit `23f0818`). **Both are currently wide open, no auth at all — re-enable before any
  real or public use.**
- Actual fix: FastAPI Cloud's dashboard restarts don't rebuild from source; only their CLI
  does a real rebuild. Ran the CLI deploy (project directory left blank in its setup
  wizard — no `pyproject.toml` anywhere in this repo; `app/` and `requirements.txt` both
  live at the repo root, not a subdirectory) — first deploy in over a week to actually pick
  up new code.
- Once that landed: `/signals/score` immediately cleared the backlog (2 hit, 1 miss, 0
  left pending), `/ingest` confirmed working again for all 4 pairs on `15min` and `1h`,
  and the `keep-fresh.yml` cron (caught red-handed still firing every 15 min with 401s the
  whole time, in FastAPI Cloud's runtime logs) will succeed going forward.
- Vercel: frontend builds were landing as `Preview` and not auto-promoting to
  `Production` (consistent with the "blocked" deploys noted last entry) — had to manually
  click "Production rebuild" to get `23f0818` live. Worth checking whether
  auto-promote-to-Production is actually enabled for this project, or every future
  frontend change will need the same manual step.
- **Known gap**: `keep-fresh.yml` only calls `/ingest` and `/signals/score` — never
  `POST /signals/{interval}/{profile}`, which is what actually *generates* a new signal.
  Generation still only happens when the frontend loads a page and calls
  `api.generateSignal(...)` client-side. Ingestion/scoring now run fully unattended; new
  signals still need someone to visit the site unless the cron gets a generation step added.

## 2026-08-11

**`keep-fresh.yml` cron fixed — was 401ing on every run since auth landed.** The
GitHub Actions ingestion cron (`.github/workflows/keep-fresh.yml`) had failed on every
single run since 2026-08-10 06:01, right after the previous day's auth rollout — its
plain `curl` calls to `/ingest/*` and `/signals/score` had no bearer token, so
`AuthMiddleware` 401'd them and candle data went stale again for about a day, the exact
failure mode the cron was built to prevent.
- Fix: added a second accepted credential to `AuthMiddleware` (`app/core/auth.py`) — a
  static `AUTOMATION_TOKEN`, sent as `X-Service-Token` instead of a user JWT, checked
  only if the env var is set (fails closed like `auth_secret_key`, same pattern).
  Workflow now sends that header on all three curl calls.
- Generated a token, set it as the `AUTOMATION_TOKEN` GitHub Actions secret on this
  repo.
- Backend setting renamed from `automation_token` to `auth_secret_key2` (env var
  `AUTH_SECRET_KEY2`) to match the variable name the user had already created on
  FastAPI Cloud — unrelated to the JWT signing key `AUTH_SECRET_KEY` despite the name.
  Only the token *value* needs to match between the GitHub secret and the FastAPI Cloud
  var; the names are independent (GitHub secret stays `AUTOMATION_TOKEN`).

## 2026-08-10

**Auth added.** Every endpoint except `/` and `/auth/login` now requires a bearer token
(`app/core/auth.py`: bcrypt password check + JWT, `AuthMiddleware`). Frontend has a
`/login` page, `AuthGuard` redirecting unauthenticated visits, token stored in
`localStorage` and attached to every API call. Closes a real exposure — `/paper-trade`
and `/ingest` were both public and unauthenticated before this.
- Login: username `AshwinMunashe`, password set by the user (not stored here).
- `AUTH_USERNAME` / `AUTH_PASSWORD_HASH` / `AUTH_SECRET_KEY` set on FastAPI Cloud via CLI.
- Backend deploy `aa6cf3af` — confirmed live: root public, protected routes 401 without
  a token, login issues a working token, wrong password rejected.

**Bugs found and fixed (backend, `app/main.py`):**
- `create_signal` and `paper_trade` were sorting candles oldest-first before limiting to
  500, so live signals were generated from ~6-month-old candles instead of the latest
  one (`get_candles` already had the correct sort/limit/reverse). Fixed both.
- `create_signal`'s duplicate-signal check only compared `direction`/`price_at_signal`,
  not `target_price`/`stop_price` — a repeat call with different
  `target_atr_mult`/`stop_atr_mult` overrides would silently return the first call's
  target/stop. Fixed to compare all four fields.
- In-process APScheduler job silently died whenever the FastAPI Cloud instance scaled
  to zero between requests, with no error surfaced — this is why candle data had gone
  stale for two days before it was caught. Removed it; replaced with
  `.github/workflows/keep-fresh.yml`, a GitHub Actions cron hitting `/ingest` +
  `/signals/score` every 15 minutes, independent of whether the backend is awake.

**Data cleanup (production MongoDB):**
- 107 duplicate resolved live signals removed (`scripts/dedupe_signals.py`) — caused by
  the stale-candle bug above: identical signal regenerated on every dashboard load,
  inflating `/signals/accuracy`'s sample size 20-30x per pair.
- 359 orphaned pending signals removed (`scripts/cleanup_orphaned_signals.py`) — live
  signals on intervals (5min/10min/4h/1day) the ingestion cron never covers, left over
  from earlier manual `curl` testing. They can never resolve since no candles for those
  intervals ever arrive.

**Swing tuning, round 4** (MACD periods + `volatility_threshold_pct`, see
`signal_engine.SWING_PAIR_OVERRIDES` comment and README "Per-pair swing target/stop
tuning" for full numbers):
- MACD: no pair got an override — every candidate was either negligible or flipped sign
  train→test.
- `volatility_threshold_pct`: EUR/USD (0.05) and GBP/USD (0.03) got real, same-sign
  overrides. USD/JPY and AUD/USD's candidates flipped sign, rejected.
- AUD/USD has now flipped sign on every candidate across all four tuning rounds
  (target/stop, EMA/RSI, MACD, volatility) — accepted as real evidence swing has no edge
  for this pair in this rule set; search closed rather than left open-ended.

**GitHub account fix.** All commits before ~09:30 were authored as
`AshwinMunashe <ashwin.m@tareeqk.ae>` even after switching `gh` push credentials to
`Ashwin2926` — the local git identity was never updated. Fixed with a repo-local
`git config user.name`/`user.email` (not global, not rewriting past commits). Logged
`gh` out of `AshwinMunashe` entirely; only `Ashwin2926` remains authenticated.

**Vercel deploy is currently blocked — unresolved.** Every deployment since roughly
09:19 (both GitHub-push-triggered and a direct `vercel --prod` CLI deploy) shows
`Blocked`/`UNKNOWN` status with a `0ms` build — it never starts, and a CLI-triggered one
eventually errored with `"Deployment not found"` after ~10 minutes. This affects the
*direct CLI deploy* too, not just git-triggered ones, which rules out a git-identity
mismatch as the sole cause. Leading theory: Vercel's Hobby-tier abuse/fraud check paused
deployments after several rapid deploys in under an hour. **Next step: user needs to open
a blocked deployment in the Vercel dashboard and report the actual block reason/banner
shown there** — not visible via CLI or API. Frontend code (login page, auth guard) is
committed and pushed (`4820aa0`) but not yet confirmed live on
`forex-assistant-five.vercel.app`.

**Deriv paper trading — separately broken, not yet fixed.** The `DERIV_API_TOKEN`
currently set in FastAPI Cloud (`...4bef`) is rejected by Deriv itself
(`"The token is invalid"`). The stored value's format (`pat_` prefix, 68 chars) doesn't
match Deriv's actual token format (~15-25 plain alphanumeric chars, no prefix) —
suspected stale/corrupted value. User needs to delete the existing token in the Deriv
dashboard, generate a fresh one while the **demo** account is active in the account
switcher, and provide the full string immediately (Deriv only shows it once).

**User also asked about swapping paper trading from Deriv to XM Global** — XM has no
public trading API (MetaTrader 4/5 only), so this would mean building an MT4/5
automation bridge (EA script or a paid service like MetaApi.cloud), not a drop-in swap
for `paper_trading.py`. Not started; user was going to try resolving the Deriv token
issue first.
