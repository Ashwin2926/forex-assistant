# Signal Stack v2 — change log

Everything done in one AI-assisted session (2026-09-10) to replace three core pieces of
the trading stack, deploy them, and fix what broke afterward. Commits are on `master`,
oldest first. This is a standalone record of that arc — the ongoing terse log for
day-to-day infra/bug work is `PROGRESS.md`; the current-state system reference is
`ARCHITECTURE.md`.

## What changed, at a glance

| Layer | Before | After |
|---|---|---|
| Trading strategies | 7 rule-based strategies (trend, Bollinger, S/R, candlestick, stoch/ADX, volume/momentum, one catch-all "smart_money") | 5 dedicated Smart Money Concepts strategies |
| ML hit-classifier | `LogisticRegression` (scikit-learn) | `XGBClassifier` (XGBoost) |
| RL algorithm | Linear Q-learning (hand-rolled, epsilon-greedy, Adagrad) | PPO (`stable-baselines3`) over a custom `gymnasium.Env` |

Consensus, the ML classifier, and RL all derive their strategy list from one shared
`STRATEGIES` array (`app/services/strategies.py`), so replacing it fanned out
automatically — none of the three had to be updated to know about the new strategy
names or count.

## Commits

### `c4c3311` — Signal Stack v2: SMC strategies, XGBoost classifier, PPO proof-of-concept

**Strategies (`app/services/strategies.py`, full rewrite).** Removed all 7 old
strategies and replaced them with 5 Smart Money Concepts strategies, each expressed in
ATR-relative terms rather than fixed pip counts:

- `call_market_structure` — Break of Structure / Change of Character
- `call_order_blocks` — the last opposing candle before an impulse move
- `call_fair_value_gap` — 3-candle imbalance gaps
- `call_liquidity_sweep` — stop-hunt wicks beyond a recent high/low that reverse
- `call_supply_demand` — origin zones of a strong impulse leg

`consensus.py`'s `STRATEGY_WEIGHTS` and `rl_engine.py`'s `STRATEGY_NAMES` both derive
from `STRATEGIES` by stripping the `call_` prefix, so both picked up the new 5-strategy
set with no changes of their own beyond a comment update.

**ML classifier (`app/services/ml_model.py`, full rewrite).** Swapped
`LogisticRegression` for `XGBClassifier`. `class_weight="balanced"` has no XGBoost
equivalent, so added `_scale_pos_weight()` (negatives/positives ratio, XGBoost's
documented equivalent). Renamed the trained model's `feature_coefficients` output to
`feature_importances` throughout the stack (`schemas.py`, `case_memory.py`, `main.py`)
since XGBoost's `feature_importances_` is unsigned gain-based importance, not a signed
coefficient — the old name implied a direction the new model doesn't have.

**RL / PPO proof-of-concept (`app/services/ppo_engine.py`, new file).** Added
`ForexTradingEnv(gymnasium.Env)` wrapping the existing replay/reward mechanics, plus
`train_and_evaluate_ppo_poc` — a synthetic-data sanity check (does a trained policy
beat a random one?) run before wiring PPO into anything live. At this commit PPO was
proof-of-concept only, not reachable from any endpoint.

**Dependencies (`requirements.txt`):** added `xgboost`, `gymnasium`,
`stable-baselines3`, `torch`. Flagged at the time as a real deploy risk (torch's
plain-PyPI wheel is the much larger CUDA-capable build, not a CPU-only one, and FastAPI
Cloud's build behavior for it was unverified from the sandbox) — resolved later in this
same session once a real deploy succeeded (see "Deploy verification" below).

### `d42dbc9` — Wire PPO into live endpoints, additive alongside linear Q-learning

Added `PPOPolicy` schema + `ppo_policies_collection`, and a second, parallel set of
endpoints (`/rl/train-ppo/{interval}`, `/rl/signal-ppo/{interval}`,
`/rl/ppo/policies`) so PPO could run live side-by-side with the existing linear
Q-learning system without touching it. `RLSignal` gained an `algo: "linear_q" | "ppo"`
field. This was an intentionally reversible, additive step — nothing about the linear
system changed or was removed here.

### `fe9c9f1` — Add PPO Agent frontend page, additive alongside the RL Agent page

Added `/rl-ppo`, a second frontend page mirroring `/rl` but pointed at the new
`-ppo`-suffixed endpoints, plus a "PPO Agent" nav link. Purely additive, matching the
backend commit above.

### `c2b1c80` — Fully retire linear Q-learning: PPO is now the only RL algorithm

Per explicit direction, PPO replaces linear Q-learning outright rather than running
alongside it. This is the commit that undid the "additive" framing of the previous two:

- **`rl_engine.py`**: deleted `LinearQPolicy`, `train_rl_policy`, `choose_action`, and
  the linear-only constants (`LEARNING_RATE`, `EPSILON_*`, `DEFAULT_EPISODES`,
  `ADAGRAD_EPSILON`). Kept every shared replay/state utility PPO depends on
  (`compute_strategy_vote_states`, `compute_ml_scores`, `full_rl_state`,
  `chronological_train_test_split`, `frozen_ml_snapshot`, the ATR-mult tables) — none
  of those were ever linear-specific.
- **`main.py`**: `POST /rl/train`, `GET /rl/policies`, and `POST /rl/signal` are now
  PPO-backed at the *same* paths (previously linear-Q-backed) — no more `-ppo` suffix.
  Removed the now-redundant `/rl/train-ppo`, `/rl/signal-ppo`, `/rl/ppo/policies`
  routes entirely. `POST /rl/train-all`, the degradation-triggered auto-retrain,
  `GET /rl/insights`, and `GET /rl/learning-curve` all updated to match
  (`profile="rl_ppo"`, `ppo_policies_collection`, `total_timesteps` instead of
  `episodes` — PPO has no warm start, every retrain is fresh). `POST /rl/reset` now
  targets the PPO collections; the retired linear ones are left alone as harmless
  orphaned history, not deleted.
- **`schemas.py`**: removed the `RLPolicy` model entirely (weights/episodes/
  `sum_sq_grad`/`warm_started_from` were all linear-specific); `RLTrainAllJob.episodes`
  renamed to `total_timesteps`.
- **Frontend**: deleted the separate `/rl-ppo` page; `/rl` is the only RL page again,
  rewritten against the PPO-backed endpoints. Kept everything algorithm-agnostic
  (accuracy, learning-curve verdicts, insights, train-all, generate-all, recent
  signals, trade-log drill-down). Dropped what had no PPO equivalent: the per-action
  weight table and the per-feature Q-value-contribution breakdown — PPO's action
  probabilities are already real probabilities, no softmax derivation needed.

### `967c0ab` — Rename RL nav label to PPO Agent, fix stale "logistic regression" copy

Two small frontend copy gaps left over from the rewrite above: the side nav still said
"RL Agent" with no mention it's PPO now (→ "PPO Agent"), and the `/ml` page's header
blurb still described the classifier as logistic regression even though `ml_model.py`
had already switched to XGBoost several commits earlier in this same session.

### `51e500d` — Fix `/ml/train`'s skip-shortcut crashing the frontend on a stale run document

**The bug:** most `/ml/train` calls hit a shortcut — "resolved signal count hasn't
changed since the last stored run, return that prior result instead of refitting on
identical data" — because signals take 100min–5hr to resolve while the cron fires every
20min. That shortcut returned the **raw MongoDB document** straight from
`ml_runs_collection`, bypassing the `MLTrainResult` Pydantic model entirely. Any
document stored *before* this session's `feature_coefficients` → `feature_importances`
rename (commit `c4c3311`, above) still had the old key and no `feature_importances` key
at all. The `/ml` page's `Object.entries(trainResult.feature_importances)` then threw
on `undefined`, crashing the page render — surfaced by the user as the ML page failing
to load right after clicking "Train model."

**The fix:** route the skip-path response through `MLTrainResult` like every other
return from this endpoint, migrating the old field name when present and letting the
model's own defaults (`{}`, `[]`, `False`) backfill anything else an older document
predates, instead of leaking whatever shape happened to be in the database.

### `d3a81ad` — Make "Train all" client-driven instead of depending on the backend's `BackgroundTasks` job

**The bug:** the RL page's "Train all" button (`POST /rl/train-all`) schedules a
20-combo training loop via FastAPI's `BackgroundTasks`, running after the HTTP response
that returns its `job_id` has already gone out. It got stuck at `0/20 completed`
indefinitely, even across an explicit cancel + retry — consistent with this project's
own documented precedent (see `keep-fresh.yml`'s history: the old in-process
APScheduler cron died the same way whenever FastAPI Cloud recycled the instance right
after a response was sent). A background coroutine scheduled this way isn't guaranteed
to survive on this platform.

**The fix:** rewrote `handleTrainAll` (frontend) to loop client-side over all 20
pair/interval combos, calling the same single-combo `POST /rl/train/{interval}` that
the plain "Train" button already uses successfully — identical in shape to
`handleGenerateAll`, which already drives "Generate all" reliably the same way, and to
`keep-fresh.yml`'s own cron loop (verified live via GitHub Actions dispatch — see
below). Cancellation is now a client-side flag checked between combos. The backend's
`POST`/`GET /rl/train-all*` endpoints are untouched and still exist, just no longer
depended on by this page. Trade-off, stated honestly in the UI copy: this now requires
keeping the browser tab open for the ~5-6 minute run, rather than the previous
(unreliable) promise that it kept running after the tab closed.

Along the way, also checked and ruled out an apparent CORS error the user hit
mid-session (`No 'Access-Control-Allow-Origin' header`) — the deployed Vercel origin is
already in `allow_origins` and `AuthMiddleware`/`CORSMiddleware` are already ordered
correctly to avoid the common "401 responses lose their CORS headers" gotcha. Chrome
reports *any* fetch that gets zero HTTP response (backend mid-restart from a fresh
push, connection reset, timeout) with that same generic wording, which is what this
almost certainly was, given two backend deploys landed in the minutes around it.

## Deploy verification

Direct HTTP access to the live backend (`forex-assistant.fastapicloud.dev`) and
frontend (`forex-assistant-five.vercel.app`) is blocked by this sandbox's outbound
network policy, so verification went through GitHub Actions instead: `keep-fresh.yml`
(the cron that ingests candles, generates signals, retrains ML, and trains/generates/
scores RL signals) was manually dispatched twice against the live backend.

- **Run `34445008398`** (commit `d42dbc9`, the additive-PPO state): all 23 steps
  passed, including a ~29-minute "Train RL policy (all intervals)" step — first
  live confirmation that the `torch`/`gymnasium`/`stable-baselines3` dependency stack
  (the deploy risk flagged in `c4c3311`) actually builds and runs on FastAPI Cloud.
- **Run `34448785277`** (commit `c2b1c80`, the final PPO-only state): all 23 steps
  passed again, this time with `Retrain ML model` (XGBoost) and the full 20-combo PPO
  training step both green end-to-end (~5.5 minutes for all 20 combos, ~16s/combo
  average) — the reference timing used to recognize the `d3a81ad` bug as "genuinely
  stuck," not just slow.

## Known follow-ups, not yet done

- The ~20 policy documents left in the retired `rl_policies_collection` (linear
  Q-learning) are orphaned but harmless — nothing reads or writes them anymore. Not
  cleaned up; left as history.
- Whether Vercel actually promotes every push to Production automatically, or leaves it
  as an un-promoted Preview (a known quirk on this project per `CLAUDE.md`), was not
  independently re-verified this session — the Vercel MCP connection available here
  only exposed an unrelated project.
