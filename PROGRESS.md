# Progress log

Running log of infrastructure/backend/frontend work on this project, most recent first.
Ruleset tuning history (backtest sweeps, per-pair overrides) lives in the README and
`signal_engine.py` instead — this file is for deploys, bugs, and ops.

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
