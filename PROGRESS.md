# Progress log

Running log of infrastructure/backend/frontend work on this project, most recent first.
Ruleset tuning history (backtest sweeps, per-pair overrides) lives in the README and
`signal_engine.py` instead — this file is for deploys, bugs, and ops.

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
