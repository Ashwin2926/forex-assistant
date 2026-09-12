# Forex Trading Assistant — v1

Signal engine for forex (EUR/USD, GBP/USD, USD/JPY, AUD/USD) with three independent
decision-makers reading the same 7 technical strategies — a rule-based engine, a weighted
consensus layer, and a learned RL agent — plus a supervised classifier that scores how likely
any of them is to actually hit. No execution, no live trading (aside from Deriv demo-account
paper trading, see below) — this generates and logs signals with full reasoning so you can
backtest and improve the logic over time.

**This README covers local setup, auth, and the rule engine's own workflow. For how the
full system (consensus, ML, RL, live quality gates) actually works end to end, see
[`ARCHITECTURE.md`](./ARCHITECTURE.md).**

## Setup (VS Code, local)

### 1. Prerequisites
- Python 3.11+ installed
- MongoDB running locally (or a free MongoDB Atlas cluster)
  - Local: `brew install mongodb-community` (Mac) or install via your package manager, then `mongod` to start it
  - Or Atlas: https://www.mongodb.com/cloud/atlas/register (free tier is enough for now)
- A free Twelve Data API key: https://twelvedata.com/pricing (free tier = 800 requests/day, plenty for dev)

### 2. Open in VS Code
```bash
cd forex-assistant
code .
```

### 3. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
```
Then edit `.env` and fill in:
- `TWELVE_DATA_API_KEY` — your key from step 1
- `MONGODB_URI` — `mongodb://localhost:27017` for local, or your Atlas connection string
- `AUTH_USERNAME`, `AUTH_PASSWORD_HASH`, `AUTH_SECRET_KEY` — see "Auth" below. Every
  endpoint except `/` and `/auth/login` requires a valid token; leaving these unset
  fails closed (nothing can log in) rather than leaving the API open.

### 5. Run the server
```bash
uvicorn app.main:app --reload
```
Server runs at `http://localhost:8000`. Interactive API docs at `http://localhost:8000/docs`.

## Using it

**Step 1 — pull historical data** (need 200+ candles for EMA200 to be meaningful):
```bash
curl -X POST http://localhost:8000/ingest/1h
```
This fetches and stores candles for all 4 pairs at the given interval.

> **Note on `pair`:** every endpoint below takes `pair` as a **query parameter**
> (`?pair=EUR/USD`), never in the URL path. Starlette's routing breaks on a literal `/`
> inside a path segment even when percent-encoded (`EUR%2FUSD` still 404s) — so `pair`
> lives in the query string on every endpoint that needs one.

**Step 2 — generate a signal:**
```bash
curl -X POST "http://localhost:8000/signals/1h/intraday?pair=EUR%2FUSD"
```
Returns a Signal object with direction (BUY/SELL/HOLD), confidence %, and a
full breakdown of which rules fired and why.

**Step 3 — view signal history:**
```bash
curl http://localhost:8000/signals
```

**Step 4 — backtest before you trust it:**
```bash
curl -X POST "http://localhost:8000/backtest/1h/intraday?pair=EUR%2FUSD"
```
Replays the same rule engine bar-by-bar over the candles you already ingested (walk-forward,
no lookahead — each decision only sees data up to that candle) and scores every BUY/SELL
signal against an ATR-based target/stop. Returns a `BacktestRun` summary: hit-rate, average
win/loss size, confidence calibration, and a per-rule breakdown of which rules actually earn
their vote vs. dead weight. Needs 200 + `max_lookforward` candles of history (defaults: 200 + 20 = 220).

Optional query params: `target_atr_mult` (default 1.5), `stop_atr_mult` (default 1.0),
`max_lookforward` (default 20 candles).

```bash
curl -X POST "http://localhost:8000/backtest/1h/intraday?pair=EUR%2FUSD&target_atr_mult=2&stop_atr_mult=1&max_lookforward=30"
curl http://localhost:8000/backtest/runs                       # list past runs
curl http://localhost:8000/backtest/runs/{run_id}               # one run's summary
curl http://localhost:8000/backtest/runs/{run_id}/signals       # every labeled signal in that run
```

Every rule threshold (EMA periods, RSI cutoffs, MACD periods, ATR period, volatility
threshold) lives in `RuleConfig` (`app/models/schemas.py`) and can be overridden per
backtest — either a single override in the `/backtest` POST body, or a **parameter sweep**
comparing several configs against the identical historical candles in one call:

```bash
curl -X POST "http://localhost:8000/backtest/sweep/1h/intraday?pair=EUR%2FUSD" \
  -H "Content-Type: application/json" \
  -d '[{"rsi_oversold": 25, "rsi_overbought": 75}, {"rsi_oversold": 35, "rsi_overbought": 65}]'
```
Returns each config's `BacktestRun` summary sorted by hit-rate.

**Step 5 — actually validate an improvement, not just eyeball a sweep:**

A config that wins a sweep might just be curve-fit to that one window of history.
`/backtest/optimize` grid-searches `RuleConfig` on the first `train_frac` (default 70%) of
your ingested history, picks the best hit-rate there, then re-scores that exact same config
on the untouched remaining tail — the two numbers side by side tell you whether the
"improvement" is real or noise. Needs enough history that both the train slice (covers
warmup for the largest config) and the test slice (covers `max_lookforward`) are meaningful —
a few hundred candles isn't enough; a couple thousand is more like it (`/ingest` accepts
`?output_size=` up to 5000 on Twelve Data's free tier).

```bash
curl -X POST "http://localhost:8000/backtest/optimize/1h/intraday?pair=USD%2FJPY"
```
Omit the body to search the built-in default grid (varies EMA responsiveness and RSI
sensitivity), or POST a JSON array of `RuleConfig` overrides to search your own. Returns
`winning_config`, `train`, `test` (both full `BacktestRun` summaries), and
`candidates_evaluated`.

The winner is picked by `expectancy_pct` by default, not raw hit-rate — pass
`?rank_by=hit_rate` to go back to the old behavior. `expectancy_pct` is the mean
`pct_move` across *every* directional signal (hit, miss, **and** expired), unlike
`avg_win_pct`/`avg_loss_pct` which only average within their own bucket — it's the actual
answer to "is this profitable on average," and it can disagree with hit-rate: a config
that's right less often can still have better expected value if its wins are big enough
relative to its losses. Breakeven hit-rate is `stop_atr_mult/(target_atr_mult+stop_atr_mult)`
— a 35% hit-rate config with a wide target and tight stop isn't "almost good," it can be
losing money on average, and `expectancy_pct` being negative says so directly where
`hit_rate_pct` alone wouldn't (this is exactly how the now-removed swing profile's original
defaults turned out to be unprofitable — see "The intraday ruleset" below).

`target_atr_mult`/`stop_atr_mult` are `RuleConfig` fields, not fixed endpoint params —
specifically so `/backtest/optimize` can search them alongside everything else instead of
holding risk/reward fixed while only tuning entry logic.

**Step 6 — let live signals resolve, then check accuracy:**

Every directional signal from `/signals` gets an ATR-based `target_price`/`stop_price`
attached at creation time (the same formula the backtester uses) and starts out
`status: "pending"`. A background job (APScheduler, every 15 minutes) re-ingests fresh
candles and checks every pending signal against them — once enough real candles have
arrived, `status` resolves to `hit`/`miss`/`expired`, exactly like a backtest, just against
real time instead of historical replay. Trigger it manually instead of waiting:

```bash
curl -X POST http://localhost:8000/signals/score
curl "http://localhost:8000/signals/accuracy?pair=EUR%2FUSD&profile=intraday"   # rolling hit-rate, live signals only
```

This is what tells you whether live performance is actually tracking what was
backtested — it often won't match at first, and that gap is itself useful signal, not a
bug to explain away. `/signals/accuracy` excludes backtest-sourced signals and anything
still pending.

**Step 7 — look at it:**
```bash
curl "http://localhost:8000/candles/1h?pair=EUR%2FUSD&profile=intraday&limit=250"
```
Stored candles with EMA/RSI/MACD/ATR attached, computed with that profile's `RuleConfig`
(so what you see matches what `generate_signal` actually used). The frontend's **Chart**
page plots this as price + EMA overlay, RSI panel, and MACD panel, with directional signal
markers (▲ BUY / ▼ SELL) placed at the candle each signal fired on.

**Step 8 — paper trade a live signal on a Deriv demo account:**

See [Paper trading (Deriv)](#paper-trading-deriv) below before running this — it executes a
real (demo-account) order, not a simulation.

```bash
curl -X POST "http://localhost:8000/paper-trade/1h/intraday?pair=EUR%2FUSD&stake=10&multiplier=100"
curl http://localhost:8000/paper-trade/open       # refreshes + lists open positions
curl http://localhost:8000/paper-trade/history    # closed positions with final P&L
```

## Auth

Every endpoint except `GET /` and `POST /auth/login` requires a bearer token — this is a
single-user tool with real (demo-account) trade execution behind `/paper-trade` and a
limited Twelve Data quota behind `/ingest`, both reachable by anyone who finds the URL if
left open. `app/core/auth.py`'s `AuthMiddleware` checks every request's `Authorization`
header against a JWT signed with `AUTH_SECRET_KEY`; a missing or invalid token gets a 401
with CORS headers still attached (so the frontend can actually read the 401 and redirect
to `/login`, instead of the browser reporting a generic CORS failure — `AuthMiddleware`
has to be added before `CORSMiddleware` in `main.py` for that to work, since Starlette
makes the *last*-added middleware the outermost one).

```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "...", "password": "..."}'
# -> {"access_token": "...", "token_type": "bearer"}
curl http://localhost:8000/signals -H "Authorization: Bearer <access_token>"
```

Setting credentials: `AUTH_PASSWORD_HASH` is a bcrypt hash, never the plaintext —
generate one with:
```bash
python -c "from app.core.auth import hash_password; print(hash_password('your-password'))"
```
Put the result in `AUTH_PASSWORD_HASH`, your chosen username in `AUTH_USERNAME`, and a
random 32+ byte string in `AUTH_SECRET_KEY` (`python -c "import secrets; print(secrets.token_hex(32))"`).
Tokens are valid for 7 days (`app.core.auth.TOKEN_TTL_SECONDS`); logging in again issues a
fresh one.

The frontend's login page (`/login`) stores the token in `localStorage` and attaches it to
every API call (`frontend/src/lib/api.ts`); `AuthGuard` (`frontend/src/components/`)
redirects unauthenticated visits to `/login`, and a 401 from any API call clears the stored
token and redirects there too, so an expired token doesn't just fail silently.

## Paper trading (Deriv)

Phase 6 of the roadmap: once a ruleset has a backtested track record, `/paper-trade`
generates a fresh live signal and executes it as a **Deriv Multipliers** contract on your
Deriv **demo** account — real order flow, fake money, so you can see how the ruleset
performs against live execution before ever risking capital.

### Why Multipliers, not a standard forex position
Deriv's WebSocket API can trade three contract types: digital Options, Multipliers, and
CFDs. Only **Options and Multipliers are reachable through the API directly** — CFDs (the
"buy 1 lot EUR/USD, close whenever" mechanics most forex trading means) require Deriv MT5,
a separate integration this project doesn't do. Multipliers are the closest API-native fit:
a leveraged derivative with stop-loss/take-profit, but P&L is `stake * multiplier *
price_move_pct`, not a literal per-unit fill. `paper_trading.py` converts each signal's
ATR-based target/stop (absolute price levels) into the take_profit/stop_loss *amounts*
Multipliers expect — treat the resulting P&L as directionally correct, not exact (it
doesn't model Deriv's spread or commission).

### Setup
1. Create a free Deriv account, then open a **demo/virtual** account inside it (this is
   separate from your real-money account, even if you never fund one).
2. Register an app at [api.deriv.com](https://api.deriv.com) for your own `DERIV_APP_ID`,
   or leave the default — `1089` is Deriv's public app ID for testing.
3. Generate an API token from **Account Settings → API token**, while your **demo** account
   is the active account. Put it in `.env` as `DERIV_API_TOKEN`.
4. Sanity-check the wiring without placing any trade: `curl http://localhost:8000/paper-trade/account`
   — confirm `is_virtual: true` and the `loginid` is the demo account you expect.

### The safety gate
`deriv_client.deriv_session()` authorizes, checks the account's `is_virtual` flag, and
**raises before yielding a session at all** if it's not a demo account — every paper-trade
call goes through this, and it is not configurable or overridable. Still: only ever generate
`DERIV_API_TOKEN` from a demo account in the first place. Don't rely on the code as the only
line of defense.

## The intraday ruleset

There used to be two profiles here, `intraday` and `swing` — a shorter-horizon rule set
(EMA 9/21 + session filter) and a longer-horizon one (EMA 50/200, no session filter). Swing
was removed: it never got past being roughly breakeven at best on two of four pairs and
net-negative on the others even after several rounds of per-pair tuning (that history is
still in git if it's ever worth revisiting). This project now runs a single validated
ruleset (`signal_engine.PROFILE_DEFAULTS["intraday"]`) across every interval, including the
longer ones (1h/4h/1day) that used to be routed to swing.

**intraday** — EMA 9/21, plus a **session filter**: outside 12:00-16:00 UTC (the London/NY
overlap — the highest-liquidity window for majors; Asian-session hours are usually too
quiet for these setups) the rule engine forces HOLD regardless of what the other rules say,
plus `volatility_threshold_pct=0.02`. target/stop are left at `RuleConfig`'s own 1.5/1.0
default — modestly positive and consistent (train +0.0005%/test +0.0002% on EUR/USD, no
train→test sign flip).

This was cross-pair validated on 15min data, 5000 candles/pair (all four majors):

| Pair | Train | Test | Drop |
|---|---|---|---|
| EUR/USD | 37.9% | 32.4% | -5.5 pts |
| GBP/USD | 44.6% | 38.6% | -6.0 pts |
| USD/JPY | 41.9% | 37.7% | -4.2 pts |
| AUD/USD | 37.4% | 31.3% | -6.1 pts |

Every pair agrees on EMA 9/21 + session filter, and every train→test drop is a believable
4-6 points. An earlier pass on only 2000 candles/pair had picked EMA 12/26 based on EUR/USD
alone and didn't hold up cross-pair — the lesson worth keeping: when cross-pair validation
disagrees, check whether there's enough data before concluding an effect isn't real.

`session_filter_enabled`/`session_start_hour_utc`/`session_end_hour_utc` are just more
`RuleConfig` fields — override them per-request the same way as any other threshold.

## Historical candle archive

Deep candle history (back to 2020 intraday, 2007-2008 daily) lives in
`data/candles/*.parquet` — 20 files (4 pairs × 5 intervals), brotli-compressed,
**committed to git**, not just in MongoDB. This exists because `candles_collection` is
deliberately kept trimmed to a rolling recent window (Atlas's free M0 tier has a hard
512MB cap), so anything that needs full history — `/backtest`, `/backtest/sweep`,
`/backtest/optimize`, `/consensus/backtest`, RL training — reads through
`app/services/candle_archive.py`'s `load_full_candle_history()`, which merges the
archive file with whatever Mongo currently holds (Mongo wins on overlap, since it's the
freshest). `load_archived_candles()` reads just the archive file directly when you don't
need Mongo's live tail.

Parquet was chosen over the original gzip-CSV after measuring both on this project's own
EUR/USD 5min file (502k rows): brotli-parquet is ~27-29% *larger* on disk (~5.2MB vs
4.1MB) but reads ~15-20x *faster* in pandas (0.024s vs 0.50s) — worth it since slow
archive reads had already caused live gateway-timeout incidents on FastAPI Cloud.

**Keeping the archive in sync:**
- `scripts/export_candles.py` (run via `.github/workflows/export-candles.yml`) reads
  each existing archive file, merges in whatever `candles_collection` currently has,
  dedupes on `timestamp`, and writes the union back — coverage only ever grows, even if
  Mongo has since been trimmed. **Never rewrite this to overwrite instead of merge** — a
  version that did lost 16 of the 20 files' full history in one run before being caught
  and restored from git history (see `PROGRESS.md`, 2026-09-12).
- `scripts/backfill_to_archive.py` (run via `.github/workflows/backfill-archive.yml`,
  `workflow_dispatch` with `interval`/`start_date`/`max_calls_per_pair` inputs) extends
  the archive's depth directly from Twelve Data, bypassing Mongo entirely — one
  continuous process per job (no per-request gateway timeout to fight), committing
  progress every 10 calls so a timeout or cancellation doesn't lose a whole pair's work.
  This is the pipeline to use for pushing history further back; it needs a
  `TWELVE_DATA_API_KEY` GitHub Actions secret to run.
- `.github/workflows/backfill-history.yml` (Mongo-routed, older) still exists and still
  backs live ingestion / the dashboard's "Sync now" button — a separate, shallower
  concern from the archive above.

## Project structure

This has grown well past the rule-engine-only description above — see **`ARCHITECTURE.md`**
for the full current system (consensus, the ML classifier, the RL agent, the live quality
gates, and how they all feed back into each other). Structure:
```
app/
  core/
    config.py      # env settings (incl. Deriv app_id/token, auth credentials)
    auth.py        # login/JWT verification + AuthMiddleware guarding every non-public route
    database.py    # MongoDB connection + collections
  models/
    schemas.py      # Candle, Signal, ConsensusSignal, RLPolicy, RLSignal, BacktestRun, PaperTrade, LoginRequest pydantic models
  services/
    data_fetcher.py     # Twelve Data API -> MongoDB
    indicators.py       # EMA, RSI, MACD, ATR, Bollinger, Stochastic, ADX, volume calculations
    strategies.py        # the 7 independent StrategyCall-producing strategies shared by every signal source
    signal_engine.py    # rule-based BUY/SELL/HOLD logic + PROFILE_DEFAULTS + label_outcome (shared with backtester)
    consensus.py         # weighted-majority additive signal source over the same 7 strategies
    backtester.py        # walk-forward replay + ATR-based outcome labeling + metrics
    outcome_scoring.py    # resolves pending LIVE signals (rule/consensus/RL) against real candles as they arrive
    ml_features.py        # Signal -> flat feature vector, shared by training and live prediction
    ml_model.py            # supervised hit/miss LogisticRegression classifier (NOT reinforcement learning)
    rl_engine.py           # the RL agent: state/actions/reward, training loop, ATR/hyperparameter tuning
    case_memory.py         # RL's k-nearest-neighbor "have we seen this before" cross-pair lookup
    deriv_client.py      # Deriv WebSocket session + virtual-account safety gate
    paper_trading.py     # Signal -> Deriv Multipliers contract execution + sync
  main.py            # FastAPI app + every endpoint
data/candles/*.parquet  # git-committed deep candle archive, see "Historical candle archive" above
scripts/              # one-off maintenance scripts (dedupe/cleanup live signals, candle archive
                       # export/backfill — see "Historical candle archive" above) — run manually
                       # or via a workflow, not on any in-process schedule
.github/workflows/
  keep-fresh.yml       # GitHub Actions cron -- the actual production heartbeat (ingest/generate/
                       # consensus/RL/score/ML-retrain); see ARCHITECTURE.md for the current schedule.
                       # NOT an in-process scheduler -- an earlier APScheduler approach silently died
                       # whenever the FastAPI Cloud instance scaled to zero between requests.
  export-candles.yml   # merges candles_collection into data/candles/*.parquet, run manually
  backfill-archive.yml # extends archive depth directly from Twelve Data (Mongo-free), workflow_dispatch
frontend/            # Next.js dashboard (signal feed + live accuracy, chart, backtesting, RL page, ML page, paper trading)
```

## What's next (not built yet)

- Now that there's a single ruleset applied to every interval, the longer intervals
  (1h/4h/1day) haven't had their own expectancy validation the way 15min intraday data
  has (see "The intraday ruleset" above, which is hit-rate/cross-pair validated on 15min
  candles specifically) — worth an `/backtest/optimize` pass per interval to confirm the
  same config holds up rather than assuming it transfers.
- **Revisit ML** — deliberately deferred until the rule-based search space was reasonably
  explored (the rule engine's failure modes — small samples, train/test sign flips — are
  easy to see and reason about; a model's failure modes usually aren't).
- The Backtesting page's "Optimize" UI runs the default grid (now includes both target/stop
  ratios, EMA/RSI variations) — it still has no input for a fully custom `configs` JSON
  body, so an exhaustive search still needs curl (see "The intraday ruleset" above)
- Ingestion runs via `.github/workflows/keep-fresh.yml`, a GitHub Actions cron every 15
  minutes hitting `/ingest/15min`, `/ingest/1h`, and `/signals/score` for all 4 pairs —
  ~768 Twelve Data calls/day if run continuously, close to the free tier's 800/day
  ceiling. Widen the cron interval if you're also calling `/ingest` manually elsewhere.
  (This used to be an in-process APScheduler job in `main.py`; moved out because it
  silently stopped running whenever the FastAPI Cloud instance scaled to zero between
  requests — the scheduler died with the process and never resumed on its own.)
- No auth/rate-limiting on any endpoint — fine for local dev, not for exposing this publicly

## Disclaimer
Signals are generated by rule-based logic for informational/educational purposes. Not
financial advice. Paper trading here uses a Deriv demo account and fake money — verify
`is_virtual: true` via `/paper-trade/account` before ever generating an API token from a
real account.
