# Forex Trading Assistant — v1

Rule-based signal engine for forex (starting with EUR/USD, GBP/USD, USD/JPY, AUD/USD).
No execution, no live trading — this generates and logs BUY/SELL/HOLD signals with
full reasoning so you can backtest and improve the logic over time.

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
curl -X POST "http://localhost:8000/signals/1h/swing?pair=EUR%2FUSD"
```
Returns a Signal object with direction (BUY/SELL/HOLD), confidence %, and a
full breakdown of which rules fired and why.

**Step 3 — view signal history:**
```bash
curl http://localhost:8000/signals
```

**Step 4 — backtest before you trust it:**
```bash
curl -X POST "http://localhost:8000/backtest/1h/swing?pair=EUR%2FUSD"
```
Replays the same rule engine bar-by-bar over the candles you already ingested (walk-forward,
no lookahead — each decision only sees data up to that candle) and scores every BUY/SELL
signal against an ATR-based target/stop. Returns a `BacktestRun` summary: hit-rate, average
win/loss size, confidence calibration, and a per-rule breakdown of which rules actually earn
their vote vs. dead weight. Needs 200 + `max_lookforward` candles of history (defaults: 200 + 20 = 220).

Optional query params: `target_atr_mult` (default 1.5), `stop_atr_mult` (default 1.0),
`max_lookforward` (default 20 candles).

```bash
curl -X POST "http://localhost:8000/backtest/1h/swing?pair=EUR%2FUSD&target_atr_mult=2&stop_atr_mult=1&max_lookforward=30"
curl http://localhost:8000/backtest/runs                       # list past runs
curl http://localhost:8000/backtest/runs/{run_id}               # one run's summary
curl http://localhost:8000/backtest/runs/{run_id}/signals       # every labeled signal in that run
```

Every rule threshold (EMA periods, RSI cutoffs, MACD periods, ATR period, volatility
threshold) lives in `RuleConfig` (`app/models/schemas.py`) and can be overridden per
backtest — either a single override in the `/backtest` POST body, or a **parameter sweep**
comparing several configs against the identical historical candles in one call:

```bash
curl -X POST "http://localhost:8000/backtest/sweep/1h/swing?pair=EUR%2FUSD" \
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
curl -X POST "http://localhost:8000/backtest/optimize/1h/swing?pair=USD%2FJPY"
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
`hit_rate_pct` alone wouldn't (this is exactly how swing's original defaults turned out to
be unprofitable — see "Intraday vs swing" below).

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
curl "http://localhost:8000/signals/accuracy?pair=EUR%2FUSD&profile=swing"   # rolling hit-rate, live signals only
```

This is what tells you whether live performance is actually tracking what was
backtested — it often won't match at first, and that gap is itself useful signal, not a
bug to explain away. `/signals/accuracy` excludes backtest-sourced signals and anything
still pending.

**Step 7 — look at it:**
```bash
curl "http://localhost:8000/candles/1h?pair=EUR%2FUSD&profile=swing&limit=250"
```
Stored candles with EMA/RSI/MACD/ATR attached, computed with that profile's `RuleConfig`
(so what you see matches what `generate_signal` actually used). The frontend's **Chart**
page plots this as price + EMA overlay, RSI panel, and MACD panel, with directional signal
markers (▲ BUY / ▼ SELL) placed at the candle each signal fired on.

**Step 8 — paper trade a live signal on a Deriv demo account:**

See [Paper trading (Deriv)](#paper-trading-deriv) below before running this — it executes a
real (demo-account) order, not a simulation.

```bash
curl -X POST "http://localhost:8000/paper-trade/1h/swing?pair=EUR%2FUSD&stake=10&multiplier=100"
curl http://localhost:8000/paper-trade/open       # refreshes + lists open positions
curl http://localhost:8000/paper-trade/history    # closed positions with final P&L
```

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

## Intraday vs swing

`profile` used to be a label only — both profiles ran the identical `RuleConfig`. They now
have separate, independently-validated defaults (`signal_engine.PROFILE_DEFAULTS`), picked
up automatically by `/signals` and `/paper-trade` whenever no explicit config is passed:

- **swing** — EMA 50/200, no session filter (unchanged), but **`target_atr_mult=0.5`/
  `stop_atr_mult=1.25`**, not the original 1.5/1.0. See "Swing's negative-expectancy fix"
  below — this isn't a guess, it's a cross-pair-validated correction to a real bug.
- **intraday** — EMA 9/21, plus a **session filter**: outside 12:00-16:00 UTC (the London/NY
  overlap — the highest-liquidity window for majors; Asian-session hours are usually too
  quiet for intraday setups) the rule engine forces HOLD regardless of what the other rules
  say, plus a tighter `volatility_threshold_pct=0.02`. target/stop left at RuleConfig's own
  1.5/1.0 default — already modestly positive and consistent (train +0.0005%/test +0.0002%
  on EUR/USD, no train→test sign flip), so unlike swing it didn't need retuning.

### Swing's negative-expectancy fix

Backtesting had validated swing's EMA 50/200 on **hit-rate** (29.7% train / 30.7% test,
EUR/USD) and called it done — hit-rate was the only metric that existed at the time. Adding
`expectancy_pct` (see `/backtest/optimize` above) retroactively invalidated that: **every
candidate in the grid, on every pair, had negative expectancy** at the original
`target_atr_mult=1.5`/`stop_atr_mult=1.0`. Breakeven at that ratio needs a 40% hit-rate
(`stop/(target+stop)`); a 29-31% hit-rate "looks" plausible in isolation, but it's well
under that bar, and swing had been treated as validated on hit-rate alone with nothing
checking the actual payout asymmetry until this metric existed.

Root cause: `target_atr_mult`/`stop_atr_mult` weren't part of `RuleConfig` — they were
fixed endpoint params, so `/backtest/optimize` could tune EMA/RSI/session/vol but never the
risk/reward ratio itself, no matter how wrong it was. Moved them into `RuleConfig`
(see `signal_engine.RuleConfig`), then swept target/stop combinations directly:

| Pair | Old (1.5/1.0) | New (0.5/1.25) train | New (0.5/1.25) test |
|---|---|---|---|
| EUR/USD | -0.0028% | +0.0024% (977 signals) | **+0.0051%** (486 signals) |
| GBP/USD | -0.0040% | -0.0001% | +0.0002% (roughly breakeven) |
| USD/JPY | -0.0108% | -0.0072% | -0.0046% (improved, still negative) |
| AUD/USD | -0.0044% | +0.0070% | -0.0032% (train→test sign flip) |

Every pair moved toward or into positive territory — a real, validated improvement over
the original ratio everywhere. But it's not a uniform win: EUR/USD shows a genuine edge,
GBP/USD is roughly breakeven, USD/JPY stays negative on both train and test (a real
pair-specific shortfall, not noise), and AUD/USD's sign flip is the same overfitting
signature as the earlier intraday lesson — don't trust that number. **Per-pair target/stop
tuning for swing is a real, still-open gap**, same as intraday's history — this fix closes
the "every candidate is doomed by the same bad ratio" bug, it doesn't claim swing is
uniformly profitable now.

### Per-pair swing target/stop tuning

The gap above — one global target/stop ratio applied to every pair — is now closed for
three of four pairs. Re-ingested to ~2500-2700 1h candles/pair (Twelve Data's actual
free-tier ceiling for 1h history, not a deliberate choice) and ran `/backtest/optimize`
per pair with a target/stop-only grid (13 candidates spanning 0.25-1.0x target, 1.0-2.0x
stop; EMA held at swing's 50/200), `train_frac=0.7`:

| Pair | Winner | Train | Test | vs. global 0.5/1.25 default (train) |
|---|---|---|---|---|
| EUR/USD | 0.75/1.25 | +0.0025% (2518 signals) | **+0.0037%** (1260 signals) | +0.0006% |
| GBP/USD | 0.75/1.5 | -0.0002% (2564 signals) | -0.0012% (1133 signals) | -0.0046% |
| USD/JPY | 0.5/2.0 | -0.0074% (2566 signals) | -0.0023% (1151 signals) | -0.0123% |
| AUD/USD | *(no override — see below)* | — | — | -0.0047% |

EUR/USD, GBP/USD, and USD/JPY all got a same-sign train→test result strictly better than
the shared default, so `signal_engine.SWING_PAIR_OVERRIDES` now applies each pair's own
winner instead of the one-size-fits-all 0.5/1.25 — `default_config_for(profile, pair)`
looks up the override when `profile="swing"` and the pair has one. GBP/USD and USD/JPY are
**still net-negative** even at their own best ratio; the override just makes them less bad,
it doesn't make them profitable. USD/JPY in particular remains a real pair-specific
shortfall that target/stop tuning alone doesn't fix — it likely needs its own EMA/RSI
tuning too, which this grid didn't search (out of scope for this pass; see "What's next").

AUD/USD deliberately keeps the global default. Its best train candidate (1.0/1.25) scored
-0.0023% on train but **flipped to +0.0021% on test** — the identical train→test
sign-flip signature already seen for AUD/USD's intraday result above. Adopting it would
mean picking a config because it happened to win on one slice of history, exactly the
failure mode this project's own optimize/train-test split exists to catch. Per-pair tuning
for AUD/USD stays an open gap rather than papering over it with an untrustworthy number.

**How that intraday default was actually reached — including a mistake worth keeping.** The
first pass (EUR/USD only, 2000 15min candles) found EMA 12/26 + session filter looking great
(40.0% train / 45.2% test) and it went straight into `PROFILE_DEFAULTS`. Checking the other
three pairs on that same 2000-candle sample overturned it: every other pair showed real
train→test degradation (GBP -8.6pts, USD/JPY -19.2pts, AUD/USD -12.6pts), and AUD/USD's own
winner *rejected* session filtering outright — direct evidence the first result was small-sample
noise (intraday test slices were only 28-404 signals). The instinct at that point was to leave
`PROFILE_DEFAULTS` alone rather than chase a new single-pair "winner" — correct given the
data available, but the data itself was the real problem: intraday only fires a few hours a
day, so 2000 candles isn't enough to tell signal from noise.

Re-ingesting to Twelve Data's free-tier max (**5000** candles/pair) and re-running the same
grid fixed it — all four pairs converged cleanly:

| Pair | Winner | Train | Test | Drop |
|---|---|---|---|---|
| EUR/USD | EMA 9/21 + session, vol 0.02 | 37.9% | 32.4% | -5.5 pts |
| GBP/USD | EMA 9/21 + session, vol 0.05 | 44.6% | 38.6% | -6.0 pts |
| USD/JPY | EMA 9/21 + session, vol 0.02 | 41.9% | 37.7% | -4.2 pts |
| AUD/USD | EMA 9/21 + session, vol 0.02 | 37.4% | 31.3% | -6.1 pts |

Every pair now agrees on EMA 9/21 + session filter, three of four agree on `vol=0.02`, every
test result clearly beats swing's ~30% baseline, and every train→test drop is a believable
4-6 points instead of the earlier 8-19. `PROFILE_DEFAULTS["intraday"]` reflects this result.
The lesson that's worth keeping past this specific number: **when cross-pair validation
disagrees, check whether there's enough data before concluding the effect isn't real** — the
session-filtering hypothesis was right all along; the first cross-check just didn't have
enough signal to see it.

`session_filter_enabled`/`session_start_hour_utc`/`session_end_hour_utc` are just more
`RuleConfig` fields — override them per-request the same way as any other threshold.

## Project structure
```
app/
  core/
    config.py      # env settings (incl. Deriv app_id/token)
    database.py    # MongoDB connection + collections
  models/
    schemas.py      # Candle, Signal, BacktestRun, PaperTrade pydantic models
  services/
    data_fetcher.py    # Twelve Data API -> MongoDB
    indicators.py       # EMA, RSI, MACD, ATR calculations
    signal_engine.py    # rule-based BUY/SELL/HOLD logic + PROFILE_DEFAULTS + label_outcome (shared with backtester)
    backtester.py        # walk-forward replay + ATR-based outcome labeling + metrics
    outcome_scoring.py    # resolves pending LIVE signals against real candles as they arrive
    deriv_client.py      # Deriv WebSocket session + virtual-account safety gate
    paper_trading.py     # Signal -> Deriv Multipliers contract execution + sync
  main.py            # FastAPI app + endpoints (incl. /candles) + APScheduler (auto-ingest + auto-score)
frontend/            # Next.js dashboard (signal feed + live accuracy, chart, backtesting, paper trading)
```

## What's next (not built yet)
- Per-pair swing target/stop tuning is done for EUR/USD, GBP/USD, and USD/JPY (see
  "Per-pair swing target/stop tuning" above) — but GBP/USD and USD/JPY are still
  expectancy-negative even at their own best ratio, and AUD/USD has no override at all
  (its best candidate was overfit-flagged, train→test sign flip). Next step for the
  still-negative pairs is likely EMA/RSI tuning per pair, not just target/stop — this
  pass only searched target/stop and held EMA at 50/200 for all four pairs
- The Backtesting page's "Optimize" UI runs the default grid (now includes both target/stop
  ratios, EMA/RSI variations) — it still has no input for a fully custom `configs` JSON
  body, so an exhaustive search still needs curl (see "Intraday vs swing" above)
- The scheduler (`AUTO_INTERVALS`/`SCHEDULER_INTERVAL_MINUTES` in `main.py`) ingests all 4
  pairs x 2 intervals every 15 minutes by default — ~768 Twelve Data calls/day if run
  continuously, close to the free tier's 800/day ceiling. Widen the interval if you're also
  calling `/ingest` manually elsewhere
- No auth/rate-limiting on any endpoint — fine for local dev, not for exposing this publicly

## Disclaimer
Signals are generated by rule-based logic for informational/educational purposes. Not
financial advice. Paper trading here uses a Deriv demo account and fake money — verify
`is_virtual: true` via `/paper-trade/account` before ever generating an API token from a
real account.
