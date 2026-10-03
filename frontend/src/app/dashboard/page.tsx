"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type {
  DailyPlan, DailyPlanResponse, LiveComboStatus, LiveDashboardResponse, LiveOpenSignal, ScorecardResponse,
} from "@/lib/types";
import { DirectionBadge, StatusBadge } from "@/components/Badges";

// Read-only page (GET /dashboard/live never generates signals), so refreshing often is cheap.
// New signals come from keep-fresh.yml's cron as candles close.
const REFRESH_MS = 60_000;
const SETTINGS_KEY = "dashboard-sizing";
// When the current settings went live (spread-aware stops, profitable-on-test-data gate) --
// the scorecard counts live results from here.
const GO_LIVE = "2026-09-30";

// Same sizing convention as /trading-signals and rl_engine.usd_per_unit: USD account, and
// USD/JPY is the only one of the 4 pairs not quoted in USD.
function lotSize(pair: string, entry: number, stop: number, riskUsd: number): number {
  const stopDistance = Math.abs(entry - stop);
  if (stopDistance === 0) return 0;
  const usdPerUnit = pair === "USD/JPY" ? 1 / entry : 1;
  return riskUsd / (stopDistance * usdPerUnit) / 100_000;
}

function priceDigits(pair: string): number {
  return pair.includes("JPY") ? 3 : 5;
}

function ago(iso: string | null): string {
  if (!iso) return "never";
  const minutes = Math.round((Date.now() - new Date(iso.endsWith("Z") ? iso : `${iso}Z`).getTime()) / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  if (minutes < 48 * 60) return `${Math.round(minutes / 60)}h ago`;
  return `${Math.round(minutes / 1440)}d ago`;
}

function formatAge(minutes: number): string {
  if (minutes < 60) return `${minutes}m`;
  if (minutes < 48 * 60) return `${Math.round(minutes / 60)}h`;
  return `${Math.round(minutes / 1440)}d`;
}

// A trade that has already covered most of the way to target (or is close to its stop) is no
// longer the setup it was at entry -- flag it instead of presenting it as fresh.
function entryVerdict(s: LiveOpenSignal): { label: string; tone: string } {
  if (s.progress_r == null) return { label: "no price", tone: "text-zinc-400" };
  const rr = Math.abs(s.target_price - s.entry_price) / Math.abs(s.entry_price - s.stop_price || 1);
  if (s.progress_r <= -0.5) return { label: "near stop — skip", tone: "text-rose-600 dark:text-rose-400" };
  if (s.progress_r >= rr * 0.5) return { label: "ran — skip", tone: "text-zinc-500" };
  if (Math.abs(s.progress_r) <= 0.25) return { label: "at entry", tone: "text-emerald-600 dark:text-emerald-400" };
  return { label: s.progress_r > 0 ? "moved in favour" : "slightly against", tone: "text-amber-600 dark:text-amber-400" };
}

export default function DashboardPage() {
  const [data, setData] = useState<LiveDashboardResponse | null>(null);
  const [plan, setPlan] = useState<DailyPlanResponse | null>(null);
  const [score, setScore] = useState<ScorecardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [balance, setBalance] = useState(10_000);
  const [riskPct, setRiskPct] = useState(1);

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "null");
      if (saved?.balance > 0) setBalance(saved.balance);
      if (saved?.riskPct > 0) setRiskPct(saved.riskPct);
    } catch {
      // storage unavailable -- defaults are fine
    }
  }, []);

  function saveSizing(next: { balance: number; riskPct: number }) {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(next));
    } catch {
      // ignore
    }
  }

  async function load() {
    try {
      setError(null);
      const [live, dailyPlan, scorecard] = await Promise.allSettled([
        api.getLiveDashboard(), api.getDailyPlan(), api.getScorecard(GO_LIVE),
      ]);
      if (dailyPlan.status === "fulfilled") setPlan(dailyPlan.value);
      if (scorecard.status === "fulfilled") setScore(scorecard.value);
      if (live.status === "rejected") throw live.reason;
      setData(live.value);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Couldn't reach the backend -- it may be restarting. Retrying every minute.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    const id = setInterval(load, REFRESH_MS);
    return () => clearInterval(id);
  }, []);

  const riskUsd = balance * (riskPct / 100);
  const tradeable = data?.open.filter((s) => {
    const v = entryVerdict(s).label;
    return v !== "ran — skip" && v !== "near stop — skip";
  }) ?? [];

  return (
    <div className="flex flex-col gap-8">
      <section className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Dashboard</h1>
          <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
            Open signals you can act on now, sized to your account. Updates every minute.
          </p>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Balance (USD)">
            <input
              type="number" min="0" step="100" value={balance} className="select w-28"
              onChange={(e) => { const b = Number(e.target.value); setBalance(b); saveSizing({ balance: b, riskPct }); }}
            />
          </Field>
          <Field label="Risk / trade (%)">
            <input
              type="number" min="0.1" max="10" step="0.1" value={riskPct} className="select w-20"
              onChange={(e) => { const r = Number(e.target.value); setRiskPct(r); saveSizing({ balance, riskPct: r }); }}
            />
          </Field>
          {data && (
            <span className="text-xs text-zinc-500 dark:text-zinc-400">Updated {ago(data.generated_at)}</span>
          )}
        </div>
      </section>

      {error && (
        <p className="rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950 dark:text-rose-300">{error}</p>
      )}
      {loading && !error && <p className="text-sm text-zinc-500">Loading…</p>}

      {data && (
        <>
          <section>
            <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
              Open signals <span className="font-normal text-zinc-500">· {tradeable.length} still near entry, {data.open.length} open</span>
            </h2>
            {data.open.length === 0 ? (
              <p className="mt-2 rounded-md bg-zinc-50 px-4 py-3 text-sm text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400">
                Nothing open right now. The table below shows what each pair/interval last decided and why.
              </p>
            ) : (
              <div className="mt-2 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
                <table className="w-full text-left text-sm">
                  <thead className="table-head text-xs uppercase">
                    <tr>
                      <th className="px-3 py-2">Pair · TF</th>
                      <th className="px-3 py-2">Side</th>
                      <th className="px-3 py-2">Entry</th>
                      <th className="px-3 py-2">Now</th>
                      <th className="px-3 py-2">Target</th>
                      <th className="px-3 py-2">Stop</th>
                      <th className="px-3 py-2">Lots</th>
                      <th className="px-3 py-2">Risk / reward</th>
                      <th className="px-3 py-2">Age</th>
                      <th className="px-3 py-2">Source</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.open.map((s) => {
                      const d = priceDigits(s.pair);
                      const verdict = entryVerdict(s);
                      const reward = riskUsd * (Math.abs(s.target_price - s.entry_price) / Math.abs(s.entry_price - s.stop_price || 1));
                      return (
                        <tr key={`${s.source}-${s.signal_id}`} className="border-t border-zinc-100 dark:border-zinc-800">
                          <td className="px-3 py-2 font-mono">{s.pair} · {s.interval}</td>
                          <td className="px-3 py-2"><DirectionBadge direction={s.direction} /></td>
                          <td className="num px-3 py-2">{s.entry_price.toFixed(d)}</td>
                          <td className="num px-3 py-2">
                            {s.current_price?.toFixed(d) ?? "—"}
                            <div className={`text-xs ${verdict.tone}`}>{verdict.label}</div>
                          </td>
                          <td className="num px-3 py-2">{s.target_price.toFixed(d)}</td>
                          <td className="num px-3 py-2">{s.stop_price.toFixed(d)}</td>
                          <td className="num px-3 py-2">{lotSize(s.pair, s.entry_price, s.stop_price, riskUsd).toFixed(2)}</td>
                          <td className="px-3 py-2 text-xs">
                            <span className="text-rose-600 dark:text-rose-400">-${riskUsd.toFixed(0)}</span>
                            {" / "}
                            <span className="text-emerald-600 dark:text-emerald-400">+${reward.toFixed(0)}</span>
                          </td>
                          <td className="px-3 py-2 text-xs">{formatAge(s.age_minutes)}</td>
                          <td className="px-3 py-2 text-xs text-zinc-500">
                            {s.source === "rl" ? `RL ${s.size_tier ?? ""}` : `Consensus ${s.agreeing_count ?? ""}/5`}
                            {s.confidence_pct != null && ` · ${s.confidence_pct.toFixed(0)}%`}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {plan && <DailyPlanSection plan={plan} />}

          <section>
            <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">What each pair/interval is saying</h2>
            <ComboGrid combos={data.combos} />
          </section>

          <section>
            <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Closed in the last 24h</h2>
            {data.recent.length === 0 ? (
              <p className="mt-2 text-sm text-zinc-500">No signals resolved in the last 24 hours.</p>
            ) : (
              <div className="mt-2 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
                <table className="w-full text-left text-sm">
                  <thead className="table-head text-xs uppercase">
                    <tr>
                      <th className="px-3 py-2">Pair · TF</th>
                      <th className="px-3 py-2">Side</th>
                      <th className="px-3 py-2">Result</th>
                      <th className="px-3 py-2">Move</th>
                      <th className="px-3 py-2">Closed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.recent.map((r, i) => (
                      <tr key={i} className="border-t border-zinc-100 dark:border-zinc-800">
                        <td className="px-3 py-2 font-mono">{r.pair} · {r.interval}</td>
                        <td className="px-3 py-2"><DirectionBadge direction={r.direction} /></td>
                        <td className="px-3 py-2"><StatusBadge status={r.status} />{r.closed_early && <span className="ml-1 text-xs text-zinc-500">early exit</span>}</td>
                        <td className="num px-3 py-2">
                          {r.outcome_pct_move != null ? `${r.outcome_pct_move >= 0 ? "+" : ""}${r.outcome_pct_move.toFixed(3)}%` : "—"}
                        </td>
                        <td className="px-3 py-2 text-xs">{ago(r.outcome_timestamp)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
          {score && <ScorecardSection score={score} />}
        </>
      )}
    </div>
  );
}

function DailyPlanSection({ plan }: { plan: DailyPlanResponse }) {
  return (
    <section>
      <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Today&apos;s plan</h2>
      <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
        How far each pair typically moves today and the levels price usually reaches — no direction call: in testing
        (2020–2026) no rule predicted the day&apos;s direction better than a coin flip. Day runs 5pm–5pm New York.
      </p>
      <div className="mt-2 grid gap-3 sm:grid-cols-2">
        {plan.pairs.map((p) => <DailyPlanCard key={p.pair} p={p} />)}
      </div>
    </section>
  );
}

function DailyPlanCard({ p }: { p: DailyPlan }) {
  if (p.error || !p.expected_range_pips || p.price == null) {
    return (
      <div className="card p-3 text-sm">
        <span className="font-mono">{p.pair}</span> <span className="text-zinc-400">— {p.error ?? "no data"}</span>
      </div>
    );
  }
  const [lo, hi] = p.expected_range_pips;
  const used = p.today_range_pips ?? 0;
  const pct = Math.min(100, Math.round((used / (p.atr_pips || 1)) * 100));
  const d = priceDigits(p.pair);
  return (
    <div className="card p-3">
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-sm font-semibold">{p.pair}</span>
        <span className="num text-sm">{p.price.toFixed(d)}</span>
      </div>
      <div className="mt-2 text-xs text-zinc-600 dark:text-zinc-300">
        Typical day <span className="num font-medium">{p.atr_pips}</span> pips (usually {lo}–{hi})
      </div>
      <div className="mt-1 flex items-center gap-2">
        <div className="h-1.5 flex-1 rounded bg-zinc-200 dark:bg-zinc-700">
          <div
            className={`h-1.5 rounded ${pct >= 100 ? "bg-amber-500" : "bg-sky-500"}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="num text-xs text-zinc-500">
          {p.today_range_pips == null ? "not started" : `${used} moved (${pct}%)`}
        </span>
      </div>
      <table className="mt-2 w-full text-xs">
        <tbody>
          {(p.levels ?? []).map((lv) => (
            <tr key={lv.name} className="border-t border-zinc-100 dark:border-zinc-800">
              <td className="py-1 text-zinc-600 dark:text-zinc-300">{lv.name}</td>
              <td className="num py-1 text-right">{lv.price.toFixed(d)}</td>
              <td className={`num py-1 text-right ${lv.distance_pips >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                {lv.distance_pips >= 0 ? "+" : ""}{lv.distance_pips}p
              </td>
              <td className="py-1 pl-2 text-right text-zinc-400">{lv.touched_today ? "✓ hit today" : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ScorecardSection({ score }: { score: ScorecardResponse }) {
  const t = score.totals;
  return (
    <section>
      <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
        Live results since {score.since}
        <span className="font-normal text-zinc-500">
          {" "}· {t.trades} closed{t.win_rate_pct != null ? `, ${t.win_rate_pct}% wins, ${t.avg_net_pct! >= 0 ? "+" : ""}${t.avg_net_pct}% avg per trade` : ""}, {t.open} open
        </span>
      </h2>
      <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
        Every live signal scored against real prices, after spread. This is the real test of whether the system works —
        judge a row only after ~30+ closed trades, and by average result, not win rate.
      </p>
      {score.rows.length === 0 ? (
        <p className="mt-2 text-sm text-zinc-500">No live signals since {score.since} yet.</p>
      ) : (
        <div className="mt-2 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
          <table className="w-full text-left text-sm">
            <thead className="table-head text-xs uppercase">
              <tr>
                <th className="px-3 py-2">Pair · TF</th>
                <th className="px-3 py-2">Source</th>
                <th className="px-3 py-2">Closed</th>
                <th className="px-3 py-2">Open</th>
                <th className="px-3 py-2">Wins</th>
                <th className="px-3 py-2">Avg / trade</th>
                <th className="px-3 py-2">Total</th>
              </tr>
            </thead>
            <tbody>
              {score.rows.map((r) => (
                <tr key={`${r.source}-${r.pair}-${r.interval}`} className="border-t border-zinc-100 dark:border-zinc-800">
                  <td className="px-3 py-2 font-mono">{r.pair} · {r.interval}</td>
                  <td className="px-3 py-2 text-xs text-zinc-500">{r.source === "rl" ? "RL" : "Consensus"}</td>
                  <td className="num px-3 py-2">{r.trades}</td>
                  <td className="num px-3 py-2">{r.open}</td>
                  <td className="num px-3 py-2">{r.win_rate_pct != null ? `${r.win_rate_pct}%` : "—"}</td>
                  <td className={`num px-3 py-2 ${(r.avg_net_pct ?? 0) >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                    {r.avg_net_pct != null ? `${r.avg_net_pct >= 0 ? "+" : ""}${r.avg_net_pct.toFixed(3)}%` : "—"}
                  </td>
                  <td className="num px-3 py-2">{r.total_net_pct >= 0 ? "+" : ""}{r.total_net_pct.toFixed(3)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function comboLine(c: LiveComboStatus): { text: string; tone: string } {
  if (c.policy_state === "none") return { text: "No trained policy", tone: "text-zinc-400" };
  if (c.policy_state === "stale") return { text: "Policy needs retraining (old format)", tone: "text-zinc-400" };
  if (c.policy_state === "excluded") {
    const ev = c.policy_eval;
    const ret = ev?.total_return_pct;
    const text = ret != null && ret <= 0
      ? `Paused — ${ret.toFixed(0)}% on test data (${ev?.trades ?? 0} trades)`
      : `Paused — only ${ev?.trades ?? 0} test trades`;
    return { text, tone: "text-rose-600 dark:text-rose-400" };
  }
  const d = c.decision;
  if (!d) return { text: "Not checked yet", tone: "text-zinc-400" };
  switch (d.outcome) {
    case "signal":
      return { text: "Signal open ↑", tone: "text-emerald-600 dark:text-emerald-400" };
    case "hold": {
      const q = d.q_values ?? {};
      const hold = Math.round(((q.HOLD ?? 0) + (q.EXIT ?? 0)) * 100);
      return { text: `Waiting (HOLD ${hold}%)`, tone: "text-zinc-500" };
    }
    case "blocked":
      return { text: "Setup blocked by safety check", tone: "text-amber-600 dark:text-amber-400" };
    case "excluded":
      return { text: "Paused by backtest result", tone: "text-rose-600 dark:text-rose-400" };
    case "error":
      return { text: `Error: ${d.detail ?? "unknown"}`, tone: "text-rose-600 dark:text-rose-400" };
  }
}

function ComboGrid({ combos }: { combos: LiveComboStatus[] }) {
  const pairs = Array.from(new Set(combos.map((c) => c.pair)));
  const intervals = Array.from(new Set(combos.map((c) => c.interval)));
  return (
    <div className="mt-2 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
      <table className="w-full text-left text-xs">
        <thead className="table-head uppercase">
          <tr>
            <th className="px-3 py-2">Pair</th>
            {intervals.map((iv) => <th key={iv} className="px-3 py-2">{iv}</th>)}
          </tr>
        </thead>
        <tbody>
          {pairs.map((pair) => (
            <tr key={pair} className="border-t border-zinc-100 align-top dark:border-zinc-800">
              <td className="px-3 py-2 font-mono text-sm">{pair}</td>
              {intervals.map((iv) => {
                const c = combos.find((x) => x.pair === pair && x.interval === iv);
                if (!c) return <td key={iv} />;
                const line = comboLine(c);
                return (
                  <td key={iv} className="px-3 py-2" title={c.policy_eval?.reason ?? c.decision?.detail ?? undefined}>
                    <div className={line.tone}>{line.text}</div>
                    <div className="mt-0.5 text-zinc-400">
                      checked {ago(c.decision?.checked_at ?? null)}
                      {c.policy_state === "outdated" && " · retrain pending"}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-zinc-500">
      {label}
      {children}
    </label>
  );
}
