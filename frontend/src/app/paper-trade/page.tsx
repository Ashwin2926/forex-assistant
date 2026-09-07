"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { INTERVALS, PAIRS, type PaperTrade, type PaperTradeAccount, type PaperTradeResult } from "@/lib/types";
import { DirectionBadge, PaperTradeStatusBadge } from "@/components/Badges";

export default function PaperTradePage() {
  const [account, setAccount] = useState<PaperTradeAccount | null>(null);
  const [accountError, setAccountError] = useState<string | null>(null);
  const [accountLoading, setAccountLoading] = useState(true);

  const [open, setOpen] = useState<PaperTrade[]>([]);
  const [history, setHistory] = useState<PaperTrade[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  const [listLoading, setListLoading] = useState(true);

  const [pair, setPair] = useState<string>(PAIRS[0]);
  const [interval, setInterval_] = useState<string>("1h");
  const [stake, setStake] = useState(10);
  const [multiplier, setMultiplier] = useState(100);
  const [running, setRunning] = useState(false);
  const [runResult, setRunResult] = useState<PaperTradeResult | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  async function loadAccount() {
    setAccountLoading(true);
    setAccountError(null);
    try {
      setAccount(await api.getPaperTradeAccount());
    } catch (e) {
      setAccountError(e instanceof ApiError ? e.message : "Failed to reach the API. Is it running?");
    } finally {
      setAccountLoading(false);
    }
  }

  async function loadTrades() {
    setListLoading(true);
    setListError(null);
    try {
      const [openTrades, historyTrades] = await Promise.all([
        api.listOpenPaperTrades(),
        api.listPaperTradeHistory({ limit: 50 }),
      ]);
      setOpen(openTrades);
      setHistory(historyTrades);
    } catch (e) {
      setListError(e instanceof ApiError ? e.message : "Failed to load paper trades.");
    } finally {
      setListLoading(false);
    }
  }

  useEffect(() => {
    loadAccount();
    loadTrades();
  }, []);

  async function handleExecute() {
    setRunning(true);
    setRunError(null);
    setRunResult(null);
    try {
      const result = await api.executePaperTrade(pair, interval, "intraday", { stake, multiplier });
      setRunResult(result);
      await loadTrades();
    } catch (e) {
      setRunError(e instanceof ApiError ? e.message : "Paper trade failed.");
    } finally {
      setRunning(false);
    }
  }

  const isDemo = account?.is_virtual === true;

  return (
    <div className="flex flex-col gap-8">
      <section>
        <h1 className="text-xl font-semibold tracking-tight">Paper Trading (Deriv)</h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          Executes a fresh live signal as a Deriv Multipliers contract on your demo account —
          real order flow, fake money. The backend refuses to trade on anything but a virtual account.
        </p>
      </section>

      <section
        className={`rounded-lg border p-4 ${
          accountError || (!accountLoading && !isDemo)
            ? "border-rose-300 bg-rose-50 dark:border-rose-900 dark:bg-rose-950"
            : "border-[var(--border)] bg-[var(--surface)]"
        }`}
      >
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Deriv account</h2>
          <button onClick={loadAccount} disabled={accountLoading} className="btn-secondary">
            {accountLoading ? "Checking…" : "Re-check"}
          </button>
        </div>
        {accountLoading && <p className="mt-2 text-sm text-zinc-500">Authorizing…</p>}
        {accountError && (
          <p className="mt-2 text-sm text-rose-700 dark:text-rose-300">
            {accountError}
            {accountError.toLowerCase().includes("deriv_api_token") && (
              <> — set <code className="font-mono">DERIV_API_TOKEN</code> in the backend&apos;s <code className="font-mono">.env</code>, generated from a Deriv <strong>demo</strong> account.</>
            )}
          </p>
        )}
        {account && !accountLoading && (
          <div className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-sm">
            <span>
              Account:{" "}
              <span className={isDemo ? "font-medium text-emerald-700 dark:text-emerald-400" : "font-bold text-rose-700 dark:text-rose-400"}>
                {isDemo ? "DEMO / virtual" : "⚠ REAL — trading disabled"}
              </span>
            </span>
            <span className="text-zinc-500">Login ID: {account.loginid}</span>
            <span className="text-zinc-500">Balance: {account.balance} {account.currency}</span>
          </div>
        )}
      </section>

      <section className="card p-4">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Execute a signal</h2>
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <Field label="Pair">
            <select value={pair} onChange={(e) => setPair(e.target.value)} className="select">
              {PAIRS.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </Field>
          <Field label="Interval">
            <select value={interval} onChange={(e) => setInterval_(e.target.value)} className="select">
              {INTERVALS.map((i) => <option key={i} value={i}>{i}</option>)}
            </select>
          </Field>
          <Field label="Stake">
            <input type="number" step="1" value={stake} onChange={(e) => setStake(Number(e.target.value))} className="select w-20" />
          </Field>
          <Field label="Multiplier">
            <input type="number" step="10" value={multiplier} onChange={(e) => setMultiplier(Number(e.target.value))} className="select w-24" />
          </Field>
          <button
            onClick={handleExecute}
            disabled={running || accountLoading || !isDemo}
            className="btn-primary"
            title={!isDemo ? "Disabled: connected account is not a demo account" : undefined}
          >
            {running ? "Executing…" : "Generate signal & trade"}
          </button>
        </div>
        {runError && (
          <p className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {runError}
          </p>
        )}
        {runResult && (
          <div className="mt-3 rounded-md bg-zinc-50 px-3 py-2 text-xs dark:bg-zinc-800">
            <p>
              Signal: <DirectionBadge direction={runResult.signal.direction} /> conf {runResult.signal.confidence}% @ {runResult.signal.price_at_signal.toFixed(5)}
            </p>
            {runResult.note && <p className="mt-1 text-zinc-500">{runResult.note}</p>}
            {runResult.paper_trade && (
              <p className="mt-1 text-zinc-500">
                {runResult.paper_trade.status === "error"
                  ? `Trade failed: ${runResult.paper_trade.error}`
                  : `Contract #${runResult.paper_trade.contract_id} opened — TP ${runResult.paper_trade.take_profit_amount} / SL ${runResult.paper_trade.stop_loss_amount} ${runResult.paper_trade.currency}`}
              </p>
            )}
          </div>
        )}
      </section>

      <section>
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Open positions</h2>
          <button onClick={loadTrades} disabled={listLoading} className="btn-secondary">
            {listLoading ? "Refreshing…" : "Refresh"}
          </button>
        </div>
        {listError && (
          <p className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {listError}
          </p>
        )}
        <TradeTable trades={open} emptyMessage="No open positions." />
      </section>

      <section>
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">History</h2>
        <TradeTable trades={history} emptyMessage="No closed trades yet." showPnl />
      </section>
    </div>
  );
}

function TradeTable({ trades, emptyMessage, showPnl }: { trades: PaperTrade[]; emptyMessage: string; showPnl?: boolean }) {
  if (trades.length === 0) {
    return <p className="mt-3 text-sm text-zinc-500">{emptyMessage}</p>;
  }
  return (
    <div className="mt-3 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
      <table className="w-full text-left text-sm">
        <thead className="table-head text-xs uppercase">
          <tr>
            <th className="px-4 py-2">Pair</th>
            <th className="px-4 py-2">Direction</th>
            <th className="px-4 py-2">Stake × Mult</th>
            <th className="px-4 py-2">TP / SL</th>
            <th className="px-4 py-2">Status</th>
            {showPnl && <th className="px-4 py-2">P&L</th>}
            <th className="px-4 py-2">Opened</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t) => (
            <tr key={t._id ?? t.contract_id} className="border-t border-zinc-100 dark:border-zinc-800">
              <td className="px-4 py-2">{t.pair} · {t.interval}</td>
              <td className="px-4 py-2"><DirectionBadge direction={t.direction} /></td>
              <td className="px-4 py-2">{t.stake} × {t.multiplier}</td>
              <td className="px-4 py-2 text-xs">{t.take_profit_amount} / {t.stop_loss_amount} {t.currency}</td>
              <td className="px-4 py-2"><PaperTradeStatusBadge status={t.status} /></td>
              {showPnl && (
                <td className={`px-4 py-2 text-xs ${t.pnl != null && t.pnl >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                  {t.pnl != null ? `${t.pnl >= 0 ? "+" : ""}${t.pnl} ${t.currency}` : "—"}
                </td>
              )}
              <td className="px-4 py-2 text-xs text-zinc-500">{new Date(t.opened_at).toLocaleString()}</td>
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
