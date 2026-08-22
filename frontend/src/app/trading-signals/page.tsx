"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { INTERVALS, PAIRS, type ConsensusSignal } from "@/lib/types";

// USD value of a 1.0 price-unit move per single unit of base currency traded, for this
// project's fixed 4-pair universe. EUR/USD, GBP/USD, AUD/USD already quote in USD, so a
// price-unit move IS a USD move. USD/JPY quotes in JPY, so a JPY-denominated move needs
// converting to USD -- dividing by the pair's own rate does that here specifically because
// USD is USD/JPY's base currency, not because this generalizes to any pair/quote-currency
// combination. Extending this to a pair that isn't USD-quoted on one side or the other would
// need a real currency-conversion rate, not this shortcut.
function usdPerUnit(pair: string, entryPrice: number): number {
  return pair === "USD/JPY" ? 1 / entryPrice : 1;
}

// Standard lot = 100,000 units of base currency -- the conventional forex sizing unit.
function calculateLotSize(pair: string, entryPrice: number, stopPrice: number, riskAmountUsd: number): number {
  const stopDistance = Math.abs(entryPrice - stopPrice);
  if (stopDistance === 0) return 0;
  const units = riskAmountUsd / (stopDistance * usdPerUnit(pair, entryPrice));
  return units / 100_000;
}

interface ActionableSignal {
  signal: ConsensusSignal;
  lots: number;
  riskUsd: number;
  potentialRewardUsd: number;
}

export default function TradingSignalsPage() {
  const [accountBalance, setAccountBalance] = useState(10000);
  const [riskPercent, setRiskPercent] = useState(1);
  const [loading, setLoading] = useState(false);
  const [signals, setSignals] = useState<ActionableSignal[]>([]);
  const [checkedCount, setCheckedCount] = useState(0);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  async function refresh() {
    setLoading(true);
    const combos = PAIRS.flatMap((p) => INTERVALS.map((i) => ({ pair: p, interval: i })));
    const riskAmountUsd = accountBalance * (riskPercent / 100);

    const results = await Promise.all(
      combos.map(async ({ pair, interval }) => {
        try {
          const result = await api.generateConsensus(pair, interval);
          return result.consensus;
        } catch {
          return null;
        }
      }),
    );

    const actionable: ActionableSignal[] = results
      .filter((s): s is ConsensusSignal => s !== null)
      .map((signal) => ({
        signal,
        lots: calculateLotSize(signal.pair, signal.entry_price, signal.stop_price, riskAmountUsd),
        riskUsd: riskAmountUsd,
        potentialRewardUsd: riskAmountUsd * (Math.abs(signal.target_price - signal.entry_price) / Math.abs(signal.entry_price - signal.stop_price)),
      }));

    setSignals(actionable);
    setCheckedCount(combos.length);
    setLastChecked(new Date());
    setLoading(false);
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex flex-col gap-8">
      <section>
        <h1 className="text-xl font-semibold tracking-tight">Trading signals</h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          Only signals where a weighted majority of independent strategies agree — every row here is
          a fired consensus signal (see the Consensus page for the full picture, including
          where nothing agreed). This is a practical stand-in while the ML angle is still
          being built, not a claim that any of this is a validated edge yet — check the
          consensus backtest results before sizing anything for real.
        </p>
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Position sizing</h2>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          Lot size = (account balance &times; risk %) &divide; (stop distance in price &times;
          USD value per unit). Standard lot = 100,000 units. This assumes a USD account and
          only correctly handles this project&apos;s 4 pairs&apos; specific quote currencies
          (EUR/USD, GBP/USD, AUD/USD quote in USD directly; USD/JPY is converted via its own
          rate) — it is not a general multi-currency calculator.
        </p>
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <Field label="Account balance (USD)">
            <input
              type="number" step="100" min="0" value={accountBalance}
              onChange={(e) => setAccountBalance(Number(e.target.value))}
              className="select w-32"
            />
          </Field>
          <Field label="Risk per trade (%)">
            <input
              type="number" step="0.1" min="0.1" max="10" value={riskPercent}
              onChange={(e) => setRiskPercent(Number(e.target.value))}
              className="select w-24"
            />
          </Field>
          <button onClick={refresh} disabled={loading} className="btn-primary">
            {loading ? `Checking ${checkedCount || 20} combinations…` : "Refresh signals"}
          </button>
          {lastChecked && !loading && (
            <span className="text-xs text-zinc-500 dark:text-zinc-400">
              Last checked {lastChecked.toLocaleTimeString()}
            </span>
          )}
        </div>
      </section>

      <section>
        {!loading && signals.length === 0 && (
          <p className="rounded-md bg-zinc-50 px-4 py-3 text-sm text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400">
            No consensus signals firing across any pair/interval right now — that&apos;s the
            normal state, not an error. Check back later or see the Consensus page for what
            each strategy is currently saying.
          </p>
        )}
        {signals.length > 0 && (
          <div className="overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
            <table className="w-full text-left text-sm">
              <thead className="bg-zinc-50 text-xs uppercase text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
                <tr>
                  <th className="px-4 py-2">Pair / interval</th>
                  <th className="px-4 py-2">Direction</th>
                  <th className="px-4 py-2">Entry</th>
                  <th className="px-4 py-2">Exit</th>
                  <th className="px-4 py-2">Stop</th>
                  <th className="px-4 py-2">Agree</th>
                  <th className="px-4 py-2">Lot size</th>
                  <th className="px-4 py-2">Risk / reward</th>
                </tr>
              </thead>
              <tbody>
                {signals.map(({ signal, lots, riskUsd, potentialRewardUsd }) => (
                  <tr key={signal._id ?? `${signal.pair}-${signal.interval}`} className="border-t border-zinc-100 dark:border-zinc-800">
                    <td className="px-4 py-2 font-mono">{signal.pair} · {signal.interval}</td>
                    <td className={`px-4 py-2 text-base font-semibold ${signal.direction === "BUY" ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                      {signal.direction}
                    </td>
                    <td className="px-4 py-2 font-mono">{signal.entry_price.toFixed(5)}</td>
                    <td className="px-4 py-2 font-mono">{signal.target_price.toFixed(5)}</td>
                    <td className="px-4 py-2 font-mono">{signal.stop_price.toFixed(5)}</td>
                    <td className="px-4 py-2">{signal.agreeing_count}/{signal.strategy_calls.length}</td>
                    <td className="px-4 py-2 font-mono">{lots.toFixed(2)}</td>
                    <td className="px-4 py-2 text-xs">
                      <span className="text-rose-600 dark:text-rose-400">-${riskUsd.toFixed(0)}</span>
                      {" / "}
                      <span className="text-emerald-600 dark:text-emerald-400">+${potentialRewardUsd.toFixed(0)}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
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
