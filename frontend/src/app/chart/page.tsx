"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ApiError } from "@/lib/api";
import { INTERVALS, PAIRS, type CandlePoint, type ConsensusSignal } from "@/lib/types";

// Validated against the dataviz skill's CVD/contrast checks (light + dark):
// node scripts/validate_palette.js "#2a78d6,#eb6834,#10b981,#e34948" --mode light  -> PASS
// EMA fast/slow (blue/orange) are a clean categorical pair. BUY/SELL reuse this app's
// established emerald/rose convention (Badges.tsx) — green/red is an inherently hard
// deuteranopia pair regardless of exact shade, so markers carry shape (triangle up/down)
// as the secondary encoding the skill requires when a CVD pair can't clear the floor.
const COLOR = {
  price: { light: "#0b0b0b", dark: "#f4f4f5" },
  emaFast: { light: "#2a78d6", dark: "#3987e5" },
  emaSlow: { light: "#eb6834", dark: "#d95926" },
  buy: { light: "#10b981", dark: "#34d399" },
  sell: { light: "#e34948", dark: "#fb7185" },
  muted: { light: "#898781", dark: "#898781" },
  grid: { light: "#e1e0d9", dark: "#2c2c2a" },
  surface: { light: "#ffffff", dark: "#18181b" }, // matches this app's card bg (white / zinc-900)
};

function useIsDark() {
  const [isDark, setIsDark] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setIsDark(document.documentElement.getAttribute("data-theme") === "dark" || (!document.documentElement.hasAttribute("data-theme") && mq.matches));
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return isDark;
}

function TrianglePoint({ cx, cy, fill, stroke, up }: { cx?: number; cy?: number; fill: string; stroke: string; up: boolean }) {
  if (cx == null || cy == null) return null;
  const size = 6;
  const points = up
    ? `${cx},${cy - size} ${cx - size},${cy + size} ${cx + size},${cy + size}`
    : `${cx},${cy + size} ${cx - size},${cy - size} ${cx + size},${cy - size}`;
  return <polygon points={points} fill={fill} stroke={stroke} strokeWidth={1.5} />;
}

export default function ChartPage() {
  const isDark = useIsDark();
  const mode = isDark ? "dark" : "light";

  const [pair, setPair] = useState<string>(PAIRS[0]);
  const [interval, setInterval_] = useState<string>("1h");

  const [candles, setCandles] = useState<CandlePoint[]>([]);
  const [signals, setSignals] = useState<ConsensusSignal[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [candleData, signalData] = await Promise.all([
        api.getCandles(pair, interval, "intraday", 250),
        api.listConsensusSignals({ pair, limit: 300 }),
      ]);
      setCandles(candleData);
      setSignals(signalData.filter((s) => s.interval === interval));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load chart data. Is the API running?");
      setCandles([]);
      setSignals([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pair, interval]);

  // Merge signal markers onto the candle series by nearest timestamp, so Scatter can
  // plot them on the same x-axis as the price line without a separate axis.
  const chartData = useMemo(() => {
    const signalByTs = new Map<string, ConsensusSignal>();
    for (const s of signals) {
      const nearest = candles.reduce((best, c) => {
        const d = Math.abs(new Date(c.timestamp).getTime() - new Date(s.timestamp).getTime());
        const bestD = best ? Math.abs(new Date(best.timestamp).getTime() - new Date(s.timestamp).getTime()) : Infinity;
        return d < bestD ? c : best;
      }, undefined as CandlePoint | undefined);
      if (nearest) signalByTs.set(nearest.timestamp, s);
    }
    return candles.map((c) => {
      const sig = signalByTs.get(c.timestamp);
      return {
        ...c,
        label: new Date(c.timestamp).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }),
        buyMarker: sig?.direction === "BUY" ? c.close : null,
        sellMarker: sig?.direction === "SELL" ? c.close : null,
      };
    });
  }, [candles, signals]);

  return (
    <div className="flex flex-col gap-6">
      <section>
        <h1 className="text-xl font-semibold tracking-tight">Chart</h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          Price with EMA overlays, RSI and MACD panels, and directional signal markers
          (▲ BUY / ▼ SELL) plotted at the candle each signal actually fired on.
        </p>
      </section>

      <section className="card p-4">
        <div className="flex flex-wrap items-end gap-3">
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
          <button onClick={load} disabled={loading} className="btn-secondary">
            {loading ? "Loading…" : "Refresh"}
          </button>
        </div>
      </section>

      {error && (
        <p className="rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950 dark:text-rose-300">
          {error}
        </p>
      )}
      {!error && !loading && chartData.length === 0 && (
        <p className="text-sm text-zinc-500">No candle data. Ingest some first from the Trading Signals page.</p>
      )}

      {chartData.length > 0 && (
        <div className="viz-root flex flex-col gap-1 card p-4">
          <div className="mb-2 flex items-center gap-4 text-xs text-zinc-500 dark:text-zinc-400">
            <LegendSwatch color={COLOR.price[mode]} label="Close" line />
            <LegendSwatch color={COLOR.emaFast[mode]} label={`EMA fast`} line />
            <LegendSwatch color={COLOR.emaSlow[mode]} label={`EMA slow`} line />
            <span className="flex items-center gap-1">
              <svg width="10" height="10" viewBox="-6 -6 12 12"><polygon points="0,-6 -6,6 6,6" fill={COLOR.buy[mode]} /></svg>
              BUY
            </span>
            <span className="flex items-center gap-1">
              <svg width="10" height="10" viewBox="-6 -6 12 12"><polygon points="0,6 -6,-6 6,-6" fill={COLOR.sell[mode]} /></svg>
              SELL
            </span>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} syncId="chart" margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={COLOR.grid[mode]} />
                <XAxis dataKey="label" tick={{ fontSize: 10 }} minTickGap={40} />
                <YAxis domain={["auto", "auto"]} tick={{ fontSize: 10 }} width={60} />
                <Tooltip content={<ChartTooltip mode={mode} />} />
                <Line type="monotone" dataKey="close" stroke={COLOR.price[mode]} strokeWidth={2} dot={false} name="Close" isAnimationActive={false} />
                <Line type="monotone" dataKey="ema_fast" stroke={COLOR.emaFast[mode]} strokeWidth={2} dot={false} name="EMA fast" isAnimationActive={false} connectNulls />
                <Line type="monotone" dataKey="ema_slow" stroke={COLOR.emaSlow[mode]} strokeWidth={2} dot={false} name="EMA slow" isAnimationActive={false} connectNulls />
                <Scatter dataKey="buyMarker" fill={COLOR.buy[mode]} shape={(p: any) => <TrianglePoint {...p} fill={COLOR.buy[mode]} stroke={COLOR.surface[mode]} up />} isAnimationActive={false} />
                <Scatter dataKey="sellMarker" fill={COLOR.sell[mode]} shape={(p: any) => <TrianglePoint {...p} fill={COLOR.sell[mode]} stroke={COLOR.surface[mode]} up={false} />} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <p className="mt-4 text-xs font-medium text-zinc-500 dark:text-zinc-400">RSI</p>
          <div className="h-32 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} syncId="chart" margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={COLOR.grid[mode]} />
                <XAxis dataKey="label" tick={{ fontSize: 10 }} minTickGap={40} hide />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} width={60} ticks={[30, 50, 70]} />
                <Tooltip content={<ChartTooltip mode={mode} />} />
                <ReferenceLine y={70} stroke={COLOR.muted[mode]} strokeDasharray="4 4" />
                <ReferenceLine y={30} stroke={COLOR.muted[mode]} strokeDasharray="4 4" />
                <Line type="monotone" dataKey="rsi" stroke={COLOR.emaFast[mode]} strokeWidth={2} dot={false} name="RSI" isAnimationActive={false} connectNulls />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <p className="mt-4 text-xs font-medium text-zinc-500 dark:text-zinc-400">MACD</p>
          <div className="h-32 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} syncId="chart" margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={COLOR.grid[mode]} />
                <XAxis dataKey="label" tick={{ fontSize: 10 }} minTickGap={40} />
                <YAxis tick={{ fontSize: 10 }} width={60} />
                <Tooltip content={<ChartTooltip mode={mode} />} />
                <ReferenceLine y={0} stroke={COLOR.muted[mode]} />
                <Bar dataKey="macd_hist" name="Histogram" isAnimationActive={false}>
                  {chartData.map((d, i) => (
                    <Cell key={i} fill={(d.macd_hist ?? 0) >= 0 ? COLOR.buy[mode] : COLOR.sell[mode]} />
                  ))}
                </Bar>
                <Line type="monotone" dataKey="macd" stroke={COLOR.emaFast[mode]} strokeWidth={2} dot={false} name="MACD" isAnimationActive={false} connectNulls />
                <Line type="monotone" dataKey="macd_signal" stroke={COLOR.emaSlow[mode]} strokeWidth={2} dot={false} name="Signal" isAnimationActive={false} connectNulls />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

function LegendSwatch({ color, label, line }: { color: string; label: string; line?: boolean }) {
  return (
    <span className="flex items-center gap-1.5">
      <span
        className="inline-block"
        style={line ? { width: 12, height: 2, background: color } : { width: 8, height: 8, borderRadius: 9999, background: color }}
      />
      {label}
    </span>
  );
}

function ChartTooltip({ active, payload, label, mode }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="card px-3 py-2 text-xs shadow-sm">
      <p className="mb-1 font-medium text-zinc-700 dark:text-zinc-200">{label}</p>
      {payload.map((p: any) => (
        p.value != null && (
          <p key={p.dataKey} style={{ color: p.color }}>
            {p.name}: {typeof p.value === "number" ? p.value.toFixed(5) : p.value}
          </p>
        )
      ))}
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
