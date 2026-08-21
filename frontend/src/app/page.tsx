"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { INTERVALS, PAIRS, PROFILES, type Profile, type Signal, type SignalAccuracy } from "@/lib/types";
import { DirectionBadge, StatusBadge } from "@/components/Badges";

export default function SignalFeedPage() {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pairFilter, setPairFilter] = useState<string>("");
  const [profileFilter, setProfileFilter] = useState<string>("");
  const [expanded, setExpanded] = useState<string | null>(null);

  const [accuracy, setAccuracy] = useState<SignalAccuracy | null>(null);
  const [accuracyError, setAccuracyError] = useState<string | null>(null);
  const [scoring, setScoring] = useState(false);

  const [triggerPair, setTriggerPair] = useState<string>(PAIRS[0]);
  const [triggerInterval, setTriggerInterval] = useState<string>("1h");
  const [triggerProfile, setTriggerProfile] = useState<Profile>("swing");
  const [busy, setBusy] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  const [generatingAll, setGeneratingAll] = useState(false);
  const [generateAllProgress, setGenerateAllProgress] = useState<string | null>(null);

  async function loadSignals() {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listSignals({ pair: pairFilter || undefined, limit: 100 });
      setSignals(profileFilter ? data.filter((s) => s.profile === profileFilter) : data);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load signals. Is the API running?");
    } finally {
      setLoading(false);
    }
  }

  async function loadAccuracy() {
    setAccuracyError(null);
    try {
      setAccuracy(await api.getSignalAccuracy({ pair: pairFilter || undefined, profile: profileFilter || undefined }));
    } catch (e) {
      setAccuracyError(e instanceof ApiError ? e.message : "Failed to load accuracy.");
    }
  }

  useEffect(() => {
    loadSignals();
    loadAccuracy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pairFilter, profileFilter]);

  async function handleScore() {
    setScoring(true);
    try {
      await api.scoreSignals();
      await Promise.all([loadSignals(), loadAccuracy()]);
    } catch (e) {
      setActionMessage(e instanceof ApiError ? `Scoring failed: ${e.message}` : "Scoring failed.");
    } finally {
      setScoring(false);
    }
  }

  async function handleIngest() {
    setBusy("ingest");
    setActionMessage(null);
    try {
      const result = await api.ingest(triggerInterval);
      setActionMessage(`Ingested ${triggerInterval}: ${JSON.stringify(result)}`);
    } catch (e) {
      setActionMessage(e instanceof ApiError ? `Ingest failed: ${e.message}` : "Ingest failed.");
    } finally {
      setBusy(null);
    }
  }

  async function handleGenerateSignal() {
    setBusy("signal");
    setActionMessage(null);
    try {
      await api.generateSignal(triggerPair, triggerInterval, triggerProfile);
      setActionMessage(`Generated signal for ${triggerPair} / ${triggerInterval} / ${triggerProfile}`);
      await loadSignals();
    } catch (e) {
      setActionMessage(e instanceof ApiError ? `Signal generation failed: ${e.message}` : "Signal generation failed.");
    } finally {
      setBusy(null);
    }
  }

  async function handleGenerateAll() {
    setGeneratingAll(true);
    setActionMessage(null);
    const combos: { interval: string; profile: Profile }[] = [
      { interval: "1h", profile: "swing" },
      { interval: "15min", profile: "intraday" },
    ];
    const jobs = PAIRS.flatMap((pair) => combos.map((c) => ({ pair, ...c })));
    const results: string[] = [];

    for (let i = 0; i < jobs.length; i++) {
      const { pair, interval, profile } = jobs[i];
      setGenerateAllProgress(`${i + 1}/${jobs.length}: ${pair} ${interval}/${profile}…`);
      try {
        const signal = await api.generateSignal(pair, interval, profile);
        results.push(`${pair} ${profile}: ${signal.direction}`);
      } catch (e) {
        results.push(`${pair} ${profile}: FAILED (${e instanceof ApiError ? e.message : "error"})`);
      }
      await loadSignals(); // refresh incrementally so results appear as they land, not just at the end
    }

    setGenerateAllProgress(null);
    setActionMessage(`Generate all done — ${results.join(" · ")}`);
    setGeneratingAll(false);
    await loadAccuracy();
  }

  return (
    <div className="flex flex-col gap-8">
      <section>
        <h1 className="text-xl font-semibold tracking-tight">Signal Feed</h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          Live BUY/SELL/HOLD signals with full rule-by-rule reasoning. Nothing here is trusted until
          it has a <a href="/backtest" className="underline underline-offset-2">backtested track record</a>.
        </p>
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Manual triggers</h2>
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <Field label="Pair">
            <select
              value={triggerPair}
              onChange={(e) => setTriggerPair(e.target.value)}
              className="select"
            >
              {PAIRS.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </Field>
          <Field label="Interval">
            <select
              value={triggerInterval}
              onChange={(e) => setTriggerInterval(e.target.value)}
              className="select"
            >
              {INTERVALS.map((i) => (
                <option key={i} value={i}>{i}</option>
              ))}
            </select>
          </Field>
          <Field label="Profile">
            <select
              value={triggerProfile}
              onChange={(e) => setTriggerProfile(e.target.value as Profile)}
              className="select"
            >
              {PROFILES.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </Field>
          <button
            onClick={handleIngest}
            disabled={busy !== null}
            className="btn-secondary"
          >
            {busy === "ingest" ? "Ingesting…" : "Ingest candles"}
          </button>
          <button
            onClick={handleGenerateSignal}
            disabled={busy !== null || generatingAll}
            className="btn-primary"
          >
            {busy === "signal" ? "Generating…" : "Generate signal"}
          </button>
          <button
            onClick={handleGenerateAll}
            disabled={busy !== null || generatingAll}
            className="btn-secondary"
            title="Generates a signal for every pair, both profiles (8 total) — ingest candles first if you haven't."
          >
            {generatingAll ? "Generating all…" : "Generate all (8)"}
          </button>
        </div>
        {generateAllProgress && (
          <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">{generateAllProgress}</p>
        )}
        {actionMessage && (
          <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">{actionMessage}</p>
        )}
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
            Live accuracy {pairFilter && `· ${pairFilter}`} {profileFilter && `· ${profileFilter}`}
          </h2>
          <button onClick={handleScore} disabled={scoring} className="btn-secondary">
            {scoring ? "Scoring…" : "Score pending signals"}
          </button>
        </div>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          Rolling hit-rate over resolved live signals only — excludes still-pending signals and
          anything from a backtest run. This is what tells you if live performance actually
          matches what was backtested; a gap here is useful signal, not a bug. Runs automatically
          every 15 minutes, or trigger it on demand above.
        </p>
        {accuracyError && (
          <p className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {accuracyError}
          </p>
        )}
        {accuracy && !accuracyError && (
          accuracy.sample_size === 0 ? (
            <p className="mt-3 text-sm text-zinc-500">
              No resolved live signals yet — generate some and give them time (or hit &quot;Score pending signals&quot;
              once enough candles have arrived).
            </p>
          ) : (
            <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <AccuracyStat label="Hit rate" value={accuracy.hit_rate_pct != null ? `${accuracy.hit_rate_pct}%` : "—"} />
              <AccuracyStat label="Sample size (last 100)" value={String(accuracy.sample_size)} />
              <AccuracyStat label="Total resolved" value={String(accuracy.total_resolved)} />
              <AccuracyStat label="Hit / Miss / Expired" value={`${accuracy.hits} / ${accuracy.misses} / ${accuracy.expired}`} />
            </div>
          )
        )}
      </section>

      <section>
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">History</h2>
          <div className="flex gap-2">
            <select value={pairFilter} onChange={(e) => setPairFilter(e.target.value)} className="select">
              <option value="">All pairs</option>
              {PAIRS.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
            <select value={profileFilter} onChange={(e) => setProfileFilter(e.target.value)} className="select">
              <option value="">All profiles</option>
              {PROFILES.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>
        </div>

        {error && (
          <p className="mt-4 rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {error}
          </p>
        )}
        {loading && !error && <p className="mt-4 text-sm text-zinc-500">Loading…</p>}
        {!loading && !error && signals.length === 0 && (
          <p className="mt-4 text-sm text-zinc-500">
            No signals yet. Ingest candle data, then generate a signal above.
          </p>
        )}

        <ul className="mt-4 flex flex-col gap-2">
          {signals.map((s) => {
            const key = s._id ?? `${s.pair}-${s.timestamp}`;
            const isOpen = expanded === key;
            return (
              <li
                key={key}
                className="rounded-lg border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900"
              >
                <button
                  onClick={() => setExpanded(isOpen ? null : key)}
                  className="flex w-full items-center justify-between gap-4 px-4 py-3 text-left"
                >
                  <div className="flex items-center gap-3">
                    <DirectionBadge direction={s.direction} />
                    <span className="text-sm font-medium">{s.pair}</span>
                    <span className="text-xs text-zinc-500">{s.interval} · {s.profile}</span>
                  </div>
                  <div className="flex items-center gap-3 text-xs text-zinc-500">
                    <span>{new Date(s.timestamp).toLocaleString()}</span>
                    <span>conf {s.confidence}%</span>
                    <StatusBadge status={s.status} />
                  </div>
                </button>
                {isOpen && (
                  <div className="border-t border-zinc-100 px-4 py-3 dark:border-zinc-800">
                    <p className="text-xs text-zinc-500">
                      Price at signal: {s.price_at_signal.toFixed(5)}
                      {s.outcome_price != null && ` · Outcome: ${s.outcome_price.toFixed(5)}`}
                      {s.outcome_pct_move != null && ` (${s.outcome_pct_move.toFixed(4)}%)`}
                    </p>
                    <ul className="mt-2 flex flex-col gap-1">
                      {s.reasons.map((r, i) => (
                        <li key={i} className="flex items-start gap-2 text-xs">
                          <span className={r.passed ? "text-emerald-600 dark:text-emerald-400" : "text-zinc-400"}>
                            {r.passed ? "✓" : "·"}
                          </span>
                          <span>
                            <span className="font-mono text-zinc-500">[{r.rule}]</span> {r.detail}
                            {r.value != null && (
                              <span className="ml-1 font-mono text-zinc-400">
                                ({parseFloat(r.value.toFixed(4))})
                              </span>
                            )}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </section>
    </div>
  );
}

function AccuracyStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-3 dark:border-zinc-800 dark:bg-zinc-800">
      <p className="text-xs text-zinc-500">{label}</p>
      <p className="mt-1 text-lg font-semibold">{value}</p>
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
