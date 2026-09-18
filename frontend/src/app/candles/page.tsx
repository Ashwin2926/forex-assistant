"use client";

import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { CandleCatchupJob } from "@/lib/types";

// Same polling cadence the other job-backed pages (ML rebuild, RL train-all) use.
const POLL_MS = 3000;

export default function CandlesPage() {
  const [job, setJob] = useState<CandleCatchupJob | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [history, setHistory] = useState<CandleCatchupJob[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);

  async function loadHistory() {
    setHistoryLoading(true);
    try {
      setHistory(await api.listCandleCatchupJobs(20));
    } catch {
      // non-critical -- a failed history load shouldn't block the trigger button
    } finally {
      setHistoryLoading(false);
    }
  }

  // Resumes polling if a catch-up kicked off earlier (this tab, another tab, or the
  // sidebar button before this page existed) is still running -- same "don't lose
  // visibility on reload" behavior every other job-backed page here has.
  useEffect(() => {
    loadHistory();
    (async () => {
      try {
        const latest = await api.getLatestCandleCatchupJob();
        if (latest) {
          setJob(latest);
          if (latest.status === "running") pollJob(latest.job_id);
        }
      } catch {
        // non-critical section
      }
    })();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function stopPolling() {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }

  function pollJob(jobId: string) {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const j = await api.getCandleCatchupJob(jobId);
        setJob(j);
        if (j.status !== "running") {
          stopPolling();
          await loadHistory();
        }
      } catch (e) {
        stopPolling();
        setStartError(e instanceof ApiError ? e.message : "Lost track of the job -- check the backend.");
      }
    }, POLL_MS);
  }

  async function handleStart() {
    setStarting(true);
    setStartError(null);
    try {
      const j = await api.startCandleCatchup();
      setJob(j);
      pollJob(j.job_id);
      await loadHistory();
    } catch (e) {
      setStartError(e instanceof ApiError ? e.message : "Failed to start -- check the backend logs.");
    } finally {
      setStarting(false);
    }
  }

  async function handleCancel() {
    if (!job) return;
    try {
      const j = await api.cancelCandleCatchupJob(job.job_id);
      setJob(j);
      stopPolling();
      await loadHistory();
    } catch (e) {
      setStartError(e instanceof ApiError ? e.message : "Failed to cancel.");
    }
  }

  const running = job?.status === "running";

  return (
    <div className="flex flex-col gap-8">
      <section>
        <h1 className="text-xl font-semibold tracking-tight">Candles</h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          Catches every pair &times; interval&apos;s candles up to now, from wherever ingestion
          last left off — for when the ingestion cron (keep-fresh.yml, every 20 min) has gone
          quiet long enough that the resulting gap is wider than a routine ingest&apos;s own
          single-call auto-widening can close. Paced by Twelve Data&apos;s rate limit and
          resumable, so it&apos;s safe to press whenever, however often.
        </p>
      </section>

      <section className="card p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Catch up now</h2>
          <div className="flex items-center gap-2">
            <button onClick={handleStart} disabled={starting || running} className="btn-primary shrink-0">
              {running ? `Catching up… (${job.completed}/${job.total})` : "Catch up candles"}
            </button>
            {running && (
              <button
                onClick={handleCancel}
                className="rounded-md border border-rose-300 px-3 py-1.5 text-xs font-medium text-rose-700 hover:bg-rose-50 dark:border-rose-800 dark:text-rose-300 dark:hover:bg-rose-950"
              >
                Stop
              </button>
            )}
          </div>
        </div>
        {startError && (
          <p className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {startError}
          </p>
        )}
        {job && job.status !== "running" && (
          <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">{summarize(job)}</p>
        )}

        {job && job.results.length > 0 && (
          <div className="mt-4 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
            <table className="w-full text-left text-xs">
              <thead className="table-head uppercase">
                <tr>
                  <th className="px-3 py-1.5">Pair</th>
                  <th className="px-3 py-1.5">Interval</th>
                  <th className="px-3 py-1.5">Calls made</th>
                  <th className="px-3 py-1.5">Candles stored</th>
                  <th className="px-3 py-1.5">Status</th>
                </tr>
              </thead>
              <tbody>
                {[...job.results].reverse().map((r, i) => (
                  <tr key={`${r.pair}-${r.interval}-${i}`} className="border-t border-zinc-100 dark:border-zinc-800">
                    <td className="px-3 py-1.5 font-mono">{r.pair}</td>
                    <td className="px-3 py-1.5 font-mono">{r.interval}</td>
                    <td className="px-3 py-1.5">{r.calls_made ?? "—"}</td>
                    <td className="px-3 py-1.5">{r.candles_stored ?? "—"}</td>
                    <td className="px-3 py-1.5">
                      {r.ok ? (
                        <span className={r.caught_up ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600 dark:text-amber-400"}>
                          {r.caught_up ? "caught up" : "partial"}
                        </span>
                      ) : (
                        <span className="text-rose-600 dark:text-rose-400" title={r.error ?? undefined}>
                          error
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section>
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Recent jobs</h2>
        {historyLoading && <p className="mt-4 text-sm text-zinc-500">Loading…</p>}
        {!historyLoading && history.length === 0 && (
          <p className="mt-4 text-sm text-zinc-500">No catch-up jobs run yet.</p>
        )}
        {history.length > 0 && (
          <div className="mt-4 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
            <table className="w-full text-left text-sm">
              <thead className="table-head text-xs uppercase">
                <tr>
                  <th className="px-4 py-2">Job</th>
                  <th className="px-4 py-2">Status</th>
                  <th className="px-4 py-2">Progress</th>
                  <th className="px-4 py-2">Candles stored</th>
                  <th className="px-4 py-2">Started</th>
                </tr>
              </thead>
              <tbody>
                {history.map((j) => (
                  <tr key={j.job_id} className="border-t border-zinc-100 dark:border-zinc-800">
                    <td className="px-4 py-2 font-mono text-xs">{j.job_id}</td>
                    <td className="px-4 py-2">
                      <span
                        className={
                          j.status === "running"
                            ? "text-sky-600 dark:text-sky-400"
                            : j.status === "cancelled"
                              ? "text-amber-600 dark:text-amber-400"
                              : "text-zinc-500 dark:text-zinc-400"
                        }
                      >
                        {j.status}
                      </span>
                    </td>
                    <td className="px-4 py-2">{j.completed}/{j.total}</td>
                    <td className="px-4 py-2">
                      {j.results.reduce((sum, r) => sum + (r.candles_stored ?? 0), 0)}
                    </td>
                    <td className="px-4 py-2 text-xs text-zinc-500">{new Date(j.created_at).toLocaleString()}</td>
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

function summarize(job: CandleCatchupJob): string {
  if (job.status === "cancelled") return `Cancelled at ${job.completed}/${job.total}.`;
  const errors = job.results.filter((r) => !r.ok).length;
  const totalCandles = job.results.reduce((sum, r) => sum + (r.candles_stored ?? 0), 0);
  return errors > 0
    ? `Done, ${errors} combo(s) had errors — ${totalCandles} candles stored.`
    : `Done — ${totalCandles} candles stored, all caught up.`;
}
