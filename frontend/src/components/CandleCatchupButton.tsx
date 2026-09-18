"use client";

import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { getToken } from "@/lib/auth";
import type { CandleCatchupJob } from "@/lib/types";

// Same polling cadence SyncNowButton/the RL page's train-all job use.
const POLL_MS = 3000;

// Manual "catch candles up to now" button -- separate from Sync now (which only tops up
// with a light output_size=5 ingest) and from the automatic keep-fresh.yml cron (same light
// top-up, every 20 min). This is for when that cron has gone quiet long enough (its own
// scheduler isn't always reliable under load, see CLAUDE.md) that the resulting gap is
// wider than /ingest's own single-call auto-widening can close -- catches every pair x
// interval combo up from wherever it last left off, all the way to now, respecting Twelve
// Data's rate limit (see data_fetcher.catch_up_batch), pressed ad hoc whenever needed.
export function CandleCatchupButton() {
  const [loggedIn, setLoggedIn] = useState(false);
  const [job, setJob] = useState<CandleCatchupJob | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    setLoggedIn(!!getToken());
  }, []);

  // Resumes polling if a catch-up kicked off earlier (this tab or another) is still
  // running -- same "don't lose visibility on refresh" behavior the RL/ML pages' jobs have.
  useEffect(() => {
    if (!loggedIn) return;
    (async () => {
      try {
        const latest = await api.getLatestCandleCatchupJob();
        if (latest && latest.status === "running") {
          setJob(latest);
          pollJob(latest.job_id);
        }
      } catch {
        // non-critical -- a failed resume check shouldn't block the button itself
      }
    })();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loggedIn]);

  if (!loggedIn) return null;

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
        if (j.status !== "running") stopPolling();
      } catch (e) {
        stopPolling();
        setError(e instanceof ApiError ? e.message : "Lost track of the job -- check the backend.");
      }
    }, POLL_MS);
  }

  async function handleClick() {
    setStarting(true);
    setError(null);
    try {
      const j = await api.startCandleCatchup();
      setJob(j);
      pollJob(j.job_id);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to start -- check the backend logs.");
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
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to cancel.");
    }
  }

  const running = job?.status === "running";

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2">
        <button
          onClick={handleClick}
          disabled={starting || running}
          title="Catches every pair x interval's candles up to now, from wherever ingestion last left off -- use this when the cron has been quiet for a while and the gap is too wide for a routine ingest to close on its own."
          className="rounded-md border border-zinc-300 px-2.5 py-1 text-xs font-medium text-zinc-600 hover:bg-zinc-100 disabled:opacity-50 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
        >
          {running ? `Catching up… (${job.completed}/${job.total})` : "Catch up candles"}
        </button>
        {running && (
          <button
            onClick={handleCancel}
            className="text-xs text-rose-600 underline underline-offset-2 hover:text-rose-700 dark:text-rose-400"
          >
            Cancel
          </button>
        )}
      </div>
      {job && job.status !== "running" && (
        <span className="text-xs text-zinc-500 dark:text-zinc-400">
          {summarize(job)}
        </span>
      )}
      {error && <span className="text-xs text-rose-600 dark:text-rose-400">{error}</span>}
    </div>
  );
}

function summarize(job: CandleCatchupJob): string {
  if (job.status === "cancelled") return `Cancelled at ${job.completed}/${job.total}.`;
  const errors = job.results.filter((r) => !r.ok).length;
  const totalCandles = job.results.reduce((sum, r) => sum + (r.candles_stored ?? 0), 0);
  return errors > 0
    ? `Done, ${errors} combo(s) had errors -- ${totalCandles} candles stored.`
    : `Done -- ${totalCandles} candles stored, all caught up.`;
}
