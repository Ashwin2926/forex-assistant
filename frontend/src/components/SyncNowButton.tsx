"use client";

import { useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { getToken } from "@/lib/auth";
import type { RunAllFlowsResult } from "@/lib/types";

// How often to poll GET /ops/run-all-flows/{job_id} while the job runs -- same cadence as
// the RL page's "train all" polling (frequent enough to feel live, not frequent enough to
// matter for load); the job itself typically finishes well under a minute since it excludes
// RL training, but can take longer on a Twelve Data hiccup or a cold backend instance.
const POLL_MS = 3000;

// Manual "catch up now" button for when the GitHub Actions cron (.github/workflows/
// keep-fresh.yml) has gone quiet for a while -- its own scheduler isn't always reliable
// under load (confirmed live 2026-08-27, a ~5hr gap with zero runs despite an active
// */20 schedule, see CLAUDE.md). Runs as a background job (POST /ops/run-all-flows returns
// a job_id immediately, this polls it) rather than a synchronous call -- the full sequence
// can take well over Cloudflare's ~100s proxy timeout, the same 524 this project already
// hit with a single /rl/train call, so waiting on one synchronous response would just be a
// bigger version of that same problem.
export function SyncNowButton() {
  const [loggedIn] = useState(() => !!getToken());
  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [summary, setSummary] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  if (!loggedIn) return null;

  function stopPolling() {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }

  async function handleClick() {
    setStatus("running");
    setSummary(null);
    try {
      const { job_id } = await api.startRunAllFlows();
      pollRef.current = setInterval(async () => {
        try {
          const job = await api.getRunAllFlowsJob(job_id);
          if (job.status === "done") {
            stopPolling();
            setSummary(summarize(job.results as RunAllFlowsResult));
            setStatus("done");
          }
        } catch (e) {
          stopPolling();
          setSummary(e instanceof ApiError ? e.message : "Lost track of the job -- check the backend.");
          setStatus("error");
        }
      }, POLL_MS);
    } catch (e) {
      setSummary(e instanceof ApiError ? e.message : "Failed to start -- check the backend logs.");
      setStatus("error");
    }
  }

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={handleClick}
        disabled={status === "running"}
        title="Manually runs one full cron cycle (ingest, generate signals, score, retrain ML) right now -- use this when the cron hasn't run in a while instead of waiting on it."
        className="rounded-md border border-zinc-300 px-2.5 py-1 text-xs font-medium text-zinc-600 hover:bg-zinc-100 disabled:opacity-50 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
      >
        {status === "running" ? "Syncing…" : "Sync now"}
      </button>
      {summary && (
        <span className={`text-xs ${status === "error" ? "text-rose-600 dark:text-rose-400" : "text-zinc-500 dark:text-zinc-400"}`}>
          {summary}
        </span>
      )}
    </div>
  );
}

function summarize(results: RunAllFlowsResult): string {
  const countErrors = (obj: Record<string, unknown>) =>
    Object.values(obj ?? {}).filter((v) => typeof v === "string" && v.startsWith("error:")).length;
  const errors =
    countErrors(results.signals) + countErrors(results.consensus) + countErrors(results.rl_signals);
  return errors > 0 ? `Done, ${errors} step(s) had errors -- see /ops/run-all-flows for detail.` : "Done -- all steps ok.";
}
