"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { getToken } from "@/lib/auth";

// How long the "Dispatched" / error confirmation stays visible before clearing itself --
// long enough to read, short enough not to linger stale after the next click.
const MESSAGE_MS = 5000;

// Manual trigger for one real .github/workflows/keep-fresh.yml cron cycle -- the lighter
// sibling of SyncNowButton (which also retrains all 20 RL combos in-process, ~20-30 min).
// This just re-fires the same ~1-6 min cycle the */20 schedule already runs (ingest,
// consensus check, RL signals, scoring, ML retrain) for when that schedule has gone quiet
// (GitHub's own scheduled triggers aren't always reliable under load -- confirmed live
// 2026-08-27 and again 2026-09-20, see PROGRESS.md) and a fast top-up is enough, without
// paying for a full RL retrain that hasn't changed since the last one.
//
// Fire-and-forget by design (see POST /ops/keep-fresh's own docstring) -- there's no job_id
// to poll, so this only ever reports whether the dispatch itself succeeded, not whether the
// resulting Actions run finished. Check the repo's Actions tab for that.
export function RefreshNowButton() {
  const [loggedIn, setLoggedIn] = useState(false);
  const [status, setStatus] = useState<"idle" | "dispatching" | "done" | "error">("idle");
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    setLoggedIn(!!getToken());
  }, []);

  useEffect(() => {
    if (!message) return;
    const id = setTimeout(() => setMessage(null), MESSAGE_MS);
    return () => clearTimeout(id);
  }, [message]);

  if (!loggedIn) return null;

  async function handleClick() {
    setStatus("dispatching");
    setMessage(null);
    try {
      await api.triggerKeepFresh();
      setStatus("done");
      setMessage("Dispatched -- check the Actions tab for progress.");
    } catch (e) {
      setStatus("error");
      setMessage(e instanceof ApiError ? e.message : "Failed to dispatch -- check the backend logs.");
    }
  }

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={handleClick}
        disabled={status === "dispatching"}
        title="Manually fires one keep-fresh.yml cycle right now (ingest, check consensus, generate RL signals, score, retrain ML) -- faster than Sync now since it skips the full RL retrain."
        className="rounded-md border border-zinc-300 px-2.5 py-1 text-xs font-medium text-zinc-600 hover:bg-zinc-100 disabled:opacity-50 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
      >
        {status === "dispatching" ? "Dispatching…" : "Refresh now"}
      </button>
      {message && (
        <span className={`text-xs ${status === "error" ? "text-rose-600 dark:text-rose-400" : "text-zinc-500 dark:text-zinc-400"}`}>
          {message}
        </span>
      )}
    </div>
  );
}
