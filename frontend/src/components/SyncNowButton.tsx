"use client";

import { useState } from "react";
import { api, ApiError } from "@/lib/api";
import { getToken } from "@/lib/auth";
import type { RunAllFlowsResult } from "@/lib/types";

// Manual "catch up now" button for when the GitHub Actions cron (.github/workflows/
// keep-fresh.yml) has gone quiet for a while -- its own scheduler isn't always reliable
// under load (confirmed live 2026-08-27, a ~5hr gap with zero runs despite an active
// */20 schedule, see CLAUDE.md). Calls POST /ops/run-all-flows, which replicates one full
// cron cycle (ingest, generate signals, score, retrain ML) across every pair/interval,
// synchronously -- deliberately excludes RL training (see that endpoint's own docstring),
// so this stays reasonably fast, but "reasonably fast" here is still many sequential calls
// and can take a minute or more; a Cloudflare proxy timeout on a very slow run would look
// like a failed request even though the work already completed server-side (each step
// commits its own result as it goes, nothing here is transactional across the whole call).
export function SyncNowButton() {
  const [loggedIn] = useState(() => !!getToken());
  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [summary, setSummary] = useState<string | null>(null);

  if (!loggedIn) return null;

  async function handleClick() {
    setStatus("running");
    setSummary(null);
    try {
      const result = await api.runAllFlows();
      setSummary(summarize(result));
      setStatus("done");
    } catch (e) {
      setSummary(e instanceof ApiError ? e.message : "Failed -- check the backend logs.");
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
        {status === "running" ? "Syncing… (can take a minute+)" : "Sync now"}
      </button>
      {summary && (
        <span className={`text-xs ${status === "error" ? "text-rose-600 dark:text-rose-400" : "text-zinc-500 dark:text-zinc-400"}`}>
          {summary}
        </span>
      )}
    </div>
  );
}

function summarize(result: RunAllFlowsResult): string {
  const countErrors = (obj: Record<string, unknown>) =>
    Object.values(obj).filter((v) => typeof v === "string" && v.startsWith("error:")).length;
  const errors =
    countErrors(result.signals) + countErrors(result.consensus) + countErrors(result.rl_signals);
  return errors > 0 ? `Done, ${errors} step(s) had errors -- see response for detail.` : "Done -- all steps ok.";
}
