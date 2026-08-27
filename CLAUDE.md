# Notes for Claude Code

Context and working notes for AI-assisted sessions on this repo, beyond what's in
`README.md` and `PROGRESS.md`.

## Architecture

FastAPI Python backend + Next.js frontend in one repo, MongoDB storage, Twelve Data for
candles, optional Deriv demo-account paper trading. Backend deploys to **FastAPI Cloud**
(`forex-assistant.fastapicloud.dev`), frontend to **Vercel**
(`forex-assistant-five.vercel.app`), both from `master`.

## Two platform gotchas that cost an entire debugging session to find (2026-08-18)

1. ~~**FastAPI Cloud dashboard "redeploy"/restart does NOT rebuild from git...**~~
   **Superseded 2026-08-27**: a plain `git push` to `master` now DOES trigger a real
   rebuild+redeploy on FastAPI Cloud — confirmed twice in the same session (new response
   fields showed up live within ~1-2 min of pushing, no CLI step run). Whatever the
   dashboard-restart-only limitation was on 2026-08-18, it no longer applies as of this
   date, or FastAPI Cloud added push-triggered deploys since. **Don't assume a CLI deploy
   is required going forward** — but if a push ever again produces no live change, check
   this assumption before spending a session re-debugging it, since it's plausible this
   could regress or the platform's behavior could vary.
2. **GitHub Actions schedule triggers go silently dormant if the workflow YAML is
   invalid**, and this is hard to detect: the run shows under the raw file path instead
   of the workflow's declared `name:`, the Jobs API returns empty for it, and
   `workflow_dispatch` rejects manual triggers with "workflow does not have that
   trigger" — all because GitHub never fully parses the file. The actual error only
   shows in the web UI's run-detail page, not via the REST API. Root cause here was an
   unquoted `run: curl ... -H "X-Service-Token: $VAR" ...` — YAML disallows a bare `: `
   (colon-space) inside a plain scalar; fix is `run: |` block-literal style.

**Related, milder finding (2026-08-27)**: even with valid YAML, `keep-fresh.yml`'s
`*/20 * * * *` schedule doesn't fire exactly on time — pulled 100 real run timestamps via
`gh run list`, median gap was 23.6 min (close enough) but 45/99 gaps exceeded 25 min and one
hit 3.5 hours. This is GitHub's own documented behavior for scheduled workflows under load,
not a bug to fix here — but worth knowing before assuming a cron-dependent job (candle
ingestion, signal scoring) ran as recently as the schedule implies.

**Vercel quirk**: frontend builds can land as `Preview` without auto-promoting to
`Production` — may need a manual "Production rebuild" click even after a successful
build.

## Working with this project's owner

- Prefers direct execution once a direction is authorized (even loosely — "yes", "add
  X") over repeated confirmation before each individual git push or deploy step. Wants
  the actual change committed, pushed to `master`, and verified end-to-end (dispatch a
  workflow and poll for it, hit an endpoint and check the response) in the same turn,
  not just asserted as done.
- Still wants genuinely risky/hard-to-reverse states flagged in the same breath as
  acting on them (e.g. "this makes the live backend fully open until re-enabled") —
  surface the risk AND do the thing, not one instead of the other.
- Replies are often short (single words, minimal punctuation) — treat that as normal
  signal density for this collaborator, not as ambiguity to double-check.
