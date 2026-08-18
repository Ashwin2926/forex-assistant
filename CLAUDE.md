# Notes for Claude Code

Context and working notes for AI-assisted sessions on this repo, beyond what's in
`README.md` and `PROGRESS.md`.

## Architecture

FastAPI Python backend + Next.js frontend in one repo, MongoDB storage, Twelve Data for
candles, optional Deriv demo-account paper trading. Backend deploys to **FastAPI Cloud**
(`forex-assistant.fastapicloud.dev`), frontend to **Vercel**
(`forex-assistant-five.vercel.app`), both from `master`.

## Two platform gotchas that cost an entire debugging session to find (2026-08-18)

1. **FastAPI Cloud dashboard "redeploy"/restart does NOT rebuild from git.** It just
   restarts the existing already-built image. Confirmed by watching `deployment_id`
   change every few minutes with zero build logs on any of them, while runtime behavior
   never changed across ~6 different deployment IDs despite env var updates and code
   pushes. The only thing that actually rebuilds from source is their **CLI** deploy
   command, run from an authenticated terminal — this repo has no `pyproject.toml`, so
   the CLI setup wizard's "directory where pyproject.toml lives" prompt should be left
   **empty**, since `app/` and `requirements.txt` both live at the repo root. **Every
   backend code change requires a manual CLI deploy afterward — a git push alone changes
   nothing live.**
2. **GitHub Actions schedule triggers go silently dormant if the workflow YAML is
   invalid**, and this is hard to detect: the run shows under the raw file path instead
   of the workflow's declared `name:`, the Jobs API returns empty for it, and
   `workflow_dispatch` rejects manual triggers with "workflow does not have that
   trigger" — all because GitHub never fully parses the file. The actual error only
   shows in the web UI's run-detail page, not via the REST API. Root cause here was an
   unquoted `run: curl ... -H "X-Service-Token: $VAR" ...` — YAML disallows a bare `: `
   (colon-space) inside a plain scalar; fix is `run: |` block-literal style.

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
