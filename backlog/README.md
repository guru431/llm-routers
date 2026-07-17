# Backlog lifecycle

This project keeps a small, deliberate backlog of side-findings and forward-looking
ideas. The files split by **privacy** and **lifecycle stage**:

| File | Tracked? | Holds |
|---|---|---|
| `FINDINGS.md` (repo root) | **gitignored** (private) | only `open` side-findings; may reference internal paths/projects |
| `IDEAS.md` (repo root) | **gitignored** (private) | feature candidates: `proposed → accepted/rejected → done` |
| `FINDINGS-archive.md` (repo root) | tracked | audit trail of closed findings (`done`/`wontfix` + resolution) |
| `IDEAS-archive.md` (repo root) | tracked | audit trail of decided ideas (`done`/`rejected` + rationale) |

**Why the split:** this is a public repo. Open notes can reference internal hosts,
projects, or people, so they stay local (gitignored). Once an item is *closed* it is
rewritten to a publishable form and moved to the tracked `*-archive.md` — that file is
the durable, reviewable record of "what was actually decided and why".

**Lifecycle rules** (canonical form lives in the global `CLAUDE.md`):

- New finding → prepend to `FINDINGS.md` with `**Status:** open`.
- Closing a finding → **first** prepend it to `FINDINGS-archive.md` (with
  `**Status:** done|wontfix` + `**Resolved:** YYYY-MM-DD — …`), **then** delete it
  from `FINDINGS.md`. Append-before-delete so a crash leaves a dup, never a loss.
- Ideas follow the same append-to-archive-then-remove flow.
- Archive entries are never deleted — they are the audit trail.

## Status manifest + stale reminder

`scripts/backlog_manifest.py` scans all four files, writes `backlog/manifest.json`
(gitignored — it echoes private titles), and flags any still-open item older than the
stale horizon (default 90 days):

```
python scripts/backlog_manifest.py                 # summary + stale list
python scripts/backlog_manifest.py --json-only      # machine-readable
python scripts/backlog_manifest.py --fail-on-stale  # exit 2 if stale (for cron alerts)
```

Wire the `--fail-on-stale` form into MonthlyStratReview / a scheduled job to get a
reminder when an open finding or idea has been sitting untouched for >90 days.

## ADRs

Larger architectural decisions live under `docs/` (gitignored — internal design
notes). When an ADR affects the public contract, summarize the decision in the
relevant tracked README or in `*-archive.md` so the public record stays coherent
without publishing the internal note verbatim.
