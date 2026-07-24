# Backlog lifecycle

This project keeps a small, deliberate backlog of side-findings and forward-looking
ideas. The files split by **privacy** and **lifecycle stage**:

| File | Tracked? | Holds |
|---|---|---|
| `FINDINGS.md` (repo root) | **gitignored** (private) | only `open` side-findings; may reference internal paths/projects |
| `IDEAS.md` (repo root) | **gitignored** (private) | feature candidates: `proposed → accepted/rejected → done` |
| `FINDINGS-archive.md` (repo root) | **gitignored** (private) | findings deliberately *not* done (`wontfix`/`deferred` + rationale) |
| `IDEAS-archive.md` (repo root) | **gitignored** (private) | ideas deliberately *not* built (`wontfix`/`deferred`/`partial` + rationale) |

**Why:** this is a public repo, and all four files can reference internal hosts,
projects, or people — so all four stay local (`.gitignore` covers all four; the
archives are a LOCAL audit trail, not a tracked one). The archives hold only what
was consciously turned down; their job is to stop the same rejected item being
filed again (weekly auto-review is the main repeat offender). Completed work is
*not* archived — `git log` and the code are its record.

**Lifecycle rules** (canonical form lives in the global `CLAUDE.md`):

- New finding → prepend to `FINDINGS.md` with `**Status:** open`.
- Finding **done** → just delete it from `FINDINGS.md`. Do not archive it.
- Finding **rejected** → **first** prepend it to `FINDINGS-archive.md` (with
  `**Status:** wontfix|deferred` + `**Resolved:** YYYY-MM-DD — why not`), **then** delete
  it from `FINDINGS.md`. Append-before-delete so a crash leaves a dup, never a loss.
- Ideas follow the same flow: shipped → deleted, rejected → archived.
- Archive entries are never deleted.

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
