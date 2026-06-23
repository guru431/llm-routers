"""In-memory job store for the async-job pattern.

Lets `council_ask_async` start a long-running deliberation in the background
and return a job_id immediately, while the caller polls `council_status` and
later picks up the result with `council_result`.

State is per-process and lost on MCP server restart — fine for our use case
where a single deliberation lasts 2-8 minutes. Jobs older than TTL are pruned
opportunistically when a new job is created.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# Jobs older than this (since creation) are GC'd. 24h is enough that the same
# session can come back later and read its result, but not so long that an
# always-on server accumulates stale jobs.
JOB_TTL_SECONDS = 24 * 3600

# Soft cap on the number of jobs kept simultaneously. When exceeded, the oldest
# finished jobs are dropped first, then oldest running. Pet-project scale.
MAX_JOBS = 64

# Hard cap on concurrently ACTIVE (non-terminal) jobs. Each council run fans out
# to 6-7 upstream LLM calls (× rounds), so unbounded async jobs would exhaust the
# connection pool / provider rate limits and silently burn paid balances
# (DeepSeek PAYG, Exa, OCG). New jobs past this limit are rejected. Mirrors
# dialogue's MAX_ACTIVE_SESSIONS.
MAX_ACTIVE_JOBS = 16

TERMINAL_PHASES = frozenset({"done", "error", "cancelled", "interrupted"})

# Best-effort on-disk persistence so an MCP-server restart can surface jobs that
# were mid-flight ("interrupted, partial result available") instead of silently
# losing them. One JSON snapshot per job; written on every phase transition and
# (coalesced) on member progress. Override the location with COUNCIL_JOBS_DIR
# (read at call time so tests can isolate it).
_DEFAULT_PERSIST_DIR = Path(__file__).parent / "logs" / "jobs"

# Per-member progress fires ~50 times per 7-member 3-round run; on a network jobs
# disk every tmp+replace is a real round-trip. A recovery snapshot only needs to
# be approximately current, so member persists are coalesced to at most one write
# per this interval. Phase transitions and terminal/cancel persists always flush
# (force=True) so the final state is never lost — and since a write always dumps
# the full current snapshot, the next forced flush captures any coalesced-away
# member progress. _last_member_persist_at tracks only coalesced (member) writes
# per job_id, so a mandatory phase flush doesn't suppress the next member update;
# entries are dropped when the job is GC'd / its file is unlinked.
_MEMBER_PERSIST_MIN_INTERVAL = 2.0
_last_member_persist_at: dict[str, float] = {}


def _persist_dir() -> Path:
    return Path(os.environ.get("COUNCIL_JOBS_DIR") or _DEFAULT_PERSIST_DIR)


def _persist(state: "JobState", *, force: bool = True) -> None:
    """Write a job's current snapshot to disk. Best-effort — never raises into
    a running council. With force=False the member write is coalesced: skipped if
    the last member persist for this job was less than _MEMBER_PERSIST_MIN_INTERVAL
    ago (the latest progress is then picked up by the next forced flush)."""
    if not force:
        now = time.time()
        last = _last_member_persist_at.get(state.job_id)
        if last is not None and now - last < _MEMBER_PERSIST_MIN_INTERVAL:
            return
    try:
        d = _persist_dir()
        d.mkdir(parents=True, exist_ok=True)
        payload = snapshot(state)
        payload["result_markdown"] = state.result_markdown
        tmp = d / f"{state.job_id}.json.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(d / f"{state.job_id}.json")
        if not force:
            _last_member_persist_at[state.job_id] = time.time()
    except OSError:
        pass
    except TypeError as e:
        # A non-serializable usage/summary (e.g. an exception object leaked in)
        # would make json.dumps raise — skip the persist, never crash the run.
        print(
            f"[mcp-council] skipping job persist for {state.job_id}: "
            f"non-serializable payload ({e})",
            file=sys.stderr,
        )


def _unlink_persisted(job_id: str) -> None:
    _last_member_persist_at.pop(job_id, None)
    try:
        (_persist_dir() / f"{job_id}.json").unlink(missing_ok=True)
    except OSError:
        pass


def _state_from_snapshot(data: dict) -> "JobState":
    """Rebuild a JobState from a persisted snapshot. Non-terminal persisted
    phases become 'interrupted' (the run died with the previous process)."""
    state = JobState(
        job_id=data["job_id"],
        question_preview=data.get("question_preview", ""),
        created_at=data.get("created_at") or time.time(),
        synthesis_requested=bool(data.get("synthesis_requested")),
        rounds_requested=data.get("rounds_requested") or 1,
    )
    state.started_at = data.get("started_at")
    state.finished_at = data.get("finished_at")
    state.error = data.get("error")
    state.dump_path = data.get("dump_path")
    state.usage = data.get("usage")
    state.summary = data.get("summary")
    state.result_markdown = data.get("result_markdown")
    for m in data.get("stage1") or []:
        state.stage1[m["id"]] = MemberProgress(**m)
    for m in data.get("stage2") or []:
        state.stage2[m["id"]] = MemberProgress(**m)
    s3 = data.get("stage3")
    if s3:
        state.stage3 = MemberProgress(**s3)
    phase = data.get("phase") or "queued"
    if phase not in TERMINAL_PHASES:
        state.phase = "interrupted"
        state.finished_at = state.finished_at or time.time()
    else:
        state.phase = phase
    return state


def load_persisted_jobs() -> int:
    """Load persisted snapshots into memory at startup, marking non-terminal
    jobs as 'interrupted'. Returns the number of jobs loaded. Synchronous —
    intended to run once before the event loop starts serving."""
    d = _persist_dir()
    if not d.exists():
        return 0
    now = time.time()
    loaded = 0
    for f in d.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if now - (data.get("created_at") or 0) > JOB_TTL_SECONDS:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        jid = data.get("job_id")
        if not jid or jid in _jobs:
            continue
        # _state_from_snapshot does MemberProgress(**m) — a snapshot with
        # extra/changed keys raises TypeError (and KeyError on a missing "id").
        # One bad file must not down the whole server: skip it with a warning
        # and keep loading the rest. Mirrors load_persisted_dialogues.
        try:
            _jobs[jid] = _state_from_snapshot(data)
        except (KeyError, TypeError, ValueError) as e:
            print(
                f"[mcp-council] skipping unreadable job snapshot {f.name}: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            continue
        loaded += 1
    return loaded


@dataclass
class MemberProgress:
    """Per-member progress snapshot for one stage."""

    id: str
    model: str
    status: str = "pending"  # pending | ok | error
    error: str | None = None
    latency_ms: int | None = None


@dataclass
class JobState:
    job_id: str
    question_preview: str  # first ~120 chars of the question, for listings
    created_at: float
    synthesis_requested: bool
    rounds_requested: int

    # Top-level phase string, monotonic: queued → stage1 → stage2 → stage3 (opt)
    # → done; or → error / cancelled at any point.
    phase: str = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None

    # Per-stage live progress (updated by on_progress callback).
    stage1: dict[str, MemberProgress] = field(default_factory=dict)
    stage2: dict[str, MemberProgress] = field(default_factory=dict)
    stage3: MemberProgress | None = None

    # Final outputs (populated when phase=done).
    result_markdown: str | None = None
    dump_path: str | None = None
    usage: dict | None = None      # cost/usage accounting (see council._compute_usage)
    summary: dict | None = None    # machine-readable verdict (see council._build_summary)

    # Internal: handle to the running asyncio.Task so we can cancel.
    _task: asyncio.Task | None = field(default=None, repr=False)


_jobs: dict[str, JobState] = {}
_jobs_lock = asyncio.Lock()


def _gc_locked(now: float) -> None:
    """Caller must hold _jobs_lock. Drops jobs older than TTL, then if still
    over MAX_JOBS drops the oldest finished ones."""
    expired = [jid for jid, j in _jobs.items() if now - j.created_at > JOB_TTL_SECONDS]
    for jid in expired:
        del _jobs[jid]
        _unlink_persisted(jid)
    if len(_jobs) <= MAX_JOBS:
        return
    finished = sorted(
        ((jid, j) for jid, j in _jobs.items() if j.phase in TERMINAL_PHASES),
        key=lambda kv: kv[1].finished_at or kv[1].created_at,
    )
    while len(_jobs) > MAX_JOBS and finished:
        jid, _ = finished.pop(0)
        _jobs.pop(jid, None)
        _unlink_persisted(jid)


def _new_job_id() -> str:
    # Short uuid suffix; collision risk negligible at our scale.
    return f"job-{uuid.uuid4().hex[:12]}"


async def create_job(
    question_preview: str,
    *,
    synthesis: bool,
    rounds: int,
) -> JobState:
    """Allocate a JobState and register it."""
    async with _jobs_lock:
        now = time.time()
        _gc_locked(now)
        active = sum(1 for j in _jobs.values() if j.phase not in TERMINAL_PHASES)
        if active >= MAX_ACTIVE_JOBS:
            raise RuntimeError(
                f"too many active council jobs ({active}/{MAX_ACTIVE_JOBS}); "
                "wait for some to finish or call council_cancel on stale ones"
            )
        jid = _new_job_id()
        state = JobState(
            job_id=jid,
            question_preview=question_preview[:120],
            created_at=now,
            synthesis_requested=synthesis,
            rounds_requested=rounds,
        )
        _jobs[jid] = state
        return state


async def get_job(job_id: str) -> JobState | None:
    async with _jobs_lock:
        return _jobs.get(job_id)


async def list_jobs(limit: int = 20) -> list[JobState]:
    async with _jobs_lock:
        items = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
        return items[:limit]


async def active_job_count() -> int:
    """Number of non-terminal (queued/running) jobs — for surfacing the
    MAX_ACTIVE_JOBS budget in council_status."""
    async with _jobs_lock:
        return sum(1 for j in _jobs.values() if j.phase not in TERMINAL_PHASES)


async def cancel_job(job_id: str) -> bool:
    """Request cancellation. Returns True if the job existed and was running.

    Race protection: instead of flipping `phase = 'cancelled'` immediately we
    cancel the task and give it a short window to settle. The task may
    finish successfully in the millisecond between our `task.cancel()` and
    the exception being delivered — in that case its `mark_phase('done')` +
    `result_markdown` win and we keep them. Only flip to 'cancelled' if the
    task either failed to handle CancelledError or didn't reach a terminal
    phase on its own.
    """
    async with _jobs_lock:
        j = _jobs.get(job_id)
        if j is None:
            return False
        if j.phase in TERMINAL_PHASES:
            return False
        task = j._task
    if task is not None and not task.done():
        task.cancel()
        # Brief drain so any CancelledError handler can run and flip phase.
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass
    if j.phase not in TERMINAL_PHASES:
        # No handler ran (e.g. bare coroutine in tests) or it ran but didn't
        # call mark_phase — finalize synchronously.
        j.phase = "cancelled"
        j.finished_at = time.time()
        _persist(j)
    return True


def attach_task(state: JobState, task: asyncio.Task) -> None:
    """Bind the background asyncio.Task to the job (caller has the state ref)."""
    state._task = task


def mark_phase(state: JobState, phase: str) -> None:
    state.phase = phase
    if state.started_at is None and phase != "queued":
        state.started_at = time.time()
    if phase in TERMINAL_PHASES:
        state.finished_at = time.time()
    _persist(state)


def update_member_stage1(state: JobState, *, id: str, model: str, status: str, error: str | None, latency_ms: int | None) -> None:
    state.stage1[id] = MemberProgress(
        id=id, model=model, status=status, error=error, latency_ms=latency_ms
    )
    _persist(state, force=False)


def update_member_stage2(state: JobState, *, id: str, model: str, status: str, error: str | None, latency_ms: int | None) -> None:
    state.stage2[id] = MemberProgress(
        id=id, model=model, status=status, error=error, latency_ms=latency_ms
    )
    _persist(state, force=False)


def update_stage3(state: JobState, *, id: str, model: str, status: str, error: str | None, latency_ms: int | None) -> None:
    state.stage3 = MemberProgress(
        id=id, model=model, status=status, error=error, latency_ms=latency_ms
    )
    _persist(state, force=False)


def snapshot(state: JobState) -> dict:
    """Plain-dict view of a JobState for serialization to MCP clients."""
    now = time.time()
    elapsed_ms = None
    if state.started_at is not None:
        end = state.finished_at if state.finished_at is not None else now
        elapsed_ms = int((end - state.started_at) * 1000)

    return {
        "job_id": state.job_id,
        "question_preview": state.question_preview,
        "phase": state.phase,
        "synthesis_requested": state.synthesis_requested,
        "rounds_requested": state.rounds_requested,
        "created_at": state.created_at,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
        "elapsed_ms": elapsed_ms,
        "error": state.error,
        "stage1": [
            {
                "id": m.id, "model": m.model, "status": m.status,
                "error": m.error, "latency_ms": m.latency_ms,
            }
            for m in state.stage1.values()
        ],
        "stage2": [
            {
                "id": m.id, "model": m.model, "status": m.status,
                "error": m.error, "latency_ms": m.latency_ms,
            }
            for m in state.stage2.values()
        ],
        "stage3": (
            {
                "id": state.stage3.id, "model": state.stage3.model,
                "status": state.stage3.status, "error": state.stage3.error,
                "latency_ms": state.stage3.latency_ms,
            }
            if state.stage3 is not None
            else None
        ),
        "has_result": state.result_markdown is not None,
        "dump_path": state.dump_path,
        "usage": state.usage,
        "summary": state.summary,
    }


# Test-only entry point: clear the store between tests.
async def _reset_for_tests() -> None:
    async with _jobs_lock:
        for j in _jobs.values():
            if j._task is not None and not j._task.done():
                j._task.cancel()
        _jobs.clear()
    _last_member_persist_at.clear()
    try:
        for f in _persist_dir().glob("*.json*"):
            f.unlink(missing_ok=True)
    except OSError:
        pass
