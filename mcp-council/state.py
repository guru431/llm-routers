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
import time
import uuid
from dataclasses import dataclass, field

# Jobs older than this (since creation) are GC'd. 24h is enough that the same
# session can come back later and read its result, but not so long that an
# always-on server accumulates stale jobs.
JOB_TTL_SECONDS = 24 * 3600

# Soft cap on the number of jobs kept simultaneously. When exceeded, the oldest
# finished jobs are dropped first, then oldest running. Pet-project scale.
MAX_JOBS = 64


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
    if len(_jobs) <= MAX_JOBS:
        return
    finished = sorted(
        ((jid, j) for jid, j in _jobs.items() if j.phase in {"done", "error", "cancelled"}),
        key=lambda kv: kv[1].finished_at or kv[1].created_at,
    )
    while len(_jobs) > MAX_JOBS and finished:
        jid, _ = finished.pop(0)
        _jobs.pop(jid, None)


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
        if j.phase in {"done", "error", "cancelled"}:
            return False
        task = j._task
    if task is not None and not task.done():
        task.cancel()
        # Brief drain so any CancelledError handler can run and flip phase.
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass
    if j.phase not in {"done", "error", "cancelled"}:
        # No handler ran (e.g. bare coroutine in tests) or it ran but didn't
        # call mark_phase — finalize synchronously.
        j.phase = "cancelled"
        j.finished_at = time.time()
    return True


def attach_task(state: JobState, task: asyncio.Task) -> None:
    """Bind the background asyncio.Task to the job (caller has the state ref)."""
    state._task = task


def mark_phase(state: JobState, phase: str) -> None:
    state.phase = phase
    if state.started_at is None and phase != "queued":
        state.started_at = time.time()
    if phase in {"done", "error", "cancelled"}:
        state.finished_at = time.time()


def update_member_stage1(state: JobState, *, id: str, model: str, status: str, error: str | None, latency_ms: int | None) -> None:
    state.stage1[id] = MemberProgress(
        id=id, model=model, status=status, error=error, latency_ms=latency_ms
    )


def update_member_stage2(state: JobState, *, id: str, model: str, status: str, error: str | None, latency_ms: int | None) -> None:
    state.stage2[id] = MemberProgress(
        id=id, model=model, status=status, error=error, latency_ms=latency_ms
    )


def update_stage3(state: JobState, *, id: str, model: str, status: str, error: str | None, latency_ms: int | None) -> None:
    state.stage3 = MemberProgress(
        id=id, model=model, status=status, error=error, latency_ms=latency_ms
    )


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
    }


# Test-only entry point: clear the store between tests.
async def _reset_for_tests() -> None:
    async with _jobs_lock:
        for j in _jobs.values():
            if j._task is not None and not j._task.done():
                j._task.cancel()
        _jobs.clear()
