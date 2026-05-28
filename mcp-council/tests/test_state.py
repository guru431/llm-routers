"""Tests for the in-memory job state module."""

import asyncio
import time

import pytest

import state


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def reset_state():
    await state._reset_for_tests()
    yield
    await state._reset_for_tests()


async def test_create_job_returns_unique_ids():
    a = await state.create_job(question_preview="q1", synthesis=False, rounds=1)
    b = await state.create_job(question_preview="q2", synthesis=True, rounds=1)
    assert a.job_id != b.job_id
    assert a.phase == "queued"
    assert a.synthesis_requested is False
    assert b.synthesis_requested is True


async def test_get_job_round_trip():
    s = await state.create_job(question_preview="hello", synthesis=False, rounds=1)
    fetched = await state.get_job(s.job_id)
    assert fetched is s


async def test_get_job_missing_returns_none():
    assert await state.get_job("nope") is None


async def test_mark_phase_transitions_timestamps():
    s = await state.create_job(question_preview="q", synthesis=False, rounds=1)
    assert s.started_at is None and s.finished_at is None
    state.mark_phase(s, "stage1")
    assert s.started_at is not None and s.finished_at is None
    state.mark_phase(s, "done")
    assert s.finished_at is not None


async def test_update_member_stage1_replaces():
    s = await state.create_job(question_preview="q", synthesis=False, rounds=1)
    state.update_member_stage1(
        s, id="glm", model="glm-5.1", status="pending", error=None, latency_ms=None
    )
    assert s.stage1["glm"].status == "pending"
    state.update_member_stage1(
        s, id="glm", model="glm-5.1", status="ok", error=None, latency_ms=5000
    )
    assert s.stage1["glm"].status == "ok"
    assert s.stage1["glm"].latency_ms == 5000


async def test_snapshot_serializes_full_state():
    s = await state.create_job(question_preview="q", synthesis=True, rounds=1)
    state.mark_phase(s, "stage1")
    state.update_member_stage1(
        s, id="glm", model="glm-5.1", status="ok", error=None, latency_ms=4000
    )
    state.update_member_stage1(
        s, id="kimi", model="kimi-k2.6", status="error", error="boom", latency_ms=120000
    )
    state.update_stage3(
        s, id="deepseek-pro", model="deepseek-v4-pro", status="ok", error=None, latency_ms=30000
    )

    snap = state.snapshot(s)
    assert snap["job_id"] == s.job_id
    assert snap["phase"] == "stage1"
    assert snap["synthesis_requested"] is True
    assert snap["elapsed_ms"] is not None and snap["elapsed_ms"] >= 0
    ids = {m["id"] for m in snap["stage1"]}
    assert ids == {"glm", "kimi"}
    assert snap["stage3"]["id"] == "deepseek-pro"
    assert snap["has_result"] is False


async def test_cancel_job_cancels_running_task():
    s = await state.create_job(question_preview="q", synthesis=False, rounds=1)

    async def long_running():
        await asyncio.sleep(10)

    task = asyncio.create_task(long_running())
    state.attach_task(s, task)
    state.mark_phase(s, "stage1")

    ok = await state.cancel_job(s.job_id)
    assert ok is True
    assert s.phase == "cancelled"
    # Give the cancel a tick to propagate.
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cancel_job_unknown_returns_false():
    assert await state.cancel_job("nope") is False


async def test_cancel_job_finished_returns_false():
    s = await state.create_job(question_preview="q", synthesis=False, rounds=1)
    state.mark_phase(s, "done")
    assert await state.cancel_job(s.job_id) is False


async def test_list_jobs_newest_first():
    s1 = await state.create_job(question_preview="first", synthesis=False, rounds=1)
    # Force a tiny gap so created_at differs.
    s1.created_at = time.time() - 5
    s2 = await state.create_job(question_preview="second", synthesis=False, rounds=1)
    lst = await state.list_jobs(limit=10)
    ids = [j.job_id for j in lst]
    assert ids[0] == s2.job_id
    assert ids[1] == s1.job_id
