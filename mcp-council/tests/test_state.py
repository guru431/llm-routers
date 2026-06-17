"""Tests for the in-memory job state module."""

import asyncio
import time

import pytest

import state


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def reset_state(tmp_path, monkeypatch):
    # Isolate on-disk job persistence into a tmp dir so tests never touch logs/.
    monkeypatch.setenv("COUNCIL_JOBS_DIR", str(tmp_path / "jobs"))
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


# --- persistence / recovery ---------------------------------------------


async def test_persist_and_recover_interrupted_job():
    s = await state.create_job(question_preview="q", synthesis=False, rounds=1)
    state.mark_phase(s, "stage1")
    state.update_member_stage1(
        s, id="glm", model="glm-5.1", status="ok", error=None, latency_ms=1000
    )
    # Simulate a server restart: drop in-memory state, keep the on-disk file.
    state._jobs.clear()
    loaded = state.load_persisted_jobs()
    assert loaded == 1
    recovered = await state.get_job(s.job_id)
    assert recovered is not None
    assert recovered.phase == "interrupted"     # non-terminal → interrupted
    assert "glm" in recovered.stage1            # partial progress preserved


async def test_persist_recover_done_job_keeps_phase_and_result():
    s = await state.create_job(question_preview="q", synthesis=False, rounds=1)
    s.result_markdown = "# final"
    state.mark_phase(s, "done")
    state._jobs.clear()
    assert state.load_persisted_jobs() == 1
    recovered = await state.get_job(s.job_id)
    assert recovered.phase == "done"            # terminal phase preserved
    assert recovered.result_markdown == "# final"


async def test_load_persisted_skips_in_memory_duplicates():
    s = await state.create_job(question_preview="q", synthesis=False, rounds=1)
    state.mark_phase(s, "stage1")
    # File exists AND job is in memory → load must not clobber the live one.
    loaded = state.load_persisted_jobs()
    assert loaded == 0
    assert (await state.get_job(s.job_id)).phase == "stage1"


def _read_persisted(job_id: str) -> dict:
    import json
    return json.loads((state._persist_dir() / f"{job_id}.json").read_text(encoding="utf-8"))


async def test_member_persists_coalesced_within_interval():
    # Rapid member updates within the min-interval write disk at most once; a
    # later forced flush (phase transition) still captures the latest progress.
    s = await state.create_job(question_preview="q", synthesis=False, rounds=1)
    state.mark_phase(s, "stage1")  # forced flush; does not arm member throttle
    # First member write goes through (no prior member persist).
    state.update_member_stage1(
        s, id="glm", model="glm-5.1", status="ok", error=None, latency_ms=1000
    )
    on_disk_ids = {m["id"] for m in _read_persisted(s.job_id)["stage1"]}
    assert on_disk_ids == {"glm"}
    # Second member arrives immediately → coalesced (skipped), file unchanged.
    state.update_member_stage1(
        s, id="kimi", model="kimi-k2.6", status="ok", error=None, latency_ms=1200
    )
    on_disk_ids = {m["id"] for m in _read_persisted(s.job_id)["stage1"]}
    assert on_disk_ids == {"glm"}
    # A phase transition forces a flush → coalesced-away member now on disk.
    state.mark_phase(s, "stage2")
    disk = _read_persisted(s.job_id)
    assert disk["phase"] == "stage2"
    assert {m["id"] for m in disk["stage1"]} == {"glm", "kimi"}


async def test_terminal_state_always_flushed():
    # The final (terminal) state must always be persisted, never coalesced away.
    s = await state.create_job(question_preview="q", synthesis=False, rounds=1)
    state.mark_phase(s, "stage1")
    state.update_member_stage1(
        s, id="glm", model="glm-5.1", status="ok", error=None, latency_ms=1000
    )
    # Immediately mark done (within the member interval) — forced flush wins.
    s.result_markdown = "# final"
    state.mark_phase(s, "done")
    disk = _read_persisted(s.job_id)
    assert disk["phase"] == "done"
    assert disk["result_markdown"] == "# final"
