"""Tests for dialogue.state — DialogueState lifecycle, snapshot, cancel, caps."""

import asyncio
import time

import pytest

from dialogue import state as dialogue_state
from dialogue.state import (
    DialogueState,
    MAX_ACTIVE_SESSIONS,
    INACTIVE_TIMEOUT_SECONDS,
    cancel_session,
    create_session,
    get_session,
    list_sessions,
    mark_phase,
    reserve_active_slot,
    snapshot,
)


async def test_create_session_returns_state_with_unique_id():
    s1 = await create_session(mode="debate", question_preview="why X?", total_rounds=5)
    s2 = await create_session(mode="panel", question_preview="why Y?", total_rounds=3)
    assert s1.session_id.startswith("dlg-")
    assert s1.session_id != s2.session_id
    assert s1.mode == "debate"
    assert s1.total_rounds == 5
    assert s1.phase == "starting"
    assert s1.current_round == 0


async def test_get_session_returns_none_for_unknown():
    assert await get_session("dlg-nope") is None


async def test_get_session_returns_state_after_create():
    s = await create_session(mode="socratic", question_preview="q", total_rounds=2)
    got = await get_session(s.session_id)
    assert got is s


async def test_list_sessions_newest_first():
    s1 = await create_session(mode="debate", question_preview="a", total_rounds=1)
    await asyncio.sleep(0.01)
    s2 = await create_session(mode="panel", question_preview="b", total_rounds=1)
    items = await list_sessions(limit=10)
    assert [s.session_id for s in items] == [s2.session_id, s1.session_id]


async def test_mark_phase_updates_timestamps():
    s = await create_session(mode="debate", question_preview="q", total_rounds=1)
    assert s.started_at is None
    mark_phase(s, "round_1_critique")
    assert s.started_at is not None
    assert s.phase == "round_1_critique"
    mark_phase(s, "done")
    assert s.finished_at is not None


async def test_cancel_session_running_marks_cancelled():
    s = await create_session(mode="debate", question_preview="q", total_rounds=1)
    mark_phase(s, "round_1_critique")
    ok = await cancel_session(s.session_id)
    assert ok is True
    assert s.phase == "cancelled"
    assert s.finished_at is not None


async def test_cancel_session_unknown_returns_false():
    assert await cancel_session("dlg-nope") is False


async def test_cancel_session_already_done_returns_false():
    s = await create_session(mode="debate", question_preview="q", total_rounds=1)
    mark_phase(s, "done")
    assert await cancel_session(s.session_id) is False


async def test_snapshot_basic_shape():
    s = await create_session(mode="debate", question_preview="q", total_rounds=5)
    s.participants = [
        {"id": "glm", "model": "glm-5.1", "position": "X is better"},
        {"id": "kimi", "model": "kimi-k2.6", "position": "Y is better"},
    ]
    s.current_round = 2
    mark_phase(s, "round_2_response")
    snap = snapshot(s)
    assert snap["session_id"] == s.session_id
    assert snap["mode"] == "debate"
    assert snap["phase"] == "round_2_response"
    assert snap["current_round"] == 2
    assert snap["total_rounds"] == 5
    assert len(snap["participants"]) == 2
    assert snap["elapsed_ms"] is not None
    assert snap["error"] is None


async def test_active_sessions_hard_cap():
    """When MAX_ACTIVE_SESSIONS reached, create_session raises."""
    for _ in range(MAX_ACTIVE_SESSIONS):
        await create_session(mode="debate", question_preview="q", total_rounds=1)
    with pytest.raises(RuntimeError) as exc:
        await create_session(mode="debate", question_preview="q", total_rounds=1)
    assert "active sessions" in str(exc.value).lower()


async def test_reserve_active_slot_raises_when_cap_full():
    """reserve_active_slot (the dialogue_continue gate) raises the same
    RuntimeError as create_session when MAX_ACTIVE_SESSIONS active sessions
    exist, so resuming a terminal session cannot bypass the cap."""
    for _ in range(MAX_ACTIVE_SESSIONS):
        s = await create_session(mode="debate", question_preview="q", total_rounds=1)
        mark_phase(s, "round_1_critique")  # active
    with pytest.raises(RuntimeError) as exc:
        await reserve_active_slot()
    assert "active sessions" in str(exc.value).lower()


async def test_reserve_active_slot_ok_when_room_for_resume():
    """A terminal session is not counted toward the cap, so when MAX-1 are
    active and one is terminal there is room to reserve a slot for resume."""
    for _ in range(MAX_ACTIVE_SESSIONS - 1):
        s = await create_session(mode="debate", question_preview="q", total_rounds=1)
        mark_phase(s, "round_1_critique")  # active
    terminal = await create_session(mode="debate", question_preview="q", total_rounds=1)
    mark_phase(terminal, "done")  # terminal -> not counted
    await reserve_active_slot()  # must not raise


async def test_terminal_sessions_do_not_block_cap():
    """Completed (terminal) sessions must not count toward the active cap, even
    when GC hasn't pruned them yet (last_activity still fresh)."""
    for _ in range(MAX_ACTIVE_SESSIONS):
        s = await create_session(mode="debate", question_preview="q", total_rounds=1)
        mark_phase(s, "done")  # terminal but fresh -> not GC'd
    # Zero active sessions -> a new one must succeed despite MAX terminal ones.
    new_s = await create_session(mode="debate", question_preview="q", total_rounds=1)
    assert new_s.phase == "starting"


async def test_full_question_preserved_separately_from_preview():
    """question keeps the full topic; question_preview is truncated to 120."""
    long_q = "Z" * 300
    s = await create_session(mode="debate", question_preview=long_q, total_rounds=1)
    assert s.question == long_q
    assert len(s.question_preview) == 120


async def test_active_sessions_cap_releases_done_first():
    """Done sessions count toward the cap until pruned, but pruning happens
    opportunistically inside create_session. Verify that marking sessions done
    + creating new ones works once over the threshold."""
    sessions = []
    for _ in range(MAX_ACTIVE_SESSIONS):
        sessions.append(await create_session(mode="debate", question_preview="q", total_rounds=1))
    # Mark half done (pruning happens on next create attempt)
    for s in sessions[: MAX_ACTIVE_SESSIONS // 2]:
        mark_phase(s, "done")
        s.last_activity = time.time() - INACTIVE_TIMEOUT_SECONDS - 10  # make them stale
    # New create should succeed because stale done sessions got GC'd.
    new_s = await create_session(mode="debate", question_preview="q", total_rounds=1)
    assert new_s.session_id.startswith("dlg-")


def test_restored_diversity_threshold_zero_is_preserved():
    """A persisted threshold of 0 ("re-prompt on ANY agreement") must survive
    restore — the old `or 7` turned it into 7. A missing key still defaults to 7."""
    base = {"session_id": "dlg-x", "mode": "panel", "total_rounds": 3, "phase": "done"}
    s_zero = dialogue_state._state_from_dump({**base, "diversity_threshold": 0})
    assert s_zero.diversity_threshold == 0
    s_missing = dialogue_state._state_from_dump(dict(base))
    assert s_missing.diversity_threshold == 7


async def test_attach_task_and_cancel_propagates():
    """cancel_session should call task.cancel() on the bound asyncio.Task."""
    s = await create_session(mode="debate", question_preview="q", total_rounds=1)
    mark_phase(s, "round_1_critique")

    cancelled_flag = {"v": False}

    async def fake_long_running():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled_flag["v"] = True
            raise

    task = asyncio.create_task(fake_long_running())
    dialogue_state.attach_task(s, task)
    await cancel_session(s.session_id)
    # Give the event loop a tick to propagate the cancel
    await asyncio.sleep(0.05)
    assert cancelled_flag["v"] is True
