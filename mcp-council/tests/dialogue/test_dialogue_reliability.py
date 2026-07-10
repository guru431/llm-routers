"""Reliability regressions for dialogue concurrency + recovery.

Covers F#5 (concurrent dialogue_continue), F#6 (recovered 'interrupted' is
readable), F#7 (cancel persists so a restart stays cancelled), F#17 (a persisted
dump self-references its own path). Each was a silent concurrency/persistence bug
with no prior coverage.

Gotchas honored (see project_dialogue_gotchas): no __init__.py under tests/dialogue
(conftest puts the package root on sys.path); recovery tests clear _sessions
WITHOUT _reset_for_tests so the on-disk dump under COUNCIL_DIALOGUES_DIR survives
the simulated restart.
"""

import asyncio
import json
import time

import pytest

from dialogue import state as dialogue_state
import server


@pytest.fixture
def fake_call(monkeypatch):
    responses: list[object] = []

    async def _fake(cfg, prompt, max_tokens, web_search):
        if not responses:
            return f"<default {cfg['id']}>"
        v = responses.pop(0)
        if isinstance(v, BaseException):
            raise v
        return v

    monkeypatch.setattr("dialogue.engine._call_model", _fake)

    def install(seq):
        responses.clear()
        responses.extend(seq)

    return install


async def _wait_done(sid, tries=100):
    for _ in range(tries):
        snap = await server.dialogue_status(sid)
        if snap["phase"] in {"done", "error"}:
            return snap
        await asyncio.sleep(0.02)
    return await server.dialogue_status(sid)


# --- F#5: two concurrent dialogue_continue on one session must serialize ---

async def test_concurrent_dialogue_continue_serializes(fake_call, tmp_path, monkeypatch):
    monkeypatch.setattr("dialogue.debate.DUMP_DIR", tmp_path)
    monkeypatch.setattr("server.DIALOGUE_DUMP_DIR", tmp_path)
    fake_call([json.dumps(["X is better", "Y is better"])] + ["a", "b"] + ["SUMMARY"])
    started = await server.model_debate(
        question="X vs Y?", participants=["glm", "kimi"], rounds=1, max_response_tokens=50,
    )
    sid = started["session_id"]
    assert (await _wait_done(sid))["phase"] == "done"

    # Slow the continuation so the winner's runner stays mid-run while the loser
    # observes a non-terminal phase and refuses.
    async def _slow(cfg, prompt, max_tokens, web_search):
        await asyncio.sleep(0.3)
        return "slow"
    monkeypatch.setattr("dialogue.engine._call_model", _slow)

    results = await asyncio.gather(
        server.dialogue_continue(session_id=sid, directive="d1", rounds=1),
        server.dialogue_continue(session_id=sid, directive="d2", rounds=1),
        return_exceptions=True,
    )
    dicts = [r for r in results if isinstance(r, dict)]
    errs = [r for r in results if isinstance(r, BaseException)]
    assert len(dicts) == 1, results
    assert len(errs) == 1 and isinstance(errs[0], RuntimeError), results
    # total_rounds bumped exactly once (1 -> 2), never twice.
    snap = await server.dialogue_status(sid)
    assert snap["total_rounds"] == 2
    await server.dialogue_cancel(sid)


# --- F#6: a recovered 'interrupted' session is terminal and result-readable ---

async def test_recovered_interrupted_result_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("COUNCIL_DIALOGUES_DIR", str(tmp_path))
    sid = "dlg-interrupted01"
    dump = {
        "session_id": sid, "mode": "debate", "question_preview": "q", "question": "q",
        "total_rounds": 3, "current_round": 2, "phase": "round_2_response",
        "participants": [
            {"id": "glm", "model": "glm-5.2", "position": "X is better"},
            {"id": "kimi", "model": "kimi-k2.7-code", "position": "Y is better"},
        ],
        "moderator": {"id": "deepseek-flash", "model": "deepseek-v4-flash"},
        "history": [
            {"round": 1, "phase": "response", "id": "glm", "text": "hi", "latency_ms": 1, "status": "ok"},
        ],
        "created_at": time.time(), "started_at": time.time(),
    }
    (tmp_path / f"{sid}.json").write_text(json.dumps(dump), encoding="utf-8")

    assert dialogue_state.load_persisted_dialogues() >= 1
    st = await dialogue_state.get_session(sid)
    assert st is not None and st.phase == "interrupted"

    res = await server.dialogue_result(sid)
    assert res["ready"] is True
    assert res["result_markdown"]
    assert "restart" in (res["error"] or "").lower()


# --- F#7: cancel persists, so a restart reloads it as 'cancelled' not 'interrupted' ---

async def test_cancel_then_restart_stays_cancelled(tmp_path, monkeypatch):
    monkeypatch.setenv("COUNCIL_DIALOGUES_DIR", str(tmp_path))
    monkeypatch.setattr("dialogue.debate.DUMP_DIR", tmp_path)
    monkeypatch.setattr("server.DIALOGUE_DUMP_DIR", tmp_path)

    async def _slow(cfg, prompt, max_tokens, web_search):
        await asyncio.sleep(1.0)
        return "x"
    monkeypatch.setattr("dialogue.engine._call_model", _slow)

    started = await server.model_debate(
        question="q", participants=["glm", "kimi"], rounds=2, max_response_tokens=50,
    )
    sid = started["session_id"]
    await asyncio.sleep(0.05)  # let the runner enter its first slow await
    assert (await server.dialogue_cancel(sid))["cancelled"] is True

    st = await dialogue_state.get_session(sid)
    # The guard handles CancelledError: mark 'cancelled' + persist, then re-raise.
    with pytest.raises(asyncio.CancelledError):
        await st._task
    assert st.phase == "cancelled"

    data = json.loads((tmp_path / f"{sid}.json").read_text(encoding="utf-8"))
    assert data["phase"] == "cancelled"

    # Simulate restart WITHOUT deleting the dump (plain _reset_for_tests would).
    dialogue_state._sessions.clear()
    dialogue_state.load_persisted_dialogues()
    st2 = await dialogue_state.get_session(sid)
    assert st2 is not None and st2.phase == "cancelled"


# --- F#17: a persisted dump records its own path, survives recovery ---

async def test_recovered_done_keeps_dump_path(tmp_path, monkeypatch):
    monkeypatch.setenv("COUNCIL_DIALOGUES_DIR", str(tmp_path))
    from dialogue.engine import write_dump

    sid = "dlg-donedump01"
    st = dialogue_state.DialogueState(
        session_id=sid, mode="debate", question_preview="q", total_rounds=1,
        created_at=time.time(), question="q", phase="done",
        history=[{"round": 1, "phase": "summary", "id": "m", "text": "S", "latency_ms": 1, "status": "ok"}],
    )
    path = write_dump(st, base_dir=tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["dump_path"] is not None
    assert data["dump_path"].endswith(f"{sid}.json")

    dialogue_state.load_persisted_dialogues()
    res = await server.dialogue_result(sid)
    assert res["ready"] is True
    assert res["dump_path"] is not None
    assert res["dump_path"].endswith(f"{sid}.json")
