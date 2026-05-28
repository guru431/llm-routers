"""Integration tests for the new dialogue MCP tools in server.py."""

import json

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


async def test_model_debate_returns_session_id_immediately(fake_call, tmp_path, monkeypatch):
    monkeypatch.setattr("dialogue.debate.DUMP_DIR", tmp_path)
    fake_call([json.dumps(["X is better", "Y is better"])] + ["resp"] * 20)

    res = await server.model_debate(
        question="X vs Y?",
        participants=["glm", "kimi"],
        moderator="deepseek-flash",
        rounds=2,
        max_response_tokens=100,
    )
    assert isinstance(res, dict)
    assert res["session_id"].startswith("dlg-")
    assert res["mode"] == "debate"
    assert res["phase"] in {"starting", "round_1_response"}
    assert res["total_rounds"] == 2
    assert len(res["participants"]) == 2


async def test_model_panel_returns_session_id(fake_call, tmp_path, monkeypatch):
    monkeypatch.setattr("dialogue.panel.DUMP_DIR", tmp_path)
    fake_call(["resp"] * 100)
    res = await server.model_panel(
        question="q",
        participants=["glm", "kimi", "deepseek-pro", "qwen"],
        monitor_model="deepseek-flash",
        rounds=2,
        max_response_tokens=100,
        diversity_monitor=False,
    )
    assert res["mode"] == "panel"
    assert len(res["participants"]) == 4


async def test_model_socratic_returns_session_id(fake_call, tmp_path, monkeypatch):
    monkeypatch.setattr("dialogue.socratic.DUMP_DIR", tmp_path)
    fake_call(["resp"] * 20)
    res = await server.model_socratic(
        topic="quantum",
        questioner="deepseek-pro",
        respondent="glm",
        rounds=2,
        max_response_tokens=100,
    )
    assert res["mode"] == "socratic"
    assert len(res["participants"]) == 2


async def test_model_debate_validates_min_participants():
    with pytest.raises(RuntimeError) as exc:
        await server.model_debate(question="q", participants=["glm"], rounds=2)
    assert "at least 2" in str(exc.value).lower() or "2 distinct" in str(exc.value).lower()


async def test_model_panel_validates_min_participants():
    with pytest.raises(RuntimeError) as exc:
        await server.model_panel(question="q", participants=["glm", "kimi"], rounds=2)
    assert "at least 4" in str(exc.value).lower() or "4 distinct" in str(exc.value).lower()


async def test_model_socratic_validates_distinct():
    with pytest.raises(RuntimeError) as exc:
        await server.model_socratic(topic="q", questioner="glm", respondent="glm", rounds=2)
    assert "distinct" in str(exc.value).lower() or "same" in str(exc.value).lower()


async def test_rounds_out_of_range_rejected():
    with pytest.raises(RuntimeError) as exc:
        await server.model_debate(question="q", rounds=21)
    assert "rounds" in str(exc.value).lower()
    with pytest.raises(RuntimeError) as exc:
        await server.model_debate(question="q", rounds=0)
    assert "rounds" in str(exc.value).lower()


# --- Task 11 ---

import asyncio
import server as _server


async def test_dialogue_status_returns_snapshot(fake_call, tmp_path, monkeypatch):
    monkeypatch.setattr("dialogue.debate.DUMP_DIR", tmp_path)
    fake_call([json.dumps(["X is better", "Y is better"])] + ["resp"] * 20)
    started = await _server.model_debate(
        question="q", participants=["glm", "kimi"], rounds=2, max_response_tokens=100,
    )
    sid = started["session_id"]

    snap = await _server.dialogue_status(sid)
    assert snap["session_id"] == sid
    assert snap["mode"] == "debate"
    assert "phase" in snap
    assert "elapsed_ms" in snap


async def test_dialogue_status_unknown_returns_error():
    snap = await _server.dialogue_status("dlg-nope")
    assert "error" in snap


async def test_dialogue_result_not_ready_returns_phase(fake_call, tmp_path, monkeypatch):
    monkeypatch.setattr("dialogue.debate.DUMP_DIR", tmp_path)

    async def _slow(cfg, prompt, max_tokens, web_search):
        await asyncio.sleep(0.5)
        return "slow"

    monkeypatch.setattr("dialogue.engine._call_model", _slow)
    started = await _server.model_debate(
        question="q", participants=["glm", "kimi"], rounds=2, max_response_tokens=100,
    )
    res = await _server.dialogue_result(started["session_id"])
    assert res["ready"] is False
    assert "phase" in res
    assert "hint" in res
    await _server.dialogue_cancel(started["session_id"])


async def test_dialogue_result_ready_returns_markdown(fake_call, tmp_path, monkeypatch):
    monkeypatch.setattr("dialogue.debate.DUMP_DIR", tmp_path)
    fake_call(
        [json.dumps(["X is better", "Y is better"])]
        + ["r1-a", "r1-b"]
        + ["FINAL SUMMARY"]
    )
    started = await _server.model_debate(
        question="X vs Y?", participants=["glm", "kimi"], rounds=1, max_response_tokens=100,
    )
    sid = started["session_id"]
    for _ in range(50):
        snap = await _server.dialogue_status(sid)
        if snap.get("phase") in {"done", "error"}:
            break
        await asyncio.sleep(0.05)
    res = await _server.dialogue_result(sid)
    assert res["ready"] is True
    assert "result_markdown" in res
    assert "FINAL SUMMARY" in res["result_markdown"]


async def test_dialogue_cancel_running(fake_call, tmp_path, monkeypatch):
    monkeypatch.setattr("dialogue.debate.DUMP_DIR", tmp_path)

    async def _slow(cfg, prompt, max_tokens, web_search):
        await asyncio.sleep(2)
        return "slow"

    monkeypatch.setattr("dialogue.engine._call_model", _slow)
    started = await _server.model_debate(
        question="q", participants=["glm", "kimi"], rounds=2, max_response_tokens=100,
    )
    out = await _server.dialogue_cancel(started["session_id"])
    assert out["cancelled"] is True


async def test_dialogue_list_sessions_returns_recent():
    out = await _server.dialogue_list_sessions(limit=5)
    assert isinstance(out, list)


# --- Task 12 ---


async def test_dialogue_continue_only_allowed_after_done(fake_call, tmp_path, monkeypatch):
    monkeypatch.setattr("dialogue.debate.DUMP_DIR", tmp_path)

    async def _slow(cfg, prompt, max_tokens, web_search):
        await asyncio.sleep(2)
        return "slow"

    monkeypatch.setattr("dialogue.engine._call_model", _slow)
    started = await _server.model_debate(
        question="q", participants=["glm", "kimi"], rounds=2, max_response_tokens=100,
    )
    with pytest.raises(RuntimeError) as exc:
        await _server.dialogue_continue(
            session_id=started["session_id"], directive="углубитесь в X", rounds=2,
        )
    assert "done" in str(exc.value).lower()
    await _server.dialogue_cancel(started["session_id"])


async def test_dialogue_continue_unknown_session_raises():
    with pytest.raises(RuntimeError):
        await _server.dialogue_continue(
            session_id="dlg-nope", directive="continue", rounds=2,
        )


async def test_dialogue_continue_extends_rounds(fake_call, tmp_path, monkeypatch):
    monkeypatch.setattr("dialogue.debate.DUMP_DIR", tmp_path)
    fake_call(
        [json.dumps(["X is better", "Y is better"])]
        + ["r1-a", "r1-b"]
        + ["FIRST SUMMARY"]
        + ["c2-a", "c2-b", "r2-a", "r2-b"]
        + ["SECOND SUMMARY"]
    )
    started = await _server.model_debate(
        question="q", participants=["glm", "kimi"], rounds=1, max_response_tokens=100,
    )
    sid = started["session_id"]
    for _ in range(50):
        snap = await _server.dialogue_status(sid)
        if snap["phase"] in {"done", "error"}:
            break
        await asyncio.sleep(0.05)
    assert snap["phase"] == "done"

    cont = await _server.dialogue_continue(session_id=sid, directive="продолжай", rounds=1)
    assert cont["session_id"] == sid
    assert cont["total_rounds"] == 2

    for _ in range(50):
        snap = await _server.dialogue_status(sid)
        if snap["phase"] in {"done", "error"}:
            break
        await asyncio.sleep(0.05)
    assert snap["phase"] == "done"
    assert snap["current_round"] == 2


async def test_dialogue_continue_total_over_cap_rejected(fake_call, tmp_path, monkeypatch):
    monkeypatch.setattr("dialogue.debate.DUMP_DIR", tmp_path)
    fake_call(
        [json.dumps(["X is better", "Y is better"])]
        + ["r1-a", "r1-b"]
        + ["FIRST SUMMARY"]
    )
    started = await _server.model_debate(
        question="q", participants=["glm", "kimi"], rounds=1, max_response_tokens=100,
    )
    sid = started["session_id"]
    for _ in range(50):
        snap = await _server.dialogue_status(sid)
        if snap["phase"] in {"done", "error"}:
            break
        await asyncio.sleep(0.05)

    with pytest.raises(RuntimeError) as exc:
        await _server.dialogue_continue(session_id=sid, directive="extend", rounds=20)
    assert "20" in str(exc.value) or "max" in str(exc.value).lower()


# --- Task 14: live smoke test (gated) ---

import os


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_KEY") or not os.environ.get("OPENCODE_GO_KEY"),
    reason="needs DEEPSEEK_KEY + OPENCODE_GO_KEY in env",
)
async def test_live_socratic_smoke():
    """End-to-end test against real DeepSeek + GLM. Costs ~$0.01 and ~30s."""
    started = await _server.model_socratic(
        topic="Why does water expand when it freezes?",
        questioner="deepseek-pro",
        respondent="glm",
        rounds=1,
        max_response_tokens=500,
    )
    sid = started["session_id"]
    for _ in range(60):
        snap = await _server.dialogue_status(sid)
        if snap["phase"] in {"done", "error"}:
            break
        await asyncio.sleep(3)
    assert snap["phase"] == "done", f"expected done, got {snap['phase']}, error={snap.get('error')}"
    res = await _server.dialogue_result(sid)
    assert res["ready"] is True
    assert "water" in res["result_markdown"].lower()
