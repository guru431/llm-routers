"""Tests for dialogue feature additions (2026-07-17 IDEAS build):
token-aware rolling memory, diversity monitor 2.0, versioned dumps + event
journal, and dialogue_fork (branching)."""

import asyncio
import json

import pytest

import server as _server
from dialogue import prompts as dprompts
from dialogue import state as dialogue_state
from dialogue.engine import DUMP_SCHEMA_VERSION, write_dump
from dialogue.panel import run_diversity_check, run_diversity_check_v2


# --- Token-aware rolling memory --------------------------------------------

def test_token_aware_history_drops_old_rounds():
    # 6 rounds, each entry ~800 chars → a tiny token budget must drop old rounds
    # but keep the most-recent verbatim window + a rolling-summary marker.
    history = []
    for rn in range(1, 7):
        history.append({"round": rn, "phase": "response", "id": "a", "text": "x" * 800})
    rendered = dprompts.format_history_section(history, max_history_tokens=300)
    assert "rolling summary" in rendered
    assert "ROUND 6" in rendered  # most recent kept
    assert "ROUND 1" not in rendered  # oldest dropped


def test_token_aware_history_unbounded_keeps_all():
    history = [{"round": rn, "phase": "response", "id": "a", "text": "hi"} for rn in range(1, 4)]
    rendered = dprompts.format_history_section(history, max_history_tokens=None)
    assert "ROUND 1" in rendered and "ROUND 3" in rendered
    assert "rolling summary" not in rendered


# --- Diversity monitor 2.0 --------------------------------------------------

@pytest.mark.asyncio
async def test_diversity_v2_distinguishes_failure_from_zero(monkeypatch):
    async def _bad(cfg, prompt, max_tokens, web_search):
        return "not json"

    monkeypatch.setattr("dialogue.engine._call_model", _bad)
    r = await run_diversity_check_v2(monitor_cfg={"id": "m", "model": "M"},
                                     responses={"a": "x", "b": "y"})
    assert r["status"] == "failed"
    assert r["score"] == 0  # score 0 but flagged as a monitor FAILURE, not consensus


@pytest.mark.asyncio
async def test_diversity_v2_parses_uncertainty(monkeypatch):
    async def _ok(cfg, prompt, max_tokens, web_search):
        return json.dumps({"score": 8, "agreers": ["a"], "uncertainty": 0.2, "reasoning": "close"})

    monkeypatch.setattr("dialogue.engine._call_model", _ok)
    r = await run_diversity_check_v2(monitor_cfg={"id": "m", "model": "M"},
                                     responses={"a": "x", "b": "y"})
    assert r["status"] == "ok" and r["score"] == 8 and r["uncertainty"] == 0.2
    # Back-compat tuple wrapper still works.
    score, agreers = await run_diversity_check(monitor_cfg={"id": "m", "model": "M"},
                                               responses={"a": "x"})
    assert score == 8 and agreers == ["a"]


# --- Versioned dumps --------------------------------------------------------

def test_write_dump_has_schema_version(tmp_path):
    s = dialogue_state.DialogueState(
        session_id="dlg-test", mode="panel", question_preview="q",
        total_rounds=1, created_at=0.0,
    )
    s.phase = "done"
    path = write_dump(s, base_dir=tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == DUMP_SCHEMA_VERSION
    assert "diversity_monitor_status" in data


# --- Dialogue fork (branching) ---------------------------------------------

async def _wait_done(sid, tries=80):
    for _ in range(tries):
        snap = await _server.dialogue_status(sid)
        if snap["phase"] in {"done", "error"}:
            return snap
        await asyncio.sleep(0.05)
    return await _server.dialogue_status(sid)


@pytest.fixture
def fake_call(monkeypatch):
    seq: list[object] = []

    async def _fake(cfg, prompt, max_tokens, web_search):
        return seq.pop(0) if seq else f"<default {cfg['id']}>"

    monkeypatch.setattr("dialogue.engine._call_model", _fake)

    def install(items):
        seq.clear()
        seq.extend(items)
    return install


@pytest.mark.asyncio
async def test_dialogue_fork_branches_without_touching_source(fake_call, tmp_path, monkeypatch):
    monkeypatch.setattr("dialogue.debate.DUMP_DIR", tmp_path)
    # 1-round debate, then a fork adds one more round on a COPY.
    fake_call(
        [json.dumps(["X better", "Y better"])]
        + ["r1-a", "r1-b", "SUMMARY1"]
        + ["c2-a", "c2-b", "r2-a", "r2-b", "SUMMARY2"] + ["z"] * 20
    )
    started = await _server.model_debate(
        question="X vs Y?", participants=["glm", "kimi"], rounds=1, max_response_tokens=80,
    )
    sid = started["session_id"]
    snap = await _wait_done(sid)
    assert snap["phase"] == "done" and snap["current_round"] == 1

    forked = await _server.dialogue_fork(session_id=sid, directive="go deeper", rounds=1)
    assert forked["session_id"] != sid
    assert forked["forked_from"] == sid

    fsnap = await _wait_done(forked["session_id"])
    assert fsnap["phase"] == "done"
    assert fsnap["current_round"] == 2

    # Source is untouched (still 1 round, no round-2 entries).
    src = await dialogue_state.get_session(sid)
    assert src.current_round == 1
    assert all(h["round"] <= 1 for h in src.history)


@pytest.mark.asyncio
async def test_dialogue_fork_rejects_running_source():
    with pytest.raises(RuntimeError):
        await _server.dialogue_fork(session_id="nonexistent", directive="x", rounds=1)
