"""Tests for dialogue.engine — turns, phases, round loop, failure handling."""

import asyncio
import time

import pytest

from dialogue import state as dialogue_state
from dialogue.engine import (
    TurnResult,
    _run_turn,
    _run_phase,
)


@pytest.fixture
def fake_turn(monkeypatch):
    """Patch _call_model to return canned responses keyed by (model_id, marker).

    Usage:
        fake_turn({"glm": "answer A", "kimi": "answer B"})
        fake_turn({"glm": RuntimeError("boom")})   # simulate failure
    """
    canned: dict[str, object] = {}

    async def _fake_call(cfg: dict, prompt: str, max_tokens: int, web_search: bool):
        v = canned.get(cfg["id"])
        if isinstance(v, BaseException):
            raise v
        if v is None:
            return f"<default response from {cfg['id']}>"
        return v

    monkeypatch.setattr("dialogue.engine._call_model", _fake_call)

    def install(responses: dict[str, object]):
        canned.clear()
        canned.update(responses)

    return install


def _cfg(id_: str) -> dict:
    return {"id": id_, "model": f"{id_}-model", "base_url": "http://fake", "env_key": "FAKE_KEY"}


async def test_run_turn_ok_returns_text_and_latency(fake_turn):
    fake_turn({"glm": "hello"})
    res = await _run_turn(
        cfg=_cfg("glm"),
        prompt="say hi",
        max_tokens=100,
        web_search=False,
    )
    assert isinstance(res, TurnResult)
    assert res.status == "ok"
    assert res.text == "hello"
    assert res.id == "glm"
    assert res.error is None
    assert res.latency_ms is not None and res.latency_ms >= 0


async def test_run_turn_error_returns_error_status(fake_turn):
    fake_turn({"glm": RuntimeError("upstream 500")})
    res = await _run_turn(
        cfg=_cfg("glm"),
        prompt="say hi",
        max_tokens=100,
        web_search=False,
    )
    assert res.status == "error"
    assert res.text == ""
    assert "upstream 500" in (res.error or "")


async def test_run_phase_runs_participants_in_parallel(fake_turn, monkeypatch):
    """Two participants, each takes ~50ms. Total wall-time should be < 80ms
    (parallel), not ~100ms (sequential)."""
    fake_turn({})

    async def _slow_call(cfg, prompt, max_tokens, web_search):
        await asyncio.sleep(0.05)
        return f"slow-{cfg['id']}"

    monkeypatch.setattr("dialogue.engine._call_model", _slow_call)

    start = time.monotonic()
    results = await _run_phase(
        participants=[_cfg("glm"), _cfg("kimi")],
        prompt_builder=lambda cfg: f"prompt for {cfg['id']}",
        max_tokens=100,
        web_search=False,
    )
    elapsed = time.monotonic() - start
    assert elapsed < 0.08
    ids = sorted(r.id for r in results)
    assert ids == ["glm", "kimi"]
    assert all(r.status == "ok" for r in results)


async def test_run_phase_partial_failure_returns_mixed_results(fake_turn):
    fake_turn({"glm": "good", "kimi": RuntimeError("boom")})
    results = await _run_phase(
        participants=[_cfg("glm"), _cfg("kimi")],
        prompt_builder=lambda cfg: "prompt",
        max_tokens=100,
        web_search=False,
    )
    by_id = {r.id: r for r in results}
    assert by_id["glm"].status == "ok"
    assert by_id["kimi"].status == "error"
    assert "boom" in by_id["kimi"].error


async def test_run_phase_prompt_builder_receives_each_cfg(fake_turn):
    fake_turn({})
    seen_ids: list[str] = []

    def builder(cfg):
        seen_ids.append(cfg["id"])
        return f"prompt for {cfg['id']}"

    await _run_phase(
        participants=[_cfg("glm"), _cfg("kimi"), _cfg("deepseek-pro")],
        prompt_builder=builder,
        max_tokens=100,
        web_search=False,
    )
    assert sorted(seen_ids) == ["deepseek-pro", "glm", "kimi"]


# --- Task 4: run_round tests ---

from dialogue.engine import run_round
from dialogue.state import DialogueState


def _make_state(mode: str = "debate", participants_ids: list[str] | None = None) -> DialogueState:
    ids = participants_ids or ["glm", "kimi"]
    s = DialogueState(
        session_id="dlg-test",
        mode=mode,
        question_preview="q",
        total_rounds=5,
        created_at=time.time(),
        participants=[
            {"id": i, "model": f"{i}-model", "position": None, "role": None}
            for i in ids
        ],
    )
    return s


async def test_run_round_critique_then_response_appends_history(fake_turn):
    fake_turn({
        "glm": "X is strong because of feature F",
        "kimi": "Y is strong because of feature G",
    })
    state = _make_state()
    state.history.append({"round": 1, "phase": "response", "id": "glm", "text": "opening A"})
    state.history.append({"round": 1, "phase": "response", "id": "kimi", "text": "opening B"})

    await run_round(
        state=state,
        round_n=2,
        topic="why X vs Y?",
        role_descriptors={"glm": "defend X is better", "kimi": "defend Y is better"},
        max_tokens=100,
        web_search=False,
        anti_agreement_rules=None,
        files_section=None,
        do_critique=True,
    )

    critiques = [h for h in state.history if h["round"] == 2 and h["phase"] == "critique"]
    responses = [h for h in state.history if h["round"] == 2 and h["phase"] == "response"]
    assert len(critiques) == 2
    assert len(responses) == 2
    assert {h["id"] for h in critiques} == {"glm", "kimi"}
    assert {h["id"] for h in responses} == {"glm", "kimi"}


async def test_run_round_skip_critique_for_socratic(fake_turn):
    fake_turn({"glm": "A2", "kimi": "B2"})
    state = _make_state()
    state.history.append({"round": 1, "phase": "response", "id": "glm", "text": "A1"})
    state.history.append({"round": 1, "phase": "response", "id": "kimi", "text": "B1"})

    await run_round(
        state=state,
        round_n=2,
        topic="q",
        role_descriptors={"glm": "r1", "kimi": "r2"},
        max_tokens=100,
        web_search=False,
        anti_agreement_rules=None,
        files_section=None,
        do_critique=False,
    )

    critiques = [h for h in state.history if h["round"] == 2 and h["phase"] == "critique"]
    responses = [h for h in state.history if h["round"] == 2 and h["phase"] == "response"]
    assert len(critiques) == 0
    assert len(responses) == 2


async def test_run_round_first_round_no_critique_even_if_requested(fake_turn):
    """Round 1 has nothing to critique (no prior responses), so critique is
    auto-skipped regardless of do_critique flag."""
    fake_turn({"glm": "opening A", "kimi": "opening B"})
    state = _make_state()
    await run_round(
        state=state,
        round_n=1,
        topic="q",
        role_descriptors={"glm": "r1", "kimi": "r2"},
        max_tokens=100,
        web_search=False,
        anti_agreement_rules=None,
        files_section=None,
        do_critique=True,
    )

    critiques = [h for h in state.history if h["round"] == 1 and h["phase"] == "critique"]
    responses = [h for h in state.history if h["round"] == 1 and h["phase"] == "response"]
    assert len(critiques) == 0
    assert len(responses) == 2


async def test_run_round_anti_agreement_rule_routes_to_named_participant(fake_turn, monkeypatch):
    """anti_agreement_rules is a dict keyed by participant id. Only that
    participant should see the rule in their prompt."""
    seen_prompts: dict[str, str] = {}

    async def capture(cfg, prompt, max_tokens, web_search):
        seen_prompts[cfg["id"]] = prompt
        return f"resp from {cfg['id']}"

    monkeypatch.setattr("dialogue.engine._call_model", capture)

    state = _make_state()
    state.history.append({"round": 1, "phase": "response", "id": "glm", "text": "opening A"})
    state.history.append({"round": 1, "phase": "response", "id": "kimi", "text": "opening B"})

    await run_round(
        state=state,
        round_n=2,
        topic="q",
        role_descriptors={"glm": "r1", "kimi": "r2"},
        max_tokens=100,
        web_search=False,
        anti_agreement_rules={"kimi": "You are the devil's advocate this round."},
        files_section=None,
        do_critique=True,
    )

    kimi_prompt = seen_prompts.get("kimi", "")
    glm_prompt = seen_prompts.get("glm", "")
    assert "devil's advocate" in kimi_prompt
    assert "devil's advocate" not in glm_prompt


# --- Task 5: run_dialogue tests ---

import json
from pathlib import Path

from dialogue.engine import run_dialogue, FAILURE_THRESHOLD


async def test_run_dialogue_executes_all_rounds(fake_turn):
    fake_turn({"glm": "ok", "kimi": "ok"})
    state = _make_state()
    state.total_rounds = 3

    await run_dialogue(
        state=state,
        topic="q",
        role_descriptors={"glm": "r1", "kimi": "r2"},
        max_tokens=100,
        web_search=False,
        files_section=None,
        do_critique=True,
        per_round_hook=None,
    )

    rounds_covered = sorted({h["round"] for h in state.history})
    assert rounds_covered == [1, 2, 3]
    assert state.current_round == 3


async def test_run_dialogue_calls_per_round_hook(fake_turn):
    fake_turn({"glm": "ok", "kimi": "ok"})
    state = _make_state()
    state.total_rounds = 2

    seen_rounds: list[int] = []

    async def hook(s, round_n):
        seen_rounds.append(round_n)

    await run_dialogue(
        state=state,
        topic="q",
        role_descriptors={"glm": "r1", "kimi": "r2"},
        max_tokens=100,
        web_search=False,
        files_section=None,
        do_critique=True,
        per_round_hook=hook,
    )

    assert seen_rounds == [1, 2]


async def test_run_dialogue_aborts_on_failure_threshold(fake_turn):
    """If >=FAILURE_THRESHOLD participants fail in a round, state.phase=error and loop stops."""
    fake_turn({"glm": RuntimeError("boom"), "kimi": RuntimeError("boom2")})
    state = _make_state()
    state.total_rounds = 5

    with pytest.raises(RuntimeError) as exc:
        await run_dialogue(
            state=state,
            topic="q",
            role_descriptors={"glm": "r1", "kimi": "r2"},
            max_tokens=100,
            web_search=False,
            files_section=None,
            do_critique=True,
            per_round_hook=None,
        )
    assert "failure threshold" in str(exc.value).lower()
    assert state.phase == "error"
    assert state.current_round < 5


async def test_run_dialogue_continues_past_one_partial_failure(fake_turn):
    """Below threshold: one of three fails — loop continues."""
    fake_turn({"glm": "ok", "kimi": "ok", "deepseek-pro": RuntimeError("transient")})
    state = _make_state(participants_ids=["glm", "kimi", "deepseek-pro"])
    state.total_rounds = 2

    await run_dialogue(
        state=state,
        topic="q",
        role_descriptors={"glm": "r1", "kimi": "r2", "deepseek-pro": "r3"},
        max_tokens=100,
        web_search=False,
        files_section=None,
        do_critique=True,
        per_round_hook=None,
    )
    assert state.current_round == 2


async def test_run_dialogue_cancellation_propagates(fake_turn, monkeypatch):
    """If state._task is cancelled mid-round, run_dialogue raises CancelledError."""

    async def slow_or_cancel(cfg, prompt, max_tokens, web_search):
        await asyncio.sleep(0.2)
        return f"resp from {cfg['id']}"

    monkeypatch.setattr("dialogue.engine._call_model", slow_or_cancel)

    state = _make_state()
    state.total_rounds = 5

    task = asyncio.create_task(
        run_dialogue(
            state=state, topic="q",
            role_descriptors={"glm": "r1", "kimi": "r2"},
            max_tokens=100, web_search=False, files_section=None,
            do_critique=True, per_round_hook=None,
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_write_dump_persists_state_to_json(tmp_path):
    from dialogue.engine import write_dump

    state = _make_state()
    state.history.append({"round": 1, "phase": "response", "id": "glm", "text": "A1"})
    state.total_rounds = 1
    state.current_round = 1

    dump_path = write_dump(state, base_dir=tmp_path)
    assert dump_path.exists()
    data = json.loads(dump_path.read_text(encoding="utf-8"))
    assert data["session_id"] == state.session_id
    assert data["mode"] == "debate"
    assert len(data["history"]) == 1
    assert data["history"][0]["text"] == "A1"
