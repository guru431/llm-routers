"""Tests for dialogue.panel — devil's advocate rotation, diversity monitor."""

import json

import pytest

from dialogue import state as dialogue_state
from dialogue.panel import (
    devils_advocate_for_round,
    run_diversity_check,
    run_panel,
    DEVILS_ADVOCATE_RULE,
)


def _cfg(id_: str) -> dict:
    return {"id": id_, "model": f"{id_}-m", "base_url": "http://fake", "env_key": "FAKE_KEY"}


def test_devils_advocate_rotates_by_round():
    participants = [_cfg("a"), _cfg("b"), _cfg("c")]
    assert devils_advocate_for_round(participants, round_n=1) == "a"
    assert devils_advocate_for_round(participants, round_n=2) == "b"
    assert devils_advocate_for_round(participants, round_n=3) == "c"
    assert devils_advocate_for_round(participants, round_n=4) == "a"


@pytest.fixture
def fake_call(monkeypatch):
    """Patch _call_model with canned sequential responses."""
    responses: list[object] = []

    async def _fake(cfg, prompt, max_tokens, web_search):
        if not responses:
            raise RuntimeError("no canned response")
        v = responses.pop(0)
        if isinstance(v, BaseException):
            raise v
        return v

    monkeypatch.setattr("dialogue.engine._call_model", _fake)

    def install(seq):
        responses.clear()
        responses.extend(seq)

    return install


async def test_run_diversity_check_parses_json(fake_call):
    fake_call([json.dumps({"score": 8, "agreers": ["a", "b"], "reasoning": "saying same thing"})])
    score, agreers = await run_diversity_check(
        monitor_cfg=_cfg("deepseek-flash"),
        responses={"a": "answer X", "b": "answer X", "c": "answer Y"},
    )
    assert score == 8
    assert agreers == ["a", "b"]


async def test_run_diversity_check_strips_code_fence(fake_call):
    fake_call(['```\n{"score": 3, "agreers": [], "reasoning": "diverse"}\n```'])
    score, agreers = await run_diversity_check(
        monitor_cfg=_cfg("deepseek-flash"),
        responses={"a": "A", "b": "B"},
    )
    assert score == 3
    assert agreers == []


async def test_run_diversity_check_invalid_json_returns_neutral(fake_call):
    """When monitor returns garbage, we fall back to score=0 (no re-prompt)."""
    fake_call(["not json at all"])
    score, agreers = await run_diversity_check(
        monitor_cfg=_cfg("deepseek-flash"),
        responses={"a": "A"},
    )
    assert score == 0
    assert agreers == []


async def test_run_panel_devils_advocate_id_recorded_per_round(fake_call, tmp_path, monkeypatch):
    monkeypatch.setattr("dialogue.panel.DUMP_DIR", tmp_path)
    # 4 participants, 2 rounds, diversity monitor off
    # Round 1: 4 responses (no critique). Round 2: 4 critique + 4 response. Summary: 1.
    fake_call(
        ["r1-a", "r1-b", "r1-c", "r1-d"]
        + ["c2-a", "c2-b", "c2-c", "c2-d", "r2-a", "r2-b", "r2-c", "r2-d"]
        + ["FINAL"]
    )

    state = await dialogue_state.create_session(
        mode="panel", question_preview="q", total_rounds=2,
    )

    await run_panel(
        state=state,
        question="q",
        participant_cfgs=[_cfg("a"), _cfg("b"), _cfg("c"), _cfg("d")],
        monitor_cfg=_cfg("deepseek-flash"),
        rounds=2,
        max_tokens=100,
        web_search=False,
        files_section=None,
        roles=None,
        diversity_monitor=False,
        diversity_threshold=7,
        devils_advocate_rotation=True,
    )

    assert state.devils_advocates == ["a", "b"]
    assert state.phase == "done"

    # Regression: dump must record phase=done (mark_phase before write_dump).
    import json as _json
    dumped = _json.loads(open(state.dump_path, encoding="utf-8").read())
    assert dumped["phase"] == "done"


async def test_run_panel_diversity_monitor_triggers_reprompt(fake_call, tmp_path, monkeypatch):
    """When diversity score > threshold, agreers should receive a re-prompt
    appended into history (with a 'reprompt' marker)."""
    monkeypatch.setattr("dialogue.panel.DUMP_DIR", tmp_path)
    # 4 participants, 1 round, monitor on.
    # Round 1 responses (4) → monitor says score=8, agreers=[a,b] → re-prompt a,b (2) → summary (1)
    fake_call(
        ["A1", "A1-copy", "B1", "C1"]
        + [json.dumps({"score": 8, "agreers": ["a", "b"], "reasoning": "saying same"})]
        + ["A1-distinct", "B1-distinct"]
        + ["FINAL"]
    )

    state = await dialogue_state.create_session(
        mode="panel", question_preview="q", total_rounds=1,
    )

    await run_panel(
        state=state,
        question="q",
        participant_cfgs=[_cfg("a"), _cfg("b"), _cfg("c"), _cfg("d")],
        monitor_cfg=_cfg("deepseek-flash"),
        rounds=1,
        max_tokens=100,
        web_search=False,
        files_section=None,
        roles=None,
        diversity_monitor=True,
        diversity_threshold=7,
        devils_advocate_rotation=False,
    )

    reprompts = [h for h in state.history if h.get("phase") == "reprompt"]
    assert len(reprompts) == 2
    assert {h["id"] for h in reprompts} == {"a", "b"}
    assert state.diversity_scores == [8]


async def test_run_panel_reprompt_only_failure_triggers_abort(fake_call, tmp_path, monkeypatch):
    """A participant that fails ONLY on the heavy reprompt call must still trip
    the failure-threshold abort. The pre-reprompt failure check cannot see that
    error, so run_panel re-checks after the reprompt."""
    monkeypatch.setattr("dialogue.panel.DUMP_DIR", tmp_path)
    # 2 participants, 1 round. Both respond OK → monitor flags both → reprompt:
    # a succeeds, b raises. 1/2 distinct failures == threshold(2)=1 → abort.
    fake_call(
        ["A1", "B1"]
        + [json.dumps({"score": 8, "agreers": ["a", "b"], "reasoning": "saying same"})]
        + ["A1-distinct", RuntimeError("reprompt boom")]
    )

    state = await dialogue_state.create_session(
        mode="panel", question_preview="q", total_rounds=1,
    )

    with pytest.raises(RuntimeError):
        await run_panel(
            state=state,
            question="q",
            participant_cfgs=[_cfg("a"), _cfg("b")],
            monitor_cfg=_cfg("deepseek-flash"),
            rounds=1,
            max_tokens=100,
            web_search=False,
            files_section=None,
            roles=None,
            diversity_monitor=True,
            diversity_threshold=7,
            devils_advocate_rotation=False,
        )

    assert state.phase == "error"
    assert "failure threshold exceeded in round 1" in (state.error or "")


async def test_run_panel_diversity_monitor_no_trigger_below_threshold(fake_call, tmp_path, monkeypatch):
    monkeypatch.setattr("dialogue.panel.DUMP_DIR", tmp_path)
    fake_call(
        ["A1", "B1"]
        + [json.dumps({"score": 4, "agreers": [], "reasoning": "diverse enough"})]
        + ["FINAL"]
    )

    state = await dialogue_state.create_session(
        mode="panel", question_preview="q", total_rounds=1,
    )

    await run_panel(
        state=state,
        question="q",
        participant_cfgs=[_cfg("a"), _cfg("b")],
        monitor_cfg=_cfg("deepseek-flash"),
        rounds=1,
        max_tokens=100,
        web_search=False,
        files_section=None,
        roles=None,
        diversity_monitor=True,
        diversity_threshold=7,
        devils_advocate_rotation=False,
    )
    reprompts = [h for h in state.history if h.get("phase") == "reprompt"]
    assert reprompts == []
    assert state.diversity_scores == [4]
