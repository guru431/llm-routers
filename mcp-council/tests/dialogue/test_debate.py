"""Tests for dialogue.debate — position generation, run_debate."""

import json

import pytest

from dialogue import state as dialogue_state
from dialogue.debate import (
    generate_positions,
    run_debate,
)


@pytest.fixture
def fake_moderator(monkeypatch):
    """Patch _call_model in engine with canned sequential responses (also used
    by debate's _call_moderator wrapper)."""
    responses: list[object] = []

    async def _fake(cfg, prompt, max_tokens, web_search):
        if not responses:
            raise RuntimeError("no canned response left for moderator")
        v = responses.pop(0)
        if isinstance(v, BaseException):
            raise v
        return v

    monkeypatch.setattr("dialogue.engine._call_model", _fake)

    def install(seq: list[object]):
        responses.clear()
        responses.extend(seq)

    return install


def _cfg(id_: str) -> dict:
    return {"id": id_, "model": f"{id_}-m", "base_url": "http://fake", "env_key": "FAKE_KEY"}


async def test_generate_positions_parses_json_array(fake_moderator):
    fake_moderator([json.dumps(["X is better because A", "Y is better because B"])])
    positions = await generate_positions(
        moderator_cfg=_cfg("deepseek-flash"),
        question="X vs Y?",
        n=2,
    )
    assert positions == ["X is better because A", "Y is better because B"]


async def test_generate_positions_strips_code_fence(fake_moderator):
    fake_moderator(['```json\n["pos1", "pos2"]\n```'])
    positions = await generate_positions(
        moderator_cfg=_cfg("deepseek-flash"),
        question="q",
        n=2,
    )
    assert positions == ["pos1", "pos2"]


async def test_generate_positions_wrong_count_raises(fake_moderator):
    fake_moderator([json.dumps(["only one"])])
    with pytest.raises(RuntimeError) as exc:
        await generate_positions(
            moderator_cfg=_cfg("deepseek-flash"),
            question="q",
            n=2,
        )
    assert "2" in str(exc.value)


async def test_generate_positions_non_json_raises(fake_moderator):
    fake_moderator(["this is not JSON"])
    with pytest.raises(RuntimeError) as exc:
        await generate_positions(
            moderator_cfg=_cfg("deepseek-flash"),
            question="q",
            n=2,
        )
    assert "json" in str(exc.value).lower()


async def test_run_debate_end_to_end(fake_moderator, tmp_path, monkeypatch):
    """Full happy path: position split → 3 rounds → summary."""
    # 1 position-gen + 2 (round 1 response, no critique)
    # + 4 (round 2 crit+resp) + 4 (round 3 crit+resp) + 1 (summary) = 12
    canned = (
        [json.dumps(["X is better", "Y is better"])]
        + ["resp glm r1", "resp kimi r1"]
        + ["crit glm r2", "crit kimi r2", "resp glm r2", "resp kimi r2"]
        + ["crit glm r3", "crit kimi r3", "resp glm r3", "resp kimi r3"]
        + ["FINAL SUMMARY"]
    )
    fake_moderator(canned)
    monkeypatch.setattr("dialogue.debate.DUMP_DIR", tmp_path)

    state = await dialogue_state.create_session(
        mode="debate", question_preview="X vs Y?", total_rounds=3,
    )

    await run_debate(
        state=state,
        question="X vs Y?",
        participant_cfgs=[_cfg("glm"), _cfg("kimi")],
        moderator_cfg=_cfg("deepseek-flash"),
        rounds=3,
        max_tokens=100,
        web_search=False,
        files_section=None,
    )

    assert state.phase == "done"
    assert state.current_round == 3
    assert state.dump_path is not None
    assert state.participants[0]["position"] == "X is better"
    assert state.participants[1]["position"] == "Y is better"
    summary_entries = [h for h in state.history if h.get("phase") == "summary"]
    assert len(summary_entries) == 1
    assert summary_entries[0]["text"] == "FINAL SUMMARY"

    # Regression: the persisted JSON must record phase=done, not an
    # intermediate phase (mark_phase must run BEFORE write_dump).
    import json as _json
    dumped = _json.loads(open(state.dump_path, encoding="utf-8").read())
    assert dumped["phase"] == "done", (
        f"dump recorded phase={dumped['phase']!r}; expected 'done'. "
        "mark_phase must be called before write_dump."
    )
