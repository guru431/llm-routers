"""Tests for dialogue.socratic — questioner/respondent alternation, moderator notes."""

import pytest

from dialogue import state as dialogue_state
from dialogue.socratic import run_socratic


@pytest.fixture
def fake_call(monkeypatch):
    responses: list[str] = []

    async def _fake(cfg, prompt, max_tokens, web_search):
        if not responses:
            raise RuntimeError("no canned response")
        return responses.pop(0)

    monkeypatch.setattr("dialogue.engine._call_model", _fake)

    def install(seq: list[str]):
        responses.clear()
        responses.extend(seq)

    return install


def _cfg(id_: str) -> dict:
    return {"id": id_, "model": f"{id_}-m", "base_url": "http://fake", "env_key": "FAKE_KEY"}


async def test_run_socratic_no_moderator_question_then_answer_each_round(fake_call, tmp_path, monkeypatch):
    monkeypatch.setattr("dialogue.socratic.DUMP_DIR", tmp_path)
    fake_call([
        "Q1 from questioner", "A1 from respondent",
        "Q2 from questioner", "A2 from respondent",
    ])

    state = await dialogue_state.create_session(
        mode="socratic", question_preview="quantum tunneling", total_rounds=2,
    )

    await run_socratic(
        state=state,
        topic="quantum tunneling",
        questioner_cfg=_cfg("deepseek-pro"),
        respondent_cfg=_cfg("glm"),
        moderator_cfg=None,
        rounds=2,
        max_tokens=100,
        web_search=False,
        files_section=None,
    )

    assert state.phase == "done"
    rounds_seen = [(h["round"], h["phase"], h["id"]) for h in state.history]
    assert (1, "question", "deepseek-pro") in rounds_seen
    assert (1, "answer", "glm") in rounds_seen
    assert (2, "question", "deepseek-pro") in rounds_seen
    assert (2, "answer", "glm") in rounds_seen
    assert not any(h["phase"] == "moderator_note" for h in state.history)
    assert not any(h["phase"] == "summary" for h in state.history)

    # Regression: dump must record phase=done (mark_phase before write_dump).
    import json as _json
    dumped = _json.loads(open(state.dump_path, encoding="utf-8").read())
    assert dumped["phase"] == "done"


async def test_run_socratic_with_moderator_adds_notes_and_summary(fake_call, tmp_path, monkeypatch):
    monkeypatch.setattr("dialogue.socratic.DUMP_DIR", tmp_path)
    fake_call([
        "Q1", "A1", "MOD NOTE 1",
        "Q2", "A2", "MOD NOTE 2",
        "FINAL SUMMARY",
    ])

    state = await dialogue_state.create_session(
        mode="socratic", question_preview="topic", total_rounds=2,
    )

    await run_socratic(
        state=state,
        topic="topic",
        questioner_cfg=_cfg("deepseek-pro"),
        respondent_cfg=_cfg("glm"),
        moderator_cfg=_cfg("deepseek-flash"),
        rounds=2,
        max_tokens=100,
        web_search=False,
        files_section=None,
    )

    notes = [h for h in state.history if h["phase"] == "moderator_note"]
    summary = [h for h in state.history if h["phase"] == "summary"]
    assert len(notes) == 2
    assert notes[0]["text"] == "MOD NOTE 1"
    assert len(summary) == 1
    assert summary[0]["text"] == "FINAL SUMMARY"
