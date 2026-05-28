"""Tests for dialogue.prompts — rendering, history truncation, role placement."""

import pytest

from dialogue.prompts import (
    HISTORY_TRUNCATE_ROUNDS,
    render_critique_prompt,
    render_response_prompt,
    render_position_split_prompt,
    render_summary_prompt,
    render_diversity_monitor_prompt,
    render_socratic_questioner_prompt,
    render_socratic_respondent_prompt,
    format_history_section,
)


def _hist(round_n: int, phase: str, id_: str, text: str) -> dict:
    return {"round": round_n, "phase": phase, "id": id_, "text": text}


def test_format_history_section_empty():
    assert format_history_section([]) == ""


def test_format_history_section_groups_by_round():
    history = [
        _hist(1, "response", "glm", "answer A1"),
        _hist(1, "response", "kimi", "answer B1"),
        _hist(2, "critique", "kimi", "B critiques A"),
        _hist(2, "critique", "glm", "A critiques B"),
        _hist(2, "response", "glm", "answer A2"),
        _hist(2, "response", "kimi", "answer B2"),
    ]
    out = format_history_section(history)
    assert "ROUND 1" in out
    assert "ROUND 2" in out
    assert "answer A1" in out
    assert "B critiques A" in out
    assert out.index("ROUND 1") < out.index("ROUND 2")


def test_format_history_section_truncates_to_last_N():
    # Use zero-padded markers so substring checks aren't ambiguous (r1 vs r10).
    history = [_hist(r, "response", "glm", f"TXT{r:02d}") for r in range(1, 20)]
    out = format_history_section(history)
    # Last HISTORY_TRUNCATE_ROUNDS (=10) rounds are kept: 10..19.
    kept_first = 20 - HISTORY_TRUNCATE_ROUNDS  # 10
    assert f"TXT{kept_first:02d}" in out
    assert f"TXT{kept_first + 9:02d}" in out  # 19
    assert "TXT01" not in out
    assert "TXT09" not in out


def test_render_critique_prompt_contains_required_sections():
    prompt = render_critique_prompt(
        topic="why X?",
        role_descriptor="you defend position: X is better",
        history=[_hist(1, "response", "glm", "X has feature F")],
        round_n=2,
        files_section=None,
        anti_agreement_rule=None,
    )
    assert "=== ROLE ===" in prompt
    assert "X is better" in prompt
    assert "=== TOPIC ===" in prompt
    assert "why X?" in prompt
    assert "=== DIALOGUE HISTORY ===" in prompt
    assert "X has feature F" in prompt
    assert "=== YOUR TASK" in prompt
    assert "critique" in prompt.lower()
    assert "round 2" in prompt.lower()


def test_render_critique_prompt_with_anti_agreement_rule():
    prompt = render_critique_prompt(
        topic="q",
        role_descriptor="r",
        history=[],
        round_n=1,
        files_section=None,
        anti_agreement_rule="You are the devil's advocate this round.",
    )
    assert "=== ANTI-AGREEMENT RULE ===" in prompt
    assert "devil's advocate" in prompt


def test_render_response_prompt_includes_critiques():
    history = [
        _hist(1, "response", "glm", "answer A"),
        _hist(1, "response", "kimi", "answer B"),
        _hist(2, "critique", "kimi", "A is weak because Z"),
    ]
    prompt = render_response_prompt(
        topic="q",
        role_descriptor="defend X",
        history=history,
        round_n=2,
        files_section=None,
        anti_agreement_rule=None,
    )
    assert "A is weak because Z" in prompt
    assert "round 2" in prompt.lower()
    assert "response" in prompt.lower()


def test_render_position_split_prompt_asks_for_n_positions():
    prompt = render_position_split_prompt(question="Rust vs Go for X?", n=2)
    assert "Rust vs Go" in prompt
    assert "2" in prompt
    assert "position" in prompt.lower() or "тезис" in prompt.lower()


def test_render_summary_prompt_includes_full_history():
    history = [_hist(1, "response", "glm", "A1"), _hist(1, "response", "kimi", "B1")]
    prompt = render_summary_prompt(topic="q", history=history, mode="debate")
    assert "A1" in prompt
    assert "B1" in prompt
    assert "summary" in prompt.lower() or "итог" in prompt.lower()


def test_render_diversity_monitor_prompt_lists_responses():
    responses = {"glm": "answer A", "kimi": "answer A almost identical"}
    prompt = render_diversity_monitor_prompt(responses=responses)
    assert "glm" in prompt
    assert "kimi" in prompt
    assert "answer A" in prompt
    assert "0" in prompt and "10" in prompt
    assert "agreer" in prompt.lower() or "agreed" in prompt.lower()


def test_render_socratic_questioner_prompt():
    prompt = render_socratic_questioner_prompt(
        topic="quantum tunneling",
        history=[_hist(1, "answer", "glm", "tunneling is...")],
        round_n=2,
        files_section=None,
    )
    assert "quantum tunneling" in prompt
    assert "questioner" in prompt.lower() or "вопрос" in prompt.lower()
    assert "tunneling is" in prompt


def test_render_socratic_respondent_prompt():
    prompt = render_socratic_respondent_prompt(
        topic="quantum tunneling",
        history=[_hist(2, "question", "deepseek-pro", "but why does X?")],
        round_n=2,
        files_section=None,
    )
    assert "quantum tunneling" in prompt
    assert "but why does X" in prompt
    assert "respondent" in prompt.lower() or "отвеч" in prompt.lower()
