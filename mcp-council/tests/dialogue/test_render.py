"""Tests for dialogue.render — markdown output."""

import pytest

from dialogue import state as dialogue_state
from dialogue.render import format_dialogue_markdown


async def test_format_debate_markdown_contains_sections():
    state = await dialogue_state.create_session(
        mode="debate", question_preview="X vs Y?", total_rounds=2,
    )
    state.participants = [
        {"id": "glm", "model": "glm-5.1", "position": "X is better", "role": None},
        {"id": "kimi", "model": "kimi-k2.6", "position": "Y is better", "role": None},
    ]
    state.moderator = {"id": "deepseek-flash", "model": "deepseek-v4-flash"}
    state.history = [
        {"round": 1, "phase": "response", "id": "glm", "text": "X has feat F", "latency_ms": 12000, "status": "ok"},
        {"round": 1, "phase": "response", "id": "kimi", "text": "Y has feat G", "latency_ms": 14000, "status": "ok"},
        {"round": 2, "phase": "critique", "id": "glm", "text": "Y is weak because Z", "latency_ms": 11000, "status": "ok"},
        {"round": 2, "phase": "critique", "id": "kimi", "text": "X is weak because W", "latency_ms": 13000, "status": "ok"},
        {"round": 2, "phase": "response", "id": "glm", "text": "X still wins", "latency_ms": 10000, "status": "ok"},
        {"round": 2, "phase": "response", "id": "kimi", "text": "Y still wins", "latency_ms": 11000, "status": "ok"},
        {"round": 2, "phase": "summary", "id": "deepseek-flash", "text": "Both made strong points; X edge on F, Y on G", "latency_ms": 5000, "status": "ok"},
    ]
    state.dump_path = "logs/dialogues/dlg-xyz.json"

    md = format_dialogue_markdown(state, "X vs Y?")
    assert "# Dialogue session" in md
    assert "mode: debate" in md
    assert "X vs Y?" in md
    assert "defending \"X is better\"" in md
    assert "defending \"Y is better\"" in md
    assert "Round 1" in md
    assert "Round 2" in md
    assert "Critique" in md
    assert "Y is weak because Z" in md
    assert "Final Summary" in md
    assert "Both made strong points" in md
    assert "logs/dialogues/dlg-xyz.json" in md


async def test_format_panel_markdown_includes_devils_advocate_and_diversity():
    state = await dialogue_state.create_session(
        mode="panel", question_preview="q", total_rounds=2,
    )
    state.participants = [
        {"id": "a", "model": "a-m", "position": None, "role": None},
        {"id": "b", "model": "b-m", "position": None, "role": None},
    ]
    state.devils_advocates = ["a", "b"]
    state.diversity_scores = [3, 7]
    state.history = [
        {"round": 1, "phase": "response", "id": "a", "text": "R1A", "latency_ms": 1000, "status": "ok"},
        {"round": 1, "phase": "response", "id": "b", "text": "R1B", "latency_ms": 1000, "status": "ok"},
    ]

    md = format_dialogue_markdown(state, "q")
    assert "diversity" in md.lower()
    assert "[3, 7]" in md or "3, 7" in md
    assert "devil" in md.lower()
    assert "a, b" in md or "[a, b]" in md or "'a', 'b'" in md


async def test_format_socratic_markdown_question_answer_layout():
    state = await dialogue_state.create_session(
        mode="socratic", question_preview="quantum", total_rounds=1,
    )
    state.participants = [
        {"id": "deepseek-pro", "model": "deepseek-v4-pro", "position": None, "role": "questioner"},
        {"id": "glm", "model": "glm-5.1", "position": None, "role": "respondent"},
    ]
    state.history = [
        {"round": 1, "phase": "question", "id": "deepseek-pro", "text": "Why tunneling?", "latency_ms": 8000, "status": "ok"},
        {"round": 1, "phase": "answer", "id": "glm", "text": "Because barrier penetration", "latency_ms": 10000, "status": "ok"},
    ]

    md = format_dialogue_markdown(state, "quantum")
    assert "Why tunneling" in md
    assert "Because barrier penetration" in md
    assert "questioner" in md.lower()
    assert "respondent" in md.lower()


async def test_format_markdown_skips_summary_section_when_no_summary():
    state = await dialogue_state.create_session(
        mode="socratic", question_preview="q", total_rounds=1,
    )
    state.participants = [
        {"id": "a", "model": "a-m", "position": None, "role": "questioner"},
        {"id": "b", "model": "b-m", "position": None, "role": "respondent"},
    ]
    state.history = [
        {"round": 1, "phase": "question", "id": "a", "text": "Q1", "latency_ms": 100, "status": "ok"},
        {"round": 1, "phase": "answer", "id": "b", "text": "A1", "latency_ms": 100, "status": "ok"},
    ]
    md = format_dialogue_markdown(state, "q")
    assert "Final Summary" not in md
