"""Regressions for the council findings resolved on 2026-07-25.

Covers: the bilingual high-risk gate, quorum counting SUPPORT rather than mere
presence in a ranking, incomplete-ranking handling, distinct-source evidence
strength, and the summary reporting rounds actually deliberated.
"""

import json

import pytest

from budget import RunBudget
from council import (
    _aggregate,
    _build_summary,
    _classify_risk,
    _evidence_strength,
    run_council,
)


def _make_members():
    return [
        {"id": "m1", "model": "M1", "base_url": "u", "env_key": "K1"},
        {"id": "m2", "model": "M2", "base_url": "u", "env_key": "K1"},
        {"id": "m3", "model": "M3", "base_url": "u", "env_key": "K1"},
    ]


@pytest.fixture(autouse=True)
def env_keys(monkeypatch):
    monkeypatch.setenv("K1", "sk-test")


# ---- Risk gate is bilingual -------------------------------------------------


@pytest.mark.parametrize("question", [
    "Удалить базу данных без резервной копии — как правильно?",
    "Раскрыть пароль и токен в конфиге, это нормально?",
    "как накатить миграцию на прод",
    "безопасно ли хранить ключ в репозитории",
    "как настроить приём платежей",
])
def test_classify_risk_matches_russian_questions(question):
    """The council is driven in Russian; an English-only marker list let
    destructive / secret-bearing questions pass as `normal` and clear
    human_review_required while their English equivalents were gated."""
    assert _classify_risk(question) == "high"


@pytest.mark.parametrize("question", [
    "какой шрифт выбрать для заголовков",
    "как отсортировать список в python",
    # Frequency corpus: ordinary technical phrasing that the broad Cyrillic stems
    # (прод\w* / ключ\w* / доступ\w* / удал\w* / затр\w* / auth\w*) all matched.
    # With every one of these classified `high`, human_review_required was a
    # constant and the gate measured nothing.
    "Как лучше продолжить рефакторинг парсера?",
    "Какие ключевые метрики выбрать для дашборда?",
    "Что доступно в новой версии библиотеки?",
    "продукт готов к релизу?",
    "это затронет другие модули?",
    "какие затраты на CI",
    "правильно ли я понял задачу",
    "как продумать структуру каталогов",
    "кто автор этой библиотеки",
    "who is the author of this paper",
])
def test_classify_risk_normal_questions_stay_normal(question):
    assert _classify_risk(question) == "normal"


@pytest.mark.parametrize("question", [
    "сбросить пароль пользователя",
    "как удалить старые записи",
    "деплой на продакшн",
    "как настроить удалённый доступ по ssh",  # "доступ" is the marker, not "удалённый"
])
def test_classify_risk_narrowed_stems_still_catch_real_risk(question):
    """Narrowing the stems must not blunt the gate: the destructive / secret /
    deploy forms these lookaheads were written around still classify high."""
    assert _classify_risk(question) == "high"


def test_classify_risk_still_matches_english():
    assert _classify_risk("delete the production database") == "high"


# ---- Quorum counts SUPPORT, not presence ------------------------------------


def test_quorum_ignores_rankers_that_placed_winner_last():
    """Two independent rankers gave the winner their MINIMUM score. Presence in
    their lists is not support, so independent_votes must not count them."""
    stage1 = [
        {"id": "glm", "model": "glm-5.2", "status": "ok"},     # opencode-go
        {"id": "gemini", "model": "gemini", "status": "ok"},   # helicone
        {"id": "codex", "model": "gpt-5.5", "status": "ok"},   # codex-agent
    ]
    stage2 = [
        {"ranker_id": "gemini", "status": "ok",
         "rankings": [{"ranked_id": "codex", "score": 9}, {"ranked_id": "glm", "score": 1}]},
        {"ranker_id": "codex", "status": "ok",
         "rankings": [{"ranked_id": "gemini", "score": 9}, {"ranked_id": "glm", "score": 1}]},
        {"ranker_id": "glm", "status": "ok",
         "rankings": [{"ranked_id": "gemini", "score": 5}, {"ranked_id": "codex", "score": 5}]},
    ]
    # Force glm as the winner regardless of the mean, to isolate the vote count.
    aggregate = [("glm", 9.9, 2), ("gemini", 5.0, 2), ("codex", 5.0, 2)]
    s = _build_summary(stage1, stage2, aggregate, None)
    assert s["winner_id"] == "glm"
    assert s["winner_ranked_by"] == 2      # both listed it …
    assert s["independent_votes"] == 0     # … neither supported it
    assert s["quorum_ok"] is False
    assert s["confidence"] != "high"
    assert s["human_review_required"] is True


def test_quorum_ignores_flat_ranker():
    """A ranker that gave every peer the same score expressed no preference."""
    stage1 = [
        {"id": "glm", "model": "glm-5.2", "status": "ok"},
        {"id": "gemini", "model": "gemini", "status": "ok"},
        {"id": "codex", "model": "gpt-5.5", "status": "ok"},
    ]
    stage2 = [
        {"ranker_id": "gemini", "status": "ok",
         "rankings": [{"ranked_id": "glm", "score": 7}, {"ranked_id": "codex", "score": 7}]},
        {"ranker_id": "codex", "status": "ok",
         "rankings": [{"ranked_id": "glm", "score": 9}, {"ranked_id": "gemini", "score": 4}]},
        {"ranker_id": "glm", "status": "ok",
         "rankings": [{"ranked_id": "gemini", "score": 5}, {"ranked_id": "codex", "score": 4}]},
    ]
    s = _build_summary(stage1, stage2, _aggregate(stage2), None)
    assert s["winner_id"] == "glm"
    assert s["independent_votes"] == 1  # codex only; gemini's 7/7 is not support
    assert s["quorum_ok"] is False


def test_incomplete_ranking_caps_confidence_and_forces_review():
    """A ranker that skipped a peer makes the means rest on different numbers of
    scores — the margin stops being a like-for-like measurement."""
    stage1 = [
        {"id": "glm", "model": "glm-5.2", "status": "ok"},
        {"id": "gemini", "model": "gemini", "status": "ok"},
        {"id": "codex", "model": "gpt-5.5", "status": "ok"},
    ]
    stage2 = [
        # gemini skipped codex entirely.
        {"ranker_id": "gemini", "status": "ok",
         "rankings": [{"ranked_id": "glm", "score": 9}]},
        {"ranker_id": "codex", "status": "ok",
         "rankings": [{"ranked_id": "glm", "score": 9}, {"ranked_id": "gemini", "score": 5}]},
        {"ranker_id": "glm", "status": "ok",
         "rankings": [{"ranked_id": "gemini", "score": 5}, {"ranked_id": "codex", "score": 4}]},
    ]
    s = _build_summary(stage1, stage2, _aggregate(stage2), None)
    assert s["incomplete_rankings"] == [{"ranker_id": "gemini", "missing": ["codex"]}]
    assert s["confidence"] != "high"
    assert s["human_review_required"] is True
    # gemini's partial list can't count as an independent supporter either.
    assert s["independent_votes"] == 1


@pytest.mark.asyncio
async def test_stage2_incomplete_ranking_triggers_one_repair():
    """An incomplete first reply is re-asked once; a still-incomplete repair is
    kept but flagged via missing_rankings instead of silently skewing aggregation."""
    members = _make_members()
    seen = {"stage2": 0}

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            seen["stage2"] += 1
            return {"content": json.dumps(
                {"rankings": [{"member": "A", "score": 7, "reasoning": "ok"}]}),
                "tokens_in": 1, "tokens_out": 1}
        return {"content": "ans-" + str(kwargs["model"]), "tokens_in": 1, "tokens_out": 1}

    result = await run_council(question="q", members=members, call_fn=fake_call)
    # 3 rankers × (first attempt + exactly one repair).
    assert seen["stage2"] == 6
    for s in result["stage2"]:
        assert s["status"] == "ok"
        assert s["missing_rankings"]  # the skipped peer is reported, not hidden
    assert result["summary"]["incomplete_rankings"]


# ---- Evidence = distinct sources, not result rows ---------------------------


def test_evidence_strength_one_query_is_not_corroboration():
    """Five rows from a single query on one host is ONE source of evidence."""
    stage1 = [{"tool_calls_log": [{
        "name": "web_search", "ok": True,
        "sources": ["https://docs.example.com/page" + str(i) for i in range(5)],
    }]}]
    ev = _evidence_strength(stage1)
    assert ev["results_total"] == 5
    assert ev["domains"] == 1
    assert ev["level"] == "weak"


def test_evidence_strength_dedups_same_url_across_members():
    """The same page reached by two members (or served from the run cache) is one
    source; tracking params / www / a trailing slash don't fork it into two."""
    stage1 = [
        {"tool_calls_log": [{"name": "web_search", "ok": True,
                             "sources": ["https://a.example/post"]}]},
        {"tool_calls_log": [{"name": "web_search", "ok": True,
                             "sources": ["https://www.a.example/post/?utm_source=x"]}]},
    ]
    ev = _evidence_strength(stage1)
    assert ev["results_total"] == 2
    assert ev["sources"] == 1
    assert ev["domains"] == 1


def test_evidence_strength_corroborated_needs_multiple_hosts():
    stage1 = [{"tool_calls_log": [{"name": "web_search", "ok": True, "sources": [
        "https://a.example/1", "https://b.example/2", "https://c.example/3",
    ]}]}]
    ev = _evidence_strength(stage1)
    assert ev["sources"] == 3 and ev["domains"] == 3
    assert ev["level"] == "corroborated"


def test_summary_source_quality_follows_distinct_hosts():
    stage1 = [
        {"id": "m1", "model": "M1", "status": "ok", "tool_calls_log": [{
            "name": "web_search", "ok": True,
            "sources": ["https://one.example/a", "https://one.example/b",
                        "https://one.example/c", "https://one.example/d"],
        }]},
    ]
    s = _build_summary(stage1, [], [], None)
    assert s["verdict"]["evidence_results_total"] == 4
    assert s["verdict"]["evidence_domains"] == 1
    assert s["verdict"]["source_quality"] == "single-source"


# ---- Summary reports rounds actually run ------------------------------------


@pytest.mark.asyncio
async def test_summary_reports_completed_rounds_not_requested():
    """A budget stop between rounds must not let the summary claim the requested
    rounds were deliberated, nor recommend a round number that was never reached."""
    members = _make_members()

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            return {"content": json.dumps({"rankings": [
                {"member": "A", "score": 5, "reasoning": ""},
                {"member": "B", "score": 5, "reasoning": ""}]}),
                "tokens_in": 1, "tokens_out": 1, "attempts": 1}
        return {"content": "answer", "tokens_in": 1, "tokens_out": 1, "attempts": 1}

    b = RunBudget(max_llm_calls=1)  # already exceeded after round 1
    result = await run_council(
        question="q", members=members, call_fn=fake_call, rounds=3, budget=b,
    )
    s = result["summary"]
    assert s["rounds_requested"] == 3
    assert s["rounds_completed"] == 1
    assert s["stop_reason"] == "budget"
    assert "rounds=4" not in s["recommended_next_action"]


@pytest.mark.asyncio
async def test_adaptive_usage_accumulates_across_passes():
    """Escalation re-runs the council; the reported usage and the budget guard
    must see probes + BOTH passes, not just the last one."""
    from council import run_adaptive_council

    # 5 members so pick_starting_subset holds some back as escalation reserves.
    members = [
        {"id": "glm", "model": "M1", "base_url": "u", "env_key": "K1"},
        {"id": "gemini", "model": "M2", "base_url": "u", "env_key": "K1"},
        {"id": "codex", "model": "M3", "base_url": "u", "env_key": "K1"},
        {"id": "qwen", "model": "M4", "base_url": "u", "env_key": "K1"},
        {"id": "kimi", "model": "M5", "base_url": "u", "env_key": "K1"},
    ]

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][-1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            return {"content": json.dumps({"rankings": [
                {"member": "A", "score": 5, "reasoning": ""},
                {"member": "B", "score": 5, "reasoning": ""}]}),
                "tokens_in": 1, "tokens_out": 1, "attempts": 1}
        return {"content": "answer", "tokens_in": 1, "tokens_out": 1, "attempts": 1}

    result = await run_adaptive_council(
        "q", members=members, call_fn=fake_call, healthcheck=False,
    )
    attempts = result["adaptive"]["attempts"]
    assert len(attempts) >= 2, "escalation should record a pass per run"
    # The top-level usage is the TOTAL across probes + every pass …
    assert result["usage"]["llm_calls"] >= attempts[0]["usage"]["llm_calls"]
    # … and the attempts trail is PER-PASS, so the rows sum to that total. It
    # used to carry the running total in every row, which made each escalation
    # look as expensive as the whole operation.
    assert sum(a["usage"]["llm_calls"] for a in attempts) == result["usage"]["llm_calls"]
    assert attempts[-1]["usage"]["llm_calls"] < result["usage"]["llm_calls"]
