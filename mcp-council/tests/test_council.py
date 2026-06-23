"""Tests for council orchestrator: stage1, stage2 anonymization, aggregation, errors."""

import json

import pytest

from council import (
    _aggregate,
    _assign_pseudonyms,
    _build_summary,
    _compute_usage,
    _council_failure_reason,
    _extract_json,
    _split_synthesis_and_analysis,
    run_council,
)
from openai_client import CouncilHTTPError


# ---- members fixture --------------------------------------------------------


def _make_members():
    return [
        {"id": "m1", "model": "M1", "base_url": "u", "env_key": "K1"},
        {"id": "m2", "model": "M2", "base_url": "u", "env_key": "K1"},
        {"id": "m3", "model": "M3", "base_url": "u", "env_key": "K1"},
    ]


@pytest.fixture(autouse=True)
def env_keys(monkeypatch):
    monkeypatch.setenv("K1", "sk-test")


# ---- helpers ---------------------------------------------------------------


def test_assign_pseudonyms_deterministic_with_seed():
    ids = ["a", "b", "c"]
    a = _assign_pseudonyms(ids, seed=42)
    b = _assign_pseudonyms(ids, seed=42)
    assert a == b
    assert set(a.values()) <= set("ABCDEFGH")


def test_assign_pseudonyms_varies_with_seed():
    ids = ["a", "b", "c", "d"]
    a = _assign_pseudonyms(ids, seed=1)
    b = _assign_pseudonyms(ids, seed=2)
    # not guaranteed but extremely likely to differ for >=4 ids
    assert a != b


def test_extract_json_plain():
    assert _extract_json('{"rankings": []}') == {"rankings": []}


def test_extract_json_from_markdown_fence():
    text = "Sure, here:\n```json\n{\"rankings\":[{\"member\":\"A\",\"score\":7}]}\n```\nDone."
    out = _extract_json(text)
    assert out == {"rankings": [{"member": "A", "score": 7}]}


def test_extract_json_failure_raises():
    with pytest.raises(ValueError):
        _extract_json("no json here")


def test_extract_json_stops_at_first_balanced_object():
    """Trailing prose with stray braces must not be swallowed into the object —
    the balanced-brace scanner stops at the first complete {...}."""
    text = 'Ranking: {"rankings": [{"member": "A", "score": 7}]} — note: use {curly} later.'
    out = _extract_json(text)
    assert out == {"rankings": [{"member": "A", "score": 7}]}


def test_extract_json_braces_inside_strings_dont_break_scan():
    """Braces inside JSON string values must not skew the depth count."""
    text = 'prose {"rankings": [{"member": "A", "score": 7, "reasoning": "use {} sparingly"}]} done'
    out = _extract_json(text)
    assert out["rankings"][0]["reasoning"] == "use {} sparingly"


def test_aggregate_means_and_sorts():
    stage2 = [
        {
            "status": "ok",
            "rankings": [
                {"ranked_id": "m1", "pseudonym": "A", "score": 8, "reasoning": ""},
                {"ranked_id": "m2", "pseudonym": "B", "score": 6, "reasoning": ""},
            ],
        },
        {
            "status": "ok",
            "rankings": [
                {"ranked_id": "m1", "pseudonym": "B", "score": 10, "reasoning": ""},
                {"ranked_id": "m3", "pseudonym": "C", "score": 4, "reasoning": ""},
            ],
        },
    ]
    out = _aggregate(stage2)
    assert out[0] == ("m1", 9.0, 2)
    assert ("m2", 6.0, 1) in out
    assert ("m3", 4.0, 1) in out
    assert out == sorted(out, key=lambda x: -x[1])


def test_aggregate_skips_error_rankers():
    stage2 = [
        {"status": "error", "rankings": []},
        {
            "status": "ok",
            "rankings": [{"ranked_id": "m1", "pseudonym": "A", "score": 5, "reasoning": ""}],
        },
    ]
    out = _aggregate(stage2)
    assert out == [("m1", 5.0, 1)]


def test_aggregate_weights_by_confidence():
    """A high-confidence ranker should shift the weighted mean more than a
    low-confidence one."""
    # Two rankers both rank m1: one with conf=10 says 10/10, the other with
    # conf=2 says 0/10. Unweighted mean = 5; weighted mean should be much
    # closer to 10 because conf=10 ranker has 5x the weight of conf=2.
    stage2 = [
        {
            "status": "ok",
            "confidence": 10,
            "rankings": [{"ranked_id": "m1", "pseudonym": "A", "score": 10, "reasoning": ""}],
        },
        {
            "status": "ok",
            "confidence": 2,
            "rankings": [{"ranked_id": "m1", "pseudonym": "A", "score": 0, "reasoning": ""}],
        },
    ]
    out = _aggregate(stage2)
    mid, mean, count = out[0]
    assert mid == "m1"
    assert count == 2
    # Weighted mean: (10*1.0 + 0*0.2) / (1.0 + 0.2) = 10/1.2 ≈ 8.33
    assert 8.0 < mean < 8.7


def test_aggregate_missing_confidence_defaults_to_full_weight():
    """No confidence field -> weight=1.0 (same as conf=10), backward compat."""
    stage2 = [
        {
            "status": "ok",
            # no "confidence" key
            "rankings": [{"ranked_id": "m1", "pseudonym": "A", "score": 8, "reasoning": ""}],
        },
        {
            "status": "ok",
            "confidence": None,
            "rankings": [{"ranked_id": "m1", "pseudonym": "A", "score": 6, "reasoning": ""}],
        },
    ]
    out = _aggregate(stage2)
    assert out == [("m1", 7.0, 2)]


async def test_stage2_extracts_confidence_from_json():
    """When the ranker returns a confidence field, it is preserved in the
    stage2 result and used in aggregation."""
    members = _make_members()

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            return {
                "content": json.dumps({
                    "confidence": 7,
                    "rankings": [
                        {"member": "A", "score": 8, "reasoning": "good"},
                        {"member": "B", "score": 5, "reasoning": "okay"},
                    ],
                }),
                "tokens_in": 1,
                "tokens_out": 1,
            }
        return {"content": f"answer from {kwargs['model']}", "tokens_in": 1, "tokens_out": 1}

    result = await run_council(question="q", members=members, call_fn=fake_call)
    # All rankers should report confidence=7.
    for s in result["stage2"]:
        if s["status"] == "ok":
            assert s["confidence"] == 7


# ---- run_council -----------------------------------------------------------


async def test_run_council_partial_stage1_failure_continues():
    members = _make_members()
    calls = {"stage1": 0, "stage2": 0}

    async def fake_call(**kwargs):
        # Stage 1 messages have the question with "=== QUESTION ===" marker;
        # Stage 2 has "=== ANSWERS TO RANK ===".
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            calls["stage2"] += 1
            return {
                "content": json.dumps(
                    {
                        "rankings": [
                            {"member": "A", "score": 7, "reasoning": "good"},
                        ]
                    }
                ),
                "tokens_in": 1,
                "tokens_out": 1,
            }
        calls["stage1"] += 1
        # Fail for m2 specifically.
        if kwargs["model"] == "M2":
            raise CouncilHTTPError("boom")
        return {"content": f"answer from {kwargs['model']}", "tokens_in": 1, "tokens_out": 1}

    result = await run_council(
        question="q", members=members, call_fn=fake_call
    )
    statuses = {s["id"]: s["status"] for s in result["stage1"]}
    assert statuses == {"m1": "ok", "m2": "error", "m3": "ok"}
    # Stage 2: only m1 and m3 rank, each sees only the other.
    assert len(result["stage2"]) == 2
    assert all(s["status"] == "ok" for s in result["stage2"])
    assert any("m2" in n and "stage1 error" in n for n in result["notes"])


async def test_run_council_all_stage1_fail_raises():
    members = _make_members()

    async def fake_call(**kwargs):
        raise CouncilHTTPError("everyone fails")

    with pytest.raises(RuntimeError, match="council fully failed"):
        await run_council(question="q", members=members, call_fn=fake_call)


async def test_run_council_stage2_invalid_json_marked_error():
    members = _make_members()

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            # m1 returns invalid JSON, others return valid.
            if kwargs["model"] == "M1":
                return {"content": "not json at all", "tokens_in": 1, "tokens_out": 1}
            return {
                "content": json.dumps(
                    {"rankings": [{"member": "A", "score": 8, "reasoning": ""}]}
                ),
                "tokens_in": 1,
                "tokens_out": 1,
            }
        return {"content": f"ans-{kwargs['model']}", "tokens_in": 1, "tokens_out": 1}

    result = await run_council(question="q", members=members, call_fn=fake_call)
    s2_by_ranker = {s["ranker_id"]: s for s in result["stage2"]}
    assert s2_by_ranker["m1"]["status"] == "error"
    assert "invalid_json" in s2_by_ranker["m1"]["error"]
    # m1's stage1 answer is preserved.
    s1_by_id = {s["id"]: s for s in result["stage1"]}
    assert s1_by_id["m1"]["status"] == "ok"


def test_compute_usage_aggregates_calls_tokens_retries():
    rounds_detail = [{
        "stage1": [
            {"attempts": 1, "tokens_in": 100, "tokens_out": 50,
             "tool_calls_log": [{}, {}]},
            {"attempts": 2, "tokens_in": 10, "tokens_out": 5, "tool_calls_log": []},
            # no-key member: never reached provider — excluded from llm_calls.
            {"attempts": None, "tokens_in": None, "tokens_out": None,
             "tool_calls_log": []},
        ],
        "stage2": [{"attempts": 1, "tokens_in": 20, "tokens_out": 8}],
        "aggregate": [],
    }]
    stage3 = {"attempts": 1, "tokens_in": 200, "tokens_out": 100, "status": "ok"}
    u = _compute_usage(rounds_detail, stage3)
    assert u["llm_calls"] == 4
    assert u["tokens_in"] == 100 + 10 + 20 + 200
    assert u["tokens_out"] == 50 + 5 + 8 + 100
    assert u["web_search_calls"] == 2
    assert u["retries"] == 1
    assert u["estimated_cost_usd"] is None


def test_compute_usage_no_double_count_carried_failed_member():
    # A failed web_search member is carried forward by identity into round 2's
    # stage1 (same dict object). Its loop_* usage must be counted only once.
    failed = {
        "id": "m3", "status": "error", "loop_calls": 3, "loop_attempts": 4,
        "loop_tokens_in": 40, "loop_tokens_out": 20,
        "tool_calls_log": [{}, {}],
    }
    ok_r1 = {"attempts": 1, "tokens_in": 100, "tokens_out": 50, "tool_calls_log": []}
    ok_r2 = {"attempts": 1, "tokens_in": 100, "tokens_out": 50, "tool_calls_log": []}
    rounds_detail = [
        {"stage1": [ok_r1, failed], "stage2": [], "aggregate": []},
        {"stage1": [ok_r2, failed], "stage2": [], "aggregate": []},
    ]
    u = _compute_usage(rounds_detail, None)
    # failed.loop_calls counted once (3) + two ok members (1 each) = 5.
    assert u["llm_calls"] == 3 + 1 + 1
    assert u["tokens_in"] == 40 + 100 + 100
    assert u["tokens_out"] == 20 + 50 + 50
    assert u["web_search_calls"] == 2  # failed member's 2 tool calls, once
    assert u["retries"] == 1  # failed: max(0, 4-3)=1; ok members: 0


@pytest.mark.parametrize("error,expected", [
    # Council-only failure the healthcheck classifier never sees: map a missing
    # env var to healthcheck's dedicated `no_key` status, not the generic error.
    ("env var OPENCODE_GO_KEY not set", "no_key"),
    # Reuse healthcheck._classify_error for the provider-level failures.
    ("http 402 insufficient_balance: no funds", "insufficient_balance"),
    ("http 401: bad key", "auth"),
    ("timeout after 3 attempts: ReadTimeout", "timeout"),
    ("overload after 3 attempts (last status 429)", "rate_limited"),
    # Council-specific stage-2 failure mode — falls through to the generic code.
    ("invalid_json: rankings is empty", "error"),
    (None, "error"),
])
def test_council_failure_reason(error, expected):
    assert _council_failure_reason(error) == expected


def test_build_summary_failed_models_carry_failure_reason():
    """Each failed_models entry gets a stable `failure_reason` code alongside the
    free-form `error` string, so automation can switch on a code."""
    stage1 = [
        {"id": "m1", "model": "M1", "status": "ok"},
        {"id": "m2", "model": "M2", "status": "error",
         "error": "http 402 insufficient_balance: no funds"},
        {"id": "m3", "model": "M3", "status": "error",
         "error": "env var OPENCODE_GO_KEY not set"},
    ]
    stage2 = [
        {"ranker_id": "m1", "status": "error", "error": "invalid_json: rankings is empty"},
    ]
    s = _build_summary(stage1, stage2, [], None)
    by_id = {f["id"]: f for f in s["failed_models"]}
    assert by_id["m2"]["failure_reason"] == "insufficient_balance"
    assert by_id["m2"]["error"] == "http 402 insufficient_balance: no funds"  # raw kept
    assert by_id["m3"]["failure_reason"] == "no_key"
    assert by_id["m1"]["failure_reason"] == "error"


def test_build_summary_winner_failed_and_disagreement():
    stage1 = [
        {"id": "m1", "model": "M1", "status": "ok"},
        {"id": "m2", "model": "M2", "status": "ok"},
        {"id": "m3", "model": "M3", "status": "error", "error": "boom"},
    ]
    stage2 = [
        {"ranker_id": "m1", "status": "ok",
         "rankings": [{"ranked_id": "m2", "score": 9}, {"ranked_id": "m3", "score": 2}]},
        {"ranker_id": "m2", "status": "ok",
         "rankings": [{"ranked_id": "m1", "score": 8}, {"ranked_id": "m3", "score": 7}]},
    ]
    aggregate = [("m2", 9.0, 1), ("m1", 8.0, 1), ("m3", 4.5, 2)]
    s = _build_summary(stage1, stage2, aggregate, None)
    assert s["winner_id"] == "m2"
    assert s["winner_model"] == "M2"
    assert any(f["id"] == "m3" and f["stage"] == "stage1" for f in s["failed_models"])
    # m3 scored 2 and 7 across rankers → spread 5 (≥3) is a disagreement.
    assert any(d["id"] == "m3" and d["spread"] == 5 for d in s["top_disagreements"])
    assert s["confidence"] in ("low", "medium", "high")


async def test_run_council_returns_usage_and_summary():
    members = _make_members()

    async def fake_call(**kwargs):
        user = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user:
            return {"content": json.dumps(
                {"rankings": [{"member": "A", "score": 8, "reasoning": ""}]}),
                "tokens_in": 5, "tokens_out": 3, "attempts": 1}
        return {"content": "ans", "tokens_in": 7, "tokens_out": 4, "attempts": 1}

    result = await run_council(question="q", members=members, call_fn=fake_call)
    assert result["usage"]["llm_calls"] > 0
    assert result["summary"]["winner_id"] is not None


async def test_run_council_stage2_all_invalid_entries_marked_error():
    """Valid JSON with a non-empty rankings list whose entries ALL fail
    normalization (unknown pseudonym, out-of-range score) must mark the ranker
    as error — not silently return an empty 'ok' ranking that degrades the
    aggregate."""
    members = _make_members()

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            if kwargs["model"] == "M1":
                # 'Z' is not an assigned pseudonym and 99 is out of [1,10];
                # every entry gets dropped during normalization.
                return {
                    "content": json.dumps(
                        {"rankings": [{"member": "Z", "score": 99, "reasoning": ""}]}
                    ),
                    "tokens_in": 1,
                    "tokens_out": 1,
                }
            return {
                "content": json.dumps(
                    {"rankings": [{"member": "A", "score": 8, "reasoning": ""}]}
                ),
                "tokens_in": 1,
                "tokens_out": 1,
            }
        return {"content": f"ans-{kwargs['model']}", "tokens_in": 1, "tokens_out": 1}

    result = await run_council(question="q", members=members, call_fn=fake_call)
    s2_by_ranker = {s["ranker_id"]: s for s in result["stage2"]}
    assert s2_by_ranker["m1"]["status"] == "error"
    assert "invalid_json" in s2_by_ranker["m1"]["error"]
    assert s2_by_ranker["m1"]["rankings"] == []


async def test_stage2_uses_pseudonyms_no_model_names_in_user_prompt():
    """Orchestrator must NOT include raw model names or member ids next to the
    answers in stage 2 prompts. (Answer text itself may say anything — we
    explicitly check the prompt's framing, not the answer body.)"""
    members = _make_members()
    seen_user_prompts: list[str] = []

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            seen_user_prompts.append(user_msg)
            return {
                "content": json.dumps(
                    {"rankings": [{"member": "A", "score": 7, "reasoning": ""}]}
                ),
                "tokens_in": 1,
                "tokens_out": 1,
            }
        # Answer text is intentionally generic — no model/id leaks from us.
        return {"content": "neutral generic answer body", "tokens_in": 1, "tokens_out": 1}

    await run_council(question="q", members=members, call_fn=fake_call)
    assert seen_user_prompts
    for prompt in seen_user_prompts:
        for m in members:
            assert m["id"] not in prompt
            assert m["model"] not in prompt
        assert "Member A" in prompt or "Member B" in prompt


async def test_self_ranking_excluded():
    members = _make_members()
    # Make m1 always say "Member X: 10" — but we don't trust that, the orchestrator
    # never sends m1's own answer to m1, so m1 cannot rank itself.
    other_answers_seen_by: dict[str, list[str]] = {}

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            # extract the model that is doing the ranking
            ranker_model = kwargs["model"]
            other_answers_seen_by[ranker_model] = user_msg
            return {
                "content": json.dumps(
                    {"rankings": [{"member": "A", "score": 5, "reasoning": ""}]}
                ),
                "tokens_in": 1,
                "tokens_out": 1,
            }
        return {"content": f"unique-ans-{kwargs['model']}", "tokens_in": 1, "tokens_out": 1}

    await run_council(question="q", members=members, call_fn=fake_call)
    assert "unique-ans-M1" not in other_answers_seen_by["M1"]
    assert "unique-ans-M2" not in other_answers_seen_by["M2"]
    assert "unique-ans-M3" not in other_answers_seen_by["M3"]


async def test_env_var_missing_marks_error(monkeypatch):
    monkeypatch.delenv("K1", raising=False)
    members = _make_members()

    async def fake_call(**kwargs):
        return {"content": "x", "tokens_in": 1, "tokens_out": 1}

    with pytest.raises(RuntimeError, match="council fully failed"):
        await run_council(question="q", members=members, call_fn=fake_call)


# ---- web_search tool loop --------------------------------------------------


async def test_web_search_disabled_no_tools_in_payload():
    """When web_search=False, the call must NOT include `tools`."""
    members = _make_members()
    captured_tools = []

    async def fake_call(**kwargs):
        captured_tools.append(kwargs.get("tools"))
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            return {
                "content": json.dumps(
                    {"rankings": [{"member": "A", "score": 7, "reasoning": "ok"}]}
                ),
                "tokens_in": 1, "tokens_out": 1,
            }
        return {"content": "answer", "tokens_in": 1, "tokens_out": 1}

    await run_council(question="q", members=members, call_fn=fake_call, web_search=False)
    # No call should have received a `tools` parameter.
    assert all(t is None for t in captured_tools), captured_tools


async def test_web_search_enabled_passes_tools_in_stage1_only(monkeypatch):
    """When web_search=True, stage1 calls get tools, but stage2 (ranking) does not."""
    from web_search import WEB_SEARCH_TOOL_SPEC
    members = _make_members()
    stage1_tool_specs = []
    stage2_tool_specs = []

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            stage2_tool_specs.append(kwargs.get("tools"))
            return {
                "content": json.dumps(
                    {"rankings": [{"member": "A", "score": 7, "reasoning": "ok"}]}
                ),
                "tokens_in": 1, "tokens_out": 1,
            }
        stage1_tool_specs.append(kwargs.get("tools"))
        # Model responds directly (no tool_calls) — easy path.
        return {"content": "answer", "tokens_in": 1, "tokens_out": 1}

    await run_council(question="q", members=members, call_fn=fake_call, web_search=True)
    # Stage 1 should receive the web_search tool spec.
    assert stage1_tool_specs and all(
        t == [WEB_SEARCH_TOOL_SPEC] for t in stage1_tool_specs
    )
    # Stage 2 should not.
    assert stage2_tool_specs and all(t is None for t in stage2_tool_specs)


async def test_tool_loop_executes_web_search_then_returns_content(monkeypatch):
    """Simulate: model asks for web_search, we execute (mocked), model then
    produces final content. Result.answer should be the final content,
    tool_calls_log should record one search."""
    from unittest.mock import patch
    members = _make_members()

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            return {
                "content": json.dumps(
                    {"rankings": [{"member": "A", "score": 7, "reasoning": "ok"}]}
                ),
                "tokens_in": 1, "tokens_out": 1,
            }
        # Per-conversation tool counter — look at how many tool messages are
        # already in the message history.
        tool_msgs = [m for m in kwargs["messages"] if m.get("role") == "tool"]
        if not tool_msgs:
            return {
                "content": None,
                "tool_calls": [{
                    "id": f"call_{kwargs['model']}",
                    "function": {"name": "web_search", "arguments": '{"query":"test"}'},
                }],
                "finish_reason": "tool_calls",
                "tokens_in": 1, "tokens_out": 1,
            }
        # After tool result has been fed back → final content.
        return {"content": f"final answer with research from {kwargs['model']}", "tokens_in": 1, "tokens_out": 1}

    # Mock Exa to avoid network.
    async def fake_search(query, num_results=5, *, api_key=None, timeout=30.0):
        return {
            "query": query,
            "results": [{"title": "T", "url": "U", "summary": "S", "highlights": []}],
            "cost_dollars": 0.001,
            "latency_ms": 10,
        }

    with patch("web_search_tool.web_search_exa", new=fake_search):
        result = await run_council(
            question="q", members=members, call_fn=fake_call, web_search=True
        )
    for s in result["stage1"]:
        if s["status"] == "ok":
            assert "final answer" in s["answer"]
            log = s.get("tool_calls_log") or []
            assert len(log) == 1
            assert log[0]["ok"] is True
            assert log[0]["query"] == "test"


async def test_tool_loop_max_iterations_marks_error():
    """Model loops forever asking for searches — after 5 iterations we abort
    with status=error and no answer is propagated to stage 2."""
    from unittest.mock import patch
    members = [_make_members()[0]]  # one member to make assertions simpler

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            # We shouldn't reach stage 2 — only one member, anyway.
            return {
                "content": json.dumps(
                    {"rankings": [{"member": "A", "score": 7, "reasoning": ""}]}
                ),
                "tokens_in": 1, "tokens_out": 1,
            }
        # Always ask for a tool — infinite loop.
        return {
            "content": None,
            "tool_calls": [{
                "id": "x", "function": {"name": "web_search", "arguments": '{"query":"loop"}'},
            }],
            "finish_reason": "tool_calls",
            "tokens_in": 1, "tokens_out": 1,
        }

    async def fake_search(query, num_results=5, *, api_key=None, timeout=30.0):
        return {"query": query, "results": [], "cost_dollars": 0.001, "latency_ms": 1}

    with patch("web_search_tool.web_search_exa", new=fake_search):
        with pytest.raises(RuntimeError, match="council fully failed"):
            # Single-member council → all-fail → RuntimeError (acceptable).
            await run_council(question="q", members=members, call_fn=fake_call, web_search=True)


async def test_tool_loop_final_turn_forces_no_tools_to_salvage_answer():
    """Graceful degradation: model burns every tool turn requesting searches,
    but on the final turn the loop passes tool_choice='none' — the model then
    writes a final answer from the collected results instead of being dropped
    with no content. Regression coverage for BUG_TOOL_ITERATIONS_LIMIT."""
    from unittest.mock import patch
    from web_search_tool import MAX_TOOL_ITERATIONS

    members = [_make_members()[0]]
    saw_force_none = []
    pre_force_tool_turns = 0

    async def fake_call(**kwargs):
        nonlocal pre_force_tool_turns
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            return {
                "content": json.dumps(
                    {"rankings": [{"member": "A", "score": 7, "reasoning": ""}]}
                ),
                "tokens_in": 1, "tokens_out": 1,
            }
        if kwargs.get("tool_choice") == "none":
            saw_force_none.append(True)
            return {
                "content": "salvaged answer from prior searches",
                "tokens_in": 1, "tokens_out": 1,
            }
        # Without the directive, keep demanding another search.
        pre_force_tool_turns += 1
        return {
            "content": None,
            "tool_calls": [{
                "id": f"x{pre_force_tool_turns}",
                "function": {"name": "web_search", "arguments": '{"query":"q"}'},
            }],
            "finish_reason": "tool_calls",
            "tokens_in": 1, "tokens_out": 1,
        }

    async def fake_search(query, num_results=5, *, api_key=None, timeout=30.0):
        return {"query": query, "results": [], "cost_dollars": 0.0, "latency_ms": 1}

    with patch("web_search_tool.web_search_exa", new=fake_search):
        result = await run_council(
            question="q", members=members, call_fn=fake_call, web_search=True
        )

    assert saw_force_none, "loop must pass tool_choice='none' on the final turn"
    # The model used up every regular turn before the salvage call.
    assert pre_force_tool_turns == MAX_TOOL_ITERATIONS
    assert result["stage1"][0]["status"] == "ok"
    assert result["stage1"][0]["answer"] == "salvaged answer from prior searches"


async def test_tool_loop_echoes_reasoning_content_for_deepseek():
    """When the model returns `reasoning_content` (DeepSeek thinking-mode),
    the tool loop must put it back into the assistant message on the next
    turn — otherwise DeepSeek's follow-up call returns http 400."""
    from unittest.mock import patch
    members = [_make_members()[0]]
    seen_assistant_msgs: list[dict] = []

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            return {
                "content": json.dumps(
                    {"rankings": [{"member": "A", "score": 7, "reasoning": ""}]}
                ),
                "tokens_in": 1, "tokens_out": 1,
            }
        # Record any assistant messages already in history.
        for m in kwargs["messages"]:
            if m.get("role") == "assistant":
                seen_assistant_msgs.append(m)
        tool_msgs = [m for m in kwargs["messages"] if m.get("role") == "tool"]
        if not tool_msgs:
            return {
                "content": None,
                "tool_calls": [{
                    "id": "x",
                    "function": {"name": "web_search", "arguments": '{"query":"q"}'},
                }],
                "reasoning_content": "my chain of thought",
                "finish_reason": "tool_calls",
                "tokens_in": 1, "tokens_out": 1,
            }
        return {"content": "final", "tokens_in": 1, "tokens_out": 1}

    async def fake_search(query, num_results=5, *, api_key=None, timeout=30.0):
        return {"query": query, "results": [], "cost_dollars": 0.0, "latency_ms": 1}

    with patch("web_search_tool.web_search_exa", new=fake_search):
        # Single-member council with web_search → tool loop runs, no stage2
        # (no peers to rank against), final content propagates as the answer.
        result = await run_council(
            question="q", members=members, call_fn=fake_call, web_search=True
        )
    assert result["stage1"][0]["status"] == "ok"
    assert result["stage1"][0]["answer"] == "final"
    # After the first tool resolve, the next call should have an assistant
    # message in history that carries the reasoning_content.
    rc_msgs = [m for m in seen_assistant_msgs if m.get("reasoning_content")]
    assert rc_msgs, "no assistant message carried reasoning_content"
    assert rc_msgs[-1]["reasoning_content"] == "my chain of thought"
    assert rc_msgs[-1]["tool_calls"][0]["id"] == "x"


async def test_tool_loop_search_error_passed_back_to_model():
    """If Exa fails, the model gets an error message back and can still produce
    a final answer based on its training data."""
    from unittest.mock import patch
    from web_search import WebSearchError
    members = _make_members()
    seen_errors: list[str] = []

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            return {
                "content": json.dumps(
                    {"rankings": [{"member": "A", "score": 7, "reasoning": ""}]}
                ),
                "tokens_in": 1, "tokens_out": 1,
            }
        # First call per turn → tool. Then look at last tool message.
        msgs = kwargs["messages"]
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        if not tool_msgs:
            return {
                "content": None,
                "tool_calls": [{
                    "id": "x", "function": {"name": "web_search",
                                            "arguments": '{"query":"q"}'},
                }],
                "finish_reason": "tool_calls",
                "tokens_in": 1, "tokens_out": 1,
            }
        # The tool message contains our error formatting. Record it and finish.
        seen_errors.append(tool_msgs[-1]["content"])
        return {"content": "fallback answer", "tokens_in": 1, "tokens_out": 1}

    async def fake_search(query, num_results=5, *, api_key=None, timeout=30.0):
        raise WebSearchError("rate limited")

    with patch("web_search_tool.web_search_exa", new=fake_search):
        result = await run_council(
            question="q", members=_make_members(), call_fn=fake_call, web_search=True
        )
    # All three members reached the fallback path.
    assert seen_errors
    for err_msg in seen_errors:
        assert "rate limited" in err_msg
    for s in result["stage1"]:
        if s["status"] == "ok":
            assert "fallback" in s["answer"]
            log = s.get("tool_calls_log") or []
            assert len(log) == 1 and log[0]["ok"] is False


# ---- Stage 3 synthesis -----------------------------------------------------


async def test_run_council_synthesis_disabled_by_default():
    members = _make_members()

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            return {
                "content": json.dumps(
                    {"rankings": [{"member": "A", "score": 7, "reasoning": "ok"}]}
                ),
                "tokens_in": 1,
                "tokens_out": 1,
            }
        return {"content": "answer", "tokens_in": 1, "tokens_out": 1}

    result = await run_council(question="q", members=members, call_fn=fake_call)
    assert result["stage3"] is None


async def test_run_council_synthesis_invokes_chairman():
    """synthesis=True triggers a stage 3 call to the highest-ranked survivor."""
    members = _make_members()
    stage3_called: dict = {}

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== COUNCIL ANSWERS ===" in user_msg:
            # Stage 3 chairman call.
            stage3_called["model"] = kwargs["model"]
            stage3_called["prompt"] = user_msg
            return {
                "content": "## Synthesized answer\n\nFinal recommendation.",
                "tokens_in": 1,
                "tokens_out": 1,
            }
        if "=== ANSWERS TO RANK ===" in user_msg:
            # Stage 2: m1 ranks m2/m3, m3 wins.
            return {
                "content": json.dumps(
                    {
                        "rankings": [
                            {"member": "A", "score": 5, "reasoning": "weaker"},
                            {"member": "B", "score": 9, "reasoning": "stronger"},
                        ]
                    }
                ),
                "tokens_in": 1,
                "tokens_out": 1,
            }
        return {"content": f"answer from {kwargs['model']}", "tokens_in": 1, "tokens_out": 1}

    result = await run_council(
        question="q", members=members, call_fn=fake_call, synthesis=True
    )
    assert result["stage3"] is not None
    assert result["stage3"]["status"] == "ok"
    assert "Synthesized answer" in result["stage3"]["synthesis"]
    # Chairman should be a stage1 survivor (one of M1/M2/M3).
    assert stage3_called["model"] in {"M1", "M2", "M3"}
    # Stage 3 prompt must include the original question and the digest.
    assert "=== ORIGINAL QUESTION ===" in stage3_called["prompt"]
    assert "=== PEER RANKINGS DIGEST ===" in stage3_called["prompt"]


# ---- Stage 3 chairman web_search -------------------------------------------


async def test_synthesis_no_web_chairman_gets_no_tools():
    """synthesis=True, web_search=False → the chairman call carries no tools."""
    members = _make_members()
    stage3_tools: list = []

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== COUNCIL ANSWERS ===" in user_msg:
            stage3_tools.append(kwargs.get("tools"))
            return {"content": "## Synthesis", "tokens_in": 1, "tokens_out": 1}
        if "=== ANSWERS TO RANK ===" in user_msg:
            return {"content": json.dumps(
                {"rankings": [{"member": "A", "score": 7, "reasoning": "ok"}]}),
                "tokens_in": 1, "tokens_out": 1}
        return {"content": "answer", "tokens_in": 1, "tokens_out": 1}

    await run_council(question="q", members=members, call_fn=fake_call,
                      synthesis=True, web_search=False)
    assert stage3_tools and all(t is None for t in stage3_tools)


async def test_synthesis_web_chairman_searches_then_synthesizes():
    """synthesis=True + web_search=True → the chairman gets the web_search tool,
    can run a search, and produces a final synthesis. The chairman's search is
    counted in usage and recorded in stage3.tool_calls_log."""
    from unittest.mock import patch
    from web_search import WEB_SEARCH_TOOL_SPEC
    members = _make_members()
    stage3_tools: list = []

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== COUNCIL ANSWERS ===" in user_msg:
            stage3_tools.append(kwargs.get("tools"))
            tool_msgs = [m for m in kwargs["messages"] if m.get("role") == "tool"]
            if not tool_msgs:
                return {
                    "content": None,
                    "tool_calls": [{
                        "id": "c1",
                        "function": {"name": "web_search",
                                     "arguments": '{"query":"verify claim"}'},
                    }],
                    "finish_reason": "tool_calls",
                    "tokens_in": 1, "tokens_out": 1,
                }
            return {"content": "## Synthesis grounded in fresh sources",
                    "tokens_in": 1, "tokens_out": 1}
        if "=== ANSWERS TO RANK ===" in user_msg:
            return {"content": json.dumps(
                {"rankings": [{"member": "A", "score": 7, "reasoning": "ok"}]}),
                "tokens_in": 1, "tokens_out": 1}
        return {"content": "answer", "tokens_in": 1, "tokens_out": 1}

    async def fake_search(query, num_results=5, *, api_key=None, timeout=30.0):
        return {"query": query, "results": [
            {"title": "T", "url": "U", "summary": "S", "highlights": []}],
            "cost_dollars": 0.001, "latency_ms": 5}

    with patch("web_search_tool.web_search_exa", new=fake_search):
        result = await run_council(question="q", members=members, call_fn=fake_call,
                                   synthesis=True, web_search=True)

    assert result["stage3"]["status"] == "ok"
    assert "fresh sources" in result["stage3"]["synthesis"]
    # Chairman received the web_search tool on every stage-3 turn.
    assert stage3_tools and all(t == [WEB_SEARCH_TOOL_SPEC] for t in stage3_tools)
    # The chairman's search is recorded and counted.
    assert len(result["stage3"]["tool_calls_log"]) == 1
    assert result["stage3"]["tool_calls_log"][0]["query"] == "verify claim"
    assert result["usage"]["web_search_calls"] >= 1


async def test_synthesis_web_chairman_loop_exhaust_marks_error_not_raise():
    """A chairman that loops forever requesting searches is marked stage3 error
    (no final content), but the council still returns stage1/stage2 materials."""
    from unittest.mock import patch
    members = _make_members()

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== COUNCIL ANSWERS ===" in user_msg:
            if kwargs.get("tool_choice") == "none":
                # Even when forbidden, return another tool request → no content.
                return {"content": None, "tool_calls": [{
                    "id": "z", "function": {"name": "web_search",
                                            "arguments": '{"query":"x"}'}}],
                    "finish_reason": "tool_calls", "tokens_in": 1, "tokens_out": 1}
            return {"content": None, "tool_calls": [{
                "id": "z", "function": {"name": "web_search",
                                        "arguments": '{"query":"x"}'}}],
                "finish_reason": "tool_calls", "tokens_in": 1, "tokens_out": 1}
        if "=== ANSWERS TO RANK ===" in user_msg:
            return {"content": json.dumps(
                {"rankings": [{"member": "A", "score": 7, "reasoning": "ok"}]}),
                "tokens_in": 1, "tokens_out": 1}
        return {"content": "answer", "tokens_in": 1, "tokens_out": 1}

    async def fake_search(query, num_results=5, *, api_key=None, timeout=30.0):
        return {"query": query, "results": [], "cost_dollars": 0.0, "latency_ms": 1}

    with patch("web_search_tool.web_search_exa", new=fake_search):
        result = await run_council(question="q", members=members, call_fn=fake_call,
                                   synthesis=True, web_search=True)

    assert result["stage3"]["status"] == "error"
    assert "no final content" in result["stage3"]["error"]
    assert all(s["status"] == "ok" for s in result["stage1"])
    assert any("stage3" in n.lower() for n in result["notes"])


# ---- Stage 3 structured analysis taxonomy ----------------------------------


def test_split_synthesis_extracts_analysis_block():
    analysis = {
        "consensus": ["use a queue"],
        "contradictions": [{"topic": "DB", "stances": [{"model": "M1", "stance": "pg"}]}],
        "partial_coverage": [{"models": ["M1"], "point": "idx"}],
        "unique_insights": [{"model": "M3", "insight": "WAL"}],
        "blind_spots": ["no backups discussed"],
    }
    content = (
        "## Final answer\n\nDo X.\n\n=== ANALYSIS (JSON) ===\n```json\n"
        + json.dumps(analysis) + "\n```\n"
    )
    prose, parsed = _split_synthesis_and_analysis(content)
    assert "Do X." in prose
    assert "ANALYSIS (JSON)" not in prose  # sentinel + block stripped from prose
    assert "```json" not in prose
    assert parsed["blind_spots"] == ["no backups discussed"]
    assert parsed["consensus"] == ["use a queue"]


def test_split_synthesis_no_block_returns_full_prose_and_none():
    content = "## Final answer\n\nJust prose, no analysis."
    prose, parsed = _split_synthesis_and_analysis(content)
    assert prose == content
    assert parsed is None


def test_split_synthesis_malformed_json_degrades_to_none():
    content = "Prose body.\n\n=== ANALYSIS (JSON) ===\n```json\n{not valid json,}\n```"
    prose, parsed = _split_synthesis_and_analysis(content)
    assert parsed is None
    assert "Prose body." in prose
    assert "ANALYSIS (JSON)" not in prose  # sentinel stripped even on parse failure


def test_split_synthesis_all_empty_categories_is_none():
    content = (
        "Prose.\n\n=== ANALYSIS (JSON) ===\n```json\n"
        + json.dumps({"consensus": [], "contradictions": [], "blind_spots": []})
        + "\n```"
    )
    _prose, parsed = _split_synthesis_and_analysis(content)
    assert parsed is None


def test_split_synthesis_ignores_json_fence_in_prose():
    """A ```json sample in the prose must not be mistaken for the analysis — only
    the sentinel-anchored block counts."""
    content = (
        'Here is config:\n```json\n{"foo": 1}\n```\nThat was an example.\n\n'
        "=== ANALYSIS (JSON) ===\n```json\n"
        + json.dumps({"blind_spots": ["X"]}) + "\n```"
    )
    prose, parsed = _split_synthesis_and_analysis(content)
    assert parsed["blind_spots"] == ["X"]
    assert parsed["consensus"] == []  # normalized to the full 5-key shape
    assert '"foo": 1' in prose  # the prose code example survives


def test_split_synthesis_block_must_follow_sentinel_immediately():
    """Extraction is anchored: the fenced JSON must follow the sentinel with only
    whitespace in between. A foreign fence placed first means the analysis is NOT
    silently pulled from a later block — the prose stays intact, analysis None.

    Old greedy `re.search` would skip the python block and capture the json block
    out of position; the anchored `re.match` refuses anything but a fence right
    after the sentinel."""
    content = (
        "## Final answer\n\nDo X.\n\n=== ANALYSIS (JSON) ===\n"
        "```python\nprint('hi')\n```\n"
        "```json\n" + json.dumps({"blind_spots": ["real one"]}) + "\n```"
    )
    prose, parsed = _split_synthesis_and_analysis(content)
    # The python fence (not valid JSON) sits immediately after the sentinel, so
    # the anchored matcher tries it and fails JSON parse → degrade to None
    # instead of reaching past it to the later json block.
    assert parsed is None
    assert "Do X." in prose


async def test_synthesis_produces_structured_analysis():
    members = _make_members()
    analysis = {"consensus": ["agreed point"], "blind_spots": ["nobody addressed scaling"]}

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== COUNCIL ANSWERS ===" in user_msg:
            return {"content": ("## Synthesis\n\nDo X.\n\n=== ANALYSIS (JSON) ===\n```json\n"
                                + json.dumps(analysis) + "\n```"),
                    "tokens_in": 1, "tokens_out": 1}
        if "=== ANSWERS TO RANK ===" in user_msg:
            return {"content": json.dumps(
                {"rankings": [{"member": "A", "score": 7, "reasoning": "ok"}]}),
                "tokens_in": 1, "tokens_out": 1}
        return {"content": "answer", "tokens_in": 1, "tokens_out": 1}

    result = await run_council(question="q", members=members, call_fn=fake_call,
                               synthesis=True)
    assert result["stage3"]["analysis"]["blind_spots"] == ["nobody addressed scaling"]
    assert result["summary"]["analysis"]["consensus"] == ["agreed point"]
    # Prose synthesis is stripped of the JSON block.
    assert "Do X." in result["stage3"]["synthesis"]
    assert "ANALYSIS (JSON)" not in result["stage3"]["synthesis"]
    assert "```json" not in result["stage3"]["synthesis"]


async def test_synthesis_missing_analysis_block_notes_and_summary_none():
    members = _make_members()

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== COUNCIL ANSWERS ===" in user_msg:
            return {"content": "## Synthesis prose only, no JSON block",
                    "tokens_in": 1, "tokens_out": 1}
        if "=== ANSWERS TO RANK ===" in user_msg:
            return {"content": json.dumps(
                {"rankings": [{"member": "A", "score": 7, "reasoning": "ok"}]}),
                "tokens_in": 1, "tokens_out": 1}
        return {"content": "answer", "tokens_in": 1, "tokens_out": 1}

    result = await run_council(question="q", members=members, call_fn=fake_call,
                               synthesis=True)
    assert result["stage3"]["status"] == "ok"  # prose intact
    assert result["stage3"]["analysis"] is None
    assert result["summary"]["analysis"] is None
    assert any("analysis" in n.lower() for n in result["notes"])


async def test_summary_analysis_none_when_no_synthesis():
    members = _make_members()

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            return {"content": json.dumps(
                {"rankings": [{"member": "A", "score": 7, "reasoning": "ok"}]}),
                "tokens_in": 1, "tokens_out": 1}
        return {"content": "answer", "tokens_in": 1, "tokens_out": 1}

    result = await run_council(question="q", members=members, call_fn=fake_call)
    assert result["summary"]["analysis"] is None


# ---- Multi-round debate ----------------------------------------------------


async def test_run_council_rounds_2_does_second_pass():
    """With rounds=2, each survivor gets a second stage1 call referencing prior
    answers + critique. Final aggregate comes from round 2."""
    members = _make_members()
    counts = {"stage1_round1": 0, "stage1_round2": 0, "stage2": 0}

    async def fake_call(**kwargs):
        sys_msg = kwargs["messages"][0]["content"]
        user_msg = kwargs["messages"][1]["content"]
        if "=== ANSWERS TO RANK ===" in user_msg:
            counts["stage2"] += 1
            return {
                "content": json.dumps(
                    {"rankings": [{"member": "A", "score": 7, "reasoning": "ok"}]}
                ),
                "tokens_in": 1,
                "tokens_out": 1,
            }
        if "improved answer" in sys_msg.lower() or "=== YOUR PREVIOUS ANSWER ===" in user_msg:
            counts["stage1_round2"] += 1
            return {"content": f"round2 answer from {kwargs['model']}", "tokens_in": 1, "tokens_out": 1}
        counts["stage1_round1"] += 1
        return {"content": f"round1 answer from {kwargs['model']}", "tokens_in": 1, "tokens_out": 1}

    result = await run_council(
        question="q", members=members, call_fn=fake_call, rounds=2
    )
    # 3 members × 2 rounds for stage1.
    assert counts["stage1_round1"] == 3
    assert counts["stage1_round2"] == 3
    # 3 members × 2 rounds for stage2 (each round each survivor ranks the other 2).
    assert counts["stage2"] == 6
    # Top-level stage1 references round 2.
    assert all("round2" in s["answer"] for s in result["stage1"] if s["status"] == "ok")
    # rounds_detail has both rounds.
    assert len(result["rounds_detail"]) == 2


async def test_run_council_round2_all_fail_restores_round1_as_final():
    """If round 2 stage1 loses every survivor, the final payload must reference
    round 1 (which had live answers): stage1 ok, stage2/aggregate populated, and
    stage3 synthesis still runs. Regression for inconsistent-restore bug."""
    members = _make_members()
    round_n = {"i": 0}

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== COUNCIL ANSWERS ===" in user_msg:
            return {"content": "## Final synthesis from round 1",
                    "tokens_in": 1, "tokens_out": 1}
        if "=== ANSWERS TO RANK ===" in user_msg:
            return {"content": json.dumps(
                {"rankings": [{"member": "A", "score": 7, "reasoning": "ok"}]}),
                "tokens_in": 1, "tokens_out": 1}
        # Round 2 stage1 (carries prior answer) → every member fails.
        if "=== YOUR PREVIOUS ANSWER ===" in user_msg:
            raise CouncilHTTPError("round2 boom")
        return {"content": f"round1 answer from {kwargs['model']}",
                "tokens_in": 1, "tokens_out": 1}

    result = await run_council(
        question="q", members=members, call_fn=fake_call, rounds=2, synthesis=True
    )
    # Final stage1 is round 1's live answers (not the all-error round 2).
    assert all(s["status"] == "ok" for s in result["stage1"])
    assert all("round1" in s["answer"] for s in result["stage1"])
    # stage2 + aggregate are consistent with that stage1.
    assert result["stage2"] and all(s["status"] == "ok" for s in result["stage2"])
    assert result["aggregate"]
    # Stage 3 ran against the restored survivors instead of being skipped.
    assert result["stage3"] is not None
    assert result["stage3"]["status"] == "ok"
    assert any("no survivors" in n for n in result["notes"])


async def test_run_council_rounds_rejects_out_of_range():
    members = _make_members()

    async def fake_call(**kwargs):
        return {"content": "x", "tokens_in": 1, "tokens_out": 1}

    with pytest.raises(ValueError, match="rounds"):
        await run_council(question="q", members=members, call_fn=fake_call, rounds=0)
    with pytest.raises(ValueError, match="rounds"):
        await run_council(question="q", members=members, call_fn=fake_call, rounds=99)


async def test_run_council_synthesis_error_noted_not_raised():
    """If the chairman call fails, stage3 is recorded as error but council
    still returns stage1+stage2 results."""
    members = _make_members()

    async def fake_call(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        if "=== COUNCIL ANSWERS ===" in user_msg:
            raise CouncilHTTPError("chairman crashed")
        if "=== ANSWERS TO RANK ===" in user_msg:
            return {
                "content": json.dumps(
                    {"rankings": [{"member": "A", "score": 7, "reasoning": "ok"}]}
                ),
                "tokens_in": 1,
                "tokens_out": 1,
            }
        return {"content": "answer", "tokens_in": 1, "tokens_out": 1}

    result = await run_council(
        question="q", members=members, call_fn=fake_call, synthesis=True
    )
    assert result["stage3"]["status"] == "error"
    assert "chairman crashed" in result["stage3"]["error"]
    assert any("stage3" in n.lower() for n in result["notes"])
    # stage1 / stage2 still populated.
    assert all(s["status"] == "ok" for s in result["stage1"])
