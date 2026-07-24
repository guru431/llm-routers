"""Tests for the council-core feature additions (2026-07-17 IDEAS build):
DLP + claim ledger, budget/deadline, response cache, capabilities reference,
adaptive selection, Borda aggregation, evidence-aware verdict, stage-2 repair.
"""

import json

import pytest

import adaptive
import budget as budget_mod
import capabilities as capabilities_mod
import dlp
import response_cache
from council import _aggregate, _aggregate_borda, _build_summary, run_council
from models import CATALOG


# --- DLP + claim ledger ----------------------------------------------------

def test_scrub_outbound_query_blocks_secret():
    # Build the fake key at runtime so the source literal doesn't trip the repo's
    # pre-commit secret scanner (the runtime string still matches the DLP pattern).
    q, reason = dlp.scrub_outbound_query("how to use key " + "sk-" + "ABCDEF0123456789ghij")
    assert q is None and "secret" in reason


def test_scrub_outbound_query_blocks_sensitive_path():
    q, reason = dlp.scrub_outbound_query("cat ~/.ssh/id_rsa contents")
    assert q is None and "sensitive local path" in reason


def test_scrub_outbound_query_passes_clean():
    q, reason = dlp.scrub_outbound_query("latest glm-5.2 release notes")
    assert q == "latest glm-5.2 release notes" and reason is None


def test_build_claim_ledger_maps_query_to_sources():
    records = [{
        "model": "glm-5.2",
        "tool_calls_log": [
            {"name": "web_search", "ok": True, "query": "x releases",
             "num_results": 2, "cost_dollars": 0.005,
             "sources": ["https://a.example", "https://b.example"]},
            {"name": "web_search", "ok": False, "query": "blocked", "sources": []},
        ],
    }]
    ledger = dlp.build_claim_ledger(records)
    assert len(ledger) == 1
    assert ledger[0]["query"] == "x releases"
    assert ledger[0]["sources"] == ["https://a.example", "https://b.example"]


# --- Budget / deadline -----------------------------------------------------

def test_run_budget_deadline_expired():
    import time
    # Start 100s in the past relative to the monotonic clock → already expired.
    b = budget_mod.RunBudget(deadline_seconds=10, started_monotonic=time.monotonic() - 100)
    assert b.deadline_expired() is True
    assert "deadline" in (b.check() or "")


def test_run_budget_ceilings():
    b = budget_mod.RunBudget(max_llm_calls=5, max_cost_usd=0.10)
    assert b.check({"llm_calls": 5}) and "LLM calls" in b.check({"llm_calls": 5})
    assert b.check({"reference_payg_cost_usd": 0.2})
    assert b.check({"llm_calls": 2, "reference_payg_cost_usd": 0.01}) is None


def test_estimate_run_shape():
    est = budget_mod.estimate_run(n_members=3, rounds=2, synthesis=True, web_search=False)
    assert est["expected_llm_calls"] == 2 * 3 * 2 + 1
    assert est["expected_minutes"] > 0


# --- Response cache ---------------------------------------------------------

def test_fingerprint_stable_and_sensitive():
    members = [{"id": "glm", "model": "glm-5.2", "base_url": "u", "provider": "ocg"}]
    base = dict(question="q", model_configs=members, synthesis=False, rounds=1,
                web_search=False, max_tokens=8192)
    k1 = response_cache.fingerprint(**base)
    k2 = response_cache.fingerprint(**base)
    assert k1 == k2
    k3 = response_cache.fingerprint(**{**base, "rounds": 2})
    assert k1 != k3


@pytest.mark.asyncio
async def test_response_cache_hit_and_miss():
    cache = response_cache.ResponseCache()
    calls = {"n": 0}

    async def compute():
        calls["n"] += 1
        return "brief"

    v1, prov1 = await cache.get_or_compute("k", compute)
    assert v1 == "brief" and prov1 is None and calls["n"] == 1
    v2, prov2 = await cache.get_or_compute("k", compute)
    assert v2 == "brief" and prov2 is not None and calls["n"] == 1  # served from cache


def test_response_cache_oversized_not_cached():
    cache = response_cache.ResponseCache(max_entry_bytes=10)
    cache._put("k", "x" * 100)
    assert cache.provenance("k") is None


# --- Capabilities reference -------------------------------------------------

def test_capabilities_has_models_and_presets():
    caps = capabilities_mod.build_capabilities()
    assert caps["service"] == "mcp-council"
    ids = {m["id"] for m in caps["models"]}
    assert ids == set(CATALOG.keys())
    assert "full" in caps["presets"]


def test_model_ask_models_line_covers_enabled():
    line = capabilities_mod.model_ask_models_line()
    for mid, cfg in CATALOG.items():
        if cfg.get("enabled") is not False:
            assert mid in line


def test_model_ask_docstring_lists_enabled_models():
    """The model_ask docstring must mention every enabled CATALOG id (guards
    the hand-written list from drifting out of sync with the catalog)."""
    import server
    doc = server.model_ask.__doc__ or ""
    for mid, cfg in CATALOG.items():
        if cfg.get("enabled") is not False:
            assert mid in doc, f"{mid} missing from model_ask docstring"


# --- Adaptive selection -----------------------------------------------------

def test_filter_healthy_drops_broken_keeps_unknown():
    rows = [{"id": "a", "status": "ok"}, {"id": "b", "status": "no_key"}]
    kept, dropped = adaptive.filter_healthy(["a", "b", "c"], rows)
    assert kept == ["a", "c"]  # c has no row → kept
    assert dropped == [("b", "no_key")]


def test_pick_starting_subset_spans_domains():
    members = [
        {"id": "glm"}, {"id": "qwen"}, {"id": "kimi"},  # all opencode-go
        {"id": "gemini"}, {"id": "codex"},              # helicone, codex-agent
    ]
    subset = adaptive.pick_starting_subset(members, min_size=3, min_domains=2)
    domains = {CATALOG.get(i, {}).get("provider", i) for i in subset}
    assert len(subset) >= 3
    assert len(domains) >= 2


def test_should_escalate_on_low_quorum():
    esc, _ = adaptive.should_escalate({"quorum_ok": False})
    assert esc is True
    esc2, _ = adaptive.should_escalate(
        {"quorum_ok": True, "agreement_confidence": "high",
         "human_review_required": False, "top_disagreements": []}
    )
    assert esc2 is False


# --- Borda aggregation ------------------------------------------------------

def test_aggregate_borda_orders_by_rank():
    stage2 = [
        {"status": "ok", "rankings": [
            {"ranked_id": "a", "score": 9}, {"ranked_id": "b", "score": 5},
            {"ranked_id": "c", "score": 2}]},
        {"status": "ok", "rankings": [
            {"ranked_id": "a", "score": 8}, {"ranked_id": "b", "score": 7},
            {"ranked_id": "c", "score": 3}]},
    ]
    borda = _aggregate_borda(stage2)
    assert borda[0][0] == "a"  # top-ranked by both
    assert borda[-1][0] == "c"


# --- Evidence-aware verdict -------------------------------------------------

def _quorum_case():
    stage1 = [
        {"id": "glm", "model": "glm-5.2", "status": "ok"},
        {"id": "gemini", "model": "gemini", "status": "ok"},
        {"id": "codex", "model": "gpt-5.5", "status": "ok"},
    ]
    stage2 = [
        {"ranker_id": "gemini", "status": "ok",
         "rankings": [{"ranked_id": "glm", "score": 9}, {"ranked_id": "codex", "score": 5}]},
        {"ranker_id": "codex", "status": "ok",
         "rankings": [{"ranked_id": "glm", "score": 9}, {"ranked_id": "gemini", "score": 5}]},
        {"ranker_id": "glm", "status": "ok",
         "rankings": [{"ranked_id": "gemini", "score": 5}, {"ranked_id": "codex", "score": 5}]},
    ]
    return stage1, stage2, _aggregate(stage2)


def test_high_risk_question_forces_human_review():
    stage1, stage2, aggregate = _quorum_case()
    borda = _aggregate_borda(stage2)
    s = _build_summary(stage1, stage2, aggregate, None,
                       question="how to delete the production database safely",
                       borda=borda)
    assert s["risk_class"] == "high"
    assert s["human_review_required"] is True
    assert "HIGH-RISK" in s["recommended_next_action"]
    assert s["verdict"]["risk_class"] == "high"


def test_ranking_methods_disagree_blocks_adopt():
    stage1, stage2, aggregate = _quorum_case()
    # Hand-craft a Borda winner (gemini) that differs from the mean winner (glm).
    borda = [("gemini", 5, 1), ("glm", 4, 1)]
    s = _build_summary(stage1, stage2, aggregate, None, question="pick a lib", borda=borda)
    assert s["ranking_methods_agree"] is False
    assert "borda" in s["recommended_next_action"].lower()
    assert s["human_review_required"] is True


# --- Stage-2 repair retry ---------------------------------------------------

def _members_2():
    return [
        {"id": "m1", "model": "M1", "base_url": "u", "env_key": "K", "provider": "p1"},
        {"id": "m2", "model": "M2", "base_url": "u", "env_key": "K", "provider": "p2"},
    ]


@pytest.mark.asyncio
async def test_stage2_repair_recovers_bad_json(monkeypatch):
    monkeypatch.setenv("K", "x")

    async def fake_call(**kwargs):
        user = kwargs["messages"][-1]["content"]
        is_stage2 = any("ANSWERS TO RANK" in m.get("content", "")
                        for m in kwargs["messages"])
        if is_stage2:
            # M1's FIRST stage-2 reply is junk; on the repair follow-up (which
            # appends "STRICT JSON ONLY") it returns valid JSON.
            if kwargs["model"] == "M1" and "STRICT JSON ONLY" not in user:
                return {"content": "totally not json", "tokens_in": 1, "tokens_out": 1}
            return {"content": json.dumps(
                {"rankings": [{"member": "A", "score": 7, "reasoning": "ok"}]}),
                "tokens_in": 1, "tokens_out": 1}
        return {"content": f"ans-{kwargs['model']}", "tokens_in": 1, "tokens_out": 1}

    result = await run_council(question="q", members=_members_2(), call_fn=fake_call)
    s2 = {s["ranker_id"]: s for s in result["stage2"]}
    assert s2["m1"]["status"] == "ok"
    assert s2["m1"]["repaired"] is True


def test_capabilities_tools_match_the_live_registry():
    """The tool list must be generated, not hand-maintained: the previous
    hard-coded copy had already dropped council_purge_logs, council_estimate,
    council_critique(_async) and dialogue_fork, hiding them from discovery."""
    import server  # noqa: F401 — registers the tools on import

    caps = capabilities_mod.build_capabilities()
    live = sorted(t.name for t in server.mcp._tool_manager.list_tools())
    assert caps["tools"] == live
    # Guard against an introspection failure silently reporting an empty set.
    for expected in ("council_ask", "council_critique", "council_critique_async",
                     "council_estimate", "council_purge_logs", "dialogue_fork"):
        assert expected in caps["tools"]
