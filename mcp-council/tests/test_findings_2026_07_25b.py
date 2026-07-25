"""Regressions for the second batch of findings resolved on 2026-07-25.

Covers: escalation gating on fixable deficits only, per-call web_search ceiling
in model_ask's path, the narrowed generic-secret DLP pattern, healthcheck error
redaction, the sandbox identity pin, refutation-angle rotation, and the async
critique job filling usage/summary.
"""

import asyncio
import os

import pytest

import adaptive
import dlp
import sandbox
from critique import REFUTE_ANGLES, pick_verifiers


# ---- should_escalate: only deficits more members can fix -------------------


def test_should_escalate_ignores_risk_only_human_review():
    """A high-risk topic is a reason for a human to look, not to re-run the
    council: risk_class depends on wording, so escalation can never clear it and
    every high-risk question burned a second full pass."""
    summary = {
        "quorum_ok": True,
        "agreement_confidence": "high",
        "human_review_required": True,   # risk_class == "high"
        "top_disagreements": [],
        "incomplete_rankings": [],
        "ranking_methods_agree": True,
    }
    escalate, _reason = adaptive.should_escalate(summary)
    assert escalate is False


@pytest.mark.parametrize("patch", [
    {"quorum_ok": False},
    {"agreement_confidence": "low"},
    {"incomplete_rankings": [{"ranker_id": "m1"}]},
    {"ranking_methods_agree": False},
    {"top_disagreements": [{"about": "x"}]},
])
def test_should_escalate_on_fixable_deficits(patch):
    summary = {
        "quorum_ok": True, "agreement_confidence": "high",
        "human_review_required": False, "top_disagreements": [],
        "incomplete_rankings": [], "ranking_methods_agree": True,
    }
    summary.update(patch)
    escalate, reason = adaptive.should_escalate(summary)
    assert escalate is True and reason


# ---- model_ask path gets the run-wide paid-search ceiling ------------------


def test_run_single_web_search_builds_a_search_cache(monkeypatch):
    """The run cap and duplicate-query dedup live in RunSearchCache. run_single
    used to pass search_cache=None, so this path had no ceiling on paid Exa
    calls at all."""
    from single_call import run_single

    monkeypatch.setenv("DEEPSEEK_KEY", "fake")
    captured = {}

    async def fake_loop(**kwargs):
        captured["search_cache"] = kwargs.get("search_cache")
        return ({"content": "answer", "finish_reason": "stop",
                 "tokens_in": 1, "tokens_out": 1}, [{"name": "web_search", "ok": True}])

    monkeypatch.setattr("single_call.run_with_tool_loop", fake_loop)
    cfg = {"id": "deepseek-flash", "model": "deepseek-v4-flash",
           "base_url": "https://x", "env_key": "DEEPSEEK_KEY"}
    sink: list[dict] = []
    out = asyncio.run(run_single(cfg, prompt="hi", max_tokens=1024,
                                 web_search=True, max_web_searches=3,
                                 tool_log_out=sink))
    assert out == "answer"
    cache = captured["search_cache"]
    assert cache is not None
    assert cache._max_searches == 3
    assert sink == [{"name": "web_search", "ok": True}]


# ---- DLP: generic key=value must look like a credential --------------------


@pytest.mark.parametrize("query", [
    "token: [REDACTED] в логе — что это значит",
    "api_key: unavailable в ответе провайдера",
    "password: смотри вики",
])
def test_generic_secret_pattern_ignores_placeholder_values(query):
    """`\\S{8,}` also matched ordinary prose, blocking legitimate searches."""
    safe, reason = dlp.scrub_outbound_query(query)
    assert reason is None and safe == query


# Credential-SHAPED fixtures, assembled at runtime. Written as literals they
# trip both the pre-commit secret guard and the CI gitleaks scan — a fake token
# that looks exactly like a real one is indistinguishable to a scanner, so the
# value never appears in the diff.
_SHAPED = "Ab3xK9zQ" + "12mnOP"


@pytest.mark.parametrize("template", [
    "api_key={} as seen in the config",
    "password: {}",
])
def test_generic_secret_pattern_still_blocks_credential_shaped_values(template):
    query = template.format(_SHAPED)
    safe, reason = dlp.scrub_outbound_query(query)
    assert safe is None and "secret" in reason


# ---- healthcheck redacts provider errors -----------------------------------


def test_healthcheck_redacts_provider_error(monkeypatch):
    """A provider error can echo a key-bearing URL; the council path redacts it,
    so the healthcheck row handed to the MCP client must too."""
    import healthcheck

    monkeypatch.setenv("OPENCODE_GO_KEY", "fake")
    # Assembled at runtime so the literal never appears in the diff (the
    # pre-commit secret guard scans for exactly this shape).
    fake_key = "sk-" + "abcdefghijklmnop"

    async def boom(**kwargs):
        raise RuntimeError(f"GET https://api/v1?api_key={fake_key} failed")

    rows = asyncio.run(healthcheck.healthcheck_models(["glm"], call_fn=boom))
    assert fake_key not in rows[0]["error"]


# ---- sandbox identity pin --------------------------------------------------


def test_read_files_rejects_identity_change(tmp_path, monkeypatch):
    """A path swapped between the path checks and the open must not be read."""
    real = tmp_path / "ctx.txt"
    real.write_text("hello", encoding="utf-8")
    other = tmp_path / "other.txt"
    other.write_text("secret-ish", encoding="utf-8")
    monkeypatch.setenv("COUNCIL_CONTEXT_ROOTS", str(tmp_path))

    real_open = os.open

    def swapping_open(path, flags, *args, **kw):
        # Simulate the swap landing after stat() but before open().
        return real_open(str(other), flags, *args, **kw)

    monkeypatch.setattr(sandbox.os, "open", swapping_open)
    with pytest.raises(sandbox.SandboxError, match="identity"):
        sandbox.read_files_with_limit([real])


def test_read_files_still_reads_unswapped_file(tmp_path, monkeypatch):
    f = tmp_path / "ctx.txt"
    f.write_text("hello", encoding="utf-8")
    monkeypatch.setenv("COUNCIL_CONTEXT_ROOTS", str(tmp_path))
    out = sandbox.read_files_with_limit([f])
    assert out[0][1] == "hello"


# ---- refutation angles rotate across findings ------------------------------


def test_refute_angles_rotate_with_finding_index():
    """With the default verifiers_per_finding=2 the angle used to be chosen by
    position alone, so only the first two of five angles were ever used."""
    members = [
        {"id": "glm", "model": "M1"},
        {"id": "gemini", "model": "M2"},
        {"id": "codex", "model": "M3"},
    ]
    finding = {"raised_by_models": [], "raised_by_domains": []}
    seen = set()
    for idx in range(len(REFUTE_ANGLES)):
        for _member, angle in pick_verifiers(finding, members, 2, angle_offset=idx):
            seen.add(angle["id"])
    assert seen == {a["id"] for a in REFUTE_ANGLES}


# ---- async critique job carries usage/summary ------------------------------


def test_run_critique_job_fills_usage_and_summary(monkeypatch):
    """council_result for an async critique returned usage=None/summary=None,
    forcing automation to parse the markdown."""
    import server
    from dialogue import state as _unused  # noqa: F401 — import side effects only

    async def fake_do_critique(*args, **kwargs):
        return ("# md", {"llm_calls": 4}, {"findings_kept": 2, "panel_quorum_ok": True})

    monkeypatch.setattr(server, "_do_critique_async", fake_do_critique)

    async def scenario():
        state = await server.job_state.create_job(
            question_preview="s", synthesis=False, rounds=1,
        )
        await server._run_critique_job(
            state, "subject", [], 1024, [{"id": "glm", "model": "M"}],
            ["correctness"], 2, 24, False,
        )
        return state

    state = asyncio.run(scenario())
    assert state.phase == "done"
    assert state.usage == {"llm_calls": 4}
    assert state.summary["findings_kept"] == 2
