"""Tests for the adversarial critique mode: lenses, dedup, verification, verdict."""

import json

import pytest

from critique import (
    CritiqueParseError,
    _apply_verdicts,
    build_critique_summary,
    cluster_findings,
    format_critique_markdown,
    normalize_findings,
    normalize_verdict,
    pick_verifiers,
    run_critique,
)
from lenses import (
    LENS_PRESETS,
    LENSES,
    UnknownLensError,
    UnknownLensPresetError,
    assign_lenses,
    resolve_lenses,
)
from openai_client import CouncilHTTPError


@pytest.fixture(autouse=True)
def env_keys(monkeypatch):
    monkeypatch.setenv("K1", "sk-test")
    monkeypatch.setenv("OPENCODE_GO_KEY", "sk-test")
    monkeypatch.setenv("HELICONE_GATEWAY_KEY", "sk-test")


def _members(n=3):
    return [
        {"id": f"m{i}", "model": f"M{i}", "base_url": "u", "env_key": "K1"}
        for i in range(1, n + 1)
    ]


def _finding(**kw):
    base = {
        "title": "t", "severity": "high", "location": "a.py:1",
        "claim": "c", "failure_scenario": "s", "fix": "f", "confidence": 7,
        "raised_by": [{"model": "m1", "lens": "correctness", "severity": "high",
                       "confidence": 7}],
        "lenses": ["correctness"], "raised_by_models": ["m1"],
        "raised_by_domains": ["m1"], "duplicates": [],
    }
    base.update(kw)
    return base


# ---- lens registry ---------------------------------------------------------


def test_every_lens_declares_focus_and_out_of_scope():
    # The out_of_scope half is what keeps critics from converging — a lens
    # without it silently degrades the whole mode to N generic reviewers.
    for lid, lens in LENSES.items():
        assert lens["focus"].strip(), lid
        assert lens["out_of_scope"].strip(), lid
        assert lens["title"].strip(), lid


def test_every_preset_references_known_lenses():
    for name, ids in LENS_PRESETS.items():
        assert all(i in LENSES for i in ids), name
        assert len(ids) >= 2, name


def test_resolve_lenses_defaults_to_code_review():
    assert resolve_lenses(None, None) == LENS_PRESETS["code-review"]


def test_resolve_lenses_preset_and_explicit_are_mutually_exclusive():
    with pytest.raises(RuntimeError, match="not both"):
        resolve_lenses(["security"], "fast-3")


def test_resolve_lenses_drops_duplicates():
    assert resolve_lenses(["security", "security", "correctness"], None) == [
        "security", "correctness",
    ]


def test_resolve_lenses_rejects_unknown_and_too_few():
    with pytest.raises(UnknownLensError):
        resolve_lenses(["nope", "security"], None)
    with pytest.raises(UnknownLensPresetError):
        resolve_lenses(None, "nope")
    with pytest.raises(RuntimeError, match="at least 2"):
        resolve_lenses(["security"], None)


def test_assign_lenses_spreads_across_provider_domains():
    # glm/qwen share the OCG domain, gemini is Helicone. Two lenses must land on
    # two DIFFERENT domains, otherwise a single gateway outage takes the panel.
    members = [
        {"id": "glm", "model": "glm", "base_url": "u", "env_key": "K1"},
        {"id": "qwen", "model": "qwen", "base_url": "u", "env_key": "K1"},
        {"id": "gemini", "model": "gemini", "base_url": "u", "env_key": "K1"},
    ]
    got = assign_lenses(["correctness", "security"], members)
    assert {a["member"]["id"] for a in got} == {"glm", "gemini"}


def test_assign_lenses_wraps_when_lenses_outnumber_members():
    got = assign_lenses(["correctness", "security", "testing"], _members(2))
    assert [a["member"]["id"] for a in got] == ["m1", "m2", "m1"]


# ---- parsing ---------------------------------------------------------------


def test_normalize_findings_happy_path():
    out = normalize_findings(json.dumps({"findings": [{
        "title": "Race on _jobs", "severity": "CRITICAL", "location": "state.py:88",
        "claim": "unlocked read", "failure_scenario": "two cancels overlap",
        "fix": "hold the lock", "confidence": 9,
    }]}))
    assert out[0]["severity"] == "critical"
    assert out[0]["confidence"] == 9


def test_normalize_findings_empty_list_is_valid():
    # "Nothing in my lane" is a real answer and must not look like a failure —
    # otherwise a clean lens reads as a broken critic.
    assert normalize_findings('{"findings": []}') == []


def test_normalize_findings_drops_unlocatable_entries():
    out = normalize_findings(json.dumps({"findings": [
        {"title": "vague", "claim": "something is off", "failure_scenario": "x"},
        {"title": "ok", "location": "a.py", "claim": "c", "failure_scenario": "s"},
    ]}))
    assert [f["title"] for f in out] == ["ok"]


def test_normalize_findings_unknown_severity_falls_back_to_medium():
    out = normalize_findings(json.dumps({"findings": [{
        "title": "t", "severity": "catastrophic", "location": "a.py",
        "claim": "c", "failure_scenario": "s",
    }]}))
    assert out[0]["severity"] == "medium"


def test_normalize_findings_wrong_shape_raises():
    with pytest.raises(CritiqueParseError):
        normalize_findings('{"findings": "lots"}')


def test_normalize_verdict_accepts_bool_and_string():
    assert normalize_verdict('{"refuted": true, "confidence": 8}')["refuted"] is True
    assert normalize_verdict('{"refuted": "no"}')["refuted"] is False


def test_normalize_verdict_missing_refuted_raises():
    # Must NOT default to "not refuted": a broken verifier would then confirm
    # every finding it was handed.
    with pytest.raises(CritiqueParseError):
        normalize_verdict('{"confidence": 9, "reasoning": "hmm"}')


# ---- dedup -----------------------------------------------------------------


def _critic(model, lens, findings):
    return {"id": model, "model": model.upper(), "status": "ok",
            "lens": lens, "findings": findings}


def test_cluster_merges_same_defect_seen_by_two_lenses():
    f1 = {"title": "Unlocked read of shared job dict", "severity": "high",
          "location": "state.py:88", "claim": "the jobs dict is read outside the lock",
          "failure_scenario": "s", "fix": "", "confidence": 8}
    f2 = {"title": "Shared jobs dict read without lock", "severity": "critical",
          "location": "state.py:91", "claim": "reading jobs dict outside its lock",
          "failure_scenario": "s2", "fix": "", "confidence": 9}
    merged = cluster_findings([
        _critic("m1", "concurrency", [f1]),
        _critic("m2", "correctness", [f2]),
    ])
    assert len(merged) == 1
    assert sorted(merged[0]["lenses"]) == ["concurrency", "correctness"]
    # Representative is the most severe statement of the defect.
    assert merged[0]["severity"] == "critical"
    assert merged[0]["duplicates"]


def test_cluster_keeps_unrelated_findings_apart():
    f1 = {"title": "SQL injection in search", "severity": "critical",
          "location": "db.py:10", "claim": "user input concatenated into query",
          "failure_scenario": "s", "fix": "", "confidence": 9}
    f2 = {"title": "Timeout missing on upload", "severity": "medium",
          "location": "http.py:44", "claim": "no timeout passed to the request",
          "failure_scenario": "s", "fix": "", "confidence": 6}
    merged = cluster_findings([_critic("m1", "security", [f1]),
                              _critic("m2", "failure-modes", [f2])])
    assert len(merged) == 2


def test_cluster_skips_failed_critics():
    merged = cluster_findings([
        {"id": "m1", "model": "M1", "status": "error", "lens": "security",
         "findings": [], "error": "boom"},
    ])
    assert merged == []


def test_cluster_sorts_severity_first():
    low = {"title": "naming", "severity": "low", "location": "a.py",
           "claim": "x", "failure_scenario": "s", "fix": "", "confidence": 5}
    crit = {"title": "data loss on restart", "severity": "critical",
            "location": "b.py", "claim": "y", "failure_scenario": "s",
            "fix": "", "confidence": 9}
    merged = cluster_findings([_critic("m1", "simplicity", [low]),
                              _critic("m2", "data-integrity", [crit])])
    assert [f["severity"] for f in merged] == ["critical", "low"]


# ---- verifier selection ----------------------------------------------------


def test_pick_verifiers_excludes_the_model_that_raised_it():
    members = _members(3)
    picked = pick_verifiers(_finding(), members, 2)
    assert "m1" not in {m["id"] for m, _ in picked}
    assert len({a["id"] for _, a in picked}) == 2  # distinct attack angles


def test_pick_verifiers_falls_back_to_raiser_when_pool_too_small():
    members = _members(2)
    picked = pick_verifiers(
        _finding(raised_by_models=["m1", "m2"], raised_by_domains=["m1", "m2"]),
        members, 2,
    )
    assert len(picked) == 2  # self-review is allowed, but flagged downstream


def test_pick_verifiers_prefers_a_domain_the_raiser_does_not_share():
    members = [
        {"id": "glm", "model": "glm", "base_url": "u", "env_key": "K1"},
        {"id": "qwen", "model": "qwen", "base_url": "u", "env_key": "K1"},
        {"id": "gemini", "model": "gemini", "base_url": "u", "env_key": "K1"},
    ]
    picked = pick_verifiers(
        _finding(raised_by_models=["glm"], raised_by_domains=["opencode-go"]),
        members, 1,
    )
    assert picked[0][0]["id"] == "gemini"


# ---- verdict folding -------------------------------------------------------


def _verdict(mid, refuted, self_review=False):
    return {"id": mid, "status": "ok", "refuted": refuted,
            "self_review": self_review, "verdict_confidence": 8}


def test_apply_verdicts_statuses():
    f = _finding()
    assert _apply_verdicts(f, [_verdict("m2", False), _verdict("m3", False)])["status"] == "confirmed"
    assert _apply_verdicts(f, [_verdict("m2", True), _verdict("m3", True)])["status"] == "refuted"
    # Half counts as refuted: verifiers only say "not refuted" when the claim is
    # clearly supported, so a split panel means it never cleared that bar.
    assert _apply_verdicts(f, [_verdict("m2", True), _verdict("m3", False)])["status"] == "refuted"
    assert _apply_verdicts(
        f, [_verdict("m2", True), _verdict("m3", False), _verdict("m4", False)]
    )["status"] == "contested"
    assert _apply_verdicts(f, [])["status"] == "unverified"


def test_apply_verdicts_ignores_errored_verifiers():
    f = _apply_verdicts(_finding(), [
        _verdict("m2", False),
        {"id": "m3", "status": "error", "refuted": None, "error": "boom"},
    ])
    assert f["status"] == "confirmed"
    assert f["verifier_count"] == 1


def test_apply_verdicts_self_review_does_not_count_as_independent():
    f = _apply_verdicts(_finding(), [
        _verdict("m1", False, self_review=True),
        _verdict("m2", False),
    ])
    assert f["independent_verifiers"] == 1
    assert f["verification_quorum_ok"] is False


# ---- end-to-end ------------------------------------------------------------


def _fake_backend(findings_by_model=None, refute=lambda title: False):
    """Build a call_fn that answers as a critic or a verifier by system prompt."""
    findings_by_model = findings_by_model or {}

    async def fake_call(**kwargs):
        system = kwargs["messages"][0]["content"]
        model = kwargs["model"]
        if "YOUR LENS" in system:
            return {"content": json.dumps(
                {"findings": findings_by_model.get(model, [])}
            ), "tokens_in": 10, "tokens_out": 5, "attempts": 1}
        # Verifier: the claim title is echoed in the user message.
        user = kwargs["messages"][-1]["content"]
        title = user.split("Title: ", 1)[1].split("\n", 1)[0]
        return {"content": json.dumps({
            "refuted": refute(title), "confidence": 8,
            "reasoning": "checked the call path",
            "strongest_counterpoint": "could be reachable via an untested path",
        }), "tokens_in": 10, "tokens_out": 5, "attempts": 1}

    return fake_call


def _f(title, severity="high", location="a.py:1"):
    return {"title": title, "severity": severity, "location": location,
            "claim": f"claim about {title}", "failure_scenario": "boom",
            "fix": "fix it", "confidence": 8}


async def test_run_critique_end_to_end_keeps_and_refutes():
    members = _members(3)
    backend = _fake_backend(
        findings_by_model={
            "M1": [_f("real defect")],
            "M2": [_f("imaginary defect", location="z.py:9")],
        },
        refute=lambda title: title == "imaginary defect",
    )
    result = await run_critique(
        subject="review this", members=members,
        lens_ids=["correctness", "security", "testing"], call_fn=backend,
    )
    by_title = {f["title"]: f for f in result["findings"]}
    assert by_title["real defect"]["status"] == "confirmed"
    assert by_title["imaginary defect"]["status"] == "refuted"
    s = result["summary"]
    assert s["findings_kept"] == 1
    assert s["findings_refuted"] == 1
    assert s["human_review_required"] is True
    assert result["usage"]["llm_calls"] > 0


async def test_run_critique_verifiers_never_review_their_own_finding():
    members = _members(3)
    backend = _fake_backend(findings_by_model={"M1": [_f("only finding")]})
    result = await run_critique(
        subject="s", members=members, lens_ids=["correctness", "security"],
        call_fn=backend, verifiers_per_finding=2,
    )
    assert result["verifiers"]
    assert all(v["self_review"] is False for v in result["verifiers"])


async def test_run_critique_all_critics_fail_raises():
    async def boom(**kwargs):
        raise CouncilHTTPError("everyone fails")

    with pytest.raises(RuntimeError, match="critique fully failed"):
        await run_critique(
            subject="s", members=_members(2),
            lens_ids=["correctness", "security"], call_fn=boom,
        )


async def test_run_critique_surviving_critic_carries_the_run():
    members = _members(2)
    inner = _fake_backend(findings_by_model={"M2": [_f("found by m2")]})

    async def flaky(**kwargs):
        if kwargs["model"] == "M1":
            raise CouncilHTTPError("m1 down")
        return await inner(**kwargs)

    result = await run_critique(
        subject="s", members=members, lens_ids=["correctness", "security"],
        call_fn=flaky, verifiers_per_finding=0,
    )
    assert result["summary"]["critics_ok"] == 1
    assert result["summary"]["failed_critics"][0]["id"] == "m1"
    assert any("m1" in n for n in result["notes"])


async def test_run_critique_zero_verifiers_marks_findings_unverified():
    backend = _fake_backend(findings_by_model={"M1": [_f("unchecked")]})
    result = await run_critique(
        subject="s", members=_members(2), lens_ids=["correctness", "security"],
        call_fn=backend, verifiers_per_finding=0,
    )
    assert result["verifiers"] == []
    assert result["findings"][0]["status"] == "unverified"
    assert any("verification skipped" in n for n in result["notes"])


async def test_run_critique_overflow_is_reported_not_dropped():
    # A silent cap reads as "that was everything" — the cap must be visible in
    # both the notes and a dedicated unverified bucket.
    many = [_f(f"defect number {i}", severity="low", location=f"f{i}.py:1")
            for i in range(5)]
    backend = _fake_backend(findings_by_model={"M1": many})
    result = await run_critique(
        subject="s", members=_members(2), lens_ids=["correctness", "security"],
        call_fn=backend, max_verified_findings=2,
    )
    assert len(result["findings"]) == 2
    assert len(result["unverified_findings"]) == 3
    assert result["summary"]["unverified_findings"] == 3
    assert any("max_verified_findings" in n for n in result["notes"])


async def test_run_critique_rejects_bad_verifier_count():
    with pytest.raises(ValueError):
        await run_critique(
            subject="s", members=_members(2),
            lens_ids=["correctness", "security"], verifiers_per_finding=99,
        )


async def test_run_critique_progress_events_cover_both_stages():
    seen = []
    backend = _fake_backend(findings_by_model={"M1": [_f("x")]})
    await run_critique(
        subject="s", members=_members(2), lens_ids=["correctness", "security"],
        call_fn=backend, on_progress=lambda t, p: seen.append((t, p)),
    )
    types = {t for t, _ in seen}
    assert {"phase", "stage1_member", "stage2_ranker"} <= types
    phases = [p["phase"] for t, p in seen if t == "phase"]
    assert phases[0] == "critique" and phases[-1] == "done"


async def test_single_provider_panel_is_not_reported_as_corroborated():
    # All three test ids are unknown to CATALOG, so each is its own domain; use
    # two real OCG ids to build a genuinely single-domain panel.
    members = [
        {"id": "glm", "model": "GLM", "base_url": "u", "env_key": "K1"},
        {"id": "qwen", "model": "QWEN", "base_url": "u", "env_key": "K1"},
    ]
    backend = _fake_backend(findings_by_model={"GLM": [_f("critical thing", "critical")]})
    result = await run_critique(
        subject="s", members=members, lens_ids=["correctness", "security"],
        call_fn=backend,
    )
    s = result["summary"]
    assert s["single_provider"] is True
    assert s["panel_quorum_ok"] is False
    assert "not an independent review" in format_critique_markdown("s", result).lower()


# ---- summary / rendering ---------------------------------------------------


def test_summary_separates_found_nothing_from_all_refuted():
    # These are different outcomes and must not collapse into one message: in the
    # second case the critics did engage and the refutations are the artifact.
    # Two provider domains, so the panel-quorum branch (which takes precedence)
    # does not fire and the found-nothing / all-refuted distinction is reachable.
    def _c(mid, lens, findings):
        return {"id": mid, "model": mid.upper(), "status": "ok",
                "lens": lens, "findings": findings}

    critics = [_c("glm", "correctness", [{"title": "x"}]), _c("gemini", "security", [])]
    nothing = build_critique_summary(
        [_c("glm", "correctness", []), _c("gemini", "security", [])],
        [], [], ["correctness", "security"],
    )
    assert "No critic raised anything" in nothing["recommended_next_action"]
    assert nothing["lenses_with_findings"] == []

    all_refuted = build_critique_summary(
        critics,
        [_apply_verdicts(_finding(), [_verdict("m2", True)])],
        [], ["correctness", "security"],
    )
    assert "refuted by verification" in all_refuted["recommended_next_action"]
    assert all_refuted["lenses_with_findings"] == ["correctness"]
    assert all_refuted["lenses_with_surviving_findings"] == []


def test_summary_counts_cross_lens_corroboration():
    verified = [
        _apply_verdicts(_finding(lenses=["a", "b"]), [_verdict("m2", False)]),
        _apply_verdicts(_finding(title="solo"), [_verdict("m2", False)]),
    ]
    s = build_critique_summary(
        [{"id": "m1", "model": "M1", "status": "ok", "lens": "correctness", "findings": []}],
        verified, [], ["correctness", "security"],
    )
    assert s["cross_lens_corroborated"] == 1


def test_markdown_lists_refuted_findings_so_they_stay_visible():
    result = {
        "summary": build_critique_summary([], [], [], ["correctness", "security"]),
        "findings": [_apply_verdicts(_finding(title="dropped"), [_verdict("m2", True)])],
        "unverified_findings": [], "notes": [], "usage": {},
    }
    md = format_critique_markdown("subject", result)
    assert "## Refuted (1)" in md
    assert "dropped" in md
