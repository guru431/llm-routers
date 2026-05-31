"""Tests for server module: markdown formatter, sandbox integration."""

import pytest

import server
from sandbox import SandboxError


def _stub_result():
    return {
        "stage1": [
            {
                "id": "m1",
                "model": "Model-1",
                "status": "ok",
                "error": None,
                "answer": "answer one",
                "latency_ms": 12000,
                "tokens_in": 100,
                "tokens_out": 50,
            },
            {
                "id": "m2",
                "model": "Model-2",
                "status": "error",
                "error": "timeout",
                "answer": None,
                "latency_ms": 120000,
                "tokens_in": None,
                "tokens_out": None,
            },
            {
                "id": "m3",
                "model": "Model-3",
                "status": "ok",
                "error": None,
                "answer": "answer three",
                "latency_ms": 30000,
                "tokens_in": 100,
                "tokens_out": 60,
            },
        ],
        "stage2": [
            {
                "ranker_id": "m1",
                "status": "ok",
                "error": None,
                "rankings": [
                    {"ranked_id": "m3", "pseudonym": "B", "score": 8, "reasoning": "solid"},
                ],
                "pseudonyms": {"m3": "B"},
                "latency_ms": 20000,
            },
            {
                "ranker_id": "m3",
                "status": "ok",
                "error": None,
                "rankings": [
                    {"ranked_id": "m1", "pseudonym": "A", "score": 9, "reasoning": "great"},
                ],
                "pseudonyms": {"m1": "A"},
                "latency_ms": 20000,
            },
        ],
        "aggregate": [("m1", 9.0, 1), ("m3", 8.0, 1)],
        "notes": ["m2 (Model-2): stage1 error — timeout; excluded from both stages"],
    }


def test_format_markdown_contains_required_sections():
    md = server.format_markdown("test question", _stub_result())
    assert "# Council deliberation" in md
    assert "## Question" in md
    assert "test question" in md
    assert "## Stage 1: Independent answers" in md
    assert "## Stage 2: Peer rankings" in md
    assert "## Aggregate scores" in md
    assert "## Notes" in md
    assert "Now synthesize the final answer based on these materials." in md


def test_format_markdown_renders_de_anonymized_members():
    md = server.format_markdown("q", _stub_result())
    # Display letters are assigned in stage1 order: m1→A, m2→B, m3→C.
    assert "Member A (Model-1)" in md
    assert "Member B (Model-2)" in md
    assert "Member C (Model-3)" in md


def test_format_markdown_shows_error_for_failed_member():
    md = server.format_markdown("q", _stub_result())
    assert "error: timeout" in md
    assert "_(no answer)_" in md


def test_format_markdown_aggregate_sorted_desc():
    md = server.format_markdown("q", _stub_result())
    # m1 (9.0) should appear before m3 (8.0) in aggregate section
    agg_section = md.split("## Aggregate scores")[1].split("## Notes")[0]
    pos_m1 = agg_section.find("Model-1")
    pos_m3 = agg_section.find("Model-3")
    assert 0 <= pos_m1 < pos_m3


def test_format_markdown_empty_stage2():
    res = _stub_result()
    res["stage2"] = []
    res["aggregate"] = []
    md = server.format_markdown("q", res)
    assert "stage 2 skipped" in md


def test_do_council_ask_sandbox_error_logged(monkeypatch, tmp_path):
    """If sandbox raises, we get RuntimeError and log_call is called once."""
    logged: list[dict] = []
    monkeypatch.setattr(server, "log_call", lambda **kw: logged.append(kw))

    def fake_resolve(paths):
        raise SandboxError("blocked")

    monkeypatch.setattr(server, "resolve_and_validate", fake_resolve)

    with pytest.raises(RuntimeError, match="sandbox"):
        server._do_council_ask("q", ["/some/blocked/path"], 1024)

    assert len(logged) == 1
    assert "sandbox" in logged[0]["status"]


def test_do_council_ask_clamps_tokens(monkeypatch, tmp_path):
    """max_response_tokens passed above hard cap is clamped to 16384."""
    from pathlib import Path as _Path
    captured: dict = {}

    async def fake_run_council(question, files_section, max_response_tokens, synthesis=False, rounds=1, web_search=False, members=None, **_kw):
        captured["max"] = max_response_tokens
        captured["synthesis"] = synthesis
        captured["rounds"] = rounds
        captured["web_search"] = web_search
        return _stub_result()

    monkeypatch.setattr(server, "run_council", fake_run_council)
    # Dump path must be under server.__file__'s parent for relative_to to work;
    # patch write_full_dump to return such a path under tmp_path is no good →
    # patch __file__-based parent by monkey-patching server.Path import.
    server_dir = _Path(server.__file__).parent
    fake_dump = server_dir / "logs" / "calls" / "stub.json"
    fake_dump.parent.mkdir(parents=True, exist_ok=True)
    fake_dump.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(server, "write_full_dump", lambda cid, dump: fake_dump)
    monkeypatch.setattr(server, "log_call", lambda **kw: None)

    server._do_council_ask("q", [], 999999)
    assert captured["max"] == 16384

    server._do_council_ask("q", [], 0)
    assert captured["max"] == 1

    # cleanup
    fake_dump.unlink(missing_ok=True)


# -------------------------------------------------------------------------
# Task 4: models param + ≥2 validation
# -------------------------------------------------------------------------

import asyncio


def test_council_ask_subset_passes_resolved_members(monkeypatch, tmp_path):
    """council_ask with models=[...] resolves the subset and passes it to run_council."""
    from pathlib import Path as _Path
    captured = {}

    async def fake_run_council(**kwargs):
        captured["members"] = kwargs.get("members")
        return _stub_result()

    monkeypatch.setattr(server, "run_council", fake_run_council)
    server_dir = _Path(server.__file__).parent
    fake_dump = server_dir / "logs" / "calls" / "stub-subset.json"
    fake_dump.parent.mkdir(parents=True, exist_ok=True)
    fake_dump.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(server, "write_full_dump", lambda cid, dump: fake_dump)
    monkeypatch.setattr(server, "log_call", lambda **kw: None)

    asyncio.run(server._do_council_ask_async(
        question="q", context_paths=[], max_response_tokens=1024,
        models=["glm", "kimi"],
    ))
    assert [m["id"] for m in captured["members"]] == ["glm", "kimi"]
    fake_dump.unlink(missing_ok=True)


def test_council_ask_single_model_rejected():
    with pytest.raises(RuntimeError, match="at least 2"):
        asyncio.run(server._do_council_ask_async(
            question="q", context_paths=[], max_response_tokens=1024,
            models=["glm"],
        ))


def test_council_ask_unknown_model_rejected():
    with pytest.raises(RuntimeError, match="unknown model_id"):
        asyncio.run(server._do_council_ask_async(
            question="q", context_paths=[], max_response_tokens=1024,
            models=["glm", "nope"],
        ))


def test_council_ask_default_models_passes_six_members(monkeypatch):
    """models=None preserves the default 6-member council."""
    from pathlib import Path as _Path
    captured = {}

    async def fake_run_council(**kwargs):
        captured["members"] = kwargs.get("members")
        return _stub_result()

    monkeypatch.setattr(server, "run_council", fake_run_council)
    server_dir = _Path(server.__file__).parent
    fake_dump = server_dir / "logs" / "calls" / "stub-default.json"
    fake_dump.parent.mkdir(parents=True, exist_ok=True)
    fake_dump.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(server, "write_full_dump", lambda cid, dump: fake_dump)
    monkeypatch.setattr(server, "log_call", lambda **kw: None)

    asyncio.run(server._do_council_ask_async(
        question="q", context_paths=[], max_response_tokens=1024,
    ))
    assert len(captured["members"]) == 7
    fake_dump.unlink(missing_ok=True)


# -------------------------------------------------------------------------
# Task 7: model_ask
# -------------------------------------------------------------------------


def test_model_ask_happy_path(monkeypatch):
    """model_ask wires sandbox + run_single + logger."""
    monkeypatch.setenv("DEEPSEEK_KEY", "fake")

    async def fake_run_single(cfg, *, prompt, max_tokens, web_search=False):
        assert cfg["id"] == "deepseek-flash"
        assert "the question" in prompt
        return "answer text"

    monkeypatch.setattr(server, "run_single", fake_run_single)
    monkeypatch.setattr(server, "log_call", lambda **kw: None)

    result = asyncio.run(server.model_ask(
        model_id="deepseek-flash",
        prompt="the question",
    ))
    assert result == "answer text"


def test_model_ask_with_context_and_examples(monkeypatch, tmp_path):
    """context_paths and example_paths form CONTEXT FILES / STYLE EXAMPLES sections."""
    monkeypatch.setenv("DEEPSEEK_KEY", "fake")
    ctx_file = tmp_path / "ctx.txt"
    ctx_file.write_text("CTX-DATA")
    ex_file = tmp_path / "ex.txt"
    ex_file.write_text("EX-DATA")

    captured = {}

    async def fake_run_single(cfg, *, prompt, max_tokens, web_search=False):
        captured["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(server, "run_single", fake_run_single)
    monkeypatch.setattr(server, "log_call", lambda **kw: None)

    asyncio.run(server.model_ask(
        model_id="deepseek-flash",
        prompt="task",
        context_paths=[str(ctx_file)],
        example_paths=[str(ex_file)],
    ))
    p = captured["prompt"]
    assert "=== CONTEXT FILES ===" in p
    assert "CTX-DATA" in p
    assert "=== STYLE EXAMPLES ===" in p
    assert "EX-DATA" in p
    assert "=== TASK ===" in p
    assert "task" in p


def test_model_ask_unknown_id_raises(monkeypatch):
    monkeypatch.setattr(server, "log_call", lambda **kw: None)
    with pytest.raises(RuntimeError, match="unknown model_id"):
        asyncio.run(server.model_ask(model_id="nope", prompt="x"))


def test_model_ask_disabled_id_raises(monkeypatch):
    monkeypatch.setattr(server, "log_call", lambda **kw: None)
    with pytest.raises(RuntimeError, match="disabled"):
        asyncio.run(server.model_ask(model_id="minimax-direct", prompt="x"))
