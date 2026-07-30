"""Offline unit tests for codex-agent-server (no live codex, no network).

Server.py is import-safe: subprocess/`codex --version` run only under main(), so
module import just computes constants. run_codex is the single seam we mock.
Named NOT test_server.py to avoid clashing with claude's live test_server.py and
the historical codex suite renamed to integration_suite.py.
"""

import io
import json
import os
from pathlib import Path

import pytest

import server


# ── pure functions ────────────────────────────────────────────────────────

def test_resolve_model_base_and_agent_suffix():
    assert server.resolve_model("gpt-5.6-sol") == ("gpt-5.6-sol", None)
    assert server.resolve_model("gpt-5.6-sol-agent") == ("gpt-5.6-sol", "workspace-write")
    # предыдущий дефолт остаётся в whitelist — уже настроенные клиенты (CCR,
    # bench/models.json) не должны сломаться от смены дефолта.
    assert server.resolve_model("gpt-5.5") == ("gpt-5.5", None)
    assert server.resolve_model(None) == (server.DEFAULT_MODEL, None)


def test_resolve_model_unknown_raises():
    with pytest.raises(server.BadRequest):
        server.resolve_model("gpt-nonexistent")


def test_resolve_sandbox_precedence():
    # tools force read-only regardless of suffix
    assert server.resolve_sandbox([{"x": 1}], None, "workspace-write") == "read-only"
    # explicit body sandbox wins over suffix
    assert server.resolve_sandbox(None, "workspace-write", None) == "workspace-write"
    # invalid body sandbox
    with pytest.raises(server.BadRequest):
        server.resolve_sandbox(None, "bogus", None)
    # suffix mode
    assert server.resolve_sandbox(None, None, "workspace-write") == "workspace-write"
    # default
    assert server.resolve_sandbox(None, None, None) == server.DEFAULT_SANDBOX


def test_resolve_workdir_containment(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKDIR", str(tmp_path))
    monkeypatch.setattr(server, "WORKDIR_ROOT", str(tmp_path))
    inside = tmp_path / "sub"
    inside.mkdir()
    assert server.resolve_workdir(str(inside)).lower().endswith("sub")
    # outside the root
    with pytest.raises(server.BadRequest):
        server.resolve_workdir(str(tmp_path.parent))
    # non-existent dir inside root
    with pytest.raises(server.BadRequest):
        server.resolve_workdir(str(tmp_path / "missing"))
    # cmd metacharacter in path
    meta = tmp_path / "a(b)"
    meta.mkdir()
    with pytest.raises(server.BadRequest):
        server.resolve_workdir(str(meta))


def test_resolve_workdir_no_root_disabled(monkeypatch):
    monkeypatch.setattr(server, "WORKDIR", None)
    monkeypatch.setattr(server, "WORKDIR_ROOT", None)
    with pytest.raises(server.BadRequest):
        server.resolve_workdir("C:/whatever")


def test_resolve_model_non_string_raises():
    # F26: a numeric `model` must be rejected as a client error, not crash on
    # name.endswith() with an AttributeError → worker crash.
    with pytest.raises(server.BadRequest):
        server.resolve_model(123)


def test_parse_tool_calls_non_dict_json_left_as_text():
    # F26: `<tool_call>1</tool_call>` is valid JSON but not an object — must not
    # crash on data.get(...); the block yields no call and stays as text.
    calls, remaining = server.parse_tool_calls("keep <tool_call>1</tool_call> me")
    assert calls == []
    assert "keep" in remaining and "me" in remaining
    calls, _ = server.parse_tool_calls('<tool_call>[1, 2]</tool_call>')
    assert calls == []


def test_tokens_collapse_privilege(monkeypatch):
    # F30: equal read/agent tokens are detected (main() exits on it).
    monkeypatch.setattr(server, "AUTH_TOKEN", "same")
    monkeypatch.setattr(server, "AGENT_AUTH_TOKEN", "same")
    assert server._tokens_collapse_privilege() is True
    monkeypatch.setattr(server, "AGENT_AUTH_TOKEN", "different")
    assert server._tokens_collapse_privilege() is False
    monkeypatch.setattr(server, "AGENT_AUTH_TOKEN", None)
    assert server._tokens_collapse_privilege() is False


def test_run_codex_isolation_flags(monkeypatch):
    # F2: every codex call carries the host-coupling isolation flags.
    captured = {}

    class _FakeProc:
        returncode = 0

        def communicate(self, input=None, timeout=None):
            return ("", "")

        def poll(self):
            return 0

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(server, "READ_ROOT", None)  # avoid an extra -C token
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    server.run_codex("hi", model_base="gpt-5.5", sandbox="read-only")
    cmd = captured["cmd"]
    assert "--ephemeral" in cmd
    assert "--ignore-user-config" in cmd
    assert "--ignore-rules" in cmd


def test_kill_process_tree_skips_dead_proc(monkeypatch):
    # F29: a process that already exited (poll() != None) needs no taskkill.
    called = {"run": False}
    monkeypatch.setattr(server.subprocess, "run",
                        lambda *a, **k: called.__setitem__("run", True))

    class _DeadProc:
        def poll(self):
            return 0

    server._kill_process_tree(_DeadProc())
    assert called["run"] is False


def test_parse_tool_calls_single_multiple_and_bad():
    calls, remaining = server.parse_tool_calls(
        'text before <tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call> after'
    )
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "a"
    assert "before" in remaining and "after" in remaining

    calls, _ = server.parse_tool_calls(
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
    )
    assert [c["function"]["name"] for c in calls] == ["a", "b"]

    calls, _ = server.parse_tool_calls("<tool_call>not json</tool_call>")
    assert calls == []


def test_extract_content_variants():
    assert server.extract_content("hi") == "hi"
    assert server.extract_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a b"
    assert server.extract_content(None) == ""


def test_child_env_is_an_allowlist(monkeypatch):
    """Guessing secret NAMES doesn't work: DATABASE_URL / GITHUB_PAT / AWS_SESSION
    carry credentials and match no suffix-or-substring rule, so the old denylist
    handed them to a CLI that can read its own environment. Only names the CLI
    needs to run get through now."""
    monkeypatch.setattr(server.os, "environ", {
        "FOO_TOKEN": "s1", "BAR_KEY": "s2", "MY_PASSWORD": "s3",
        # These are exactly the ones a denylist misses.
        "DATABASE_URL": "postgres://u:p@h/db", "GITHUB_PAT": "ghp_x",
        "AWS_SESSION": "s", "STRIPE_SK": "sk_live_x",
        "PATH": "/usr/bin", "HOME": "/home/u", "PLAIN": "not-needed",
    })
    env = server._child_env_without_secrets(EXTRA="1")
    for leaked in ("FOO_TOKEN", "BAR_KEY", "MY_PASSWORD", "DATABASE_URL",
                   "GITHUB_PAT", "AWS_SESSION", "STRIPE_SK", "PLAIN"):
        assert leaked not in env, f"{leaked} must not reach the child CLI"
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/home/u"
    assert env["EXTRA"] == "1"


def test_child_env_passthrough_opt_in(monkeypatch):
    """A deployment that genuinely needs another variable opts in explicitly."""
    monkeypatch.setattr(server.os, "environ", {
        "HTTPS_PROXY": "http://proxy:3128", "PATH": "/usr/bin",
        "AGENT_CHILD_ENV_PASSTHROUGH": "https_proxy",
    })
    env = server._child_env_without_secrets()
    assert env["HTTPS_PROXY"] == "http://proxy:3128"


def test_build_tools_system_prompt_lists_functions():
    prompt = server.build_tools_system_prompt([
        {"function": {"name": "weather", "description": "d",
                      "parameters": {"properties": {"loc": {"type": "string"}}, "required": ["loc"]}}}
    ])
    assert "weather" in prompt and "loc" in prompt and "[required]" in prompt


# ── handler (_handle_chat / do_GET), run_codex mocked ──────────────────────

def _handler(monkeypatch, headers=None):
    h = server.Handler.__new__(server.Handler)
    h.headers = headers or {}
    h.sent = []
    h.stream_calls = []
    h.stream_live_calls = []
    h._send = lambda code, data, headers=None: h.sent.append((code, data))
    h._send_stream = lambda *a: h.stream_calls.append(a)
    h._send_stream_live = lambda *a: h.stream_live_calls.append(a)
    return h


def test_happy_path_shape(monkeypatch):
    monkeypatch.setattr(server, "run_codex", lambda *a, **k: "hello there")
    h = _handler(monkeypatch)
    h._handle_chat({"messages": [{"role": "user", "content": "hi"}]})
    assert len(h.sent) == 1
    code, data = h.sent[0]
    assert code == 200
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "hello there"
    assert data["usage"]["estimate"] is True
    assert data["usage"]["sandbox"] == "read-only"


def test_tools_return_tool_calls(monkeypatch):
    canned = '<tool_call>{"name": "weather", "arguments": {"location": "Moscow"}}</tool_call>'
    monkeypatch.setattr(server, "run_codex", lambda *a, **k: canned)
    h = _handler(monkeypatch)
    h._handle_chat({
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"type": "function", "function": {"name": "weather", "parameters": {}}}],
    })
    code, data = h.sent[0]
    assert code == 200
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "weather"
    assert data["choices"][0]["finish_reason"] == "tool_calls"


def test_tools_force_readonly_reported_in_usage(monkeypatch):
    # F#16: gpt-5.5-agent + tools runs read-only; usage.sandbox reflects it.
    monkeypatch.setattr(server, "run_codex", lambda *a, **k: "ok")
    h = _handler(monkeypatch)
    h._handle_chat({
        "model": "gpt-5.5-agent",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
    })
    code, data = h.sent[0]
    assert code == 200
    assert data["model"] == "gpt-5.5-agent"       # echoes request (OpenAI convention)
    assert data["usage"]["sandbox"] == "read-only"  # but effective mode surfaced


def test_bad_body_returns_400(monkeypatch):
    h = _handler(monkeypatch)
    h._handle_chat({"messages": "not a list"})
    code, data = h.sent[0]
    assert code == 400
    assert data["error"]["type"] == "invalid_request_error"


def test_tools_null_entry_returns_400(monkeypatch):
    # F26: `tools:[null]` must be a clean 400, not an AttributeError in
    # build_tools_system_prompt → worker crash.
    monkeypatch.setattr(server, "run_codex", lambda *a, **k: "x")
    h = _handler(monkeypatch)
    h._handle_chat({"messages": [{"role": "user", "content": "hi"}], "tools": [None]})
    code, data = h.sent[0]
    assert code == 400
    assert data["error"]["type"] == "invalid_request_error"


def test_numeric_model_returns_400(monkeypatch):
    # F26: a numeric `model` is a client error → 400, not a 500/crash.
    monkeypatch.setattr(server, "run_codex", lambda *a, **k: "x")
    h = _handler(monkeypatch)
    h._handle_chat({"model": 123, "messages": [{"role": "user", "content": "hi"}]})
    code, data = h.sent[0]
    assert code == 400
    assert data["error"]["type"] == "invalid_request_error"


def test_404_is_openai_shaped(monkeypatch):
    h = _handler(monkeypatch)
    h.path = "/nope"
    h.do_GET()
    code, data = h.sent[0]
    assert code == 404
    assert isinstance(data["error"], dict)
    assert "message" in data["error"] and "type" in data["error"]


def test_stream_tool_calls_indexed():
    # F#11 golden: streaming tool_calls are OpenAI-shaped indexed deltas.
    h = server.Handler.__new__(server.Handler)
    h.send_response = lambda *a, **k: None
    h.send_header = lambda *a, **k: None
    h.end_headers = lambda *a, **k: None
    h.wfile = io.BytesIO()
    resp_message = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
        {"id": "call_2", "type": "function", "function": {"name": "b", "arguments": '{"x":1}'}},
    ]}
    h._send_stream("id1", 123, "gpt-5.5", resp_message, "tool_calls", {"estimate": True, "sandbox": "read-only"})
    blocks = [b for b in h.wfile.getvalue().decode("utf-8").split("\n\n") if b.strip()]
    assert blocks[-1] == "data: [DONE]"
    payloads = [json.loads(b[len("data: "):]) for b in blocks if not b.endswith("[DONE]")]
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    # content is None → no content delta emitted
    assert all("content" not in p["choices"][0]["delta"] for p in payloads)
    tc = [p["choices"][0]["delta"]["tool_calls"][0] for p in payloads
          if "tool_calls" in p["choices"][0]["delta"]]
    assert [t["index"] for t in tc] == [0, 1]
    assert [t["function"]["name"] for t in tc] == ["a", "b"]
    finish = [p for p in payloads if p["choices"][0]["finish_reason"] == "tool_calls"]
    assert finish and finish[0]["usage"]["estimate"] is True


# ── Idea 1: capability profiles ────────────────────────────────────────────

def test_resolve_profile_and_sandbox_explicit():
    assert server.resolve_profile_and_sandbox(None, None, "chat", None) == ("chat", "read-only")
    assert server.resolve_profile_and_sandbox(None, None, "research", None) == ("research", "read-only")
    assert server.resolve_profile_and_sandbox(None, None, "agent", None) == ("agent", "workspace-write")


def test_resolve_profile_agent_with_tools_conflicts():
    with pytest.raises(server.BadRequest):
        server.resolve_profile_and_sandbox([{"x": 1}], None, "agent", None)


def test_resolve_profile_invalid():
    with pytest.raises(server.BadRequest):
        server.resolve_profile_and_sandbox(None, None, "bogus", None)


def test_resolve_profile_legacy_fallback():
    # `-agent` suffix (workspace-write) → reported as agent profile
    assert server.resolve_profile_and_sandbox(None, None, None, "workspace-write") == ("agent", "workspace-write")
    # nothing → chat + default sandbox (read-only)
    assert server.resolve_profile_and_sandbox(None, None, None, None) == ("chat", server.DEFAULT_SANDBOX)


def test_handle_chat_reports_profile(monkeypatch):
    monkeypatch.setattr(server, "run_codex", lambda *a, **k: "hi")
    h = _handler(monkeypatch)
    h._handle_chat({"messages": [{"role": "user", "content": "hi"}], "profile": "research"})
    code, data = h.sent[0]
    assert code == 200
    assert data["usage"]["profile"] == "research"
    assert data["usage"]["sandbox"] == "read-only"


# ── Idea 12: structured output helpers + validation/repair ─────────────────

def test_json_schema_errors_required_and_nested():
    schema = {"type": "object", "required": ["a", "b"],
              "properties": {"b": {"type": "object", "required": ["c"]}}}
    assert server.json_schema_errors({"a": 1, "b": {"c": 2}}, schema) == []
    errs = server.json_schema_errors({"a": 1}, schema)
    assert any("b" in e for e in errs)
    errs = server.json_schema_errors({"a": 1, "b": {}}, schema)
    assert any("c" in e for e in errs)  # nested required missing
    assert server.json_schema_errors([], schema)  # not an object


def test_validate_structured_output_fence_and_schema():
    ok, _ = server.validate_structured_output('```json\n{"x": 1}\n```', None)
    assert ok
    ok, err = server.validate_structured_output("not json", None)
    assert not ok and "not valid JSON" in err
    ok, err = server.validate_structured_output('{"x": 1}', {"type": "object", "required": ["y"]})
    assert not ok and "y" in err


def test_build_response_format_prompt_types():
    assert server.build_response_format_prompt({"type": "json_object"}).startswith("# OUTPUT FORMAT")
    p = server.build_response_format_prompt(
        {"type": "json_schema", "json_schema": {"schema": {"type": "object", "required": ["a"]}}})
    assert "JSON Schema" in p and '"required"' in p
    assert server.build_response_format_prompt({"type": "text"}) is None


def test_build_tools_system_prompt_includes_full_schema():
    prompt = server.build_tools_system_prompt([
        {"function": {"name": "f", "description": "d", "parameters": {
            "type": "object", "properties": {"x": {"type": "string", "enum": ["a", "b"]}},
            "required": ["x"]}}}
    ])
    assert "Full JSON Schema" in prompt and '"enum"' in prompt


def test_tool_calls_schema_errors_detects_missing():
    tools = [{"function": {"name": "f", "parameters": {"type": "object", "required": ["x"]}}}]
    good = [{"function": {"name": "f", "arguments": '{"x": 1}'}}]
    bad = [{"function": {"name": "f", "arguments": '{}'}}]
    assert server.tool_calls_schema_errors(good, tools) == []
    assert server.tool_calls_schema_errors(bad, tools)


def test_handle_chat_structured_output_repair(monkeypatch):
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        return "not json" if calls["n"] == 1 else '{"a": 1}'

    monkeypatch.setattr(server, "run_codex", fake)
    h = _handler(monkeypatch)
    h._handle_chat({"messages": [{"role": "user", "content": "hi"}],
                    "response_format": {"type": "json_object"}})
    code, data = h.sent[0]
    assert code == 200
    assert calls["n"] == 2  # one repair-retry
    assert data["usage"]["structured_output"] is True
    assert data["choices"][0]["message"]["content"] == '{"a": 1}'


# ── Idea 11: streaming runner + live SSE + cancellation fallback ────────────

class _FakeStdin:
    def write(self, s):
        pass

    def close(self):
        pass


class _FakeStream:
    def __init__(self, lines):
        self._it = iter(lines)

    def __iter__(self):
        return self._it

    def close(self):
        pass


class _FakeStreamProc:
    def __init__(self, lines):
        self.stdout = _FakeStream(lines)
        self.stderr = _FakeStream([])
        self.stdin = _FakeStdin()
        self.returncode = 0
        self.pid = 4242

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0


def test_run_codex_stream_parses_events(monkeypatch):
    lines = [
        json.dumps({"type": "item.completed",
                    "item": {"type": "agent_message", "text": "Hello world"}}) + "\n",
        json.dumps({"type": "turn.completed",
                    "usage": {"input_tokens": 10, "output_tokens": 5}}) + "\n",
    ]
    monkeypatch.setattr(server, "READ_ROOT", None)
    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **k: _FakeStreamProc(lines))
    items = list(server.run_codex_stream("hi", model_base="gpt-5.5", sandbox="read-only"))
    kinds = [i[0] for i in items]
    assert ("text", "Hello world") in items
    assert kinds[-1] == "meta"
    meta = items[-1][1]
    assert meta["usage"]["estimate"] is False
    assert meta["usage"]["prompt_tokens"] == 10 and meta["usage"]["completion_tokens"] == 5


def test_run_codex_stream_unsupported_raises(monkeypatch):
    monkeypatch.setattr(server, "READ_ROOT", None)
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda *a, **k: _FakeStreamProc(["this is not json\n"]))
    gen = server.run_codex_stream("hi", model_base="gpt-5.5", sandbox="read-only")
    with pytest.raises(server.StreamUnsupported):
        next(gen)


def test_send_stream_live_emits_real_usage():
    def gen_items():
        yield ("text", "Hel")
        yield ("text", "lo")
        yield ("meta", {"usage": {"prompt_tokens": 1, "completion_tokens": 2,
                                  "total_tokens": 3, "estimate": False}, "stop_reason": "stop"})

    g = gen_items()
    first = next(g)
    h = server.Handler.__new__(server.Handler)
    h.send_response = lambda *a, **k: None
    h.send_header = lambda *a, **k: None
    h.end_headers = lambda *a, **k: None
    h.wfile = io.BytesIO()
    h._send_stream_live(g, first, "id1", 123, "gpt-5.5", {"profile": "chat", "estimate": True, "sandbox": "read-only"})
    blocks = [b for b in h.wfile.getvalue().decode("utf-8").split("\n\n") if b.strip()]
    assert blocks[-1] == "data: [DONE]"
    payloads = [json.loads(b[len("data: "):]) for b in blocks if not b.endswith("[DONE]")]
    contents = [p["choices"][0]["delta"].get("content") for p in payloads
                if "content" in p["choices"][0]["delta"]]
    assert contents == ["Hel", "lo"]
    finish = [p for p in payloads if p["choices"][0]["finish_reason"] == "stop"]
    assert finish and finish[0]["usage"]["estimate"] is False
    assert finish[0]["usage"]["profile"] == "chat"  # base_usage preserved


def test_handle_chat_stream_routes_to_live(monkeypatch):
    def fake_stream(*a, **k):
        yield ("text", "hi")
        yield ("meta", {"usage": None, "stop_reason": "stop"})

    monkeypatch.setattr(server, "run_codex_stream", fake_stream)
    h = _handler(monkeypatch)
    h._handle_chat({"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert h.stream_live_calls and not h.sent


def test_handle_chat_stream_fallback_on_unsupported(monkeypatch):
    def fake_stream(*a, **k):
        raise server.StreamUnsupported("nope")
        yield  # pragma: no cover

    monkeypatch.setattr(server, "run_codex_stream", fake_stream)
    monkeypatch.setattr(server, "run_codex", lambda *a, **k: "buffered answer")
    h = _handler(monkeypatch)
    h._handle_chat({"messages": [{"role": "user", "content": "hi"}], "stream": True})
    # fell back to buffered → pseudo-stream
    assert h.stream_calls and not h.stream_live_calls


# ── Idea 13: readiness / metrics / bounded queue ───────────────────────────

def test_acquire_slot_and_release():
    assert server._acquire_slot() is True
    server._CODEX_SEM.release()


def test_metrics_snapshot_shape():
    snap = server.METRICS.snapshot(42)
    for k in ("total_requests", "active", "rejected_overload", "timeouts",
              "killed_processes", "latency_median_s", "latency_p90_s", "max_queue"):
        assert k in snap


def test_do_get_metrics_and_ready(monkeypatch):
    # /metrics requires auth; present a valid bearer if the server has a token.
    hdrs = {"Authorization": f"Bearer {server.AUTH_TOKEN}"} if server.AUTH_TOKEN else {}
    h = _handler(monkeypatch, headers=hdrs)
    h.path = "/metrics"
    h.do_GET()
    code, data = h.sent[-1]
    assert code == 200 and "total_requests" in data
    h.sent.clear()
    h.path = "/ready"
    h.do_GET()
    code, data = h.sent[-1]
    assert code in (200, 503)
    assert "ready" in data and "checks" in data
    assert set(data["checks"]) >= {"auth_token_configured", "cli_found", "not_overloaded"}


# ── FINDINGS 2026-07-21: streaming honesty + no double workspace-write ──────


@pytest.fixture(autouse=True)
def _reset_stream_support():
    """The support memo is process-wide; keep tests independent."""
    server._STREAM_JSON_SUPPORTED[0] = None
    yield
    server._STREAM_JSON_SUPPORTED[0] = None


class _FailingStreamProc(_FakeStreamProc):
    """A CLI that emits one recognized event, then dies with a non-zero code."""

    def __init__(self, lines, returncode=1):
        super().__init__(lines)
        self._rc = returncode
        self.stderr = _FakeStream(["codex: boom\n"])

    def poll(self):
        return self._rc

    def wait(self, timeout=None):
        return self._rc


def test_stream_nonzero_exit_raises_instead_of_clean_finish(monkeypatch):
    """EOF on stdout is not success: a CLI that emitted one delta and then failed
    used to be reported to the client as a completed answer."""
    lines = [json.dumps({"type": "agent_message_delta", "delta": "partial"}) + "\n"]
    monkeypatch.setattr(server, "READ_ROOT", None)
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda *a, **k: _FailingStreamProc(lines))
    gen = server.run_codex_stream("hi", model_base="gpt-5.5", sandbox="read-only")
    assert next(gen) == ("text", "partial")
    with pytest.raises(server.StreamFailed):
        next(gen)


def test_stream_timeout_is_reported_not_silently_completed(monkeypatch):
    lines = [json.dumps({"type": "agent_message_delta", "delta": "partial"}) + "\n"]
    monkeypatch.setattr(server, "READ_ROOT", None)
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda *a, **k: _FakeStreamProc(lines))

    class _ImmediateTimer:
        """Fire the watchdog callback as soon as it is started."""

        def __init__(self, interval, fn, args=()):
            self._fn, self._args = fn, args
            self.daemon = True

        def start(self):
            self._fn(*self._args)

        def cancel(self):
            pass

    monkeypatch.setattr(server.threading, "Timer", _ImmediateTimer)
    monkeypatch.setattr(server, "_kill_process_tree", lambda proc: None)
    gen = server.run_codex_stream("hi", model_base="gpt-5.5", sandbox="read-only")
    items = []
    with pytest.raises(server.StreamFailed, match="timed out"):
        for item in gen:
            items.append(item)
    assert ("text", "partial") in items


def test_stream_reconciles_outfile_after_partial_deltas(monkeypatch, tmp_path):
    """One early delta used to suppress the authoritative `-o` outfile entirely,
    so a truncated event stream silently became the whole answer."""
    outfile = tmp_path / "codex-out.txt"
    outfile.write_text("partial and the rest", encoding="utf-8")
    monkeypatch.setattr(server, "READ_ROOT", None)
    monkeypatch.setattr(server.tempfile, "mkstemp",
                        lambda **k: (os.open(outfile, os.O_RDONLY), str(outfile)))
    lines = [json.dumps({"type": "agent_message_delta", "delta": "partial"}) + "\n"]
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda *a, **k: _FakeStreamProc(lines))
    items = list(server.run_codex_stream("hi", model_base="gpt-5.5", sandbox="read-only"))
    texts = [p for kind, p in items if kind == "text"]
    assert texts == ["partial", " and the rest"]
    assert items[-1][1]["text"] == "partial and the rest"


def test_stream_unsupported_never_reruns_a_workspace_write(monkeypatch):
    """The buffered fallback re-executes the SAME prompt. For workspace-write the
    first process may already have edited files, so the agent action would be
    applied twice — report the failure instead of retrying."""
    def fake_stream(*a, **k):
        raise server.StreamUnsupported("nope")
        yield  # pragma: no cover

    ran = []
    monkeypatch.setattr(server, "run_codex_stream", fake_stream)
    monkeypatch.setattr(server, "run_codex",
                        lambda *a, **k: ran.append(1) or "buffered answer")
    monkeypatch.setattr(server, "resolve_workdir", lambda *a, **k: str(Path.cwd()))
    monkeypatch.setattr(server, "AGENT_AUTH_TOKEN", "agent-tok")
    h = _handler(monkeypatch)
    h._check_agent_auth = lambda: True
    h._handle_chat({"messages": [{"role": "user", "content": "hi"}],
                    "model": "gpt-5.5-agent", "stream": True})
    assert ran == [], "workspace-write must not be re-executed"
    assert h.sent and h.sent[0][0] == 502
    assert "will NOT be retried" in h.sent[0][1]["error"]["message"]


def test_stream_unsupported_is_remembered(monkeypatch):
    """After learning the CLI has no --json, later requests skip the doomed pass."""
    calls = []

    def fake_stream(*a, **k):
        calls.append(1)
        raise server.StreamUnsupported("nope")
        yield  # pragma: no cover

    monkeypatch.setattr(server, "run_codex_stream", fake_stream)
    monkeypatch.setattr(server, "run_codex", lambda *a, **k: "buffered answer")
    for _ in range(3):
        h = _handler(monkeypatch)
        h._handle_chat({"messages": [{"role": "user", "content": "hi"}], "stream": True})
        assert h.stream_calls  # buffered pseudo-stream each time
    assert len(calls) == 1


def test_send_stream_live_reports_failure_instead_of_done():
    """A failed run must not sign off with a successful finish chunk + [DONE]."""
    def gen_items():
        yield ("text", "partial")
        raise server.StreamFailed("codex timed out after 300s")

    g = gen_items()
    first = next(g)
    h = server.Handler.__new__(server.Handler)
    h.send_response = lambda *a, **k: None
    h.send_header = lambda *a, **k: None
    h.end_headers = lambda *a, **k: None
    h.wfile = io.BytesIO()
    before = server.METRICS.timeouts
    h._send_stream_live(g, first, "id1", 123, "gpt-5.5", {"estimate": True})
    body = h.wfile.getvalue().decode("utf-8")
    assert "[DONE]" not in body
    assert '"finish_reason": "error"' in body
    assert "timed out" in body
    assert server.METRICS.timeouts == before + 1


def test_unknown_tool_name_is_an_error():
    """A hallucinated tool used to be skipped by the validator and reach the
    client as a legitimate finish_reason=tool_calls."""
    tools = [{"function": {"name": "weather", "parameters": {
        "type": "object", "properties": {"loc": {"type": "string"}}}}}]
    errs = server.tool_calls_schema_errors(
        [{"function": {"name": "rm_rf", "arguments": "{}"}}], tools)
    assert errs and "not one of the offered tools" in errs[0]
    assert server.tool_calls_schema_errors(
        [{"function": {"name": "weather", "arguments": '{"loc": "Moscow"}'}}], tools) == []


def test_structured_repair_is_revalidated(monkeypatch):
    """A second invalid response used to return 200 with structured_output=true."""
    monkeypatch.setattr(server, "run_codex", lambda *a, **k: "still not json")
    h = _handler(monkeypatch)
    h._handle_chat({
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_object"},
    })
    code, data = h.sent[0]
    assert code == 502 and "after one repair" in data["error"]["message"]


def test_malformed_message_tool_calls_is_400(monkeypatch):
    monkeypatch.setattr(server, "run_codex", lambda *a, **k: "x")
    h = _handler(monkeypatch)
    h._handle_chat({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": {"id": "1"}},
    ]})
    assert h.sent[0][0] == 400


def test_ready_checks_read_root_and_reasoning(monkeypatch, tmp_path):
    """/ready stayed green while every read-only call failed on a bad `-C`, and
    an invalid CODEX_AGENT_REASONING was never validated at all."""
    sent = []
    h = server.Handler.__new__(server.Handler)
    h._send = lambda code, data, headers=None: sent.append((code, data))

    monkeypatch.setattr(server, "AUTH_TOKEN", "tok")
    monkeypatch.setattr(server.shutil, "which", lambda name: "codex")
    monkeypatch.setattr(server, "WORKDIR_ROOT", None)
    monkeypatch.setattr(server, "DEFAULT_SANDBOX", "read-only")
    monkeypatch.setattr(server, "REASONING_ENV_VALID", True)

    monkeypatch.setattr(server, "READ_ROOT", str(tmp_path / "missing"))
    h._send_ready()
    assert sent[-1][0] == 503
    assert sent[-1][1]["checks"]["read_root_exists"] is False

    monkeypatch.setattr(server, "READ_ROOT", str(tmp_path))
    h._send_ready()
    assert sent[-1][0] == 200

    monkeypatch.setattr(server, "REASONING_ENV_VALID", False)
    h._send_ready()
    assert sent[-1][0] == 503
    assert sent[-1][1]["checks"]["reasoning_valid"] is False


def test_ready_requires_agent_config_when_default_is_workspace_write(monkeypatch, tmp_path):
    sent = []
    h = server.Handler.__new__(server.Handler)
    h._send = lambda code, data, headers=None: sent.append((code, data))
    monkeypatch.setattr(server, "AUTH_TOKEN", "tok")
    monkeypatch.setattr(server.shutil, "which", lambda name: "codex")
    monkeypatch.setattr(server, "READ_ROOT", str(tmp_path))
    monkeypatch.setattr(server, "WORKDIR_ROOT", None)
    monkeypatch.setattr(server, "WORKDIR", None)
    monkeypatch.setattr(server, "AGENT_AUTH_TOKEN", None)
    monkeypatch.setattr(server, "REASONING_ENV_VALID", True)
    monkeypatch.setattr(server, "DEFAULT_SANDBOX", "workspace-write")
    h._send_ready()
    code, data = sent[-1]
    assert code == 503
    assert data["checks"]["default_route_workdir_configured"] is False
    assert data["checks"]["default_route_agent_token_configured"] is False
