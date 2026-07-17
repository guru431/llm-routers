"""Offline handler/runner tests for claude-agent-server (no live claude).

server.py is import-safe: `claude --version` runs only under main(), so importing
the module just computes constants. run_claude's subprocess.Popen is the single
seam we mock. Covers the F1 (argv injection), F2 (isolation flags) and F26 (body
validation / malformed tool blocks) regressions.
"""

import io
import json

import pytest

import server


# ── run_claude command construction (Popen mocked) ─────────────────────────

class _FakeProc:
    def __init__(self, stdout='{"result": "ok"}'):
        self._stdout = stdout
        self.returncode = 0

    def communicate(self, input=None, timeout=None):
        return (self._stdout, "")

    def poll(self):
        return 0


def _capture_popen(monkeypatch, stdout='{"result": "ok"}'):
    captured = {}

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        captured["kwargs"] = kw
        return _FakeProc(stdout)

    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    return captured


def test_system_prompt_not_in_argv_uses_file(monkeypatch):
    # F1 (BatBadBut): CLAUDE_BIN is a claude.CMD shim on Windows, so subprocess
    # routes argv through cmd.exe. A client-controlled system prompt with cmd
    # metacharacters must NOT reach argv — it goes via --system-prompt-file.
    captured = _capture_popen(monkeypatch)
    evil = 'hi" & calc.exe ^ | echo (%PATH%)'
    out = server.run_claude("prompt", system_prompt=evil, model="claude-opus-4-8")
    assert out == "ok"
    cmd = captured["cmd"]
    assert "--system-prompt-file" in cmd
    # No argv token carries the raw system prompt, and the old injectable
    # `--system-prompt=<value>` form is gone.
    assert all(evil not in str(tok) for tok in cmd)
    assert not any(str(tok).startswith("--system-prompt=") for tok in cmd)


def test_isolation_flags_present(monkeypatch):
    # F2: the chat profile disables built-in tools, user MCP and session
    # persistence on every call.
    captured = _capture_popen(monkeypatch)
    server.run_claude("prompt", system_prompt=None, model="claude-opus-4-8")
    cmd = captured["cmd"]
    assert "--tools" in cmd
    assert cmd[cmd.index("--tools") + 1] == ""  # `--tools ""` disables all tools
    assert "--strict-mcp-config" in cmd
    assert "--no-session-persistence" in cmd


def test_parse_tool_calls_non_dict_left_as_text():
    # F26: `<tool_call>1</tool_call>` is valid JSON but not an object — no call,
    # left as text (byte-identical guard with codex-agent-server).
    calls, remaining = server.parse_tool_calls("a <tool_call>1</tool_call> b")
    assert calls == []
    assert "a" in remaining and "b" in remaining


# ── handler body validation (run_claude not reached) ───────────────────────

def _handler(headers=None):
    h = server.Handler.__new__(server.Handler)
    h.headers = headers or {}
    h.sent = []
    h.stream_calls = []
    h.stream_live_calls = []
    h._send = lambda code, data, headers=None: h.sent.append((code, data))
    h._send_stream = lambda *a: h.stream_calls.append(a)
    h._send_stream_live = lambda *a: h.stream_live_calls.append(a)
    return h


def test_bad_body_returns_400():
    h = _handler()
    h._handle_chat({"messages": "not a list"})
    code, _ = h.sent[0]
    assert code == 400


def test_tools_null_entry_returns_400(monkeypatch):
    # F26: `tools:[null]` must be a clean 400, not an AttributeError in
    # build_tools_system_prompt → worker crash → RemoteDisconnected.
    monkeypatch.setattr(server, "run_claude", lambda *a, **k: "x")
    h = _handler()
    h._handle_chat({"messages": [{"role": "user", "content": "hi"}], "tools": [None]})
    code, data = h.sent[0]
    assert code == 400
    assert data["error"]["type"] == "invalid_request_error"


# ── Idea 1: capability profiles (chat/research; agent rejected) ─────────────

def test_profile_agent_rejected(monkeypatch):
    monkeypatch.setattr(server, "run_claude", lambda *a, **k: "x")
    h = _handler()
    h._handle_chat({"messages": [{"role": "user", "content": "hi"}], "profile": "agent"})
    code, data = h.sent[0]
    assert code == 400
    assert "agent" in data["error"]["message"]


def test_profile_research_reported(monkeypatch):
    monkeypatch.setattr(server, "CACHE", None)
    monkeypatch.setattr(server, "run_claude", lambda *a, **k: "ok")
    h = _handler()
    h._handle_chat({"messages": [{"role": "user", "content": "hi"}], "profile": "research"})
    code, data = h.sent[0]
    assert code == 200
    assert data["usage"]["profile"] == "research"


def test_profile_invalid_rejected(monkeypatch):
    h = _handler()
    h._handle_chat({"messages": [{"role": "user", "content": "hi"}], "profile": "bogus"})
    code, data = h.sent[0]
    assert code == 400


# ── Idea 12: structured output (byte-identical helpers, handler wiring) ─────

def test_structured_helpers_present():
    assert server.json_schema_errors({"a": 1}, {"type": "object", "required": ["b"]})
    ok, _ = server.validate_structured_output('{"a": 1}', None)
    assert ok
    assert server.build_response_format_prompt({"type": "json_object"}) is not None
    prompt = server.build_tools_system_prompt([
        {"function": {"name": "f", "parameters": {"type": "object",
         "properties": {"x": {"enum": [1, 2]}}, "required": ["x"]}}}])
    assert "Full JSON Schema" in prompt


def test_handle_chat_structured_repair(monkeypatch):
    monkeypatch.setattr(server, "CACHE", None)
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        return "nope" if calls["n"] == 1 else '{"ok": true}'

    monkeypatch.setattr(server, "run_claude", fake)
    h = _handler()
    h._handle_chat({"messages": [{"role": "user", "content": "hi"}],
                    "response_format": {"type": "json_object"}})
    code, data = h.sent[0]
    assert code == 200
    assert calls["n"] == 2
    assert data["usage"]["structured_output"] is True


# ── Idea 11: streaming runner + live SSE + fallback ─────────────────────────

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


def test_run_claude_stream_parses_events(monkeypatch):
    lines = [
        json.dumps({"type": "system", "subtype": "init"}) + "\n",
        json.dumps({"type": "assistant", "message": {
            "content": [{"type": "text", "text": "Hi there"}], "stop_reason": "end_turn"}}) + "\n",
        json.dumps({"type": "result", "subtype": "success",
                    "usage": {"input_tokens": 7, "output_tokens": 3}}) + "\n",
    ]
    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **k: _FakeStreamProc(lines))
    items = list(server.run_claude_stream("hi", model="claude-opus-4-8"))
    assert ("text", "Hi there") in items
    meta = items[-1]
    assert meta[0] == "meta"
    assert meta[1]["usage"]["estimate"] is False
    assert meta[1]["usage"]["prompt_tokens"] == 7 and meta[1]["usage"]["completion_tokens"] == 3
    assert meta[1]["stop_reason"] == "stop"


def test_run_claude_stream_unsupported_raises(monkeypatch):
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda *a, **k: _FakeStreamProc(["plain error, not json\n"]))
    gen = server.run_claude_stream("hi", model="claude-opus-4-8")
    with pytest.raises(server.StreamUnsupported):
        next(gen)


def test_send_stream_live_real_usage_overrides_estimate():
    def gen_items():
        yield ("text", "ab")
        yield ("meta", {"usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                  "total_tokens": 2, "estimate": False}, "stop_reason": "stop"})

    g = gen_items()
    first = next(g)
    h = server.Handler.__new__(server.Handler)
    h.send_response = lambda *a, **k: None
    h.send_header = lambda *a, **k: None
    h.end_headers = lambda *a, **k: None
    h.wfile = io.BytesIO()
    h._send_stream_live(g, first, "id1", 1, "claude-opus-4-8", {"profile": "chat", "estimate": True, "cached": False})
    out = h.wfile.getvalue().decode("utf-8")
    blocks = [b for b in out.split("\n\n") if b.strip()]
    assert blocks[-1] == "data: [DONE]"
    payloads = [json.loads(b[len("data: "):]) for b in blocks if not b.endswith("[DONE]")]
    finish = [p for p in payloads if p["choices"][0]["finish_reason"] == "stop"]
    assert finish and finish[0]["usage"]["estimate"] is False
    assert finish[0]["usage"]["profile"] == "chat"


def test_handle_chat_stream_routes_to_live(monkeypatch):
    monkeypatch.setattr(server, "CACHE", None)

    def fake_stream(*a, **k):
        yield ("text", "hi")
        yield ("meta", {"usage": None, "stop_reason": "stop"})

    monkeypatch.setattr(server, "run_claude_stream", fake_stream)
    h = _handler()
    h._handle_chat({"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert h.stream_live_calls and not h.sent


def test_handle_chat_stream_fallback_on_unsupported(monkeypatch):
    monkeypatch.setattr(server, "CACHE", None)

    def fake_stream(*a, **k):
        raise server.StreamUnsupported("nope")
        yield  # pragma: no cover

    monkeypatch.setattr(server, "run_claude_stream", fake_stream)
    monkeypatch.setattr(server, "run_claude", lambda *a, **k: "buffered")
    h = _handler()
    h._handle_chat({"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert h.stream_calls and not h.stream_live_calls


# ── Idea 13: readiness / metrics / bounded queue ───────────────────────────

def test_acquire_slot_and_release():
    assert server._acquire_slot() is True
    server._CLAUDE_SEM.release()


def test_metrics_snapshot_shape():
    snap = server.METRICS.snapshot(10)
    for k in ("total_requests", "active", "cache_hits", "cache_misses",
              "latency_median_s", "latency_p90_s", "max_queue"):
        assert k in snap


def test_do_get_ready_and_metrics():
    hdrs = {"Authorization": f"Bearer {server.AUTH_TOKEN}"} if server.AUTH_TOKEN else {}
    h = _handler(headers=hdrs)
    h.path = "/ready"
    h.do_GET()
    code, data = h.sent[-1]
    assert code in (200, 503)
    assert set(data["checks"]) >= {"auth_token_configured", "cli_found", "not_overloaded"}
    h.sent.clear()
    h.path = "/metrics"
    h.do_GET()
    code, data = h.sent[-1]
    assert code == 200 and "total_requests" in data
