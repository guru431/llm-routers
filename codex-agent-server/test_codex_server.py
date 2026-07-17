"""Offline unit tests for codex-agent-server (no live codex, no network).

Server.py is import-safe: subprocess/`codex --version` run only under main(), so
module import just computes constants. run_codex is the single seam we mock.
Named NOT test_server.py to avoid clashing with claude's live test_server.py and
the historical codex suite renamed to integration_suite.py.
"""

import io
import json

import pytest

import server


# ── pure functions ────────────────────────────────────────────────────────

def test_resolve_model_base_and_agent_suffix():
    assert server.resolve_model("gpt-5.5") == ("gpt-5.5", None)
    assert server.resolve_model("gpt-5.5-agent") == ("gpt-5.5", "workspace-write")
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


def test_child_env_strips_secrets(monkeypatch):
    monkeypatch.setattr(server.os, "environ", {
        "FOO_TOKEN": "s1", "BAR_KEY": "s2", "MY_PASSWORD": "s3", "PLAIN": "keep",
    })
    env = server._child_env_without_secrets(EXTRA="1")
    assert "FOO_TOKEN" not in env and "BAR_KEY" not in env and "MY_PASSWORD" not in env
    assert env["PLAIN"] == "keep"
    assert env["EXTRA"] == "1"


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
    h._send = lambda code, data: h.sent.append((code, data))
    h._send_stream = lambda *a: h.stream_calls.append(a)
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
