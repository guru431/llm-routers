"""Offline handler/runner tests for claude-agent-server (no live claude).

server.py is import-safe: `claude --version` runs only under main(), so importing
the module just computes constants. run_claude's subprocess.Popen is the single
seam we mock. Covers the F1 (argv injection), F2 (isolation flags) and F26 (body
validation / malformed tool blocks) regressions.
"""

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

def _handler():
    h = server.Handler.__new__(server.Handler)
    h.headers = {}
    h.sent = []
    h._send = lambda code, data: h.sent.append((code, data))
    h._send_stream = lambda *a: h.sent.append(("stream", a))
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
