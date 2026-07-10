"""Golden SSE tests for claude-agent-server _send_stream (F#11 / F#14).

Drives the real _send_stream with a stubbed wfile and asserts the OpenAI stream
shape: role delta, indexed tool_calls deltas, finish chunk with usage, [DONE].
Mirrors codex-agent-server/test_codex_server.py::test_stream_tool_calls_indexed
(the method is byte-identical — enforced by test_byte_identical.py).
"""

import io
import json

import server


def _stream(resp_message, usage):
    h = server.Handler.__new__(server.Handler)
    h.send_response = lambda *a, **k: None
    h.send_header = lambda *a, **k: None
    h.end_headers = lambda *a, **k: None
    h.wfile = io.BytesIO()
    h._send_stream("id1", 123, "claude-opus-4-8", resp_message, "tool_calls", usage)
    blocks = [b for b in h.wfile.getvalue().decode("utf-8").split("\n\n") if b.strip()]
    return blocks


def test_stream_tool_calls_indexed():
    resp_message = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
        {"id": "call_2", "type": "function", "function": {"name": "b", "arguments": '{"x":1}'}},
    ]}
    blocks = _stream(resp_message, {"estimate": True})
    assert blocks[-1] == "data: [DONE]"
    payloads = [json.loads(b[len("data: "):]) for b in blocks if not b.endswith("[DONE]")]
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    tc = [p["choices"][0]["delta"]["tool_calls"][0] for p in payloads
          if "tool_calls" in p["choices"][0]["delta"]]
    assert [t["index"] for t in tc] == [0, 1]
    assert [t["function"]["name"] for t in tc] == ["a", "b"]
    finish = [p for p in payloads if p["choices"][0]["finish_reason"] == "tool_calls"]
    assert finish and finish[0]["usage"]["estimate"] is True


def test_stream_no_empty_content_delta():
    # F#14: falsy content must not emit an empty content delta.
    blocks = _stream({"role": "assistant", "content": ""}, {"estimate": True})
    payloads = [json.loads(b[len("data: "):]) for b in blocks if not b.endswith("[DONE]")]
    assert all("content" not in p["choices"][0]["delta"] for p in payloads)
