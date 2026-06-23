"""Tests for openai_client: HTTP retry, error parsing, <think> stripping."""

import httpx
import pytest

import circuit_breaker
import openai_client
from openai_client import CouncilHTTPError, _strip_think, call_openai_compat


@pytest.fixture(autouse=True)
def _reset_breaker():
    # Failures simulated across tests must not leave a host's breaker open and
    # short-circuit later tests.
    circuit_breaker.reset()
    yield
    circuit_breaker.reset()


@pytest.fixture(autouse=True)
def _reset_module_client():
    # call_openai_compat now reuses a module-level AsyncClient. Tests patch
    # httpx.AsyncClient per-test via `patch_httpx`; reset the cache so each test
    # gets its own fake instead of the first test's cached one.
    openai_client._CLIENT = None
    yield
    openai_client._CLIENT = None


def _make_response(status_code: int, json_data: dict | None = None, text: str | None = None):
    """Build an httpx.Response without going through httpx.MockTransport."""
    if json_data is not None:
        content = httpx.Response(status_code, json=json_data).content
    else:
        content = (text or "").encode("utf-8")
    return httpx.Response(status_code, content=content)


class _FakeClient:
    def __init__(self, responses: list[httpx.Response]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        if not self._responses:
            raise RuntimeError("no more fake responses")
        return self._responses.pop(0)


@pytest.fixture
def patch_httpx(monkeypatch):
    """Patches httpx.AsyncClient. Returns the fake client list for assertions."""
    holder: dict = {}

    def _factory(*args, **kwargs):
        client = holder["client"]
        return client

    monkeypatch.setattr("openai_client.httpx.AsyncClient", _factory)
    return holder


def _ok_response():
    return _make_response(
        200,
        json_data={
            "choices": [{"message": {"content": "hello world"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


async def test_open_breaker_short_circuits_without_http(patch_httpx):
    # Force the host's breaker open, then assert the call raises immediately and
    # never touches the (would-be-failing) HTTP client.
    host = "blocked.example.com"
    for _ in range(circuit_breaker.FAILURE_THRESHOLD):
        circuit_breaker.record_failure(host)
    fake = _FakeClient([])  # no responses — a real request would RuntimeError
    patch_httpx["client"] = fake
    with pytest.raises(CouncilHTTPError, match="circuit_open"):
        await call_openai_compat(
            base_url=f"https://{host}/v1",
            api_key="sk-test",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
        )
    assert fake.calls == []


async def test_success_clears_breaker_failures(patch_httpx):
    host = "flaky.example.com"
    for _ in range(circuit_breaker.FAILURE_THRESHOLD - 1):
        circuit_breaker.record_failure(host)
    patch_httpx["client"] = _FakeClient([_ok_response()])
    await call_openai_compat(
        base_url=f"https://{host}/v1",
        api_key="sk-test",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
    )
    assert circuit_breaker.snapshot().get(host) is None  # streak reset on success


async def test_success_with_record_breaker_false_keeps_streak(patch_httpx):
    # healthcheck passes record_breaker=False so a probe can't touch the breaker.
    # A successful probe must NOT reset an accumulated fail-streak either.
    host = "probed.example.com"
    for _ in range(circuit_breaker.FAILURE_THRESHOLD - 1):
        circuit_breaker.record_failure(host)
    patch_httpx["client"] = _FakeClient([_ok_response()])
    await call_openai_compat(
        base_url=f"https://{host}/v1",
        api_key="sk-test",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        record_breaker=False,
    )
    assert circuit_breaker.snapshot()[host]["fails"] == circuit_breaker.FAILURE_THRESHOLD - 1


async def test_call_returns_content_and_usage(patch_httpx):
    patch_httpx["client"] = _FakeClient([_ok_response()])
    out = await call_openai_compat(
        base_url="https://example.com/v1",
        api_key="sk-test",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
    )
    assert out["content"] == "hello world"
    assert out["tokens_in"] == 10
    assert out["tokens_out"] == 5


async def test_strip_think():
    assert _strip_think("<think>reasoning</think>answer") == "answer"
    assert _strip_think("plain") == "plain"
    assert _strip_think("<think>a</think>  hello  ") == "hello"


async def test_strip_think_applied_in_response(patch_httpx):
    resp = _make_response(
        200,
        json_data={
            "choices": [{"message": {"content": "<think>reasoning</think>final answer"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )
    patch_httpx["client"] = _FakeClient([resp])
    out = await call_openai_compat(
        base_url="https://example.com/v1",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "q"}],
        max_tokens=10,
    )
    assert out["content"] == "final answer"


async def test_http_402_no_retry(patch_httpx):
    patch_httpx["client"] = _FakeClient(
        [_make_response(402, text='{"error":"insufficient_balance"}')]
    )
    with pytest.raises(CouncilHTTPError, match="402"):
        await call_openai_compat(
            base_url="https://example.com/v1",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "q"}],
            max_tokens=10,
        )


async def test_http_429_retried_then_succeeds(patch_httpx, monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("openai_client.asyncio.sleep", fake_sleep)
    patch_httpx["client"] = _FakeClient(
        [_make_response(429, text=""), _ok_response()]
    )
    out = await call_openai_compat(
        base_url="https://example.com/v1",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "q"}],
        max_tokens=10,
    )
    assert out["content"] == "hello world"
    # First-attempt backoff per RETRY_BACKOFFS[0].
    assert sleeps == [15]


async def test_reasoning_content_surfaced_for_deepseek_thinking(patch_httpx):
    """DeepSeek thinking-mode returns `reasoning_content` alongside content/
    tool_calls. The client must surface it so the tool-loop can echo it back
    on the next call (the DeepSeek API rejects follow-up calls otherwise)."""
    body = {
        "choices": [{
            "message": {
                "content": None,
                "reasoning_content": "thinking step 1 then step 2",
                "tool_calls": [{
                    "id": "call_x",
                    "function": {"name": "web_search", "arguments": '{"query":"q"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    patch_httpx["client"] = _FakeClient([_make_response(200, json_data=body)])
    out = await call_openai_compat(
        base_url="https://api.deepseek.com/v1",
        api_key="k",
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "q"}],
        max_tokens=10,
    )
    assert out["reasoning_content"] == "thinking step 1 then step 2"
    assert out["tool_calls"][0]["id"] == "call_x"
    assert out["finish_reason"] == "tool_calls"


async def test_http_500_retried_then_succeeds(patch_httpx, monkeypatch):
    """500 from upstream is transient (OCG flakiness) — retry, don't fail fast."""
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("openai_client.asyncio.sleep", fake_sleep)
    patch_httpx["client"] = _FakeClient(
        [_make_response(500, text="oops"), _ok_response()]
    )
    out = await call_openai_compat(
        base_url="https://example.com/v1",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "q"}],
        max_tokens=10,
    )
    assert out["content"] == "hello world"
    assert sleeps == [15]


async def test_http_500_exhausted_raises(patch_httpx, monkeypatch):
    """If 500 persists past all retries, raise CouncilHTTPError (overload)."""
    async def fake_sleep(_s):
        pass

    monkeypatch.setattr("openai_client.asyncio.sleep", fake_sleep)
    patch_httpx["client"] = _FakeClient(
        [
            _make_response(500, text="oops"),
            _make_response(500, text="oops"),
            _make_response(500, text="oops"),
        ]
    )
    with pytest.raises(CouncilHTTPError, match="overload"):
        await call_openai_compat(
            base_url="https://example.com/v1",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "q"}],
            max_tokens=10,
        )


async def test_empty_content_raises(patch_httpx):
    resp = _make_response(
        200,
        json_data={
            "choices": [{"message": {"content": ""}}],
            "usage": {},
        },
    )
    patch_httpx["client"] = _FakeClient([resp])
    with pytest.raises(CouncilHTTPError, match="empty"):
        await call_openai_compat(
            base_url="https://example.com/v1",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "q"}],
            max_tokens=10,
        )


async def test_extra_payload_merged(patch_httpx):
    fake = _FakeClient([_ok_response()])
    patch_httpx["client"] = fake
    await call_openai_compat(
        base_url="https://example.com/v1",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "q"}],
        max_tokens=10,
        extra_payload={"thinking": {"type": "disabled"}},
    )
    sent = fake.calls[0]["json"]
    assert sent["thinking"] == {"type": "disabled"}
    assert sent["model"] == "m"


async def test_extra_payload_overrides_temperature_but_not_protected_keys(patch_httpx):
    # kimi-k2.7-code accepts only temperature=1; the catalog forces it via
    # `extra` (must win over the council default). model/messages/stream are
    # protected — extra must NOT be able to clobber them.
    fake = _FakeClient([_ok_response()])
    patch_httpx["client"] = fake
    await call_openai_compat(
        base_url="https://example.com/v1",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "q"}],
        max_tokens=10,
        temperature=0.3,
        extra_payload={"temperature": 1, "model": "evil", "stream": True},
    )
    sent = fake.calls[0]["json"]
    assert sent["temperature"] == 1     # extra overrode the 0.3 default
    assert sent["model"] == "m"          # protected — not clobbered
    assert sent["stream"] is False       # protected — not clobbered


async def test_network_error_raises(patch_httpx, monkeypatch):
    class _RaisingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            raise httpx.ConnectError("boom")

    patch_httpx["client"] = _RaisingClient()
    with pytest.raises(CouncilHTTPError, match="network error"):
        await call_openai_compat(
            base_url="https://example.com/v1",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "q"}],
            max_tokens=10,
        )
