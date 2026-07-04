"""Universal OpenAI-compatible async HTTP client used by mcp-council.

Supports OCG (OpenCode Go), DeepSeek direct, Helicone Gateway — все они принимают
один и тот же /v1/chat/completions схему. Различия (thinking/reasoning_effort,
min max_tokens) задаются через `extra` и `min_max_tokens` в config.COUNCIL.

Retry: до 2 повторов на HTTP 408/429/500/502/503/504/529 и на timeout, backoff
[15s, 45s] между попытками (RETRY_BACKOFFS).
402 (insufficient balance) — без retry, сразу ошибка.

Strip <think>...</think> блоки из ответа (некоторые модели — Kimi, GLM —
возвращают reasoning в этом виде даже когда thinking отключён).
"""

import asyncio
import re
from urllib.parse import urlparse

import httpx

import circuit_breaker

# Per-phase timeout. The 600s READ ceiling is needed for thinking-style models
# routed via OCG (they hold the connection 2-5 min before any bytes; 120s caused
# Kimi/Qwen/MiniMax to ReadTimeout silently). But a bare float would apply 600s
# to CONNECT too, so a dead/black-holed host could eat a long connect wait — pin
# a short connect/pool timeout while keeping the long read ceiling.
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)
# 500/502/503: transient upstream errors observed at OCG (5 outages / 7 weeks
# per project notes). Worth retrying — they typically clear within a minute.
# 504: gateway timeout (typical for an overloaded OCG gateway). 408: request
# timeout. 529: Anthropic-style overload. 429: rate limit.
RETRY_STATUSES = (408, 429, 500, 502, 503, 504, 529)
# Backoff for 5xx is shorter than for 429 because the upstream is usually back
# within seconds; we still keep two attempts so a longer outage falls through.
RETRY_BACKOFFS = (15, 45)  # seconds between attempts


class CouncilHTTPError(Exception):
    """Любая ошибка вызова OpenAI-compatible endpoint."""


# Module-level client, lazily created on first use. Reused across every call so
# the TCP+TLS connection pool survives between requests — council fan-out has
# 4/6 default members on the same OCG host, so they share connections instead of
# re-handshaking per attempt. Must be created inside a running event loop (httpx
# binds to the loop), hence lazy init rather than a module-import-time singleton.
_CLIENT: "httpx.AsyncClient | None" = None


def _get_client() -> "httpx.AsyncClient":
    global _CLIENT
    if _CLIENT is None:
        # Explicit pool limits so behaviour under several concurrent async jobs
        # (each fanning out to 7 members × multi-turn web_search) is predictable
        # instead of relying on httpx defaults (100/20).
        _CLIENT = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        )
    return _CLIENT


async def close_client() -> None:
    """Close and reset the module-level AsyncClient.

    Wire into a server shutdown/lifecycle hook so the connection pool is
    released cleanly. Also lets cross-loop tests / hot-reload drop a client
    bound to a now-dead event loop — the next call lazily re-creates it.
    """
    global _CLIENT
    if _CLIENT is not None:
        await _CLIENT.aclose()
        _CLIENT = None


def _strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()


async def call_openai_compat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.3,
    extra_payload: dict | None = None,
    response_format: dict | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    timeout: "float | httpx.Timeout" = DEFAULT_TIMEOUT,
    max_attempts: int | None = None,
    record_breaker: bool = True,
) -> dict:
    """Один POST к {base_url}/chat/completions с retry на 408/429/500/502/503/504/529.

    `max_attempts`: cap on total HTTP attempts, clamped to a minimum of 1
    (1 = no retries; values <1 are treated as 1). None = full RETRY_BACKOFFS
    budget (default council behavior).
    `record_breaker`: when False, infra failures do NOT feed the circuit
    breaker (healthcheck uses this so a probe can't trip the breaker for the
    real council path).

    Returns::
        {
            "content": str | None,         # None if the model chose to call tools
            "tool_calls": list[dict] | None,  # OpenAI-style tool_calls or None
            "finish_reason": str | None,
            "tokens_in": int | None,
            "tokens_out": int | None,
        }

    Raises CouncilHTTPError on network/HTTP/parsing failure or after exhausting retries.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    host = urlparse(base_url).netloc or base_url
    # Short-circuit if this provider was recently marked down — don't spend the
    # full retry/timeout budget on a host we already know is failing.
    cooldown = circuit_breaker.open_for(host)
    if cooldown:
        raise CouncilHTTPError(
            f"circuit_open for {host}: provider marked down, cooling down "
            f"{int(cooldown)}s"
        )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if extra_payload:
        # Protected keys must come from explicit args, never from a catalog
        # `extra` dict (which is attacker-distant but still a footgun). temperature
        # / max_tokens / response_format stay overridable on purpose — the kimi
        # catalog entry forces temperature=1 via extra.
        protected = {"model", "messages", "stream"}
        payload.update({k: v for k, v in extra_payload.items() if k not in protected})
    if response_format is not None:
        payload["response_format"] = response_format
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    last_error: str | None = None
    # Reused module-level AsyncClient — saves TCP+TLS handshake per attempt and,
    # for council runs, lets multiple members reuse the same connection pool to
    # the same host (OCG serves 4/6 council members in the default catalog).
    # timeout is per-request so different callers (council vs healthcheck) can
    # set their own ceiling on the shared client.
    client = _get_client()
    # Retry budget: full RETRY_BACKOFFS by default; max_attempts caps it (1 = no
    # retries) so healthcheck can probe without burning the full backoff budget.
    # Clamp to >=1 attempt so max_attempts=0 still makes one request (the
    # docstring contract is "1 = no retries", there is no "0 attempts" mode).
    max_retries = len(RETRY_BACKOFFS)
    if max_attempts is not None:
        max_retries = min(max_retries, max(0, max(1, max_attempts) - 1))

    def _record_failure() -> None:
        if record_breaker:
            circuit_breaker.record_failure(host)

    for attempt in range(max_retries + 1):
        try:
            resp = await client.post(url, headers=headers, json=payload, timeout=timeout)
        except httpx.TimeoutException as e:
            # ReadTimeout / ConnectTimeout / PoolTimeout. Thinking-mode models
            # can hold the connection for minutes and a transient blip from the
            # provider shouldn't kill an otherwise viable request. Apply the
            # same backoff as HTTP 5xx for consistency.
            detail = str(e) or type(e).__name__
            if attempt >= max_retries:
                _record_failure()
                raise CouncilHTTPError(
                    f"timeout after {attempt + 1} attempts: {detail}"
                ) from e
            # A concurrent member of the same fan-out may have tripped the breaker
            # while we were failing — re-check before spending another backoff so
            # one provider outage doesn't make every co-hosted member burn its
            # full retry budget before the breaker helps.
            if circuit_breaker.open_for(host):
                raise CouncilHTTPError(f"circuit_open for {host}: tripped mid-retry")
            await asyncio.sleep(RETRY_BACKOFFS[attempt])
            last_error = f"timeout {detail} (retry)"
            continue
        except httpx.HTTPError as e:
            # Non-timeout transport error (DNS, TLS, etc.). str(e) is often
            # empty on these — fall back to the class name so logs aren't blank.
            detail = str(e) or type(e).__name__
            _record_failure()
            raise CouncilHTTPError(f"network error: {detail}") from e

        if resp.status_code == 402:
            body = resp.text[:200] if resp.text else ""
            raise CouncilHTTPError(f"http 402 insufficient_balance: {body}")

        if resp.status_code in RETRY_STATUSES:
            if attempt >= max_retries:
                _record_failure()
                raise CouncilHTTPError(
                    f"overload after {attempt + 1} attempts (last status {resp.status_code})"
                )
            # See the timeout branch: short-circuit if a sibling already opened
            # the breaker for this host mid-retry.
            if circuit_breaker.open_for(host):
                raise CouncilHTTPError(f"circuit_open for {host}: tripped mid-retry")
            await asyncio.sleep(RETRY_BACKOFFS[attempt])
            last_error = f"http {resp.status_code} (retry)"
            continue

        if resp.status_code != 200:
            body = resp.text[:200] if resp.text else ""
            raise CouncilHTTPError(f"http {resp.status_code}: {body}")

        # HTTP 200 = the host is up: clear any accumulated infra-failure streak
        # NOW, even if the body turns out empty/malformed below (those are not
        # infra outages). record_failure stays reserved for transport/timeout/5xx.
        if record_breaker:
            circuit_breaker.record_success(host)

        try:
            data = resp.json()
        except ValueError as e:
            raise CouncilHTTPError(f"invalid JSON in response: {e}") from e

        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise CouncilHTTPError(f"invalid response structure: {e}") from e

        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        # DeepSeek thinking-mode returns a separate `reasoning_content` alongside
        # tool_calls. The DeepSeek API STRICTLY REQUIRES it to be echoed back in
        # the next assistant message — otherwise the follow-up call rejects with
        # http 400: "The `reasoning_content` in the thinking mode must be passed
        # back to the API." We surface it here so the caller can put it back
        # into the conversation. Other providers leave the field absent and the
        # caller passing it back is harmless (extra key is ignored).
        reasoning_content = msg.get("reasoning_content")
        finish_reason = choice.get("finish_reason")

        # Accept the response when either (a) we got actual content, or (b) the
        # model decided to call tools. Only fail when both are missing — that's
        # a degenerate response (often max_tokens spent on hidden reasoning).
        if not content and not tool_calls:
            raise CouncilHTTPError(
                f"empty content (finish_reason={finish_reason})"
            )

        usage = data.get("usage", {}) or {}
        return {
            "content": _strip_think(content) if content else None,
            "tool_calls": tool_calls,
            "reasoning_content": reasoning_content,
            "finish_reason": finish_reason,
            "tokens_in": usage.get("prompt_tokens"),
            "tokens_out": usage.get("completion_tokens"),
            # Number of HTTP attempts spent (1 = succeeded first try). Used by
            # council usage-accounting to count retries on the success path.
            "attempts": attempt + 1,
        }

    raise CouncilHTTPError(last_error or "unreachable")  # pragma: no cover
