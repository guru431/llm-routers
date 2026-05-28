"""Universal OpenAI-compatible async HTTP client used by mcp-council.

Supports OCG (OpenCode Go), DeepSeek direct, Helicone Gateway — все они принимают
один и тот же /v1/chat/completions схему. Различия (thinking/reasoning_effort,
min max_tokens) задаются через `extra` и `min_max_tokens` в config.COUNCIL.

Retry: 2 раза на HTTP 429/529 с backoff [60s, 120s].
402 (insufficient balance) — без retry, сразу ошибка.

Strip <think>...</think> блоки из ответа (некоторые модели — Kimi, GLM —
возвращают reasoning в этом виде даже когда thinking отключён).
"""

import asyncio
import re

import httpx

# Read timeout for the upstream LLM. Thinking-style models routed via OCG can
# spend 2-5 minutes before any bytes are returned (full response held until
# completion); 120s caused Kimi/Qwen/MiniMax to all ReadTimeout silently.
# DeepSeek direct streams its body in chunks so it never hit the cap even when
# total wall-time exceeded 6 minutes.
DEFAULT_TIMEOUT = 600.0
# 500/502/503: transient upstream errors observed at OCG (5 outages / 7 weeks
# per project notes). Worth retrying — they typically clear within a minute.
# 529: Anthropic-style overload. 429: rate limit.
RETRY_STATUSES = (429, 500, 502, 503, 529)
# Backoff for 5xx is shorter than for 429 because the upstream is usually back
# within seconds; we still keep two attempts so a longer outage falls through.
RETRY_BACKOFFS = (15, 45)  # seconds between attempts


class CouncilHTTPError(Exception):
    """Любая ошибка вызова OpenAI-compatible endpoint."""


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
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """Один POST к {base_url}/chat/completions с retry на 429/500/502/503/529.

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
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if extra_payload:
        payload.update(extra_payload)
    if response_format is not None:
        payload["response_format"] = response_format
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    last_error: str | None = None
    # One AsyncClient across retries — saves TCP+TLS handshake per attempt and,
    # for council runs, lets multiple members reuse the same connection pool to
    # the same host (OCG serves 4/6 council members in the default catalog).
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(len(RETRY_BACKOFFS) + 1):
            try:
                resp = await client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException as e:
                # ReadTimeout / ConnectTimeout / PoolTimeout. Thinking-mode models
                # can hold the connection for minutes and a transient blip from the
                # provider shouldn't kill an otherwise viable request. Apply the
                # same backoff as HTTP 5xx for consistency.
                detail = str(e) or type(e).__name__
                if attempt >= len(RETRY_BACKOFFS):
                    raise CouncilHTTPError(
                        f"timeout after {attempt + 1} attempts: {detail}"
                    ) from e
                await asyncio.sleep(RETRY_BACKOFFS[attempt])
                last_error = f"timeout {detail} (retry)"
                continue
            except httpx.HTTPError as e:
                # Non-timeout transport error (DNS, TLS, etc.). str(e) is often
                # empty on these — fall back to the class name so logs aren't blank.
                detail = str(e) or type(e).__name__
                raise CouncilHTTPError(f"network error: {detail}") from e

            if resp.status_code == 402:
                body = resp.text[:200] if resp.text else ""
                raise CouncilHTTPError(f"http 402 insufficient_balance: {body}")

            if resp.status_code in RETRY_STATUSES:
                if attempt >= len(RETRY_BACKOFFS):
                    raise CouncilHTTPError(
                        f"overload after {attempt + 1} attempts (last status {resp.status_code})"
                    )
                await asyncio.sleep(RETRY_BACKOFFS[attempt])
                last_error = f"http {resp.status_code} (retry)"
                continue

            if resp.status_code != 200:
                body = resp.text[:200] if resp.text else ""
                raise CouncilHTTPError(f"http {resp.status_code}: {body}")

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
            }

    raise CouncilHTTPError(last_error or "unreachable")  # pragma: no cover
