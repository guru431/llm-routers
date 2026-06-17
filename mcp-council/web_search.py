"""Exa.ai web search client for the council `web_search` tool.

When `web_search=True` is passed to `council_ask`, each council member can call
`web_search(query)` during stage 1. This module is the executor that runs the
actual HTTP call to Exa and renders results in a compact form suitable to feed
back into the LLM as a `tool` message.

Exa was chosen because:
- It returns dense, LLM-friendly snippets (summary + highlights) per result;
- The Exa MCP server is already used in this project, so the API key is
  already in the vault;
- Single HTTP POST → batched search + content fetch in one round-trip.

Keep this dependency-free of council/orchestrator code so it can be unit-tested
on its own.
"""

from __future__ import annotations

import asyncio
import os
import time

import httpx

EXA_API_URL = "https://api.exa.ai/search"
EXA_TIMEOUT_SECONDS = 30.0
# Fixed result count per search. The tool spec exposes only `query`, so callers
# never tune this; keeping it constant avoids a runaway loop costing too much
# (Exa charges ~$0.005-0.01 per request at the volume we use).
NUM_RESULTS = 5


class WebSearchError(Exception):
    """Recoverable error from Exa — the model gets the error text and decides
    how to proceed (try a different query, or give up on search and answer
    from training data)."""


async def web_search_exa(
    query: str,
    *,
    api_key: str | None = None,
    timeout: float = EXA_TIMEOUT_SECONDS,
) -> dict:
    """Run a single Exa search. Returns a dict with results and metadata.

    Shape::
        {
            "query": str,
            "results": [
                {"title": str, "url": str, "summary": str,
                 "highlights": [str, ...]},
                ...
            ],
            "cost_dollars": float | None,
            "latency_ms": int,
        }

    On HTTP / network error raises WebSearchError with a short reason. Caller
    is expected to feed `{"error": str(e)}` back into the model rather than
    propagate the exception.
    """
    if not query or not query.strip():
        raise WebSearchError("empty query")

    key = api_key or os.environ.get("EXA_API_KEY")
    if not key:
        raise WebSearchError("EXA_API_KEY not set")

    payload = {
        "query": query.strip(),
        "numResults": NUM_RESULTS,
        "contents": {
            # ~800 chars of text is enough to anchor the LLM without bloating
            # the context window.
            "text": {"maxCharacters": 800},
            "summary": True,
            "highlights": {"numSentences": 2, "highlightsPerUrl": 2},
        },
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(EXA_API_URL, headers=headers, json=payload)
    except httpx.HTTPError as e:
        detail = str(e) or type(e).__name__
        raise WebSearchError(f"network error: {detail}") from e

    if resp.status_code != 200:
        body = resp.text[:200] if resp.text else ""
        raise WebSearchError(f"http {resp.status_code}: {body}")

    try:
        data = resp.json()
    except ValueError as e:
        raise WebSearchError(f"invalid JSON: {e}") from e

    results: list[dict] = []
    for r in data.get("results", []) or []:
        results.append({
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "summary": (r.get("summary") or "").strip(),
            "highlights": list(r.get("highlights") or []),
        })

    return {
        "query": query,
        "results": results,
        "cost_dollars": data.get("costDollars"),
        "latency_ms": int((time.monotonic() - start) * 1000),
    }


def format_results_for_llm(result: dict) -> str:
    """Render a successful Exa search result into a compact string the LLM
    can consume as the content of a `tool` role message.

    Format:
        # Web search results for: <query>
        Results: 3, latency 1820ms

        ## 1. <title>
        <url>
        Summary: <summary>
        Highlights:
        - <hl1>
        - <hl2>

        ## 2. ...
    """
    query = result.get("query", "")
    results = result.get("results", []) or []
    lines: list[str] = []
    lines.append(f"# Web search results for: {query}")
    lines.append(f"Results: {len(results)}, latency {result.get('latency_ms', 0)}ms")
    lines.append("")
    if not results:
        lines.append("_(no results)_")
        return "\n".join(lines)
    for i, r in enumerate(results, 1):
        lines.append(f"## {i}. {r['title']}")
        lines.append(r["url"])
        if r["summary"]:
            lines.append(f"Summary: {r['summary']}")
        if r["highlights"]:
            lines.append("Highlights:")
            for h in r["highlights"]:
                # Collapse whitespace, drop the [...] placeholders Exa inserts
                # when it strings highlights together.
                h_clean = " ".join(h.split())
                lines.append(f"- {h_clean}")
        lines.append("")
    return "\n".join(lines)


def format_error_for_llm(error: str, query: str) -> str:
    """Render a search error back to the LLM so it can try a different query
    or proceed without search."""
    return (
        f"# Web search failed for: {query}\n"
        f"Error: {error}\n"
        f"You may try a different query or proceed without web search."
    )


# Public spec for the tool, embedded into council `tools` payload.
WEB_SEARCH_TOOL_SPEC: dict = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web (Exa.ai) for current information beyond your "
            "training data. Use sparingly: prefer to combine multiple aspects "
            "into one focused query rather than firing many narrow searches. "
            "Returns up to 5 results with title, url, summary and key "
            "highlight passages. Good for: recent product versions, KB "
            "articles, current CVEs, post-2024 events. Skip if the question "
            "is purely about your existing knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Be specific — include product "
                        "name, version, year, error code where relevant."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}
