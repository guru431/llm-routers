"""Model catalog for mcp-council.

Single source of truth for both council deliberation members and single-model
routine workers. Replaces the old config.py + duplicate configs in
mcp-deepseek/mcp-minimax.

Quirks (`extra`, `min_max_tokens`) per model are documented inline below —
they encode provider-specific requirements (e.g. GLM needs thinking disabled,
Kimi k2.7-code needs reasoning_effort "minimal") to avoid truncated/garbage output.

Env key names (`env_key`) are read from the process environment; see the
project README for how keys are provided to the MCP server.

`provider` is the independent failure/credential DOMAIN — models sharing one
(same gateway + same key) go down together (outage, rate-limit, revoked key).
The council counts distinct domains, not raw member count, before it will call
a verdict independently corroborated (see council._build_summary).

Pricing (`price_in`/`price_out`, USD per 1M tokens) is a REFERENCE PAYG list
price, NOT what the run is billed. Every default council member actually bills
flat-rate via a subscription (OCG $10/mo: glm/kimi/deepseek-pro/qwen/minimax/
deepseek-flash; ChatGPT-flat: codex; Helicone gemini has no listed price), so
the real incremental cost of a run is ≈ $0 + any Exa web_search. The DeepSeek
numbers are kept only as a "what this would cost at DeepSeek-direct PAYG"
yardstick and feed council usage.reference_payg_cost_usd — deliberately named
so automation never mistakes it for actual spend. DeepSeek list prices as of
2026-05.
"""

from __future__ import annotations

OCG = "https://opencode.ai/zen/go/v1"
DS = "https://api.deepseek.com/v1"
HEL = "https://ai-gateway.helicone.ai/v1"
MM = "https://api.minimaxi.chat/v1"


CATALOG: dict[str, dict] = {
    # --- Council members (default participants of council_ask) ---
    "glm": {
        "model": "glm-5.2",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
        "provider": "opencode-go",
        "extra": {"thinking": {"type": "disabled"}},
        # OCG flat-rate subscription — no published per-token price.
        "price_in": None,
        "price_out": None,
    },
    "kimi": {
        "model": "kimi-k2.7-code",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
        "provider": "opencode-go",
        # k2.7-code не поддерживает reasoning_effort="none" (HTTP 400, в отличие
        # от k2.6) — допустимы minimal|low|medium. minimal — ближайшее к none.
        # k2.7-code (Moonshot) принимает ТОЛЬКО temperature=1 ("invalid
        # temperature: only 1 is allowed for this model") — overrides council
        # default 0.3 через extra (payload.update(extra) wins, см. openai_client).
        "extra": {"reasoning_effort": "minimal", "temperature": 1},
        "min_max_tokens": 30000,
        "price_in": None,
        "price_out": None,
    },
    "deepseek-pro": {
        # via OCG-прокси с 2026-06-07 (DeepSeek direct PAYG исчерпан, вряд ли вернётся)
        "model": "deepseek-v4-pro",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
        "provider": "opencode-go",
        # Reference PAYG list price (billed flat-rate via OCG — not per-token).
        # DeepSeek-direct list price (50% off promo): $0.435/1M in, $0.87/1M out.
        "price_in": 0.435,
        "price_out": 0.87,
    },
    "qwen": {
        "model": "qwen3.6-plus",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
        "provider": "opencode-go",
        "price_in": None,
        "price_out": None,
    },
    "minimax": {
        "model": "minimax-m3",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
        "provider": "opencode-go",
        "min_max_tokens": 30000,
        "price_in": None,
        "price_out": None,
    },
    "gemini": {
        "model": "gemini-3.1-pro-preview",
        "base_url": HEL,
        "env_key": "HELICONE_GATEWAY_KEY",
        "provider": "helicone",
        "min_max_tokens": 30000,
        # No published price for 3.1-pro-preview via Helicone Gateway.
        "price_in": None,
        "price_out": None,
    },
    "codex": {
        # codex-agent-server (local OpenAI-compatible wrapper over `codex exec`,
        # ChatGPT subscription). `sandbox: read-only` forces pure text generation
        # — без него дефолт сервера тоже read-only, но члену совета агентный режим
        # не нужен ни при каких настройках сервера. Сервер должен быть запущен на
        # :8766; CODEX_AGENT_TOKEN передаётся через окружение MCP-сервера.
        "model": "gpt-5.5",
        "base_url": "http://127.0.0.1:8766/v1",
        "env_key": "CODEX_AGENT_TOKEN",
        "provider": "codex-agent",
        "extra": {"sandbox": "read-only"},
        # `codex exec gpt-5.5` is a reasoning model spawned as a subprocess
        # (cold start) — a real POST, not a light /health GET. The default 12s
        # probe almost always ReadTimeouts on a healthy server, so give this
        # local agent-server member a longer healthcheck ceiling.
        "healthcheck_timeout": 75.0,
        # ChatGPT/Codex flat subscription — no per-token price.
        "price_in": None,
        "price_out": None,
    },

    # --- Routine workers (model_ask only) ---
    "deepseek-flash": {
        # via OCG-прокси с 2026-06-07 (DeepSeek direct PAYG исчерпан, вряд ли вернётся)
        "model": "deepseek-v4-flash",
        "base_url": OCG,
        "env_key": "OPENCODE_GO_KEY",
        "provider": "opencode-go",
        # Reference PAYG list price (billed flat-rate via OCG — not per-token).
        # DeepSeek-direct list price: $0.14/1M in, $0.28/1M out.
        "price_in": 0.14,
        "price_out": 0.28,
    },
    "minimax-direct": {
        "model": "abab7-chat-preview",
        "base_url": MM,
        "env_key": "MINIMAX_API_KEY",
        "provider": "minimax-direct",
        "enabled": False,
        "price_in": None,
        "price_out": None,
    },
}


COUNCIL_DEFAULT: list[str] = [
    "glm",
    "kimi",
    "deepseek-pro",
    "qwen",
    "minimax",
    "gemini",
    "codex",
]


# Named council presets — convenience over hand-listing model ids. Kept as
# EXPLICIT lists so a change is a one-line edit that never silently reshuffles a
# caller's council. No "local" preset: the catalog has no local-runtime members.
#
# CAVEAT: these labels are a HEURISTIC, not a bench-validated product ranking.
# The last local bench run predates the current catalog (models have since
# changed — e.g. GLM 5.2, Kimi K2.7-code), and raw bench results are gitignored,
# so "best"/"balanced"/"cheap" are NOT reproducible claims for today's model
# versions. Editing CATALOG can silently degrade a preset. Treat as a starting
# point; re-run bench/ and update deliberately before relying on preset quality.
# Note also "cheap" is a single-provider pair (both OCG) — a two-model, one-domain
# council never earns a quorum-backed "adopt" verdict (see council._build_summary).
PRESETS: dict[str, list[str]] = {
    "best": list(COUNCIL_DEFAULT),                    # all strongest members
    "balanced": ["deepseek-pro", "glm", "gemini"],    # strong + mid mix, fewer calls
    "cheap": ["glm", "qwen"],                          # lowest-cost OCG pair
}


class UnknownModelError(RuntimeError):
    """Raised when a model_id is not present in CATALOG."""


class DisabledModelError(RuntimeError):
    """Raised when a model_id is present but disabled (enabled: False)."""


class UnknownPresetError(RuntimeError):
    """Raised when a preset name is not in PRESETS."""


def resolve_preset(name: str) -> list[str]:
    """Return the model-id list for a named preset (copy). Raises UnknownPresetError."""
    if name not in PRESETS:
        raise UnknownPresetError(
            f"unknown preset: '{name}'. Available: {sorted(PRESETS)}"
        )
    return list(PRESETS[name])


def provider_domain(model_id: str) -> str:
    """Return the independent failure/credential domain for a model id.

    Members sharing a domain (same gateway + same key) fail together, so the
    council counts DISTINCT domains — not raw member count — when judging whether
    a verdict rests on independent sources. Unknown ids (e.g. test stubs not in
    CATALOG) map to themselves, so each counts as its own domain."""
    cfg = CATALOG.get(model_id)
    if not cfg:
        return model_id
    return cfg.get("provider") or model_id


def resolve_member(id: str) -> dict:
    """Return cfg dict with `id` injected. Raises UnknownModelError / DisabledModelError."""
    if id not in CATALOG:
        raise UnknownModelError(
            f"unknown model_id: '{id}'. Available: {sorted(CATALOG.keys())}"
        )
    cfg = CATALOG[id]
    if cfg.get("enabled") is False:
        raise DisabledModelError(f"model '{id}' is disabled in catalog")
    return {"id": id, **cfg}


def resolve_members(ids: list[str] | None) -> list[dict]:
    """Resolve a list of model_ids into cfgs. None → COUNCIL_DEFAULT.

    Preserves input order, dropping duplicate ids (a model can appear only once —
    duplicates would collide on council pseudonyms and skew aggregation).
    Raises on first invalid id.
    """
    if ids is None:
        ids = COUNCIL_DEFAULT
    seen: set[str] = set()
    unique: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return [resolve_member(i) for i in unique]
