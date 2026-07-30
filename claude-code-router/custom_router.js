/**
 * custom_router.js — Claude Code Router
 *
 * Routing logic:
 *   1. Read model name from request body (req.body.model)
 *   2. If it's a known OpenCode Go model — pass through as-is
 *   3. If it's a Claude name (claude-*) — fall back to Router.default (set in config.json)
 *   4. Missing model — fall back to CCR's built-in scenario routing (null)
 *   5. Anything else (unknown/typo'd/decommissioned id) — throw, so the bad id
 *      surfaces instead of being silently rerouted to Router.default
 *
 * How per-project model selection works:
 *   In each project's .claude/settings.local.json set:
 *     "env": {
 *       "ANTHROPIC_MODEL": "minimax-m3"   // or glm-5.2, qwen3.7-plus, etc.
 *     }
 *   Claude Code extension reads this and puts the model in the request body.
 *   Our router sees it and routes to opencode provider with that exact model id.
 *
 * Falls back to Router.default if the model is a claude-* name (extension forgot
 * to apply env var) or if model is missing. An unknown model id is rejected
 * (thrown) rather than silently rerouted, so typos don't run on the wrong model.
 *
 * Traceability (Idea 21): every routing decision is logged as ONE structured
 * JSON line `{"ccr_route":{provider,model,reason}}` to stdout. CCR's custom-router
 * contract only hands us `(req, config)` and expects a "provider,model" string
 * back (or null) — there is no response object to attach a trace header to — so
 * a structured stdout log is the honest maximum. `reason` names WHY each route
 * was chosen so a grep over CCR logs reconstructs the decision path.
 *
 * Single-provider by design: this deployment has exactly one upstream (opencode),
 * so there is no real health/cost/latency multi-provider fallback to make. The
 * only fallback is Router.default (CCR built-in) for claude-* / missing-model
 * requests, traced with reason 'router_default_fallback'. See README
 * ("Каталог моделей и трассировка") for how to keep the model list in sync.
 */

// Fallback list of OpenCode Go models if config isn't passed to the router.
// Normally we derive the set from config.Providers[opencode].models (below) so
// the two stay in sync; this hardcode is only the safety net and its use is
// WARNED about at runtime (a silent hardcode fallback hid config drift before).
// SOURCE OF TRUTH: config.example.json's Providers[opencode].models.
//
// NOT a mirror of mcp-council/models.py::CATALOG. CCR's catalog is the models the
// OpenCode Go subscription exposes (a SUPERSET); CATALOG is the subset the council
// actually calls, with prices/quirks/provider-domain attached. The two diverge on
// purpose — keep THESE TWO copies in sync (config.example.json + this list), not
// three; see README "Каталог моделей и трассировка".
const OPENCODE_MODELS_FALLBACK = new Set([
  'glm-5.2',
  'kimi-k3', 'kimi-k2.7-code',
  'deepseek-v4-pro', 'deepseek-v4-flash',
  'qwen3.7-max', 'qwen3.7-plus',
  'minimax-m3',
  'mimo-v2.5-pro', 'mimo-v2.5',
  'grok-4.5', 'hy3',
]);

// Emit one structured trace line per routing decision. Kept to a single JSON
// object so `grep ccr_route <log>` reconstructs every route + its reason.
function trace(provider, model, reason) {
  try {
    console.log(JSON.stringify({ ccr_route: { provider, model, reason } }));
  } catch (_e) {
    // Never let logging break routing.
  }
}

module.exports = async function router(req, config) {
  const model = req.body?.model;

  if (!model) {
    // No model in request — let CCR's built-in scenario routing handle it.
    trace(null, null, 'no_model_builtin_routing');
    return null;
  }

  // Single source of truth: the opencode provider's model list from config.json.
  // An EMPTY/absent list means the config is broken, not that there are zero
  // models — so we keep routing off the hardcoded net and WARN loudly. This is a
  // NOISY fallback, NOT fail-closed: requests still route, they just route off a
  // list that may be stale. Fail-closed would mean refusing to route at all.
  const configModels = config?.Providers?.find((p) => p.name === 'opencode')?.models;
  let opencodeModels;
  if (configModels?.length) {
    opencodeModels = new Set(configModels);
  } else {
    console.warn(
      '[custom_router] WARNING: config.Providers[opencode].models is empty or ' +
      'missing — using the hardcoded OPENCODE_MODELS_FALLBACK. Fix ' +
      'Providers[opencode].models in ~/.claude-code-router/config.json so routing ' +
      'reflects the real catalog (see README).'
    );
    opencodeModels = OPENCODE_MODELS_FALLBACK;
  }

  // Per-project override: model id matches an OpenCode Go model → route through opencode.
  if (opencodeModels.has(model)) {
    trace('opencode', model, 'opencode_model_match');
    return `opencode,${model}`;
  }

  // Claude name (claude-opus-5, claude-sonnet-5, claude-haiku-4-5, etc.)
  // → fall back to built-in routing (Router.default / .background / .think etc.).
  // Single-provider deployment: Router.default IS the only fallback by design.
  if (typeof model === 'string' && model.startsWith('claude-')) {
    trace(null, model, 'router_default_fallback');
    return null;
  }

  // Unknown model id (typo'd / decommissioned / wrong provider). Returning null
  // here would silently run the request on Router.default (a DIFFERENT model)
  // and mask the mistake. Fail closed with an explicit, traceable error so the
  // bad model id surfaces to the caller instead of being silently rerouted.
  trace(null, model, 'unknown_model_rejected');
  throw new Error(
    `[custom_router] unknown model "${model}": not an OpenCode Go model and not a ` +
    `claude-* name. Fix ANTHROPIC_MODEL, or add the id to Providers[opencode].models.`
  );
}
