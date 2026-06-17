/**
 * custom_router.js — Claude Code Router
 *
 * Routing logic:
 *   1. Read model name from request body (req.body.model)
 *   2. If it's a known OpenCode Go model — pass through as-is
 *   3. If it's a Claude name (claude-*) — fall back to Router.default (set in config.json)
 *
 * How per-project model selection works:
 *   In each project's .claude/settings.local.json set:
 *     "env": {
 *       "ANTHROPIC_MODEL": "minimax-m3"   // or glm-5.1, qwen3.6-plus, etc.
 *     }
 *   Claude Code extension reads this and puts the model in the request body.
 *   Our router sees it and routes to opencode provider with that exact model id.
 *
 * Falls back to Router.default if the model is a claude-* name (extension forgot
 * to apply env var) or if model is missing.
 */

// Fallback list of OpenCode Go models if config isn't passed to the router.
// Normally we derive the set from config.Providers[opencode].models (below) so
// the two stay in sync; this hardcode is only the safety net.
// SOURCE OF TRUTH: config.example.json's Providers[opencode].models. This list is
// a hand-maintained copy of it — keep both in sync when models change (no build step).
const OPENCODE_MODELS_FALLBACK = new Set([
  'glm-5.1', 'glm-5',
  'kimi-k2.5', 'kimi-k2.6',
  'mimo-v2-pro', 'mimo-v2-omni', 'mimo-v2.5-pro', 'mimo-v2.5',
  'minimax-m3', 'minimax-m2.5',
  'qwen3.6-plus', 'qwen3.5-plus', 'qwen3.7-plus',
  'deepseek-v4-pro', 'deepseek-v4-flash',
]);

module.exports = async function router(req, config) {
  const model = req.body?.model;

  if (!model) {
    // No model in request — let CCR's built-in scenario routing handle it
    return null;
  }

  // Single source of truth: the opencode provider's model list from config.json.
  // Falls back to the hardcoded set if config isn't available in this signature.
  const configModels = config?.Providers?.find((p) => p.name === 'opencode')?.models;
  const opencodeModels = new Set(
    configModels?.length ? configModels : OPENCODE_MODELS_FALLBACK
  );

  // Per-project override: model id matches an OpenCode Go model → route through opencode
  if (opencodeModels.has(model)) {
    return `opencode,${model}`;
  }

  // Unknown non-claude model: silent fallback to Router.default would mask a
  // typo'd/decommissioned model id, so surface it.
  if (!model.startsWith('claude-')) {
    console.warn(`[custom_router] unknown model "${model}" — falling back to Router.default`);
  }

  // Claude name (claude-opus-4-8, claude-sonnet-4-6, claude-haiku-4-5, etc.)
  // → fall back to built-in routing (Router.default / .background / .think etc.)
  return null;
}
