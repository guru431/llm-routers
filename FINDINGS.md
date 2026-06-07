# Findings
Побочные находки. Ревизия: MonthlyStratReview 1-го числа. Stale >90 дней → alert.

## 2026-06-07 · CLAUDE.md/AGENTS.md sync drift — llm_routers [P3]
**Контекст:** еженедельный sync-check (`cron/agents-md-sync-check.py`)
**Что:** обнаружены расхождения между CLAUDE.md и AGENTS.md.
**Предложение:** свести руками либо принять расхождение как намеренное.
**Статус:** open

<details>
<summary>Диагностика от DeepSeek</summary>

### CRITICAL_MISSING_IN_AGENTS
- **`CODEX_AGENT_TOKEN`** — в разделе «Env keys» AGENTS.md перечислены 4 из 6 ключей из CLAUDE.md, но пропущен `CODEX_AGENT_TOKEN` (bearer для `codex-agent-server` на :8766). `codex` — один из 7 default council members, без этого токена `council_ask` падает с `CouncilHTTPError`.

### OUTDATED_IN_AGENTS
- (нет)

### CONTRADICTIONS
- (нет)
</details>

