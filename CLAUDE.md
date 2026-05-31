# CLAUDE.md — llm_routers

## ⚠️ Публичный репозиторий — не допускать утечек

Этот репозиторий **публичный** (GitHub). При любом редактировании НЕ коммить:
- секреты, ключи, токены, пароли (только через env-переменные / `*.example.env` с пустыми значениями);
- реальные приватные IP (`192.168.x`, `10.x`) и внутренние хосты/SSH-порты;
- внутренние домены и персональные данные (email, телефон, ФИО, адреса);
- имена внутренних проектов, серверов и людей.

Защита — pre-commit hook [`.githooks/pre-commit`](.githooks/pre-commit): generic-сканер форматов ключей + локальный `.sanitize-patterns` (gitignored denylist конкретных значений). После клона активировать: `git config core.hooksPath .githooks`. Сам `.sanitize-patterns` НИКОГДА не коммить.

Зонтичный проект для MCP-серверов routing'а LLM-запросов из Claude Code.

## Что внутри

- `mcp-council/` — единый MCP-сервер с двумя группами tool'ов:
  - **Council (single-shot deliberation):**
    - **`council_ask(question, models=None, ...)`** — Karpathy 3-stage council из 7 моделей (или подмножества через `models=[...]`, **минимум 2**). Для архитектурных решений, спорных вопросов, важного code review, debug сложных багов. **НЕ для рутины** (2-8 мин, дорого).
    - **`model_ask(model_id, prompt, context_paths=[], example_paths=[], ...)`** — один прямой вызов конкретной модели из `models.CATALOG`. Для тяжёлой суммаризации логов, шаблонной генерации, переводов. Заменяет старые `deepseek_read/draft` и `minimax_read/draft` (пакеты `mcp-deepseek` и `mcp-minimax` удалены 2026-05-21).
    - Async-pattern для council: `council_ask_async` + `council_status/result/cancel/list_jobs`.
  - **Dialogue (продолжительные диалоги между моделями с anti-convergence):**
    - **`model_debate(question, participants=None, moderator=None, rounds=5, ...)`** — 2+ моделей с противоположными позициями (модератор автогенерирует), N раундов critique/response. Default participants `["glm","kimi","codex"]`.
    - **`model_panel(question, participants=None, roles=None, diversity_monitor=True, devils_advocate_rotation=True, rounds=5, ...)`** — 4-6 моделей в свободной дискуссии, devil's advocate ротация + diversity monitor (re-prompt согласившимся). Default participants = DEFAULT_PANEL_PARTICIPANTS (7, вкл. codex).
    - **`model_socratic(topic, questioner=None, respondent=None, moderator=None, rounds=5, ...)`** — questioner задаёт углубляющие вопросы, respondent отвечает, optional moderator пишет note + summary. Default: deepseek-pro / glm.
    - **`dialogue_continue(session_id, directive, rounds=3)`** — продолжить done-сессию ещё N раундов с user-directive ("углубитесь в X").
    - **`dialogue_status/result/cancel/list_sessions`** — наблюдение/выгрузка/отмена. Все 3 starter tool'а async-only (длительность 5-50 мин). Hard cap rounds=20, активных сессий=20.
- `claude-code-router/` — HTTP-прокси `ccr` (npm `@musistudio/claude-code-router`, ставится глобально). В репо живут `config.example.json` (template без ключей) + `custom_router.js` (симлинк в `~/.claude-code-router/`). Реальный `config.json` с ключами — только в `~/.claude-code-router/`, в git не попадает. См. [claude-code-router/README.md](claude-code-router/README.md) для деталей привязки и потоков запросов.
- `claude-agent-server/` — обёртка `claude -p` CLI в OpenAI-compatible HTTP API (port 8765). Превращает подписку Claude Max/Pro в локальный endpoint для n8n, чат-ботов и любых OpenAI-клиентов. Опционально — boot-task в Windows Task Scheduler (`install_task.ps1`). Tool calling **не работает** (claude -p воспринимает custom tools как prompt injection). См. [claude-agent-server/README.md](claude-agent-server/README.md).
- `codex-agent-server/` — обёртка `codex exec` CLI в OpenAI-compatible HTTP API (port 8766). Превращает подписку ChatGPT/Codex в локальный endpoint. Один API, два режима: **read-only** (чистый чат — дефолт) и **workspace-write** (агент правит файлы). Режим выбирается именем модели (`gpt-5.5` vs `gpt-5.5-agent`) или полем `sandbox` в body; `tools` всегда форсят read-only. Для агентного — containment-проверка `workdir` внутри `CODEX_AGENT_WORKDIR_ROOT`. Дизайн: [docs/superpowers/specs/2026-05-31-codex-agent-server-design.md](docs/superpowers/specs/2026-05-31-codex-agent-server-design.md). См. [codex-agent-server/README.md](codex-agent-server/README.md).
- `bench/` — раннер LLM-бенчмарков (`run.py`, `judge.py`, `report.py`, `models.json`, `prompts/`, `results/`). Пишет markdown-отчёт в корень репо.

## Реестр моделей (`mcp-council/models.py`)

Единый `CATALOG` — source of truth для обоих tool'ов:

| id | model | назначение |
|---|---|---|
| `glm` | glm-5.1 | council member (OCG) |
| `kimi` | kimi-k2.6 | council member (OCG) |
| `deepseek-pro` | deepseek-v4-pro | council member (DeepSeek direct) |
| `qwen` | qwen3.6-plus | council member (OCG) |
| `minimax` | minimax-m2.7 | council member (OCG) |
| `gemini` | gemini-3.1-pro-preview | council member (Helicone Gateway) |
| `codex` | gpt-5.5 | council member (codex-agent-server :8766, read-only) |
| `deepseek-flash` | deepseek-v4-flash | routine worker (model_ask only) |
| `minimax-direct` | abab7-chat-preview | disabled (billing off) |

`COUNCIL_DEFAULT` = первые 7 (без flash и direct). При `models=None` в `council_ask` совещание идёт ровно по этому списку. **`codex` требует запущенного `codex-agent-server` на :8766** — если он недоступен, member падает с `CouncilHTTPError` и совет продолжает остальными.

## Принципы

- **Stateless** — каждый MCP-вызов независим.
- **Sandbox** — `sandbox.py` блокирует `.env`, ключи, secrets, settings.json. Лимит 50 файлов / 500 KB суммарно.
- **Single source of truth** — `models.py::CATALOG` хранит всех моделей, дубликатов `sandbox.py`/`logger.py` больше нет.

## Ключи

Все ключи читаются `mcp-council` (передаются через `~/.claude.json` → `mcpServers.council.env`):
- `DEEPSEEK_KEY` — для deepseek-pro и deepseek-flash
- `OPENCODE_GO_KEY` — для glm, kimi, qwen, minimax (через OCG)
- `HELICONE_GATEWAY_KEY` — для gemini
- `CODEX_AGENT_TOKEN` — для codex (bearer к локальному codex-agent-server :8766)
- `MINIMAX_API_KEY` — для minimax-direct (currently disabled в catalog, billing off)
- `EXA_API_KEY` — для web_search в любом tool

Значения передаются через окружение MCP-сервера и в репозиторий не попадают.

## Registration в Claude Code

MCP-сервер регистрируется в `~/.claude.json` под top-level `mcpServers` как `council`. Tool-пути: `mcp__council__council_ask`, `mcp__council__model_ask`, `mcp__council__council_ask_async`, `mcp__council__council_status/result/cancel/list_jobs`.

## Tests

```bash
cd mcp-council
pip install -e ".[dev]"
pytest -v
```

## Связанные документы

- Бенчи моделей: раннер [`bench/`](bench/)
- Per-model quirks (thinking/reasoning_effort) задаются в `mcp-council/models.py::CATALOG`.

## Coexistence

См. [AGENTS.md](AGENTS.md) для общих правил (для Codex CLI и других AI-агентов).
