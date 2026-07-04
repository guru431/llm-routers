# llm_routers

Зонтичный проект для MCP-серверов и прокси, маршрутизирующих запросы к разным
LLM-провайдерам из Claude Code и OpenAI-совместимых клиентов.

## Содержимое

| Пакет | Тип | Назначение |
|---|---|---|
| [`mcp-council/`](mcp-council/) | MCP | Единый router LLM-запросов. Tool `council_ask` — Karpathy 3-stage council (несколько моделей или подмножество ≥2). Tool `model_ask` — один прямой вызов конкретной модели. Плюс группа dialogue-tool'ов (`model_debate` / `model_panel` / `model_socratic`) для продолжительных диалогов между моделями с anti-convergence. |
| [`claude-code-router/`](claude-code-router/) | HTTP-proxy | Кастомизация [`@musistudio/claude-code-router`](https://github.com/musistudio/claude-code-router) (`ccr`): `config.example.json` + `custom_router.js`. Прокси Anthropic-API → произвольный OpenAI-совместимый upstream с per-project выбором модели. |
| [`claude-agent-server/`](claude-agent-server/) | HTTP-proxy | Обёртка `claude -p` CLI в OpenAI-compatible HTTP API. Превращает подписку Claude Max/Pro в локальный API endpoint для n8n, чат-ботов и любых OpenAI-клиентов. |
| [`codex-agent-server/`](codex-agent-server/) | HTTP-proxy | Обёртка `codex exec` CLI в OpenAI-compatible HTTP API на порту 8766. Превращает подписку ChatGPT в локальный API endpoint для Codex CLI; член совета 'codex' в mcp-council. |
| [`bench/`](bench/) | benchmark | Раннер LLM-бенчей: `run.py` (вызовы моделей с замером TTFT/quality), `judge.py` (LLM-as-judge), `report.py` (генерация markdown-отчёта). |

Раньше существовали отдельные пакеты `mcp-deepseek/` и `mcp-minimax/`. Их
функционал слит в `mcp-council/model_ask` через расширение `models.CATALOG`;
каталоги удалены.

## Shared lib политика

Python-пакетов теперь четыре (`mcp-council`, `claude-agent-server`,
`codex-agent-server`, `bench`), но общая lib намеренно не выделяется:

- `mcp-council` самодостаточен (`sandbox.py` / `logger.py` / `openai_client.py`
  живут внутри пакета).
- Оба agent-сервера — независимые standalone-пакеты, каждый запускается из своего
  каталога как отдельный boot-таск. Несколько хелперов (`_load_dotenv`,
  `_child_env_without_secrets`, `build_tools_system_prompt`, `parse_tool_calls`,
  `extract_content`, `_send`/`_send_stream`) держатся **byte-identical** в обеих
  копиях — правку вносить сразу в обе (об этом же напоминает `NOTE:`-комментарий
  над ними). Выделять ради этого shared lib значило бы связать deploy двух
  независимых серверов из-за ~200 строк — не оправдано.

## Ключи

Все ключи передаются через переменные окружения (см. README каждого пакета —
для `mcp-council` это `~/.claude.json` → `mcpServers.council.env`). Реальные
значения в репозиторий не попадают.

## Лицензия

MIT — см. [LICENSE](LICENSE).
