# mcp-council

MCP-сервер: Karpathy 3-stage council из 7 LLM с опциональной auto-synthesis, multi-round debate, confidence-weighted aggregation и per-model web search через Exa.ai.

## Workflow

1. **Stage 1: Independent** — 7 моделей независимо отвечают на вопрос (parallel).
2. **Stage 2: Anonymized peer-ranking** — каждая модель оценивает чужие ответы под псевдонимами Member A/B/C… (parallel, self-ranking исключён, per-ranker random shuffle псевдонимов для устранения positional bias). Каждый ranker самооценивает уверенность (1-10), которая используется как вес в aggregate.
3. **Stage 3: Synthesis** — опционально (`synthesis=true`); по умолчанию синтез делает основной Claude-агент в сессии. Если включён — chairman (модель с наивысшим weighted rank, fallback DeepSeek) сам пишет финальный ответ внутри MCP.

## Council members (7)

| id | model | provider | env_key |
|---|---|---|---|
| glm | glm-5.2 | OpenCode Go | `OPENCODE_GO_KEY` |
| kimi | kimi-k2.7-code | OpenCode Go | `OPENCODE_GO_KEY` |
| deepseek-pro | deepseek-v4-pro | OpenCode Go | `OPENCODE_GO_KEY` |
| qwen | qwen3.6-plus | OpenCode Go | `OPENCODE_GO_KEY` |
| minimax | minimax-m3 | OpenCode Go | `OPENCODE_GO_KEY` |
| gemini | gemini-3.1-pro-preview | Helicone AI Gateway | `HELICONE_GATEWAY_KEY` |
| codex | gpt-5.5 | codex-agent-server :8766 (read-only) | `CODEX_AGENT_TOKEN` |

`deepseek-pro` теперь идёт через OCG-прокси (DeepSeek direct PAYG исчерпан с 2026-06-07): при OCG outage живым голосом остаётся только Gemini. См. также `_pick_chairman` — DeepSeek используется как fallback chairman.

## Tools

### `council_ask` — синхронный (блокирующий, 2-8 мин)

```python
council_ask(
    question: str,
    context_paths: list[str] | None = None,
    max_response_tokens: int = 8192,
    synthesis: bool = False,    # True → MCP сам делает stage 3 синтез
    rounds: int = 1,             # 2-3 → multi-round debate
    web_search: bool = False,    # True → каждая модель в stage 1 имеет web_search tool через Exa.ai
) -> str
```

Возвращает один markdown-документ с stage1+stage2+stage3 (опц.) + aggregate.

### `council_ask_async` — неблокирующий, через job_id

```python
council_ask_async(question, context_paths=None, max_response_tokens=8192,
                  synthesis=False, rounds=1, web_search=False) -> {"job_id": "job-…", ...}
```

Возвращает за ~50 мс. Затем:

- `council_status(job_id)` — текущая фаза, per-member progress, elapsed.
- `council_result(job_id)` — финальный markdown, когда `phase=="done"`.
- `council_cancel(job_id)` — отмена.
- `council_list_jobs(limit=20)` — список последних job'ов.

Используй когда вызывающий агент (Claude в сессии) хочет продолжать отвечать пользователю пока council работает.

### Real-time event stream (Monitor-friendly)

Каждый `council_ask_async` создаёт `logs/events/<job_id>.jsonl` (один JSON-event на строку, line-buffered). `council_ask_async` возвращает путь в поле `event_log`. Внешний наблюдатель — например Claude в основной сессии с tool `Monitor` — может `tail -F <event_log>` и реагировать на события в реальном времени без polling'а `council_status`.

Event types:
- `phase` — `{"phase": queued|stage1|stage2|stage3|done|error|cancelled, ...}`
- `stage1_member` — `{"id": "...", "model": "...", "status": "ok"|"error", "latency_ms": int, "tool_calls_count": int}`
- `stage2_ranker` — то же для ranker'ов
- `stage3` — то же для chairman synthesis
- `tool_call` — `{"member_id": ..., "name": "web_search", "query": str, "status": "ok"|"error", "num_results": int|None}` — испускается каждый раз когда модель сделала web_search во время своего stage 1
- `result_ready` — `{"status": "ok"|"error"|"cancelled", "dump_path": str|None}` — финальный event, файл затем закрывается

### Web search per-model (`web_search=True`)

Когда включено: каждая stage-1 модель получает OpenAI-style tool `web_search(query)` через Exa.ai. Модели **независимо** формулируют свои queries (probe показал что 6 моделей выдают 6 разных формулировок — от простых до boolean syntax типа `"A" OR "B"`), исполнитель в MCP дёргает Exa, отдаёт title/url/summary/highlights, модель может вызвать ещё раз или сразу написать финальный ответ. Max 5 iterations на модель (защита от зацикливания).

Stage 2 (peer-ranking) и Stage 3 (chairman synthesis) — **без** tools, они работают с собранными в stage 1 материалами.

Trade-off: каждая модель тратит +30-90s на 1-3 search iterations. Стоимость Exa ~$0.005-0.01/query × 5-15 queries за council = $0.05-0.15.

## Когда применять

Архитектурное решение, спорный технический вопрос, важный code review, разбор сложного бага. **НЕ** используй для рутины (быстрых вопросов, шаблонной генерации) — дорого и медленно.

Дополнительно:
- `rounds=2` примерно удваивает время и токены — оправдано когда answers сильно расходятся в round 1, и хочется чтобы модели увидели критику и улучшили ответы.
- `synthesis=true` экономит контекст в основной сессии за счёт ещё одного API-вызова к chairman'у; теряется преимущество "chairman знает весь разговор".

## Install

```bash
cd llm_routers/mcp-council
pip install -e ".[dev]"
```

## Run tests

```bash
pytest -v
```

## Run server (stdio)

```bash
OPENCODE_GO_KEY=<...> HELICONE_GATEWAY_KEY=<...> CODEX_AGENT_TOKEN=<...> EXA_API_KEY=<...> python server.py
```

`EXA_API_KEY` обязательный только если кто-то вызывает с `web_search=True`. Все 4 ключа задаются через переменные окружения (в Claude Code — через `~/.claude.json` → `mcpServers.council.env`).

## Design

Async-job исполнение council живёт в `state.py`; промпты стадий — в `prompts.py` (`STAGE3_SYSTEM` / `STAGE1_ROUND_N_SYSTEM`); выбор chairman — `_pick_chairman` в `council.py`.

## Sandbox

Список заблокированных путей — в `sandbox.py` (тот же blacklist, что в `mcp-deepseek/sandbox.py`; дубликат by design).

## Logging

- JSONL события per-call: `logs/council_YYYY-MM-DD.log` (метаданные).
- Полный дамп per-call: `logs/calls/<timestamp>-<hash>.json` (question + stage1 ответы + stage2 рейтинги + stage3 synthesis + errors + latency). Используется для анализа качества council.

## HTTP behaviour

- `DEFAULT_TIMEOUT = 600s` (thinking-модели через OCG могут долго думать без emitting bytes).
- Retry on HTTP 429/500/502/503/529: 2 попытки, backoff (15s, 45s).
- HTTP 402 (insufficient balance) — без retry, сразу error.
- Pустой `str(exception)` в httpx ошибках заменяется на `type(e).__name__` (`ReadTimeout`, `ConnectTimeout`, …).
