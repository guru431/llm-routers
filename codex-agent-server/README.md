# Codex Agent Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![No deps](https://img.shields.io/badge/dependencies-stdlib%20only-brightgreen.svg)](#)

HTTP-прокси для **Codex CLI** (OpenAI, подписка ChatGPT) с OpenAI-compatible API. Аналог [`claude-agent-server`](../claude-agent-server/), но поверх `codex exec`. Один API — два мира потребителей:

- **агентный** (`workspace-write`) — Codex реально правит файлы / запускает shell (локальные агентные клиенты);
- **read-only** — чистая генерация текста (mcp-council, claude-code-router, code-review).

Один файл `server.py`, Python 3.10+ stdlib, без зависимостей.

## Зачем

Подписка ChatGPT (Codex) — фиксированная цена. Сервер открывает доступ к Codex-моделям через локальный OpenAI-endpoint для любых проектов: агенты, n8n, чат-боты, советы моделей. По дефолту ведёт себя как обычная chat-модель; агентность включается осознанно.

## Требования

- Python 3.10+ (stdlib only)
- Codex CLI установлен и авторизован (`codex --version`, `codex login`)
- Активная подписка ChatGPT с доступом к Codex

## Установка

```bash
cd codex-agent-server
export CODEX_AGENT_TOKEN='cas-<random hex>'   # обязателен
python server.py
```

Сервер стартует на `127.0.0.1:8766`. Проверка:

```bash
curl http://localhost:8766/health
```

### Опции запуска

```bash
python server.py --port 9000        # другой порт
python server.py --host 0.0.0.0     # открыть на LAN (токен обязателен)
```

### Автозапуск на Windows (Task Scheduler)

```powershell
.\install_task.ps1
```

Создаёт задачу `\codex_agent_server` с запуском при старте системы (через `pythonw.exe`). Как и `claude_agent_server`, задача ведётся отдельно от общего реестра задач.

## Режимы и переключение

Режим sandbox разрешается по приоритету (первое сработавшее побеждает):

| # | Условие | Режим |
|---|---|---|
| 1 | в запросе есть `tools` | `read-only` (форс) |
| 2 | явное поле `sandbox` в body | его значение |
| 3 | суффикс `-agent` в имени модели | `workspace-write` |
| 4 | env `CODEX_AGENT_DEFAULT_SANDBOX` | дефолт (`read-only`) |

Это даёт два независимых способа выбрать режим:

- **именем модели** — для клиентов, умеющих задавать только строку (CCR, агентные клиенты с `models.json`): `gpt-5.5` (read-only) vs `gpt-5.5-agent` (workspace-write);
- **полем `sandbox` в body** — для клиентов, умеющих добавлять поля (mcp-council через `extra_payload`).

`tools` всегда форсят `read-only`: клиентские OpenAI-функции несовместимы с агентным режимом, где Codex дёргает собственные инструменты.

## Endpoints

### `POST /v1/chat/completions` — OpenAI-compatible

```bash
curl -X POST http://localhost:8766/v1/chat/completions \
  -H "Authorization: Bearer $CODEX_AGENT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "2+2=?"}]
  }'
```

Параметры body:
- `messages` (required) — массив `{role, content}` (роли `system`/`user`/`assistant`/`tool`).
- `model` (optional) — `gpt-5.5` или `gpt-5.5-agent` (см. `/v1/models`).
- `sandbox` (optional) — `read-only` | `workspace-write` (приоритет 2).
- `workdir` / `cwd` (optional) — рабочий корень для агентного режима (внутри `CODEX_AGENT_WORKDIR_ROOT`).
- `tools` (optional) — OpenAI-функции; форсят read-only, парсятся из `<tool_call>` блоков.
- `timeout` (optional) — секунды, клампится в `[10, 600]` (default 300).
- `reasoning` (optional) — `minimal|low|medium|high`; per-request override `model_reasoning_effort`. Default = env `CODEX_AGENT_REASONING` (medium). Невалидное → дефолт. Полезно токен-чувствительным потребителям (code-review шлёт `low`) без смены глобального дефолта для council/CCR.

### `GET /v1/models`

Список моделей (каждая база + `-agent` вариант), `owned_by: "openai"`.

### `GET /health`

```bash
curl http://localhost:8766/health
# {"status":"ok","model":"gpt-5.5","default_sandbox":"read-only","uptime":3600,"security":"authenticated"}
```

## Агентный режим (workspace-write)

```bash
# Codex создаст/изменит файлы в CODEX_AGENT_WORKDIR
curl -X POST http://localhost:8766/v1/chat/completions \
  -H "Authorization: Bearer $CODEX_AGENT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5-agent",
    "messages": [{"role":"user","content":"Создай README в корне проекта"}],
    "workdir": "C:/codex-workspace/myrepo"
  }'
```

`workdir` обязан лежать внутри `CODEX_AGENT_WORKDIR_ROOT` (дефолт = `CODEX_AGENT_WORKDIR`), иначе `400`. Если `workdir` не передан — берётся `CODEX_AGENT_WORKDIR`.

## Конфигурация (env)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `CODEX_AGENT_MODEL` | `gpt-5.5` | модель по умолчанию |
| `CODEX_AGENT_MODELS` | `gpt-5.5` | базовые id для whitelist (через запятую) |
| `CODEX_AGENT_DEFAULT_SANDBOX` | `read-only` | дефолт режима |
| `CODEX_AGENT_PORT` | `8766` | порт |
| `CODEX_AGENT_HOST` | `127.0.0.1` | bind-адрес |
| `CODEX_AGENT_TOKEN` | _(обязателен)_ | bearer-токен; без него exit 2 |
| `CODEX_AGENT_WORKDIR` | _(для агентного)_ | корень работы агента |
| `CODEX_AGENT_WORKDIR_ROOT` | = `WORKDIR` | разрешённый корень для override |
| `CODEX_AGENT_REASONING` | `medium` | `model_reasoning_effort` |
| `CODEX_AGENT_MAX_BODY` | `10485760` (10 MB) | макс. размер тела запроса; больше → `413` |
| `CODEX_AGENT_MAX_CONCURRENCY` | `4` | макс. параллельных codex-вызовов; сверх → `429` |

## Использование как провайдера

| Потребитель | Как выбрать режим |
|---|---|
| Агентный клиент | модель `gpt-5.5-agent` + `workdir` |
| Чат-клиент | модель `gpt-5.5` |
| mcp-council | `extra: {"sandbox": "read-only"}` в `models.py::CATALOG` |
| claude-code-router | provider config; `tools` форсят read-only |

### Python (openai SDK)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8766/v1", api_key="cas-...")
resp = client.chat.completions.create(model="gpt-5.5", messages=[{"role":"user","content":"Привет"}])
print(resp.choices[0].message.content)
```

## Tool calling

В read-only режиме tool calling эмулируется через prompt-injection (как в `claude-agent-server`): описания функций инжектируются в system-промпт, модель возвращает `<tool_call>{...}</tool_call>`, парсер конвертирует в `tool_calls`. Точность ниже native tool use. В агентном режиме Codex использует **свои** нативные инструменты — клиентские `tools` туда не передаются.

## Безопасность

- Bind по умолчанию `127.0.0.1`. Для LAN (`--host 0.0.0.0`) — bearer-токен обязателен (и так требуется).
- `CODEX_AGENT_TOKEN` **обязателен** — без него сервер не стартует (exit 2). Все endpoint кроме `/health` требуют `Authorization: Bearer`.
- **Агентный режим = HTTP-запрос может писать файлы и запускать shell.** Защита: обязательный токен, containment-проверка `workdir` внутри `CODEX_AGENT_WORKDIR_ROOT`, безопасный дефолт `read-only`.
- **Реальный write-containment делегирован codex**, а не `-C`/realpath-проверке (та лишь выбирает cwd и отсекает запросы вне корня). На каждый агентный вызов сервер пиннит `-c sandbox_workspace_write.writable_roots=[<workdir>]`, чтобы границу записи enforce'ил сам `codex --sandbox workspace-write`, а не дефолтное поведение cwd.
- Глобальные MCP-серверы Codex отключаются на каждом вызове (`-c mcp_servers={}`), чтобы сервис не триггерил сторонние интеграции.

## Тесты

Это live-интеграционный сьют (CLI, не pytest) — бьёт по запущенному серверу на :8766:

```bash
python integration_suite.py --token $CODEX_AGENT_TOKEN              # text/tool/system/multi-turn
python integration_suite.py --token $CODEX_AGENT_TOKEN --agentic    # + workspace-write (медленно)
python integration_suite.py --token $CODEX_AGENT_TOKEN --cat TextGen
```

Агентные тесты требуют writable `CODEX_AGENT_WORKDIR` на хосте сервера и живой `codex login`.

## Архитектура

Весь сервер — один файл `server.py`:

- `HTTPServer` + `BaseHTTPRequestHandler` принимают запросы.
- `resolve_model()` снимает суффикс `-agent`, `resolve_sandbox()` применяет приоритет режима, `resolve_workdir()` делает containment-проверку.
- `run_codex()` зовёт `codex exec - -m <model> --sandbox <mode> --skip-git-repo-check -o <tmpfile>` через `subprocess.run` (промпт через stdin — обход Windows-лимита cmdline). Финальный ответ читается из tmp-файла `-o`, а не из JSONL.
- `CREATE_NO_WINDOW` на Windows подавляет вспышки консоли от `codex.cmd` shim.
- System-промпт складывается в текст (у Codex нет `--system-prompt`).

`usage` — приблизительный (`len(text)//4`); Codex не отдаёт реальные счётчики токенов в `-o`. Каждый вызов несёт базовый overhead системного промпта Codex (~25-27k токенов) — это съедает подписку ChatGPT.

## License

MIT — см. [LICENSE](LICENSE).
