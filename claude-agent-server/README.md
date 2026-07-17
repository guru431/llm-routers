# Claude Agent Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![No deps](https://img.shields.io/badge/dependencies-stdlib%20only-brightgreen.svg)](#)

HTTP-прокси для Claude Code CLI с OpenAI-compatible API. Превращает локально установленный `claude` CLI (через подписку Claude Max/Pro) в API-сервер, к которому можно подключать любые проекты — OpenAI SDK, Open WebUI, n8n, чат-боты и т.д.

Один файл `server.py`, Python 3.10+ stdlib, без зависимостей.

## Зачем

Подписка Claude Max/Pro — фиксированная цена. Этот сервер открывает доступ к Opus/Sonnet/Haiku через HTTP API на любом порту/хосте. Вместо платы за каждый токен через Anthropic API — Claude становится «локальной» моделью для всех проектов в сети.

## Требования

- Python 3.10+ (stdlib only)
- Claude Code CLI установлен и авторизован (`claude --version` должно работать)
- Активная подписка Claude Max или Pro

## Установка

```bash
git clone https://github.com/guru431/claude-agent-server.git
cd claude-agent-server
python server.py
```

Сервер запустится на `127.0.0.1:8765` (только loopback). Проверка:

```bash
curl http://localhost:8765/health
```

### Опции запуска

```bash
python server.py --port 9000              # другой порт
python server.py --host 0.0.0.0           # открыть на LAN (по умолчанию loopback)
```

### Автозапуск на Windows (Task Scheduler)

```powershell
.\install_task.ps1
```

Создаёт задачу `\claude_agent_server` с запуском при старте системы (через `pythonw.exe`, без консольного окна).

## Endpoints

### `POST /v1/chat/completions` — OpenAI-compatible

Совместим с клиентами OpenAI Chat Completions в пределах поддерживаемого subset'а
(см. [Совместимость OpenAI API](#совместимость-openai-api) и [Tool calling](#tool-calling)) —
не полная реализация API.

```bash
curl -X POST http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "Ты переводчик."},
      {"role": "user", "content": "Переведи: Hello world"}
    ]
  }'
```

Ответ:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "claude-opus-4-8",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Привет, мир"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
}
```

Параметры в body:
- `messages` (required) — массив `{role, content}`. Роли: `system`, `user`, `assistant`, `tool`
- `model` (optional) — `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` (см. `/v1/models`)
- `tools` (optional) — массив определений функций в OpenAI-формате; вызовы парсятся из `<tool_call>` блоков ответа
- `timeout` (optional) — таймаут в секундах (default 300)
- `stream` (optional) — если `true`, ответ отдаётся как OpenAI SSE (`text/event-stream`, чанки `chat.completion.chunk` + `data: [DONE]`). Псевдо-стрим: CLI возвращает ответ целиком, сервер режет его на чанки для совместимости (Open WebUI)

### `GET /v1/models`

Список моделей в OpenAI-формате. Используется Open WebUI для селектора.

```bash
curl http://localhost:8765/v1/models
```

### `GET /health`

```bash
curl http://localhost:8765/health
# без токена: {"status": "ok"}
# с токеном:  {"status": "ok", "model": "claude-opus-4-8", "default_profile": "chat", "profiles": [...], "uptime": 3600, "security": "authenticated", "cache": {...}}
```

### `GET /ready` — readiness probe

Отличается от `/health` (liveness): проверяет, что сервер реально готов обслуживать. Без токена (только булевы флаги, без путей). `200 {ready:true}` либо `503 {ready:false, checks:{...}}`.

```bash
curl http://localhost:8765/ready
# {"ready": true, "checks": {"auth_token_configured": true, "cli_found": true, "not_overloaded": true}}
```

### `GET /metrics` — счётчики (JSON, не prometheus)

Требует bearer. `total_requests`, `active` (in-flight), `rejected_overload`, `timeouts`, `killed_processes`, `cache_hits`/`cache_misses`, `uptime`, latency `median`/`p90` (ring buffer последних N).

```bash
curl -H "Authorization: Bearer $CLAUDE_AGENT_TOKEN" http://localhost:8765/metrics
```

## Конфигурация

Через переменные окружения:

| Переменная | По умолчанию | Описание |
|---|---|---|
| `CLAUDE_AGENT_MODEL` | `claude-opus-4-8` | Модель по умолчанию |
| `CLAUDE_AGENT_PORT` | `8765` | Порт сервера |
| `CLAUDE_AGENT_TOKEN` | _(обязателен)_ | Bearer-токен. Сервер **не стартует** без него (exit 2). Обязателен в `Authorization: Bearer <token>` на всех endpoints кроме `/health`. |
| `CLAUDE_AGENT_CACHE` | `1` | Включить response cache (`0`/`false` — выключить) |
| `CLAUDE_AGENT_CACHE_SIZE` | `256` | Макс. записей в кэше (LRU eviction) |
| `CLAUDE_AGENT_CACHE_TTL` | `3600` | TTL записи в секундах |
| `CLAUDE_AGENT_CACHE_BYTES` | `67108864` (64 MB) | Макс. суммарный размер значений кэша; больше → LRU eviction |
| `CLAUDE_AGENT_MAX_BODY` | `10485760` (10 MB) | Макс. размер тела запроса; больше → `413` |
| `CLAUDE_AGENT_MAX_CONCURRENCY` | `4` | Макс. параллельных claude-вызовов; сверх → bounded queue |
| `CLAUDE_AGENT_QUEUE_WAIT` | `5` | Сек. ожидания свободного слота в bounded queue перед `429` (Idea 13) |
| `CLAUDE_AGENT_MAX_QUEUE` | `2×concurrency` | Макс. ожидающих в очереди; переполнение → сразу `429 + Retry-After` |
| `CLAUDE_AGENT_MAX_SYSTEM_PROMPT` | `7000` | Макс. длина system-prompt (символов). Он идёт в `--system-prompt=` argv; на Windows больше ~8191 символов cmdline упирается в лимит cmd.exe. Больше → `400`. Поднять, если подтверждён больший реальный лимит. |

## Профили, structured output, стриминг и наблюдаемость

### Capability profiles (`profile`)

Явные именованные «позы» запроса поверх дефолтного chat-режима. Выбираются полем `profile` в body:

| profile | claude | Что означает |
|---|---|---|
| `chat` (дефолт) | текущее поведение | чистая генерация, без tools/session persistence |
| `research` | = chat + tool/web эмуляция | ярлык для единообразия с codex-agent-server; у claude нет реального FS-доступа, поэтому research ≡ chat + prompt-injection tools (не настоящий OS-доступ) |
| `agent` | **`400`** | claude не поддерживает workspace-write; для agent-профиля используйте `codex-agent-server` |

Активный профиль отражается в `usage.profile` ответа и в `/health` (`default_profile`, `profiles`). **Профиль — это ярлык, НЕ OS-песочница** (см. [Безопасность](#безопасность)).

### Structured output (`response_format`)

Поддержаны `{"type":"json_object"}` и `{"type":"json_schema","json_schema":{...}}`. Требование вернуть валидный JSON (и сама схема при json_schema) инжектируется в system-prompt; ответ валидируется (парсится как JSON; при json_schema — проверяются required-поля верхнего уровня и вложенных object-required, без внешних libs); при провале — **один** repair-retry с сообщением об ошибке. В ответе `usage.structured_output: true`. Tool-эмуляция теперь передаёт **полную** JSON Schema функции (nested objects/items/enum) в промпт, а после парсинга `<tool_call>` валидирует required-аргументы (тоже один repair-retry).

### Настоящий стриминг + cancellation (`stream: true`)

Для **чистого текста** (без tools и без response_format) сервер читает `claude --output-format stream-json` построчно и отдаёт реальные инкрементальные OpenAI SSE-deltas по мере поступления, с реальными `usage`/`stop_reason` (`estimate:false`). Если клиент отключается — запись в сокет падает, и дерево процессов `claude` немедленно убивается (client-disconnect → cancellation). Если формат stream-json не распознан — прозрачный fallback на буферный режим. Для `tools`/`response_format` стрим остаётся псевдо-стримом (нужно буферизовать весь ответ для парсинга/валидации), `usage` — оценка `len//4` (`estimate:true`).

### Readiness, metrics, bounded queue

`/ready` и `/metrics` — см. [Endpoints](#endpoints). При перегрузке вместо мгновенного `429` запрос ждёт свободный слот в bounded queue (до `CLAUDE_AGENT_QUEUE_WAIT` сек); не дождался или очередь переполнена → `429` с заголовком `Retry-After`.

## Использование

### Python (openai SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8765/v1", api_key="unused")
resp = client.chat.completions.create(
    model="claude-opus-4-8",
    messages=[{"role": "user", "content": "Привет"}],
)
print(resp.choices[0].message.content)
```

### Python (stdlib)

```python
import json, urllib.request

req = urllib.request.Request(
    "http://localhost:8765/v1/chat/completions",
    data=json.dumps({"messages": [{"role": "user", "content": "2+2=?"}]}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as r:
    print(json.loads(r.read())["choices"][0]["message"]["content"])
```

### curl

```bash
curl -s http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "2+2=?"}]}' \
  | jq -r '.choices[0].message.content'
```

## Tool calling

Tool calling эмулируется через prompt injection: описания функций инжектируются в system-prompt, модель возвращает `<tool_call>{...}</tool_call>`, парсер конвертирует в OpenAI-формат `tool_calls`.

**Ограничение:** это не настоящий native tool use Anthropic API — точность ниже, чем у прямого вызова `claude` CLI с MCP-серверами. На штатных бенчмарках tool-calling работает примерно в 7 случаях из 12 (см. `test_server.py`).

**Схема функций.** В system-prompt инжектируется плоское описание (`name: type [required] —
description`) **плюс полная JSON Schema** параметров (`Full JSON Schema: {...}`) — вложенные
`object`/`array`, `items`, `enum`, `oneOf`/`anyOf` теперь доходят до модели дословно. После
парсинга `<tool_call>` наличие required-полей валидируется; при их отсутствии — один
repair-retry. `content` сообщений поддерживается как строка или массив `{"type":"text"}`
частей (image/audio-части игнорируются).

```python
client.chat.completions.create(
    model="claude-opus-4-8",
    messages=[{"role": "user", "content": "Какая погода в Москве?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }],
)
```

## Совместимость OpenAI API

Сервер — drop-in для клиентов OpenAI, но `claude -p` CLI не отдаёт часть knob'ов. Что учитывается и что игнорируется:

| Поле | Поведение |
|---|---|
| `messages` | учитывается (роли system/user/assistant/tool) |
| `model` | учитывается (whitelist; неизвестная → `400`) |
| `stream` | учитывается — **настоящий** стрим для чистого текста (построчный `--output-format stream-json`, реальные deltas + реальный usage `estimate:false` + cancellation при disconnect); для `tools`/`response_format` — псевдо-стрим (буфер → SSE-чанки); `tool_calls` идут индексированными delta по OpenAI-спеке |
| `tools` | эмулируется через prompt-injection (не native tool use); полная JSON Schema в промпте + валидация required-аргументов + один repair-retry |
| `response_format` | учитывается — `json_object` и `json_schema` (инъекция + валидация + один repair-retry; `usage.structured_output:true`). См. [Structured output](#structured-output-response_format) |
| `profile` | учитывается — `chat`\|`research` (`agent` → `400`). См. [Profiles](#capability-profiles-profile) |
| `timeout` | учитывается, зажимается в `[10, 600]` секунд |
| `tool_choice` | **игнорируется** — CLI не умеет форсить/запрещать конкретный вызов; модель решает сама |
| `temperature`, `top_p`, `max_tokens`, `n`, `stop` | **игнорируются** — у `claude -p` нет соответствующих ключей |

`usage` — приблизительная оценка токенов (`len // 4`, `estimate:true`) в буферном режиме; в настоящем стриме чистого текста — **реальные** счётчики из stream-json (`estimate:false`). Поле `usage.profile` показывает активный профиль.

## Безопасность

Биндится по умолчанию на `127.0.0.1` (только loopback). Для доступа из LAN задать `--host 0.0.0.0` (или `CLAUDE_AGENT_HOST=0.0.0.0`, или через install_task). Bearer-токен через `CLAUDE_AGENT_TOKEN` **обязателен** — без него сервер не стартует (exit 2):

```bash
export CLAUDE_AGENT_TOKEN='cas-<random hex>'
python server.py
```

Все endpoints кроме `/health` требуют `Authorization: Bearer <token>` — иначе 401. `/health` работает без токена (liveness-проба) и отдаёт только `{"status": "ok"}`; полные поля (`model`, `uptime`, `security`, `cache`) — лишь при валидном bearer.

Клиент с токеном:

```bash
curl -X POST http://host:8765/v1/chat/completions \
  -H "Authorization: Bearer $CLAUDE_AGENT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hi"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://host:8765/v1", api_key="sk-local-<random>")
```

**Транспорт — только loopback либо TLS-прокси.** Сервер говорит по plain HTTP.
Единственный встроенно-поддерживаемый режим — bind на `127.0.0.1` (default). Bearer
поверх незашифрованного HTTP на LAN недостаточен: токен уходит по сети в открытом
виде, его можно перехватить/переиграть. Если нужен доступ из LAN — не биндить
`0.0.0.0` напрямую, а поставить перед сервером **TLS/mTLS reverse proxy** (nginx/caddy)
или пускать трафик через **VPN**, оставив сам сервер на `127.0.0.1` за прокси. При
bind не на loopback сервер печатает предупреждение на старте.

**Профиль «чат» НЕ изолирован от файловой системы хоста.** `claude` запускается тем
же OS-пользователем, что и сервер. Чтобы урезать поверхность, каждый вызов идёт с
`--tools ""` (все встроенные инструменты Claude отключены — сервер эмулирует
tool-calling через prompt-injection и никогда не даёт Claude реально исполнять
Bash/Edit/Read), `--strict-mcp-config` (пользовательские MCP-серверы не загружаются)
и `--no-session-persistence` (сессии не пишутся на диск). Это снижает host-action
surface, но **не является песочницей**: истинная изоляция (отдельный low-privilege
OS-user / контейнер с allowlisted mount) — на усмотрение оператора развёртывания.

Рекомендации:
- В open Internet — не выставлять.
- В LAN — только за TLS/mTLS reverse proxy или через VPN; сам сервер на `127.0.0.1`.
- Для локальной разработки — `--host 127.0.0.1` (тогда токен не нужен).

## Тесты

```bash
python test_server.py                          # все тесты на localhost:8765
python test_server.py --url http://host:8765   # другой адрес
python test_server.py --cat ToolCall           # только категория ToolCall
```

12 тестов: tool calling, генерация текста, system-prompt adherence, multi-turn.

## Архитектура

Весь сервер — один файл `server.py` (~370 строк):

- `HTTPServer` + `BaseHTTPRequestHandler` принимают запросы
- `run_claude()` вызывает `claude -p -` через `subprocess.run` (промпт идёт через stdin — обход Windows-лимита cmdline ~32K)
- `CREATE_NO_WINDOW` на Windows подавляет вспышки консольных окон от `claude.cmd` shim
- Ответ парсится из JSON-output Claude CLI и возвращается в OpenAI-формате
- Multi-turn собирается простой конкатенацией `User: ...\n\nAssistant: ...`
- Tool calling: см. `build_tools_system_prompt()` и `parse_tool_calls()`

`usage` в ответе — приблизительные токены (`len(text) // 4`), не реальные значения от Anthropic.

## License

MIT — см. [LICENSE](LICENSE).
