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

Сервер запустится на `0.0.0.0:8765`. Проверка:

```bash
curl http://localhost:8765/health
```

### Опции запуска

```bash
python server.py --port 9000              # другой порт
python server.py --host 127.0.0.1         # bind только на localhost
```

### Автозапуск на Windows (Task Scheduler)

```powershell
.\install_task.ps1
```

Создаёт задачу `\claude_agent_server` с запуском при старте системы (через `pythonw.exe`, без консольного окна).

## Endpoints

### `POST /v1/chat/completions` — OpenAI-compatible

Drop-in замена для любого клиента, который умеет OpenAI API.

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

### `GET /v1/models`

Список моделей в OpenAI-формате. Используется Open WebUI для селектора.

```bash
curl http://localhost:8765/v1/models
```

### `GET /health`

```bash
curl http://localhost:8765/health
# {"status": "ok", "model": "claude-opus-4-8", "uptime": 3600}
```

## Конфигурация

Через переменные окружения:

| Переменная | По умолчанию | Описание |
|---|---|---|
| `CLAUDE_AGENT_MODEL` | `claude-opus-4-8` | Модель по умолчанию |
| `CLAUDE_AGENT_PORT` | `8765` | Порт сервера |

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

## Безопасность

Сервер **не имеет аутентификации**. Биндится по умолчанию на `0.0.0.0` — доступен всем в сети. Рекомендации:

- В open Internet — не выставлять. Только LAN или за firewall.
- Для локального использования — `--host 127.0.0.1`.
- Если нужна аутентификация — поставить reverse proxy (nginx/caddy) с basic auth или token-проверкой.

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
