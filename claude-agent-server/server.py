"""
Claude Agent Server — универсальный HTTP-прокси для Claude CLI.

Endpoints:
    POST /v1/chat/completions  — OpenAI-compatible (messages + tools)
    GET  /v1/models            — Model list (OpenAI-compatible)
    GET  /health               — Healthcheck (включает cache stats, security mode)
    DELETE /cache              — Очистить response cache

Env:
    CLAUDE_AGENT_MODEL      — модель (default: claude-opus-4-8)
    CLAUDE_AGENT_PORT       — порт (default: 8765)
    CLAUDE_AGENT_TOKEN      — bearer-токен (ОБЯЗАТЕЛЕН — без него сервер
                              не стартует). Требуется на всех endpoints
                              кроме /health (Authorization: Bearer ...)
    CLAUDE_AGENT_CACHE      — '1'/'0' включить response cache (default: '1')
    CLAUDE_AGENT_CACHE_SIZE — макс. записей в кэше (default: 256, LRU eviction)
    CLAUDE_AGENT_CACHE_TTL  — TTL записи в секундах (default: 3600 = 1h)
    CLAUDE_AGENT_MAX_BODY   — макс. размер тела запроса в байтах (default: 10 MB; >лимит → 413)
    CLAUDE_AGENT_MAX_CONCURRENCY — макс. параллельных claude-вызовов (default: 4; сверх → 429)

Caching:
    Сервер кэширует ответы по ключу (model, system_prompt, prompt). Запросы с
    `tools` и с `cache: false` в payload НЕ кэшируются. Cache hit возвращает
    ответ мгновенно (CLI ~5-30s) и помечает `cached: true` в usage.
"""

import argparse
import hmac
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cache import ResponseCache

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("claude-agent-server")

MODEL = os.getenv("CLAUDE_AGENT_MODEL", "claude-opus-4-8")
MODELS = [
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

# Suppress console windows on Windows when calling claude CLI (.cmd shim)
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Resolve the claude binary once. On Windows the npm shim is `claude.CMD`;
# CreateProcess won't append PATHEXT, so `subprocess.run(["claude", ...])` fails
# with FileNotFoundError. shutil.which() respects PATHEXT and returns the full
# path subprocess can launch directly. Mirrors codex-agent-server's CODEX_BIN.
CLAUDE_BIN = shutil.which("claude") or "claude"


# ── Response cache ──────────────────────────────────────────────────────────

CACHE_ENABLED = os.getenv("CLAUDE_AGENT_CACHE", "1") not in ("0", "false", "False", "")
try:
    _CACHE_SIZE = max(1, int(os.getenv("CLAUDE_AGENT_CACHE_SIZE", "256")))
except ValueError:
    _CACHE_SIZE = 256
try:
    _CACHE_TTL = max(1.0, float(os.getenv("CLAUDE_AGENT_CACHE_TTL", "3600")))
except ValueError:
    _CACHE_TTL = 3600.0

CACHE = ResponseCache(max_size=_CACHE_SIZE, ttl_seconds=_CACHE_TTL) if CACHE_ENABLED else None

# Mandatory bearer auth. Server refuses to start without it; required on every
# endpoint except /health.
AUTH_TOKEN = os.getenv("CLAUDE_AGENT_TOKEN") or None

# Reject oversized request bodies before reading them into memory (DoS guard).
# Mirrors codex-agent-server's MAX_BODY_SIZE.
try:
    MAX_BODY_SIZE = max(1024, int(os.getenv("CLAUDE_AGENT_MAX_BODY", str(10 * 1024 * 1024))))
except ValueError:
    MAX_BODY_SIZE = 10 * 1024 * 1024

# Cap concurrent claude invocations. Each request spawns a heavy `claude` CLI
# subprocess (Opus on the Max plan); without a cap, many parallel authed
# requests exhaust threads/processes and burn the Max quota. Excess → 429.
# Mirrors codex-agent-server's MAX_CONCURRENCY.
try:
    MAX_CONCURRENCY = max(1, int(os.getenv("CLAUDE_AGENT_MAX_CONCURRENCY", "4")))
except ValueError:
    MAX_CONCURRENCY = 4
_CLAUDE_SEM = threading.BoundedSemaphore(MAX_CONCURRENCY)


# ============================================================
# Tool calling via prompt injection
# ============================================================

def build_tools_system_prompt(tools: list) -> str:
    """Build a system prompt section that describes available tools and
    forces the model to use structured JSON for tool calls."""
    lines = ["# Available Functions\n"]
    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        sig_parts = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "string")
            req = " [required]" if pname in required else ""
            pdesc = pinfo.get("description", "")
            sig_parts.append(f"  {pname}: {ptype}{req} — {pdesc}")
        lines.append(f"## {name}\n{desc}")
        if sig_parts:
            lines.append("Parameters:\n" + "\n".join(sig_parts))
        lines.append("")

    lines.append("""# TOOL CALLING RULES — READ CAREFULLY

You have access to the functions listed above. You MUST follow these rules:

1. When the user's request needs real-time data, external info, system state, file contents, or any action — you MUST call the appropriate function.
2. NEVER fabricate or guess data that should come from a function call. If the user asks about weather, disk space, file contents, search results — CALL THE FUNCTION.
3. To call a function, your ENTIRE response must be ONLY this JSON (no text before/after):

<tool_call>
{"name": "function_name", "arguments": {"param1": "value1"}}
</tool_call>

4. For multiple calls, use multiple <tool_call> blocks.
5. If the request does NOT need a function (general knowledge, opinions, text generation) — respond normally with text.
6. If in doubt whether to call a function — CALL IT. Never guess.""")

    return "\n".join(lines)


def parse_tool_calls(text: str) -> tuple[list[dict], str]:
    """Parse <tool_call>...</tool_call> blocks from response."""
    calls = []
    remaining = text

    pattern = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
    for match in pattern.finditer(text):
        try:
            data = json.loads(match.group(1))
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": data.get("name", ""),
                    "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False),
                }
            })
        except json.JSONDecodeError:
            pass

    if calls:
        remaining = pattern.sub("", text).strip()

    return calls, remaining


# ============================================================
# Claude CLI runner
# ============================================================

def run_claude(prompt: str, system_prompt: str | None = None,
               model: str | None = None, timeout: int = 300) -> str:
    """Call claude CLI and return result text."""
    m = model or MODEL
    if m not in MODELS:
        raise ValueError(f"model not in whitelist: {m!r}")
    cmd = [CLAUDE_BIN, "--model", m, "-p", "-", "--output-format", "json"]
    if system_prompt:
        # `--system-prompt=VALUE` (single argv with `=`) prevents argument
        # injection: even if VALUE starts with `--`, argparse binds it as
        # the value of --system-prompt rather than parsing it as a new flag.
        cmd.append(f"--system-prompt={system_prompt}")
    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        creationflags=CREATE_NO_WINDOW,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"claude exit code {result.returncode}")
    # Parse JSON output to extract result
    try:
        data = json.loads(result.stdout.strip())
        if data.get("is_error"):
            raise RuntimeError(data.get("result", "Unknown error"))
        return data.get("result", "").strip()
    except json.JSONDecodeError:
        return result.stdout.strip()


def extract_content(content) -> str:
    """Extract text from string or OpenAI content array."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            c.get("text", "") for c in content
            if isinstance(c, dict) and c.get("type") == "text"
        )
    return str(content) if content else ""


# ============================================================
# HTTP Handler
# ============================================================

class Handler(BaseHTTPRequestHandler):
    # Socket timeout (seconds), applied by StreamRequestHandler.setup() to the
    # whole connection. Guards against a lying/partial Content-Length that pins
    # a worker thread on a blocking rfile.read() forever. Only counts against
    # idle socket ops, so it won't interrupt a long in-flight claude call (no
    # socket I/O happens while the subprocess runs).
    timeout = 60

    def log_message(self, format, *args):
        logger.info("%s %s", self.address_string(), format % args)

    def _check_auth(self) -> bool:
        """Enforce bearer-auth if CLAUDE_AGENT_TOKEN is configured.
        Returns False after sending 401; caller must abort."""
        if not AUTH_TOKEN:
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            self._send(401, {"error": {"message": "missing bearer token", "type": "auth_error"}})
            return False
        presented = header[len("Bearer "):].strip()
        if not hmac.compare_digest(presented.encode("utf-8"), AUTH_TOKEN.encode("utf-8")):
            self._send(401, {"error": {"message": "invalid bearer token", "type": "auth_error"}})
            return False
        return True

    def do_GET(self):
        if self.path == "/health":
            payload = {
                "status": "ok",
                "model": MODEL,
                "uptime": int(time.time() - SERVER_START),
                "security": "authenticated" if AUTH_TOKEN else "unauthenticated",
            }
            if CACHE is not None:
                payload["cache"] = CACHE.stats()
            else:
                payload["cache"] = {"enabled": False}
            self._send(200, payload)
        elif self.path == "/v1/models":
            if not self._check_auth():
                return
            self._send(200, {
                "object": "list",
                "data": [{
                    "id": m,
                    "object": "model",
                    "created": int(SERVER_START),
                    "owned_by": "anthropic",
                } for m in MODELS],
            })
        else:
            self._send(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            if not self._check_auth():
                return
            body = self._read_body()
            if body is None:
                return
            self._handle_chat(body)
        else:
            self._send(404, {"error": "Not found"})

    def do_DELETE(self):
        if self.path == "/cache":
            if not self._check_auth():
                return
            if CACHE is None:
                self._send(404, {"error": "cache disabled"})
                return
            CACHE.clear()
            self._send(200, {"status": "cleared", "stats": CACHE.stats()})
        else:
            self._send(404, {"error": "Not found"})

    def _handle_chat(self, body: dict):
        """OpenAI-compatible chat completions with tool calling support."""
        messages = body.get("messages", [])
        if not messages:
            self._send(400, {"error": {"message": "messages is required", "type": "invalid_request_error"}})
            return

        model = body.get("model")
        # Clamp client-provided timeout to [10s, 600s] to prevent DoS via
        # `timeout: 0` (instant fail) or `timeout: 999999` (hung worker).
        try:
            timeout = int(body.get("timeout", 300))
        except (TypeError, ValueError):
            timeout = 300
        timeout = max(10, min(timeout, 600))
        tools = body.get("tools")

        # Separate system prompt from conversation
        system_parts = []
        conversation = []
        for msg in messages:
            role = msg.get("role", "user")
            content = extract_content(msg.get("content", ""))
            if role == "system":
                system_parts.append(content)
            elif role == "tool":
                # Carry tool_call_id so multi-turn loops can match each result
                # back to the assistant tool_call that produced it. Without
                # this the LLM has to guess pairing when >1 tool was called
                # in the same assistant turn.
                tool_name = msg.get("name", "function")
                tool_call_id = msg.get("tool_call_id")
                header = f"[Tool {tool_name}"
                if tool_call_id:
                    header += f" id={tool_call_id}"
                header += f"]: {content}"
                conversation.append(("tool", header))
            else:
                conversation.append((role, content))

        # Inject tool descriptions into system prompt
        if tools:
            system_parts.append(build_tools_system_prompt(tools))

        system_prompt = "\n\n".join(system_parts) if system_parts else None

        # Build conversation prompt
        if len(conversation) == 1:
            prompt = conversation[0][1]
        elif len(conversation) == 0:
            prompt = ""
        else:
            parts = []
            for role, content in conversation:
                if role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
                elif role == "tool":
                    parts.append(content)
            prompt = "\n\n".join(parts)

        # Cache lookup: skip если есть tools (нестабильные ответы) или явный bypass
        cache_bypass = body.get("cache") is False
        cache_eligible = CACHE is not None and not tools and not cache_bypass
        cached = None
        if cache_eligible:
            cached = CACHE.get(model or MODEL, system_prompt, prompt)

        logger.info("Chat: %d msgs (%d sys, %d conv), tools=%s, %d chars, model=%s, cache=%s",
                     len(messages), len(system_parts), len(conversation),
                     len(tools) if tools else 0, len(prompt), model or MODEL,
                     "hit" if cached is not None else ("miss" if cache_eligible else "skip"))

        try:
            if cached is not None:
                result = cached
            else:
                # Cap concurrent claude subprocesses: reject (429) rather than
                # pile up processes and burn the Max quota under parallel load.
                # Cache hits skip this — they don't spawn a subprocess.
                if not _CLAUDE_SEM.acquire(blocking=False):
                    self._send(429, {"error": {
                        "message": f"server busy: >{MAX_CONCURRENCY} concurrent claude requests",
                        "type": "rate_limit_error"}})
                    return
                try:
                    result = run_claude(prompt, system_prompt=system_prompt,
                                        model=model, timeout=timeout)
                finally:
                    _CLAUDE_SEM.release()
                if cache_eligible and result:
                    CACHE.put(model or MODEL, system_prompt, prompt, result)

            # Parse tool calls if tools were provided
            tool_calls = []
            content = result
            if tools and result:
                tool_calls, content = parse_tool_calls(result)

            # Build response
            resp_message = {"role": "assistant"}
            if tool_calls:
                resp_message["tool_calls"] = tool_calls
                resp_message["content"] = content if content else None
            else:
                resp_message["content"] = content

            self._send(200, {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model or MODEL,
                "choices": [{
                    "index": 0,
                    "message": resp_message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }],
                # Rough estimate (chars/4). Accurate only for ASCII English;
                # for ru/CJK 1 char ≈ 2-3 tokens, so these undercount badly.
                # claude CLI doesn't expose real token counts in -p output.
                "usage": {
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": len(result) // 4,
                    "total_tokens": (len(prompt) + len(result)) // 4,
                    "estimate": True,
                    "cached": cached is not None,
                },
            })
        except subprocess.TimeoutExpired:
            self._send(504, {"error": {"message": "claude timeout", "type": "timeout"}})
        except Exception as exc:
            logger.exception("claude error")
            self._send(500, {"error": {"message": str(exc), "type": "server_error"}})

    def _read_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY_SIZE:
            self._send(413, {"error": {
                "message": f"request body too large ({length} > {MAX_BODY_SIZE} bytes)",
                "type": "invalid_request_error"}})
            return None
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            self._send(400, {"error": "Invalid JSON"})
            return None

    def _send(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


SERVER_START = time.time()


def main():
    parser = argparse.ArgumentParser(description="Claude Agent Server")
    parser.add_argument(
        "--host",
        default=os.getenv("CLAUDE_AGENT_HOST", "127.0.0.1"),
        help="Bind address. Default 127.0.0.1 (loopback only). "
             "Set to 0.0.0.0 explicitly to expose on LAN.",
    )
    parser.add_argument("--port", type=int, default=int(os.getenv("CLAUDE_AGENT_PORT", "8765")))
    args = parser.parse_args()

    if not AUTH_TOKEN:
        logger.error(
            "CLAUDE_AGENT_TOKEN env var is required — server refuses to start without "
            "bearer auth. Set it via [Environment]::SetEnvironmentVariable(\"CLAUDE_AGENT_TOKEN\", "
            "\"<token>\", \"Machine\") (Windows) or export CLAUDE_AGENT_TOKEN=<token> (POSIX) "
            "and restart."
        )
        sys.exit(2)

    try:
        subprocess.run([CLAUDE_BIN, "--version"], capture_output=True, check=True, creationflags=CREATE_NO_WINDOW)
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.error("claude CLI not found. Install: https://claude.ai/code")
        sys.exit(1)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("Claude Agent Server started: http://%s:%d", args.host, args.port)
    logger.info("Model: %s", MODEL)
    if CACHE is not None:
        logger.info("Cache: enabled (max=%d entries, ttl=%.0fs)", _CACHE_SIZE, _CACHE_TTL)
    else:
        logger.info("Cache: disabled (CLAUDE_AGENT_CACHE=0)")
    logger.info("Auth: bearer token required on /v1/* and DELETE /cache (token len=%d)", len(AUTH_TOKEN))
    logger.info("Endpoints: POST /v1/chat/completions, GET /v1/models, GET /health, DELETE /cache")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
