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
    CLAUDE_AGENT_CACHE_BYTES — макс. суммарный размер значений кэша в байтах
                              (default: 67108864 = 64 MB; LRU eviction)
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
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cache import ResponseCache

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("claude-agent-server")


# NOTE: _load_dotenv, _child_env_without_secrets, build_tools_system_prompt,
# parse_tool_calls and extract_content are kept byte-identical with
# codex-agent-server/server.py (no shared module on purpose) — apply any fix
# to both copies.
def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from a .env file next to this script into the
    environment, without overwriting variables already set. Lets the server
    read its token (and other config) from a co-located .env without a
    python-dotenv dependency; an explicit env var still wins. Required for
    boot-launched deployments where the process has no inherited shell env."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


_load_dotenv()


# Names whose upper-cased form ends with one of these (or contains SECRET/
# PASSWORD/TOKEN) are treated as secrets and stripped from any child-process env.
# The child codex/claude CLI authenticates via its own ~/.codex / ~/.claude login,
# never via these, so removing them is safe — and it closes the /proc/self/environ
# and `echo $VAR` exfiltration vectors for provider keys AND this server's own
# bearer tokens (a read-only prompt-injection must not be able to harvest the
# workspace-write token and self-escalate).
_SECRET_ENV_SUFFIXES = ("_TOKEN", "_KEY", "_SECRET", "_PASSWORD", "_APIKEY", "_CREDENTIALS")
_SECRET_ENV_SUBSTRINGS = ("SECRET", "PASSWORD", "TOKEN")


def _child_env_without_secrets(**overrides: str) -> dict:
    """Copy os.environ with every secret-looking var removed, then apply
    overrides. Pass to subprocess `env=` so a spawned CLI can't leak our keys."""
    env = {
        k: v for k, v in os.environ.items()
        if not (k.upper().endswith(_SECRET_ENV_SUFFIXES)
                or any(s in k.upper() for s in _SECRET_ENV_SUBSTRINGS))
    }
    env.update(overrides)
    return env


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
try:
    _CACHE_BYTES = max(1024, int(os.getenv("CLAUDE_AGENT_CACHE_BYTES", str(64 * 1024 * 1024))))
except ValueError:
    _CACHE_BYTES = 64 * 1024 * 1024

CACHE = ResponseCache(max_size=_CACHE_SIZE, ttl_seconds=_CACHE_TTL,
                      max_bytes=_CACHE_BYTES) if CACHE_ENABLED else None

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

# The system prompt (all system messages + injected tool descriptions) is passed
# as a single `--system-prompt=<value>` argv. CLAUDE_BIN resolves to the npm
# `claude.CMD` shim, so subprocess routes it through cmd.exe, whose command-line
# ceiling is ~8191 chars (the raw CreateProcessW limit is 32767). A large system
# prompt + many tools would silently overflow → truncated flag or spawn failure
# surfacing as a generic 500. Reject with a clear 400 instead. Default 7000 is
# conservative for cmd.exe minus the rest of argv; raise via env if you confirm a
# higher real limit (e.g. POSIX shells).
try:
    SYSTEM_PROMPT_ARGV_LIMIT = max(1024, int(os.getenv("CLAUDE_AGENT_MAX_SYSTEM_PROMPT", "7000")))
except ValueError:
    SYSTEM_PROMPT_ARGV_LIMIT = 7000


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
    # Сигнал хукам Claude Code (~/.claude/settings.json: SessionStart/SessionEnd),
    # что это headless-вызов сервера: тяжёлая инъекция wiki-контекста (~162K токенов,
    # ~$3/вызов, упор в лимит Max → "claude exit code 1") должна быть пропущена.
    # Scrub the bearer token AND every provider key/secret from the child env:
    # the claude CLI authenticates via its own ~/.claude login, so it needs none
    # of them, and leaving them in place leaks them into subprocess inspection /
    # crash dumps / further-spawned tools. CLAUDE_AGENT_SERVER signals hooks that
    # this is a headless server call.
    child_env = _child_env_without_secrets(CLAUDE_AGENT_SERVER="1")

    # Start in its own process group/session so a timeout can kill the WHOLE
    # tree, not just the launcher. On Windows CLAUDE_BIN is a `claude.CMD` shim
    # that spawns node.exe; killing only the shim orphans node.exe (it keeps
    # running and burns the Max quota). CREATE_NEW_PROCESS_GROUP lets taskkill
    # /T walk the tree; start_new_session does the same via a POSIX process group.
    popen_kwargs = dict(
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=child_env,
    )
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the whole tree, then reap so we don't leave a zombie/orphan.
        # Bound both the kill and the reap so a hung taskkill/communicate can't
        # pin the _CLAUDE_SEM slot forever (concurrency leak → eventual 429s).
        if sys.platform == "win32":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    creationflags=CREATE_NO_WINDOW,
                    timeout=15,
                )
            except subprocess.TimeoutExpired:
                # proc may already be dead (ProcessLookupError) — don't let a
                # fallback kill turn a 504 timeout into a 500.
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        raise

    if proc.returncode != 0:
        # Log the raw claude stderr/stdout server-side only — it can carry the
        # home dir / username, Max-quota internals and local paths. The exception
        # message stays generic so _handle_chat never leaks it to the client even
        # if the surrounding error handling changes. Mirrors codex-agent-server's
        # run_codex (generic "codex command failed").
        detail = (stderr or "").strip() or (stdout or "").strip()[:800]
        logger.error("claude exit code %s; detail: %s", proc.returncode, detail or "(empty)")
        raise RuntimeError("claude command failed")
    # Parse JSON output to extract result
    try:
        data = json.loads((stdout or "").strip())
        if data.get("is_error"):
            raise RuntimeError(data.get("result", "Unknown error"))
        return data.get("result", "").strip()
    except json.JSONDecodeError:
        return (stdout or "").strip()


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

    def _authed(self) -> bool:
        """Side-effect-free auth check (no 401 sent). True if no token is
        configured, or a valid bearer was presented. Used by /health to decide
        how much config to disclose."""
        if not AUTH_TOKEN:
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return False
        presented = header[len("Bearer "):].strip()
        return hmac.compare_digest(presented.encode("utf-8"), AUTH_TOKEN.encode("utf-8"))

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
            # Liveness probe must work without a token (200 + minimal body).
            # Config details (model/uptime/security/cache) are only disclosed to
            # an authenticated caller, so an anonymous probe can't fingerprint
            # the deployment.
            if not self._authed():
                self._send(200, {"status": "ok"})
                return
            payload = {
                "status": "ok",
                "model": MODEL,
                "uptime": int(time.monotonic() - SERVER_START_MONO),
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
            self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            if not self._check_auth():
                return
            body = self._read_body()
            if body is None:
                return
            self._handle_chat(body)
        else:
            self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_DELETE(self):
        if self.path == "/cache":
            if not self._check_auth():
                return
            if CACHE is None:
                self._send(404, {"error": {"message": "cache disabled", "type": "invalid_request_error"}})
                return
            CACHE.clear()
            self._send(200, {"status": "cleared", "stats": CACHE.stats()})
        else:
            self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def _handle_chat(self, body: dict):
        """OpenAI-compatible chat completions with tool calling support."""
        # Validate body shape before touching it: a non-dict body or non-list
        # messages/tools would otherwise raise AttributeError/TypeError before
        # the try below → a bare worker crash with no HTTP response. Return 400
        # instead. Mirrors codex-agent-server.
        if not isinstance(body, dict):
            self._send(400, {"error": {"message": "request body must be a JSON object", "type": "invalid_request_error"}})
            return
        messages = body.get("messages", [])
        if not isinstance(messages, list):
            self._send(400, {"error": {"message": "messages must be a list", "type": "invalid_request_error"}})
            return
        if not messages:
            self._send(400, {"error": {"message": "messages is required", "type": "invalid_request_error"}})
            return
        if not all(isinstance(m, dict) for m in messages):
            self._send(400, {"error": {"message": "each message must be an object", "type": "invalid_request_error"}})
            return
        tools_raw = body.get("tools")
        if tools_raw is not None and not isinstance(tools_raw, list):
            self._send(400, {"error": {"message": "tools must be a list", "type": "invalid_request_error"}})
            return

        model = body.get("model")
        # Reject an unknown model with 400 invalid_request_error rather than
        # letting run_claude raise ValueError → broad except → 500 server_error.
        # `model` omitted falls back to the (validated-at-startup) default MODEL.
        if model is not None and model not in MODELS:
            self._send(400, {"error": {
                "message": f"model not in whitelist: {model!r}",
                "type": "invalid_request_error"}})
            return
        # Clamp client-provided timeout to [10s, 600s] to prevent DoS via
        # `timeout: 0` (instant fail) or `timeout: 999999` (hung worker).
        try:
            timeout = int(body.get("timeout", 300))
        except (TypeError, ValueError):
            timeout = 300
        timeout = max(10, min(timeout, 600))
        tools = body.get("tools")
        stream = bool(body.get("stream"))

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
            elif role == "assistant" and msg.get("tool_calls"):
                # A pure tool-call assistant turn has empty content but carries
                # tool_calls. Render them (with their id) into the text so the
                # id referenced by the following tool result actually appears in
                # the prompt — otherwise multi-turn tool loops become incoherent
                # (result id points at a call absent from the conversation).
                call_lines = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    call_lines.append(
                        f"[Called tool {fn.get('name', '')} id={tc.get('id', '')} "
                        f"with {fn.get('arguments', '')}]"
                    )
                merged = "\n".join(call_lines)
                if content:
                    merged = content + "\n" + merged
                conversation.append((role, merged))
            else:
                conversation.append((role, content))

        # Inject tool descriptions into system prompt
        if tools:
            system_parts.append(build_tools_system_prompt(tools))

        system_prompt = "\n\n".join(system_parts) if system_parts else None

        # Guard the CLI argv length (see SYSTEM_PROMPT_ARGV_LIMIT): a system
        # prompt too large for `--system-prompt=` on Windows would otherwise fail
        # opaquely as a 500. A 400 is honest and actionable.
        if system_prompt and len(system_prompt) > SYSTEM_PROMPT_ARGV_LIMIT:
            self._send(400, {"error": {
                "message": (
                    f"system prompt too large for CLI argv "
                    f"({len(system_prompt)} > {SYSTEM_PROMPT_ARGV_LIMIT} chars); "
                    f"reduce system messages or number/size of tools"),
                "type": "invalid_request_error"}})
            return

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

        # Reject empty input (e.g. messages with only a system role) BEFORE taking
        # a concurrency slot: an empty stdin makes the claude CLI spawn for nothing
        # and waste a _CLAUDE_SEM slot. Fail fast with 400 instead.
        if not prompt.strip():
            self._send(400, {"error": {
                "message": "no user content to send to claude (prompt is empty)",
                "type": "invalid_request_error"}})
            return

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

            resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            resp_model = model or MODEL
            finish_reason = "tool_calls" if tool_calls else "stop"
            # Rough estimate (chars/4). Accurate only for ASCII English; for
            # ru/CJK 1 char ≈ 2-3 tokens, so these undercount badly. claude CLI
            # doesn't expose real token counts in -p output. Same usage object is
            # emitted on both transports (non-stream body and the stream finish
            # chunk) so streaming clients also get token counters.
            usage = {
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(result) // 4,
                "total_tokens": (len(prompt) + len(result)) // 4,
                "estimate": True,
                "cached": cached is not None,
            }

            if stream:
                # The CLI gives us the full answer at once, so we can't truly
                # stream. We DO buffer the whole result, then emit it as SSE
                # chunks so OpenAI-streaming clients (Open WebUI) don't break on
                # a single JSON blob. Same id/created/model as the non-stream body.
                self._send_stream(resp_id, created, resp_model, resp_message, finish_reason, usage)
                return

            self._send(200, {
                "id": resp_id,
                "object": "chat.completion",
                "created": created,
                "model": resp_model,
                "choices": [{
                    "index": 0,
                    "message": resp_message,
                    "finish_reason": finish_reason,
                }],
                "usage": usage,
            })
        except subprocess.TimeoutExpired:
            self._send(504, {"error": {"message": "claude timeout", "type": "timeout"}})
        except Exception:
            # Full traceback (incl. claude CLI stderr: local paths, home dir /
            # username, Max-quota internals) goes to the server log only; the
            # client gets a generic message so a bound LAN peer can't harvest
            # internals from the 500 body. Mirrors codex-agent-server.
            logger.exception("claude error")
            self._send(500, {"error": {"message": "internal server error", "type": "server_error"}})

    def _read_body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            self._send(400, {"error": {"message": "invalid Content-Length", "type": "invalid_request_error"}})
            return None
        if length < 0:
            self._send(400, {"error": {"message": "invalid Content-Length", "type": "invalid_request_error"}})
            return None
        if length > MAX_BODY_SIZE:
            self._send(413, {"error": {
                "message": f"request body too large ({length} > {MAX_BODY_SIZE} bytes)",
                "type": "invalid_request_error"}})
            return None
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            self._send(400, {"error": {"message": "invalid JSON body", "type": "invalid_request_error"}})
            return None

    def _send(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        # A client that disconnected (common after a timeout) makes wfile.write
        # raise ConnectionError/BrokenPipe. Swallow it so it doesn't bubble to
        # _handle_chat's `except Exception`, which would log a false "claude error"
        # and try to write a second 500 status to the already-dead socket.
        try:
            self.wfile.write(body)
        except (ConnectionError, BrokenPipeError):
            return

    def _send_stream(self, resp_id: str, created: int, model: str,
                     resp_message: dict, finish_reason: str, usage: dict):
        """Emit the (already-complete) response as OpenAI SSE chunks.

        The claude CLI returns the whole answer at once, so this is pseudo-stream:
        a role chunk, one content chunk (if any text), an optional tool_calls
        chunk, the finish chunk (carrying usage), then `[DONE]`. Each line is
        `data: {json}\\n\\n`, object="chat.completion.chunk", sharing the
        non-stream id/created/model.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(delta: dict, finish=None, usage=None):
            chunk = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if usage is not None:
                chunk["usage"] = usage
            line = "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
            self.wfile.write(line.encode("utf-8"))

        # Once the 200 + headers are sent, the response has begun. A mid-stream
        # client disconnect makes wfile.write raise ConnectionError/BrokenPipe;
        # swallow it here and return rather than let it bubble to _handle_chat's
        # `except Exception`, which would log a false error and try to write a
        # second 500 status to the already-dead socket.
        try:
            # 1) role
            emit({"role": "assistant"})
            # 2) content (if any text was produced)
            content = resp_message.get("content")
            if content:
                emit({"content": content})
            # 3) tool_calls — one indexed delta per call, OpenAI-stream shaped so
            # strict SDKs that accumulate by `index` reassemble them (a single
            # non-indexed blob got dropped). We hold the whole call, so each
            # delta carries the full arguments — no need to fragment.
            tool_calls = resp_message.get("tool_calls")
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    fn = tc.get("function", {})
                    emit({"tool_calls": [{
                        "index": i,
                        "id": tc.get("id"),
                        "type": tc.get("type", "function"),
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", ""),
                        },
                    }]})
            # 4) finish (carries usage) + 5) [DONE]
            emit({}, finish=finish_reason, usage=usage)
            self.wfile.write(b"data: [DONE]\n\n")
        except (ConnectionError, BrokenPipeError):
            return


class SingleInstanceServer(ThreadingHTTPServer):
    # HTTPServer sets allow_reuse_address=1 (SO_REUSEADDR). On Windows that lets
    # a SECOND process bind the same port and the OS load-balances connections
    # between them — restarts left stale instances live, so requests hit servers
    # with different code intermittently (the "duplicate instance" bug). Disabling
    # reuse makes a second bind fail fast (WSAEADDRINUSE) → only one instance ever
    # listens on the port. A killed listener's socket is freed immediately (no
    # TIME_WAIT on a non-connected listening socket), so restart-after-crash is fine.
    allow_reuse_address = False


SERVER_START = time.time()
# Monotonic clock for uptime: immune to wall-clock jumps (NTP sync, manual set,
# DST) that can make a time.time()-based uptime go negative or spike.
SERVER_START_MONO = time.monotonic()


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

    # Fail fast on a bad default model. run_claude() whitelists `m` and raises
    # ValueError on a non-whitelisted model → without this every request that
    # omits `model` would 500. Better to refuse to start with a clear message.
    if MODEL not in MODELS:
        logger.error(
            "CLAUDE_AGENT_MODEL=%r is not in the supported list %s — server refuses "
            "to start (every request without an explicit `model` would fail). Set "
            "CLAUDE_AGENT_MODEL to a supported id and restart.",
            MODEL, MODELS,
        )
        sys.exit(2)

    try:
        subprocess.run([CLAUDE_BIN, "--version"], capture_output=True, check=True, creationflags=CREATE_NO_WINDOW)
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.error("claude CLI not found. Install: https://claude.ai/code")
        sys.exit(1)

    try:
        server = SingleInstanceServer((args.host, args.port), Handler)
    except OSError as exc:
        logger.error("cannot bind %s:%d — another instance already listening? (%s)",
                     args.host, args.port, exc)
        sys.exit(1)
    logger.info("Claude Agent Server started: http://%s:%d", args.host, args.port)
    logger.info("Model: %s", MODEL)
    if CACHE is not None:
        logger.info("Cache: enabled (max=%d entries, ttl=%.0fs)", _CACHE_SIZE, _CACHE_TTL)
    else:
        logger.info("Cache: disabled (CLAUDE_AGENT_CACHE=0)")
    logger.info("Auth: bearer token required on /v1/* and DELETE /cache")
    logger.info("Endpoints: POST /v1/chat/completions, GET /v1/models, GET /health, DELETE /cache")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
