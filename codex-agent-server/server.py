"""
Codex Agent Server — универсальный HTTP-прокси для Codex CLI.

Превращает локально установленный `codex` (подписка ChatGPT) в OpenAI-compatible
HTTP-endpoint. Один API, два мира потребителей:
  - агентный (workspace-write) — Codex правит файлы / запускает shell (агентные клиенты);
  - read-only — чистая генерация текста (mcp-council, claude-code-router, code-review).

Endpoints:
    POST /v1/chat/completions  — OpenAI-compatible (messages + tools + sandbox/workdir)
    GET  /v1/models            — список моделей (base + `-agent` варианты)
    GET  /health               — healthcheck (security mode, default sandbox)

Режим sandbox разрешается по приоритету (первое сработавшее побеждает):
    1. есть `tools` в запросе          → read-only (клиентские tools несовместимы с агентным)
    2. явное поле `sandbox` в body      → оно (`read-only` | `workspace-write`)
    3. суффикс `-agent` в имени модели  → workspace-write
    4. env CODEX_AGENT_DEFAULT_SANDBOX  → дефолт (read-only)

Env:
    CODEX_AGENT_MODEL          — модель по умолчанию (default: gpt-5.5)
    CODEX_AGENT_MODELS         — базовые id для whitelist через запятую (default: gpt-5.5)
    CODEX_AGENT_DEFAULT_SANDBOX— дефолт режима (default: read-only)
    CODEX_AGENT_PORT           — порт (default: 8766)
    CODEX_AGENT_HOST           — bind (default: 127.0.0.1)
    CODEX_AGENT_TOKEN          — bearer-токен (ОБЯЗАТЕЛЕН — без него сервер не стартует)
    CODEX_AGENT_WORKDIR        — корень работы агента (обязателен для workspace-write)
    CODEX_AGENT_WORKDIR_ROOT   — разрешённый корень для per-request override (default = WORKDIR)
    CODEX_AGENT_REASONING      — model_reasoning_effort (default: medium)
    CODEX_AGENT_MAX_BODY       — макс. размер тела запроса в байтах (default: 10 MB; >лимит → 413)
    CODEX_AGENT_MAX_CONCURRENCY— макс. параллельных codex-вызовов (default: 4; сверх → 429)

Tool calling эмулируется через prompt-injection (только в read-only). `usage` —
приблизительный (chars/4); Codex не отдаёт реальные счётчики токенов в `-o`.
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
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("codex-agent-server")


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

# Suppress console windows on Windows when calling codex CLI (.cmd shim)
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Resolve the codex binary once. On Windows the npm shim is `codex.CMD`;
# CreateProcess won't append PATHEXT, so `subprocess.run(["codex", ...])` fails
# with FileNotFoundError. shutil.which() respects PATHEXT and returns the full
# path that subprocess can launch directly.
CODEX_BIN = shutil.which("codex") or "codex"

AGENT_SUFFIX = "-agent"
SANDBOX_MODES = ("read-only", "workspace-write")

DEFAULT_MODEL = os.getenv("CODEX_AGENT_MODEL", "gpt-5.5")
BASE_MODELS = [m.strip() for m in os.getenv("CODEX_AGENT_MODELS", "gpt-5.5").split(",") if m.strip()]
if DEFAULT_MODEL not in BASE_MODELS:
    BASE_MODELS.insert(0, DEFAULT_MODEL)

# Exposed model ids: each base plus its `-agent` variant.
EXPOSED_MODELS = []
for _b in BASE_MODELS:
    EXPOSED_MODELS.append(_b)
    EXPOSED_MODELS.append(_b + AGENT_SUFFIX)

DEFAULT_SANDBOX = os.getenv("CODEX_AGENT_DEFAULT_SANDBOX", "read-only")
if DEFAULT_SANDBOX not in SANDBOX_MODES:
    DEFAULT_SANDBOX = "read-only"

REASONING = os.getenv("CODEX_AGENT_REASONING", "medium") or None
WORKDIR = os.getenv("CODEX_AGENT_WORKDIR") or None
WORKDIR_ROOT = os.getenv("CODEX_AGENT_WORKDIR_ROOT") or WORKDIR

# Mandatory bearer auth. Server refuses to start without it; required on every
# endpoint except /health.
AUTH_TOKEN = os.getenv("CODEX_AGENT_TOKEN") or None

# Reject oversized request bodies before reading them into memory (DoS guard).
try:
    MAX_BODY_SIZE = max(1024, int(os.getenv("CODEX_AGENT_MAX_BODY", str(10 * 1024 * 1024))))
except ValueError:
    MAX_BODY_SIZE = 10 * 1024 * 1024

# Cap concurrent codex invocations. Each request spawns a heavy `codex exec`
# subprocess (and burns the ChatGPT subscription); without a cap, many parallel
# authed requests exhaust threads/processes. Excess requests get 429.
try:
    MAX_CONCURRENCY = max(1, int(os.getenv("CODEX_AGENT_MAX_CONCURRENCY", "4")))
except ValueError:
    MAX_CONCURRENCY = 4
_CODEX_SEM = threading.BoundedSemaphore(MAX_CONCURRENCY)


# ============================================================
# Model + sandbox resolution
# ============================================================

class BadRequest(Exception):
    """Client error → HTTP 400 with the message."""


def resolve_model(requested: str | None) -> tuple[str, str | None]:
    """Map a requested model id to (base_model, suffix_mode).

    `<base>-agent` → (base, "workspace-write"); `<base>` → (base, None).
    Raises BadRequest if the base model is not in the whitelist.
    """
    name = requested or DEFAULT_MODEL
    suffix_mode = None
    base = name
    if name.endswith(AGENT_SUFFIX):
        base = name[: -len(AGENT_SUFFIX)]
        suffix_mode = "workspace-write"
    if base not in BASE_MODELS:
        raise BadRequest(f"model not in whitelist: {name!r}. Available: {EXPOSED_MODELS}")
    return base, suffix_mode


def resolve_sandbox(tools, body_sandbox: str | None, suffix_mode: str | None) -> str:
    """Resolve the sandbox mode per the documented priority order."""
    if tools:
        return "read-only"
    if body_sandbox is not None:
        if body_sandbox not in SANDBOX_MODES:
            raise BadRequest(f"invalid sandbox: {body_sandbox!r}. Allowed: {list(SANDBOX_MODES)}")
        return body_sandbox
    if suffix_mode:
        return suffix_mode
    return DEFAULT_SANDBOX


def resolve_workdir(req_workdir: str | None) -> str:
    """Resolve and containment-check the working dir for workspace-write.

    Falls back to CODEX_AGENT_WORKDIR. The resolved real path must be inside
    CODEX_AGENT_WORKDIR_ROOT, else BadRequest.

    Security note: this check picks *where* codex runs (its cwd) and rejects an
    out-of-root request early. The actual write-containment boundary is enforced
    by codex's own `--sandbox workspace-write`, not by this realpath check
    (which a TOCTOU symlink swap could in principle defeat). run_codex pins
    `sandbox_workspace_write.writable_roots` to this resolved path so the boundary
    is enforced by codex itself.
    """
    # A configured root is required to containment-check; without one we cannot
    # safely allow file-writing requests. Guard first so a request-supplied
    # `workdir` can't reach os.path.realpath(None) (TypeError → uncontrolled 500).
    root_base = WORKDIR_ROOT or WORKDIR
    if not root_base:
        raise BadRequest(
            "workspace-write disabled: server has no CODEX_AGENT_WORKDIR / "
            "CODEX_AGENT_WORKDIR_ROOT configured"
        )
    base = req_workdir or WORKDIR
    if not base:
        raise BadRequest(
            "workspace-write requires a working dir: set CODEX_AGENT_WORKDIR or pass "
            "`workdir` in the request body"
        )
    real = os.path.realpath(base)
    root = os.path.realpath(root_base)
    # Compare case-insensitively on case-insensitive filesystems (Windows):
    # os.path.realpath does not normalize case, so `C:\Codex` vs `C:\codex`
    # would otherwise be wrongly rejected. normcase also unifies path separators.
    nreal = os.path.normcase(real)
    nroot = os.path.normcase(root)
    if nreal != nroot and not nreal.startswith(nroot + os.sep):
        raise BadRequest(f"workdir outside allowed root: {real!r} not under {root!r}")
    if not os.path.isdir(real):
        raise BadRequest(f"workdir is not a directory: {real!r}")
    return real


# ============================================================
# Tool calling via prompt injection (read-only mode only)
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
# Codex CLI runner
# ============================================================

def run_codex(prompt: str, *, model_base: str, sandbox: str,
              workdir: str | None = None, reasoning: str | None = None,
              timeout: int = 300) -> str:
    """Call `codex exec` and return the final agent message.

    The final message is read from a temp file via `-o` (cleaner than parsing
    JSONL). MCP servers from the global config are disabled (`mcp_servers={}`)
    so the service doesn't trigger the user's zabbix/n8n/etc. on every call.
    """
    # `-` (stdin) goes first as the PROMPT positional; flags follow.
    cmd = [
        CODEX_BIN, "exec", "-",
        "-m", model_base,
        "--sandbox", sandbox,
        "--skip-git-repo-check",
        "--color", "never",
        "-c", "mcp_servers={}",
    ]
    if reasoning:
        cmd += ["-c", f"model_reasoning_effort={reasoning}"]
    if sandbox == "workspace-write" and workdir:
        cmd += ["-C", workdir]
        # Real write-containment is enforced by codex's own `--sandbox
        # workspace-write`, NOT by resolve_workdir()'s -C/realpath check (that
        # check only picks the cwd). Pin the enforced writable root to the
        # already-containment-checked workdir so codex itself — not just our
        # choice of cwd — is the security boundary. json.dumps escapes Windows
        # backslashes into valid JSON, which codex parses for the `-c` value.
        cmd += ["-c", f"sandbox_workspace_write.writable_roots={json.dumps([workdir])}"]

    fd, outfile = tempfile.mkstemp(suffix=".txt", prefix="codex-out-")
    os.close(fd)
    cmd += ["-o", outfile]
    try:
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
            raise RuntimeError(result.stderr.strip() or f"codex exit code {result.returncode}")
        with open(outfile, encoding="utf-8") as f:
            return f.read().strip()
    finally:
        # On Windows the spawned codex process (or an AV scan) may still hold
        # `outfile` for a moment after exit, making os.remove raise. Retry a few
        # times with a short sleep — almost always clears within ~250ms — so the
        # temp file doesn't leak. Mirrors dialogue/engine.py's replace-retry.
        for _ in range(5):
            try:
                os.remove(outfile)
                break
            except FileNotFoundError:
                break
            except OSError:
                time.sleep(0.05)


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
    # idle socket ops, so it won't interrupt a long in-flight codex call (no
    # socket I/O happens while the subprocess runs).
    timeout = 60

    def log_message(self, format, *args):
        logger.info("%s %s", self.address_string(), format % args)

    def _check_auth(self) -> bool:
        """Enforce bearer-auth. Returns False after sending 401; caller aborts."""
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
            self._send(200, {
                "status": "ok",
                "model": DEFAULT_MODEL,
                "default_sandbox": DEFAULT_SANDBOX,
                "uptime": int(time.time() - SERVER_START),
                "security": "authenticated" if AUTH_TOKEN else "unauthenticated",
            })
        elif self.path == "/v1/models":
            if not self._check_auth():
                return
            self._send(200, {
                "object": "list",
                "data": [{
                    "id": m,
                    "object": "model",
                    "created": int(SERVER_START),
                    "owned_by": "openai",
                } for m in EXPOSED_MODELS],
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

    def _handle_chat(self, body: dict):
        """OpenAI-compatible chat completions with sandbox/workdir routing."""
        messages = body.get("messages", [])
        if not messages:
            self._send(400, {"error": {"message": "messages is required", "type": "invalid_request_error"}})
            return

        try:
            timeout = int(body.get("timeout", 300))
        except (TypeError, ValueError):
            timeout = 300
        timeout = max(10, min(timeout, 600))
        tools = body.get("tools")

        # Per-request reasoning effort overrides the server default (REASONING).
        # Token-sensitive consumers (code-review) can request 'low'/'minimal'
        # без смены глобального дефолта для council/CCR. Невалидное → дефолт.
        req_reasoning = body.get("reasoning")
        if req_reasoning not in ("minimal", "low", "medium", "high"):
            req_reasoning = None
        effective_reasoning = req_reasoning or REASONING

        try:
            model_base, suffix_mode = resolve_model(body.get("model"))
            sandbox = resolve_sandbox(tools, body.get("sandbox"), suffix_mode)
            workdir = resolve_workdir(body.get("workdir") or body.get("cwd")) \
                if sandbox == "workspace-write" else None
        except BadRequest as exc:
            self._send(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            return

        # Separate system prompt from conversation
        system_parts = []
        conversation = []
        for msg in messages:
            role = msg.get("role", "user")
            content = extract_content(msg.get("content", ""))
            if role == "system":
                system_parts.append(content)
            elif role == "tool":
                tool_name = msg.get("name", "function")
                tool_call_id = msg.get("tool_call_id")
                header = f"[Tool {tool_name}"
                if tool_call_id:
                    header += f" id={tool_call_id}"
                header += f"]: {content}"
                conversation.append(("tool", header))
            else:
                conversation.append((role, content))

        if tools:
            system_parts.append(build_tools_system_prompt(tools))

        # Codex has no --system-prompt flag → fold system into the prompt text.
        parts = []
        if system_parts:
            parts.append("# System\n" + "\n\n".join(system_parts))
        if len(conversation) == 1 and not system_parts:
            parts.append(conversation[0][1])
        else:
            for role, content in conversation:
                if role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
                elif role == "tool":
                    parts.append(content)
        prompt = "\n\n".join(parts)

        logger.info("Chat: %d msgs (%d sys, %d conv), tools=%s, %d chars, model=%s, sandbox=%s",
                     len(messages), len(system_parts), len(conversation),
                     len(tools) if tools else 0, len(prompt), model_base, sandbox)

        # Cap concurrent codex subprocesses: reject (429) rather than pile up
        # threads/processes and burn the subscription under parallel load.
        if not _CODEX_SEM.acquire(blocking=False):
            self._send(429, {"error": {
                "message": f"server busy: >{MAX_CONCURRENCY} concurrent codex requests",
                "type": "rate_limit_error"}})
            return

        try:
            result = run_codex(prompt, model_base=model_base, sandbox=sandbox,
                               workdir=workdir, reasoning=effective_reasoning, timeout=timeout)

            tool_calls = []
            content = result
            if tools and result:
                tool_calls, content = parse_tool_calls(result)

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
                "model": body.get("model") or DEFAULT_MODEL,
                "choices": [{
                    "index": 0,
                    "message": resp_message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }],
                # Rough estimate (chars/4). Codex doesn't expose real token
                # counts in -o output; for ru/CJK this undercounts.
                "usage": {
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": len(result) // 4,
                    "total_tokens": (len(prompt) + len(result)) // 4,
                    "estimate": True,
                },
            })
        except subprocess.TimeoutExpired:
            self._send(504, {"error": {"message": "codex timeout", "type": "timeout"}})
        except Exception as exc:
            logger.exception("codex error")
            self._send(500, {"error": {"message": str(exc), "type": "server_error"}})
        finally:
            _CODEX_SEM.release()

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


def main():
    parser = argparse.ArgumentParser(description="Codex Agent Server")
    parser.add_argument(
        "--host",
        default=os.getenv("CODEX_AGENT_HOST", "127.0.0.1"),
        help="Bind address. Default 127.0.0.1 (loopback only). "
             "Set to 0.0.0.0 explicitly to expose on LAN.",
    )
    parser.add_argument("--port", type=int, default=int(os.getenv("CODEX_AGENT_PORT", "8766")))
    args = parser.parse_args()

    if not AUTH_TOKEN:
        logger.error(
            "CODEX_AGENT_TOKEN env var is required — server refuses to start without "
            "bearer auth. Set it via [Environment]::SetEnvironmentVariable(\"CODEX_AGENT_TOKEN\", "
            "\"<token>\", \"Machine\") (Windows) or export CODEX_AGENT_TOKEN=<token> (POSIX) "
            "and restart."
        )
        sys.exit(2)

    try:
        subprocess.run([CODEX_BIN, "--version"], capture_output=True, check=True, creationflags=CREATE_NO_WINDOW)
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.error("codex CLI not found. Install: https://github.com/openai/codex")
        sys.exit(1)

    try:
        server = SingleInstanceServer((args.host, args.port), Handler)
    except OSError as exc:
        logger.error("cannot bind %s:%d — another instance already listening? (%s)",
                     args.host, args.port, exc)
        sys.exit(1)
    logger.info("Codex Agent Server started: http://%s:%d", args.host, args.port)
    logger.info("Models: %s", EXPOSED_MODELS)
    logger.info("Default sandbox: %s", DEFAULT_SANDBOX)
    if WORKDIR:
        logger.info("Workdir root: %s (allowed: %s)", WORKDIR, WORKDIR_ROOT)
    else:
        logger.info("Workdir: not set (workspace-write requests need `workdir` in body)")
    logger.info("Auth: bearer token required on /v1/* (token len=%d)", len(AUTH_TOKEN))
    logger.info("Endpoints: POST /v1/chat/completions, GET /v1/models, GET /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
