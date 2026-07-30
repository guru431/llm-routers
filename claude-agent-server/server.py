"""
Claude Agent Server — универсальный HTTP-прокси для Claude CLI.

Endpoints:
    POST /v1/chat/completions  — OpenAI-compatible (messages + tools + profile +
                                 response_format + real streaming)
    GET  /v1/models            — Model list (OpenAI-compatible)
    GET  /health               — Healthcheck (включает cache stats, security mode)
    GET  /ready                — readiness probe (200 ready / 503 with per-check details)
    GET  /metrics              — JSON counters (requests/active/overload/timeouts/cache/latency)
    DELETE /cache              — Очистить response cache

Env:
    CLAUDE_AGENT_MODEL      — модель (default: claude-opus-5)
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
    CLAUDE_AGENT_MAX_CONCURRENCY — макс. параллельных claude-вызовов (default: 4; сверх → bounded queue)
    CLAUDE_AGENT_QUEUE_WAIT — сек. ожидания слота перед 429 (default: 5)
    CLAUDE_AGENT_MAX_QUEUE  — макс. ожидающих в очереди (default: 2×concurrency; переполнение → 429+Retry-After)

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
import tempfile
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


# The child env is an ALLOW-list, not a deny-list. Guessing secret NAMES does not
# work: `DATABASE_URL`, `GITHUB_PAT`, `AWS_SESSION`, `STRIPE_SK` and every
# in-house naming convention carry credentials while matching no suffix or
# substring rule, so the old denylist happily handed them to the child CLI. That
# matters most for codex, where even the read-only sandbox executes commands and
# can read its own environment — one prompt injection was enough to print any
# credential the denylist missed.
#
# So: only names the CLI actually needs to run are passed through. The child
# codex/claude CLI authenticates via its own ~/.codex / ~/.claude login, never via
# environment credentials, so nothing else is required.
_CHILD_ENV_ALLOWLIST = frozenset({
    # Process / shell basics
    "PATH", "PATHEXT", "COMSPEC", "SHELL", "TERM", "COLORTERM", "NO_COLOR",
    # Windows system layout
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "OS", "PROCESSOR_ARCHITECTURE",
    "PROCESSOR_IDENTIFIER", "NUMBER_OF_PROCESSORS",
    "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432",
    "COMMONPROGRAMFILES", "COMMONPROGRAMFILES(X86)", "COMMONPROGRAMW6432",
    # Home / config / scratch — the CLI reads ~/.claude or ~/.codex from these
    "HOME", "HOMEDRIVE", "HOMEPATH", "USERPROFILE", "USERNAME", "USERDOMAIN",
    "APPDATA", "LOCALAPPDATA", "TEMP", "TMP", "TMPDIR",
    "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME",
    # Locale / time / TLS trust
    "LANG", "LC_ALL", "LC_CTYPE", "TZ",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS",
})

# Extra names to pass through, comma-separated. For deployments that genuinely
# need something else (a proxy, a corporate CA path). Proxy variables are NOT in
# the default allowlist on purpose: an authenticated proxy URL embeds
# credentials, so passing it to the child has to be a deliberate choice.
_CHILD_ENV_PASSTHROUGH_VAR = "AGENT_CHILD_ENV_PASSTHROUGH"


def _child_env_allowlist() -> frozenset:
    extra = os.getenv(_CHILD_ENV_PASSTHROUGH_VAR, "")
    names = {n.strip().upper() for n in extra.split(",") if n.strip()}
    return _CHILD_ENV_ALLOWLIST | names


def _child_env_without_secrets(**overrides: str) -> dict:
    """Build the child-process environment from an ALLOW-list, then apply
    overrides. Pass to subprocess `env=` so a spawned CLI sees only what it needs
    to run — never this server's tokens, nor any unrelated credential that
    happens not to look like one."""
    allowed = _child_env_allowlist()
    env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
    env.update(overrides)
    return env


# Дефолт поднят до opus-5 / sonnet-5 (2026-07-30). Предыдущее поколение
# (4-8/4-6) осталось в whitelist: на нём завязаны строки bench/models.json и
# уже настроенные клиенты, а `claude -p --model claude-opus-4-8` продолжает
# работать. Список — то, что реально принимает CLI на текущей подписке;
# несуществующее имя CLI отклоняет ("issue with the selected model").
MODEL = os.getenv("CLAUDE_AGENT_MODEL", "claude-opus-5")
MODELS = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

# Capability profiles (Idea 1). claude-agent-server has NO real filesystem access
# (every call runs `--tools ""` — see run_claude), so:
#   chat     — default; plain generation, no tools/session persistence.
#   research — chat + web-search TOOL EMULATION (prompt-injected tools that never
#              actually touch the host FS). Functionally chat+tools here, since
#              claude has no real read/write surface to gate — it is a LABEL that
#              lets callers express intent uniformly across both agent servers.
#   agent    — NOT supported (no workspace-write mode): rejected with 400. Use
#              codex-agent-server for the agent profile.
# A profile is a label, NOT an OS sandbox (see README security section).
CLAUDE_PROFILES = ("chat", "research")

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

# Bounded request queue (Idea 13): under load, instead of an instant 429, wait a
# short bounded time for a free slot; refuse (429 + Retry-After) only if the queue
# is already full or the wait times out. Keeps callers from hammering while the
# waiting set stays bounded so we never pile up unboundedly.
try:
    QUEUE_WAIT_SECONDS = max(0.0, float(os.getenv("CLAUDE_AGENT_QUEUE_WAIT", "5")))
except ValueError:
    QUEUE_WAIT_SECONDS = 5.0
try:
    MAX_QUEUE = max(0, int(os.getenv("CLAUDE_AGENT_MAX_QUEUE", str(MAX_CONCURRENCY * 2))))
except ValueError:
    MAX_QUEUE = MAX_CONCURRENCY * 2
_QUEUE_LOCK = threading.Lock()
_QUEUE_WAITING = 0
# Retry-After seconds advertised on an overload 429 (integer, ≥1).
RETRY_AFTER = max(1, int(round(QUEUE_WAIT_SECONDS)) or 1)


def _acquire_slot() -> bool:
    """Bounded-queue slot acquisition. Fast path: take a free concurrency slot.
    Under load: wait up to QUEUE_WAIT_SECONDS in a bounded queue; refuse if the
    queue is already full (>MAX_QUEUE waiters) or the wait times out. Returns True
    on success (caller must release _CLAUDE_SEM), else False (caller → 429)."""
    global _QUEUE_WAITING
    if _CLAUDE_SEM.acquire(blocking=False):
        return True
    with _QUEUE_LOCK:
        if _QUEUE_WAITING >= MAX_QUEUE:
            return False
        _QUEUE_WAITING += 1
    try:
        return _CLAUDE_SEM.acquire(timeout=QUEUE_WAIT_SECONDS)
    finally:
        with _QUEUE_LOCK:
            _QUEUE_WAITING -= 1


class Metrics:
    """Thread-safe in-process counters + a small latency ring buffer for /metrics
    (Idea 13). Byte-identical with the other agent server; cache_* stay 0 where a
    server has no response cache."""

    def __init__(self, latency_window: int = 128):
        self._lock = threading.Lock()
        self._latency_window = latency_window
        self.total_requests = 0
        self.active = 0
        self.rejected_overload = 0
        self.timeouts = 0
        self.killed_processes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self._latencies: list = []

    def inc(self, field: str, n: int = 1):
        with self._lock:
            setattr(self, field, getattr(self, field) + n)

    def enter(self):
        with self._lock:
            self.total_requests += 1
            self.active += 1

    def leave(self):
        with self._lock:
            if self.active > 0:
                self.active -= 1

    def record_latency(self, seconds: float):
        with self._lock:
            self._latencies.append(seconds)
            excess = len(self._latencies) - self._latency_window
            if excess > 0:
                del self._latencies[0:excess]

    @staticmethod
    def _percentile(sorted_values: list, pct: float) -> float:
        if not sorted_values:
            return 0.0
        k = int(round((pct / 100.0) * (len(sorted_values) - 1)))
        k = max(0, min(len(sorted_values) - 1, k))
        return sorted_values[k]

    def snapshot(self, uptime: int) -> dict:
        with self._lock:
            lat = sorted(self._latencies)
            return {
                "uptime": uptime,
                "total_requests": self.total_requests,
                "active": self.active,
                "rejected_overload": self.rejected_overload,
                "timeouts": self.timeouts,
                "killed_processes": self.killed_processes,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "latency_samples": len(lat),
                "latency_median_s": round(self._percentile(lat, 50), 3),
                "latency_p90_s": round(self._percentile(lat, 90), 3),
                "max_concurrency": MAX_CONCURRENCY,
                "max_queue": MAX_QUEUE,
            }


METRICS = Metrics()

# Ceiling on the system prompt (all system messages + injected tool
# descriptions).
#
# This is NO LONGER an argv limit. The prompt is written to a temp file and
# passed as `--system-prompt-file <path>` (see run_claude), so the ~8191-char
# cmd.exe command-line ceiling stopped applying — yet the old 7000-char guard
# stayed, rejecting perfectly valid long instructions and tool schemas for a
# reason that no longer existed.
#
# What remains worth bounding is the BODY: the system prompt is prepended to
# every request's context, so an unbounded one silently burns the model's window
# and the caller's budget. 200k chars ≈ 50k tokens — generous for real system
# prompts and tool schemas, still a bound. Raise/lower via env.
try:
    MAX_SYSTEM_PROMPT_CHARS = max(1024, int(os.getenv("CLAUDE_AGENT_MAX_SYSTEM_PROMPT", "200000")))
except ValueError:
    MAX_SYSTEM_PROMPT_CHARS = 200000


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
        # Full JSON Schema of the parameters (Idea 12): nested objects/arrays,
        # `items`, `enum`, `oneOf` etc. that the flat signature above drops still
        # reach the model verbatim. Serialized deterministically (sort_keys) so
        # the injected prompt — and claude's cache key built from it — stays stable.
        if isinstance(params, dict) and params:
            lines.append("Full JSON Schema:\n" + json.dumps(params, ensure_ascii=False, sort_keys=True))
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
            # A block whose JSON is not an object (e.g. a bare `1` or a list)
            # is not a valid call: leave it as text instead of crashing on
            # data.get(...). Malformed model tool blocks degrade to text.
            if not isinstance(data, dict):
                continue
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
# Structured output + schema validation (Idea 12)
#   byte-identical helpers with the other agent server — apply fixes to both.
#   Dependency-free: no jsonschema. json_schema_errors is a STRUCTURAL gate
#   (object type + required-keys, recursive into required object props), not a
#   full validator — value types beyond object containment are not checked.
# ============================================================

_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def json_schema_errors(obj, schema) -> list:
    """Minimal structural check of `obj` against a JSON-Schema subset. Returns a
    list of human-readable error strings (empty = ok). Checks only: object schemas
    (type=='object', or `properties`/`required` present) require a dict with all
    `required` keys, recursing into required object-typed properties."""
    errors: list = []
    if not isinstance(schema, dict):
        return errors
    props = schema.get("properties")
    required = schema.get("required") or []
    if not isinstance(required, list):
        required = []
    is_object = schema.get("type") == "object" or isinstance(props, dict) or bool(required)
    if is_object:
        if not isinstance(obj, dict):
            errors.append(f"expected a JSON object, got {type(obj).__name__}")
            return errors
        for key in required:
            if key not in obj:
                errors.append(f"missing required field: {key!r}")
        if isinstance(props, dict):
            for key in required:
                sub = props.get(key)
                if key in obj and isinstance(sub, dict):
                    errors.extend(json_schema_errors(obj[key], sub))
    return errors


def response_format_schema(response_format):
    """Return the JSON-Schema dict for a `{"type":"json_schema"}` response_format,
    else None (json_object accepts any JSON; unsupported types are ignored)."""
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") != "json_schema":
        return None
    js = response_format.get("json_schema") or {}
    schema = js.get("schema") if isinstance(js, dict) else None
    return schema if isinstance(schema, dict) else None


def build_response_format_prompt(response_format):
    """Build a system-prompt section enforcing structured JSON output for a
    supported `response_format` (`json_object` | `json_schema`). Returns None for
    anything unsupported, so the caller leaves the request unconstrained (old
    behaviour preserved — response_format was previously ignored)."""
    if not isinstance(response_format, dict):
        return None
    rf_type = response_format.get("type")
    if rf_type == "json_object":
        return (
            "# OUTPUT FORMAT — STRICT\n"
            "Respond with a SINGLE valid JSON value and NOTHING else: no prose, no "
            "explanation, no markdown code fences. The entire response must parse as JSON."
        )
    if rf_type == "json_schema":
        head = (
            "# OUTPUT FORMAT — STRICT\n"
            "Respond with a SINGLE valid JSON value and NOTHING else: no prose, no "
            "explanation, no markdown code fences. The entire response must parse as JSON"
        )
        schema = response_format_schema(response_format)
        if schema:
            return head + " and MUST conform to this JSON Schema:\n" + json.dumps(
                schema, ensure_ascii=False, sort_keys=True)
        return head + " object."
    return None


def strip_json_fences(text: str) -> str:
    """Return the inside of a single ```json ... ``` fence if the whole text is one
    fenced block, else the stripped text unchanged."""
    candidate = (text or "").strip()
    m = _JSON_FENCE_RE.match(candidate)
    return m.group(1).strip() if m else candidate


def validate_structured_output(text: str, schema) -> tuple:
    """Validate `text` is JSON (json_object) and — if `schema` is given
    (json_schema) — satisfies json_schema_errors. Tolerates one ```json fence.
    Returns (ok: bool, error_message: str)."""
    candidate = strip_json_fences(text)
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError) as exc:
        return False, f"response is not valid JSON: {exc}"
    if schema is not None:
        errs = json_schema_errors(parsed, schema)
        if errs:
            return False, "; ".join(errs)
    return True, ""


def tool_calls_schema_errors(tool_calls: list, tools: list) -> list:
    """For each parsed tool call, check the tool EXISTS in `tools` and that its
    arguments carry the required keys of that tool's parameters schema. Returns a
    flat list of human-readable errors (empty = ok).

    An unknown name is an error, not something to skip: the emulation layer puts
    the offered functions in the system prompt and parses whatever comes back, so
    a model that invents `delete_everything` used to sail straight through this
    validator and reach the client as a legitimate `finish_reason=tool_calls`.
    A tool with no object schema still can't be argument-checked — but its NAME
    is now verified."""
    schemas: dict = {}
    for t in tools or []:
        if isinstance(t, dict):
            fn = t.get("function", {})
            if isinstance(fn, dict) and fn.get("name"):
                schemas[fn["name"]] = fn.get("parameters") or {}
    errors: list = []
    for tc in tool_calls:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        name = fn.get("name", "") if isinstance(fn, dict) else ""
        if name not in schemas:
            errors.append(
                f"tool {name!r}: not one of the offered tools "
                f"({', '.join(sorted(schemas)) or 'none'})"
            )
            continue
        schema = schemas.get(name)
        if not isinstance(schema, dict) or not schema:
            continue
        raw = fn.get("arguments", "")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, ValueError):
            errors.append(f"tool {name!r}: arguments are not valid JSON")
            continue
        errors.extend(f"tool {name!r}: {e}" for e in json_schema_errors(args, schema))
    return errors


# ============================================================
# Claude CLI runner
# ============================================================

def run_claude(prompt: str, system_prompt: str | None = None,
               model: str | None = None, timeout: int = 300) -> str:
    """Call claude CLI and return result text."""
    m = model or MODEL
    if m not in MODELS:
        raise ValueError(f"model not in whitelist: {m!r}")
    # Chat-profile isolation flags (F2): `--tools ""` disables ALL built-in
    # tools (this server emulates tools via prompt injection and never wants
    # claude to actually run Bash/Edit/Read on the host); `--strict-mcp-config`
    # with no `--mcp-config` loads no MCP servers; `--no-session-persistence`
    # stops session files being written to disk. These do NOT sandbox the
    # filesystem — claude still runs as this OS user (see README security
    # section) — they only shrink the host-action surface a "chat" bearer reaches.
    cmd = [
        CLAUDE_BIN, "--model", m, "-p", "-", "--output-format", "json",
        "--tools", "", "--strict-mcp-config", "--no-session-persistence",
    ]
    # Pass the client-controlled system prompt via a temp FILE
    # (`--system-prompt-file`), never as a `--system-prompt=<value>` argv (F1).
    # On Windows CLAUDE_BIN is a `claude.CMD` shim, so subprocess routes argv
    # through cmd.exe, whose metacharacter re-parsing (BatBadBut) a crafted
    # system prompt (embedded quote + `&|^<>()%`) could exploit to break out of
    # the quoting and run a command as this service. Only the server-generated
    # temp path — which has no shell metacharacters — enters argv now.
    sysprompt_file = None
    if system_prompt:
        fd, sysprompt_file = tempfile.mkstemp(suffix=".txt", prefix="claude-sys-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(system_prompt)
        cmd += ["--system-prompt-file", sysprompt_file]
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

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
        try:
            stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the whole tree, then reap so we don't leave a zombie/orphan.
            # Bound both the kill and the reap so a hung taskkill/communicate can't
            # pin the _CLAUDE_SEM slot forever (concurrency leak → eventual 429s).
            if sys.platform == "win32":
                try:
                    r = subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        capture_output=True,
                        creationflags=CREATE_NO_WINDOW,
                        timeout=15,
                    )
                    # F29: taskkill can fail (race, elevation) WITHOUT raising —
                    # verify the process actually died and hard-kill the shim if
                    # not, so a live child can't keep burning the Max quota.
                    if r.returncode != 0 and proc.poll() is None:
                        proc.kill()
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
    finally:
        # Remove the system-prompt temp file. Retry briefly: on Windows the
        # child (or an AV scan) may still hold it for a moment after exit.
        if sysprompt_file:
            for _ in range(5):
                try:
                    os.remove(sysprompt_file)
                    break
                except FileNotFoundError:
                    break
                except OSError:
                    time.sleep(0.05)
            else:
                logger.warning("could not delete claude system-prompt temp file (leaked): %s", sysprompt_file)


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
# Real streaming (Idea 11) — line-by-line JSON events + cancellation
# ============================================================

def _kill_process_tree(proc: "subprocess.Popen") -> None:
    """Terminate a claude subprocess and its descendants (the `.cmd` shim spawns
    a detached `node`). On Windows: `taskkill /T /F`; on POSIX: kill the session
    process group. Best-effort — never raises. Byte-identical with the other
    agent server (used by the streaming runner's watchdog / cancellation)."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            r = subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                creationflags=CREATE_NO_WINDOW,
                timeout=15,
            )
            # taskkill can fail (race, elevation) WITHOUT raising — verify the
            # process actually died and hard-kill the shim if not, so a live
            # orphaned node can't keep burning the subscription / holding outfile.
            if r.returncode != 0 and proc.poll() is None:
                proc.kill()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


class StreamUnsupported(Exception):
    """Raised (before any SSE byte is sent) when the CLI's streaming JSON output
    can't be used, so _handle_chat falls back to the buffered path."""


class StreamFailed(Exception):
    """Raised AFTER deltas have been emitted when the run did not actually
    succeed — watchdog timeout, non-zero exit, or a CLI error event.

    Distinct from StreamUnsupported (which means "nothing ran usefully, retry
    buffered"): here the run started and failed, so the SSE writer must emit an
    error and NOT a clean `[DONE]`. Without this, a killed or crashed CLI whose
    stdout simply reached EOF after one recognized delta was reported to the
    client as a completed answer, with timeout/kill metrics left at zero."""


def _map_stop_reason(sr) -> str:
    """Map an Anthropic stop_reason to an OpenAI finish_reason."""
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(sr, "stop")


def _claude_text_from_blocks(content) -> str:
    """Join text from an Anthropic message content block list."""
    if not isinstance(content, list):
        return ""
    return "".join(
        b.get("text", "") for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )


def _claude_usage(u):
    """Build an OpenAI-style usage dict (real, estimate=False) from an Anthropic
    usage block, or None if no token counts are present. cache read/creation input
    tokens (also real input) are folded into prompt_tokens."""
    if not isinstance(u, dict):
        return None
    inp = u.get("input_tokens")
    out = u.get("output_tokens")
    if not isinstance(inp, int) and not isinstance(out, int):
        return None
    inp = inp if isinstance(inp, int) else 0
    out = out if isinstance(out, int) else 0
    inp += u.get("cache_read_input_tokens", 0) or 0
    inp += u.get("cache_creation_input_tokens", 0) or 0
    return {"prompt_tokens": inp, "completion_tokens": out,
            "total_tokens": inp + out, "estimate": False}


def run_claude_stream(prompt, *, system_prompt=None, model=None, timeout=300):
    """Generator streaming a `claude -p --output-format stream-json` run. Yields
    ('text', delta) as text arrives (partial content_block_delta events when the
    CLI emits them, else whole assistant-message text), then one ('meta', {usage,
    stop_reason, text}) with REAL usage/stop_reason (estimate=False) when the
    result/assistant events carry them. Reads stdout line-by-line so deltas
    surface as emitted. Kills the whole process tree on GeneratorExit (client
    disconnect) or timeout. Raises StreamUnsupported before the first yield if the
    CLI produced no JSON (e.g. stream-json unsupported) so _handle_chat falls back
    to buffered run_claude."""
    m = model or MODEL
    if m not in MODELS:
        raise ValueError(f"model not in whitelist: {m!r}")
    cmd = [
        CLAUDE_BIN, "--model", m, "-p", "-",
        "--output-format", "stream-json", "--verbose",
        "--tools", "", "--strict-mcp-config", "--no-session-persistence",
    ]
    sysprompt_file = None
    if system_prompt:
        fd, sysprompt_file = tempfile.mkstemp(suffix=".txt", prefix="claude-sys-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(system_prompt)
        cmd += ["--system-prompt-file", sysprompt_file]
    child_env = _child_env_without_secrets(CLAUDE_AGENT_SERVER="1")
    popen_kwargs = dict(
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", env=child_env,
    )
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    # The watchdog used to kill the tree and leave NO trace: stdout hit EOF, the
    # loop ended normally and the caller emitted a successful `[DONE]`. Record the
    # kill so the post-loop check can tell "finished" from "was killed".
    killed = {"timeout": False}

    def _on_timeout():
        killed["timeout"] = True
        _kill_process_tree(proc)

    watchdog = threading.Timer(timeout, _on_timeout)
    watchdog.daemon = True
    watchdog.start()

    def _feed():
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
        except (OSError, ValueError):
            pass

    threading.Thread(target=_feed, daemon=True).start()

    # Drain stderr concurrently. Undrained, a chatty CLI fills the pipe buffer and
    # DEADLOCKS on write while we wait on stdout; it also gives the failure path
    # something to log.
    stderr_lines: list = []

    def _drain_stderr():
        try:
            for line in proc.stderr:
                if len(stderr_lines) < 200:
                    stderr_lines.append(line.rstrip())
        except (OSError, ValueError):
            pass

    threading.Thread(target=_drain_stderr, daemon=True).start()

    emitted = ""
    usage_meta = None
    stop_reason = "stop"
    saw_json = False
    try:
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                if not saw_json:
                    raise StreamUnsupported("first stream-json line is not JSON")
                continue
            saw_json = True
            if not isinstance(evt, dict):
                continue
            etype = evt.get("type")
            if etype == "content_block_delta":
                delta = evt.get("delta") or {}
                txt = delta.get("text")
                if isinstance(txt, str) and txt:
                    emitted += txt
                    yield ("text", txt)
            elif etype == "assistant":
                msg = evt.get("message") or {}
                sr = msg.get("stop_reason")
                if sr:
                    stop_reason = _map_stop_reason(sr)
                mu = _claude_usage(msg.get("usage"))
                if mu:
                    usage_meta = mu
                if not emitted:
                    txt = _claude_text_from_blocks(msg.get("content"))
                    if txt:
                        emitted += txt
                        yield ("text", txt)
            elif etype == "result":
                ru = _claude_usage(evt.get("usage"))
                if ru:
                    usage_meta = ru
                if not emitted:
                    rtxt = evt.get("result")
                    if isinstance(rtxt, str) and rtxt:
                        emitted += rtxt
                        yield ("text", rtxt)
        if not saw_json:
            raise StreamUnsupported("no stream-json output")

        # EOF on stdout is NOT success. Wait for the real exit status before
        # calling the run complete.
        try:
            returncode = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            returncode = proc.poll()
        if killed["timeout"]:
            logger.error("claude stream timed out after %ss; stderr: %s",
                         timeout, " | ".join(stderr_lines[-5:]) or "(empty)")
            raise StreamFailed(f"claude timed out after {timeout}s")
        if returncode not in (0, None):
            logger.error("claude stream exit code %s; stderr: %s",
                         returncode, " | ".join(stderr_lines[-5:]) or "(empty)")
            raise StreamFailed("claude command failed")

        yield ("meta", {"usage": usage_meta, "stop_reason": stop_reason, "text": emitted})
    finally:
        watchdog.cancel()
        _kill_process_tree(proc)
        for pipe in (proc.stdout, proc.stderr):
            try:
                pipe.close()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        if sysprompt_file:
            for _ in range(5):
                try:
                    os.remove(sysprompt_file)
                    break
                except FileNotFoundError:
                    break
                except OSError:
                    time.sleep(0.05)
            else:
                logger.warning("could not delete claude system-prompt temp file (leaked): %s", sysprompt_file)


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

    def _send_ready(self):
        """Readiness probe (Idea 13): 200 {ready:true,...} when the server can
        actually serve, else 503 {ready:false, checks:{...}}. Unauthenticated
        (like /health) and exposes only booleans, never paths. Checks: bearer
        token configured, claude CLI resolvable, and the concurrency pool isn't
        saturated."""
        snap = METRICS.snapshot(int(time.monotonic() - SERVER_START_MONO))
        checks = {
            "auth_token_configured": bool(AUTH_TOKEN),
            "cli_found": shutil.which("claude") is not None,
            "not_overloaded": snap["active"] < MAX_CONCURRENCY,
        }
        ready = all(checks.values())
        self._send(200 if ready else 503, {"ready": ready, "checks": checks})

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
                "default_profile": "chat",
                "profiles": list(CLAUDE_PROFILES),
                "uptime": int(time.monotonic() - SERVER_START_MONO),
                "security": "authenticated" if AUTH_TOKEN else "unauthenticated",
            }
            if CACHE is not None:
                payload["cache"] = CACHE.stats()
            else:
                payload["cache"] = {"enabled": False}
            self._send(200, payload)
        elif self.path == "/ready":
            self._send_ready()
        elif self.path == "/metrics":
            if not self._check_auth():
                return
            self._send(200, METRICS.snapshot(int(time.monotonic() - SERVER_START_MONO)))
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
        """OpenAI-compatible chat completions with profiles (Idea 1), structured
        output (Idea 12), real streaming (Idea 11) and tool-calling emulation."""
        METRICS.enter()
        req_start = time.monotonic()
        slot = False
        try:
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
            # Each tool must be an object with an object `function` (a `tools:[null]`
            # or `function: 1` would otherwise reach build_tools_system_prompt and
            # crash on .get(...) → worker exception → RemoteDisconnected. Client error → 400.
            if isinstance(tools_raw, list):
                for t in tools_raw:
                    if not isinstance(t, dict) or not isinstance(t.get("function", {}), dict):
                        self._send(400, {"error": {
                            "message": "each tool must be an object with a function object",
                            "type": "invalid_request_error"}})
                        return

                    # Nested schema shape: build_tools_system_prompt and
                    # json_schema_errors walk `parameters`/`properties`/`required`
                    # assuming dict/list. A scalar there crashed rendering and
                    # surfaced as a 500 — the client's JSON is the client's error.
                    params = t.get("function", {}).get("parameters")
                    if params is not None and not isinstance(params, dict):
                        self._send(400, {"error": {
                            "message": "tool function.parameters must be an object",
                            "type": "invalid_request_error"}})
                        return
                    if isinstance(params, dict):
                        props = params.get("properties")
                        if props is not None and not isinstance(props, dict):
                            self._send(400, {"error": {
                                "message": "tool parameters.properties must be an object",
                                "type": "invalid_request_error"}})
                            return
                        req = params.get("required")
                        if req is not None and not isinstance(req, list):
                            self._send(400, {"error": {
                                "message": "tool parameters.required must be a list",
                                "type": "invalid_request_error"}})
                            return
            # Assistant `tool_calls` are rendered into the prompt with
            # tc.get("id") — a dict instead of a list iterates its KEYS, and a
            # non-dict element has no .get, both raising inside rendering.
            for m in messages:
                tcs = m.get("tool_calls")
                if tcs is None:
                    continue
                if not isinstance(tcs, list) or not all(isinstance(tc, dict) for tc in tcs):
                    self._send(400, {"error": {
                        "message": "message.tool_calls must be a list of objects",
                        "type": "invalid_request_error"}})
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

            # Capability profile (Idea 1). claude has no real filesystem/agentic
            # mode, so only chat|research are valid; agent is an explicit 400 that
            # points at codex-agent-server.
            profile = body.get("profile")
            if profile is None:
                profile = "chat"
            elif profile == "agent":
                self._send(400, {"error": {
                    "message": ("profile 'agent' is not supported by claude-agent-server "
                                "(no workspace-write/agentic mode); use codex-agent-server "
                                "for the agent profile"),
                    "type": "invalid_request_error"}})
                return
            elif profile not in CLAUDE_PROFILES:
                self._send(400, {"error": {
                    "message": f"invalid profile: {profile!r}. Allowed: {list(CLAUDE_PROFILES)} "
                               f"(agent → use codex-agent-server)",
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

            # Structured output (Idea 12): supported response_format types inject a
            # strict-JSON instruction and enable validation + one repair-retry.
            response_format = body.get("response_format")
            rf_prompt = build_response_format_prompt(response_format) if response_format is not None else None
            structured = rf_prompt is not None
            structured_schema = response_format_schema(response_format) if structured else None

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

            # Inject tool descriptions + response_format into the system prompt
            if tools:
                system_parts.append(build_tools_system_prompt(tools))
            if rf_prompt:
                system_parts.append(rf_prompt)

            system_prompt = "\n\n".join(system_parts) if system_parts else None

            # Bound the system prompt BODY (see MAX_SYSTEM_PROMPT_CHARS) — it is
            # prepended to every request, so an unbounded one eats the context
            # window. This is no longer an argv limit: the prompt goes to the CLI
            # through --system-prompt-file.
            if system_prompt and len(system_prompt) > MAX_SYSTEM_PROMPT_CHARS:
                self._send(400, {"error": {
                    "message": (
                        f"system prompt too large "
                        f"({len(system_prompt)} > {MAX_SYSTEM_PROMPT_CHARS} chars); "
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
            # and waste a slot. Fail fast with 400 instead.
            if not prompt.strip():
                self._send(400, {"error": {
                    "message": "no user content to send to claude (prompt is empty)",
                    "type": "invalid_request_error"}})
                return

            # Cache lookup: skip если есть tools / structured output / явный bypass
            cache_bypass = body.get("cache") is False
            cache_eligible = CACHE is not None and not tools and not structured and not cache_bypass
            cached = None
            if cache_eligible:
                cached = CACHE.get(model or MODEL, system_prompt, prompt)

            logger.info("Chat: %d msgs (%d sys, %d conv), tools=%s, %d chars, model=%s, profile=%s, cache=%s",
                         len(messages), len(system_parts), len(conversation),
                         len(tools) if tools else 0, len(prompt), model or MODEL, profile,
                         "hit" if cached is not None else ("miss" if cache_eligible else "skip"))

            resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            resp_model = model or MODEL

            # Cache hit → serve immediately (no slot, no subprocess). cache is only
            # eligible without tools/structured, so no tool parse is needed here.
            def serve_cached(text: str) -> None:
                METRICS.inc("cache_hits")
                usage = {
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": len(text) // 4,
                    "total_tokens": (len(prompt) + len(text)) // 4,
                    "estimate": True,
                    "cached": True,
                    "profile": profile,
                }
                resp_message = {"role": "assistant", "content": text}
                if stream:
                    self._send_stream(resp_id, created, resp_model, resp_message, "stop", usage)
                else:
                    self._send(200, {
                        "id": resp_id, "object": "chat.completion", "created": created,
                        "model": resp_model,
                        "choices": [{"index": 0, "message": resp_message, "finish_reason": "stop"}],
                        "usage": usage,
                    })
                METRICS.record_latency(time.monotonic() - req_start)

            if cached is not None:
                serve_cached(cached)
                return

            # Bounded queue (Idea 13): wait briefly for a slot, else 429+Retry-After.
            if not _acquire_slot():
                METRICS.inc("rejected_overload")
                self._send(429, {"error": {
                    "message": f"server busy: >{MAX_CONCURRENCY} concurrent claude requests, queue full",
                    "type": "rate_limit_error"}},
                    headers={"Retry-After": str(RETRY_AFTER)})
                return
            slot = True

            # Re-check the cache now that we hold a slot: the lookup above and the
            # CACHE.put below straddle the queue wait, so an identical request that
            # was already running may have finished and cached its answer while we
            # queued. Serving it here saves a duplicate CLI subprocess. (It does not
            # collapse two requests that miss simultaneously — the second still runs
            # its own claude; the queue wait is much shorter than a claude call.)
            if cache_eligible:
                # `peek`, not `get`: this is a SECOND look at the same request
                # (a concurrent winner may have filled the entry while we queued),
                # not a new cache access. Counting it in cache stats gave an
                # ordinary uncached request two misses while request-level
                # METRICS.cache_misses rose by one, so /health's hit rate stopped
                # matching the server's own request metrics.
                cached = CACHE.peek(model or MODEL, system_prompt, prompt)
                if cached is not None:
                    serve_cached(cached)
                    return
                METRICS.inc("cache_misses")

            # Real streaming (Idea 11) — only for plain text: tools and structured
            # output must buffer the whole answer to parse/validate/repair it. A
            # streamed plain-text response is not written to the cache (the text is
            # consumed incrementally); cache still serves subsequent identical asks.
            if stream and not tools and not structured:
                base_usage = {
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": 0,
                    "total_tokens": len(prompt) // 4,
                    "estimate": True,
                    "cached": False,
                    "profile": profile,
                }
                gen = run_claude_stream(prompt, system_prompt=system_prompt, model=model, timeout=timeout)
                first_item = None
                try:
                    first_item = next(gen)
                except StreamUnsupported:
                    logger.warning("claude stream-json unsupported; falling back to buffered mode")
                    try:
                        gen.close()
                    except Exception:
                        pass
                    gen = None
                except StopIteration:
                    gen = None
                    first_item = None
                if gen is not None:
                    self._send_stream_live(gen, first_item, resp_id, created, resp_model, base_usage)
                    METRICS.record_latency(time.monotonic() - req_start)
                    return

            # Buffered path (tools, structured output, or streaming fallback).
            result = run_claude(prompt, system_prompt=system_prompt, model=model, timeout=timeout)

            # Structured output: validate + ONE repair-retry (Idea 12). The
            # REPAIRED payload is validated again — without that, a second
            # invalid response still returned 200 with structured_output=true,
            # i.e. the server asserted a contract it had just seen violated.
            if structured and result:
                ok, err = validate_structured_output(result, structured_schema)
                if not ok:
                    logger.info("structured output invalid (%s); one repair-retry", err)
                    repair = (prompt + "\n\n# REPAIR — YOUR PREVIOUS RESPONSE WAS INVALID\n"
                              + err + "\nReturn ONLY the corrected JSON value, nothing else.")
                    result = run_claude(repair, system_prompt=system_prompt, model=model, timeout=timeout)
                    ok, err = validate_structured_output(result or "", structured_schema)
                    if not ok:
                        logger.warning("structured output still invalid after repair: %s", err)
                        self._send(502, {"error": {
                            "message": ("model did not produce output matching the "
                                        f"requested response_format after one repair: {err}"),
                            "type": "upstream_error"}})
                        return

            # Parse tool calls if tools were provided
            tool_calls = []
            content = result
            if tools and result:
                tool_calls, content = parse_tool_calls(result)
                # Validate the call (known tool + required args) + ONE repair-retry
                # (Idea 12), then re-validate the repaired call for the same reason
                # as above.
                errs = tool_calls_schema_errors(tool_calls, tools)
                if errs:
                    logger.info("invalid tool call (%s); one repair-retry", "; ".join(errs))
                    repair = (prompt + "\n\n# REPAIR — YOUR PREVIOUS TOOL CALL WAS INVALID\n"
                              + "; ".join(errs) + "\nReissue the <tool_call> block with ALL required fields set.")
                    result = run_claude(repair, system_prompt=system_prompt, model=model, timeout=timeout)
                    tool_calls, content = parse_tool_calls(result)
                    errs = tool_calls_schema_errors(tool_calls, tools)
                    if errs:
                        logger.warning("tool call still invalid after repair: %s", "; ".join(errs))
                        self._send(502, {"error": {
                            "message": ("model did not produce a valid tool call after "
                                        f"one repair: {'; '.join(errs)}"),
                            "type": "upstream_error"}})
                        return

            if cache_eligible and result:
                CACHE.put(model or MODEL, system_prompt, prompt, result)

            # Build response
            resp_message = {"role": "assistant"}
            if tool_calls:
                resp_message["tool_calls"] = tool_calls
                resp_message["content"] = content if content else None
            else:
                resp_message["content"] = content

            finish_reason = "tool_calls" if tool_calls else "stop"
            # Rough estimate (chars/4). Accurate only for ASCII English; for
            # ru/CJK 1 char ≈ 2-3 tokens, so these undercount badly. Buffered
            # `claude -p` output doesn't expose real token counts; the live
            # streaming path surfaces real counts from stream-json usage events.
            usage = {
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(result) // 4,
                "total_tokens": (len(prompt) + len(result)) // 4,
                "estimate": True,
                "cached": False,
                "profile": profile,
            }
            if structured:
                usage["structured_output"] = True

            if stream:
                # Buffered pseudo-stream (tools/structured/fallback): whole result
                # sliced into SSE chunks so OpenAI-streaming clients don't break.
                self._send_stream(resp_id, created, resp_model, resp_message, finish_reason, usage)
                METRICS.record_latency(time.monotonic() - req_start)
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
            METRICS.record_latency(time.monotonic() - req_start)
        except subprocess.TimeoutExpired:
            METRICS.inc("timeouts")
            METRICS.inc("killed_processes")
            self._send(504, {"error": {"message": "claude timeout", "type": "timeout"}})
        except Exception:
            # Full traceback (incl. claude CLI stderr: local paths, home dir /
            # username, Max-quota internals) goes to the server log only; the
            # client gets a generic message so a bound LAN peer can't harvest
            # internals from the 500 body. Mirrors codex-agent-server.
            logger.exception("claude error")
            self._send(500, {"error": {"message": "internal server error", "type": "server_error"}})
        finally:
            if slot:
                _CLAUDE_SEM.release()
            METRICS.leave()

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

    def _send(self, code: int, data: dict, headers: dict | None = None):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Extra response headers (e.g. Retry-After on an overload 429). Byte-
        # identical with the other agent server.
        if headers:
            for hk, hv in headers.items():
                self.send_header(hk, str(hv))
        self.end_headers()
        # A client that disconnected (common after a timeout) makes wfile.write
        # raise ConnectionError/BrokenPipe. Swallow it so it doesn't bubble to
        # _handle_chat's `except Exception`, which would log a false "claude error"
        # and try to write a second 500 status to the already-dead socket.
        try:
            self.wfile.write(body)
        except (ConnectionError, BrokenPipeError):
            return

    def _send_stream_live(self, gen, first_item, resp_id, created, model, base_usage):
        """Stream ('text', delta) items from generator `gen` as OpenAI SSE content
        deltas in real time (Idea 11), then a finish chunk carrying real usage /
        stop_reason. On a client write failure (disconnect) close the generator —
        its finally kills the CLI process tree, which is the disconnect→cancellation
        link. `base_usage` seeds usage; the generator's ('meta') real token counts
        override it (estimate=False). Byte-identical with the other agent server."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(delta, finish=None, usage=None):
            chunk = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if usage is not None:
                chunk["usage"] = usage
            self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode("utf-8"))

        finish_reason = "stop"
        usage = dict(base_usage)
        disconnected = False
        emitted_chars = 0
        try:
            emit({"role": "assistant"})
            item = first_item
            while item is not None:
                kind, payload = item
                if kind == "text":
                    if payload:
                        emitted_chars += len(payload)
                        emit({"content": payload})
                elif kind == "meta":
                    meta = payload or {}
                    finish_reason = meta.get("stop_reason") or "stop"
                    real = meta.get("usage")
                    if real:
                        usage.update(real)
                try:
                    item = next(gen)
                except StopIteration:
                    item = None
            emit({}, finish=finish_reason, usage=usage)
            self.wfile.write(b"data: [DONE]\n\n")
        except (ConnectionError, BrokenPipeError):
            disconnected = True
        except StreamFailed as e:
            # The run started and FAILED (timeout / non-zero exit). Deltas may
            # already be on the wire, so we can't switch to a JSON error response
            # — but we must not sign off with a successful finish chunk and
            # `[DONE]` either, which is exactly how a killed CLI used to look
            # indistinguishable from a completed answer.
            METRICS.inc("timeouts")
            METRICS.inc("killed_processes")
            # emitted_chars, not len(str(usage)): the message says "chars", and
            # the length of a stringified usage dict is unrelated to how much of
            # the answer actually reached the client before the run died.
            logger.error("live stream failed after %d chars: %s", emitted_chars, e)
            try:
                emit({}, finish="error", usage=usage)
                self.wfile.write(
                    ("data: " + json.dumps(
                        {"error": {"message": str(e), "type": "upstream_error"}},
                        ensure_ascii=False) + "\n\n").encode("utf-8")
                )
            except (ConnectionError, BrokenPipeError):
                disconnected = True
        finally:
            # Cancellation: closing the generator throws GeneratorExit into it, so
            # its finally kills the CLI tree. Harmless if it already finished.
            try:
                gen.close()
            except Exception:
                pass
            if disconnected:
                METRICS.inc("killed_processes")

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
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        # F3: this server speaks plain HTTP. Bound off loopback, the bearer
        # token crosses the LAN in clear text and can be sniffed/replayed. The
        # only supported LAN exposure is behind a TLS/mTLS reverse proxy or a VPN.
        logger.warning(
            "Bound to %s (NOT loopback) over PLAIN HTTP: the bearer token travels "
            "the network unencrypted. Do not expose directly on the LAN — put a "
            "TLS/mTLS reverse proxy (nginx/caddy) or a VPN in front, and bind this "
            "server to 127.0.0.1 behind it.", args.host,
        )
    logger.info("Model: %s", MODEL)
    if CACHE is not None:
        logger.info("Cache: enabled (max=%d entries, ttl=%.0fs)", _CACHE_SIZE, _CACHE_TTL)
    else:
        logger.info("Cache: disabled (CLAUDE_AGENT_CACHE=0)")
    logger.info("Auth: bearer token required on /v1/* and DELETE /cache")
    logger.info("Profiles: %s (default chat)", list(CLAUDE_PROFILES))
    logger.info("Concurrency: %d, queue wait %.1fs, max queue %d", MAX_CONCURRENCY, QUEUE_WAIT_SECONDS, MAX_QUEUE)
    logger.info("Endpoints: POST /v1/chat/completions, GET /v1/models, GET /health, GET /ready, GET /metrics, DELETE /cache")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
