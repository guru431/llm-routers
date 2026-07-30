"""
Codex Agent Server — универсальный HTTP-прокси для Codex CLI.

Превращает локально установленный `codex` (подписка ChatGPT) в OpenAI-compatible
HTTP-endpoint. Один API, два мира потребителей:
  - агентный (workspace-write) — Codex правит файлы / запускает shell (агентные клиенты);
  - read-only — чистая генерация текста (mcp-council, claude-code-router, code-review).

Endpoints:
    POST /v1/chat/completions  — OpenAI-compatible (messages + tools + sandbox/workdir/
                                 profile + response_format + real streaming)
    GET  /v1/models            — список моделей (base + `-agent` варианты)
    GET  /health               — healthcheck (liveness only; config fields require read-token)
    GET  /ready                — readiness probe (200 ready / 503 with per-check details)
    GET  /metrics              — JSON counters (requests/active/overload/timeouts/latency)

Режим sandbox разрешается по приоритету (первое сработавшее побеждает):
    1. есть `tools` в запросе          → read-only (клиентские tools несовместимы с агентным)
    2. явное поле `sandbox` в body      → оно (`read-only` | `workspace-write`)
    3. суффикс `-agent` в имени модели  → workspace-write
    4. env CODEX_AGENT_DEFAULT_SANDBOX  → дефолт (read-only)

Env:
    CODEX_AGENT_MODEL          — модель по умолчанию (default: gpt-5.6-sol)
    CODEX_AGENT_MODELS         — базовые id для whitelist через запятую (default: gpt-5.6-sol,gpt-5.5)
    CODEX_AGENT_DEFAULT_SANDBOX— дефолт режима (default: read-only)
    CODEX_AGENT_PORT           — порт (default: 8766)
    CODEX_AGENT_HOST           — bind (default: 127.0.0.1)
    CODEX_AGENT_TOKEN          — bearer-токен для read-only (ОБЯЗАТЕЛЕН — без него сервер не стартует)
    CODEX_AGENT_AGENT_TOKEN    — ОТДЕЛЬНЫЙ bearer-токен для workspace-write (если не задан — workspace-write недоступен, 403)
    CODEX_AGENT_WORKDIR        — корень работы агента (обязателен для workspace-write)
    CODEX_AGENT_WORKDIR_ROOT   — разрешённый корень для per-request override (default = WORKDIR)
    CODEX_AGENT_READ_ROOT      — рабочая директория для read-only codex (default = WORKDIR_ROOT; если не задан — codex видит весь хост)
    CODEX_AGENT_REASONING      — model_reasoning_effort (default: medium)
    CODEX_AGENT_MAX_BODY       — макс. размер тела запроса в байтах (default: 10 MB; >лимит → 413)
    CODEX_AGENT_MAX_CONCURRENCY— макс. параллельных codex-вызовов (default: 4; сверх → bounded queue)
    CODEX_AGENT_QUEUE_WAIT     — сек. ожидания слота перед 429 (default: 5)
    CODEX_AGENT_MAX_QUEUE      — макс. ожидающих в очереди (default: 2×concurrency; переполнение → 429+Retry-After)

Tool calling эмулируется через prompt-injection (только в read-only). Буферный
`usage` — приблизительный (chars/4); Codex не отдаёт реальные счётчики в `-o`. В
настоящем стриме чистого текста usage реальный, когда codex --json его отдаёт.
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

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("codex-agent-server")


# NOTE: _load_dotenv, _child_env_without_secrets, build_tools_system_prompt,
# parse_tool_calls and extract_content are kept byte-identical with
# claude-agent-server/server.py (no shared module on purpose) — apply any fix
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


# Suppress console windows on Windows when calling codex CLI (.cmd shim)
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Resolve the codex binary once. On Windows the npm shim is `codex.CMD`;
# CreateProcess won't append PATHEXT, so `subprocess.run(["codex", ...])` fails
# with FileNotFoundError. shutil.which() respects PATHEXT and returns the full
# path that subprocess can launch directly.
CODEX_BIN = shutil.which("codex") or "codex"

AGENT_SUFFIX = "-agent"
SANDBOX_MODES = ("read-only", "workspace-write")

# Capability profiles (Idea 1): explicit named request postures layered over the
# existing sandbox/`-agent`-suffix mechanism (both preserved for back-compat).
#   chat     — pure generation, no filesystem writes/session persistence (read-only)
#   research — read-only + tool/web emulation, still never writes (read-only)
#   agent    — writes files / runs shell; needs the separate agent token + workdir
#              containment (workspace-write). codex-only; claude rejects it (400).
# A profile is a LABEL + a mode selector; it is NOT an OS sandbox (read-only codex
# still has full host read — see README security section).
PROFILES = ("chat", "research", "agent")

# `gpt-5.6-sol` — имя, которое Codex CLI принимает на подписке ChatGPT (голые
# `gpt-5.6`/`gpt-5.6-codex` → 400 "not supported when using Codex with a ChatGPT
# account"). `gpt-5.5` остаётся в whitelist для уже настроенных клиентов (CCR,
# bench/models.json), но дефолтом с 2026-07-30 идёт 5.6.
DEFAULT_MODEL = os.getenv("CODEX_AGENT_MODEL", "gpt-5.6-sol")
BASE_MODELS = [m.strip() for m in os.getenv("CODEX_AGENT_MODELS", "gpt-5.6-sol,gpt-5.5").split(",") if m.strip()]
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

# Reasoning effort. The per-request override is validated against this set, but
# the ENV value used to skip validation entirely — an invalid one only surfaced
# as a codex CLI failure on every call, with /ready reporting green. Fall back to
# the default and record it so readiness can report the misconfiguration.
REASONING_LEVELS = ("minimal", "low", "medium", "high")
_REASONING_RAW = os.getenv("CODEX_AGENT_REASONING", "medium") or None
REASONING_ENV_VALID = _REASONING_RAW is None or _REASONING_RAW in REASONING_LEVELS
REASONING = _REASONING_RAW if REASONING_ENV_VALID else "medium"
WORKDIR = os.getenv("CODEX_AGENT_WORKDIR") or None
WORKDIR_ROOT = os.getenv("CODEX_AGENT_WORKDIR_ROOT") or WORKDIR

# Working root for read-only codex. read-only sandbox still grants the model
# full host *read* access (codex read-only == full host read access), so `-C`
# is NOT a read boundary — it only pins the cwd (relative paths resolve here,
# modest defense-in-depth). Default to WORKDIR_ROOT; when unset, codex sees the
# whole host (a startup warning is logged).
READ_ROOT = os.getenv("CODEX_AGENT_READ_ROOT") or WORKDIR_ROOT

# Mandatory bearer auth for read-only. Server refuses to start without it;
# required on every /v1/* endpoint.
AUTH_TOKEN = os.getenv("CODEX_AGENT_TOKEN") or None

# Separate bearer for workspace-write (agentic) requests. A leaked read-only
# token must NOT grant file-write/exec. When unset, workspace-write is refused
# with 403 while read-only keeps working with AUTH_TOKEN (backward-compatible).
AGENT_AUTH_TOKEN = os.getenv("CODEX_AGENT_AGENT_TOKEN") or None


def _tokens_collapse_privilege() -> bool:
    """True when a workspace-write agent token is set AND equal to the read-only
    token — which collapses the read/agent privilege separation (a leaked read
    token would also unlock workspace-write). Timing-safe compare."""
    return bool(AGENT_AUTH_TOKEN) and bool(AUTH_TOKEN) and hmac.compare_digest(
        AGENT_AUTH_TOKEN.encode("utf-8"), AUTH_TOKEN.encode("utf-8"))

# cmd.exe metacharacters: codex resolves to a `.cmd` shim on Windows, so a
# workdir path containing these would be reinterpreted by cmd.exe (BatBadBut)
# even though it passed realpath containment. Reject such workdirs early.
_CMD_METACHARS = set('&|^<>()"%')

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

# Bounded request queue (Idea 13): under load, instead of an instant 429, wait a
# short bounded time for a free slot; refuse (429 + Retry-After) only if the queue
# is already full or the wait times out. Keeps callers from hammering while the
# waiting set stays bounded so we never pile up unboundedly.
try:
    QUEUE_WAIT_SECONDS = max(0.0, float(os.getenv("CODEX_AGENT_QUEUE_WAIT", "5")))
except ValueError:
    QUEUE_WAIT_SECONDS = 5.0
try:
    MAX_QUEUE = max(0, int(os.getenv("CODEX_AGENT_MAX_QUEUE", str(MAX_CONCURRENCY * 2))))
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
    on success (caller must release _CODEX_SEM), else False (caller → 429)."""
    global _QUEUE_WAITING
    if _CODEX_SEM.acquire(blocking=False):
        return True
    with _QUEUE_LOCK:
        if _QUEUE_WAITING >= MAX_QUEUE:
            return False
        _QUEUE_WAITING += 1
    try:
        return _CODEX_SEM.acquire(timeout=QUEUE_WAIT_SECONDS)
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
    # A non-string `model` (e.g. numeric JSON) would blow up on name.endswith()
    # below with an AttributeError → worker crash. Reject it as a client error.
    if requested is not None and not isinstance(requested, str):
        raise BadRequest(f"model must be a string: {requested!r}")
    name = requested or DEFAULT_MODEL
    suffix_mode = None
    base = name
    # Only treat `-agent` as the workspace-write suffix when stripping it leaves
    # a known base model. Otherwise a base model whose own name happens to end
    # in `-agent` would be mangled (and fail the whitelist) — keep the full name.
    if name.endswith(AGENT_SUFFIX) and name[: -len(AGENT_SUFFIX)] in BASE_MODELS:
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


def resolve_profile_and_sandbox(tools, body_sandbox: str | None,
                                body_profile: str | None, suffix_mode: str | None) -> tuple[str, str]:
    """Resolve (profile, sandbox). An explicit `profile` in the body takes
    precedence and maps: chat/research → read-only, agent → workspace-write.
    Without a profile, fall back to the legacy sandbox resolution (tools /
    `sandbox` field / `-agent` suffix / default) and derive a reporting profile
    name from the effective sandbox. `tools` always force read-only (client tools
    are incompatible with the agentic mode), so profile='agent' + tools is a
    conflict → BadRequest."""
    if body_profile is not None:
        if body_profile not in PROFILES:
            raise BadRequest(f"invalid profile: {body_profile!r}. Allowed: {list(PROFILES)}")
        if body_profile == "agent":
            if tools:
                raise BadRequest(
                    "profile 'agent' is incompatible with `tools` "
                    "(client tools force read-only execution)")
            return "agent", "workspace-write"
        return body_profile, "read-only"  # chat | research → read-only
    sandbox = resolve_sandbox(tools, body_sandbox, suffix_mode)
    profile = "agent" if sandbox == "workspace-write" else "chat"
    return profile, sandbox


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
    # BatBadBut: codex resolves to a `.cmd` shim, so a workdir path that contains
    # cmd.exe metacharacters would be reinterpreted by the shell even though it
    # passed realpath containment (a client can mkdir such a dir inside the root).
    if _CMD_METACHARS & set(real):
        raise BadRequest(
            f"workdir contains shell metacharacters: {real!r} "
            f"(disallowed: {''.join(sorted(_CMD_METACHARS))})"
        )
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
# Codex CLI runner
# ============================================================

def _kill_process_tree(proc: "subprocess.Popen") -> None:
    """Terminate a codex subprocess and its descendants (the `.cmd` shim spawns
    a detached `node`). On Windows: `taskkill /T /F`; on POSIX: kill the session
    process group. Best-effort — never raises."""
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
            # F29: taskkill can fail (race, elevation) WITHOUT raising — verify
            # the process actually died and hard-kill the shim if not, so a live
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


def run_codex(prompt: str, *, model_base: str, sandbox: str,
              workdir: str | None = None, reasoning: str | None = None,
              timeout: int = 300) -> str:
    """Call `codex exec` and return the final agent message.

    The final message is read from a temp file via `-o` (cleaner than parsing
    JSONL). MCP servers from the global config are disabled (`mcp_servers={}`)
    so the service doesn't trigger the user's zabbix/n8n/etc. on every call.
    """
    # `-` (stdin) goes first as the PROMPT positional; flags follow.
    # Isolation flags (F2): `--ephemeral` writes no session files to disk;
    # `--ignore-user-config` skips ~/.codex/config.toml (auth still uses
    # CODEX_HOME, so the subscription login keeps working) so the request can't
    # pick up the user's providers/model/settings; `--ignore-rules` skips
    # user/project execpolicy .rules. These shrink host coupling but do NOT
    # sandbox host reads — read-only codex still has full host read access
    # (see README security section).
    cmd = [
        CODEX_BIN, "exec", "-",
        "-m", model_base,
        "--sandbox", sandbox,
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
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
    elif sandbox == "read-only" and READ_ROOT:
        # read-only codex has full host *read* access (full host read access),
        # so this `-C` does NOT sandbox reads — it only pins the cwd so relative
        # paths resolve inside READ_ROOT (modest defense-in-depth). When READ_ROOT
        # is unset, codex runs in the server's cwd with whole-host read access
        # (a startup warning is logged in main()).
        cmd += ["-C", READ_ROOT]

    fd, outfile = tempfile.mkstemp(suffix=".txt", prefix="codex-out-")
    os.close(fd)
    cmd += ["-o", outfile]

    # New process group / session so a timeout can kill the whole tree. On
    # Windows codex spawns a `node` child under the `.cmd` shim; killing only
    # the shim PID (what subprocess.run did) orphaned that node. CREATE_NEW_PROCESS_GROUP
    # lets taskkill /T reach the tree; start_new_session on POSIX enables killpg.
    # Strip our bearer tokens and provider keys from the child codex env: a
    # read-only codex has full host read (incl. /proc/self/environ), so an
    # inherited CODEX_AGENT_AGENT_TOKEN would let a prompt-injected read-only
    # request harvest it and self-escalate to workspace-write. Mirrors
    # claude-agent-server's child_env scrub.
    popen_kwargs = {"env": _child_env_without_secrets()}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            **popen_kwargs,
        )
        try:
            _, stderr = proc.communicate(input=prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the whole tree, not just the shim, then reap so no orphaned
            # node lingers holding the subscription / outfile open.
            _kill_process_tree(proc)
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            raise
        if proc.returncode != 0:
            # Log the raw codex stderr server-side only — it can contain workspace
            # paths, code fragments and CLI internals. The exception message stays
            # generic so _handle_chat never leaks it to the client (worse on LAN).
            detail = (stderr or "").strip()
            logger.error("codex exit code %s; stderr: %s", proc.returncode, detail or "(empty)")
            raise RuntimeError("codex command failed")
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
        else:
            # Don't swallow a real leak silently — surface the path so the temp
            # file can be reaped out-of-band.
            logger.warning("could not delete codex temp output file (leaked): %s", outfile)


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

# Process-wide memo of whether `codex exec --json` actually streams events on
# this machine: None = unknown, True/False = learned from a real attempt. Once we
# know it doesn't, streaming is skipped up front so no request pays for a doomed
# first pass — and a workspace-write request never starts one it can't retry.
_STREAM_JSON_SUPPORTED = [None]


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


def _codex_event_text(evt) -> tuple[str, bool]:
    """Best-effort text extraction from a `codex exec --json` stream event.
    Returns (text, is_delta). Handles a few known event shapes; unknown/control
    events return ('', False). Codex's exact JSONL schema varies by version, so
    this stays defensive — the authoritative final answer still comes from the
    `-o` outfile, never from these events alone."""
    if not isinstance(evt, dict):
        return "", False
    etype = evt.get("type")
    if etype in ("agent_message_delta", "output_text.delta", "response.output_text.delta"):
        t = evt.get("delta") or evt.get("text")
        if isinstance(t, str):
            return t, True
    item = evt.get("item")
    if isinstance(item, dict) and item.get("type") in ("agent_message", "assistant_message"):
        t = item.get("text") or item.get("message")
        if isinstance(t, str):
            return t, False
    msg = evt.get("msg")
    if isinstance(msg, dict):
        mt = msg.get("type")
        if mt == "agent_message_delta":
            t = msg.get("delta") or msg.get("message") or msg.get("text")
            if isinstance(t, str):
                return t, True
        if mt in ("agent_message", "assistant_message"):
            t = msg.get("message") or msg.get("text")
            if isinstance(t, str):
                return t, False
    return "", False


def _codex_event_usage(evt):
    """Extract a real token-usage dict (estimate=False) from a codex event if one
    carries input/output token counts, else None."""
    if not isinstance(evt, dict):
        return None
    for key in ("usage", "token_count", "info"):
        u = evt.get(key)
        if isinstance(u, dict):
            inp = u.get("input_tokens", u.get("prompt_tokens", u.get("total_input_tokens")))
            out = u.get("output_tokens", u.get("completion_tokens", u.get("total_output_tokens")))
            if isinstance(inp, int) or isinstance(out, int):
                inp = inp if isinstance(inp, int) else 0
                out = out if isinstance(out, int) else 0
                return {"prompt_tokens": inp, "completion_tokens": out,
                        "total_tokens": inp + out, "estimate": False}
    msg = evt.get("msg")
    if isinstance(msg, dict):
        return _codex_event_usage(msg)
    return None


def run_codex_stream(prompt, *, model_base, sandbox, workdir=None, reasoning=None, timeout=300):
    """Generator streaming a `codex exec --json` run. Yields ('text', delta) as
    events arrive, then one ('meta', {usage, stop_reason, text}) at the end. The
    `-o` outfile still captures the authoritative final message: if the JSONL
    parser recognized no text, the file's contents are emitted as the final delta,
    so correctness never depends on the version-variable event schema. Kills the
    whole process tree on GeneratorExit (client disconnect) or timeout. Raises
    StreamUnsupported before the first yield if `--json` produced no JSON at all
    (caller falls back to buffered run_codex)."""
    cmd = [
        CODEX_BIN, "exec", "-",
        "-m", model_base,
        "--sandbox", sandbox,
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--color", "never",
        "-c", "mcp_servers={}",
        "--json",
    ]
    if reasoning:
        cmd += ["-c", f"model_reasoning_effort={reasoning}"]
    if sandbox == "workspace-write" and workdir:
        cmd += ["-C", workdir]
        cmd += ["-c", f"sandbox_workspace_write.writable_roots={json.dumps([workdir])}"]
    elif sandbox == "read-only" and READ_ROOT:
        cmd += ["-C", READ_ROOT]

    fd, outfile = tempfile.mkstemp(suffix=".txt", prefix="codex-out-")
    os.close(fd)
    cmd += ["-o", outfile]

    popen_kwargs = {"env": _child_env_without_secrets()}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", **popen_kwargs,
    )
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
                    raise StreamUnsupported("first --json line is not JSON")
                continue
            saw_json = True
            u = _codex_event_usage(evt)
            if u:
                usage_meta = u
            txt, is_delta = _codex_event_text(evt)
            if not txt:
                continue
            if is_delta:
                emitted += txt
                yield ("text", txt)
            elif not emitted:
                emitted += txt
                yield ("text", txt)
        if not saw_json:
            raise StreamUnsupported("no --json output")

        # EOF on stdout is NOT success. Wait for the real exit status before
        # calling the run complete.
        try:
            returncode = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            returncode = proc.poll()
        if killed["timeout"]:
            logger.error("codex stream timed out after %ss; stderr: %s",
                         timeout, " | ".join(stderr_lines[-5:]) or "(empty)")
            raise StreamFailed(f"codex timed out after {timeout}s")
        if returncode not in (0, None):
            logger.error("codex stream exit code %s; stderr: %s",
                         returncode, " | ".join(stderr_lines[-5:]) or "(empty)")
            raise StreamFailed("codex command failed")

        # The `-o` outfile is authoritative. Reconcile it against what the events
        # produced even when deltas WERE emitted — a single early delta used to
        # suppress this entirely, so a truncated event stream silently became the
        # whole answer.
        try:
            with open(outfile, encoding="utf-8") as f:
                final = f.read().strip()
        except OSError:
            final = ""
        if final and not emitted:
            emitted = final
            yield ("text", final)
        elif final and final != emitted:
            if final.startswith(emitted):
                remainder = final[len(emitted):]
                emitted = final
                yield ("text", remainder)
            else:
                # Diverged (not merely truncated) — don't double-emit prose the
                # client already has; hand the authoritative text to the caller
                # via meta and say so in the log.
                logger.warning(
                    "codex stream text diverges from the -o outfile "
                    "(streamed %d chars, file %d chars)", len(emitted), len(final)
                )
                emitted = final
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
        for _ in range(5):
            try:
                os.remove(outfile)
                break
            except FileNotFoundError:
                break
            except OSError:
                time.sleep(0.05)
        else:
            logger.warning("could not delete codex temp output file (leaked): %s", outfile)


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
        """Enforce bearer-auth at the transport level. Accepts EITHER the
        read-only token OR the (distinct) workspace-write token, so a request
        carrying the agent token passes this gate and reaches _handle_chat,
        where _check_agent_auth is the sole write gate. Returns False after
        sending 401; caller aborts."""
        if not AUTH_TOKEN:
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            self._send(401, {"error": {"message": "missing bearer token", "type": "auth_error"}})
            return False
        presented = header[len("Bearer "):].strip().encode("utf-8")
        ok = hmac.compare_digest(presented, AUTH_TOKEN.encode("utf-8"))
        if AGENT_AUTH_TOKEN:
            ok = hmac.compare_digest(presented, AGENT_AUTH_TOKEN.encode("utf-8")) or ok
        if not ok:
            self._send(401, {"error": {"message": "invalid bearer token", "type": "auth_error"}})
            return False
        return True

    def _has_valid_read_token(self) -> bool:
        """Non-sending bearer check against the read token. Used by /health to
        decide whether to expose config, without emitting a 401."""
        if not AUTH_TOKEN:
            return False
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return False
        presented = header[len("Bearer "):].strip()
        return hmac.compare_digest(presented.encode("utf-8"), AUTH_TOKEN.encode("utf-8"))

    def _check_agent_auth(self) -> bool:
        """Enforce the workspace-write bearer. The presented token (already a
        valid read-only token) must ALSO match CODEX_AGENT_AGENT_TOKEN. If that
        env is unset, workspace-write is disabled entirely. Sends 403 + returns
        False on failure; caller aborts."""
        if not AGENT_AUTH_TOKEN:
            self._send(403, {"error": {
                "message": "workspace-write requires CODEX_AGENT_AGENT_TOKEN (not configured on this server)",
                "type": "auth_error"}})
            return False
        header = self.headers.get("Authorization", "")
        presented = header[len("Bearer "):].strip() if header.startswith("Bearer ") else ""
        if not hmac.compare_digest(presented.encode("utf-8"), AGENT_AUTH_TOKEN.encode("utf-8")):
            self._send(403, {"error": {
                "message": "invalid workspace-write token (CODEX_AGENT_AGENT_TOKEN required for agentic mode)",
                "type": "auth_error"}})
            return False
        return True

    def _send_ready(self):
        """Readiness probe (Idea 13): 200 {ready:true,...} when the server can
        actually serve the route it DEFAULTS to, else 503 {ready:false,
        checks:{...}}. Unauthenticated (like /health) and exposes only booleans,
        never paths.

        Readiness has to reflect the configuration the default route actually
        uses. Checking only "CLI resolvable" left it green while every request
        failed: a READ_ROOT pointing at a missing/непapка directory makes codex
        fail on `-C`, and an invalid CODEX_AGENT_REASONING made it fail on `-c`.
        Both are checked here, and when the default sandbox is workspace-write
        the workdir/agent-token config it needs is required too."""
        snap = METRICS.snapshot(int(time.monotonic() - SERVER_START_MONO))
        checks = {
            "auth_token_configured": bool(AUTH_TOKEN),
            "cli_found": shutil.which("codex") is not None,
            # Only required when configured — an unset root just disables
            # workspace-write, read-only keeps working, so treat unset as OK.
            "workdir_root_exists": (WORKDIR_ROOT is None) or os.path.isdir(WORKDIR_ROOT),
            # Passed as `-C` on every read-only call: a bad path fails them ALL.
            "read_root_exists": (READ_ROOT is None) or os.path.isdir(READ_ROOT),
            # An out-of-range env value is rejected by the CLI, not by us.
            "reasoning_valid": REASONING_ENV_VALID,
            "not_overloaded": snap["active"] < MAX_CONCURRENCY,
        }
        if DEFAULT_SANDBOX == "workspace-write":
            # The default route is agentic: it needs a containment root and the
            # separate agent bearer, or every default request is refused.
            checks["default_route_workdir_configured"] = bool(WORKDIR_ROOT or WORKDIR)
            checks["default_route_agent_token_configured"] = bool(AGENT_AUTH_TOKEN)
        ready = all(checks.values())
        self._send(200 if ready else 503, {"ready": ready, "checks": checks,
                                           "default_sandbox": DEFAULT_SANDBOX})

    def do_GET(self):
        if self.path == "/health":
            # Unauthenticated callers (LAN liveness probes) get only liveness.
            # Config details (model/sandbox/uptime/security) require a valid
            # read-token so the endpoint can't fingerprint the server.
            if not self._has_valid_read_token():
                self._send(200, {"status": "ok"})
                return
            self._send(200, {
                "status": "ok",
                "model": DEFAULT_MODEL,
                "default_sandbox": DEFAULT_SANDBOX,
                "default_profile": "agent" if DEFAULT_SANDBOX == "workspace-write" else "chat",
                "profiles": list(PROFILES),
                "uptime": int(time.monotonic() - SERVER_START_MONO),
                "security": "authenticated" if AUTH_TOKEN else "unauthenticated",
            })
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
                    "owned_by": "openai",
                } for m in EXPOSED_MODELS],
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

    def _handle_chat(self, body: dict):
        """OpenAI-compatible chat completions with profile/sandbox/workdir routing
        (Idea 1), structured output (Idea 12) and real streaming (Idea 11)."""
        METRICS.enter()
        req_start = time.monotonic()
        slot = False
        try:
            # Validate body shape before touching it: a non-dict body or non-list
            # messages/tools would otherwise raise AttributeError/TypeError → a bare
            # 500 (and via str(exc) could leak internals). Return 400 instead.
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
            tools = body.get("tools")
            if tools is not None and not isinstance(tools, list):
                self._send(400, {"error": {"message": "tools must be a list", "type": "invalid_request_error"}})
                return
            # Each tool must be an object with an object `function` (a `tools:[null]`
            # or `function: 1` would otherwise reach build_tools_system_prompt and
            # crash on .get(...) → worker exception. Client error → 400.
            if isinstance(tools, list):
                for t in tools:
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

            try:
                timeout = int(body.get("timeout", 300))
            except (TypeError, ValueError):
                timeout = 300
            timeout = max(10, min(timeout, 600))

            # Per-request reasoning effort overrides the server default (REASONING).
            # Token-sensitive consumers (code-review) can request 'low'/'minimal'
            # без смены глобального дефолта для council/CCR. Невалидное → дефолт.
            req_reasoning = body.get("reasoning")
            if req_reasoning not in REASONING_LEVELS:
                req_reasoning = None
            effective_reasoning = req_reasoning or REASONING

            stream = bool(body.get("stream"))

            # Structured output (Idea 12): supported response_format types inject a
            # strict-JSON instruction and enable validation + one repair-retry.
            response_format = body.get("response_format")
            rf_prompt = build_response_format_prompt(response_format) if response_format is not None else None
            structured = rf_prompt is not None
            structured_schema = response_format_schema(response_format) if structured else None

            try:
                model_base, suffix_mode = resolve_model(body.get("model"))
                profile, sandbox = resolve_profile_and_sandbox(
                    tools, body.get("sandbox"), body.get("profile"), suffix_mode)
                workdir = resolve_workdir(body.get("workdir") or body.get("cwd")) \
                    if sandbox == "workspace-write" else None
            except BadRequest as exc:
                self._send(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
                return

            # workspace-write requires the *separate* agent token: a leaked read-only
            # token must not grant file-write/exec. Checked here (not in _check_auth)
            # because the mode is only known after model/sandbox resolution. read-only
            # already passed _check_auth against AUTH_TOKEN.
            if sandbox == "workspace-write" and not self._check_agent_auth():
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
                elif role == "assistant" and msg.get("tool_calls"):
                    # A pure tool-call assistant turn has empty content but carries
                    # tool_calls. Render them (with their id) into the text so the
                    # id referenced by the following tool result actually appears in
                    # the prompt — otherwise multi-turn tool loops become incoherent
                    # (result id points at a call absent from the conversation).
                    # Kept in sync with claude-agent-server's identical branch.
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

            if tools:
                system_parts.append(build_tools_system_prompt(tools))
            if rf_prompt:
                system_parts.append(rf_prompt)

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

            logger.info("Chat: %d msgs (%d sys, %d conv), tools=%s, %d chars, model=%s, profile=%s, sandbox=%s",
                         len(messages), len(system_parts), len(conversation),
                         len(tools) if tools else 0, len(prompt), model_base, profile, sandbox)

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            resp_model = body.get("model") or DEFAULT_MODEL

            # Bounded queue (Idea 13): wait briefly for a slot, else 429+Retry-After.
            if not _acquire_slot():
                METRICS.inc("rejected_overload")
                self._send(429, {"error": {
                    "message": f"server busy: >{MAX_CONCURRENCY} concurrent codex requests, queue full",
                    "type": "rate_limit_error"}},
                    headers={"Retry-After": str(RETRY_AFTER)})
                return
            slot = True

            # Real streaming (Idea 11) — only for plain text: tools and structured
            # output must buffer the whole answer to parse/validate/repair it.
            # `_STREAM_JSON_SUPPORTED` remembers a previous StreamUnsupported so a
            # CLI without `--json` doesn't burn a doomed first pass on EVERY
            # request (which matters most for workspace-write, where that pass can
            # already have edited files).
            if stream and not tools and not structured and _STREAM_JSON_SUPPORTED[0] is not False:
                base_usage = {
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": 0,
                    "total_tokens": len(prompt) // 4,
                    "estimate": True,
                    "sandbox": sandbox,
                    "profile": profile,
                }
                gen = run_codex_stream(prompt, model_base=model_base, sandbox=sandbox,
                                       workdir=workdir, reasoning=effective_reasoning, timeout=timeout)
                first_item = None
                try:
                    first_item = next(gen)
                    _STREAM_JSON_SUPPORTED[0] = True
                except StreamUnsupported:
                    # No SSE byte sent yet — the buffered path could re-run the
                    # same prompt. That is safe ONLY when the run is idempotent.
                    # A workspace-write run is not: the first CLI process may
                    # already have edited files before its stdout turned out to
                    # carry no JSON, and re-running would apply the agent's
                    # changes a second time. Report the failure instead.
                    _STREAM_JSON_SUPPORTED[0] = False
                    logger.warning("codex --json streaming unsupported")
                    try:
                        gen.close()
                    except Exception:
                        pass
                    gen = None
                    if sandbox == "workspace-write":
                        self._send(502, {"error": {
                            "message": (
                                "codex --json streaming is unavailable and this is a "
                                "workspace-write request: the run already started and "
                                "may have modified files, so it will NOT be retried "
                                "automatically. Re-issue with stream=false."
                            ),
                            "type": "upstream_error"}})
                        return
                except StopIteration:
                    gen = None
                    first_item = None
                if gen is not None:
                    self._send_stream_live(gen, first_item, completion_id, created, resp_model, base_usage)
                    METRICS.record_latency(time.monotonic() - req_start)
                    return

            # Buffered path (tools, structured output, or streaming fallback).
            result = run_codex(prompt, model_base=model_base, sandbox=sandbox,
                               workdir=workdir, reasoning=effective_reasoning, timeout=timeout)

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
                    result = run_codex(repair, model_base=model_base, sandbox=sandbox,
                                       workdir=workdir, reasoning=effective_reasoning, timeout=timeout)
                    ok, err = validate_structured_output(result or "", structured_schema)
                    if not ok:
                        logger.warning("structured output still invalid after repair: %s", err)
                        self._send(502, {"error": {
                            "message": ("model did not produce output matching the "
                                        f"requested response_format after one repair: {err}"),
                            "type": "upstream_error"}})
                        return

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
                    result = run_codex(repair, model_base=model_base, sandbox=sandbox,
                                       workdir=workdir, reasoning=effective_reasoning, timeout=timeout)
                    tool_calls, content = parse_tool_calls(result)
                    errs = tool_calls_schema_errors(tool_calls, tools)
                    if errs:
                        logger.warning("tool call still invalid after repair: %s", "; ".join(errs))
                        self._send(502, {"error": {
                            "message": ("model did not produce a valid tool call after "
                                        f"one repair: {'; '.join(errs)}"),
                            "type": "upstream_error"}})
                        return

            resp_message = {"role": "assistant"}
            if tool_calls:
                resp_message["tool_calls"] = tool_calls
                resp_message["content"] = content if content else None
            else:
                resp_message["content"] = content

            finish_reason = "tool_calls" if tool_calls else "stop"
            usage = {
                # Rough estimate (chars/4). Codex's -o output doesn't expose real
                # token counts, so buffered responses stay estimate=True; the live
                # streaming path surfaces real counts when codex --json provides them.
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(result) // 4,
                "total_tokens": (len(prompt) + len(result)) // 4,
                "estimate": True,
                # Effective sandbox actually used. May differ from the requested
                # `-agent` model: `tools` in the request force read-only, so a
                # `gpt-5.5-agent` + tools call runs read-only despite resp_model
                # echoing the agent id. Surfaced here (carried into the stream
                # finish chunk too) so routers/logs see the real execution mode.
                "sandbox": sandbox,
                "profile": profile,
            }
            if structured:
                usage["structured_output"] = True

            if stream:
                self._send_stream(completion_id, created, resp_model,
                                  resp_message, finish_reason, usage)
            else:
                self._send(200, {
                    "id": completion_id,
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
            self._send(504, {"error": {"message": "codex timeout", "type": "timeout"}})
        except Exception:
            # Full traceback (incl. any workspace paths / codex output) goes to the
            # server log only; the client gets a generic message so a bound LAN peer
            # can't harvest internals from the 500 body.
            logger.exception("codex error")
            self._send(500, {"error": {"message": "internal server error", "type": "server_error"}})
        finally:
            if slot:
                _CODEX_SEM.release()
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
            # The read itself may have failed on a dropped connection
            # (ConnectionResetError); the error _send would then also fail writing
            # to the dead socket. Best-effort — don't let it raise out of the
            # handler as worker-thread noise.
            try:
                self._send(400, {"error": {"message": "invalid JSON body", "type": "invalid_request_error"}})
            except OSError:
                pass
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
        # A client that disconnected (common after a long codex run) makes
        # wfile.write raise ConnectionError/BrokenPipe. Swallow it so it doesn't
        # bubble to _handle_chat's `except Exception`, which would log a false
        # "codex error" and try to write a second 500 to the dead socket.
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

        The CLI returns the whole answer at once, so this is pseudo-stream:
        a role chunk, one content chunk (if any text), an indexed tool_calls
        chunk per call, the finish chunk (carrying usage), then `[DONE]`. Kept
        byte-identical with claude-agent-server (docstrings excepted).
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
# Monotonic clock for uptime: time.time() can jump (NTP/manual clock shift) and
# make uptime go negative. SERVER_START stays wall-clock for the `created` field.
SERVER_START_MONO = time.monotonic()


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

    # Equal read-only and agent tokens collapse the privilege separation: a
    # leaked read-only token would then also unlock workspace-write. Refuse to
    # start (F30) — a mere warning is invisible under pythonw and would leave the
    # write gate silently open.
    if _tokens_collapse_privilege():
        logger.error(
            "CODEX_AGENT_AGENT_TOKEN equals CODEX_AGENT_TOKEN — read-only/agent "
            "privilege separation is broken (a leaked read token would also grant "
            "workspace-write). Set a DISTINCT agent token, or unset it to disable "
            "workspace-write, and restart."
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
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        # F3: this server speaks plain HTTP. Bound off loopback, the bearer
        # token crosses the LAN in clear text — and it unlocks full host read
        # (read-only) or file-write/exec (agent). The only supported LAN
        # exposure is behind a TLS/mTLS reverse proxy or a VPN.
        logger.warning(
            "Bound to %s (NOT loopback) over PLAIN HTTP: the bearer token travels "
            "the network unencrypted and unlocks full host read (and file-write/exec "
            "with the agent token). Do not expose directly on the LAN — put a "
            "TLS/mTLS reverse proxy or a VPN in front and bind to 127.0.0.1 behind it.",
            args.host,
        )
    logger.info("Models: %s", EXPOSED_MODELS)
    logger.info("Default sandbox: %s", DEFAULT_SANDBOX)
    if WORKDIR:
        logger.info("Workdir root: %s (allowed: %s)", WORKDIR, WORKDIR_ROOT)
    else:
        logger.info("Workdir: not set (workspace-write requests need `workdir` in body)")
    if READ_ROOT:
        logger.info("Read root (read-only cwd): %s", READ_ROOT)
    else:
        logger.warning(
            "CODEX_AGENT_READ_ROOT not set: read-only codex runs with full host "
            "read access (full host read access). Set CODEX_AGENT_READ_ROOT to pin its cwd."
        )
    logger.info("Auth: bearer token required on /v1/*")
    if AGENT_AUTH_TOKEN:
        logger.info("workspace-write: enabled (separate agent token)")
    else:
        logger.info("workspace-write: DISABLED (set CODEX_AGENT_AGENT_TOKEN to enable agentic mode)")
    logger.info("Profiles: %s (default %s)", list(PROFILES),
                "agent" if DEFAULT_SANDBOX == "workspace-write" else "chat")
    logger.info("Concurrency: %d, queue wait %.1fs, max queue %d", MAX_CONCURRENCY, QUEUE_WAIT_SECONDS, MAX_QUEUE)
    logger.info("Endpoints: POST /v1/chat/completions, GET /v1/models, GET /health, GET /ready, GET /metrics")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
