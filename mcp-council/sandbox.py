"""Path validation and blacklist for DeepSeek MCP helper."""

import os

BLOCK_NAMES = frozenset({
    ".env",
    ".git-credentials",
    "project-knowledge-base.yaml",
    "credentials.json",
    "secrets.yaml",
    "secrets.yml",
    # Token/credential files that carry no secret-y extension.
    ".netrc",
    ".npmrc",
    ".pgpass",
    ".dockercfg",
    "kubeconfig",
    # gcloud Application Default Credentials (the real default filename; the
    # ~/.kube/config / ~/.docker/config.json basenames are too generic to block
    # by name, so they're caught by BLOCK_DIR_SEGMENTS below instead).
    "application_default_credentials.json",
    # Common private-key basenames copied outside ~/.ssh (no extension).
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
})

BLOCK_NAME_PREFIXES = (".env.", ".credentials")
# Key/cert/keystore extensions. Deny-list is best-effort — the PEM content-sniff
# in _has_secret_header() is the real safety net for renamed/extensionless keys.
BLOCK_NAME_SUFFIXES = (
    ".pem", ".key", ".p12", ".pfx", ".crt", ".cer",
    ".kdbx", ".keystore", ".jks", ".ppk",
    # Any *.env (vault.env, prod.env, …), not just the literal ".env" name.
    ".env",
)
# Directory segments whose contents are credentials regardless of basename:
# ~/.kube/config, ~/.config/gcloud/*, ~/.docker/config.json and ~/.azure/* all
# carry live cloud creds under generic file names ('config', 'config.json').
BLOCK_DIR_SEGMENTS = (
    "/.ssh/", "/.aws/", "/.gcp/", "/secrets/",
    "/.kube/", "/.config/gcloud/", "/.docker/", "/.azure/",
)
BLOCK_NAMES_IN_CLAUDE_DIR = frozenset({"settings.json", "settings.local.json"})


def is_blocked(path: str) -> bool:
    """True если путь подпадает под blacklist.

    Проверка идёт по ДВУМ нормализациям:
      1) raw — expanduser без realpath, чтобы поймать запросы, где symlink
         в исходном пути ссылается на секреты (или будет ссылаться после
         TOCTOU между проверкой и open).
      2) real — после realpath, на случай косвенных путей через каталог-
         symlink (`/tmp/link/.ssh/id_rsa` → `/home/user/.ssh/id_rsa`).
    Если ЛЮБАЯ из нормализаций попадает в blacklist — блокируем.
    """
    raw = os.path.expanduser(path).replace("\\", "/").lower()
    real = os.path.realpath(os.path.expanduser(path)).replace("\\", "/").lower()

    for p in (raw, real):
        name = os.path.basename(p)
        if name in BLOCK_NAMES:
            return True
        if any(name.startswith(prefix) for prefix in BLOCK_NAME_PREFIXES):
            return True
        if any(name.endswith(suffix) for suffix in BLOCK_NAME_SUFFIXES):
            return True
        if any(seg in p for seg in BLOCK_DIR_SEGMENTS):
            return True
        if "/.claude/" in p and name in BLOCK_NAMES_IN_CLAUDE_DIR:
            return True
    return False


from pathlib import Path

MAX_TOTAL_BYTES = 500 * 1024  # 500 KB
MAX_FILE_COUNT = 50


class SandboxError(Exception):
    """Любая нарушение sandbox-правил (blacklist, size, count, missing file)."""


# Allow-list root(s). The deny-list above is best-effort: a prompt-injected
# context_path can still exfiltrate any non-blacklisted private file (a neutral-
# named private working doc passes every name/content check). So context files
# are FAIL-CLOSED: set COUNCIL_CONTEXT_ROOTS (os.pathsep-separated, e.g. the
# repo/workspace dir) to require every context file to resolve INSIDE one of
# those roots. With no roots set, context_paths are REJECTED — set
# COUNCIL_CONTEXT_FAIL_OPEN=1 to restore the old deny-list-only behavior.
_CONTEXT_ROOTS_ENV = "COUNCIL_CONTEXT_ROOTS"
_CONTEXT_FAIL_OPEN_ENV = "COUNCIL_CONTEXT_FAIL_OPEN"


def fail_open() -> bool:
    """True if the operator opted out of fail-closed context handling.

    When no COUNCIL_CONTEXT_ROOTS is configured, context files are rejected by
    default (fail-closed). Setting COUNCIL_CONTEXT_FAIL_OPEN=1 restores the old
    deny-list-only mode (any non-blacklisted file passes) — an explicit, logged
    choice rather than an implicit hole."""
    return os.environ.get(_CONTEXT_FAIL_OPEN_ENV, "").strip().lower() in ("1", "true", "yes")


def context_roots_configured() -> bool:
    """True if COUNCIL_CONTEXT_ROOTS is set to at least one non-empty root.

    When False the sandbox runs deny-list-only: a prompt-injected context_path
    can still ship any non-blacklisted file to a third-party LLM. Callers
    (server startup, healthcheck) use this to surface the missing guardrail.
    """
    return bool(_allowed_roots())


def _allowed_roots() -> list[Path]:
    raw = os.environ.get(_CONTEXT_ROOTS_ENV, "").strip()
    if not raw:
        return []
    roots: list[Path] = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if part:
            roots.append(Path(os.path.expanduser(part)).resolve())
    return roots


def _within_allowed_roots(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


_SECRET_SNIFF_BYTES = 8192

# Substrings that mark a file as a private key / credential regardless of its
# name or extension. The deny-list above catches known *names*; this catches a
# private key copied to /tmp/mykey, id_rsa renamed to backup, a .pem renamed to
# .txt, etc. — the actual exfiltration risk when such a file is passed as a
# context_path and shipped to a third-party LLM API.
_SECRET_HEADER_MARKERS = (
    b"PRIVATE KEY-----",        # -----BEGIN (RSA|EC|DSA|OPENSSH|generic) PRIVATE KEY-----
    b"PuTTY-User-Key-File",     # PuTTY .ppk private key
)


def _has_secret_header(p: Path) -> bool:
    """True if the file's first 8KB contain a private-key / credential header.
    Content-sniff safety net for renamed or extensionless secrets."""
    try:
        with p.open("rb") as fh:
            chunk = fh.read(_SECRET_SNIFF_BYTES)
    except OSError:
        return False
    return any(marker in chunk for marker in _SECRET_HEADER_MARKERS)


def resolve_and_validate(paths: list[str]) -> list[Path]:
    """Нормализовать и провалидировать список путей. Возвращает list[Path].

    Порядок результата соответствует порядку входных paths (consumers, например
    server._do_draft, опираются на этот invariant для разделения context/examples).

    Raises:
        SandboxError если: количество > MAX_FILE_COUNT, путь в blacklist,
        путь не существует или не является файлом.
    """
    if len(paths) > MAX_FILE_COUNT:
        raise SandboxError(
            f"file count limit exceeded: {len(paths)} > {MAX_FILE_COUNT}"
        )
    roots = _allowed_roots()
    # Fail-closed: with no allowed roots and no explicit opt-out, refuse to read
    # any context file — the deny-list can't be a trust boundary for neutral-named
    # private docs. Guard on `paths` so an empty request stays a no-op.
    if paths and not roots and not fail_open():
        raise SandboxError(
            f"context files disabled: set {_CONTEXT_ROOTS_ENV} to an allowed "
            f"workspace root (fail-closed default), or {_CONTEXT_FAIL_OPEN_ENV}=1 "
            f"to restore deny-list-only behavior"
        )
    resolved: list[Path] = []
    for p in paths:
        if is_blocked(p):
            raise SandboxError(f"blocked by sandbox: {p}")
        path = Path(os.path.expanduser(p)).resolve()
        if roots and not _within_allowed_roots(path, roots):
            raise SandboxError(
                f"path outside allowed roots ({_CONTEXT_ROOTS_ENV}): {p}"
            )
        if not path.is_file():
            raise SandboxError(f"not a file: {p}")
        if _has_secret_header(path):
            raise SandboxError(f"blocked by sandbox (private-key/credential content): {p}")
        resolved.append(path)
    return resolved


_BINARY_SNIFF_BYTES = 8192


def _looks_binary_bytes(chunk: bytes) -> bool:
    """Heuristic on an in-memory buffer: binary if the first 8KB contain a NUL
    byte (same rule git uses). Exception: UTF-16/UTF-32 text (PowerShell's
    Out-File -Encoding unicode/utf32) is full of NULs but legitimate — a leading
    BOM marks it as text. UTF-32 is checked FIRST: its LE BOM (FF FE 00 00)
    starts with the UTF-16 LE BOM, so the shorter test would claim it."""
    if chunk[:4] in (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"):
        return False
    if chunk[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return False
    return b"\x00" in chunk[:_BINARY_SNIFF_BYTES]


def _looks_binary(p: Path) -> bool:
    """Path wrapper around _looks_binary_bytes (kept for callers/tests)."""
    with p.open("rb") as fh:
        return _looks_binary_bytes(fh.read(_BINARY_SNIFF_BYTES))


def _decode_text(raw: bytes) -> str:
    """BOM-aware decode. The deny-list/sniff path already promised to read UTF-16
    correctly, but the old `read_text(encoding="utf-8")` mangled it into
    replacement chars / interleaved NULs. Honour the BOM: UTF-8-SIG, UTF-16 LE/BE,
    else plain UTF-8 (errors='replace' for the odd stray byte)."""
    if raw[:3] == b"\xef\xbb\xbf":
        return raw[3:].decode("utf-8", errors="replace")
    # UTF-32 before UTF-16: the UTF-32 LE BOM (FF FE 00 00) has the UTF-16 LE BOM
    # as its prefix, so testing 2 bytes first would decode it as UTF-16 garbage.
    if raw[:4] == b"\xff\xfe\x00\x00":
        return raw[4:].decode("utf-32-le", errors="replace")
    if raw[:4] == b"\x00\x00\xfe\xff":
        return raw[4:].decode("utf-32-be", errors="replace")
    if raw[:2] == b"\xff\xfe":
        return raw[2:].decode("utf-16-le", errors="replace")
    if raw[:2] == b"\xfe\xff":
        return raw[2:].decode("utf-16-be", errors="replace")
    return raw.decode("utf-8", errors="replace")


def read_files_with_limit(paths: list[Path]) -> list[tuple[Path, str]]:
    """Читать файлы, проверяя суммарный размер.

    Порядок результата соответствует порядку входных paths.

    Каждый файл открывается РОВНО ОДИН раз: размер считается по фактически
    прочитанным байтам (read budget+1), а не отдельным stat() — это закрывает
    TOCTOU-окно между проверкой размера и чтением. Декодирование BOM-aware.

    Raises SandboxError если суммарный размер превышает MAX_TOTAL_BYTES
    или один из файлов выглядит бинарным (NUL byte в первых 8KB).
    """
    total = 0
    out: list[tuple[Path, str]] = []
    for p in paths:
        remaining = MAX_TOTAL_BYTES - total
        # Read one byte past the remaining budget so overflow is detected from the
        # bytes actually read — no separate stat() → no size/read TOCTOU window.
        try:
            with p.open("rb") as fh:
                raw = fh.read(remaining + 1)
        except OSError as e:
            raise SandboxError(f"cannot read file: {p} ({e})")
        total += len(raw)
        if total > MAX_TOTAL_BYTES:
            raise SandboxError(
                f"size limit exceeded: > {MAX_TOTAL_BYTES // 1024} KB"
            )
        if _looks_binary_bytes(raw):
            raise SandboxError(f"binary file rejected: {p}")
        out.append((p, _decode_text(raw)))
    return out
