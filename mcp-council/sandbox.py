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
)
BLOCK_DIR_SEGMENTS = ("/.ssh/", "/.aws/", "/.gcp/")
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


# Optional allow-list root(s). The deny-list above is best-effort: a prompt-
# injected context_path can still exfiltrate any non-blacklisted private file.
# Set COUNCIL_CONTEXT_ROOTS (os.pathsep-separated, e.g. the repo/workspace dir)
# to require every context file to resolve INSIDE one of those roots. Unset =
# deny-list-only (backward compatible).
_CONTEXT_ROOTS_ENV = "COUNCIL_CONTEXT_ROOTS"


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


def _looks_binary(p: Path) -> bool:
    """Heuristic: file is binary if its first 8KB contain a NUL byte. Same
    rule git uses (`git diff` falls back to "binary patch" on a NUL hit).
    Cheap, avoids feeding garbage to the LLM."""
    with p.open("rb") as fh:
        chunk = fh.read(_BINARY_SNIFF_BYTES)
    return b"\x00" in chunk


def read_files_with_limit(paths: list[Path]) -> list[tuple[Path, str]]:
    """Читать файлы, проверяя суммарный размер.

    Порядок результата соответствует порядку входных paths.

    Raises SandboxError если суммарный размер превышает MAX_TOTAL_BYTES
    или один из файлов выглядит бинарным (NUL byte в первых 8KB).
    """
    total = 0
    out: list[tuple[Path, str]] = []
    for p in paths:
        total += p.stat().st_size
        if total > MAX_TOTAL_BYTES:
            raise SandboxError(
                f"size limit exceeded: {total // 1024} KB > "
                f"{MAX_TOTAL_BYTES // 1024} KB"
            )
        if _looks_binary(p):
            raise SandboxError(f"binary file rejected: {p}")
        text = p.read_text(encoding="utf-8", errors="replace")
        out.append((p, text))
    return out
