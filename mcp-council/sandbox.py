"""Path validation and blacklist for DeepSeek MCP helper."""

import os

BLOCK_NAMES = frozenset({
    ".env",
    ".git-credentials",
    "project-knowledge-base.yaml",
    "credentials.json",
    "secrets.yaml",
    "secrets.yml",
})

BLOCK_NAME_PREFIXES = (".env.", ".credentials")
BLOCK_NAME_SUFFIXES = (".pem", ".key")
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
    resolved: list[Path] = []
    for p in paths:
        if is_blocked(p):
            raise SandboxError(f"blocked by sandbox: {p}")
        path = Path(os.path.expanduser(p)).resolve()
        if not path.is_file():
            raise SandboxError(f"not a file: {p}")
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
