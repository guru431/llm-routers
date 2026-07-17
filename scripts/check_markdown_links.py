#!/usr/bin/env python3
"""Markdown relative-link checker for CI (docs contract).

Scans every tracked *.md file for inline links `[text](target)` and reference
definitions, and fails if a RELATIVE link points at a path that does not exist.
External (http/https/mailto), pure-anchor (#...), and template-ish (<...>,
containing a space) targets are skipped — only real in-repo file links are
verified, which is what prevents a README from linking a moved/renamed file.

Exit 0 = all relative links resolve; exit 1 = at least one broken link (printed).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# [text](target) and [text](target "title"); also bare reference defs [id]: target
_INLINE = re.compile(r"\[[^\]]*\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
_REFDEF = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)", re.MULTILINE)


def _tracked_md() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "*.md"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: walk the tree if git isn't available.
        return [p for p in ROOT.rglob("*.md") if ".git" not in p.parts]
    return [ROOT / line for line in out.splitlines() if line.strip()]


def _is_external_or_skip(target: str) -> bool:
    if target.startswith(("http://", "https://", "mailto:", "#", "//")):
        return True
    if target.startswith("<") or " " in target:  # template placeholder
        return True
    return False


def check_file(md: Path) -> list[str]:
    broken: list[str] = []
    text = md.read_text(encoding="utf-8", errors="replace")
    targets = _INLINE.findall(text) + _REFDEF.findall(text)
    for raw in targets:
        target = raw.split("#", 1)[0].strip()  # drop anchor fragment
        if not target or _is_external_or_skip(raw):
            continue
        resolved = (md.parent / target).resolve()
        if not resolved.exists():
            broken.append(f"{md.relative_to(ROOT)} -> {target}")
    return broken


def main() -> int:
    all_broken: list[str] = []
    for md in _tracked_md():
        if md.exists():
            all_broken.extend(check_file(md))
    if all_broken:
        print("Broken relative markdown links:")
        for b in all_broken:
            print(f"  {b}")
        return 1
    print("All relative markdown links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
