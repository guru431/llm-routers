#!/usr/bin/env python3
"""Canonical all-tests runner for llm_routers.

Each subproject is an independent package with colliding top-level module names
(three different `server.py`, a `cache.py`, three `test_server.py`), so they
CANNOT share one pytest process — `import server` would resolve to whichever
copy lands in sys.modules first. We run each pytest suite in its own
interpreter instead, mirroring the standalone invocation that already works.

Profiles:
    --quick        (default) pytest suites only, no live servers
    --full         quick + compileall (byte-compile every package)
    --integration  live codex-agent-server/integration_suite.py (needs token+server)

Usage:
    python run_tests.py                  # quick
    python run_tests.py --full
    python run_tests.py --integration
    python run_tests.py --quick -k foo   # extra args after the profile go to pytest
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Subprojects that ship pytest suites. Each runs in its own process with cwd set
# to its own directory, so its conftest.py / pyproject.toml resolve exactly as in
# a standalone `pytest` run.
SUITES = ["mcp-council", "claude-agent-server"]

# Packages to byte-compile in --full (catches syntax errors the test suites
# don't import). The agent-server integration suite is excluded — it's live-only.
COMPILEALL_TARGETS = ["mcp-council", "claude-agent-server", "codex-agent-server", "bench"]


def _run(cmd: list[str], cwd: Path) -> int:
    return subprocess.run(cmd, cwd=cwd).returncode


def run_pytest(extra: list[str]) -> list[str]:
    failed: list[str] = []
    for suite in SUITES:
        print(f"\n=== pytest: {suite} ===", flush=True)
        if _run([sys.executable, "-m", "pytest", "-q", *extra], ROOT / suite) != 0:
            failed.append(f"pytest:{suite}")
    return failed


def run_compileall() -> list[str]:
    failed: list[str] = []
    for target in COMPILEALL_TARGETS:
        print(f"\n=== compileall: {target} ===", flush=True)
        if _run([sys.executable, "-m", "compileall", "-q", target], ROOT) != 0:
            failed.append(f"compileall:{target}")
    return failed


def run_integration(extra: list[str]) -> list[str]:
    suite = ROOT / "codex-agent-server" / "integration_suite.py"
    print(f"\n=== integration: {suite.name} ===", flush=True)
    print("(needs a running codex-agent-server on :8766 and CODEX_AGENT_TOKEN)", flush=True)
    if _run([sys.executable, str(suite), *extra], ROOT) != 0:
        return ["integration:codex-agent-server"]
    return []


def main() -> int:
    argv = sys.argv[1:]
    profile = "quick"
    if argv and argv[0] in ("--quick", "--full", "--integration"):
        profile = argv[0][2:]
        argv = argv[1:]
    extra = argv  # forwarded to pytest (quick/full) or the integration suite

    failed: list[str] = []
    if profile == "integration":
        failed += run_integration(extra)
    else:
        failed += run_pytest(extra)
        if profile == "full":
            failed += run_compileall()

    print("\n=== summary ===")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print(f"all checks passed ({profile})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
