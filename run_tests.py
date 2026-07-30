#!/usr/bin/env python3
"""Canonical all-tests runner for llm_routers.

Each subproject is an independent package with colliding top-level module names
(three different `server.py`, a `cache.py`, two `test_server.py` — the codex
agent-server suite was renamed to integration_suite.py), so they CANNOT share
one pytest process — `import server` would resolve to whichever copy lands in
sys.modules first. We run each pytest suite in its own interpreter instead,
mirroring the standalone invocation that already works.

Profiles:
    --quick        (default) pytest suites only, no live servers
    --full         quick + compileall (byte-compile every package)
    --integration  live codex-agent-server/integration_suite.py (needs token+server)
    --doctor       preflight: are the dev deps importable in THIS interpreter?

Usage:
    python run_tests.py                  # quick
    python run_tests.py --full
    python run_tests.py --integration
    python run_tests.py --doctor         # distinguish "deps missing" from "tests failing"
    python run_tests.py --quick -k foo   # extra args after the profile go to pytest
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Subprojects that ship pytest suites. Each runs in its own process with cwd set
# to its own directory, so its conftest.py / pyproject.toml resolve exactly as in
# a standalone `pytest` run.
SUITES = ["mcp-council", "claude-agent-server", "codex-agent-server", "bench", "tools"]

# Packages to byte-compile in --full (catches syntax errors the test suites
# don't import). The agent-server integration suite is excluded — it's live-only.
COMPILEALL_TARGETS = ["mcp-council", "claude-agent-server", "codex-agent-server", "bench", "tools"]


def _run(cmd: list[str], cwd: Path) -> int:
    return subprocess.run(cmd, cwd=cwd).returncode


def _has_selector(extra: list[str]) -> bool:
    """True only if `extra` contains an actual test SELECTOR: -k/-m (with or
    without an attached value) or a positional node-id/path. A non-selector flag
    like --disable-warnings must NOT count — otherwise rc 5 ("no tests collected")
    would be silently tolerated on a run that was supposed to collect everything."""
    for a in extra:
        if a in ("-k", "-m") or (len(a) > 2 and a.startswith(("-k", "-m"))):
            return True
        if not a.startswith("-") and ("::" in a or "/" in a or a.endswith(".py")):
            return True
    return False


def run_pytest(extra: list[str]) -> list[str]:
    # rc 5 == "no tests collected". Tolerate it only when a real -k/-m/node
    # selector is in effect — then a suite legitimately matching nothing is a
    # skip. For an UNFILTERED run (or one carrying only non-selector flags like
    # --disable-warnings), rc 5 means the suite lost all its tests
    # (file deleted/renamed) and MUST fail loudly instead of reporting success.
    filtered = _has_selector(extra)
    ok_codes = (0, 5) if filtered else (0,)
    failed: list[str] = []
    for suite in SUITES:
        print(f"\n=== pytest: {suite} ===", flush=True)
        rc = _run([sys.executable, "-m", "pytest", "-q", *extra], ROOT / suite)
        if rc == 5 and not filtered:
            print(f"!!! {suite}: no tests collected (rc 5) on an unfiltered run", flush=True)
        if rc not in ok_codes:
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


def run_doctor() -> int:
    """Stdlib-only preflight: is each dev dependency importable in THIS
    interpreter? Uses importlib.util.find_spec so a MISSING module isn't itself
    imported (works before deps are installed). Returns 1 if anything is missing.
    Serves the global-CLAUDE.md note that 'No module named pytest' is an
    environment problem, not a red suite."""
    import importlib.util

    required = ["pytest", "pytest_asyncio", "httpx", "mcp"]
    print(f"python: {sys.version.split()[0]}  ({sys.executable})")
    missing: list[str] = []
    for mod in required:
        ok = importlib.util.find_spec(mod) is not None
        print(f"  {'OK     ' if ok else 'MISSING'} {mod}")
        if not ok:
            missing.append(mod)
    if missing:
        print(f"\nMISSING: {', '.join(missing)} — pip install -r requirements-dev.txt")
        return 1
    print("\nall dev deps present")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    profile = "quick"
    if argv and argv[0] in ("--quick", "--full", "--integration", "--doctor"):
        profile = argv[0][2:]
        argv = argv[1:]
    extra = argv  # forwarded to pytest (quick/full) or the integration suite

    if profile == "doctor":
        return run_doctor()

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
