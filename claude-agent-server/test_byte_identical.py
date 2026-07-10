"""Guard: helpers duplicated across the two standalone agent servers must stay
code-identical (no shared module — extraction is a rejected trade-off, see the
NOTE headers in both server.py). This replaces the manual "apply to both copies"
discipline that already drifted once (the _send_stream Connection header).

Compares ASTs (docstrings stripped, so server-name mentions in docstrings don't
count as drift). Runs in the claude-agent-server suite; finds codex via a repo-
relative path.
"""

import ast
from pathlib import Path

import pytest

CLAUDE = Path(__file__).resolve().parent / "server.py"
CODEX = Path(__file__).resolve().parents[1] / "codex-agent-server" / "server.py"

# Functions that MUST be code-identical in both servers.
IDENTICAL_FUNCS = [
    "_load_dotenv",
    "_child_env_without_secrets",
    "build_tools_system_prompt",
    "parse_tool_calls",
    "extract_content",
    "Handler._send",
    "Handler._send_stream",
]

# Module constants that must match.
IDENTICAL_CONSTS = ["_SECRET_ENV_SUFFIXES", "_SECRET_ENV_SUBSTRINGS"]


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _find_func(tree: ast.Module, name: str):
    if "." in name:
        cls, meth = name.split(".")
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for n in node.body:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == meth:
                        return n
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _dump_no_doc(fn) -> str:
    """ast.dump of a function with its leading docstring stripped."""
    body = fn.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        fn.body = body[1:]
    return ast.dump(fn)


def _find_const(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.dump(node.value)
    return None


@pytest.mark.parametrize("name", IDENTICAL_FUNCS)
def test_shared_helpers_are_code_identical(name):
    a = _find_func(_tree(CLAUDE), name)
    b = _find_func(_tree(CODEX), name)
    assert a is not None, f"{name} missing in claude-agent-server/server.py"
    assert b is not None, f"{name} missing in codex-agent-server/server.py"
    assert _dump_no_doc(a) == _dump_no_doc(b), (
        f"{name} diverged between claude-agent-server and codex-agent-server. "
        "These helpers have no shared module by design — apply the fix to BOTH copies."
    )


@pytest.mark.parametrize("name", IDENTICAL_CONSTS)
def test_shared_constants_are_identical(name):
    a = _find_const(_tree(CLAUDE), name)
    b = _find_const(_tree(CODEX), name)
    assert a is not None and b is not None, f"{name} missing in one server"
    assert a == b, f"{name} diverged between the two servers"
