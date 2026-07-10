"""Tests for sandbox.resolve_and_validate — opt-in allow-list root."""

import pytest

from sandbox import (
    SandboxError,
    resolve_and_validate,
    context_roots_configured,
    _CONTEXT_ROOTS_ENV,
    _CONTEXT_FAIL_OPEN_ENV,
)


def test_no_roots_env_rejects_fail_closed(tmp_path, monkeypatch):
    # Fail-closed default: with no allowed roots and no opt-out, context files
    # are refused rather than shipped to a third-party LLM.
    monkeypatch.delenv(_CONTEXT_ROOTS_ENV, raising=False)
    monkeypatch.delenv(_CONTEXT_FAIL_OPEN_ENV, raising=False)
    f = tmp_path / "note.txt"
    f.write_text("hello")
    with pytest.raises(SandboxError, match=_CONTEXT_ROOTS_ENV):
        resolve_and_validate([str(f)])
    # An empty request is still a no-op (no files to guard).
    assert resolve_and_validate([]) == []


def test_fail_open_env_restores_deny_list_only(tmp_path, monkeypatch):
    monkeypatch.delenv(_CONTEXT_ROOTS_ENV, raising=False)
    monkeypatch.setenv(_CONTEXT_FAIL_OPEN_ENV, "1")
    f = tmp_path / "note.txt"
    f.write_text("hello")
    assert resolve_and_validate([str(f)]) == [f.resolve()]


def test_path_inside_allowed_root_passes(tmp_path, monkeypatch):
    monkeypatch.setenv(_CONTEXT_ROOTS_ENV, str(tmp_path))
    f = tmp_path / "sub" / "note.txt"
    f.parent.mkdir()
    f.write_text("hello")
    assert resolve_and_validate([str(f)]) == [f.resolve()]


def test_path_outside_allowed_root_rejected(tmp_path, monkeypatch):
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("data")
    monkeypatch.setenv(_CONTEXT_ROOTS_ENV, str(root))
    with pytest.raises(SandboxError, match="outside allowed roots"):
        resolve_and_validate([str(outside)])


def test_context_roots_configured_reflects_env(tmp_path, monkeypatch):
    monkeypatch.delenv(_CONTEXT_ROOTS_ENV, raising=False)
    assert context_roots_configured() is False
    monkeypatch.setenv(_CONTEXT_ROOTS_ENV, "   ")
    assert context_roots_configured() is False
    monkeypatch.setenv(_CONTEXT_ROOTS_ENV, str(tmp_path))
    assert context_roots_configured() is True
