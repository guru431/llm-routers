"""Tests for sandbox.resolve_and_validate — opt-in allow-list root."""

import pytest

from sandbox import SandboxError, resolve_and_validate, _CONTEXT_ROOTS_ENV


def test_no_roots_env_allows_any_nonblacklisted_file(tmp_path, monkeypatch):
    monkeypatch.delenv(_CONTEXT_ROOTS_ENV, raising=False)
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
