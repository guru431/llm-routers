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


def test_read_files_decodes_utf16_le_bom(tmp_path):
    # F20: a UTF-16 LE file (PowerShell 5.1 default) must decode to real text,
    # not UTF-8-mangled replacement chars / interleaved NULs.
    from sandbox import read_files_with_limit
    p = tmp_path / "u16.txt"
    p.write_bytes(b"\xff\xfe" + "Привет, мир".encode("utf-16-le"))
    (_, text), = read_files_with_limit([p])
    assert text == "Привет, мир"
    assert "\x00" not in text and "�" not in text


def test_read_files_decodes_utf8_bom(tmp_path):
    from sandbox import read_files_with_limit
    p = tmp_path / "u8.txt"
    p.write_bytes(b"\xef\xbb\xbf" + "hello".encode("utf-8"))
    (_, text), = read_files_with_limit([p])
    assert text == "hello"


def test_read_rejects_secret_swapped_in_after_validation(tmp_path, monkeypatch):
    """TOCTOU: the validated path is replaced with a private key between
    resolve_and_validate() and the read. The read must reject it — the sandbox
    boundary has to hold for the bytes actually shipped, not just for the object
    that happened to be there at validation time."""
    from sandbox import read_files_with_limit

    monkeypatch.setenv(_CONTEXT_ROOTS_ENV, str(tmp_path))
    f = tmp_path / "note.txt"
    f.write_text("harmless", encoding="utf-8")
    validated = resolve_and_validate([str(f)])

    # …swap happens here… (header assembled at runtime so the source literal
    # doesn't trip the repo's own pre-commit secret scanner)
    header = b"-----BEGIN OPENSSH " + b"PRIVATE KEY-----"
    f.write_bytes(header + b"\nAAAA\n")

    with pytest.raises(SandboxError, match="private-key"):
        read_files_with_limit(validated)


def test_read_rejects_path_moved_outside_allowed_root(tmp_path, monkeypatch):
    """A validated Path object whose target now sits outside the allow-list is
    refused at read time, not silently read."""
    from sandbox import read_files_with_limit

    root = tmp_path / "allowed"
    root.mkdir()
    inside = root / "note.txt"
    inside.write_text("hello", encoding="utf-8")
    monkeypatch.setenv(_CONTEXT_ROOTS_ENV, str(root))
    validated = resolve_and_validate([str(inside)])

    outside = tmp_path / "elsewhere.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(SandboxError, match="outside allowed roots"):
        read_files_with_limit([outside.resolve()])
    # The legitimately-validated path still reads fine.
    assert read_files_with_limit(validated)[0][1] == "hello"


def test_read_rejects_directory(tmp_path):
    from sandbox import read_files_with_limit
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(SandboxError):
        read_files_with_limit([d])
