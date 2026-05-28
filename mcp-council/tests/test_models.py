"""Tests for the model catalog and resolver."""
import pytest

from models import (
    CATALOG,
    COUNCIL_DEFAULT,
    DisabledModelError,
    UnknownModelError,
    resolve_member,
    resolve_members,
)


def test_council_default_has_six_members():
    assert len(COUNCIL_DEFAULT) == 6
    assert "deepseek-pro" in COUNCIL_DEFAULT
    assert "deepseek" not in COUNCIL_DEFAULT  # renamed
    for mid in COUNCIL_DEFAULT:
        assert mid in CATALOG


def test_catalog_has_routine_workers():
    assert "deepseek-flash" in CATALOG
    assert "minimax-direct" in CATALOG
    assert CATALOG["minimax-direct"].get("enabled") is False


def test_resolve_member_returns_cfg_with_id():
    cfg = resolve_member("deepseek-flash")
    assert cfg["id"] == "deepseek-flash"
    assert cfg["model"] == "deepseek-v4-flash"
    assert cfg["env_key"] == "DEEPSEEK_KEY"


def test_resolve_member_unknown_raises():
    with pytest.raises(UnknownModelError) as exc:
        resolve_member("nope")
    msg = str(exc.value)
    assert "nope" in msg
    assert "Available:" in msg


def test_resolve_member_disabled_raises():
    with pytest.raises(DisabledModelError):
        resolve_member("minimax-direct")


def test_resolve_members_default_returns_council():
    members = resolve_members(None)
    assert [m["id"] for m in members] == COUNCIL_DEFAULT


def test_resolve_members_subset_preserves_order():
    members = resolve_members(["qwen", "glm"])
    assert [m["id"] for m in members] == ["qwen", "glm"]


def test_resolve_members_invalid_raises():
    with pytest.raises(UnknownModelError):
        resolve_members(["glm", "nope"])
