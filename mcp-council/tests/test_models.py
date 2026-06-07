"""Tests for the model catalog and resolver."""
import pytest

from models import (
    CATALOG,
    COUNCIL_DEFAULT,
    PRESETS,
    DisabledModelError,
    UnknownModelError,
    UnknownPresetError,
    resolve_member,
    resolve_members,
    resolve_preset,
)


def test_council_default_has_seven_members():
    assert len(COUNCIL_DEFAULT) == 7
    assert "deepseek-pro" in COUNCIL_DEFAULT
    assert "codex" in COUNCIL_DEFAULT
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
    assert cfg["env_key"] == "OPENCODE_GO_KEY"


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


def test_resolve_members_drops_duplicates_preserving_order():
    # Duplicate ids would collide on council pseudonyms; resolver keeps first.
    members = resolve_members(["glm", "kimi", "glm"])
    assert [m["id"] for m in members] == ["glm", "kimi"]


def test_presets_resolve_to_valid_catalog_ids_min_two():
    for name, ids in PRESETS.items():
        assert len(ids) >= 2, f"preset {name} must have >=2 members"
        resolved = resolve_members(resolve_preset(name))  # raises if any id bad/disabled
        assert [m["id"] for m in resolved] == ids


def test_resolve_preset_unknown_raises():
    with pytest.raises(UnknownPresetError):
        resolve_preset("nope")


def test_resolve_preset_returns_copy():
    got = resolve_preset("cheap")
    got.append("MUTATED")
    # Mutating the returned list must not leak into PRESETS or future resolves.
    assert "MUTATED" not in resolve_preset("cheap")
    assert "MUTATED" not in PRESETS["cheap"]
