"""Tests for the model catalog and resolver."""
import pytest

from models import (
    CATALOG,
    COUNCIL_DEFAULT,
    PRESETS,
    DisabledModelError,
    UnknownModelError,
    UnknownPresetError,
    provider_domain,
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
    # "cheap" is a legacy alias for the neutral name — resolve_preset accepts both.
    got = resolve_preset("cheap")
    got.append("MUTATED")
    # Mutating the returned list must not leak into PRESETS or future resolves.
    assert "MUTATED" not in resolve_preset("cheap")
    assert "MUTATED" not in PRESETS["fast-2-single-provider"]


def test_resolve_preset_legacy_aliases():
    # Old quality-implying names still resolve to the neutral presets.
    assert resolve_preset("best") == PRESETS["full"]
    assert resolve_preset("balanced") == PRESETS["diverse-3"]
    assert resolve_preset("cheap") == PRESETS["fast-2-single-provider"]


def test_provider_domain_groups_ocg_and_isolates_independent_backends():
    # The 5 OCG members share one failure/credential domain; gemini (Helicone)
    # and codex (local codex-agent-server) are independent.
    assert provider_domain("glm") == provider_domain("qwen") == "opencode-go"
    assert provider_domain("deepseek-pro") == "opencode-go"
    assert provider_domain("gemini") == "helicone"
    assert provider_domain("codex") == "codex-agent"
    assert len({provider_domain("glm"), provider_domain("gemini"),
                provider_domain("codex")}) == 3
    # Unknown ids (test stubs) map to themselves so each is its own domain.
    assert provider_domain("m1") == "m1"


def test_every_catalog_member_has_a_provider_domain():
    for mid, cfg in CATALOG.items():
        assert cfg.get("provider"), f"{mid} missing provider domain"
