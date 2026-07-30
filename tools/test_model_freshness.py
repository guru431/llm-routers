"""Тесты чистых функций model_freshness (разбор версий, диф, генерация кандидатов,
дедуп находок, вставка в FINDINGS). Сетевые вызовы и CLI-пробы не трогаются."""
from __future__ import annotations

import model_freshness as mf


# ── разбор имён ────────────────────────────────────────────────────────────

def test_parse_model_id_families_and_versions():
    assert mf.parse_model_id("kimi-k2.7-code")[:3] == ("kimi-k", (2, 7), "-code")
    assert mf.parse_model_id("kimi-k3")[:3] == ("kimi-k", (3,), "")
    assert mf.parse_model_id("gpt-5.6-sol")[:3] == ("gpt-", (5, 6), "-sol")
    assert mf.parse_model_id("claude-opus-4-8")[:3] == ("claude-opus-", (4, 8), "")
    # разделитель запоминается — от него зависит стиль кандидатов
    assert mf.parse_model_id("claude-opus-4-8")[3] == "-"
    assert mf.parse_model_id("gpt-5.6-sol")[3] == "."
    # имя без цифр не ломает разбор
    assert mf.parse_model_id("hy3")[:2] == ("hy", (3,))
    assert mf.parse_model_id("nova")[1] == ()


def test_is_newer_pads_short_versions():
    assert mf.is_newer((3,), (2, 7)) is True
    assert mf.is_newer((2, 7), (3,)) is False
    assert mf.is_newer((5, 6), (5, 5)) is True
    assert mf.is_newer((5,), (5, 1)) is False     # 5.0 < 5.1
    assert mf.is_newer((), (5,)) is False          # без версии сравнивать нечего


# ── диф листинга ───────────────────────────────────────────────────────────

def test_diff_listing_reports_newer_same_family():
    res = mf.diff_listing(["kimi-k2.7-code"], ["kimi-k2.7-code", "kimi-k3", "kimi-k2.5"])
    assert [x["available"] for x in res["newer"]] == ["kimi-k3"]
    assert res["newer"][0]["configured"] == "kimi-k2.7-code"
    assert res["new_family"] == []


def test_diff_listing_reports_unknown_family():
    # grok нашёлся бы ТОЛЬКО так: семейства grok в конфиге нет вообще
    res = mf.diff_listing(["glm-5.2"], ["glm-5.2", "grok-4.5"])
    assert [x["available"] for x in res["new_family"]] == ["grok-4.5"]
    assert res["newer"] == []


def test_listing_providers_declare_new_family_policy():
    # OCG: выдача = каталог подписки, новое семейство там сигнал.
    # Helicone: маркетплейс на сотню моделей — иначе 100 строк шума за прогон.
    assert mf.LISTING_PROVIDERS["opencode-go"]["new_families"] is True
    assert mf.LISTING_PROVIDERS["helicone"]["new_families"] is False


def test_diff_listing_reports_disappeared():
    res = mf.diff_listing(["qwen3.6-plus", "glm-5.2"], ["glm-5.2"])
    assert res["disappeared"] == ["qwen3.6-plus"]


def test_diff_listing_ignores_older_and_equal():
    res = mf.diff_listing(["qwen3.7-plus"], ["qwen3.7-plus", "qwen3.6-plus", "qwen3.5-plus"])
    assert res["newer"] == []
    assert res["new_family"] == []
    assert res["disappeared"] == []


def test_diff_listing_same_version_other_suffix_is_not_newer():
    # qwen3.7-max — не «новее» qwen3.7-plus, это соседний вариант той же версии
    res = mf.diff_listing(["qwen3.7-plus"], ["qwen3.7-plus", "qwen3.7-max"])
    assert res["newer"] == []


# ── кандидаты для проб ─────────────────────────────────────────────────────

def test_bump_candidates_two_part_version():
    cands = mf.bump_candidates("gpt-5.6-sol")
    assert "gpt-5.7-sol" in cands and "gpt-6-sol" in cands
    assert "gpt-5.6-sol" not in cands
    assert len(cands) <= 3


def test_bump_candidates_single_part_tries_both_separators():
    # разделителя в id нет, а вендоры пишут по-разному (claude-opus-4-8 vs gpt-5.6)
    cands = mf.bump_candidates("claude-opus-5")
    assert cands == ["claude-opus-6", "claude-opus-5-1", "claude-opus-5.1"]


def test_bump_candidates_without_version_is_empty():
    assert mf.bump_candidates("nova") == []


def test_newest_per_family_keeps_one_id_per_family():
    got = mf.newest_per_family(
        ["claude-opus-4-8", "claude-opus-5", "claude-sonnet-4-6", "claude-sonnet-5"])
    assert sorted(got) == ["claude-opus-5", "claude-sonnet-5"]


# ── дедуп находок ──────────────────────────────────────────────────────────

def test_unreported_filters_already_seen():
    report = {"opencode-go": {"newer": [{"available": "kimi-k3", "configured": "kimi-k2.7-code"}],
                              "new_family": [{"available": "grok-4.5"}],
                              "disappeared": ["qwen3.6-plus"]}}
    fresh = mf.unreported(report, {"reported": ["opencode-go:newer:kimi-k3"]})
    assert fresh == ["opencode-go:new:grok-4.5", "opencode-go:gone:qwen3.6-plus"]


def test_unreported_counts_probe_hits():
    report = {"claude-agent": {"probe_hits": ["claude-opus-6"], "probed": ["claude-opus-6"]}}
    assert mf.unreported(report, {}) == ["claude-agent:probe:claude-opus-6"]


# ── запись в FINDINGS ──────────────────────────────────────────────────────

def test_write_finding_inserts_above_existing_entries(tmp_path):
    p = tmp_path / "FINDINGS.md"
    p.write_text("# Findings — x\nшапка\n\n## 2026-01-01 · Старое [P3]\n**Status:** open\n",
                 encoding="utf-8")
    mf.write_finding("## 2026-07-30 · Новое [P3]\n**Status:** open\n", path=p)
    text = p.read_text(encoding="utf-8")
    assert text.index("2026-07-30") < text.index("2026-01-01")   # новые записи сверху
    assert text.startswith("# Findings — x")                      # шапка на месте


def test_write_finding_creates_file_with_header(tmp_path):
    p = tmp_path / "FINDINGS.md"
    mf.write_finding("## 2026-07-30 · Новое [P3]\n**Status:** open\n", path=p)
    text = p.read_text(encoding="utf-8")
    assert text.startswith("# Findings — llm_routers")
    assert "2026-07-30" in text


def test_render_finding_has_canonical_fields():
    entry = mf.render_finding({}, ["opencode-go:newer:kimi-k3"], "2026-07-30")
    assert entry.startswith("## 2026-07-30 · ")
    for field in ("**Context:**", "**What:**", "**Proposal:**", "**Status:** open"):
        assert field in entry
    assert "[P3]" in entry
