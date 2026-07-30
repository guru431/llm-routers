"""Проверка свежести каталога моделей: что провайдер уже отдаёт, а мы ещё не зовём
(и наоборот — что мы зовём, а провайдер больше не отдаёт).

Зачем: новые поколения моделей выходят чаще, чем кто-либо перечитывает
`mcp-council/models.py`. Обнаружилось на практике — kimi k3, gpt-5.6 и opus 5
вышли за месяц, а конфиги продолжали звать k2.7-code / gpt-5.5 / opus-4-8.

Два класса провайдеров, и проверяются они по-разному:

  * **со списком** (`GET /v1/models`) — OpenCode Go, Helicone Gateway. Здесь
    инвентарь известен точно, поэтому диф двусторонний: и новые версии, и
    пропавшие из выдачи id (легаси-поколение выключили).
  * **без списка** (Codex CLI, Claude CLI — подписочные, каталог наружу не
    отдают) — тут только ПРОБА: генерим правдоподобные следующие имена версий
    и пытаемся сделать самый дешёвый вызов. Успех = модель есть И у аккаунта к
    ней доступ. Это важнее листинга: `gpt-5.6` существует, но на подписке
    ChatGPT отвечает 400 — «есть в природе» и «доступно нам» это разные вещи.

Что НЕ делается намеренно: скрипт ничего не переключает сам. Смена модели
совета меняет состав голосов и квоты (у kimi-k3, например, 2x usage и 220
запросов/5ч против 1150 у k2.7-code) — это решение человека, а не крона.
Скрипт только пишет находку в FINDINGS.md.

Ключи читаются из `secrets/vault.env` (gitignored) или окружения — как в
`bench/judge.py`; путь переопределяется env `VAULT_PATH`.

Usage:
    python tools/model_freshness.py                  # отчёт в stdout
    python tools/model_freshness.py --json           # то же машинно
    python tools/model_freshness.py --no-probe       # только листинги (быстро, без CLI)
    python tools/model_freshness.py --write-findings # + запись в FINDINGS.md (для крона)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "logs" / "model_freshness_state.json"
LOG_PATH = ROOT / "logs" / "model_freshness.log"
FINDINGS_PATH = ROOT / "FINDINGS.md"

# Провайдеры с листингом: id домена (как в models.py::CATALOG "provider") → откуда брать.
#
# `new_families` — сообщать ли о семействах, которых у нас нет вовсе. Для OCG да:
# там выдача = каталог подписки, и grok-4.5 иначе не всплывёт. Для Helicone нет:
# это маркетплейс на сотню моделей всех вендоров, откуда мы намеренно берём одну
# (gemini) — «новое семейство» там означало бы 100 строк шума на каждом прогоне.
LISTING_PROVIDERS = {
    "opencode-go": {"models_url": "https://opencode.ai/zen/go/v1/models",
                    "env_key": "OPENCODE_GO_KEY", "new_families": True},
    "helicone": {"models_url": "https://ai-gateway.helicone.ai/v1/models",
                 "env_key": "HELICONE_GATEWAY_KEY", "new_families": False},
}

# Провайдеры без листинга — проверяются пробой через локальный CLI.
# `argv` — шаблон; `{model}` подставляется. Успех определяется по коду возврата:
# оба CLI отдают 1 на неизвестной/недоступной модели (проверено 2026-07-30).
PROBE_PROVIDERS = {
    "codex-agent": {
        "argv": ["codex", "exec", "--model", "{model}", "--ignore-user-config",
                 "--sandbox", "read-only", "--skip-git-repo-check", "ok"],
        "timeout": 240,
    },
    "claude-agent": {
        "argv": ["claude", "-p", "--model", "{model}", "ok"],
        "timeout": 240,
    },
}

_VERSION_RE = re.compile(r"^(?P<family>.*?)(?P<version>\d+(?:[.\-]\d+)*)(?P<suffix>.*)$")


# ── разбор имён моделей ────────────────────────────────────────────────────

def parse_model_id(model_id: str) -> tuple[str, tuple[int, ...], str, str]:
    """('kimi-k2.7-code') → ('kimi-k', (2, 7), '-code', '.').

    Семейство — всё до ПЕРВОЙ группы цифр, версия — сама группа (точки и дефисы
    внутри неё считаются разделителями), дальше суффикс и использованный
    разделитель (нужен, чтобы генерить кандидатов в том же стиле: claude-opus-5-1
    против gpt-5.6). Модель без цифр вообще → версия (), сравнение по ней всегда
    ложно, что и нужно."""
    m = _VERSION_RE.match(model_id)
    if not m:
        return model_id, (), "", "."
    raw = m.group("version")
    sep = "." if "." in raw else ("-" if "-" in raw else ".")
    parts = tuple(int(x) for x in re.split(r"[.\-]", raw))
    return m.group("family"), parts, m.group("suffix"), sep


def is_newer(candidate: tuple[int, ...], current: tuple[int, ...]) -> bool:
    """Сравнение версий покомпонентно, короткая добивается нулями:
    (3,) > (2,7) — True, (2,7) > (2,7,1) — False."""
    if not candidate or not current:
        return False
    n = max(len(candidate), len(current))
    a = candidate + (0,) * (n - len(candidate))
    b = current + (0,) * (n - len(current))
    return a > b


def diff_listing(configured: list[str], available: list[str]) -> dict:
    """Диф конфигурации против инвентаря провайдера.

    newer        — у семейства, которое мы зовём, есть версия старше нашей;
    new_family   — провайдер отдаёт семейство, которого у нас нет вообще;
    disappeared  — мы зовём id, которого в выдаче больше нет.

    `new_family` намеренно не молчит: kimi-k3 нашёлся бы и по `newer`, а вот
    grok-4.5 — только так (семейства grok у нас не было).
    """
    conf_by_family: dict[str, list[tuple[tuple[int, ...], str]]] = {}
    for cid in configured:
        fam, ver, _suffix, _sep = parse_model_id(cid)
        conf_by_family.setdefault(fam, []).append((ver, cid))

    newer, new_family = [], []
    for aid in available:
        if aid in configured:
            continue
        fam, ver, _suffix, _sep = parse_model_id(aid)
        known = conf_by_family.get(fam)
        if known is None:
            new_family.append({"available": aid, "family": fam})
            continue
        best_ver, best_id = max(known)
        if is_newer(ver, best_ver):
            newer.append({"available": aid, "configured": best_id, "family": fam})

    disappeared = [c for c in configured if c not in available]
    return {"newer": newer, "new_family": new_family, "disappeared": disappeared}


def newest_per_family(model_ids: list[str]) -> list[str]:
    """По одному самому свежему id на семейство — от него и пляшем кандидатами.
    Пробы стоят запуска CLI, а от `claude-opus-4-8` и `claude-opus-5` кандидаты
    почти совпадают."""
    best: dict[str, tuple[tuple[int, ...], str]] = {}
    for mid in model_ids:
        fam, ver, _suffix, _sep = parse_model_id(mid)
        if fam not in best or ver > best[fam][0]:
            best[fam] = (ver, mid)
    return [mid for _ver, mid in best.values()]


def bump_candidates(model_id: str) -> list[str]:
    """Правдоподобные имена следующей версии — для провайдеров без листинга.

    'claude-opus-5' → ['claude-opus-6', 'claude-opus-5-1', 'claude-opus-5.1']
    'gpt-5.6-sol'   → ['gpt-5.7-sol', 'gpt-6-sol', 'gpt-6.0-sol']
    Больше трёх не генерим: каждая проба — реальный запуск CLI.

    У односоставной версии разделитель взять неоткуда (в самом id его нет), а
    вендоры пишут по-разному — Anthropic `claude-opus-4-8`, OpenAI `gpt-5.6`.
    Поэтому там пробуем обе формы, а не угадываем одну.
    """
    family, ver, suffix, sep = parse_model_id(model_id)
    if not ver:
        return []
    out = []
    if len(ver) >= 2:
        out.append(f"{family}{ver[0]}{sep}{ver[1] + 1}{suffix}")
        out.append(f"{family}{ver[0] + 1}{suffix}")
        out.append(f"{family}{ver[0] + 1}{sep}0{suffix}")
    else:
        out.append(f"{family}{ver[0] + 1}{suffix}")
        out.append(f"{family}{ver[0]}-1{suffix}")
        out.append(f"{family}{ver[0]}.1{suffix}")
    seen, uniq = set(), []
    for c in out:
        if c != model_id and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq[:3]


# ── источники конфигурации ─────────────────────────────────────────────────

def load_vault() -> dict[str, str]:
    """secrets/vault.env (gitignored) — тот же формат, что читает bench/judge.py."""
    path = Path(os.environ.get("VAULT_PATH") or (ROOT / "secrets" / "vault.env"))
    env: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def load_catalog() -> dict[str, list[str]]:
    """{provider_domain: [model ids]} из mcp-council/models.py::CATALOG.

    Импорт по пути: mcp-council не пакет (дефис в имени), поэтому importlib, а не
    обычный import."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_council_models", ROOT / "mcp-council" / "models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    by_provider: dict[str, list[str]] = {}
    for cfg in mod.CATALOG.values():
        if cfg.get("enabled") is False:
            continue
        by_provider.setdefault(cfg.get("provider", "?"), []).append(cfg["model"])
    return by_provider


def load_bench_models() -> dict[str, list[str]]:
    """{provider: [model ids]} из bench/models.json (только не-skip строки)."""
    data = json.loads((ROOT / "bench" / "models.json").read_text(encoding="utf-8"))
    by_provider: dict[str, list[str]] = {}
    for m in data["models"]:
        if m.get("skip_reason"):
            continue
        by_provider.setdefault(m["provider"], []).append(m["model"])
    return by_provider


# ── обращения к провайдерам ────────────────────────────────────────────────

def fetch_listing(url: str, api_key: str | None, timeout: float = 30.0) -> list[str]:
    """GET /v1/models → список id. httpx, а не urllib: Cloudflare перед OCG
    отдаёт 403 (error 1010) на дефолтный UA urllib."""
    import httpx

    headers = {"User-Agent": "llm-routers-model-freshness/1"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = httpx.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    return [m["id"] for m in payload.get("data", payload.get("models", [])) if m.get("id")]


def probe_model(provider: str, model_id: str) -> bool:
    """Один самый дешёвый вызов CLI. True = модель есть и доступна аккаунту."""
    cfg = PROBE_PROVIDERS[provider]
    argv = [a.format(model=model_id) for a in cfg["argv"]]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=cfg["timeout"], stdin=subprocess.DEVNULL)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return proc.returncode == 0


# ── FINDINGS ───────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def unreported(report: dict, state: dict) -> list[str]:
    """Ключи находок, которых ещё не было в прошлых прогонах. Без этого крон
    каждую неделю заводил бы одну и ту же запись в FINDINGS."""
    seen = set(state.get("reported", []))
    keys = []
    for provider, res in report.items():
        for item in res.get("newer", []):
            keys.append(f"{provider}:newer:{item['available']}")
        for item in res.get("new_family", []):
            keys.append(f"{provider}:new:{item['available']}")
        for item in res.get("disappeared", []):
            keys.append(f"{provider}:gone:{item}")
        for item in res.get("probe_hits", []):
            keys.append(f"{provider}:probe:{item}")
    return [k for k in keys if k not in seen]


def render_finding(report: dict, fresh_keys: list[str], today: str) -> str:
    """P3-запись в формате FINDINGS.md (поля английские — это канон)."""
    lines = [f"## {today} · Каталог моделей разошёлся с провайдерами [P3]",
             "**Context:** `tools/model_freshness.py` (weekly cron)",
             "**What:** " + "; ".join(fresh_keys[:12]) +
             ("; …" if len(fresh_keys) > 12 else ""),
             "**Proposal:** сверить с лимитами подписки (у новых поколений бывает "
             "другой вес запроса), затем обновить `mcp-council/models.py::CATALOG`, "
             "`bench/models.json` и каталог CCR; исчезнувшие id удалить.",
             "**Status:** open", ""]
    return "\n".join(lines)


def write_finding(entry: str, path: Path = FINDINGS_PATH) -> None:
    """Вставка НАД первой существующей записью (новые записи сверху), шапка
    сохраняется. Файла нет → создаётся с шапкой."""
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = ("# Findings — llm_routers\n"
                "Побочные находки, только `open`. Ревизия: MonthlyStratReview 1-го числа. "
                "Stale >90 дней → alert.\nНовые записи сверху. Выполненные — удаляются, "
                "отклонённые — в FINDINGS-archive.md.\n\n")
    idx = text.find("\n## ")
    if idx == -1:
        new_text = text.rstrip("\n") + "\n\n" + entry
    else:
        new_text = text[:idx + 1] + entry + text[idx + 1:]
    path.write_text(new_text, encoding="utf-8")


# ── основной прогон ────────────────────────────────────────────────────────

def build_report(probe: bool = True) -> dict:
    vault = load_vault()
    catalog = load_catalog()
    bench = load_bench_models()
    # bench использует свои имена провайдеров (opencode_go), council — свои
    # (opencode-go): нормализуем к домену council'а.
    alias = {"opencode_go": "opencode-go", "helicone": "helicone",
             "claude_agent": "claude-agent", "codex_agent": "codex-agent"}
    configured: dict[str, list[str]] = {k: list(v) for k, v in catalog.items()}
    for prov, models in bench.items():
        key = alias.get(prov, prov)
        for m in models:
            configured.setdefault(key, [])
            if m not in configured[key]:
                configured[key].append(m)

    report: dict[str, dict] = {}
    for provider, cfg in LISTING_PROVIDERS.items():
        key = os.environ.get(cfg["env_key"]) or vault.get(cfg["env_key"])
        if not key:
            report[provider] = {"error": f"нет ключа {cfg['env_key']}"}
            continue
        try:
            available = fetch_listing(cfg["models_url"], key)
        except Exception as exc:                       # noqa: BLE001 — отчёт, не падение
            report[provider] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        res = diff_listing(configured.get(provider, []), available)
        if not cfg.get("new_families", True):
            res["new_family"] = []
        res["available_count"] = len(available)
        report[provider] = res

    for provider in PROBE_PROVIDERS:
        if not probe:
            report[provider] = {"skipped": "--no-probe"}
            continue
        conf = configured.get(provider, [])
        hits, tried = [], []
        for model_id in newest_per_family(conf):
            for cand in bump_candidates(model_id):
                # кандидат, который уже в конфиге, — не находка (иначе
                # claude-opus-4-8 «открыл» бы claude-opus-5, который мы и так зовём)
                if cand in tried or cand in conf:
                    continue
                tried.append(cand)
                if probe_model(provider, cand):
                    hits.append(cand)
        report[provider] = {"probe_hits": hits, "probed": tried}
    return report


def render_text(report: dict) -> str:
    out = []
    for provider, res in sorted(report.items()):
        out.append(f"[{provider}]")
        if "error" in res:
            out.append(f"  ! {res['error']}")
        if "skipped" in res:
            out.append(f"  - пропущено ({res['skipped']})")
        for item in res.get("newer", []):
            out.append(f"  ↑ новее: {item['available']}  (используем {item['configured']})")
        for item in res.get("new_family", []):
            out.append(f"  + новое семейство: {item['available']}")
        for item in res.get("disappeared", []):
            out.append(f"  × пропало из выдачи: {item}")
        for item in res.get("probe_hits", []):
            out.append(f"  ↑ проба удалась: {item}")
        if res.get("probed") and not res.get("probe_hits"):
            out.append(f"  = пробы без попаданий: {', '.join(res['probed'])}")
        if not any(k in res for k in ("error", "skipped")) and not (
                res.get("newer") or res.get("new_family") or res.get("disappeared")
                or res.get("probe_hits")):
            out.append("  = расхождений нет")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true", help="машинный вывод")
    ap.add_argument("--no-probe", action="store_true",
                    help="не запускать CLI-пробы (быстро; только листинги)")
    ap.add_argument("--write-findings", action="store_true",
                    help="дописать P3-запись в FINDINGS.md, если есть новое")
    args = ap.parse_args(argv)

    # Крон пишет stdout в файл под Windows-локалью (cp1251) — без этого
    # кириллица в отчёте превращается в UnicodeEncodeError или в кракозябры.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    report = build_report(probe=not args.no_probe)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json
          else render_text(report))

    if args.write_findings:
        # Крон запускает нас скрытым окном без redirect'а — сохраняем отчёт сами,
        # иначе после ночного прогона нечего смотреть, кроме факта записи в FINDINGS.
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(f"=== {date.today().isoformat()}\n{render_text(report)}\n")
        except OSError as exc:                         # noqa: BLE001 — лог не критичен
            print(f"log write failed: {exc}", file=sys.stderr)

        state = load_state()
        fresh = unreported(report, state)
        if fresh:
            write_finding(render_finding(report, fresh, date.today().isoformat()))
            state["reported"] = sorted(set(state.get("reported", [])) | set(fresh))
            state["last_run"] = date.today().isoformat()
            save_state(state)
            print(f"\nFINDINGS.md: добавлена запись ({len(fresh)} нов.)", file=sys.stderr)
        else:
            state["last_run"] = date.today().isoformat()
            save_state(state)
            print("\nFINDINGS.md: нового нет", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
