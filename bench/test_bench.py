"""Offline tests for the bench runner/judge/report (no network, no LLM).

Covers the findings resolved on 2026-07-25: a truncated stream must not count as
a completed response, a run must carry a real lifecycle, repeats must be judged
individually, and the deterministic gate must check VALUES rather than shape.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _store  # noqa: E402
import judge  # noqa: E402
import report  # noqa: E402
import run as bench_run  # noqa: E402


# ── streams must reach their own terminal marker (F: truncated = success) ────

class _FakeResponse:
    def __init__(self, lines, status_code=200, headers=None):
        self._lines = lines
        self.status_code = status_code
        self.headers = headers or {"content-type": "text/event-stream"}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        yield from self._lines

    def read(self):
        return b""


class _FakeClient:
    def __init__(self, lines, headers=None):
        self._lines = lines
        self._headers = headers

    def stream(self, *a, **k):
        return _FakeResponse(self._lines, headers=self._headers)


def test_openai_partial_text_without_done_is_an_error():
    """One chunk then EOF used to be recorded as a clean success because the old
    guard only fired when the text was empty."""
    chunk = json.dumps({"choices": [{"delta": {"content": "half an ans"}}]})
    res = bench_run.call_openai(_FakeClient([f"data: {chunk}"]), "http://x", "m", "s", "u", None)
    assert res["text"] == "half an ans"          # partial text kept for diagnosis
    assert res["error"] and "[DONE]" in res["error"]


def test_openai_done_marker_is_success():
    chunk = json.dumps({"choices": [{"delta": {"content": "full"}}]})
    res = bench_run.call_openai(
        _FakeClient([f"data: {chunk}", "data: [DONE]"]), "http://x", "m", "s", "u", None)
    assert res["error"] is None and res["text"] == "full"


def test_openai_finish_reason_is_success():
    chunk = json.dumps({"choices": [{"delta": {"content": "full"}, "finish_reason": "stop"}]})
    res = bench_run.call_openai(_FakeClient([f"data: {chunk}"]), "http://x", "m", "s", "u", None)
    assert res["error"] is None


def test_gemini_requires_finish_reason():
    """The Gemini path never confirmed a terminal event: ANY EOF looked complete."""
    chunk = json.dumps({"candidates": [{"content": {"parts": [{"text": "partial"}]}}]})
    res = bench_run.call_gemini(_FakeClient([f"data: {chunk}"]), "http://x", "m", "s", "u", "k")
    assert res["text"] == "partial"
    assert res["error"] and "finishReason" in res["error"]

    done = json.dumps({"candidates": [
        {"content": {"parts": [{"text": "full"}]}, "finishReason": "STOP"}]})
    ok = bench_run.call_gemini(_FakeClient([f"data: {done}"]), "http://x", "m", "s", "u", "k")
    assert ok["error"] is None

    cut = json.dumps({"candidates": [
        {"content": {"parts": [{"text": "cut"}]}, "finishReason": "MAX_TOKENS"}]})
    truncated = bench_run.call_gemini(_FakeClient([f"data: {cut}"]), "http://x", "m", "s", "u", "k")
    assert truncated["error"] and "MAX_TOKENS" in truncated["error"]


def test_ollama_requires_done_true():
    line = json.dumps({"message": {"content": "partial"}})
    res = bench_run.call_ollama(_FakeClient([line]), "http://x", "m", "s", "u")
    assert res["text"] == "partial"
    assert res["error"] and "done=true" in res["error"]

    done = json.dumps({"message": {"content": "full"}, "done": True, "eval_count": 3})
    ok = bench_run.call_ollama(_FakeClient([done]), "http://x", "m", "s", "u")
    assert ok["error"] is None and ok["tok_out"] == 3


# ── run lifecycle (F: `latest` can publish an unfinished run) ────────────────

def _write_manifest(run_dir: Path, **fields):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(fields), encoding="utf-8")


def test_run_status_distinguishes_partial_from_complete(tmp_path):
    partial = tmp_path / "r1"
    _write_manifest(partial, run_id="r1", status="started",
                    expected_cells=8, completed_cells=3)
    assert _store.run_status(partial)["complete"] is False

    full = tmp_path / "r2"
    _write_manifest(full, run_id="r2", status="completed",
                    expected_cells=8, completed_cells=8)
    assert _store.run_status(full)["complete"] is True

    # A legacy manifest with no lifecycle fields is NOT assumed complete.
    legacy = tmp_path / "r3"
    _write_manifest(legacy, run_id="r3")
    assert _store.run_status(legacy)["complete"] is False


def test_resolve_run_dir_prefers_the_complete_pointer(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    (runs / "partial").mkdir(parents=True)
    (runs / "done").mkdir(parents=True)
    monkeypatch.setattr(_store, "RUNS_DIR", runs)
    monkeypatch.setattr(_store, "LATEST_TXT", runs / "latest.txt")
    monkeypatch.setattr(_store, "LATEST_COMPLETE_TXT", runs / "latest-complete.txt")
    (runs / "latest.txt").write_text("partial", encoding="utf-8")
    (runs / "latest-complete.txt").write_text("done", encoding="utf-8")
    assert _store.resolve_run_dir().name == "done"
    # …and without a complete pointer it still falls back (callers warn).
    (runs / "latest-complete.txt").unlink()
    assert _store.resolve_run_dir().name == "partial"


# ── deterministic judge must check VALUES, not just shape ───────────────────

_T4 = {
    "id": "T4_json_extract",
    "category": "structured",
    "expected": {"fields": {
        "person": {"any_of": ["иван", "петров"]},
        "date": {"any_of": ["21", "мая"]},
        "time": {"any_of": ["17:30"]},
        "action": {"any_of": ["позвон", "договор"]},
    }},
}


def test_t4_correct_values_score_five():
    text = json.dumps({"person": "Иван Петров", "date": "21 мая",
                       "time": "17:30", "action": "позвонить"}, ensure_ascii=False)
    assert judge.deterministic_score(_T4, text, exec_code=False) == (5, True)


def test_t4_right_shape_wrong_values_is_not_a_pass():
    """All four keys present but every value wrong used to score a perfect 5/5."""
    text = json.dumps({"person": "Барак Обама", "date": "3 января",
                       "time": "09:00", "action": "выгулять собаку"}, ensure_ascii=False)
    score, passed = judge.deterministic_score(_T4, text, exec_code=False)
    assert passed is False
    assert score <= 1


def test_t4_one_wrong_field_is_partial():
    text = json.dumps({"person": "Иван Петров", "date": "21 мая",
                       "time": "09:00", "action": "позвонить"}, ensure_ascii=False)
    score, passed = judge.deterministic_score(_T4, text, exec_code=False)
    assert (score, passed) == (3, False)


_SUM = {"id": "T2_yt_summary_en", "category": "summarization",
        "system": "Summarize.", "user": "TRANSCRIPT: ...",
        "expected": {"bullets": 5}}


def test_summarization_wrong_bullet_count_defers_and_caps():
    """A single nonsense bullet used to be forced to exactly 3/5 without ever
    reaching the semantic judge."""
    text = "- only one bullet, and it is nonsense"
    assert judge.deterministic_score(_SUM, text, exec_code=False) is None
    cap = judge.format_cap(_SUM, text)
    assert cap is not None and cap[0] == 3


def test_summarization_exact_count_has_no_cap():
    text = "\n".join(f"- point {i}" for i in range(5))
    assert judge.format_cap(_SUM, text) is None


def test_score_pair_caps_llm_score_on_format_miss(monkeypatch):
    monkeypatch.setattr(judge, "JUDGE_MODELS", ["fake-judge"])
    monkeypatch.setattr(judge, "call_judge",
                        lambda prompt, model, bypass_cache=False: {"score": 5, "reason": "great"})
    out = judge.score_pair(_SUM, "- one bullet", exec_code=False, bypass_cache=False)
    assert out["score"] == 3
    assert "capped" in out["reason"]


# ── repeats are judged and aggregated individually ──────────────────────────

def test_load_run_keeps_every_repeat_score(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "m1.jsonl").write_text("\n".join(
        json.dumps({"task_id": "T1", "repeat_idx": i, "text": "x",
                    "total_s": 1.0 + i, "ttft_s": 0.5, "ts": "2026-07-25T00:00:00"})
        for i in range(3)
    ), encoding="utf-8")
    (run_dir / "_judge.jsonl").write_text("\n".join(
        json.dumps({"model_id": "m1", "task_id": "T1", "repeat_idx": i, "score": s})
        for i, s in enumerate([3, 5, 4])
    ), encoding="utf-8")
    _recs, _last, judges, _mf, _rd = report.load_run(str(run_dir))
    assert judges[("m1", "T1")]["scores"] == [3, 5, 4]
    assert judges[("m1", "T1")]["last"]["score"] == 4


def test_load_judged_keys_include_repeat(tmp_path):
    """Without repeat_idx in the key, judging repeat #1 marked #0..#N as done."""
    jf = tmp_path / "_judge.jsonl"
    jf.write_text(json.dumps(
        {"model_id": "m1", "task_id": "T1", "repeat_idx": 0, "score": 4,
         "resp_hash": "abc"}) + "\n", encoding="utf-8")
    judged = judge.load_judged(jf)
    assert ("m1", "T1", 0, "abc") in judged
    assert ("m1", "T1", 1, "abc") not in judged


# ── adjudication: explicit rounding rule, panel-sized disagreement ──────────

def test_adjudicate_rounds_half_up_not_to_even():
    """round() is banker's rounding: (2,3)→2 but (3,4)→4 — a systematic
    asymmetry on exactly the split pairs a panel exists to resolve."""
    assert judge.adjudicate([2, 3])[0] == 3
    assert judge.adjudicate([3, 4])[0] == 4


def test_adjudicate_disagreement_threshold_scales_with_panel():
    """With 2 judges a spread of 3 on a 0-5 scale almost never fires."""
    assert judge.adjudicate([2, 4])[1] is True      # 2 judges → threshold 2
    assert judge.adjudicate([3, 3, 4])[1] is False  # 3 judges → still 3
    assert judge.adjudicate([1, 3, 4])[1] is True


# ── report: CI matches the statistic shown, baseline uses the same one ──────

def test_quality_ci_is_for_the_mean_of_task_means():
    """The flat bootstrap reported the CI of the MEDIAN of a pooled sample —
    a different estimator, dominated by between-task variance."""
    groups = [[5.0, 5.0], [1.0, 1.0]]  # two tasks, stable within each
    ci = report.bootstrap_ci_task_mean(groups, repeats=2)
    assert ci is not None
    lo, hi = ci
    assert 1.0 <= lo <= 3.0 <= hi <= 5.0
    assert report.quality_from_groups(groups) == 3.0


def test_baseline_and_current_quality_use_one_estimator(tmp_path):
    """Baseline was mean(all scores) while the run was mean(task means): with an
    unequal repeat count per task the ⚠️REGRESSION flag could fire on nothing."""
    judges = {
        ("m1", "T1"): {"scores": [5, 5, 5], "last": {}, "n_repeats": 3},
        ("m1", "T2"): {"scores": [1], "last": {}, "n_repeats": 1},
    }
    groups = report.task_score_groups(judges, "m1", ["T1", "T2"])
    assert groups == [[5, 5, 5], [1]]
    assert report.quality_from_groups(groups) == 3.0  # NOT 16/4 = 4.0


# ── report: judge panel comes from the manifest, not a hardcode ─────────────

def test_self_judged_follows_the_actual_panel():
    claude_model = {"id": "claude-opus-4-7", "model": "claude-opus-4-7",
                    "provider": "claude_agent"}
    glm_model = {"id": "ocg-glm-5", "model": "glm-5", "provider": "opencode_go"}
    assert report.self_judged(claude_model, ["claude-opus-4-8"]) is True
    assert report.self_judged(glm_model, ["claude-opus-4-8"]) is False
    # Panel switched → the flag moves with it.
    assert report.self_judged(claude_model, ["glm-5"]) is False
    assert report.self_judged(glm_model, ["glm-5"]) is True


def test_judge_panel_is_recorded_in_the_manifest(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run1"}), encoding="utf-8")
    _store.update_manifest(run_dir, judge_panel=["a", "b"])
    assert _store.load_manifest(run_dir)["judge_panel"] == ["a", "b"]
    assert _store.load_manifest(run_dir)["run_id"] == "run1"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
