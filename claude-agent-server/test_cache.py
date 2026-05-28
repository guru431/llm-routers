"""Unit tests for ResponseCache (TTL + LRU)."""

import time
import threading
import pytest


def test_miss_then_hit():
    from cache import ResponseCache
    c = ResponseCache(max_size=10, ttl_seconds=60)
    assert c.get("model-a", "sys", "prompt-1") is None
    c.put("model-a", "sys", "prompt-1", "response-1")
    assert c.get("model-a", "sys", "prompt-1") == "response-1"


def test_different_keys_dont_collide():
    from cache import ResponseCache
    c = ResponseCache(max_size=10, ttl_seconds=60)
    c.put("model-a", "sys", "prompt-1", "a1")
    c.put("model-b", "sys", "prompt-1", "b1")
    c.put("model-a", "sys2", "prompt-1", "a2-sys2")
    c.put("model-a", "sys", "prompt-2", "a3")
    assert c.get("model-a", "sys", "prompt-1") == "a1"
    assert c.get("model-b", "sys", "prompt-1") == "b1"
    assert c.get("model-a", "sys2", "prompt-1") == "a2-sys2"
    assert c.get("model-a", "sys", "prompt-2") == "a3"


def test_none_system_prompt_treated_consistently():
    from cache import ResponseCache
    c = ResponseCache(max_size=10, ttl_seconds=60)
    c.put("m", None, "p", "r")
    assert c.get("m", None, "p") == "r"
    # Empty string и None — разные ключи (явное намерение caller'а)
    assert c.get("m", "", "p") is None


def test_lru_eviction():
    from cache import ResponseCache
    c = ResponseCache(max_size=3, ttl_seconds=60)
    c.put("m", "s", "p1", "r1")
    c.put("m", "s", "p2", "r2")
    c.put("m", "s", "p3", "r3")
    # Touch p1 чтобы он стал most-recently-used
    assert c.get("m", "s", "p1") == "r1"
    # Добавляем p4 — должен вытеснить p2 (least recently used)
    c.put("m", "s", "p4", "r4")
    assert c.get("m", "s", "p1") == "r1"  # MRU
    assert c.get("m", "s", "p2") is None  # вытеснен
    assert c.get("m", "s", "p3") == "r3"
    assert c.get("m", "s", "p4") == "r4"


def test_ttl_expiration():
    from cache import ResponseCache
    c = ResponseCache(max_size=10, ttl_seconds=0.1)  # 100ms TTL для быстрого теста
    c.put("m", "s", "p", "r")
    assert c.get("m", "s", "p") == "r"
    time.sleep(0.15)
    assert c.get("m", "s", "p") is None


def test_overwrite_updates_value_and_recency():
    from cache import ResponseCache
    c = ResponseCache(max_size=2, ttl_seconds=60)
    c.put("m", "s", "p1", "r1")
    c.put("m", "s", "p2", "r2")
    # Перезаписываем p1 — он становится MRU
    c.put("m", "s", "p1", "r1_new")
    # Добавляем p3 — вытесняется p2 (now LRU), не p1
    c.put("m", "s", "p3", "r3")
    assert c.get("m", "s", "p1") == "r1_new"
    assert c.get("m", "s", "p2") is None
    assert c.get("m", "s", "p3") == "r3"


def test_stats_counters():
    from cache import ResponseCache
    c = ResponseCache(max_size=10, ttl_seconds=60)
    c.put("m", "s", "p", "r")
    c.get("m", "s", "p")  # hit
    c.get("m", "s", "p")  # hit
    c.get("m", "s", "missing")  # miss
    stats = c.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["size"] == 1
    assert stats["max_size"] == 10
    assert stats["hit_rate"] == pytest.approx(2 / 3, rel=0.01)


def test_clear():
    from cache import ResponseCache
    c = ResponseCache(max_size=10, ttl_seconds=60)
    c.put("m", "s", "p1", "r1")
    c.put("m", "s", "p2", "r2")
    c.clear()
    assert c.get("m", "s", "p1") is None
    assert c.get("m", "s", "p2") is None
    stats = c.stats()
    assert stats["size"] == 0


def test_thread_safety():
    """Многопоточные put/get не падают и не теряют данные (smoke test)."""
    from cache import ResponseCache
    c = ResponseCache(max_size=1000, ttl_seconds=60)
    errors = []

    def worker(i):
        try:
            for j in range(50):
                key = f"prompt-{i}-{j}"
                c.put("m", "s", key, f"r{i}-{j}")
                assert c.get("m", "s", key) == f"r{i}-{j}"
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"Thread errors: {errors}"


def test_expired_entry_evicted_on_access():
    """Просроченная запись не возвращается и удаляется из size."""
    from cache import ResponseCache
    c = ResponseCache(max_size=10, ttl_seconds=0.1)
    c.put("m", "s", "p", "r")
    assert c.stats()["size"] == 1
    time.sleep(0.15)
    assert c.get("m", "s", "p") is None
    # После expired get — size должен уменьшиться
    assert c.stats()["size"] == 0
