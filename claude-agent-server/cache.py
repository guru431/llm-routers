"""Response cache для claude-agent-server.

Кэширует ответы claude CLI по ключу (model, system_prompt, prompt). Поскольку
сервер обёртка над `claude -p`, prompt caching уровня Anthropic API (через
`cache_control: ephemeral`) тут неприменим — этот модуль кэширует ПОЛНЫЕ
ответы при идентичных входах.

Выгоды:
- мгновенный отклик при повторных запросах (CLI занимает 5-30s)
- ускоряет retry-сценарии у клиентов (например run-format retry в translate_bot)
- стоимость нулевая (Max подписка flat)

Caller должен решать когда НЕ использовать кэш:
- запросы с tools (нестабильные ответы)
- explicit bypass через `cache: false` в payload
- TODO: запросы с temperature != 0 если когда-то понадобится
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Optional


class ResponseCache:
    """Thread-safe LRU cache с TTL. Используется в claude-agent-server.

    Args:
        max_size: максимум записей в кэше (LRU eviction при превышении).
        ttl_seconds: время жизни записи (после — get возвращает None).
    """

    def __init__(self, max_size: int = 256, ttl_seconds: float = 3600.0):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._data: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(model: Optional[str], system_prompt: Optional[str], prompt: str) -> str:
        """Стабильный ключ через sha256 (защита от случайных коллизий при
        длинных prompts и от хранения чувствительных данных в plain ключе).

        None отличается от пустой строки — это явный contract.
        """
        h = hashlib.sha256()
        for part in (model, system_prompt, prompt):
            # \x1e (Record Separator) — байт-разделитель который не встречается
            # в обычном тексте; защищает от ambiguity ("ab"+"" vs "a"+"b").
            h.update(b"\x1e")
            if part is None:
                h.update(b"\x00")
            else:
                h.update(part.encode("utf-8", errors="replace"))
        return h.hexdigest()

    def get(self, model: Optional[str], system_prompt: Optional[str], prompt: str) -> Optional[str]:
        """Возвращает кэшированный ответ или None если miss/expired."""
        key = self._make_key(model, system_prompt, prompt)
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            timestamp, value = entry
            if now - timestamp > self._ttl:
                # Expired — удаляем
                del self._data[key]
                self._misses += 1
                return None
            # MRU: переносим в конец
            self._data.move_to_end(key)
            self._hits += 1
            return value

    def put(self, model: Optional[str], system_prompt: Optional[str], prompt: str, value: str) -> None:
        """Сохраняет ответ. Перезапись обновляет recency."""
        key = self._make_key(model, system_prompt, prompt)
        now = time.monotonic()
        with self._lock:
            if key in self._data:
                # Перезапись — обновляем значение и переносим в MRU
                self._data.move_to_end(key)
            self._data[key] = (now, value)
            # LRU eviction
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        """Удаляет все записи, hits/misses не сбрасываются."""
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        """Возвращает счётчики для /health endpoint."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._data),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / total) if total else 0.0,
            }
