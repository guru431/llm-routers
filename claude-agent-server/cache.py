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
        max_bytes: максимум суммарного размера значений в байтах (LRU eviction
            при превышении). Защита от OOM при немногих, но огромных ответах
            (один ответ может быть в сотни KB; 256 таких записей — десятки MB).
    """

    def __init__(self, max_size: int = 256, ttl_seconds: float = 3600.0,
                 max_bytes: int = 64 * 1024 * 1024):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._max_bytes = max_bytes
        # Каждая запись: key -> (timestamp, value, nbytes). Сумму байтов держим
        # в _total_bytes, чтобы не пересчитывать на каждой эвикции.
        self._data: "OrderedDict[str, tuple[float, str, int]]" = OrderedDict()
        self._total_bytes = 0
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
                # surrogatepass (не replace): иначе разные суррогаты/эмодзи
                # схлопываются в U+FFFD → коллизия ключа → чужой кэш-ответ.
                h.update(part.encode("utf-8", errors="surrogatepass"))
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
            timestamp, value, nbytes = entry
            if now - timestamp > self._ttl:
                # Expired — удаляем
                del self._data[key]
                self._total_bytes -= nbytes
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
        nbytes = len(value.encode("utf-8", errors="replace"))
        with self._lock:
            old = self._data.get(key)
            if old is not None:
                # Перезапись — вычитаем старый размер, переносим в MRU
                self._total_bytes -= old[2]
                self._data.move_to_end(key)
            self._data[key] = (now, value, nbytes)
            self._total_bytes += nbytes
            # LRU eviction по числу записей ИЛИ суммарным байтам. Не выселяем
            # единственную (только что добавленную) запись, даже если она одна
            # превышает max_bytes — иначе put стал бы no-op.
            while (len(self._data) > self._max_size
                   or self._total_bytes > self._max_bytes) and len(self._data) > 1:
                _, evicted = self._data.popitem(last=False)
                self._total_bytes -= evicted[2]

    def clear(self) -> None:
        """Удаляет все записи, hits/misses не сбрасываются."""
        with self._lock:
            self._data.clear()
            self._total_bytes = 0

    def stats(self) -> dict:
        """Возвращает счётчики для /health endpoint."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._data),
                "max_size": self._max_size,
                "bytes": self._total_bytes,
                "max_bytes": self._max_bytes,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / total) if total else 0.0,
            }
