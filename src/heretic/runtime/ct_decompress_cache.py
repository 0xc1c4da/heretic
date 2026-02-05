# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import torch


def _env_float(name: str, default: float) -> float:
    try:
        v = os.environ.get(name, "").strip()
        return float(v) if v else float(default)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        v = os.environ.get(name, "").strip()
        return int(v) if v else int(default)
    except Exception:
        return int(default)


def _as_bytes(x: int | float) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _tensor_nbytes(t: torch.Tensor) -> int:
    try:
        return int(t.numel()) * int(t.element_size())
    except Exception:
        return 0


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    inserts: int = 0
    evictions: int = 0
    bytes_cached: int = 0
    bytes_evicted: int = 0


class _DeviceLRU:
    """
    Byte-budgeted LRU cache for a single CUDA device.

    Keys are arbitrary hashables; values are CUDA tensors.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: OrderedDict[Any, torch.Tensor] = OrderedDict()
        self._nbytes: OrderedDict[Any, int] = OrderedDict()
        self.budget_bytes: int = 0
        self.safety_bytes: int = 0
        self.stats = CacheStats()
        self._last_log_t: float = 0.0

    def configure(self, *, budget_bytes: int, safety_bytes: int) -> None:
        with self._lock:
            self.budget_bytes = max(0, int(budget_bytes))
            self.safety_bytes = max(0, int(safety_bytes))
            # If budget shrinks, evict immediately.
            self._evict_until_locked(0)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._nbytes.clear()
            self.stats = CacheStats()

    def get(self, key: Any) -> torch.Tensor | None:
        with self._lock:
            t = self._data.get(key)
            if t is None:
                self.stats.misses += 1
                return None
            # Move to MRU.
            self._data.move_to_end(key, last=True)
            self._nbytes.move_to_end(key, last=True)
            self.stats.hits += 1
            return t

    def maybe_put(
        self,
        *,
        key: Any,
        tensor: torch.Tensor,
        device_index: int,
    ) -> None:
        """
        Insert tensor if it fits within budget and safety constraints.
        """
        if self.budget_bytes <= 0:
            return

        nbytes = _tensor_nbytes(tensor)
        if nbytes <= 0:
            return

        try:
            free_now, total = torch.cuda.mem_get_info(int(device_index))
        except Exception:
            return

        # Do not cache if it would violate the safety headroom.
        if int(free_now) - int(nbytes) < int(self.safety_bytes):
            return

        with self._lock:
            # Another thread might have inserted in the meantime.
            if key in self._data:
                self._data.move_to_end(key, last=True)
                self._nbytes.move_to_end(key, last=True)
                return

            # Evict until it fits.
            self._evict_until_locked(nbytes)

            # If it still doesn't fit, give up.
            current = sum(self._nbytes.values())
            if current + nbytes > int(self.budget_bytes):
                return

            self._data[key] = tensor
            self._nbytes[key] = nbytes
            self.stats.inserts += 1
            self.stats.bytes_cached = current + nbytes

    def _evict_until_locked(self, incoming_nbytes: int) -> None:
        """
        Assumes caller holds lock.
        """
        if self.budget_bytes <= 0:
            self._data.clear()
            self._nbytes.clear()
            self.stats.bytes_cached = 0
            return

        while self._data:
            current = sum(self._nbytes.values())
            if current + int(incoming_nbytes) <= int(self.budget_bytes):
                break
            k, v = self._data.popitem(last=False)
            nb = self._nbytes.pop(k, 0)
            self.stats.evictions += 1
            self.stats.bytes_evicted += int(nb)
            # Allow tensor to be freed.
            del v
        self.stats.bytes_cached = sum(self._nbytes.values())

    def maybe_log(self, logger: Callable[[str], None], *, device_index: int) -> None:
        if os.environ.get("HERETIC_CT_CACHE_STATS", "").strip() != "1":
            return
        interval_s = _env_float("HERETIC_CT_CACHE_STATS_INTERVAL_S", 10.0)
        now = time.time()
        with self._lock:
            if (now - self._last_log_t) < interval_s:
                return
            self._last_log_t = now
            st = self.stats
            logger(
                "* ct_cache "
                f"cuda:{device_index} "
                f"budget={self.budget_bytes/2**30:.1f}GiB "
                f"safety={self.safety_bytes/2**30:.1f}GiB "
                f"cached={st.bytes_cached/2**30:.1f}GiB "
                f"hits={st.hits} misses={st.misses} "
                f"inserts={st.inserts} evictions={st.evictions}"
            )


class DecompressedWeightCache:
    """
    Global registry of per-device LRU caches.

    This is intentionally process-global so it can be used by a forward monkeypatch.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._per_device: dict[int, _DeviceLRU] = {}

    def _get_device_cache(self, device_index: int) -> _DeviceLRU:
        with self._lock:
            c = self._per_device.get(int(device_index))
            if c is None:
                c = _DeviceLRU()
                self._per_device[int(device_index)] = c
            return c

    def clear_all(self) -> None:
        with self._lock:
            for c in self._per_device.values():
                c.clear()

    def configure_from_max_memory(
        self,
        logger: Callable[[str], None],
        *,
        max_memory: dict[int | str, str] | None,
        beta: float | None = None,
    ) -> None:
        """
        Auto-size cache budgets using the user's `max_memory` headroom policy.

        If max_memory is missing or does not include a device entry, budget defaults to 0 (cache disabled)
        to preserve safety.
        """
        if not torch.cuda.is_available():
            return

        beta_f = float(beta) if beta is not None else _env_float("HERETIC_CT_CACHE_BETA", 0.5)
        beta_f = max(0.0, min(1.0, beta_f))

        # Lazy import; available in our env (Accelerate is a dependency).
        try:
            from accelerate.utils.modeling import convert_file_size_to_int
        except Exception:
            convert_file_size_to_int = None  # type: ignore[assignment]

        n = int(torch.cuda.device_count())
        for i in range(n):
            try:
                free_now, total = torch.cuda.mem_get_info(i)
            except Exception:
                continue

            planned = None
            if isinstance(max_memory, dict):
                planned = max_memory.get(i)
                if planned is None:
                    planned = max_memory.get(str(i))

            planned_bytes = None
            if isinstance(planned, str) and callable(convert_file_size_to_int):
                try:
                    planned_bytes = int(convert_file_size_to_int(planned))
                except Exception:
                    planned_bytes = None

            # If user didn't specify max_memory for this device, disable caching by default.
            if planned_bytes is None:
                budget_bytes = 0
                safety_bytes = 0
            else:
                planned_headroom = max(0, int(total) - int(planned_bytes))
                safety_bytes = planned_headroom
                usable_for_cache = max(0, int(free_now) - int(safety_bytes))
                budget_bytes = int(usable_for_cache * beta_f)

            self._get_device_cache(i).configure(
                budget_bytes=_as_bytes(budget_bytes),
                safety_bytes=_as_bytes(safety_bytes),
            )

        if os.environ.get("HERETIC_CT_CACHE_STATS", "").strip() == "1":
            logger("* ct_cache: configured per-device budgets from max_memory")

    def get(self, *, device_index: int, key: Any) -> torch.Tensor | None:
        return self._get_device_cache(device_index).get(key)

    def maybe_put(self, *, device_index: int, key: Any, tensor: torch.Tensor) -> None:
        self._get_device_cache(device_index).maybe_put(
            key=key, tensor=tensor, device_index=device_index
        )

    def maybe_log(self, logger: Callable[[str], None], *, device_index: int) -> None:
        self._get_device_cache(device_index).maybe_log(logger, device_index=device_index)


GLOBAL_CT_CACHE = DecompressedWeightCache()

