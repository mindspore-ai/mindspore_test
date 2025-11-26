"""Client-side caching utilities for microbatch data."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from .rpc_adapter import ProcessedSample


class SampleNotReady(KeyError):
    """Raised when a requested sample is not available in the cache."""


@dataclass
class _CacheEntry:
    sample: ProcessedSample


class SampleCache:
    """A simple FIFO cache that stores processed samples keyed by index."""

    def __init__(self, max_size: Optional[int] = None) -> None:
        self._data: "OrderedDict[int, _CacheEntry]" = OrderedDict()
        self._max_size = max_size

    def put(self, sample: ProcessedSample) -> None:
        idx = sample.index
        if idx in self._data:
            self._data[idx] = _CacheEntry(sample)
            self._data.move_to_end(idx)
            return
        if self._max_size is not None and len(self._data) >= self._max_size:
            self._data.popitem(last=False)
        self._data[idx] = _CacheEntry(sample)

    def bulk_put(self, samples: Iterable[ProcessedSample]) -> None:
        for sample in samples:
            self.put(sample)

    def pop(self, index: int) -> ProcessedSample:
        try:
            entry = self._data.pop(index)
        except KeyError as exc:
            raise SampleNotReady(f"Sample {index} is not ready yet") from exc
        return entry.sample

    def has(self, index: int) -> bool:
        return index in self._data

    def clear(self) -> None:
        self._data.clear()

    def snapshot(self) -> Dict[int, ProcessedSample]:
        return {idx: entry.sample for idx, entry in self._data.items()}
