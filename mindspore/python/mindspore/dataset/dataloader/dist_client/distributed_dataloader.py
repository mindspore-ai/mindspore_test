# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""MindSpore-friendly distributed data loader for the client side."""

from __future__ import annotations

import os

import math
import socket
import time
from collections.abc import Mapping, Sequence as SequenceABC
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, List, Optional, Protocol, Sequence

import numpy as np
from mindspore import Tensor, ops

from .cache import SampleCache, SampleNotReady
from .rpc_adapter import (
    BatchAssignment,
    ClientInfo,
    FetchRequest,
    ServerNodeClient,
    get_rpc_factories,
)


class Dataset(Protocol):
    """Minimal dataset protocol compatible with MindSpore tensors."""

    def __len__(self) -> int:  # pragma: no cover - protocol method
        ...

    def __getitem__(self, index: int) -> Any:  # pragma: no cover - protocol method
        ...


class Sampler(Protocol):
    """Protocol for index samplers."""

    def __iter__(self) -> Iterator[int]:  # pragma: no cover - protocol method
        ...

    def __len__(self) -> int:  # pragma: no cover - protocol method
        ...


class SequentialSampler:
    """Yield dataset indices sequentially."""

    def __init__(self, data_source: Dataset) -> None:
        self._length = len(data_source)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._length))

    def __len__(self) -> int:
        return self._length


class RandomSampler:
    """Yield dataset indices in a random order."""

    def __init__(self, data_source: Dataset, seed: Optional[int] = None) -> None:
        self._length = len(data_source)
        self._initial_seed = seed
        self._rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[int]:
        permuted = self._rng.permutation(self._length)
        for idx in permuted:
            yield int(idx)

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        seed = (self._initial_seed or 0) + epoch
        self._rng = np.random.default_rng(seed)


class BatchSampler(Iterable[List[int]]):
    """Group indices from another sampler into fixed-size batches."""

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch: List[int] = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        length = len(self.sampler)
        if self.drop_last:
            return length // self.batch_size
        return math.ceil(length / self.batch_size)


def _default_collate(batch: Sequence[Any]) -> Any:
    """Collate a sequence of samples into MindSpore tensors when possible."""

    if not batch:
        return batch

    elem = batch[0]
    if isinstance(elem, Tensor):
        return ops.stack(batch)
    if isinstance(elem, np.ndarray):
        return Tensor(np.stack(batch))
    if isinstance(elem, (int, float, bool)):
        return Tensor(np.asarray(batch))
    if isinstance(elem, Mapping):
        return {key: _default_collate([d[key] for d in batch]) for key in elem}
    if isinstance(elem, SequenceABC) and not isinstance(elem, (str, bytes)):
        transposed = list(zip(*batch))
        return [_default_collate(list(samples)) for samples in transposed]
    return batch



@dataclass
class _BatchSession:
    assignment: BatchAssignment
    server_client: ServerNodeClient


class _MicrobatchSequence(SequenceABC):
    """Lazy, indexable view over the microbatches in a single batch."""

    def __init__(
        self,
        loader: "DistributedDataLoader",
        session: _BatchSession,
        batch_indices: Sequence[int],
    ) -> None:
        self._loader = loader
        self._session = session
        self._batch_indices = list(batch_indices)
        self._microbatch_size = loader.microbatch_size
        self._materialized: List[Any] = []
        self._num_microbatches = math.ceil(len(self._batch_indices) / self._microbatch_size)

    def __len__(self) -> int:
        return self._num_microbatches

    def __iter__(self) -> Iterator[Any]:
        for idx in range(self._num_microbatches):
            yield self[idx]


    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._num_microbatches))]
        normalized = index
        if normalized < 0:
            normalized += self._num_microbatches
        if normalized < 0 or normalized >= self._num_microbatches:
            raise IndexError("microbatch index out of range")
        self._materialize_until(normalized)
        return self._materialized[normalized]

    def _materialize_until(self, target: int) -> None:
        while len(self._materialized) <= target:
            self._materialize_next()

    def _materialize_next(self) -> None:
        offset = len(self._materialized) * self._microbatch_size
        micro_indices = self._batch_indices[offset : offset + self._microbatch_size]
        if not micro_indices:
            return
        microbatch = self._loader._materialize_microbatch(self._session, micro_indices)
        self._materialized.append(microbatch)


class DistributedDataLoader(Iterable[Any]):
    """Client-side loader that fetches remote data before yielding each batch."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        microbatch_size: int,
        *,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        drop_last: bool = False,
        collate_fn: Optional[Callable[[Sequence[Any]], Any]] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        if microbatch_size > batch_size:
            raise ValueError("microbatch_size cannot exceed batch_size")
        if sampler is not None and shuffle:
            raise ValueError("shuffle cannot be True when a sampler is provided")

        self.dataset = dataset
        self.batch_size = batch_size
        self.microbatch_size = microbatch_size
        coordinator_factory, server_factory = get_rpc_factories()
        self._coordinator_client = coordinator_factory()
        self._server_client_factory = server_factory

        base_sampler = self._resolve_sampler(dataset, sampler, shuffle)
        self._batch_sampler = BatchSampler(base_sampler, batch_size=batch_size, drop_last=drop_last)

        self._collate = collate_fn or _default_collate

        # Configure distributed dataloader cache behavior
        self.cache_size = 1000
        self._cache = SampleCache(max_size=self.cache_size)

        client_id = f"{socket.gethostname()}-{os.getpid()}"
        self._client_info = ClientInfo(
            client_id=client_id,
            batch_size=batch_size,
            microbatch_size=microbatch_size)

    def __iter__(self) -> Iterator[Any]:
        session_id = self._coordinator_client.register_client(self._client_info)
        for batch_indices in self._batch_sampler:
            session = self._open_batch(session_id, batch_indices)
            yield _MicrobatchSequence(self, session, batch_indices)

    def __len__(self) -> int:
        return len(self._batch_sampler)

    @staticmethod
    def _resolve_sampler(
        dataset: Dataset[Any], sampler: Optional[Sampler[int]], shuffle: bool
    ) -> Sampler[int]:
        if sampler is not None:
            return sampler
        if shuffle:
            return RandomSampler(dataset)
        return SequentialSampler(dataset)

    def _open_batch(self, session_id: str, batch_indices: Sequence[int]) -> _BatchSession:
        assignment = self._coordinator_client.assign_server_node(session_id, batch_indices)
        server_client = self._server_client_factory(assignment)
        return _BatchSession(assignment=assignment, server_client=server_client)

    def _ensure_remote_samples(self, session: _BatchSession, indices: Sequence[int]) -> None:
        missing = [idx for idx in indices if not self._cache.has(idx)]
        if not missing:
            return
        request = FetchRequest(
            session_id=session.assignment.session_id,
            server_node_id=session.assignment.server_node_id,
            indices=missing,
            extra=session.assignment.metadata,
        )
        
        start_time = time.time()
        samples = session.server_client.fetch(request)
        end_time = time.time()
        latency = end_time - start_time
        # Report performance metrics back to the coordinator using this server node ID
        self._coordinator_client.report_completion(
            node_id=session.assignment.server_node_id,
            latency=latency
        )

        if not samples:
            raise SampleNotReady("Server node returned no samples for requested indices")
        self._cache.bulk_put(samples)

    def _materialize_microbatch(
        self, session: _BatchSession, indices: Sequence[int]
    ) -> Any:
        self._ensure_remote_samples(session, indices)
        payloads = [self._cache.pop(idx).payload for idx in indices]
        return self._collate(payloads)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self._batch_sampler.sampler, "set_epoch"):
            self._batch_sampler.sampler.set_epoch(epoch)


