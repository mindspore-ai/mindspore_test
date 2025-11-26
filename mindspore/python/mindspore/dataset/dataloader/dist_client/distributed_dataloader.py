"""PyTorch-compatible distributed data loader for the client side."""

from __future__ import annotations

import sys
import os

# Enable package-relative imports when executed as a script
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "client"

import time
import math
import socket
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence

try:
    from torch.utils.data import BatchSampler, Dataset, RandomSampler, Sampler, SequentialSampler
    from torch.utils.data._utils.collate import default_collate
except ModuleNotFoundError as exc:  # pragma: no cover - enforced at import time
    raise ImportError("DistributedDataLoader requires PyTorch to be installed") from exc

from .cache import SampleCache, SampleNotReady
from .rpc_adapter import (
    BatchAssignment,
    ClientInfo,
    CoordinatorClient,
    FetchRequest,
    ProcessedSample,
    ServerNodeClient,
    get_rpc_factories,
)
from .dataset import DummyTextDataset




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


    # 这边是生成一个 microbatch_size 大小的 batch，将会通过 
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
    
        self._collate = collate_fn or default_collate

        ### 这里是设置 distributed dataloader 相关逻辑
        self.cache_size = 1000  # 设置缓存大小，可以根据需要调整
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


    # 这个函数是一个 fetch cache 接口
    # 如果本地缓存有缺失，就发起 RPC 调用获取
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
        # 自动向 Coordinator 汇报性能
        # 这里的 server_node_id 就是之前 assign 拿到的 ID
        self._coordinator_client.report_completion(
            node_id=session.assignment.server_node_id,
            latency=latency
        )
        
        
        #samples = session.server_client.fetch(request)
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



### example
if __name__ == "__main__":
    from .rpc_adapter import build_inmemory_rpc, configure_rpc_clients

    # Configure RPC with in-memory mocks for demo purposes
    coordinator_factory, server_factory = build_inmemory_rpc()
    configure_rpc_clients(coordinator_factory, server_factory)

    # 3. Define a Dataset shell that only carries metadata/length info
    class RemoteDataset(DummyTextDataset):
        def __init__(self, size, seq_length):
            super().__init__(size, seq_length)

        def __getitem__(self, index):
            raise RuntimeError(
                "RemoteDataset samples are materialized via DistributedDataLoader RPCs"
            )

    # 4. Run Test
    print("Initializing DistributedDataLoader...")
    # Create a dataset with 20 items
    dataset = RemoteDataset(size=20, seq_length=10)
    
    # Create loader: batch_size=4, microbatch_size=2
    # This means each batch (4 items) is split into 2 microbatches (2 items each)
    loader = DistributedDataLoader(
        dataset, 
        batch_size=4, 
        microbatch_size=2,
        shuffle=False
    )

    print("Starting iteration...")
    for i, micro_batches in enumerate(loader):
        print(f"Batch {i} received. Contains {len(micro_batches)} microbatches.")

        first_micro = micro_batches[0]
        print(f"  First microbatch tensor shape: {first_micro['input_ids'].shape}")

        for mb_idx, micro_batch in enumerate(micro_batches):
            input_ids = micro_batch["input_ids"]
            print(f"  Microbatch {mb_idx}: input_ids shape {input_ids.shape}")
            print(f"    First token ids: {input_ids[0].tolist()}")

    print("Done.")

