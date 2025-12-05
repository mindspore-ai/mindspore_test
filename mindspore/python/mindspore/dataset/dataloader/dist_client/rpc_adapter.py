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

"""Typed RPC shims for the coordinator and server nodes."""

from __future__ import annotations

import io
import pickle
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
from mindspore import Tensor, ops



try:  # Prefer the fully-qualified package when installed inside MindSpore
    from mindspore.dataset.dataloader.dist_rpc.common import RPCMethod
    from mindspore.dataset.dataloader.dist_rpc.client_rpc import (
        CoordinatorRPCClient,
        ServerNodeRPCClient,
    )

except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from dist_rpc.common import RPCMethod
    from dist_rpc.client_rpc import CoordinatorRPCClient, ServerNodeRPCClient



def _slice_value(value: Any, index: int) -> Any:
    """Slice the first dimension of nested payloads."""

    if isinstance(value, Tensor):
        return value[index]
    if isinstance(value, np.ndarray):
        return Tensor(value[index])
    if isinstance(value, list):
        return value[index]
    if isinstance(value, dict):
        return {k: _slice_value(v, index) for k, v in value.items()}
    return value


def _tensor_batch_to_samples(batch: Any) -> List[Any]:
    """Split a batched payload into a list of per-sample payloads."""

    if isinstance(batch, Tensor):
        if batch.ndim == 0:
            raise ValueError("Expected at least 1D tensor for batched samples")
        return [batch[i] for i in range(batch.shape[0])]

    if isinstance(batch, np.ndarray):
        if batch.ndim == 0:
            raise ValueError("Expected at least 1D array for batched samples")
        return [Tensor(batch[i]) for i in range(batch.shape[0])]

    if isinstance(batch, dict):
        keys = list(batch.keys())
        if not keys:
            return []
        batch_size = len(batch[keys[0]])
        samples: List[Dict[str, Any]] = []
        for i in range(batch_size):
            sample = {k: _slice_value(batch[k], i) for k in keys}
            samples.append(sample)
        return samples

    if isinstance(batch, list):
        return batch

    raise TypeError(f"Unsupported batch type: {type(batch)}")


@dataclass(frozen=True)
class ClientInfo:
    client_id: str
    batch_size: int
    microbatch_size: int
    extra: Dict[str, Any] | None = None


@dataclass(frozen=True)
class BatchAssignment:
    session_id: str
    server_node_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FetchRequest:
    session_id: str
    server_node_id: str
    indices: Sequence[int]
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessedSample:
    index: int
    payload: Any
    
    
class CoordinatorClient(Protocol):
    def register_client(self, info: ClientInfo) -> str:
        ...

    def assign_server_node(self, session_id: str, batch_indices: Sequence[int]) -> BatchAssignment:
        ...
        
    def report_completion(self, node_id: str, latency: float) -> None:
        ...


class ServerNodeClient(Protocol):
    def fetch(self, request: FetchRequest) -> Sequence[ProcessedSample]:
        ...


CoordinatorFactory = Callable[[], CoordinatorClient]
ServerClientFactory = Callable[[BatchAssignment], ServerNodeClient]

_RPC_COORDINATOR_FACTORY: Optional[CoordinatorFactory] = None
_RPC_SERVER_FACTORY: Optional[ServerClientFactory] = None


def configure_rpc_clients(
    coordinator_factory: CoordinatorFactory,
    server_client_factory: ServerClientFactory,
) -> None:
    """Register the factories used by DistributedDataLoader."""

    global _RPC_COORDINATOR_FACTORY, _RPC_SERVER_FACTORY
    _RPC_COORDINATOR_FACTORY = coordinator_factory
    _RPC_SERVER_FACTORY = server_client_factory


def get_rpc_factories() -> Tuple[CoordinatorFactory, ServerClientFactory]:
    if _RPC_COORDINATOR_FACTORY is None or _RPC_SERVER_FACTORY is None:
        raise RuntimeError(
            "DistributedDataLoader RPC stack is not configured. Call configure_rpc_clients(...) first."
        )
    return _RPC_COORDINATOR_FACTORY, _RPC_SERVER_FACTORY


class InMemoryCoordinatorClient:
    """A reference coordinator client for local testing."""

    def __init__(self, server_node_id: str = "server-0") -> None:
        self._server_node_id = server_node_id

    def register_client(self, info: ClientInfo) -> str:
        _ = info
        return str(uuid.uuid4())

    def assign_server_node(self, session_id: str, batch_indices: Sequence[int]) -> BatchAssignment:
        _ = batch_indices
        return BatchAssignment(session_id=session_id, server_node_id=self._server_node_id)


def build_inmemory_rpc() -> Tuple[CoordinatorFactory, ServerClientFactory]:
    """Return factories for an in-memory mock coordinator/server pair."""

    class MockCoordinator(CoordinatorClient):
        def register_client(self, info: ClientInfo) -> str:
            _ = info
            return "mock_session"

        def assign_server_node(self, session_id: str, indices: Sequence[int]) -> BatchAssignment:
            _ = indices
            return BatchAssignment(session_id=session_id, server_node_id="mock_server")

    class MockServer(ServerNodeClient):
        def fetch(self, request: FetchRequest) -> Sequence[ProcessedSample]:
            values = []
            for idx in request.indices:
                values.append(Tensor(np.array([idx, idx], dtype=np.int32)))
            batch = ops.stack(values) if values else Tensor(np.empty((0, 2), dtype=np.int32))
            payloads = _tensor_batch_to_samples(batch)
            return [ProcessedSample(index=idx, payload=payload) for idx, payload in zip(request.indices, payloads)]

    return (lambda: MockCoordinator(), lambda _: MockServer())


class RemoteCoordinatorClient(CoordinatorClient):
    """Coordinator client backed by the socket-based RPC implementation."""

    def __init__(self, host: str, port: int, *, timeout: float = 10.0) -> None:
        self._rpc = CoordinatorRPCClient(host, port, timeout=timeout)

    def register_client(self, info: ClientInfo) -> str:
        payload = {
            "client_id": info.client_id,
            "batch_size": info.batch_size,
            "microbatch_size": info.microbatch_size,
            "extra": info.extra or {},
        }
        acknowledged = self._rpc.register_client(payload)
        if not acknowledged:
            raise RuntimeError("Coordinator rejected client registration")
        return info.client_id

    def assign_server_node(self, session_id: str, batch_indices: Sequence[int]) -> BatchAssignment:
        assignment = self._rpc.assign_servernode(session_id, list(batch_indices))
        host = assignment.get("host")
        port = assignment.get("port")
        if host is None or port is None:
            raise ValueError("Coordinator assignment must include host and port")
        server_node_id = assignment.get("server_node_id") or f"{host}:{port}"
        metadata = dict(assignment)
        metadata.setdefault("host", host)
        metadata.setdefault("port", port)
        return BatchAssignment(session_id=session_id, server_node_id=server_node_id, metadata=metadata)
    
    def report_completion(self, node_id: str, latency: float) -> None:
        payload = {"node_id": node_id, "latency": latency}
        # Use the predefined RPC method for reporting latency metrics
        self._rpc._call(RPCMethod.REPORT_COMPLETION, payload)


class RemoteServerNodeClient(ServerNodeClient):
    """Server-node client backed by the socket-based RPC implementation."""

    def __init__(self, host: str, port: int, *, timeout: float = 10.0) -> None:
        self._rpc = ServerNodeRPCClient(host, port, timeout=timeout)

    def fetch(self, request: FetchRequest) -> Sequence[ProcessedSample]:
        client_id = request.extra.get("client_id") if request.extra else request.session_id

        raw_data = self._rpc.fetch(client_id, list(request.indices))
        if isinstance(raw_data, bytes):
            batch = pickle.loads(raw_data)
        elif isinstance(raw_data, Tensor):
            batch = raw_data
        elif isinstance(raw_data, np.ndarray):
            batch = Tensor(raw_data)
        elif isinstance(raw_data, dict):
            batch = raw_data
        elif isinstance(raw_data, list):
            batch = raw_data
        else:
            raise TypeError(f"Unexpected payload type from server: {type(raw_data)}")

        payloads = _tensor_batch_to_samples(batch)
        results = [ProcessedSample(index=idx, payload=payload) for idx, payload in zip(request.indices, payloads)]
        return results


def build_socket_rpc(
    coordinator_host: str,
    coordinator_port: int,
    *,
    timeout: float = 10.0,
) -> Tuple[CoordinatorFactory, ServerClientFactory]:
    """Create RPC factories that talk to the real coordinator/server nodes."""

    def coordinator_factory() -> CoordinatorClient:
        return RemoteCoordinatorClient(coordinator_host, coordinator_port, timeout=timeout)

    def server_factory(assignment: BatchAssignment) -> ServerNodeClient:
        host = assignment.metadata.get("host")
        port = assignment.metadata.get("port")
        if host is None or port is None:
            raise ValueError("BatchAssignment metadata must include host and port")
        return RemoteServerNodeClient(str(host), int(port), timeout=timeout)

    return coordinator_factory, server_factory


