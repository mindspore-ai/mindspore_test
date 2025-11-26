"""Typed RPC shims for the coordinator and server nodes."""

from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple
import io
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from rpc.common import RPCMethod



_start_coordinator = None
_start_servernode = None

try:  # Prefer installed package when available
    from Distdataloader.rpc.client_rpc import CoordinatorRPCClient, ServerNodeRPCClient
    from Distdataloader.rpc.example import _start_coordinator, _start_servernode
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from rpc.client_rpc import CoordinatorRPCClient, ServerNodeRPCClient
    from rpc.example import _start_coordinator, _start_servernode


def _tensor_batch_to_samples(batch: Any) -> List[torch.Tensor]:
    """Split a batch tensor into a list of per-sample tensors along dim 0."""
    '''       
    if batch.ndim == 0:
        raise ValueError("Expected at least 1D tensor for batched samples")
    if batch.size(0) == 0:
        return []
    return list(batch.unbind(dim=0))
    '''
    if isinstance(batch, torch.Tensor): 
        if batch.ndim == 0:
            raise ValueError("Expected at least 1D tensor")
        return list(batch.unbind(dim=0))
    
    elif isinstance(batch, dict):
        keys = list(batch.keys())
        if not keys:
            return []
        
        batch_size = len(batch[keys[0]])
        samples = []
        for i in range(batch_size):
            # 把每个 key 的第 i 个数据取出来，组成一个小字典
            sample = {k: batch[k][i] for k in keys}
            samples.append(sample)
        return samples
        
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
    def fetch(self, request: FetchRequest) -> Sequence[torch.Tensor]:
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
        def fetch(self, request: FetchRequest) -> Sequence[torch.Tensor]:
            values = []
            for idx in request.indices:
                values.append(torch.tensor([idx, idx], dtype=torch.long))
            batch = torch.stack(values) if values else torch.empty(0, dtype=torch.long)
            return _tensor_batch_to_samples(batch)

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
        # 发送 RPC 请求
        # 注意：这里使用了 rpc/common.py 里定义的 RPCMethod.REPORT_COMPLETION
        self._rpc._call(RPCMethod.REPORT_COMPLETION, payload)


class RemoteServerNodeClient(ServerNodeClient):
    """Server-node client backed by the socket-based RPC implementation."""

    def __init__(self, host: str, port: int, *, timeout: float = 10.0) -> None:
        self._rpc = ServerNodeRPCClient(host, port, timeout=timeout)

    def fetch(self, request: FetchRequest) -> Sequence[torch.Tensor]:
        client_id = request.extra.get("client_id") if request.extra else request.session_id
        
        raw_data = self._rpc.fetch(client_id, list(request.indices))
        if isinstance(raw_data, bytes):
            with io.BytesIO(raw_data) as f:
                # 这一步会把 bytes 还原成 {'images': Tensor, ...}
                batch = torch.load(f)
        elif isinstance(raw_data, torch.Tensor):
            # 兼容旧逻辑（如果服务器只返回 Tensor）
            batch = raw_data
        elif isinstance(raw_data, dict):
            batch = raw_data           
        else:
            raise TypeError(f"Unexpected payload type from server: {type(raw_data)}")
        payloads = _tensor_batch_to_samples(batch)
        

        # 4. 打包成 ProcessedSample (绑定 index 和 payload)
        results = []
        for idx, payload in zip(request.indices, payloads):
            results.append(ProcessedSample(index=idx, payload=payload))
            
        return results
        #return _tensor_batch_to_samples(batch)


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


def _demo_run() -> None:
    """Spin up coordinator/server RPC servers and run a sample client flow."""

    if _start_coordinator is None or _start_servernode is None:
        raise RuntimeError("Demo helpers are unavailable; ensure ray_demo.rpc is importable")

    coordinator_port = 19100
    server_port = 19200
    coord = _start_coordinator("127.0.0.1", coordinator_port)
    server = _start_servernode("127.0.0.1", server_port)

    try:
        time.sleep(0.2)
        coordinator_client = RemoteCoordinatorClient("127.0.0.1", coordinator_port)
        server_client = RemoteServerNodeClient("127.0.0.1", server_port)

        client_info = ClientInfo(client_id="demo", batch_size=4, microbatch_size=2)
        coordinator_client.register_client(client_info)
        assignment = coordinator_client.assign_server_node(client_info.client_id, [0, 1, 2, 3])
        samples = server_client.fetch(
            FetchRequest(
                session_id=client_info.client_id,
                server_node_id=assignment.server_node_id,
                indices=[5, 6, 7],
            )
        )
        print(f"Demo assignment: {assignment}")
        print(f"Fetched {len(samples)} samples; first sample: {samples[0] if samples else None}")
    finally:
        coord.close()
        server.close()


if __name__ == "__main__":
    _demo_run()


