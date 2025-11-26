# SPDX-License-Identifier: Apache-2.0
"""Client-side RPC helpers for coordinator and server nodes."""

# Standard
from dataclasses import dataclass
import socket
from typing import Any, Dict, Iterable, Sequence

# Third Party
import torch

# First Party
from .common import PayloadType, REQUEST_HEADER, RESPONSE_HEADER, RPCMethod
from .serde import (
    TorchDeserializer,
    deserialize_message,
    dtype_from_name,
    serialize_message,
)


@dataclass
class RPCResponse:
    status_code: int
    payload: Any
    payload_type: PayloadType
    dtype: str | None = None


class _BaseRPCClient:
    """Shared socket helpers for coordinator/server RPC clients."""

    def __init__(self, host: str, port: int, timeout: float = 10.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._tensor_deserializers: Dict[str, TorchDeserializer] = {}

    def _call(self, method: RPCMethod, payload: Any) -> RPCResponse:
        method_bytes = method.value.encode("utf-8")
        payload_bytes = serialize_message(payload)
        header = REQUEST_HEADER.pack(len(method_bytes), PayloadType.JSON.value, len(payload_bytes))
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            self._send_all(sock, header)
            self._send_all(sock, method_bytes)
            if payload_bytes:
                self._send_all(sock, payload_bytes)

            response_header = self._receive_all(sock, RESPONSE_HEADER.size)
            status_code, resp_type_val, dtype_len, body_len = RESPONSE_HEADER.unpack(response_header)
            dtype_name = None
            if dtype_len:
                dtype_name = self._receive_all(sock, dtype_len).decode("utf-8")
            body = self._receive_all(sock, body_len) if body_len else b""

        payload_type = PayloadType(resp_type_val)
        payload_obj = self._decode_payload(payload_type, body, dtype_name)
        return RPCResponse(status_code=status_code, payload=payload_obj, payload_type=payload_type, dtype=dtype_name)

    def _decode_payload(self, payload_type: PayloadType, body: bytes, dtype_name: str | None) -> Any:
        if payload_type == PayloadType.JSON:
            return deserialize_message(body)
        if payload_type == PayloadType.BYTES:
            return body
        if payload_type == PayloadType.TENSOR:
            if dtype_name is None:
                raise ValueError("Tensor response missing dtype metadata")
            deserializer = self._tensor_deserializers.get(dtype_name)
            if deserializer is None:
                torch_dtype = dtype_from_name(dtype_name)
                deserializer = TorchDeserializer(torch_dtype)
                self._tensor_deserializers[dtype_name] = deserializer
            return deserializer.from_bytes(body)
        raise ValueError(f"Unknown payload type: {payload_type}")

    def _send_all(self, sock: socket.socket, data: bytes) -> None:
        view = memoryview(data)
        total_sent = 0
        while total_sent < len(data):
            sent = sock.send(view[total_sent:])
            if sent == 0:
                raise RuntimeError("Socket connection broken while sending")
            total_sent += sent

    def _receive_all(self, sock: socket.socket, nbytes: int) -> bytes:
        data = bytearray()
        while len(data) < nbytes:
            chunk = sock.recv(nbytes - len(data))
            if not chunk:
                raise RuntimeError("Socket connection closed unexpectedly")
            data.extend(chunk)
        return bytes(data)


class CoordinatorRPCClient(_BaseRPCClient):
    """Client used by GPU workers to contact the coordinator."""

    def register_client(self, client_info: Dict[str, Any]) -> bool:
        response = self._call(RPCMethod.REGISTER_CLIENT, client_info)
        return bool(response.payload)

    def assign_servernode(self, client_id: str, indices: Sequence[int]) -> Dict[str, Any]:
        payload = {"client_id": client_id, "indices": list(indices)}
        response = self._call(RPCMethod.ASSIGN_SERVERNODE, payload)
        if not isinstance(response.payload, dict):
            raise ValueError("Coordinator did not return a server node mapping")
        return response.payload
    
    def register_servernode(self, node_id: str, port: int, weight: float = 1.0) -> bool:
        payload = {
            "client_id": node_id, 
            "port": port, 
            "weight": weight
        }
        response = self._call(RPCMethod.REGISTER_SERVERNODE, payload)
        return bool(response.payload)


class ServerNodeRPCClient(_BaseRPCClient):
    """Client used by GPU workers to fetch processed samples from server nodes."""

    def fetch(self, client_id: str, indices: Sequence[int]) -> Any:
        payload = {"client_id": client_id, "indices": list(indices)}
        response = self._call(RPCMethod.FETCH, payload)
        if response.payload_type not in (PayloadType.TENSOR, PayloadType.BYTES):
             raise ValueError(f"Fetch RPC returned unexpected type: {response.payload_type}")
        #assert isinstance(response.payload, torch.Tensor)
        return response.payload


def fetch_samples(server_host: str, server_port: int, client_id: str, indices: Iterable[int]) -> torch.Tensor:
    """Utility to perform a one-off fetch call without keeping a client instance."""
    client = ServerNodeRPCClient(server_host, server_port)
    return client.fetch(client_id, list(indices))
