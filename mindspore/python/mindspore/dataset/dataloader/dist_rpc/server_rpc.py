# SPDX-License-Identifier: Apache-2.0
"""Reusable RPC server that mirrors LMCache's socket server structure."""

# Standard
from dataclasses import dataclass
import socket
import threading
from typing import Any, Callable, Dict, Optional

# Third Party
import torch

# First Party
from .common import PayloadType, REQUEST_HEADER, RESPONSE_HEADER, RPCMethod
from .serde import TorchSerializer, deserialize_message, dtype_to_name, serialize_message


RPCHandler = Callable[[Any, "RPCRequestContext"], Any]


@dataclass
class RPCRequestContext:
    method: str
    client_address: tuple[str, int]


class RPCServer:
    """Simple threaded RPC server that lets each role register handlers."""

    def __init__(self, host: str, port: int, role: str = "server") -> None:
        self.host = host
        self.port = port
        self.role = role
        self._serializer = TorchSerializer()
        self._handlers: Dict[str, RPCHandler] = {}
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((host, port))
        self._server_socket.listen()
        self._stop_event = threading.Event()
        self._serve_thread: Optional[threading.Thread] = None

    def register_handler(self, method: str | RPCMethod, handler: RPCHandler) -> None:
        key = method.value if isinstance(method, RPCMethod) else method
        self._handlers[key] = handler

    def serve_forever(self) -> None:
        self._server_socket.settimeout(1.0)
        while not self._stop_event.is_set():
            try:
                client_socket, addr = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop_event.is_set():
                    break
                raise
            threading.Thread(target=self._handle_client, args=(client_socket, addr), daemon=True).start()

    def serve_in_thread(self) -> None:
        if self._serve_thread is not None and self._serve_thread.is_alive():
            return
        self._serve_thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._serve_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        self._server_socket.close()
        if self._serve_thread is not None:
            self._serve_thread.join(timeout=1.0)

    def _handle_client(self, client_socket: socket.socket, addr: tuple[str, int]) -> None:
        try:
            while not self._stop_event.is_set():
                header = self._receive_all(client_socket, REQUEST_HEADER.size)
                if not header:
                    break
                method_len, payload_type_val, payload_len = REQUEST_HEADER.unpack(header)
                method = self._receive_all(client_socket, method_len).decode("utf-8")
                payload_bytes = self._receive_all(client_socket, payload_len) if payload_len else b""
                payload_type = PayloadType(payload_type_val)
                payload = self._decode_request_payload(payload_type, payload_bytes)
                ctx = RPCRequestContext(method=method, client_address=addr)
                response = self._dispatch(method, payload, ctx)
                self._send_response(client_socket, response)
        finally:
            client_socket.close()

    def _dispatch(self, method: str, payload: Any, ctx: RPCRequestContext) -> tuple[int, PayloadType, bytes, Optional[str]]:
        handler = self._handlers.get(method)
        if handler is None:
            body = serialize_message({"error": f"Unknown method {method}"})
            return 404, PayloadType.JSON, body, None
        try:
            result = handler(payload, ctx)
        except Exception as exc:  # pragma: no cover - defensive logging hook
            body = serialize_message({"error": str(exc)})
            return 500, PayloadType.JSON, body, None
        return self._encode_response_payload(result)

    def _send_response(self, client_socket: socket.socket, response: tuple[int, PayloadType, bytes, Optional[str]]) -> None:
        status_code, payload_type, body, dtype_name = response
        dtype_bytes = dtype_name.encode("utf-8") if dtype_name else b""
        header = RESPONSE_HEADER.pack(status_code, payload_type.value, len(dtype_bytes), len(body))
        client_socket.sendall(header)
        if dtype_bytes:
            client_socket.sendall(dtype_bytes)
        if body:
            client_socket.sendall(body)

    def _receive_all(self, client_socket: socket.socket, nbytes: int) -> bytes:
        data = bytearray()
        while len(data) < nbytes:
            chunk = client_socket.recv(nbytes - len(data))
            if not chunk:
                return b""
            data.extend(chunk)
        return bytes(data)

    def _decode_request_payload(self, payload_type: PayloadType, data: bytes) -> Any:
        if payload_type == PayloadType.JSON:
            return deserialize_message(data)
        if payload_type == PayloadType.BYTES:
            return data
        if payload_type == PayloadType.TENSOR:
            raise ValueError("Tensor payloads are not supported for inbound RPC calls")
        raise ValueError(f"Unknown payload type {payload_type}")

    def _encode_response_payload(self, result: Any) -> tuple[int, PayloadType, bytes, Optional[str]]:
        status_code = 200
        if result is None:
            return status_code, PayloadType.JSON, b"", None
        if isinstance(result, torch.Tensor):
            dtype_name = dtype_to_name(result.dtype)
            body = self._serializer.to_bytes(result)
            return status_code, PayloadType.TENSOR, body, dtype_name
        if isinstance(result, (bytes, bytearray)):
            return status_code, PayloadType.BYTES, bytes(result), None
        if isinstance(result, (str, int, float, bool, list, dict)) or result is None:
            return status_code, PayloadType.JSON, serialize_message(result), None
        raise TypeError(f"Unsupported response type: {type(result)}")


class CoordinatorRPCServer(RPCServer):
    """Coordinator server exposing register/assign RPC handlers."""

    def on_register_client(self, handler: RPCHandler) -> None:
        self.register_handler(RPCMethod.REGISTER_CLIENT, handler)

    def on_assign_servernode(self, handler: RPCHandler) -> None:
        self.register_handler(RPCMethod.ASSIGN_SERVERNODE, handler)


class ServerNodeRPCServer(RPCServer):
    """Server node that only needs to handle fetch RPCs."""

    def on_fetch(self, handler: RPCHandler) -> None:
        self.register_handler(RPCMethod.FETCH, handler)
