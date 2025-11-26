# SPDX-License-Identifier: Apache-2.0
"""Minimal end-to-end demo for the custom RPC stack."""

# Standard
import sys
import time
from pathlib import Path
from typing import Any

# Third Party
import torch

# Local (support running as script or module)
if __package__ is None or __package__ == "":
    this_dir = Path(__file__).resolve().parent
    workspace_root = this_dir.parent
    sys.path.insert(0, str(workspace_root.parent))
    from ray_demo.rpc.client_rpc import CoordinatorRPCClient, ServerNodeRPCClient
    from ray_demo.rpc.server_rpc import CoordinatorRPCServer, ServerNodeRPCServer
else:
    from .client_rpc import CoordinatorRPCClient, ServerNodeRPCClient
    from .server_rpc import CoordinatorRPCServer, ServerNodeRPCServer


def _start_coordinator(host: str, port: int) -> CoordinatorRPCServer:
    server = CoordinatorRPCServer(host, port, role="coordinator")

    def handle_register(payload: Any, _ctx):
        print(f"[Coordinator] register_client payload={payload}")
        return True

    def handle_assign(payload: Any, _ctx):
        print(f"[Coordinator] assign_servernode payload={payload}")
        # Masquerade as a router that sends everyone to the fetch server.
        return {"host": "127.0.0.1", "port": 9200}

    server.on_register_client(handle_register)
    server.on_assign_servernode(handle_assign)
    server.serve_in_thread()
    return server


def _start_servernode(host: str, port: int) -> ServerNodeRPCServer:
    server = ServerNodeRPCServer(host, port, role="servernode")

    def handle_fetch(payload: Any, _ctx):
        print(f"[ServerNode] fetch payload={payload}")
        indices = payload.get("indices", [])
        tensor = torch.tensor(indices, dtype=torch.float32)
        return tensor

    server.on_fetch(handle_fetch)
    server.serve_in_thread()
    return server


def main() -> None:
    coord = _start_coordinator("127.0.0.1", 9100)
    node = _start_servernode("127.0.0.1", 9200)

    time.sleep(0.2)  # Allow sockets to start listening

    coordinator_client = CoordinatorRPCClient("127.0.0.1", 9100)
    server_client = ServerNodeRPCClient("127.0.0.1", 9200)

    assert coordinator_client.register_client({"client_id": "demo", "batch_size": 2})
    assignment = coordinator_client.assign_servernode("demo", [0, 1, 2])
    print(f"Assignment: {assignment}")

    tensor = server_client.fetch("demo", [5, 6, 7])
    print(f"Fetch result tensor: {tensor}")

    coord.close()
    node.close()


if __name__ == "__main__":
    main()
