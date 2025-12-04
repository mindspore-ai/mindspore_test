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
"""
RPC server for datanode.
"""


import argparse
import pandas as pd
import mindspore
import io
import time

from concurrent.futures import ThreadPoolExecutor
from dist_rpc.server_rpc import ServerNodeRPCServer
from dist_rpc.client_rpc import CoordinatorRPCClient
from dist_server.data_node.processor import DataProcessor

#global processor
processor = None
executor = ThreadPoolExecutor(max_workers=4)

def handle_fetch(payload):
    """
    Payload Format: {'indices': [1, 2, 3], 'client_id': '...'}
    """
    indices = payload.get("indices", [])
    if not indices:
        return None

    future = executor.submit(processor.get_batch, indices)
    result_dict = future.result()

    if result_dict is None:
        return None
    
    buffer = io.BytesIO()
    mindspore.save(result_dict, buffer)
    return buffer.getvalue() 

def register_to_coordinator(coordinator_host, coordinator_port, local_port, node_id="node_0"):
    client = CoordinatorRPCClient(coordinator_host, coordinator_port)
    while True:
        try:
            print(f"Attempting to register to Coordinator at {coordinator_host}:{coordinator_port}...")
            success = client.register_servernode(node_id=node_id, port=local_port)
            if success:
                print(">>> Registration SUCCESS!")
                break
        except Exception as e:
            print(f"Registration failed ({e}), retrying in 2s...")
            time.sleep(2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--coord_host", type=str, default="127.0.0.1") 
    parser.add_argument("--coord_port", type=int, default=9100)        
    parser.add_argument("--node_id", type=str, default="node_0")       
    parser.add_argument("--metadata", type=str)
    parser.add_argument("--tokenizer", type=str)
    args = parser.parse_args()

    print(f"Loading metadata from {args.metadata}...")
    df = pd.read_parquet(args.metadata)
    
    global processor
    processor = DataProcessor(args.tokenizer, df)

    #start up RPC Server
    server = ServerNodeRPCServer("0.0.0.0", args.port)
    server.on_fetch(handle_fetch)
    
    import threading
    t = threading.Thread(target=register_to_coordinator, args=(
        args.coord_host, 
        args.coord_port, 
        args.port,
        args.node_id
    ), daemon=True)
    t.start()
    
    
    print(f"ServerNode running on port {args.port}...")
    server.serve_forever()

if __name__ == "__main__":
    main()