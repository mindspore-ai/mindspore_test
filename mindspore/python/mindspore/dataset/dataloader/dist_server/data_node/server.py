import argparse
import pandas as pd
import torch
import io
import time
from concurrent.futures import ThreadPoolExecutor

from dist_rpc.server_rpc import ServerNodeRPCServer
from dist_rpc.client_rpc import CoordinatorRPCClient
from dist_server.data_node.processor import DataProcessor

# 全局处理器
processor = None
executor = ThreadPoolExecutor(max_workers=4)

def handle_fetch(payload, ctx):
    """
    Payload 格式: {'indices': [1, 2, 3], 'client_id': '...'}
    """
    indices = payload.get("indices", [])
    if not indices:
        return None

    future = executor.submit(processor.get_batch, indices)
    result_dict = future.result()

    if result_dict is None:
        return None
    
    buffer = io.BytesIO()
    torch.save(result_dict, buffer)
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
    parser.add_argument("--metadata", type=str, default="/home/lyh/ray_temp/metadata.parquet")
    parser.add_argument("--tokenizer", type=str, default="/home/lyh/ray_temp/models--openai--clip-vit-base-patch32")
    args = parser.parse_args()

    # 1. 加载数据
    print(f"Loading metadata from {args.metadata}...")
    df = pd.read_parquet(args.metadata)
    
    # 2. 初始化处理器
    global processor
    processor = DataProcessor(args.tokenizer, df)

    # 3. 启动 RPC Server
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